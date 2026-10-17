import os

import torch
from scipy.io import wavfile

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.infer.my_utils import load_audio
from rvc.infer.pipeline import VC
from rvc.modules.predictor.utils import check_rvc.modules.predictors, check_embedders, MODELS_DIR

# Local fairseq implementation (src/tools/fairseq.py) - replaces the pip "fairseq" package.
from tools.fairseq import load_model as load_fairseq_model


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        # Accept both plain strings and torch.device objects.
        device_str = str(self.device)

        if torch.cuda.is_available():
            i_device = int(device_str.split(":")[-1]) if ":" in device_str else 0
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )

        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
            self.is_half = False
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            # Half precision is not supported for CPU inference.
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError(f'Unsupported RVC model version: {version}')

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def resolve_embedder_path(embedder_name, rvc_models_dir):
    """Locate the embedder (.pt) file, preferring the shared rvc_models directory."""
    candidates = [
        os.path.join(rvc_models_dir, embedder_name + ".pt"),
        os.path.join(MODELS_DIR, embedder_name + ".pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def rvc_infer(
    index_path,
    index_rate,
    input_path,
    output_path,
    pitch_change,
    f0_method,
    cpt,
    version,
    net_g,
    filter_radius,
    tgt_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    vc,
    config,
    embedder_name,
    rvc_models_dir,
    f0autotune,
    f0_min=50,
    f0_max=1100
):
    # Download missing rvc.modules.predictor/embedder files before inference.
    check_predictors(f0_method)
    check_embedders(embedder_name)

    device = torch.device(config.device)
    print(f'[CoverGen] Inference device: {device}')

    embedder_model_path = resolve_embedder_path(embedder_name, rvc_models_dir)
    if not os.path.exists(embedder_model_path):
        raise FileNotFoundError(f'[ERROR] Embedder model not found: {embedder_model_path}')

    # Load the HuBERT/ContentVec embedder with the local fairseq tool (src/tools/fairseq.py).
    hubert_model = load_fairseq_model(embedder_model_path).to(device).eval()
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()

    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        input_path,
        times,
        pitch_change,
        f0_method,
        index_path,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        0,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        f0autotune,
        f0_file=None,
        f0_min=f0_min,
        f0_max=f0_max
    )
    wavfile.write(output_path, tgt_sr, audio_opt)
