import os
import gc
import sys
import torch
import codecs
import librosa
import requests

import numpy as np
import soundfile as sf
import torch.nn.functional as F

# Absolute directory that stores downloaded predictor/embedder files (independent of CWD).
MODELS_DIR = os.path.join(os.getcwd(), 'models')

PREDICTOR_DOWNLOAD_LINK = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13")
EMBEDDER_DOWNLOAD_LINK = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13") + "fairseq/"


def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())


def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(url))
    elif output_path is None:
        output_path = os.path.basename(url)

    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                f.write(chunk)

        return output_path
    else: raise ValueError(f'Failed to download {url}: HTTP {response.status_code}')

def check_predictors(method):
    def download(predictors):
        target = os.path.join(MODELS_DIR, predictors)
        if not os.path.exists(target):
            os.makedirs(MODELS_DIR, exist_ok=True)
            HF_download_file(PREDICTOR_DOWNLOAD_LINK + predictors, target)

    model_dict = {
        **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"),
        **dict.fromkeys(["fcpe"], "fcpe.pt"),
        **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"),
        **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"),
        **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"),
        **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"),
        **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"),
        **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"),
    }

    if method in model_dict: download(model_dict[method])

def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "spin"]:
        model_path = os.path.join(MODELS_DIR, hubert + ".pt")
        if not os.path.exists(model_path):
            os.makedirs(MODELS_DIR, exist_ok=True)
            HF_download_file(EMBEDDER_DOWNLOAD_LINK + hubert + ".pt", model_path)

def load_audio(file, sample_rate=16000):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(f"[ERROR] Not found audio: {file}")

        try:
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error reading audio file: {e}")

    return audio.flatten()

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0
