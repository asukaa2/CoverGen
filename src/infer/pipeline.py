from functools import lru_cache
import numpy as np, parselmouth, torch, sys, os
from time import time as ttime
import torch.nn.functional as F
from scipy import signal
from torch import Tensor
import faiss, librosa
import random
import gc
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
now_dir = os.path.join(BASE_DIR, 'src')
sys.path.append(now_dir)

from predictor.generator import Generator, Autotune, post_process

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()

    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device

        # Initialize the new Generator with the same parameters
        self.generator = Generator(
            sample_rate=self.sr,
            hop_length=self.window,
            f0_min=50,
            f0_max=1100,
            is_half=self.is_half,
            device=self.device,
        )

        self.ref_freqs = [
            65.41,
            82.41,
            110.00,
            146.83,
            196.00,
            246.94,
            329.63,
            440.00,
            587.33,
            783.99,
            1046.50,
        ]
        self.note_dict = self.generate_interpolated_frequencies()

    def generate_interpolated_frequencies(self):
        note_dict = []
        for i in range(len(self.ref_freqs) - 1):
            freq_low = self.ref_freqs[i]
            freq_high = self.ref_freqs[i + 1]
            interpolated_freqs = np.linspace(
                freq_low, freq_high, num=10, endpoint=False
            )
            note_dict.extend(interpolated_freqs)
        note_dict.append(self.ref_freqs[-1])
        return note_dict

    def autotune_f0(self, f0, f0_autotune_strength=1.0):
        """Autotune f0 using the new Autotune class from generator."""
        autotune = Autotune(self.note_dict)
        return autotune.autotune_f0(f0, f0_autotune_strength)

    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index % torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def normalize_f0_method(self, f0_method):
        """Map UI-level method names to the generator's supported methods."""
        aliases = {
            'rmvpe+': 'rmvpe',
            'mangio-crepe': 'mangio-crepe-full',
            'crepe': 'crepe-full',
        }
        return aliases.get(f0_method, f0_method)

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        crepe_hop_length,
        f0autotune,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        # Propagate user settings into the pitch generator.
        self.generator.f0_min = f0_min
        self.generator.f0_max = f0_max

        if f0_method.startswith('hybrid[') and f0_method.endswith(']'):
            # Hybrid methods: average the f0 curves of multiple trackers.
            sub_methods = f0_method[len('hybrid['):-1].split('+')
            f0_curves = []
            for sub in sub_methods:
                sub_normalized = self.normalize_f0_method(sub.strip())
                self.sync_hop_length(sub_normalized, crepe_hop_length)
                f0_curves.append(self.generator.compute_f0(
                    sub_normalized, x, p_len,
                    filter_radius if filter_radius % 2 != 0 else filter_radius + 1
                ))
            if isinstance(f0_curves[0], tuple): f0_curves = [c[0] for c in f0_curves]
            stacked = np.vstack([np.asarray(c, dtype=np.float64) for c in f0_curves])
            voiced = stacked > 0
            voiced_count = voiced.sum(axis=0)
            f0_bak_avg = np.where(voiced_count > 0, stacked.sum(axis=0) / np.maximum(voiced_count, 1), 0.0)
            f0_coarse, f0bak = post_process(f0_bak_avg, f0_up_key, f0_mel_min, f0_mel_max)
        else:
            normalized = self.normalize_f0_method(f0_method)
            self.sync_hop_length(normalized, crepe_hop_length)
            f0_coarse, f0bak = self.generator.calculator(
                f0_method=normalized,
                x=x,
                f0_up_key=f0_up_key,
                p_len=p_len,
                filter_radius=filter_radius,
                f0_autotune=bool(f0autotune),
                f0_autotune_strength=1.0,
            )

        # Handle inp_f0 replacement if provided
        if inp_f0 is not None:
            tf0 = self.sr // self.window
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0bak[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0bak[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
            # Recompute f0_coarse after replacement
            f0_coarse, f0bak = post_process(
                f0bak.copy() / pow(2, f0_up_key / 12),  # undo the pitch shift
                f0_up_key,
                f0_mel_min,
                f0_mel_max,
            )

        return f0_coarse, f0bak

    def sync_hop_length(self, f0_method, crepe_hop_length):
        """Keep the generator's hop length in sync for crepe-based trackers.

        Cached crepe models are invalidated when the hop length changes because
        the hop length is baked into the model instance at construction time.
        """
        if 'crepe' not in f0_method:
            return
        if self.generator.hop_length != crepe_hop_length:
            self.generator.hop_length = crepe_hop_length
            if hasattr(self.generator, 'crepe'): del self.generator.crepe
            if hasattr(self.generator, 'mangio_crepe'): del self.generator.mangio_crepe

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch != None and pitchf != None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        f0autotune,
        f0_file=None,
        f0_min=50,
        f0_max=1100,
    ):
        if file_index != "" and os.path.exists(file_index) == True and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(error)
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception as error:
                print(error)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                crepe_hop_length,
                f0autotune,
                inp_f0,
                f0_min,
                f0_max,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
