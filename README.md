# CoverGen

**AI Song Cover Generator** — separate vocals from any song, swap the singer's voice with an RVC model, and mix everything back together into a finished AI cover.

CoverGen bundles three pieces:

1. A **vocal/instrument separator** (vendored `audio_separator` package — MDX, VR, Demucs, Roformer architectures).
2. A **voice-conversion engine** (RVC v1/v2 inference pipeline with multiple F0 trackers: rmvpe+, fcpe, crepe/mangio-crepe, harvest, dio, yin, pyin).
3. A **FastAPI backend** + a small static frontend (HTML/CSS/JS), plus an optional Gradio web UI for power users.

---

## Features

- **Three input modes**: YouTube link, local audio file path, or browser upload.
- **Multi-stage pipeline**: vocals/instrumental split → dereverb → main/backup vocal split → RVC voice conversion → final mix.
- **Multiple F0 methods** including hybrid combinations (e.g. `hybrid[rmvpe+fcpe]`).
- **REST API** for programmatic access — every feature exposed at `/api/*`.
- **Optional Gradio UI** (`webui.py`) for users who prefer the classic RVC webui look.
- **CPU + GPU + Apple MPS** inference (auto-selected by torch).

---

## Project layout

```
CoverGen/
├── app.py                 # FastAPI backend (REST API + static frontend)
├── webui.py               # Optional Gradio web UI
├── core.py                # Pipeline: yt download → separate → RVC → mix
├── requirements.txt       # Python dependencies
├── tools/
│   └── fairseq.py         # Local HuBERT/ContentVec loader (replaces pip fairseq)
├── rvc/
│   ├── static/            # HTML/CSS/JS frontend for app.py
│   ├── infer/             # RVC inference (Config, get_vc, pipeline)
│   └── modules/
│       ├── audio_separator/   # Vendored python-audio-separator
│       ├── infer_pack/        # RVC SynthesizerTrn model definitions
│       └── predictor/         # F0 trackers: RMVPE, FCPE, CREPE, PYWORLD
├── assets/
│   ├── rvc_models/        # Voice model folders (.pth + optional .index)
│   ├── separator_models/  # MDX/Roformer/VR separation model files
│   └── song_output/       # Generated covers + intermediate stems
└── docs/                  # See the docs/ folder for detailed guides
```

---

## Quick start

### 1. Install dependencies

```bash
git clone https://github.com/asukaa2/CoverGen.git
cd CoverGen
pip install -r requirements.txt
```

> **Note**: On macOS skip `onnxruntime-gpu` — `requirements.txt` already handles that automatically. On Windows/Linux with an NVIDIA GPU, both `onnxruntime-gpu` and `torch` CUDA builds are recommended.

> **System dependency**: `ffmpeg` must be installed and on your `PATH` (used for audio decoding and YouTube downloads). Install via `apt install ffmpeg`, `brew install ffmpeg`, or the official Windows build.

### 2. Add an RVC voice model

Drop a folder containing a `.pth` file (and optional `.index` file) into `assets/rvc_models/`:

```
assets/rvc_models/
└── my-voice/
    ├── my-voice.pth
    └── my-voice.index
```

You can also install a model from a zip URL at runtime via the web UI's "Download model" tab or the `POST /api/models/download` endpoint.

### 3. (First run only) Download separator + predictor models

The first time you generate a cover, the pipeline will auto-download the required F0 trackers (rmvpe.pt, fcpe.pt, crepe_*.pth) and HuBERT embedder (hubert_base.pt) from HuggingFace into a `models/` directory at the project root.

The MDX/Roformer **separation** models are NOT auto-downloaded — copy them into `assets/separator_models/`. The pipeline expects:

| File                                            | Purpose                          |
|-------------------------------------------------|----------------------------------|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt`     | Vocal / instrumental split       |
| `UVR-DeEcho-DeReverb.pth`                       | Vocal dereverb                   |
| `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | Main / backup vocal split |

You can find these on the [UVR models archive](https://github.com/TRvlvr/model_scheme/releases) and the [audio-separator releases page](https://github.com/nomadkaraoke/python-audio-separator/releases).

### 4. Run the server

```bash
python app.py --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860> in your browser, pick a voice model, paste a YouTube URL or upload an audio file, and click **Generate AI Cover**.

### 5. (Optional) Use the Gradio UI instead

```bash
python webui.py --listen --listen-port 7860
```

### 6. (Optional) Use the CLI

```bash
python core.py \
    -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    -dir my-voice \
    -p -12 \
    -k \
    -oformat mp3
```

Run `python core.py --help` for the full list of options.

---

## REST API

| Method | Endpoint                  | Purpose                                                         |
|--------|---------------------------|-----------------------------------------------------------------|
| GET    | `/api/models`             | List installed voice models (JSON `{"models": [...]}`).         |
| POST   | `/api/models/download`    | Install a voice model from a HuggingFace/Pixeldrain/GDrive URL. |
| POST   | `/api/models/upload`      | Install a voice model from an uploaded zip (raw body + `?dir_name=`). |
| POST   | `/api/cover`              | Generate an AI cover; returns the binary audio file.            |

The `/api/cover` endpoint accepts `multipart/form-data` with these fields:

| Field             | Type    | Default  | Description                                              |
|-------------------|---------|----------|----------------------------------------------------------|
| `voice_model`     | string  | required | Name of the voice model folder.                          |
| `song_input`      | string  | `""`     | YouTube URL or local file path.                          |
| `audio_file`      | file    | optional | Uploaded audio file (overrides `song_input`).           |
| `pitch`           | int     | `0`      | Pitch shift in semitones (−24 to +24).                  |
| `keep_files`      | bool    | `false`  | Keep intermediate stems in `assets/song_output/`.       |
| `output_format`   | string  | `mp3`    | `mp3`, `flac`, or `wav`.                                 |
| `f0_method`       | string  | `rmvpe+` | Pitch detection method.                                 |
| `index_rate`      | float   | `0.5`    | Timbre leakage control (0–1).                            |
| `filter_radius`   | int     | `3`      | Median filter radius (0–7).                              |
| `rms_mix_rate`    | float   | `0.25`   | Loudness preservation (0=original, 1=fixed).            |
| `protect`         | float   | `0.33`   | Voiceless-consonant protection (0–0.5; 0.5 disables).   |
| `crepe_hop_length`| int     | `128`    | Hop length for crepe-based trackers.                    |
| `f0autotune`      | bool    | `false`  | Apply pitch correction.                                  |
| `f0_min` / `f0_max` | int   | `50`/`1100` | Min/max pitch (Hz) for F0 detection.                  |
| `main_gain` / `backup_gain` / `inst_gain` | int | `0` | Per-stem volume adjustment (dB). |

The response body is the generated audio file itself. Stem URLs are listed in the `X-Stems` response header (URL-encoded JSON array). See `docs/API.md` for full examples.

Interactive API docs are available at `/docs` once the server is running.

---

## CLI usage

```bash
python core.py -i <song-input> -dir <voice-model-name> -p <pitch> [options]
```

Key flags:

| Flag | Description |
|------|-------------|
| `-i, --song-input` | YouTube URL or local audio path (required). |
| `-dir, --rvc-dirname` | Voice model folder name under `assets/rvc_models/` (required). |
| `-p, --pitch-change` | Pitch shift in semitones (required). |
| `-k, --keep-files / --no-keep-files` | Keep intermediate audio files. |
| `-oformat, --output-format` | `mp3` (default), `flac`, or `wav`. |
| `-palgo, --pitch-detection-algo` | `rmvpe` (default), `fcpe`, `mangio-crepe`, etc. |
| `-autotune / --no-autotune` | Enable pitch correction. |
| `-minpitch`, `-maxpitch` | Min/max pitch in Hz. |

Run `python core.py --help` for the complete list.

---

## How the pipeline works

```
Input (YouTube / file / upload)
        │
        ▼
[1] Stereo conversion (ffmpeg)
        │
        ▼
[2] Vocal / Instrumental split  ── model_bs_roformer_ep_317_sdr_12.9755.ckpt
        │
        ▼
[3] Vocal dereverb              ── UVR-DeEcho-DeReverb.pth
        │
        ▼
[4] Main / Backup vocal split   ── mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
        │
        ▼
[5] RVC voice conversion        ── HuBERT/ContentVec embedder + RVC v2 .pth
        │
        ▼
[6] Mix AI vocals + backup + instrumental  ── pydub
        │
        ▼
Final AI cover (mp3 / flac / wav)
```

Intermediate files are written to `assets/song_output/<song_id>/` and deleted when `keep_files=False`.

---



## License

See [LICENSE](LICENSE) for the project's license terms.

The vendored `rvc/modules/audio_separator/` package is derived from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and retains its upstream license.
