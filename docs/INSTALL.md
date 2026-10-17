# Installation

CoverGen runs on Linux, macOS (including Apple Silicon), and Windows. The
pipeline auto-selects the best available torch device (CUDA → MPS → CPU).

## Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| **Python 3.10–3.12** | Runtime | <https://www.python.org/downloads/> |
| **ffmpeg** | Audio decoding, YouTube downloads, stereo conversion | Linux: `apt install ffmpeg` · macOS: `brew install ffmpeg` · Windows: <https://www.gyan.dev/ffmpeg/builds/> |
| **git** | Clone the repo | `apt install git` / `brew install git` |
| **(optional) NVIDIA driver + CUDA** | GPU inference | <https://developer.nvidia.com/cuda-downloads> |

Python 3.13 is not recommended yet — `pyworld` and `faiss-cpu` wheels are
still catching up. If you must use 3.13, install `pyworld` from source.

## 1. Clone the repo

```bash
git clone https://github.com/asukaa2/CoverGen.git
cd CoverGen
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
```

## 3. Install PyTorch

Pick the wheel index that matches your hardware. The general-purpose CPU
wheel always works; for GPU use the matching CUDA index.

```bash
# CPU only (works everywhere)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

## 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file uses [PEP 508 environment markers](https://peps.python.org/pep-0508/)
to install `onnxruntime` on macOS and `onnxruntime-gpu` everywhere else.

## 5. Verify the install

```bash
python -c "import torch, librosa, gradio, fastapi, yt_dlp; print('OK')"
python -c "import sys; sys.path.insert(0,'.'); import app, core, webui; print('CoverGen modules OK')"
```

If both lines print `OK`, you're ready to run the server.

## 6. (Optional) Install `fairseq`

The pipeline can load HuBERT / ContentVec embedders via `torch.hub` (the
default, see `tools/fairseq.py`). If `torch.hub` fails on your network,
install the full `fairseq` package as a fallback:

```bash
pip install fairseq
```

`fairseq` is large and has strict version requirements; only install it
if the default loader doesn't work.

## Common install issues

### `error: command 'gcc' failed` while installing `pyworld`

`pyworld` needs a C compiler. On Debian/Ubuntu: `apt install build-essential python3-dev`.
On macOS: `xcode-select --install`. On Windows: install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### `ModuleNotFoundError: No module named 'pkg_resources'`

You're on setuptools ≥ 81, which removed `pkg_resources`. Downgrade:

```bash
pip install "setuptools<81"
```

### `onnxruntime-gpu` fails to find CUDA

Either install CUDA (see step 3) or fall back to the CPU wheel:

```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

### `Could not find a version that satisfies the requirement torchcodec==0.9.1`

`torchcodec` is no longer required (it was a leftover from an older
revision). The current `requirements.txt` does not include it; if you're
upgrading an old checkout, remove it from your env:

```bash
pip uninstall torchcodec
```

## Next steps

- Add a voice model — see [MODELS.md](MODELS.md).
- Start the server — see [../README.md](../README.md#quick-start).
- Browse the API — see [API.md](API.md).
