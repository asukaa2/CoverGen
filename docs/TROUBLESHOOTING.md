# Troubleshooting

This guide lists the most common errors and their fixes. If your error
isn't here, check the [issues on GitHub](https://github.com/asukaa2/CoverGen/issues)
or open a new one.

---

## Install / import errors

### `ModuleNotFoundError: No module named 'torch'`

PyTorch is not installed. Install it from the official wheel index:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For GPU, replace `cpu` with your CUDA version (e.g. `cu121`).

### `ModuleNotFoundError: No module named 'pkg_resources'`

You're on `setuptools ≥ 81`, which removed `pkg_resources`. Downgrade:

```bash
pip install "setuptools<81"
```

### `ModuleNotFoundError: No module named 'fairseq'`

Install the optional `fairseq` package, or rely on the built-in
`tools/fairseq.py` shim which loads HuBERT via `torch.hub`. If
`torch.hub` is unreachable, install `fairseq`:

```bash
pip install fairseq
```

### `ModuleNotFoundError: No module named 'audio_separator'`

CoverGen vendors `audio_separator` under `rvc/modules/`. The code in
`core.py` and `app.py` automatically adds that directory to
`sys.path`. If you're importing `core` from outside the project root,
make sure you `sys.path.insert(0, '/path/to/CoverGen/rvc/modules')`
before importing.

### `ImportError: cannot import name 'Server' from 'gradio'`

You're running an old version of `app.py` that called `gradio.Server`.
The current version uses FastAPI directly. Make sure you have the
latest `app.py` from this repo and that there is no file named
`gradio.py` in the project root (it would shadow the real `gradio`
package). The legacy `gradio.py` was renamed to `webui.py`.

### `ImportError: cannot import name 'check_rvc.modules.predictors'`

You're running an old version of `rvc/infer/infer.py` that had a typo
in an import. The current version uses `check_predictors`. Re-pull the
latest code.

### `ModuleNotFoundError: No module named 'local_attention'`

Install it: `pip install local-attention` (it's required by the FCPE
pitch tracker). The current `requirements.txt` already includes it.

### `ModuleNotFoundError: No module named 'pyworld'`

`pip install pyworld`. You may need a C compiler (`build-essential` on
Debian, `xcode-select --install` on macOS, MSVC Build Tools on Windows).

---

## Runtime errors during cover generation

### `Missing MDX-Net separation model(s): Kim_Vocal_2.onnx, ...`

The three separator model files are missing from
`assets/separator_models/`. See [MODELS.md](MODELS.md#2-separator-models)
for the list of files and where to download them.

### `Voice model "X" was not found. Refresh the model list.`

The model folder doesn't exist under `assets/rvc_models/`, or it's
filtered out by `ITEMS_TO_EXCLUDE` (which skips `hubert_base.pt`,
`rmvpe.pt`, `fcpe.pt`, and `MODELS.txt`). Make sure your model is in
its own subfolder:

```
assets/rvc_models/
└── my-voice/
    └── my-voice.pth
```

### `No .pth model file was found in the extracted zip`

The zip must contain a `.pth` file larger than 40 MB. Some RVC model
zips ship the weights inside a nested folder — CoverGen walks the zip
recursively, but if your `.pth` is smaller than 40 MB it will be
skipped (to avoid picking up test checkpoints).

Verify the zip contents:

```bash
unzip -l model.zip | grep -E '\.pth|\.index'
```

### `Incorrect format for <path>. Use a voice model trained using RVC v2 instead.`

The `.pth` file is not an RVC checkpoint (it might be a PyTorch state
dict without the `config`/`weight`/`version` keys RVC expects). Use a
model trained with [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

### `Could not load a fairseq/HuBERT model from '<path>'`

The embedder file (e.g. `hubert_base.pt`) couldn't be loaded. Either:

1. The file is missing — let the pipeline auto-download it on first
   run, or download it manually into `models/` or `assets/rvc_models/`.
2. The file is corrupt — re-download it from the
   [RVC v2 models repo](https://huggingface.co/lj1995/VoiceConversionWebUI).
3. `torch.hub` can't reach GitHub — install the `fairseq` package
   (`pip install fairseq`) as a fallback loader.

### `[WARNING] Feature index '<name>.index' contains 0 vectors`

The `.index` file is empty or corrupt. The pipeline will continue
without index retrieval (lower timbre accuracy but still functional).
Re-download or re-upload the model's `.index` file to restore full
quality.

### `Failed to load audio: ...` (from `my_utils.load_audio`)

ffmpeg is not installed or not on your `PATH`. Install it:

- Linux: `apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: <https://www.gyan.dev/ffmpeg/builds/>

### `Pipeline produced no audio output.`

The pipeline ran but produced no files. This usually means an upstream
separator failed silently. Check the server logs (stderr) for the
actual error. Common causes:

- Missing separator models (see above).
- ffmpeg not installed.
- The YouTube URL is invalid or geo-restricted.
- Out of memory — try a shorter song or switch to CPU inference.

---

## Performance issues

### Cover generation is very slow

- **Use a GPU.** CPU inference is 10–50× slower. Check
  `torch.cuda.is_available()` — if it returns `False`, install the
  CUDA-enabled torch wheel.
- **Pick a faster F0 method.** `pm` and `dio` are fastest but less
  accurate. `rmvpe+` (the default) is a good balance. `mangio-crepe`
  with a small hop length is the slowest.
- **Increase `crepe_hop_length`.** Larger values (256, 512) are faster
  but slightly less pitch-accurate.
- **Disable `keep_files`** if you don't need intermediate stems — saves
  disk I/O during cleanup.

### Out of memory (CUDA)

- Reduce `index_rate` to 0 (disables faiss retrieval, which is
  memory-heavy).
- Lower `filter_radius` (smaller median filter window).
- Use a shorter input song.
- Restart the server between runs (frees GPU memory held by previous
  `net_g` instances).

### The web UI is slow to respond

The Gradio UI runs the pipeline synchronously in a single event loop.
If you're generating a long cover, the UI will be unresponsive until
it's done. Use the REST API (`app.py`) for non-blocking generation —
you can submit the request and poll the connection for the audio
response.

---

## Still stuck?

1. Read [PIPELINE.md](PIPELINE.md) to understand what each stage does.
2. Check the server logs (stderr) — most errors print a full traceback.
3. Open an issue on GitHub with the full error message, your OS,
   Python version, and torch version (`python -c "import torch; print(torch.__version__)"`).
