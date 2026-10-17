# Voice & separator models

CoverGen needs three kinds of model files:

| Kind | Purpose | Default location |
|------|---------|------------------|
| **RVC voice models** | The "target voice" the AI sings in. One folder per voice. | `assets/rvc_models/<voice_name>/` |
| **Separator models** | MDX / Roformer / VR weights used to split vocals, dereverb, etc. | `assets/separator_models/` |
| **Pitch + embedder models** | F0 trackers (rmvpe, fcpe, crepe) + HuBERT/ContentVec embedder. Auto-downloaded. | `models/` at project root |

---

## 1. RVC voice models

Each voice lives in its own folder under `assets/rvc_models/`:

```
assets/rvc_models/
└── my-voice/
    ├── my-voice.pth          # required, RVC v2 weights (≥ 40 MB)
    └── my-voice.index        # optional, faiss feature index
```

### Where to find them

The RVC community publishes trained voice models in many places. Some
popular starting points:

- The [RVC v2 models collection](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/hubert_base)
  on HuggingFace (mostly demos + the HuBERT base embedder).
- Community Discords / subreddits dedicated to AI cover generation
  (search for "RVC models" + the voice you want).
- Train your own — see the [RVC v2 GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  for training instructions.

### Installing a voice model

**Manual**: unzip the downloaded archive, create a folder under
`assets/rvc_models/`, and copy the `.pth` (and optional `.index`) into
it. Click "Refresh Models List" in the web UI (or call
`GET /api/models`) to see it appear.

**Via the web UI**: open the "Download model" tab, paste the zip URL
(supports HuggingFace, Pixeldrain, Google Drive), give the model a
unique name, and click "Download & Install Model".

**Via the REST API**:

```bash
# From a URL
curl -X POST http://127.0.0.1:7860/api/models/download \
     -F "url=https://huggingface.co/.../model.zip" \
     -F "dir_name=my-voice"

# From a local zip
curl -X POST "http://127.0.0.1:7860/api/models/upload?dir_name=my-voice" \
     --data-binary @model.zip \
     -H "Content-Type: application/zip"
```

### Zip file requirements

CoverGen walks the zip recursively looking for:

- A `.pth` file **larger than 40 MB** — this is the RVC v2 weights file.
- An optional `.index` file **larger than 100 KB** — this is the faiss
  feature index used for timbre retrieval.

Small `.pth` files (e.g. test checkpoints) are skipped to avoid
accidentally picking up the wrong file. After extraction, all
non-`.pth`/`.index` files and nested folders are deleted, leaving a
flat folder with just the model + index.

---

## 2. Separator models

The pipeline expects three specific files in `assets/separator_models/`:

| Filename | Purpose | Architecture |
|----------|---------|--------------|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | Vocal / instrumental split | BS-Roformer |
| `UVR-DeEcho-DeReverb.pth` | Vocal dereverb | VR |
| `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | Main / backup vocal split | Mel-Band Roformer |

### Where to find them

These are NOT auto-downloaded. Download them from:

- The [python-audio-separator releases page](https://github.com/nomadkaraoke/python-audio-separator/releases)
  (look for the `model-configs` asset — it bundles model YAMLs).
- The [UVR model archive](https://github.com/TRvlvr/model_scheme/releases).
- Direct HuggingFace mirrors (search for the filename above).

Copy the three files into `assets/separator_models/`:

```
assets/separator_models/
├── model_bs_roformer_ep_317_sdr_12.9755.ckpt
├── UVR-DeEcho-DeReverb.pth
└── mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
```

### Using different separator models

If you want to swap in a different separator (e.g. a newer vocal split
model), edit the three constants at the top of `core.py`:

```python
VOCAL_INST_MODEL = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
DEREVERB_MODEL   = 'UVR-DeEcho-DeReverb.pth'
KARAOKE_MODEL    = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
```

Then drop the new file into `assets/separator_models/`. The pipeline
will use it on the next run.

---

## 3. Pitch trackers + embedder (auto-downloaded)

The first time `rvc_infer()` runs, it calls `check_predictors()` and
`check_embedders()` (in `rvc/modules/predictor/utils.py`), which
download the required files into a `models/` directory at the project
root:

```
models/
├── rmvpe.pt                    # RMVPE pitch tracker
├── fcpe.pt                     # FCPE pitch tracker (and fcpe_legacy.pt)
├── crepe_full.pth              # CREPE pitch trackers (also large/medium/small/tiny)
└── hubert_base.pt              # HuBERT / ContentVec embedder
```

These come from the
[`IVrganzrfr-EIP-Project`](https://huggingface.co/) HuggingFace repo
(URL is ROT13-encoded in `utils.py` for legacy reasons). You need
internet access on first run. After that, the files are cached and no
further downloads are needed.

### Manually placing embedder files

If you're offline or want a specific embedder, drop the `.pt` file into
either `models/` or `assets/rvc_models/`. The `resolve_embedder_path()`
function in `rvc/infer/infer.py` checks both locations in that order.

The default embedder name is `hubert_base`. To use a different one (e.g.
`contentvec_base`), modify the call to `rvc_infer()` in `core.py`:

```python
rvc_infer(..., embedder_name='contentvec_base', ...)
```

Available embedders: `contentvec_base`, `hubert_base`,
`japanese_hubert_base`, `korean_hubert_base`, `chinese_hubert_base`,
`portuguese_hubert_base`, `spin`.

---

## Verifying your setup

After installing all models, run:

```bash
python -c "
import os
sep = 'assets/separator_models'
rvc = 'assets/rvc_models'
print('Separator models:', sorted(os.listdir(sep)))
print('RVC voice models:', sorted(d for d in os.listdir(rvc) if os.path.isdir(os.path.join(rvc, d))))
"
```

You should see the three separator files and at least one voice model
folder. Then start the server and call `GET /api/models` to confirm the
voice is detected.
