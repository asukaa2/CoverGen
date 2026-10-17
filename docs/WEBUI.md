# Gradio web UI (`webui.py`)

CoverGen ships with two frontends:

- **`app.py`** — a FastAPI server that serves a custom HTML/CSS/JS
  frontend at `/` and exposes REST endpoints at `/api/*`. This is the
  recommended default.
- **`webui.py`** — an alternative Gradio Blocks UI for users who prefer
  the classic RVC webui look. Functionally equivalent.

This document covers `webui.py`. For `app.py` see the
[top-level README](../README.md) and [API.md](API.md).

## Starting the Gradio UI

```bash
python webui.py [--share] [--listen] [--listen-host HOST] [--listen-port PORT]
```

| Flag | Description |
|------|-------------|
| `--share` | Create a public `https://<random>.gradio.live` URL (via Gradio's tunnel). |
| `--listen` | Bind to `0.0.0.0` so the UI is reachable from your LAN. |
| `--listen-host <host>` | Custom bind host (default: `0.0.0.0` when `--listen` is set). |
| `--listen-port <port>` | Custom port (default: Gradio's `7860`). |

Examples:

```bash
# Local only
python webui.py

# Reachable from the LAN on port 8080
python webui.py --listen --listen-port 8080

# Public Gradio share link (no port forwarding needed)
python webui.py --share
```

## UI layout

The Gradio UI has two tabs:

### CoverGen tab

| Section | Controls |
|---------|----------|
| **Voice Models** | Dropdown of installed models + "Refresh Models List" button. |
| **Input Song** | YouTube/file-path text field + "Upload File from Device" button. |
| **Voice Pitch Change** | Slider (−24 to +24 semitones) + autotune checkbox. |
| **Voice Conversion Settings** (collapsible) | Index rate, filter radius, RMS mix rate, protect rate, F0 method, hop length, min/max pitch. |
| **Audio Mixing Settings** (collapsible) | Main/backup/inst gain sliders. |
| **Output** | AI cover audio player + intermediate-stem players (collapsible). Format dropdown + reset button. |

### Model Upload tab

Two sub-tabs:

- **Upload via Link** — paste a HuggingFace/Pixeldrain/Google Drive URL
  + unique name → "Download Model".
- **Upload Locally** — pick a local zip + unique name → "Download
  Model" (despite the label, this uploads a local zip).

## How it differs from `app.py`

| Feature | `app.py` (FastAPI) | `webui.py` (Gradio) |
|---------|--------------------|---------------------|
| Frontend | Custom HTML/CSS/JS | Gradio Blocks components |
| API | REST `/api/*` endpoints | Gradio queue (Python-only) |
| Theme customization | Built-in theme customizer modal | Default Gradio theme |
| File upload size | No Gradio limit | Gradio's default upload limit |
| Best for | Production / programmatic access | Quick local experimentation |

Both share the same `core.song_cover_pipeline()` backend, so cover
quality is identical.

## Switching between them

You can run both at the same time on different ports — they don't
conflict (each gets its own `_pipeline_lock`, but they share the same
`assets/song_output/` directory, so use different `song_id`s if you
don't want them to collide).

```bash
# Terminal 1
python app.py --port 7860

# Terminal 2
python webui.py --listen-port 7861
```

## Common UI issues

**"Voice model directory X already exists"** — When uploading a model,
the name you chose is already taken. Pick a different name, or delete
the existing folder under `assets/rvc_models/`.

**The dropdown doesn't show my new model** — Click the "Refresh Models
List" button (CoverGen tab) or restart the server. The dropdown is
populated at startup and not auto-refreshed.

**`gradio.Error: No .pth model file was found in the extracted zip`** —
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
