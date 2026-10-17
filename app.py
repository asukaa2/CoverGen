"""CoverGen backend server (FastAPI + uvicorn).

This module exposes:
  * A REST API under ``/api/*`` for listing models, uploading / downloading
    voice models, and generating AI covers.
  * A small static frontend served from ``/`` (HTML/CSS/JS in ``rvc/static``).

The cover generation pipeline lives in :mod:`core` and is shared between
this REST API and the optional Gradio web UI in ``webui.py``.
"""

import json
import os
import shutil
import sys
import threading
import urllib.parse
from argparse import ArgumentParser
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable so ``from core import song_cover_pipeline``
# and the ``rvc.*`` packages work no matter what directory the server is
# launched from.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# The vendored ``audio_separator`` package lives under ``rvc/modules/`` and
# uses top-level ``audio_separator.*`` imports internally, so expose its
# parent directory on sys.path.
sys.path.insert(0, str(BASE_DIR / 'rvc' / 'modules'))

STATIC_DIR = BASE_DIR / 'rvc' / 'static'
RVC_MODELS_DIR = BASE_DIR / 'assets' / 'rvc_models'
OUTPUT_DIR = BASE_DIR / 'assets' / 'song_output'
SEPARATOR_MODELS_DIR = BASE_DIR / 'assets' / 'separator_models'

# Make sure the runtime directories exist before the pipeline uses them.
RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEPARATOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

from core import song_cover_pipeline  # noqa: E402

ITEMS_TO_EXCLUDE = {'hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt'}

# Media types for the binary audio REST response.
AUDIO_MEDIA_TYPES = {
    'mp3': 'audio/mpeg',
    'flac': 'audio/flac',
    'wav': 'audio/wav',
    'ogg': 'audio/ogg',
    'm4a': 'audio/mp4',
    'aac': 'audio/aac',
}

# Serialise heavy generation jobs so concurrent requests don't blow up GPU
# memory (mirrors gradio's default_concurrency_limit=1).
_pipeline_lock = threading.Lock()

app = FastAPI(
    title='CoverGen API',
    description='AI song cover generation backend (MDX vocal separation + RVC voice conversion).',
    version='2.0.0',
)


def list_voice_models() -> list:
    """Return the names of all installed RVC voice model directories."""
    if not RVC_MODELS_DIR.exists():
        return []
    return sorted(
        item.name
        for item in RVC_MODELS_DIR.iterdir()
        if item.name not in ITEMS_TO_EXCLUDE and item.is_dir()
    )


def extract_zip(extraction_folder: Path, zip_name: str) -> None:
    """Extract a zipped RVC model and flatten its .pth/.index files."""
    import zipfile

    extraction_folder = Path(extraction_folder)
    extraction_folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    try:
        os.remove(zip_name)
    except OSError:
        pass

    index_filepath, model_filepath = None, None
    for root, _dirs, files in os.walk(extraction_folder):
        for name in files:
            full = os.path.join(root, name)
            if name.endswith('.index') and os.stat(full).st_size > 1024 * 100:
                index_filepath = full
            if name.endswith('.pth') and os.stat(full).st_size > 1024 * 1024 * 40:
                model_filepath = full

    if not model_filepath:
        shutil.rmtree(extraction_folder, ignore_errors=True)
        raise ValueError(
            f'No .pth model file was found in the extracted zip. '
            f'Please check {extraction_folder}.'
        )

    shutil.move(model_filepath, str(extraction_folder / os.path.basename(model_filepath)))
    if index_filepath:
        shutil.move(index_filepath, str(extraction_folder / os.path.basename(index_filepath)))

    for item in extraction_folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)


def download_online_model_logic(url: str, dir_name: str) -> str:
    """Download and install a zipped RVC model from a public URL."""
    import urllib.request

    if not url or not dir_name:
        raise ValueError('Both the model download link and the model name are required.')

    extraction_folder = RVC_MODELS_DIR / dir_name
    if extraction_folder.exists():
        raise ValueError(
            f'Voice model directory {dir_name} already exists! '
            f'Choose a different name for your voice model.'
        )

    zip_name = url.split('/')[-1]
    if 'huggingface.co' in url:
        urllib.request.urlretrieve(url, zip_name)
    elif 'pixeldrain.com' in url:
        zip_name = f'{dir_name}.zip'
        urllib.request.urlretrieve(
            f'https://pixeldrain.com/api/file/{dir_name}.zip', zip_name
        )
    elif 'drive.google.com' in url:
        try:
            from gdown import download as gdown_download
        except ImportError as exc:
            raise ValueError(
                'Google Drive downloads require the "gdown" package. '
                'Install it with "pip install gdown".'
            ) from exc
        zip_name = f'{dir_name}.zip'
        gdown_download(id=url.split('/')[-2], output=zip_name, quiet=True)
    else:
        raise ValueError(
            'Unsupported download link. Use HuggingFace, Pixeldrain or Google Drive URLs.'
        )

    extract_zip(extraction_folder, zip_name)
    return f'[+] Model {dir_name} successfully downloaded!'


# ============================================================
# Shared pipeline runner (used by all REST routes)
# ============================================================

def run_cover_pipeline(
    *,
    voice_model: str,
    song_input: str,
    uploaded_path: str | None,
    pitch: int,
    keep_files: bool,
    main_gain: int,
    backup_gain: int,
    inst_gain: int,
    index_rate: float,
    filter_radius: int,
    rms_mix_rate: float,
    f0_method: str,
    crepe_hop_length: int,
    protect: float,
    output_format: str,
    f0autotune: bool,
    f0_min: int,
    f0_max: int,
) -> list:
    """Validate inputs and run the full cover pipeline. Returns result paths."""
    resolved_song_input = (song_input or '').strip()
    if uploaded_path:
        resolved_song_input = uploaded_path

    if not resolved_song_input:
        raise ValueError(
            'Provide a YouTube link, a local file path, or upload an audio file.'
        )
    if not voice_model:
        raise ValueError('Select a voice model first.')
    if voice_model not in list_voice_models():
        raise ValueError(
            f'Voice model "{voice_model}" was not found. Refresh the model list.'
        )

    output_format = (output_format or 'mp3').lower()
    if output_format not in AUDIO_MEDIA_TYPES:
        raise ValueError(
            f'Unsupported output format "{output_format}". '
            f'Use one of: {", ".join(AUDIO_MEDIA_TYPES)}.'
        )

    with _pipeline_lock:
        result_paths = song_cover_pipeline(
            resolved_song_input,
            voice_model,
            pitch,
            keep_files,
            is_webui=0,
            main_gain=main_gain,
            backup_gain=backup_gain,
            inst_gain=inst_gain,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            f0_method=f0_method,
            crepe_hop_length=crepe_hop_length,
            protect=protect,
            output_format=output_format,
            progress=None,
            f0autotune=f0autotune,
            f0_min=f0_min,
            f0_max=f0_max,
        )
    return [p for p in result_paths if p and os.path.exists(str(p))]


# ============================================================
# REST endpoints (plain FastAPI routes)
# ============================================================

@app.get('/api/models')
async def rest_list_models():
    """REST: list installed voice models as plain JSON."""
    return JSONResponse({'models': list_voice_models()})


@app.post('/api/models/download')
async def rest_download_model(
    url: str = Form(''),
    dir_name: str = Form(''),
):
    """REST: download and install a zipped voice model."""
    try:
        message = download_online_model_logic(url.strip(), dir_name.strip())
        return JSONResponse({'ok': True, 'message': message})
    except Exception as error:
        return JSONResponse({'ok': False, 'message': str(error)}, status_code=400)


@app.post('/api/models/upload')
async def rest_upload_model(request: Request, dir_name: str = ''):
    """REST: install an uploaded zip file as a voice model.

    The raw request body must be the zip bytes; the model name is passed
    either as the ?dir_name= query parameter or the X-Model-Name header.
    """
    name = (dir_name or request.headers.get('X-Model-Name', '')).strip()
    zip_bytes = await request.body()
    if not zip_bytes or not name:
        return JSONResponse(
            {
                'ok': False,
                'message': 'Send the zip bytes as the request body and the model name as ?dir_name=.',
            },
            status_code=400,
        )
    extraction_folder = RVC_MODELS_DIR / name
    if extraction_folder.exists():
        return JSONResponse(
            {
                'ok': False,
                'message': f'Voice model directory {name} already exists! Choose a different name.',
            },
            status_code=400,
        )
    temp_zip = BASE_DIR / f'__upload_{name}.zip'
    try:
        temp_zip.write_bytes(zip_bytes)
        extract_zip(extraction_folder, str(temp_zip))
        return JSONResponse(
            {'ok': True, 'message': f'[+] Model {name} successfully uploaded!'}
        )
    except Exception as error:
        shutil.rmtree(extraction_folder, ignore_errors=True)
        return JSONResponse({'ok': False, 'message': str(error)}, status_code=400)
    finally:
        temp_zip.unlink(missing_ok=True)


@app.post('/api/cover')
async def rest_generate_cover(
    voice_model: str = Form(''),
    song_input: str = Form(''),
    pitch: int = Form(0),
    keep_files: bool = Form(False),
    main_gain: int = Form(0),
    backup_gain: int = Form(0),
    inst_gain: int = Form(0),
    index_rate: float = Form(0.5),
    filter_radius: int = Form(3),
    rms_mix_rate: float = Form(0.25),
    f0_method: str = Form('rmvpe+'),
    crepe_hop_length: int = Form(128),
    protect: float = Form(0.33),
    output_format: str = Form('mp3'),
    f0autotune: bool = Form(False),
    f0_min: int = Form(50),
    f0_max: int = Form(1100),
    audio_file: UploadFile | None = File(None),
):
    """REST: generate an AI cover and return the BINARY AUDIO file itself.

    Accepts multipart/form-data (fields mirror the gradio API endpoint,
    with an optional uploaded audio file in the "audio_file" field) or a
    standard form body. The response body is the actual generated cover
    audio (audio/mpeg, audio/flac or audio/wav) - never JSON.

    Stem files (when keep_files=true) are served from /song_output/ and
    their relative URLs are listed in the X-Stems response header as a
    URL-encoded JSON array.
    """
    uploaded_path = None
    if audio_file is not None:
        suffix = Path(getattr(audio_file, 'filename', '') or 'upload.audio').suffix or '.audio'
        temp_upload = BASE_DIR / f'__upload_song{suffix}'
        content = await audio_file.read()
        if not content:
            return JSONResponse(
                {'ok': False, 'message': 'Uploaded audio file is empty.'},
                status_code=400,
            )
        temp_upload.write_bytes(content)
        uploaded_path = str(temp_upload)

    try:
        result_paths = run_cover_pipeline(
            voice_model=voice_model,
            song_input=song_input,
            uploaded_path=uploaded_path,
            pitch=pitch,
            keep_files=keep_files,
            main_gain=main_gain,
            backup_gain=backup_gain,
            inst_gain=inst_gain,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            f0_method=f0_method,
            crepe_hop_length=crepe_hop_length,
            protect=protect,
            output_format=output_format,
            f0autotune=f0autotune,
            f0_min=f0_min,
            f0_max=f0_max,
        )
    except ValueError as error:
        return JSONResponse({'ok': False, 'message': str(error)}, status_code=400)
    except Exception as error:
        return JSONResponse(
            {'ok': False, 'message': f'Generation failed: {error}'},
            status_code=500,
        )
    finally:
        if uploaded_path and uploaded_path.startswith(str(BASE_DIR / '__upload_song')):
            Path(uploaded_path).unlink(missing_ok=True)

    if not result_paths:
        return JSONResponse(
            {'ok': False, 'message': 'Pipeline produced no audio output.'},
            status_code=500,
        )

    cover_path = Path(result_paths[0])
    media_type = AUDIO_MEDIA_TYPES.get(
        cover_path.suffix.lstrip('.').lower(), 'application/octet-stream'
    )

    # List intermediate stem files (result_paths[1:]) as relative URLs.
    stems = []
    for stem_path in result_paths[1:]:
        try:
            relative = Path(stem_path).relative_to(OUTPUT_DIR).as_posix()
            stems.append(f'/song_output/{urllib.parse.quote(relative)}')
        except ValueError:
            # Stem lives outside the public output dir - skip it.
            continue

    headers = {
        'X-Output-Filename': urllib.parse.quote(cover_path.name),
        'X-Stems': urllib.parse.quote(json.dumps(stems)),
    }
    return FileResponse(
        path=str(cover_path),
        media_type=media_type,
        filename=cover_path.name,
        headers=headers,
    )


# ============================================================
# Frontend + static assets
# ============================================================

@app.get('/', response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    """Serve the custom HTML/CSS/JS frontend."""
    return (STATIC_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / 'favicon.svg')


# Static assets (style.css, app.js) and generated stems served directly.
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
app.mount('/song_output', StaticFiles(directory=str(OUTPUT_DIR)), name='song_output')


if __name__ == '__main__':
    import uvicorn

    parser = ArgumentParser(description='CoverGen backend server (FastAPI + uvicorn).')
    parser.add_argument(
        '--host', type=str, default='127.0.0.1',
        help='Hostname the server will bind to (default: 127.0.0.1).',
    )
    parser.add_argument(
        '--port', type=int, default=7860,
        help='Listening port (default: 7860).',
    )
    parser.add_argument(
        '--reload', action='store_true',
        help='Enable uvicorn auto-reload (for development).',
    )
    args = parser.parse_args()

    uvicorn.run(
        'app:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
