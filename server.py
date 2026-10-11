"""CoverGen backend built on gradio.Server.

Architecture follows the official Gradio announcement:
https://huggingface.co/blog/introducing-gradio-server

- gradio.Server extends FastAPI: custom routes, middleware and static files
  work natively, while Gradio's API engine adds queuing, SSE streaming,
  concurrency control and gradio_client compatibility.
- @app.api() endpoints are queued and callable through the Gradio JS client
  and through gradio_client.
- Plain FastAPI REST routes live alongside @app.api() endpoints:
    GET  /api/models          -> JSON list of voice models
    POST /api/cover           -> BINARY AUDIO (the generated AI cover file)
    POST /api/models/download -> JSON status message
    POST /api/models/upload   -> JSON status message
  POST /api/cover returns the actual audio file (Content-Type audio/mpeg,
  audio/flac, ...) - never JSON - so any client (browser, curl, mobile app)
  receives playable audio directly.

Run:
    python server.py --host 0.0.0.0 --port 7860
"""

import json
import os
import shutil
import sys
import threading
import urllib.parse
import urllib.request
import zipfile
from argparse import ArgumentParser
from pathlib import Path

from fastapi import File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server
from gradio.data_classes import FileData

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
STATIC_DIR = BASE_DIR / 'static'
RVC_MODELS_DIR = BASE_DIR / 'rvc_models'
OUTPUT_DIR = BASE_DIR / 'song_output'

# The inference pipeline lives in src/ and uses intra-package imports.
sys.path.insert(0, str(SRC_DIR))

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

# Serialise heavy generation jobs (mirrors gradio's default_concurrency_limit=1).
_pipeline_lock = threading.Lock()

app = Server(
    title='CoverGen API',
    description='AI song cover generation backend (MDX vocal separation + RVC voice conversion).',
    version='2.0.0',
)


def list_voice_models() -> list:
    """Return the names of all installed RVC voice model directories."""
    return sorted(
        item for item in os.listdir(RVC_MODELS_DIR)
        if item not in ITEMS_TO_EXCLUDE and os.path.isdir(RVC_MODELS_DIR / item)
    )


def extract_zip(extraction_folder: Path, zip_name: str) -> None:
    """Extract a zipped RVC model and flatten its .pth/.index files."""
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        shutil.rmtree(extraction_folder, ignore_errors=True)
        raise ValueError(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    os.rename(model_filepath, extraction_folder / os.path.basename(model_filepath))
    if index_filepath:
        os.rename(index_filepath, extraction_folder / os.path.basename(index_filepath))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(extraction_folder / filepath):
            shutil.rmtree(extraction_folder / filepath)


def download_online_model_logic(url: str, dir_name: str) -> str:
    """Download and install a zipped RVC model from a public URL."""
    if not url or not dir_name:
        raise ValueError('Both the model download link and the model name are required.')

    extraction_folder = RVC_MODELS_DIR / dir_name
    if extraction_folder.exists():
        raise ValueError(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

    zip_name = url.split('/')[-1]
    if 'huggingface.co' in url:
        urllib.request.urlretrieve(url, zip_name)
    elif 'pixeldrain.com' in url:
        zip_name = f'{dir_name}.zip'
        urllib.request.urlretrieve(f'https://pixeldrain.com/api/file/{zip_name}', zip_name)
    elif 'drive.google.com' in url:
        from gdown import download as gdown_download
        zip_name = f'{dir_name}.zip'
        gdown_download(id=url.split('/')[-2], output=zip_name, quiet=True)
    else:
        raise ValueError('Unsupported download link. Use HuggingFace, Pixeldrain or Google Drive URLs.')

    extract_zip(extraction_folder, zip_name)
    return f'[+] Model {dir_name} successfully downloaded!'


def filedata_path(file_data) -> str | None:
    """Extract the local file path from an api() FileData argument.

    Depending on the gradio version the value arrives either as a pydantic
    FileData object or as a plain dict (as in the official gradio.Server
    examples: file_data["path"]).
    """
    if file_data is None:
        return None
    if isinstance(file_data, dict):
        return file_data.get('path')
    return getattr(file_data, 'path', None)


# ============================================================
# Gradio API endpoints (queued, gradio_client compatible)
# ============================================================

@app.api(name='get_models')
def get_models() -> list:
    """List all available RVC voice models."""
    return list_voice_models()


@app.api(name='generate_cover')
def generate_cover(
    voice_model: str,
    song_input: str = '',
    audio_file: FileData = None,
    pitch: int = 0,
    keep_files: bool = False,
    main_gain: int = 0,
    backup_gain: int = 0,
    inst_gain: int = 0,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    f0_method: str = 'rmvpe+',
    crepe_hop_length: int = 128,
    protect: float = 0.33,
    output_format: str = 'mp3',
    f0autotune: bool = False,
    f0_min: int = 50,
    f0_max: int = 1100,
) -> list:
    """Generate an AI cover from a YouTube link, a local path or an uploaded file.

    Returns [ai_cover, converted_vocals, dereverbed_vocals, backup_vocals, instrumentals].
    """
    result_paths = run_cover_pipeline(
        voice_model=voice_model,
        song_input=song_input,
        uploaded_path=filedata_path(audio_file),
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
    return [
        FileData(path=str(result_path))
        for result_path in result_paths
        if result_path and os.path.exists(str(result_path))
    ]


@app.api(name='download_online_model')
def download_online_model(url: str, dir_name: str) -> str:
    """Download and install a zipped RVC voice model from a public URL."""
    return download_online_model_logic(url, dir_name)


@app.api(name='upload_local_model')
def upload_local_model(zip_file: FileData, dir_name: str) -> str:
    """Install an RVC voice model from a locally uploaded zip file."""
    if not dir_name:
        raise ValueError('Give your uploaded model a unique name.')
    zip_path = filedata_path(zip_file)
    if not zip_path:
        raise ValueError('No zip file was received.')
    extraction_folder = RVC_MODELS_DIR / dir_name.strip()
    if extraction_folder.exists():
        raise ValueError(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

    extract_zip(extraction_folder, zip_path)
    return f'[+] Model {dir_name} successfully uploaded!'


# ============================================================
# Shared pipeline runner (used by both api() and REST routes)
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
        raise ValueError('Provide a YouTube link, a local file path, or upload an audio file.')
    if not voice_model:
        raise ValueError('Select a voice model first.')
    if voice_model not in list_voice_models():
        raise ValueError(f'Voice model "{voice_model}" was not found. Refresh the model list.')

    # MDX-Net separation models are not auto-downloaded; fail fast with a clear message.
    missing_mdx = [
        name for name in ('Kim_Vocal_2.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx')
        if not (BASE_DIR / 'mdxnet_models' / name).exists()
    ]
    if missing_mdx:
        raise ValueError(
            'Missing MDX-Net separation model(s): '
            + ', '.join(missing_mdx)
            + '. Run "python src/download_models.py" first.'
        )

    output_format = output_format.lower() if output_format else 'mp3'
    if output_format not in AUDIO_MEDIA_TYPES:
        raise ValueError(f'Unsupported output format "{output_format}". Use one of: {", ".join(AUDIO_MEDIA_TYPES)}.')

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
# REST endpoints (plain FastAPI routes on the gradio.Server)
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
            {'ok': False, 'message': 'Send the zip bytes as the request body and the model name as ?dir_name=.'},
            status_code=400,
        )
    extraction_folder = RVC_MODELS_DIR / name
    if extraction_folder.exists():
        return JSONResponse(
            {'ok': False, 'message': f'Voice model directory {name} already exists! Choose a different name.'},
            status_code=400,
        )
    temp_zip = BASE_DIR / f'__upload_{name}.zip'
    try:
        temp_zip.write_bytes(zip_bytes)
        extract_zip(extraction_folder, str(temp_zip))
        return JSONResponse({'ok': True, 'message': f'[+] Model {name} successfully uploaded!'})
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

    Accepts multipart/form-data (fields mirror the gradio api() endpoint,
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
            return JSONResponse({'ok': False, 'message': 'Uploaded audio file is empty.'}, status_code=400)
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
        return JSONResponse({'ok': False, 'message': f'Generation failed: {error}'}, status_code=500)
    finally:
        if uploaded_path and uploaded_path.startswith(str(BASE_DIR / '__upload_song')):
            Path(uploaded_path).unlink(missing_ok=True)

    if not result_paths:
        return JSONResponse({'ok': False, 'message': 'Pipeline produced no audio output.'}, status_code=500)

    cover_path = Path(result_paths[0])
    media_type = AUDIO_MEDIA_TYPES.get(cover_path.suffix.lstrip('.').lower(), 'application/octet-stream')

    # List intermediate stem files (result_paths[1:]) as relative URLs.
    stems = []
    for stem_path in result_paths[1:]:
        relative = Path(stem_path).relative_to(BASE_DIR).as_posix()
        stems.append(f'/song_output/{urllib.parse.quote(relative[len("song_output/"):])}')

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
    parser = ArgumentParser(description='CoverGen backend server (gradio.Server mode).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Hostname the server will use.')
    parser.add_argument('--port', type=int, default=7860, help='Listening port the server will use.')
    parser.add_argument('--share', action='store_true', help='Create a public Gradio share link.')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        share=args.share,
    )
