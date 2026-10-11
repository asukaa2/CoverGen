"""CoverGen backend built on gradio.Server.

Architecture follows the official Gradio announcement:
https://huggingface.co/blog/introducing-gradio-server

- gradio.Server extends FastAPI: custom routes, middleware and static files
  work natively, while Gradio's API engine adds queuing, SSE streaming,
  concurrency control and gradio_client compatibility.
- @app.api() endpoints are queued and callable through the Gradio JS client
  (used by the custom HTML/CSS/JS frontend) and through gradio_client.
- @app.get("/") serves the custom frontend instead of a Gradio UI.

Run:
    python server.py --host 0.0.0.0 --port 7860
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from argparse import ArgumentParser
from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse
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

app = Server(
    title='CoverGen API',
    description='AI song cover generation backend (MDX vocal separation + RVC voice conversion).',
    version='1.0.0',
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
    resolved_song_input = (song_input or '').strip()
    uploaded_path = filedata_path(audio_file)
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


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    """Serve the custom HTML/CSS/JS frontend."""
    return (STATIC_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / 'favicon.svg')


# Static assets (style.css, app.js) served as regular FastAPI static files.
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


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
