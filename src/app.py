import json
import os
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr

from core import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in models_list if item not in items_to_remove and os.path.isdir(os.path.join(models_dir, item))]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
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
        raise gr.Error(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'huggingface.co' in url:
            urllib.request.urlretrieve(url, zip_name)
        elif 'pixeldrain.com' in url:
            zip_name = dir_name + '.zip'
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)
        elif 'drive.google.com' in url:
            zip_name = dir_name + '.zip'
            file_id = url.split('/')[-2]
            output = os.path.join('.', f'{dir_name}.zip')
            gdown.download(id=file_id, output=output, quiet=False)

        progress(0.5, desc='[~] Extracting zip file...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} successfully downloaded!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        # In Gradio 5+/6 the uploaded file is passed as a plain filepath string.
        zip_name = zip_path if isinstance(zip_path, str) else zip_path.name
        progress(0.5, desc='[~] Extracting zip file...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} successfully uploaded!'

    except Exception as e:
        raise gr.Error(str(e))

def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)

def process_file_upload(file):
    # In Gradio 5+/6 the uploaded file is passed as a plain filepath string.
    file_name = file if isinstance(file, str) else file.name
    return file_name, gr.update(value=file_name)

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ['rmvpe+', 'mangio-crepe', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def show_pitch_slider(pitch_detection_algo):
    if pitch_detection_algo in ['rmvpe+']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_f0_method(use_hybrid_methods):
    if use_hybrid_methods:
        return gr.update(choices=['hybrid[rmvpe+fcpe]', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]'], value='hybrid[rmvpe+fcpe]')
    else:
        return gr.update(choices=['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Allow sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI accessible from your local network.")
    parser.add_argument('--listen-host', type=str, help='Hostname the server will use.')
    parser.add_argument('--listen-port', type=int, help='Listening port the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)

    with gr.Blocks(title='CoverGen - AsukaDev') as app:
        with gr.Tab("CoverGen"):
            with gr.Row():
                with gr.Column():
                    rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Directory "CoverGen/rvc_models". After adding new models to this directory, click the "Refresh Models List" button')
                    ref_btn = gr.Button('Refresh Models List 🔁', variant='primary')

                with gr.Column() as yt_link_col:
                    song_input = gr.Text(label='Input Song', info='Link to a song on YouTube or the full path to a local file')
                    song_input_file = gr.UploadButton('Upload File from Device', file_types=['audio'], variant='primary')

                with gr.Column(visible=False) as file_upload_col:
                    local_file = gr.File(label='Audio File')
                    show_yt_link_button = gr.Button('Insert YouTube Link / File Path')
                    song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

                with gr.Column():
                    pitch = gr.Slider(-24, 24, value=0, step=1, label='Voice Pitch Change', info='-24 - male voice || 24 - female voice')
                    f0autotune = gr.Checkbox(label="Autotune", info='Automatically corrects pitch for a more harmonious vocal sound', value=False)

            with gr.Accordion('Voice Conversion Settings', open=False):
                gr.Markdown('<center><h2>Main Settings</h2></center>')
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much accent of the AI voice to keep in the vocals. Choosing smaller values can help reduce artifacts present in the audio")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='If >=3: applies median filtering to the harvested pitch results. Can reduce breathiness')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate', info="Controls how accurately the original vocal's loudness (0) or a fixed loudness (1) is reproduced")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect Rate', info='Protects voiceless consonants and breath sounds. Increasing the parameter to the maximum value of 0.5 provides full protection')
                gr.Markdown('<center><h2>Pitch Detection Settings</h2></center>')
                with gr.Row():
                    with gr.Column():
                        use_hybrid_methods = gr.Checkbox(label="Use Hybrid Methods", value=False)
                        f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Pitch Detection Method')
                        use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                    crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=True, label='Hop Length', info='Smaller values lead to longer conversions and a higher risk of voice cracks, but better pitch accuracy')
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                    f0_min = gr.Slider(label="Minimum Pitch Range:", info="Specify the minimum pitch range for inference in Hertz. This parameter determines the lower bound of the pitch range that the algorithm will use to determine the fundamental frequency (F0) in the audio signal. (VOICE WILL BE SOFTER)", step=1, minimum=1, value=50, maximum=16000, visible=True)
                    f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_min)
                    f0_max = gr.Slider(label="Maximum Pitch Range:", info="Specify the maximum pitch range for inference in Hertz. This parameter determines the upper bound of the pitch range that the algorithm will use to determine the fundamental frequency (F0) in the audio signal. (VOICE WILL BE ROUGHER)", step=1, minimum=1, value=1100, maximum=16000, visible=True)
                    f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_max)
                keep_files = gr.Checkbox(label='Keep Intermediate Files', info='Keep all audio files created in the song_output/id directory, e.g. Isolated Vocals/Instrumentals', visible=False)

            with gr.Accordion('Audio Mixing Settings', open=False):
                gr.Markdown('<center><h2>Volume Changes (decibels)</h2></center>')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Instrumentals')

            with gr.Row():
                with gr.Column(scale=2, min_width=100):
                    generate_btn = gr.Button("Generate", variant='primary', scale=1, min_width=100)

                with gr.Column(scale=5):
                    with gr.Group():
                        ai_cover = gr.Audio(label='AI Cover')
                        with gr.Accordion("Intermediate Audio Files", open=False):
                            ai_vocals = gr.Audio(label='Converted Vocals')
                            main_vocals_dereverb = gr.Audio(label='Vocals')
                            backup_vocals = gr.Audio(label='Backup Vocals')
                            instrumentals = gr.Audio(label='Instrumentals')

                with gr.Column(scale=1, min_width=100):
                    output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Output File Type', scale=0.5)
                    clear_btn = gr.ClearButton(value='Reset All Parameters', components=[keep_files, use_hybrid_methods], min_width=100)


            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                              inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                      inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                      protect, output_format, f0autotune, f0_min, f0_max],
                              outputs=[ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])
            clear_btn.click(lambda: [0, 0.5, 3, 0.25, 0.33, 128,
                                    0, 0, 0, False, 50, 1100,
                                    None, None, None, None, None],
                            outputs=[pitch, index_rate, filter_radius, rms_mix_rate, protect,
                                    crepe_hop_length, main_gain, backup_gain, inst_gain,
                                    f0autotune, f0_min, f0_max,
                                    ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])

        with gr.Tab('Model Upload'):
            with gr.Tab('Upload via Link'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Model Download Link', info='This should be a link to a zip file containing a .pth model file and an optional .index file.', scale = 3)
                    model_name = gr.Text(label='Model Name', info='Give your uploaded model a unique name, different from other voice models.', scale = 1.5)

                with gr.Row(equal_height=True):
                    dl_output_message = gr.Text(label='Output Message', interactive=False, scale=3)
                    download_btn = gr.Button('Download Model', variant='primary', scale=1.5)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Upload Locally'):
                gr.Markdown('## Upload a locally trained RVC v2 model and index file')
                gr.Markdown('- Find the model file (weights folder) and optional index file (logs/[name] folder)')
                gr.Markdown('- Compress the files into a zip file')
                gr.Markdown('- Upload the zip file and give the voice a unique name')
                gr.Markdown('- Click the "Download Model" button')

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        zip_file = gr.File(label='Zip File')

                    with gr.Column(scale=1.5):
                        local_model_name = gr.Text(label='Model Name', info='Give your uploaded model a unique name, different from other voice models.')
                        model_upload_button = gr.Button('Download Model', variant='primary')

                with gr.Row():
                    local_upload_output_message = gr.Text(label='Output Message', interactive=False)
                    model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
