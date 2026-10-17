import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import yt_dlp

from pydub import AudioSegment

from rvc.audio_separator.separator import Separator

from rvc.infer.infer import Config, get_vc, rvc_infer

BASE_DIR = os.getcwd()

separator_models_dir = os.path.join(BASE_DIR, 'assets', 'separator_models')
rvc_models_dir = os.path.join(BASE_DIR, 'assets', 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'assets', 'song_output')

# Model filenames used by python-audio-separator.
# Place these files inside `assets/separator_models` (or let the library download them).
VOCAL_INST_MODEL = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
DEREVERB_MODEL = 'UVR-DeEcho-DeReverb.pth'
KARAOKE_MODEL = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'


def get_youtube_video_id(url, ignore_playlist=True):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    return None


def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'The directory {model_dir} is missing a model file.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def display_progress(message, percent, is_webui, progress=None):
    if is_webui and progress is not None:
        try:
            progress(percent, desc=message)
        except Exception:
            # Progress tracking is unavailable outside a Gradio event context.
            print(message)
    else:
        print(message)


def _find_output_file(output_files, keyword):
    """Pick the file from a separator output list whose name contains `keyword`."""
    for f in output_files:
        if keyword.lower() in os.path.basename(f).lower():
            return f
    return None


def _rename(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)


def preprocess_song(song_input, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    os.makedirs(song_output_dir, exist_ok=True)
    orig_song_path = convert_to_stereo(orig_song_path)

    orig_basename = os.path.splitext(os.path.basename(orig_song_path))[0]

    # 1) Split vocals from instrumental
    display_progress('[~] Separating vocals from instrumental...', 0.1, is_webui, progress)
    separator = Separator(output_dir=song_output_dir, output_format='wav')
    separator.load_model(model_filename=VOCAL_INST_MODEL)
    voc_inst = separator.separate(orig_song_path)

    # Depending on model, order is (Vocals, Instrumental) or (Instrumental, Vocals).
    vocals_src = _find_output_file(voc_inst, 'vocal')
    inst_src = _find_output_file(voc_inst, 'instrument')
    if vocals_src is None:
        vocals_src = voc_inst[0]
    if inst_src is None:
        inst_src = voc_inst[1]

    vocals_path = os.path.join(song_output_dir, f'{orig_basename}_Vocals.wav')
    instrumentals_path = os.path.join(song_output_dir, f'{orig_basename}_Instrumental.wav')
    _rename(vocals_src, vocals_path)
    _rename(inst_src, instrumentals_path)

    # 2) DeReverb vocals
    display_progress('[~] Applying DeReverb to vocals...', 0.3, is_webui, progress)
    separator = Separator(output_dir=song_output_dir, output_format='wav')
    separator.load_model(model_filename=DEREVERB_MODEL)
    voc_no_reverb = separator.separate(vocals_path)

    # UVR-DeEcho-DeReverb returns (No Reverb, Reverb)
    no_reverb_src = _find_output_file(voc_no_reverb, 'no reverb')
    if no_reverb_src is None:
        no_reverb_src = voc_no_reverb[0]
    reverb_src = _find_output_file(voc_no_reverb, 'reverb')
    if reverb_src is not None and reverb_src == no_reverb_src:
        reverb_src = None

    main_vocals_dereverb_path = os.path.join(song_output_dir, f'{orig_basename}_Vocals_Main_DeReverb.wav')
    _rename(no_reverb_src, main_vocals_dereverb_path)

    # Keep/remove the leftover reverb file
    if reverb_src is not None and os.path.exists(reverb_src):
        os.remove(reverb_src)

    # 3) Split main vocals vs backup vocals
    display_progress('[~] Splitting main vocals and backup vocals...', 0.2, is_webui, progress)
    separator = Separator(output_dir=song_output_dir, output_format='wav')
    separator.load_model(model_filename=KARAOKE_MODEL)
    backing_voc = separator.separate(main_vocals_dereverb_path)

    backing_src = _find_output_file(backing_voc, 'back')
    if backing_src is None:
        backing_src = backing_voc[0]
    lead_src = _find_output_file(backing_voc, 'lead')
    if lead_src is None:
        lead_src = backing_voc[1]

    backup_vocals_path = os.path.join(song_output_dir, f'{orig_basename}_Vocals_Backup.wav')
    main_vocals_path = os.path.join(song_output_dir, f'{orig_basename}_Vocals_Main.wav')
    _rename(backing_src, backup_vocals_path)
    _rename(lead_src, main_vocals_path)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, f0autotune, f0_min, f0_max, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = Config(device, True)

    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g,
              filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, config, 'hubert_base',
              rvc_models_dir, f0autotune, f0_min, f0_max)

    gc.collect()


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files, is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, f0_method='rmvpe',
                        crepe_hop_length=128, protect=0.33, output_format='mp3', progress=None, f0autotune=False, f0_min=50, f0_max=1100):

    try:
        if not song_input or not voice_model:
            raise_exception('Make sure the song input field and voice model field are filled in.', is_webui)

        display_progress('[~] Starting AI cover generation pipeline...', 0, is_webui, progress)

        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Invalid YouTube URL.'
                raise_exception(error_msg, is_webui)
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} does not exist.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, song_id, is_webui, input_type, progress)
        else:
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir)

            if any(path is None for path in paths) or keep_files:
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, song_id, is_webui, input_type, progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

        ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}_{voice_model}_converted_voice.wav')
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{output_format}')

        if os.path.exists(ai_vocals_path):
            os.remove(ai_vocals_path)
        if os.path.exists(ai_cover_path):
            os.remove(ai_cover_path)

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting vocals...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate,
                         filter_radius, rms_mix_rate, protect, crepe_hop_length, f0autotune, f0_min, f0_max, is_webui)

        display_progress('[~] Combining AI vocals and instrumental...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_path, backup_vocals_path, instrumentals_path], ai_cover_path, main_gain, backup_gain, inst_gain, output_format)

        intermediate_files = [vocals_path, main_vocals_path]

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return [ai_cover_path, ai_vocals_path, main_vocals_dereverb_path, backup_vocals_path, instrumentals_path]


    except Exception as e:
        raise_exception(str(e), is_webui)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoverGen V3.', add_help=True)
    parser.add_argument('-i', '--song-input', type=str, required=True, help='Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)')
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction, help='Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals')
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5, help='A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset')
    parser.add_argument('-fr', '--filter-radius', type=int, default=3, help='A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.')
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25, help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe', help='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128, help='If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. The higher the value, the faster the conversion and less risk of voice cracks, but there is less pitch accuracy. Recommended: 128.')
    parser.add_argument('-pro', '--protect', type=float, default=0.33, help='A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.')
    parser.add_argument('-mv', '--main-vol', type=int, default=0, help='Volume change for AI main vocals in decibels. Use -3 to decrease by 3 decibels and 3 to increase by 3 decibels')
    parser.add_argument('-bv', '--backup-vol', type=int, default=0, help='Volume change for backup vocals in decibels')
    parser.add_argument('-iv', '--inst-vol', type=int, default=0, help='Volume change for instrumentals in decibels')
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0, help='Not used by the pipeline anymore; kept for CLI compatibility')
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3', help='Output format of audio file. mp3 for smaller file size, wav for best quality')
    parser.add_argument('-minpitch', '--minimum-pitch', type=int, default=50, help='Specify minimal pitch for inference [HZ]')
    parser.add_argument('-maxpitch', '--maximum-pitch', type=int, default=1100, help='Specify max pitch for inference [HZ]')
    parser.add_argument('-autotune', '--autotune', type=argparse.BooleanOptionalAction, help='Enable autotune')
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} was not found.')

    cover_path = song_cover_pipeline(args.song_input, rvc_dirname, args.pitch_change, args.keep_files,
                                     main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
                                     index_rate=args.index_rate, filter_radius=args.filter_radius,
                                     rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
                                     crepe_hop_length=args.crepe_hop_length, protect=args.protect,
                                     output_format=args.output_format, f0autotune=args.autotune,
                                     f0_min=args.minimum_pitch, f0_max=args.maximum_pitch)
    print(f'[+] Cover created at {cover_path}')