# The cover-generation pipeline

This document describes how `core.song_cover_pipeline()` turns an input
song into a finished AI cover. Use it as a reference if you want to
extend the pipeline, swap out a stage, or debug a failure.

## High-level flow

```
Input (YouTube URL / local path / uploaded file)
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ song_cover_pipeline()  (core.py)                         │
│   ├── resolve input type (yt / local)                    │
│   ├── compute song_id (YouTube ID or file hash)          │
│   ├── if song_output/<id>/ exists → reuse cached stems   │
│   │                                                       │
│   └── preprocess_song()                                  │
│         ├── [yt only] yt_download()                      │
│         ├── convert_to_stereo() (ffmpeg)                 │
│         ├── [1] Separator: vocals / instrumental split   │
│         ├── [2] Separator: vocal dereverb                │
│         └── [3] Separator: main / backup vocal split     │
│                                                           │
│   ├── voice_change()  (RVC inference)                    │
│   │     ├── Config (device, half precision, gpu_mem)     │
│   │     ├── get_vc() — load .pth, build net_g            │
│   │     ├── load HuBERT/ContentVec embedder              │
│   │     ├── VC.pipeline()                                │
│   │     │     ├── F0 estimation (rmvpe / fcpe / crepe…)  │
│   │     │     ├── feature extraction via HuBERT          │
│   │     │     ├── faiss index retrieval (if .index)      │
│   │     │     └── synthesizer net_g.infer()              │
│   │     └── wavfile.write(ai_vocals.wav)                 │
│   │                                                       │
│   └── combine_audio()                                    │
│         ├── main vocal (-4 dB + main_gain)               │
│         ├── backup vocal (-6 dB + backup_gain)           │
│         ├── instrumental (-7 dB + inst_gain)             │
│         └── pydub overlay → export mp3/flac/wav          │
└──────────────────────────────────────────────────────────┘
                  │
                  ▼
       assets/song_output/<song_id>/
       ├── <song>_<voice>_converted_voice.wav
       └── <song> (<voice> Ver).mp3
```

## Stage 1 — Input resolution

`song_cover_pipeline(song_input, voice_model, pitch_change, ...)` first
decides what kind of input it received:

- If `urlparse(song_input).scheme == 'https'` → YouTube mode. The video
  ID is extracted via `get_youtube_video_id()`, and the audio is
  downloaded by `yt_download()` using `yt_dlp` with the
  `FFmpegExtractAudio` postprocessor (output is always mp3).
- Otherwise → local-file mode. The path is stripped of stray quotes,
  and `get_hash()` (BLAKE2b, 11-char hex) is used as the song ID.

The song ID doubles as the cache key: all stems for a given input are
stored under `assets/song_output/<song_id>/`. Reusing a song ID skips
re-separation on subsequent runs (unless `keep_files=False` deletes the
stems first).

## Stage 2 — Stereo conversion

`convert_to_stereo(audio_path)` uses `librosa.load(mono=False)` to
check if the input is mono. If so, ffmpeg is invoked with `-ac 2` to
upmix to stereo. The MDX/Roformer separators expect stereo audio.

## Stage 3 — Three-pass separation

All three separations use the vendored `audio_separator.Separator`
class with `model_file_dir=assets/separator_models/`. The pipeline
passes `output_dir=<song_output_dir>` and `output_format='wav'` to each
`Separator` instance.

### Pass 1 — Vocal / instrumental split

- Model: `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (BS-Roformer).
- Output: `*_Vocals.wav` + `*_Instrumental.wav`.

### Pass 2 — Vocal dereverb

- Model: `UVR-DeEcho-DeReverb.pth` (VR architecture).
- Input: the `*_Vocals.wav` from pass 1.
- Output: `*_Vocals_Main_DeReverb.wav` (the "No Reverb" stem).

### Pass 3 — Main / backup vocal split

- Model: `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`.
- Input: `*_Vocals_Main_DeReverb.wav`.
- Output: `*_Vocals_Main.wav` + `*_Vocals_Backup.wav`.

The pipeline uses `_find_output_file(output_list, keyword)` to robustly
identify which output file is which — separator models don't always
return stems in a fixed order, so checking the filename for `"vocal"`,
`"instrument"`, `"no reverb"`, `"back"`, `"lead"` etc. is more reliable
than positional indexing.

## Stage 4 — RVC voice conversion

`voice_change(voice_model, vocals_path, output_path, ...)` is where the
actual voice swap happens.

1. **Load the voice model** — `get_rvc_model()` finds the `.pth` (and
   optional `.index`) inside `assets/rvc_models/<voice_model>/`.
2. **Build the RVC config** — `Config(device, True)` auto-detects CUDA
   / MPS / CPU, sets half-precision based on the GPU family (16xx / 10xx
   / P40 are forced to FP32), and computes per-device padding constants.
3. **Instantiate the synthesizer** — `get_vc()` reads the checkpoint
   dict, picks `SynthesizerTrnMs256NSFsid[_nono]` (v1) or
   `SynthesizerTrnMs768NSFsid[_nono]` (v2) based on the checkpoint's
   `version` and `f0` flags, loads the weights, and moves the model to
   the chosen device.
4. **Load the embedder** — `tools/fairseq.py:load_model()` loads the
   HuBERT/ContentVec `.pt` file (default: `hubert_base.pt`) via
   `torch.hub`, or falls back to the official `fairseq` package.
5. **Run inference** — `rvc_infer()` calls `VC.pipeline()`, which:
   - Loads the input audio at 16 kHz mono.
   - Optionally loads the feature index via `faiss.read_index()`.
   - High-pass filters the audio at 48 Hz (removes rumble).
   - Pads the audio and splits it into chunks at low-energy boundaries.
   - For each chunk: extracts HuBERT features, optionally blends in
     indexed features via `faiss.search`, computes F0 with the chosen
     method, applies autotune if enabled, and synthesizes with
     `net_g.infer(feats, p_len, pitch, pitchf, sid)`.
   - Concatenates the chunks, applies RMS matching to preserve the
     original vocal loudness, and writes the result to `output_path`.

### F0 method normalization

The UI uses names like `rmvpe+` and `mangio-crepe`; the underlying
`Generator` class expects `rmvpe` and `mangio-crepe-full`.
`VC.normalize_f0_method()` translates between the two. Hybrid methods
like `hybrid[rmvpe+mangio-crepe]` are parsed by `VC.get_f0()`, which
runs each sub-tracker independently and averages their F0 curves.

### Index handling

If the `.index` file is missing, empty, or fails to load, the pipeline
prints a `[WARNING]` and continues without index retrieval (i.e. pure
HuBERT features). The cover will still be generated but with slightly
less timbre accuracy. Re-download or re-upload the model's `.index`
file to restore full quality.

## Stage 5 — Mixing

`combine_audio([ai_vocals, backup_vocals, instrumentals], output_path,
gains, format)` uses `pydub.AudioSegment`:

- Main AI vocals: `-4 dB + main_gain`.
- Backup vocals: `-6 dB + backup_gain`.
- Instrumentals: `-7 dB + inst_gain`.

The three stems are overlaid with `.overlay()` and exported to the
requested format via `.export(format=...)`. `pydub` requires ffmpeg for
mp3 export.

## Caching and intermediate files

By default (`keep_files=False`), only the final mix and the converted
vocals are kept in `assets/song_output/<song_id>/`. The intermediate
`*_Vocals.wav` and `*_Vocals_Main.wav` are deleted. With
`keep_files=True`, everything stays on disk, which lets you re-run the
mixing stage without re-separating.

## Extending the pipeline

Common extension points:

- **Add a new separation stage** — instantiate another `Separator`,
  call `load_model()` and `separate()`, and add the result path to the
  return list of `preprocess_song()`.
- **Add a new F0 method** — extend the `compute_f0` dispatch table in
  `rvc/modules/predictor/generator.py` and (if needed) the
  `normalize_f0_method` alias map in `rvc/infer/pipeline.py`.
- **Add a new REST endpoint** — define a new `@app.post('/api/...')`
  function in `app.py`. Use `_pipeline_lock` for any endpoint that runs
  the full pipeline.
- **Use a different embedder** — pass a different `embedder_name`
  argument to `rvc_infer()` (e.g. `contentvec_base`, `japanese_hubert_base`).
  The chosen embedder must exist in `models/<name>.pt` or
  `assets/rvc_models/<name>.pt`.
