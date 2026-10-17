# REST API reference

CoverGen's FastAPI backend (`app.py`) exposes four endpoints under `/api`.
All of them accept and return JSON, except `/api/cover` which returns a
binary audio file as the response body.

Interactive OpenAPI docs are also served at `/docs` once the server is
running.

---

## `GET /api/models`

List the voice models currently installed under `assets/rvc_models/`.

### Response

```json
{
  "models": ["my-voice", "another-voice"]
}
```

### Example

```bash
curl http://127.0.0.1:7860/api/models
```

---

## `POST /api/models/download`

Download and install a zipped voice model from a public URL. The zip must
contain a `.pth` file (≥ 40 MB) and may optionally include an `.index`
file.

### Request body (`application/x-www-form-urlencoded` or `multipart/form-data`)

| Field      | Type   | Required | Description                              |
|------------|--------|----------|------------------------------------------|
| `url`      | string | yes      | HuggingFace, Pixeldrain, or Google Drive URL. |
| `dir_name` | string | yes      | Unique name for the new voice model folder. |

### Response

```json
{ "ok": true, "message": "[+] Model my-voice successfully downloaded!" }
```

On failure the response has HTTP 400 and a `message` field describing the
error.

### Example

```bash
curl -X POST http://127.0.0.1:7860/api/models/download \
     -F "url=https://huggingface.co/<user>/<repo>/resolve/main/model.zip" \
     -F "dir_name=my-voice"
```

---

## `POST /api/models/upload`

Install a voice model from a zip file uploaded directly in the request
body. The model name is passed as a query parameter or as the
`X-Model-Name` header.

### Request

- **Body**: raw zip file bytes (`Content-Type: application/zip`).
- **Query**: `?dir_name=<name>` (or header `X-Model-Name: <name>`).

### Response

```json
{ "ok": true, "message": "[+] Model my-voice successfully uploaded!" }
```

### Example

```bash
curl -X POST "http://127.0.0.1:7860/api/models/upload?dir_name=my-voice" \
     --data-binary @model.zip \
     -H "Content-Type: application/zip"
```

---

## `POST /api/cover`

Generate an AI cover. The response body is the generated audio file
itself (`audio/mpeg`, `audio/flac`, or `audio/wav`). Stem URLs are listed
in the `X-Stems` response header as a URL-encoded JSON array.

### Request body (`multipart/form-data`)

| Field              | Type    | Default  | Description                                              |
|--------------------|---------|----------|----------------------------------------------------------|
| `voice_model`      | string  | required | Name of the voice model folder.                          |
| `song_input`       | string  | `""`     | YouTube URL or local file path. Mutually exclusive with `audio_file`. |
| `audio_file`       | file    | optional | Uploaded audio file. Overrides `song_input`.            |
| `pitch`            | int     | `0`      | Pitch shift in semitones (−24 to +24).                  |
| `keep_files`       | bool    | `false`  | Keep intermediate stems in `assets/song_output/`.       |
| `output_format`    | string  | `mp3`    | `mp3`, `flac`, or `wav`.                                 |
| `f0_method`        | string  | `rmvpe+` | Pitch detection method (see F0 methods below).          |
| `index_rate`       | float   | `0.5`    | Timbre leakage control (0–1).                            |
| `filter_radius`    | int     | `3`      | Median filter radius (0–7).                              |
| `rms_mix_rate`     | float   | `0.25`   | Loudness preservation (0=original, 1=fixed).            |
| `protect`          | float   | `0.33`   | Voiceless-consonant protection (0–0.5; 0.5 disables).   |
| `crepe_hop_length` | int     | `128`    | Hop length for crepe-based trackers.                    |
| `f0autotune`       | bool    | `false`  | Apply pitch correction.                                  |
| `f0_min`           | int     | `50`     | Min pitch (Hz).                                          |
| `f0_max`           | int     | `1100`   | Max pitch (Hz).                                          |
| `main_gain`        | int     | `0`      | Main vocals volume (dB).                                 |
| `backup_gain`      | int     | `0`      | Backup vocals volume (dB).                               |
| `inst_gain`        | int     | `0`      | Instrumentals volume (dB).                               |

### F0 methods

- `rmvpe+` (recommended) — best clarity.
- `rmvpe`, `fcpe`, `rmvpe-legacy`, `fcpe-legacy`.
- `mangio-crepe`, `crepe-full` / `large` / `medium` / `small` / `tiny`.
- `pm`, `harvest`, `dio`, `yin`, `pyin`.
- Hybrids: `hybrid[rmvpe+fcpe]`, `hybrid[rmvpe+mangio-crepe]`,
  `hybrid[mangio-crepe+rmvpe]`, `hybrid[mangio-crepe+fcpe]`,
  `hybrid[mangio-crepe+rmvpe+fcpe]`.

### Response

- **200 OK** — body is the audio file. Headers:
  - `Content-Type: audio/mpeg` (or `audio/flac`, `audio/wav`).
  - `Content-Disposition: attachment; filename="..."`.
  - `X-Output-Filename`: URL-encoded final filename.
  - `X-Stems`: URL-encoded JSON array of stem URLs (only present when
    `keep_files=true` and intermediate stems exist).
- **400 Bad Request** — JSON `{ "ok": false, "message": "..." }` for
  validation errors (missing model, bad output format, etc.).
- **500 Internal Server Error** — JSON `{ "ok": false, "message": "..." }`
  for pipeline failures.

### Examples

Generate from a YouTube URL:

```bash
curl -X POST http://127.0.0.1:7860/api/cover \
     -F "voice_model=my-voice" \
     -F "song_input=https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
     -F "pitch=-12" \
     -F "output_format=mp3" \
     -o cover.mp3
```

Generate from an uploaded file, keeping intermediate stems:

```bash
curl -X POST http://127.0.0.1:7860/api/cover \
     -F "voice_model=my-voice" \
     -F "audio_file=@song.mp3" \
     -F "pitch=-12" \
     -F "keep_files=true" \
     -D headers.txt \
     -o cover.mp3
# headers.txt now contains the X-Stems list, e.g.:
# X-Stems: %5B%22%2Fsong_output%2Fabcd%2Fsong_Vocals.wav%22%5D
```

Decode the `X-Stems` header:

```bash
python -c "import urllib.parse, json; print(json.loads(urllib.parse.unquote('%5B%22%2Fsong_output%2Fabcd%2Fsong_Vocals.wav%22%5D')))"
# ['/song_output/abcd/song_Vocals.wav']
```

---

## Error conventions

All JSON error responses follow this shape:

```json
{ "ok": false, "message": "<human-readable description>" }
```

Validation errors (missing fields, unknown model, etc.) return HTTP 400.
Internal pipeline errors return HTTP 500. Network/IO errors during model
downloads return HTTP 400.

## Rate limiting / concurrency

CoverGen serializes all cover-generation jobs with a single
`threading.Lock` (see `_pipeline_lock` in `app.py`). Concurrent
`POST /api/cover` requests will queue — only one cover is generated at a
time. This protects GPU memory and prevents ffmpeg file collisions.

If you need parallel generation, run multiple `app.py` processes on
different ports and load-balance in front of them.
