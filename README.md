# tEODor

Mute profanity in video and audio files using AI-powered transcription.

tEODor uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2-based Whisper implementation) to transcribe audio, locate profanity with word-level timestamps, and mute those specific moments using FFmpeg. The original video and all other audio are untouched -- only the offending words are silenced.

## How It Works

1. **Subtitle pre-check** (video mode) -- If subtitles are available (embedded or external `.srt`), tEODor scans them first. If no profanity is found in subtitles, it can skip transcription entirely (`--subtitle-only` mode).
2. **Audio extraction** -- The audio track is extracted from the video container, preserving the original codec and channel layout.
3. **AI transcription** -- faster-whisper transcribes the audio with word-level timestamps to pinpoint exactly where profanity occurs.
4. **Subtitle-enhanced detection** -- When subtitles are available, tEODor cross-references Whisper results with subtitle text. If subtitles indicate profanity that Whisper missed, it does a targeted second-pass transcription on those segments. This catches words that full-file transcription sometimes misses.
5. **Precision muting** -- FFmpeg applies volume-zero filters at the exact timestamps of each detected word.
6. **Clean output** -- For video, the muted audio is added as a second track labeled "Defused (CLEAN) Track". For audio-only files, the original is replaced with the clean version.

## Installation

Requires **Python 3.10+** and **FFmpeg** installed on your system.

### pipx (recommended)

```bash
pipx install git+https://github.com/ktwingstrom/tEODor.git
```

### pip

```bash
pip install git+https://github.com/ktwingstrom/tEODor.git
```

### From source

```bash
git clone https://github.com/ktwingstrom/tEODor.git
cd tEODor
pip install -e .
```

### GPU / CUDA Setup

tEODor automatically uses GPU acceleration when CUDA is available. If you need a specific CUDA version of PyTorch, install it before tEODor:

```bash
python3 -m venv ~/.venvs/teodor
source ~/.venvs/teodor/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/ktwingstrom/tEODor.git
```

## Usage

### `defuse` -- Mute profanity in video files

```bash
defuse -i movie.mkv
defuse -i episode1.mkv episode2.mkv episode3.mkv
```

Creates a `-CLEAN` version of each file with the muted audio added as a second audio track.

**Flags:**

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input video file(s) (required) |
| `--ignore-subtitles` | Skip subtitle detection entirely |
| `--subtitle-only` | Only process files that have profanity in subtitles |
| `--no-subtitle-enhance` | Disable subtitle-enhanced detection (Whisper only) |
| `--no-sync-check` | Disable ffsubsync subtitle sync verification |
| `--preserve-original` | Keep the original file (default: delete after creating clean version) |
| `--output-transcription` | Save transcription to a text file for debugging |
| `--model MODEL` | Whisper model to use (default: `nyrahealth/faster_CrisperWhisper`) |

### `defuse-audio` -- Mute profanity in audio files

```bash
defuse-audio -i audiobook.mp3
```

Creates a `-CLEAN` version of the audio file.

**Flags:**

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input audio file (required) |

Supports long files (audiobooks, podcasts) with automatic chunking for files over 2 hours.

### `mask-subtitles` -- Mask profanity in subtitle files

```bash
mask-subtitles -i subtitle.srt
mask-subtitles -i subtitle.srt -o clean.srt
mask-subtitles -i subtitle.srt --in-place
```

Replaces profanity with asterisks (e.g., "fucking" becomes "****ing") while preserving subtitle structure and encoding.

**Flags:**

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input subtitle file (required) |
| `-o`, `--output` | Output file (default: `input-CLEAN.srt`) |
| `--in-place` | Modify the file in place (creates `.bak` backup) |

## Batch Processing

Process an entire directory of video files:

```bash
defuse -i /path/to/shows/Season01/*
```

Or use the included shell scripts for recursive directory processing:

```bash
./defuse-all.sh -i /path/to/shows/
./defuse-all-audio.sh -i /path/to/audiobooks/
```

## System Requirements

- **Python** >= 3.10
- **FFmpeg** and **FFprobe** -- must be installed and on your PATH
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **ffsubsync** (optional) -- for subtitle sync verification. Installed automatically as a dependency.

## Updating

```bash
# pipx
pipx upgrade teodor

# pip
pip install --upgrade git+https://github.com/ktwingstrom/tEODor.git
```

## License

[GPL-3.0](LICENSE)
