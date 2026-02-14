# tEODor Project Context

## Project Overview
tEODor is a tool for muting profanity (F-bombs, etc.) in video and audio content using AI transcription (Whisper) and FFMPEG.

## Current System
- CPU: Intel i5-14500
- GPU: Tesla P4 (8GB VRAM, Pascal architecture sm_61)
- Connection: Gigabit fiber

## Current Branch
`main` - All features merged

## Latest Updates (2026-02-13)

### v1.1.0 - Batched GPU inference and cleaner output

Added `BatchedInferencePipeline` from faster-whisper to process multiple audio segments in parallel on the GPU, giving ~3-4x transcription speedup. Previously sequential processing was not fully utilizing GPU bandwidth — a full movie took ~1 hour for transcription, now expected ~15-20 minutes.

**Changes:**
- Wrap `WhisperModel` in `BatchedInferencePipeline` for parallel segment processing
- New `--batch-size` CLI flag (default 8, use 0 to fall back to sequential)
- Fixed misleading transcription timing — was only measuring model load, now measures actual transcription
- Added `language="en"` to all transcribe calls (skips language detection)
- Suppressed verbose ffmpeg output from extract/mute steps (capture_output)
- Removed giant filter string dump from mute_audio
- Final remux uses `-loglevel warning -stats` to show only progress line
- Bumped `faster-whisper` dependency to `>=1.1.0`

**Files changed:** `teodor/defuse.py`, `pyproject.toml`
**Tag:** `v1.1.0` pushed to GitHub

**To update:** `pipx upgrade teodor`

**Notes:**
- Tesla P4 (8GB): start with `--batch-size 8`, try `--batch-size 4` if OOM
- Batched mode processes segments independently (no context passing), WER may be very slightly higher
- Re-analysis of subtitle-missed clips still uses raw model (clip_timestamps not compatible with batched)

---

### v1.0.1 - Fix ffsubsync detection in pipx installs

When installed globally via `pipx install git+https://github.com/ktwingstrom/tEODor.git`, ffsubsync was not detected because it lives inside pipx's isolated venv and isn't on the system PATH.

**Fix:** Added `_find_ffsubsync()` helper that checks the running Python's bin directory first (finding it inside the pipx venv), then falls back to system PATH lookup. Also cleaned up redundant inline imports (`re`, `shutil`).

**Files changed:** `teodor/defuse.py`, `pyproject.toml`
**Tag:** `v1.0.1` pushed to GitHub

---

## Previous Updates (2026-02-03)

### Subtitle Masking Integrated into Pipeline

The standalone `mask-subtitles.py` functionality is now integrated into `defuse.py`. After successfully creating a clean video, subtitles are automatically masked.

**Pipeline Flow:**
1. Process video → create `video-CLEAN.mkv` (muted audio)
2. If subtitles exist → create `video-CLEAN.srt` (masked profanity)
3. Clean up intermediate files (including extracted `_subtitles.srt`)

**New Functions:**
- `mask_subtitle_file()` - Masks profanity in subtitle files, handles multiple encodings
- `mask_for_log()` - Masks profanity in log output (displays `f**k` instead of full word)

**Log Output:**
- All log messages now display masked profanity for cleaner output
- Example: `Whisper found: 'f**king' at 123.45s` instead of the full word

---

## Previous Updates (2026-01-27)

### New Script: mask-subtitles.py

Added a standalone script for masking profanity in subtitle files without processing video/audio.

**Features:**
- Handles multiple encodings (UTF-8, UTF-8-BOM, Latin-1, CP1252)
- Preserves original encoding when writing
- Matches "fuck" variations (fuck, fucking, fucker, motherfucker, etc.)
- Replaces with asterisks matching the length of matched word
- Creates backup when using `--in-place` flag

**Usage:**
```bash
python3 mask-subtitles.py -i subtitle.srt                    # Creates subtitle-CLEAN.srt
python3 mask-subtitles.py -i subtitle.srt -o output.srt      # Specify output
python3 mask-subtitles.py -i subtitle.srt --in-place         # Modify in place (creates .bak)
```

**Test Results:**
- Tested on Pluribus S01E01 subtitles
- Found and masked 18 instances of profanity
- Encoding preserved correctly

---

## Previous Updates (2026-01-15)

### Bug Fixes and Improvements

**1. English Subtitle Stream Selection**
- Added `find_english_subtitle_stream()` function to properly select English subtitles
- Previously grabbed first subtitle stream (often wrong language like Czech forced subs)
- Now uses ffprobe to find `eng`/`en` language tracks
- Prefers full subtitles over forced subtitles
- Falls back to first subtitle if no English found

**2. `--preserve-original` Flag**
- Added flag to keep original file after creating clean version
- Default behavior: delete original (normal workflow)
- With flag: keep original (useful for testing)

**Test Results:**
- Tested on Pluribus S01E01 (56 min, 60+ subtitle tracks)
- English subtitle stream correctly found at index 28
- Whisper detected: 11 profanity instances
- Subtitles found: 28 profanity instances
- Subtitle fallbacks added: 17 (ones Whisper missed)
- Total muted: 28

---

## Previous Updates (2025-12-14)

### GPU Support Added to Audio-Only Script

Added the same GPU acceleration to `defuse-audio-only.py` that was already in `defuse.py`.

**Changes:**
- Added `load_whisper_model()` function with CUDA auto-detection
- Updated `transcribe_audio()` to use the new function
- Now uses Tesla P4 GPU with int8 compute type for faster audiobook processing

Both scripts now have identical GPU support.

---

## Previous Updates (2025-12-02)

### Tesla P4 GPU Support Added

Successfully added GPU acceleration for the Tesla P4.

**Key Discovery:**
- PyTorch 2.5+ dropped support for Pascal GPUs (sm_61 like Tesla P4)
- Python 3.13 doesn't have PyTorch 2.4.x wheels available
- `insanely-fast-whisper` requires PyTorch, so it won't work with Tesla P4

**Solution: faster-whisper with CTranslate2**
- CTranslate2 fully supports Pascal GPUs including Tesla P4
- Uses int8 compute type (float16 requires Volta sm_70+)
- No PyTorch dependency needed for GPU inference

**New `load_whisper_model()` function:**
- Auto-detects CUDA via `ctranslate2.get_supported_compute_types("cuda")`
- Selects optimal compute type: float16 → int8 → float32 (in order of preference)
- Falls back to CPU with int8 if no CUDA available

**GPU Status:**
- Device: Tesla P4 detected and working
- Compute type: int8
- All 11 unit tests passing

### Subtitle-Enhanced Profanity Detection

Implemented a multi-pass detection system that uses SRT subtitles as a reference guide to catch profanity that Whisper might miss.

**How It Works:**
1. Full transcription pass with Whisper (precise word timestamps)
2. Parse SRT subtitles for known profanity locations
3. Merge results - use Whisper timing when available, subtitle timing as fallback
4. For missed words, perform targeted re-analysis on subtitle time windows
5. If targeted analysis fails, use subtitle timing with small buffer

**Key Functions:**
- `load_whisper_model()` - Load model with auto GPU/CPU detection
- `transcribe_audio()` - Main transcription with subtitle enhancement
- `parse_srt_for_profanity()` - Extract profanity with timing from SRT files
- `merge_profanity_results()` - Combine Whisper and subtitle detections
- `subtitle_guided_transcription()` - Targeted re-analysis of missed segments
- `check_subtitles_for_profanity()` - Quick check for profanity in subtitles
- `get_subtitle_file_path()` - Handle external and embedded subtitles
- `find_english_subtitle_stream()` - Find best English subtitle track in multi-track files
- `mask_subtitle_file()` - Mask profanity in subtitle files (integrated pipeline)
- `mask_for_log()` - Mask profanity for log output display

**CLI Options:**
- `--subtitle-only` - Only process files that have subtitles with profanity
- `--no-subtitle-enhance` - Disable subtitle enhancement (Whisper only)
- `--ignore-subtitles` - Ignore subtitles entirely
- `--output-transcription` - Save transcription to file for debugging
- `--preserve-original` - Keep original file after creating clean version
- `--batch-size N` - Batch size for parallel GPU inference (default 8, 0 to disable)
- `--no-sync-check` - Disable ffsubsync subtitle sync verification
- `--model NAME` - Whisper model to use (default: nyrahealth/faster_CrisperWhisper)

**Profanity Patterns:**
- Catches compound words (motherfucker, bullshit, etc.)
- Patterns: fuck*, *fucker, shit*, *shit, n-word variants

## Files Overview

- `defuse.py` - Main script for video files (GPU-accelerated)
- `defuse-audio-only.py` - Audio-only processing (GPU-accelerated)
- `mask-subtitles.py` - Standalone subtitle profanity masking (no video/audio processing)
- `test_subtitle_detection.py` - Unit tests (11 tests)
- `requirements.txt` - Dependencies

## Dependencies
- faster-whisper>=1.1.0 (uses CTranslate2 for GPU, BatchedInferencePipeline)
- pysrt>=1.1.2
- ffmpeg-python>=0.2.0

## Important Notes
- FFmpeg must be installed separately on system
- Script automatically detects CUDA and switches between CPU/GPU
- Video streams are copied (not re-encoded), only audio is processed
- Tesla P4 uses int8 compute (not float16) due to Pascal architecture limitations
