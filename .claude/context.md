# tEODor Project Context

## Project Overview
tEODor is a tool for muting profanity (F-bombs, etc.) in video and audio content using AI transcription (Whisper) and FFMPEG.

## Current System
- CPU: Intel i5-14500
- GPU: Tesla P4 (8GB VRAM, Pascal architecture sm_61)
- Connection: Gigabit fiber

## Current Branch
`main` - All features merged

## Latest Updates (2026-03-10)

### v1.3.0 - Directory input support for batch processing

- `defuse -i` now accepts directories in addition to individual files (e.g. `defuse -i Season*/`)
- Directories are expanded to all video files inside them, sorted alphabetically
- Supported extensions: `.mkv`, `.mp4`, `.avi`, `.m4v`, `.mov`, `.wmv`, `.flv`, `.webm`, `.ts`, `.mpg`, `.mpeg`
- Already-defused (`-CLEAN`) files are automatically skipped
- `VIDEO_EXTENSIONS` constant added to `defuse.py`
- Version bumped to 1.3.0

### Previous: v1.2.0 - Custom profanity words via --swears flag

- Added `--swears word1 word2 ...` flag to all three commands (`defuse`, `defuse-audio`, `mask-subtitles`)
- Users can now mute additional words beyond the hardcoded f-word
- `word_to_pattern()` helper converts plain words to flexible regex patterns with letter dedup (e.g. "goddamn"/"godamn" produce the same pattern, catches compound forms like "goddamnit")
- Patterns threaded through all detection/masking functions via `patterns=None` parameter
- 6 new unit tests for `word_to_pattern()` (19 total)
- Version bumped to 1.2.0

### Previous: v1.1.3 - Fix setuptools and clarify output

- Pin `setuptools>=68.0,<76` — versions 76+ removed `pkg_resources` as a top-level module, breaking `webrtcvad`/ffsubsync on Python 3.13. Now installed correctly via `pipx install --force`.
- Changed misleading "using subtitle timing" messages to "using estimated word timing from subtitles" to clarify that character interpolation narrows the mute window to the word position, not the whole subtitle line.

### v1.1.2 - Fix GPU OOM and transcription output

- Lowered default `--batch-size` from 8 to 6 (Tesla P4 OOM'd with 8 on `large-v3`)
- Transcription file (`--output-transcription`) now writes progressively during segment processing instead of after completion — partial output saved even on crash

### v1.1.1 - Fix model compatibility and Python 3.13 support

- **Switched default model from CrisperWhisper to `large-v3`**: `BatchedInferencePipeline` produces broken word-level timestamps with CrisperWhisper (garbled tokens, 3/29 detected on Zombieland). `large-v3` works correctly (24/29 detected). CrisperWhisper still usable via `--model nyrahealth/faster_CrisperWhisper --batch-size 0`.
- Added `setuptools` to dependencies for `pkg_resources` (Python 3.13)

### v1.1.0 - Batched GPU inference and cleaner output

- `BatchedInferencePipeline` for parallel segment processing (~5 min transcription for a movie)
- `--batch-size` CLI flag (default 6, use 0 for sequential)
- Suppressed verbose ffmpeg output, fixed transcription timing measurement
- Added `language="en"` to skip language detection

### v1.0.1 - Fix ffsubsync detection in pipx installs

- `_find_ffsubsync()` helper checks Python's bin directory first (pipx venv), then system PATH

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
- `-i` / `--input` - Input video files or directories containing video files
- `--subtitle-only` - Only process files that have subtitles with profanity
- `--no-subtitle-enhance` - Disable subtitle enhancement (Whisper only)
- `--ignore-subtitles` - Ignore subtitles entirely
- `--output-transcription` - Save transcription to file for debugging
- `--preserve-original` - Keep original file after creating clean version
- `--batch-size N` - Batch size for parallel GPU inference (default 6, 0 to disable)
- `--no-sync-check` - Disable ffsubsync subtitle sync verification
- `--model NAME` - Whisper model to use (default: large-v3)
- `--swears WORD ...` - Additional profanity words to mute (builds flexible regex patterns)

**Profanity Patterns:**
- Hardcoded: f-word (always active)
- User-extensible via `--swears`: `word_to_pattern()` builds `\w*<letter>+...\w*` regexes
- Catches compound words (motherfucker, bullshit, goddamnit, etc.)

## Files Overview

- `defuse.py` - Main script for video files (GPU-accelerated)
- `defuse-audio-only.py` - Audio-only processing (GPU-accelerated)
- `mask-subtitles.py` - Standalone subtitle profanity masking (no video/audio processing)
- `test_subtitle_detection.py` - Unit tests (19 tests)
- `requirements.txt` - Dependencies

## Dependencies
- faster-whisper>=1.1.0 (uses CTranslate2 for GPU, BatchedInferencePipeline)
- pysrt>=1.1.2
- ffmpeg-python>=0.2.0
- ffsubsync>=0.4.0
- setuptools>=68.0,<76 (provides pkg_resources for webrtcvad on Python 3.13+)

## Important Notes
- FFmpeg must be installed separately on system
- Script automatically detects CUDA and switches between CPU/GPU
- Video streams are copied (not re-encoded), only audio is processed
- Tesla P4 uses int8 compute (not float16) due to Pascal architecture limitations
- **Install/upgrade:** `pipx install --force git+https://github.com/ktwingstrom/tEODor.git` (use `--force` to ensure dependency pins are applied)
- CrisperWhisper model does NOT work with BatchedInferencePipeline (broken word timestamps); use `large-v3` (default) or run CrisperWhisper with `--batch-size 0`

## Zombieland Test Results (2026-02-14)
- Model: large-v3, batch_size=6, Tesla P4 int8
- Transcription: ~5 min (309s)
- Whisper detected: 24/29, Subtitle fallbacks: 6, Refined: 3/6
- Total muted: 30
- Subtitle character interpolation correctly narrows mute windows to estimated word position
