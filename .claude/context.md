# tEODor Project Context

## Project Overview
tEODor is a tool for muting profanity (F-bombs, etc.) in video and audio content using AI transcription (Whisper) and FFMPEG.

## Current System
- CPU: Intel i5-14500
- GPU: Tesla P4 (8GB VRAM, Pascal architecture sm_61)
- Connection: Gigabit fiber

## Current Branch
`feature/subtitle-enhanced-detection` - Contains GPU support and subtitle-enhanced detection

## Latest Updates (2025-12-14)

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

**CLI Options:**
- `--subtitle-only` - Only process files that have subtitles with profanity
- `--no-subtitle-enhance` - Disable subtitle enhancement (Whisper only)
- `--ignore-subtitles` - Ignore subtitles entirely
- `--output-transcription` - Save transcription to file for debugging

**Profanity Patterns:**
- Catches compound words (motherfucker, bullshit, etc.)
- Patterns: fuck*, *fucker, shit*, *shit, n-word variants

## Files Overview

- `defuse.py` - Main script for video files (GPU-accelerated)
- `defuse-audio-only.py` - Audio-only processing (CPU with faster-whisper)
- `test_subtitle_detection.py` - Unit tests (11 tests)
- `requirements.txt` - Dependencies

## Dependencies
- faster-whisper>=1.0.0 (uses CTranslate2 for GPU)
- pysrt>=1.1.2
- ffmpeg-python>=0.2.0

## Important Notes
- FFmpeg must be installed separately on system
- Script automatically detects CUDA and switches between CPU/GPU
- Video streams are copied (not re-encoded), only audio is processed
- Tesla P4 uses int8 compute (not float16) due to Pascal architecture limitations
