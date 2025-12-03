# tEODor Project Context

## Project Overview
tEODor is a tool for muting F-bombs in video and audio content using AI transcription (Whisper) and FFMPEG.

## Recent Changes (2025-11-23)

### Migration to insanely-fast-whisper
Upgraded from `openai-whisper` to `insanely-fast-whisper` for significant performance improvements.

**Reason for Change:**
- User has Tesla P4 GPU on the way
- insanely-fast-whisper offers 10-30x speedup on GPU
- Also provides 5-10x speedup on CPU (i5-14500)

**Files Modified:**
1. `requirements.txt` - Updated dependencies
2. `defuse.py` - Rewrote `transcribe_audio()` function
3. `defuse-audio-only.py` - Same updates for consistency

**Key Configuration:**
- Uses `transformers.pipeline` API
- Disabled Flash Attention 2 (Tesla P4 doesn't support it)
- Batch size: 24 (optimized for P4's 8GB VRAM)
- torch_dtype: float16 for GPU, float32 for CPU
- Model: "openai/whisper-base"

## Current System
- CPU: Intel i5-14500
- GPU: None yet (Tesla P4 incoming)
- Connection: Gigabit fiber

## Latest Updates (2025-11-23 Evening)

### Dependency Compatibility Fixes
Fixed multiple compatibility issues with PyTorch and transformers libraries:

**Issue 1: torch/torchvision Version Mismatch**
- Problem: torch 2.8.0 incompatible with torchvision, causing `RuntimeError: operator torchvision::nms does not exist`
- Solution: Upgraded to torch 2.9.1 + torchvision 0.24.1
- All imports now work correctly

**Issue 2: Deprecated transformers API**
- Problem: `torch_dtype` parameter deprecated, should use `dtype` instead
- Problem: `model_kwargs={"use_flash_attention_2": False}` not supported in newer transformers
- Solution: Updated both `defuse.py` and `defuse-audio-only.py`:
  - Changed `torch_dtype` → `dtype`
  - Removed `model_kwargs` parameter entirely

**Issue 3: CPU Memory Allocation Error**
- Problem: `batch_size=24` too large for CPU, causing `std::length_error: vector::reserve`
- Solution: Made batch_size dynamic based on device:
  - CPU: `batch_size=1` (conservative to avoid memory issues)
  - GPU: `batch_size=24` (optimized for 8GB VRAM)
- Files updated: `defuse.py` (line 303-308), `defuse-audio-only.py` (line 70-74)

**Issue 4: CPU Chunking Memory Error**
- Problem: Even with `batch_size=1`, CPU still crashed with memory error
- Root cause: `chunk_length_s=30` parameter is experimental and incompatible with CPU processing
- Attempted Solution: Disabled chunking for CPU - Still failed

**Issue 5: CPU Word-Level Timestamp Memory Error**
- Problem: Even without chunking, CPU crashed when requesting word-level timestamps
- Root cause: Word-level timestamp extraction requires loading entire audio into memory
- Solution: Use segment-level timestamps for CPU, word-level for GPU:
  - CPU: `return_timestamps=True` (segment timestamps - less memory intensive)
  - GPU: `return_timestamps="word"` (word timestamps - more accurate)
  - Segment timestamps still provide sufficient accuracy for F-bomb detection
- Files updated: `defuse.py` (line 325-339), `defuse-audio-only.py` (line 88-102)

### Current Status
- All dependencies installed and working
- torch 2.9.1+cu128 with CUDA 12.8 support ready
- transformers 4.57.1 compatible
- Script currently running on CPU with batch_size=1
- Testing with "The Boondock Saints" (1080p, 1:48:26 duration)

## Latest Updates (2025-12-02)

### Subtitle-Enhanced Profanity Detection (feature/subtitle-enhanced-detection branch)

Implemented a multi-pass detection system that uses SRT subtitles as a reference guide to catch profanity that Whisper might miss.

**Problem Solved:**
- Whisper sometimes misses F-words during transcription
- Subtitles contain accurate text but less precise timing
- Combined approach catches more profanity with better accuracy

**How It Works:**
1. Full transcription pass with Whisper (precise word timestamps)
2. Parse SRT subtitles for known profanity locations
3. Merge results - use Whisper timing when available, subtitle timing as fallback
4. For missed words, perform targeted re-analysis on subtitle time windows
5. If targeted analysis fails, use subtitle timing with small buffer

**New Functions:**
- `parse_srt_for_profanity()` - Extract profanity with timing from SRT files
- `merge_profanity_results()` - Combine Whisper and subtitle detections
- `check_subtitles_for_profanity()` - Quick check for profanity in subtitles
- `get_subtitle_file_path()` - Handle external and embedded subtitles

**New CLI Options:**
- `--subtitle-only` - Only process files that have subtitles with profanity
- `--no-subtitle-enhance` - Disable subtitle enhancement (Whisper only)
- `--ignore-subtitles` - Ignore subtitles entirely

**Profanity Patterns:**
- Now catches compound words (motherfucker, bullshit, etc.)
- Patterns: fuck*, *fucker, shit*, *shit, n-word variants

**Testing:**
- Unit test suite added: `test_subtitle_detection.py`
- Tests pattern matching, SRT parsing, result merging, subtitle checking
- All 11 tests passing

**Files Modified:**
- `defuse.py` - Major rewrite of subtitle and transcription functions
- `test_subtitle_detection.py` - New unit test file

### Next Steps
1. Test with real video files with subtitles
2. When Tesla P4 arrives, GPU will be automatically detected
3. Performance should improve dramatically on GPU (10-30x faster)

## Technical Notes

### Why insanely-fast-whisper over faster-whisper?
- insanely-fast-whisper chosen because user has GPU coming
- faster-whisper is better for CPU-only (doesn't need PyTorch)
- insanely-fast-whisper excels at GPU batching for maximum speed

### PyTorch Installation
- Large download (~2-3GB with CUDA)
- PyPI servers can be slow despite gigabit connection
- Alternative: Use PyTorch's direct index for faster downloads
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- For Tesla P4, will need CUDA 12.1 version later

## Dependencies
- insanely-fast-whisper>=0.0.13
- transformers>=4.30.0
- torch>=2.0.0
- pysrt>=1.1.2
- tqdm>=4.65.0
- ffmpeg-python>=0.2.0

## Important Reminders
- FFmpeg must be installed separately on system
- Script automatically detects CUDA and switches between CPU/GPU
- Video streams are copied (not re-encoded), only audio is processed
- QuickSync is used by ffmpeg for video operations but can't accelerate Whisper AI inference
