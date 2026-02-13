# Session: Migration to insanely-fast-whisper
**Date:** 2025-11-23
**Duration:** ~45 minutes

## Summary
Migrated tEODor from openai-whisper to insanely-fast-whisper to take advantage of incoming Tesla P4 GPU and provide immediate CPU performance improvements.

## Discussion Points

### Initial Question: QuickSync Support
- User asked if script uses Intel QuickSync
- Clarified that QuickSync is only for video encoding/decoding (H.264, H.265, etc.)
- Script uses `-c:v copy` so video isn't re-encoded (no need for QuickSync)
- Only audio is processed

### Whisper Hardware Acceleration Options
- **QuickSync (Intel)**: Cannot accelerate Whisper - only for video encoding
- **CUDA (NVIDIA)**: Current method - works with Whisper
- **OpenVINO (Intel)**: Could theoretically work but complex setup
- User's system: i5-14500 with UHD Graphics 770, no discrete GPU yet

### Performance Options Discussed

**1. openai-whisper (current/original)**
- Most compatible but slowest
- What was originally installed

**2. faster-whisper**
- 5-10x faster on CPU
- 2-4x faster on GPU
- Doesn't require PyTorch (uses CTranslate2)
- Best for CPU-only systems

**3. insanely-fast-whisper (chosen)**
- 5-10x faster on CPU (similar to faster-whisper)
- 10-30x faster on GPU (much better than faster-whisper)
- Requires PyTorch + Transformers
- Best when GPU is available
- **Decision driver**: User has Tesla P4 GPU on the way

## Changes Made

### 1. requirements.txt
```diff
- openai-whisper>=20231117
+ insanely-fast-whisper>=0.0.13
+ transformers>=4.30.0
```

### 2. defuse.py
- Changed imports: `from transformers import pipeline`
- Completely rewrote `transcribe_audio()` function:
  - Uses `pipeline("automatic-speech-recognition")` instead of `whisper.load_model()`
  - Configured for Tesla P4: `use_flash_attention_2: False`
  - Batch processing: `batch_size=24`, `chunk_length_s=30`
  - Auto-detects GPU: `device="cuda:0"` or `"cpu"`
  - Uses float16 on GPU, float32 on CPU
  - Word timestamp format changed from `segments[].words[]` to `chunks[]`

### 3. defuse-audio-only.py
- Applied same changes as defuse.py for consistency
- Updated transcription function with identical logic

## Technical Details

### Configuration for Tesla P4
- **Flash Attention 2**: Disabled (P4 doesn't support it)
- **Batch size**: 24 (optimal for 8GB VRAM)
- **Precision**: float16 (faster inference)
- **Model**: openai/whisper-base (same as before)

### API Changes
**Old (openai-whisper):**
```python
model = whisper.load_model("base", device=device)
result = model.transcribe(audio_file, word_timestamps=True)
# Access: result['segments'][i]['words'][j]
```

**New (insanely-fast-whisper):**
```python
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", ...)
result = pipe(audio_file, return_timestamps="word")
# Access: result['chunks'][i]['timestamp']
```

## Installation Issues

### PyPI Download Speed
- PyTorch installation extremely slow despite gigabit fiber
- **Not** a user connection issue
- **Cause**: PyPI CDN can be slow/throttled for large packages
- **Size**: ~2-3GB for CUDA-enabled PyTorch
- **Solution**: Use PyTorch's direct index for faster downloads:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

### Installation Status at Session End
- PyTorch still downloading
- User decided to continue tomorrow
- Remaining steps:
  1. Let PyTorch finish installing
  2. Run `pip install -r requirements.txt` for remaining dependencies
  3. Test the script

## Expected Results

### On CPU (i5-14500)
- Current speed with openai-whisper: baseline
- Expected with insanely-fast-whisper: **5-10x faster**

### On GPU (Tesla P4 when it arrives)
- Expected speed: **10-30x faster** than original
- Will automatically detect and use GPU
- No code changes needed

## Files Modified
1. `/home/kevin/scripts/tEODor/requirements.txt`
2. `/home/kevin/scripts/tEODor/defuse.py` (transcribe_audio function + imports)
3. `/home/kevin/scripts/tEODor/defuse-audio-only.py` (transcribe_audio function + imports)

## Next Session TODO
- [ ] Verify PyTorch installation completed
- [ ] Install remaining requirements
- [ ] Test script on CPU with sample video
- [ ] Compare transcription speed (should be 5-10x faster)
- [ ] When P4 arrives: verify CUDA detection
- [ ] When P4 arrives: test GPU performance (expect 10-30x improvement)

## Questions Answered
1. Does script use QuickSync? No, and doesn't need to (video not re-encoded)
2. Can Whisper use QuickSync? No, QuickSync is only for video encoding
3. What about insanely-fast-whisper? Yes, better for GPU use case
4. Still need PyTorch? Yes, insanely-fast-whisper requires it
5. Why is PyPI so slow? Common issue, not user's connection

## References
- insanely-fast-whisper: Optimized for GPU batching
- faster-whisper: CTranslate2-based, good for CPU
- Tesla P4: 8GB VRAM, CUDA compute capability 6.1
- No Flash Attention 2 support on P4 (requires compute 8.0+)
