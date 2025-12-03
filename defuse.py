import subprocess
import os
import argparse
import json
import time
import pysrt
import re
from faster_whisper import WhisperModel

# Map out common extensions to their codec. Used in multiple functions
AUDIO_EXTENSION_MAP = {
    'aac':        'aac',
    'ac3':        'ac3',
    'eac3':       'eac3',
    'dts':        'dts',
    'mp3':        'mp3',
    'libmp3lame': 'mp3',
    'flac':       'flac',
    'opus':       'opus',
    'libopus':    'opus',
    'vorbis':     'wav',
    'libvorbis':  'wav',
    'wav':        'wav',
    'pcm_s16le':  'wav',
    'pcm_s24le':  'wav'
}

###############################################################################
#                           GET INFO                                          #
###############################################################################
def get_info(video_file):
    print("##########\nGetting audio and subtitle info from video file...\n##########")

    # Use ffprobe to get audio stream info including channels and language tags.
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=index,channels,codec_name,bit_rate,duration:stream_tags=language',
        '-of', 'json',
        video_file
    ]
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)

    # Default values
    audio_stream_index = None
    audio_codec = 'aac'
    bit_rate = '320000'
    duration = None
    subtitles_exist = False
    channels = None

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            audio_streams = data.get('streams', [])

            # Try to find an English audio track first
            for stream in audio_streams:
                tags = stream.get('tags', {})
                language = tags.get('language', '').lower()
                if language == 'eng':
                    audio_stream_index = stream.get('index')
                    audio_codec = stream.get('codec_name', audio_codec)
                    bit_rate = stream.get('bit_rate', bit_rate)
                    duration = stream.get('duration', duration)
                    channels = stream.get('channels')
                    break

            # If no English track, use the first audio stream
            if audio_stream_index is None and audio_streams:
                first_stream = audio_streams[0]
                audio_stream_index = first_stream.get('index')
                audio_codec = first_stream.get('codec_name', audio_codec)
                bit_rate = first_stream.get('bit_rate', bit_rate)
                duration = first_stream.get('duration', duration)
                channels = first_stream.get('channels')

        except json.JSONDecodeError:
            print("Error: Failed to parse stream information JSON.")
    else:
        print("Error: Failed to get stream information via ffprobe.")

    # Check for subtitle streams
    ffprobe_subs_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 's',
        '-show_entries', 'stream=index',
        '-of', 'json',
        video_file
    ]
    result_subs = subprocess.run(ffprobe_subs_cmd, capture_output=True, text=True)
    if result_subs.returncode == 0:
        subs_data = json.loads(result_subs.stdout)
        subtitles_exist = 'streams' in subs_data and len(subs_data['streams']) > 0
    else:
        subtitles_exist = False

    # Debug output
    print(f"##########\nAudio Stream Index Chosen: {audio_stream_index}")
    print(f"Audio Codec: {audio_codec}")
    print(f"Bitrate: {bit_rate}")
    print(f"Duration: {duration} seconds")
    print(f"Channels: {channels}\n##########")
    print(f"##########\nSubtitles Exist in Video File: {subtitles_exist}\n##########")

    # Check for an external SRT file
    base_name, _ = os.path.splitext(video_file)
    subtitle_file = base_name + ".srt"
    external_srt_exists = os.path.isfile(subtitle_file)
    print(f"##########\nExternal SRT Subtitle File Exists: {external_srt_exists}\n##########")

    # Return all needed info including channel count
    return audio_stream_index, audio_codec, bit_rate, duration, subtitles_exist, external_srt_exists, channels

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################
def get_audio_extension(codec_name: str) -> str:
    """
    Given an audio codec name, return the proper file extension.
    """
    return AUDIO_EXTENSION_MAP.get(codec_name, codec_name)

def get_extracted_filename(video_file: str, codec_name: str) -> str:
    """
    Returns the filename for the extracted audio.
    """
    base_name, _ = os.path.splitext(video_file)
    ext = get_audio_extension(codec_name)
    return f"{base_name}.{ext}"

def get_defused_filename(audio_file: str, codec_name: str) -> str:
    """
    Returns the filename for the defused (muted) audio.
    """
    base_name, _ = os.path.splitext(audio_file)
    ext = get_audio_extension(codec_name)
    return f"{base_name}-DEFUSED-AUDIO.{ext}"

def get_ac3_or_copy(audio_file: str):
    """
    Decide how to encode the defused audio.
    """
    _, ext = os.path.splitext(audio_file)
    ext = ext.lower()
    
    if ext == '.wav':
        out_codec = 'ac3'
        extra_args = ['-b:a', '384k']
        defused_ext = '.ac3'
    else:
        out_codec = 'aac'
        extra_args = ['-b:a', '256k', '-strict', 'experimental']
        defused_ext = '.m4a'
    return out_codec, extra_args, defused_ext

###############################################################################
#                           SUBTITLE FUNCTIONS                                #
###############################################################################

# Profanity patterns to detect (can be extended)
# These patterns match both standalone words and compound words
PROFANITY_PATTERNS = [
    r'\w*f+u+c+k+\w*',     # fuck and variations (fucker, fucking, motherfucker, etc.)
    r'\w*n+i+g+g+e+r+\w*', # n-word and variations
    r'\w*s+h+i+t+\w*',     # shit and variations (bullshit, shitty, etc.)
]

def get_subtitle_file_path(video_file, subtitles_exist, external_srt_exists):
    """
    Get or extract the subtitle file path.
    Returns the path to the SRT file (either external or extracted from video).
    """
    base_name, _ = os.path.splitext(video_file)

    if external_srt_exists:
        return base_name + ".srt"
    elif subtitles_exist:
        extracted_subtitle_file = base_name + "_subtitles.srt"
        if not os.path.exists(extracted_subtitle_file):
            print("##########\nExtracting subtitles from video file...\n##########")
            cmd = ['ffmpeg', '-y', '-i', video_file, '-map', '0:s:0', extracted_subtitle_file]
            subprocess.run(cmd, text=True, capture_output=True)
        return extracted_subtitle_file
    return None


def parse_srt_for_profanity(subtitle_file):
    """
    Parse an SRT file and extract timing information for profanity.

    Returns a list of tuples: (word, start_time, end_time, subtitle_text)
    where times are in seconds.
    """
    print("##########\nParsing subtitles for profanity...\n##########")

    if not os.path.exists(subtitle_file):
        print(f"##########\nSubtitle file not found: {subtitle_file}\n##########")
        return []

    try:
        subs = pysrt.open(subtitle_file, encoding='utf-8')
    except Exception as e:
        print(f"##########\nError reading subtitle file: {e}\n##########")
        try:
            subs = pysrt.open(subtitle_file, encoding='latin-1')
        except Exception as e2:
            print(f"##########\nFailed to read subtitle file with fallback encoding: {e2}\n##########")
            return []

    profanity_instances = []
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in PROFANITY_PATTERNS]

    for sub in subs:
        text = sub.text.replace('\n', ' ')
        start_seconds = sub.start.ordinal / 1000.0
        end_seconds = sub.end.ordinal / 1000.0

        # Check each profanity pattern
        for pattern in compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                word = match.group()
                # Estimate word position within the subtitle duration
                # This is approximate - we'll use the full subtitle window
                profanity_instances.append({
                    'word': word,
                    'start': start_seconds,
                    'end': end_seconds,
                    'subtitle_text': text,
                    'source': 'subtitle'
                })
                print(f"  Found in subtitles: '{word}' at {start_seconds:.2f}s - {end_seconds:.2f}s")

    print(f"##########\nFound {len(profanity_instances)} profanity instances in subtitles.\n##########")
    return profanity_instances


def check_subtitles_for_profanity(video_file, subtitles_exist, external_srt_exists):
    """
    Quick check if subtitles contain any profanity.
    Returns (has_profanity, subtitle_file_path)
    """
    subtitle_file = get_subtitle_file_path(video_file, subtitles_exist, external_srt_exists)

    if not subtitle_file:
        return False, None

    try:
        with open(subtitle_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(subtitle_file, 'r', encoding='latin-1') as file:
            content = file.read()

    # Quick check for any profanity
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            print(f"##########\nProfanity found in subtitles.\n##########")
            return True, subtitle_file

    print("##########\nNo profanity found in subtitles.\n##########")
    return False, subtitle_file


def merge_profanity_results(whisper_swears, subtitle_swears, tolerance=2.0):
    """
    Merge profanity detected by Whisper with profanity found in subtitles.

    Strategy:
    1. Start with all Whisper detections (they have precise timestamps)
    2. For each subtitle profanity, check if Whisper found something in that time window
    3. If Whisper missed it, add the subtitle's time window as a "fallback" detection

    Args:
        whisper_swears: List of tuples (word, start, end) from Whisper
        subtitle_swears: List of dicts with 'word', 'start', 'end' from subtitles
        tolerance: Time window (seconds) to consider as a match

    Returns:
        Merged list of tuples (word, start, end) for muting
    """
    print("##########\nMerging Whisper and subtitle profanity results...\n##########")

    # Convert whisper results to a more workable format
    merged = []
    whisper_times = []

    for swear in whisper_swears:
        word, start, end = swear
        merged.append({
            'word': word,
            'start': start,
            'end': end,
            'source': 'whisper'
        })
        whisper_times.append((start, end))

    # Check each subtitle detection
    missed_count = 0
    for sub_swear in subtitle_swears:
        sub_start = sub_swear['start']
        sub_end = sub_swear['end']

        # Check if Whisper found anything in this time window (with tolerance)
        found_match = False
        for w_start, w_end in whisper_times:
            # Check for overlap with tolerance
            if (w_start <= sub_end + tolerance and w_end >= sub_start - tolerance):
                found_match = True
                break

        if not found_match:
            # Whisper missed this one - add from subtitle
            missed_count += 1
            # Add a small buffer around the subtitle window
            merged.append({
                'word': sub_swear['word'],
                'start': max(0, sub_start - 0.2),  # Small buffer before
                'end': sub_end + 0.3,  # Buffer after
                'source': 'subtitle_fallback'
            })
            print(f"  Whisper missed: '{sub_swear['word']}' at {sub_start:.2f}s - using subtitle timing")

    print(f"##########\nWhisper found: {len(whisper_swears)}, Subtitle fallbacks added: {missed_count}\n##########")
    print(f"##########\nTotal profanity to mute: {len(merged)}\n##########")

    # Sort by start time and convert to tuple format expected by mute_audio
    merged.sort(key=lambda x: x['start'])
    return [(m['word'], m['start'], m['end']) for m in merged]


def subtitle_guided_transcription(audio_file, subtitle_file, model, output_transcription=False):
    """
    Perform targeted transcription on specific audio segments where subtitles
    indicate profanity exists but Whisper's full transcription may have missed it.

    This does a second-pass analysis on the subtitle time windows to try to
    get more accurate word-level timestamps.

    Args:
        audio_file: Path to the audio file
        subtitle_file: Path to the SRT file
        model: Loaded Whisper model
        output_transcription: Whether to save transcription to file

    Returns:
        List of (word, start, end) tuples for detected profanity
    """
    print("##########\nPerforming subtitle-guided transcription analysis...\n##########")

    # Get subtitle profanity windows
    subtitle_profanity = parse_srt_for_profanity(subtitle_file)

    if not subtitle_profanity:
        print("##########\nNo profanity in subtitles to guide transcription.\n##########")
        return []

    refined_swears = []
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in PROFANITY_PATTERNS]

    # For each subtitle window with profanity, do targeted transcription
    for sub_item in subtitle_profanity:
        window_start = max(0, sub_item['start'] - 1.0)  # 1 second buffer
        window_end = sub_item['end'] + 1.0

        print(f"  Analyzing window: {window_start:.2f}s - {window_end:.2f}s for '{sub_item['word']}'")

        try:
            # Transcribe just this segment with word timestamps
            segments, _ = model.transcribe(
                audio_file,
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,  # Disable VAD for small segments
                clip_timestamps=[window_start, window_end]
            )

            segment_found = False
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        # Check if this word matches any profanity pattern
                        for pattern in compiled_patterns:
                            if pattern.search(word.word):
                                end_time = word.end + 0.1
                                refined_swears.append((word.word, word.start, end_time))
                                print(f"    Found precise timing: '{word.word}' at {word.start:.2f}s - {end_time:.2f}s")
                                segment_found = True
                                break

            if not segment_found:
                # Whisper still couldn't find it - use subtitle timing as fallback
                print(f"    Whisper couldn't detect in segment - using subtitle timing as fallback")
                refined_swears.append((
                    sub_item['word'],
                    max(0, sub_item['start'] - 0.2),
                    sub_item['end'] + 0.3
                ))

        except Exception as e:
            print(f"    Error analyzing segment: {e} - using subtitle timing")
            refined_swears.append((
                sub_item['word'],
                max(0, sub_item['start'] - 0.2),
                sub_item['end'] + 0.3
            ))

    return refined_swears

###############################################################################
#                           EXTRACT AUDIO                                     #
###############################################################################
def extract_audio(video_file, audio_index, audio_codec, bit_rate, duration, channels):
    print("##########\nExtracting audio from video file...\n##########")
    
    # Lookup table for specific codec settings
    codec_map = {
        'dts': {
            'ext': '.wav',
            'ffmpeg_codec': 'pcm_s16le',
            'extra_args': ['-ar', '16000']  # Removed forced mono setting
        },
        'aac': {
            'ext': '.aac',
            'ffmpeg_codec': 'copy',
            'extra_args': ['-strict', '-2']
        },
        'vorbis': {
            "ext": ".wav",
            "ffmpeg_codec": "pcm_s16le",
            "extra_args": []
        },
        'libvorbis': {
            "ext": ".wav",
            "ffmpeg_codec": "pcm_s16le",
            "extra_args": []
        }
    }

    if audio_codec in codec_map:
        chosen_ext = codec_map[audio_codec]['ext']
        chosen_codec = codec_map[audio_codec]['ffmpeg_codec']
        extra_args = codec_map[audio_codec]['extra_args']
    else:
        # In fallback, if the channel count is less than 2, force re-encoding to stereo.
        try:
            orig_channels = int(channels) if channels is not None else 0
        except ValueError:
            orig_channels = 0
        if orig_channels < 2:
            chosen_codec = 'aac'
            extra_args = ['-b:a', '256k', '-strict', '-2']
            chosen_ext = '.aac'
        else:
            chosen_ext = f".{audio_codec}"
            chosen_codec = 'copy'
            extra_args = ['-strict', '-2']

    # Compute the final channel count (force at least stereo)
    try:
        orig_channels = int(channels) if channels is not None else 0
    except ValueError:
        orig_channels = 0
    final_channels = orig_channels if orig_channels >= 2 else 2

    output_audio = os.path.splitext(video_file)[0] + chosen_ext

    cmd = [
        'ffmpeg',
        '-i', video_file,
        '-map', f'0:{audio_index}',
        '-vn',
        '-acodec', chosen_codec,
        *extra_args,
    ]
    # If re-encoding (i.e. not copying), set the channel count
    if chosen_codec != 'copy':
        cmd.extend(['-ac', str(final_channels)])
    if duration:
        cmd.extend(['-t', str(duration)])
    cmd.append(output_audio)

    subprocess.run(cmd, text=True)
    return output_audio

###############################################################################
#                           EXTRACT WAV                                       #
###############################################################################

def extract_for_transcription(video_file, audio_index, duration=None):
    """
    Extract a mono, 16 kHz, 16-bit PCM WAV just for Whisper.
    This file is deleted once transcription is done.
    """
    wav_path = os.path.splitext(video_file)[0] + "_whisper.wav"
    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-map', f'0:{audio_index}',
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
    ]
    if duration:
        cmd += ['-t', str(duration)]
    cmd.append(wav_path)

    subprocess.run(cmd, check=True)
    return wav_path

###############################################################################
#                           TRANSCRIBE AUDIO                                  #
###############################################################################
def load_whisper_model():
    """
    Load the Whisper model using faster-whisper with GPU support.
    Returns the model and device string.
    """
    # Check for CUDA availability by trying to import and use ctranslate2
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        cuda_available = len(cuda_types) > 0
    except Exception:
        cuda_available = False
        cuda_types = set()

    if cuda_available:
        device = "cuda"
        # Tesla P4 (Pascal) supports int8 for best performance
        # float16 requires Volta (sm_70) or newer
        if "float16" in cuda_types:
            compute_type = "float16"
        elif "int8" in cuda_types:
            compute_type = "int8"
        else:
            compute_type = "float32"
        print(f"##########\nCUDA available! Using GPU acceleration.\n##########")
    else:
        device = "cpu"
        compute_type = "int8"  # CPU optimized
        print("##########\nCUDA not available. Using CPU.\n##########")

    print(f"##########\nLoading Whisper model (faster-whisper)...\n##########")
    print(f"##########\nDevice: {device}, Compute type: {compute_type}\n##########")

    # Load the model
    # model_size can be: tiny, base, small, medium, large-v2, large-v3
    model = WhisperModel("base", device=device, compute_type=compute_type)

    return model, device


def transcribe_audio(audio_file, subtitle_file=None, output_transcription=False):
    """
    Transcribe audio to find profanity with word-level timestamps.

    If subtitle_file is provided, uses subtitle-enhanced detection:
    1. Full transcription pass with Whisper
    2. Parse subtitles for known profanity locations
    3. Merge results, using subtitle timing as fallback for missed words

    Args:
        audio_file: Path to audio file (WAV format for best results)
        subtitle_file: Optional path to SRT subtitle file
        output_transcription: Whether to save full transcription to file

    Returns:
        List of tuples (word, start_time, end_time) for each profanity found
    """
    print("##########\nTranscribing audio into text to find profanity...\n##########")
    start_time = time.time()

    # Load the model
    model, device = load_whisper_model()

    print("##########\nTranscribing audio (full pass)...\n##########")

    # Transcribe the audio file using faster-whisper
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,  # Enable word-level timestamps
        vad_filter=True  # Voice activity detection to improve accuracy
    )

    # Measure the end time
    end_time_transcription = time.time()
    duration = end_time_transcription - start_time

    print(f"##########\nDetected language '{info.language}' with probability {info.language_probability:.2f}\n##########")
    print(f"##########\nTranscription completed in {duration:.2f} seconds\n##########")

    # Compile profanity patterns for matching
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in PROFANITY_PATTERNS]

    # Instantiate empty lists
    whisper_swear_list = []
    transcribed_text_parts = []

    # Process segments and words
    print("##########\nProcessing transcription for profanity...\n##########")
    for segment in segments:
        transcribed_text_parts.append(segment.text)

        # Process words in the segment if available
        if segment.words:
            for word in segment.words:
                # Check if word matches any profanity pattern
                for pattern in compiled_patterns:
                    if pattern.search(word.word):
                        # Add 0.1 second buffer to the end
                        end_time = word.end + 0.1
                        whisper_swear_list.append((word.word, word.start, end_time))
                        print(f"  Whisper found: '{word.word}' at {word.start:.2f}s - {end_time:.2f}s")
                        break

    # Write transcription to a text file for troubleshooting if requested
    if output_transcription:
        transcription_file = os.path.splitext(audio_file)[0] + "_transcription.txt"
        with open(transcription_file, 'w', encoding='utf-8') as file:
            file.write(' '.join(transcribed_text_parts))
        print(f"##########\nTranscription saved to: {transcription_file}\n##########")

    print(f"##########\nWhisper detected {len(whisper_swear_list)} profanity instances\n##########")

    # If we have subtitles, use them to catch any missed profanity
    if subtitle_file and os.path.exists(subtitle_file):
        print("##########\nEnhancing detection with subtitle data...\n##########")

        # Parse subtitles for profanity
        subtitle_swears = parse_srt_for_profanity(subtitle_file)

        if subtitle_swears:
            # Merge whisper results with subtitle hints
            final_swear_list = merge_profanity_results(whisper_swear_list, subtitle_swears)

            # If we found more via subtitles, do targeted re-analysis
            if len(final_swear_list) > len(whisper_swear_list):
                print("##########\nSubtitles indicated additional profanity - doing targeted analysis...\n##########")

                # Find which subtitle items weren't matched by whisper
                whisper_times = [(s[1], s[2]) for s in whisper_swear_list]
                missed_subs = []
                for sub in subtitle_swears:
                    found = False
                    for w_start, w_end in whisper_times:
                        if (w_start <= sub['end'] + 2.0 and w_end >= sub['start'] - 2.0):
                            found = True
                            break
                    if not found:
                        missed_subs.append(sub)

                # Do targeted transcription on missed segments
                if missed_subs:
                    print(f"##########\nAttempting to refine {len(missed_subs)} subtitle-only detections...\n##########")
                    for missed in missed_subs:
                        window_start = max(0, missed['start'] - 1.0)
                        window_end = missed['end'] + 1.0
                        print(f"  Re-analyzing: {window_start:.2f}s - {window_end:.2f}s for '{missed['word']}'")

                        try:
                            # Transcribe just this segment with word timestamps
                            segs, _ = model.transcribe(
                                audio_file,
                                beam_size=5,
                                word_timestamps=True,
                                vad_filter=False,
                                clip_timestamps=[window_start, window_end]
                            )

                            found_refined = False
                            for seg in segs:
                                if seg.words:
                                    for w in seg.words:
                                        for pattern in compiled_patterns:
                                            if pattern.search(w.word):
                                                # Update the fallback entry with precise timing
                                                for i, (word, start, end) in enumerate(final_swear_list):
                                                    if (abs(start - missed['start']) < 1.0 or
                                                        abs(start - (missed['start'] - 0.2)) < 0.5):
                                                        final_swear_list[i] = (w.word, w.start, w.end + 0.1)
                                                        print(f"    Refined: '{w.word}' at {w.start:.2f}s - {w.end + 0.1:.2f}s")
                                                        found_refined = True
                                                        break
                                            if found_refined:
                                                break
                                    if found_refined:
                                        break
                                if found_refined:
                                    break

                            if not found_refined:
                                print(f"    Could not refine - keeping subtitle timing")

                        except Exception as e:
                            print(f"    Error in targeted analysis: {e}")

            return final_swear_list
        else:
            print("##########\nNo profanity in subtitles - using Whisper results only\n##########")

    print(f"##########\nTotal profanity to mute: {len(whisper_swear_list)}\n##########")
    return whisper_swear_list


###############################################################################
#                           MUTE AUDIO                                        #
###############################################################################
def mute_audio(audio_only_file, swears):
    """
    Mute the swear words in the given audio file by applying volume=0 filters.
    """
    print("##########\nIterating through swear list and muting...\n##########")
    filter_expressions = []
    for swear in swears:
        print("Swear tuple:", swear)
        start = float(swear[1])
        end = float(swear[2])
        filter_expressions.append({'start': start, 'end': end})

    filter_string = ', '.join(
        f"volume=enable='between(t,{expr['start']},{expr['end']}):volume=0'"
        for expr in filter_expressions
    )
    print("##########\nMuting all F-words...\n##########")
    print(f"Filter String: {filter_string}")

    out_codec, extra_args, defused_ext = get_ac3_or_copy(audio_only_file)
    base_name, _ = os.path.splitext(audio_only_file)
    defused_audio_file = f"{base_name}-DEFUSED-AUDIO{defused_ext}"

    cmd = [
        'ffmpeg',
        '-i', audio_only_file,
        '-vn',
        '-af', filter_string,
        '-c:a', out_codec,
        *extra_args,
        defused_audio_file
    ]
    subprocess.run(cmd, text=True)
    return defused_audio_file

###############################################################################
#                           REMOVE INTERMEDIATE FILES                         #
###############################################################################
def remove_int_files(*file_paths):
    print("##########\nRemoving intermediate files...\n##########")
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"##########\nDeleted: {file_path}\n##########")
        else:
            print(f"##########\nFile not found: {file_path}\n##########")

###############################################################################
#                           MAIN                                              #
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='Process video files and mute profanity.')
    parser.add_argument('-i', '--input', nargs='+', help='Input video files', required=True)
    parser.add_argument('--ignore-subtitles', action='store_true',
                        help='Ignore subtitles entirely (do not use for detection or pre-check)')
    parser.add_argument('--subtitle-only', action='store_true',
                        help='Only process files that have subtitles with profanity')
    parser.add_argument('--output-transcription', action='store_true',
                        help='Output transcription to a file for debugging')
    parser.add_argument('--no-subtitle-enhance', action='store_true',
                        help='Disable subtitle-enhanced detection (use Whisper only)')
    args = parser.parse_args()

    video_files = args.input
    ignore_subtitles = args.ignore_subtitles
    subtitle_only = args.subtitle_only
    output_transcription = args.output_transcription
    use_subtitle_enhance = not args.no_subtitle_enhance

    for video_file in video_files:
        video_file = os.path.abspath(video_file)

        if not os.path.isfile(video_file):
            print(f"##########\nError: File not found: {video_file}\n##########")
            continue

        directory, filename = os.path.split(video_file)
        base_name, extension = os.path.splitext(filename)

        clean_filename = f"{base_name}-CLEAN{extension}"
        clean_file_path = os.path.join(directory, clean_filename)
        if os.path.exists(clean_file_path):
            print(f"A defused file with the name '{clean_filename}' already exists. Skipping: {video_file}")
            continue

        # Get audio info including channel count
        audio_index, audio_codec, bit_rate, duration, subtitles_exist, external_srt_exists, channels = get_info(video_file)

        # Get subtitle file path (if available)
        subtitle_file = None
        if not ignore_subtitles and (subtitles_exist or external_srt_exists):
            has_profanity, subtitle_file = check_subtitles_for_profanity(
                video_file, subtitles_exist, external_srt_exists
            )

            if subtitle_only and not has_profanity:
                print("##########\nNo profanity found in subtitles. Skipping (--subtitle-only mode).\n##########")
                continue

            if has_profanity:
                print(f"##########\nSubtitle file with profanity found: {subtitle_file}\n##########")
                if use_subtitle_enhance:
                    print("##########\nSubtitle-enhanced detection ENABLED\n##########")
        elif subtitle_only:
            print("##########\nNo subtitles available. Skipping (--subtitle-only mode).\n##########")
            continue

        # Determine subtitle file for enhancement
        enhance_subtitle_file = subtitle_file if use_subtitle_enhance and not ignore_subtitles else None

        # Extract audio track (using detected channel info)
        audio_only_file = extract_audio(video_file, audio_index, audio_codec, bit_rate, duration, channels)

        # Extract wav file for transcription only
        wav = extract_for_transcription(video_file, audio_index, duration)

        # Transcribe audio with optional subtitle enhancement
        swears = transcribe_audio(wav, subtitle_file=enhance_subtitle_file, output_transcription=output_transcription)

        # Remove the wav file as we don't need it anymore
        os.remove(wav)

        if not swears:
            print("##########\nNo profanity found in audio. Exiting.\n##########")
            remove_int_files(audio_only_file)
            continue

        # Mute the swear words in the original extracted audio file
        defused_audio_file = mute_audio(audio_only_file, swears)

        # Combine the defused audio with the original video
        clean_video_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")
        print("##########\nAdding edited audio as a second audio stream...\n##########")

        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-i', defused_audio_file,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '0:a:0',
            '-map', '1:a:0',
            '-metadata:s:a:1', 'language=eng',
            '-metadata:s:a:1', 'title=Defused (CLEAN) Track',
            clean_video_file
        ]
        subprocess.run(cmd)

        if os.path.exists(clean_video_file):
            print(f"##########\nSuccessfully created clean file: {clean_video_file}\n##########")
            remove_int_files(defused_audio_file, audio_only_file, video_file)
        else:
            print(f"##########\nFailed to create clean file: {clean_video_file}. Keeping original.\n##########")
            remove_int_files(defused_audio_file, audio_only_file)

if __name__ == "__main__":
    main()

