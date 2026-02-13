#!/usr/bin/env python3
import subprocess
import os
import argparse
import json
import time
import pysrt
import re
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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

    # Check for an external SRT file (handles .en.srt, .en.hi.srt, etc.)
    external_srt = find_external_srt(video_file)
    external_srt_exists = external_srt is not None
    if external_srt_exists:
        print(f"##########\nExternal SRT Subtitle File Found: {external_srt}\n##########")
    else:
        print(f"##########\nExternal SRT Subtitle File Exists: False\n##########")

    # Return all needed info including channel count
    return audio_stream_index, audio_codec, bit_rate, duration, subtitles_exist, external_srt_exists, channels

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################
def mask_for_log(word):
    """Mask profanity for log output (e.g., 'fucking' -> 'f**king')."""
    return re.sub(r'(?i)uck', '**', word)


def find_external_srt(video_file):
    """
    Search for an external SRT file matching the video file.
    Handles common naming patterns like .srt, .en.srt, .en.hi.srt, etc.
    Returns the path if found, None otherwise.
    """
    base_name, _ = os.path.splitext(video_file)
    directory = os.path.dirname(video_file)
    video_basename = os.path.basename(base_name)

    # Check exact match first
    exact = base_name + ".srt"
    if os.path.isfile(exact):
        return exact

    # Search for SRT files with language/tag suffixes (e.g., .en.srt, .en.hi.srt)
    for f in os.listdir(directory):
        if f.lower().endswith('.srt') and f.startswith(video_basename + '.'):
            return os.path.join(directory, f)

    return None


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
    r'\w*f+u+c+k+\w*'     
    #r'\w*n+i+g+g+e+r+\w*',
    #r'\w*s+h+i+t+\w*',
]

def find_english_subtitle_stream(video_file):
    """
    Find the best English subtitle stream in a video file.
    Prefers full subtitles over forced subtitles.
    Returns the subtitle stream index (e.g., '0:s:0') or None if not found.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 's',
        '-show_entries', 'stream=index:stream_tags=language,title',
        '-of', 'json', video_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
    except json.JSONDecodeError:
        return None

    if not streams:
        return None

    # Find English subtitle streams
    english_streams = []
    first_sub_index = None

    for i, stream in enumerate(streams):
        if first_sub_index is None:
            first_sub_index = i

        tags = stream.get('tags', {})
        lang = tags.get('language', '').lower()
        title = tags.get('title', '').lower()

        if lang == 'eng' or lang == 'en':
            # Check if it's a forced subtitle (usually for foreign dialogue only)
            is_forced = 'forced' in title or 'foreign' in title
            english_streams.append((i, is_forced))

    if english_streams:
        # Prefer non-forced English subtitles
        non_forced = [s for s in english_streams if not s[1]]
        if non_forced:
            selected_index = non_forced[0][0]
            print(f"##########\nFound English subtitle stream at index {selected_index}\n##########")
            return f'0:s:{selected_index}'
        else:
            # Fall back to forced English if that's all we have
            selected_index = english_streams[0][0]
            print(f"##########\nFound English (forced) subtitle stream at index {selected_index}\n##########")
            return f'0:s:{selected_index}'

    # No English found, fall back to first subtitle
    if first_sub_index is not None:
        print(f"##########\nNo English subtitles found, using first subtitle stream\n##########")
        return f'0:s:{first_sub_index}'

    return None


def get_subtitle_file_path(video_file, subtitles_exist, external_srt_exists):
    """
    Get or extract the subtitle file path.
    Returns the path to the SRT file (either external or extracted from video).
    """
    base_name, _ = os.path.splitext(video_file)

    # Check for external SRT (handles .en.srt, .en.hi.srt, etc.)
    external_srt = find_external_srt(video_file)
    if external_srt:
        return external_srt
    elif subtitles_exist:
        # Find the best English subtitle stream
        sub_stream = find_english_subtitle_stream(video_file)
        if sub_stream is None:
            sub_stream = '0:s:0'  # Fallback to first subtitle

        # Check if subtitle codec is bitmap-based (can't extract to SRT)
        sub_idx = int(sub_stream.split(':')[-1])
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', f's:{sub_idx}',
            '-show_entries', 'stream=codec_name',
            '-of', 'json', video_file
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        bitmap_codecs = {'hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle', 'xsub'}
        if probe_result.returncode == 0:
            try:
                probe_data = json.loads(probe_result.stdout)
                probe_streams = probe_data.get('streams', [])
                if probe_streams:
                    codec = probe_streams[0].get('codec_name', '')
                    if codec in bitmap_codecs:
                        print(f"##########\nSubtitle stream is bitmap-based ({codec}), cannot extract to SRT\n##########")
                        return None
            except json.JSONDecodeError:
                pass

        extracted_subtitle_file = base_name + "_subtitles.srt"
        if not os.path.exists(extracted_subtitle_file):
            print("##########\nExtracting subtitles from video file...\n##########")
            cmd = ['ffmpeg', '-y', '-i', video_file, '-map', sub_stream, extracted_subtitle_file]
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0 or not os.path.exists(extracted_subtitle_file) or os.path.getsize(extracted_subtitle_file) == 0:
                print("##########\nFailed to extract subtitles from video file\n##########")
                if os.path.exists(extracted_subtitle_file) and os.path.getsize(extracted_subtitle_file) == 0:
                    os.remove(extracted_subtitle_file)
                return None
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
        duration = end_seconds - start_seconds
        text_len = len(text) if len(text) > 0 else 1
        for pattern in compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                word = match.group()
                # Estimate word position via character offset interpolation
                char_ratio_start = match.start() / text_len
                char_ratio_end = match.end() / text_len
                word_start = start_seconds + char_ratio_start * duration
                word_end = start_seconds + char_ratio_end * duration
                # Add ±0.3s buffer, clamped to subtitle boundaries
                buf_start = max(start_seconds, word_start - 0.3)
                buf_end = min(end_seconds, word_end + 0.3)
                profanity_instances.append({
                    'word': word,
                    'start': buf_start,
                    'end': buf_end,
                    'subtitle_text': text,
                    'source': 'subtitle'
                })
                print(f"  Found in subtitles: '{mask_for_log(word)}' at {buf_start:.2f}s - {buf_end:.2f}s")

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
            print(f"  Whisper missed: '{mask_for_log(sub_swear['word'])}' at {sub_start:.2f}s - using subtitle timing")

    print(f"##########\nWhisper found: {len(whisper_swears)}, Subtitle fallbacks added: {missed_count}\n##########")
    print(f"##########\nTotal profanity to mute: {len(merged)}\n##########")

    # Sort by start time
    merged.sort(key=lambda x: x['start'])

    # Deduplicate overlapping entries
    if merged:
        deduped = [merged[0]]
        for entry in merged[1:]:
            prev = deduped[-1]
            if entry['start'] < prev['end'] - 0.1:
                # Overlapping - merge into wider window, prefer whisper source
                if entry['source'] == 'whisper' and prev['source'] != 'whisper':
                    deduped[-1] = {
                        'word': entry['word'],
                        'start': min(prev['start'], entry['start']),
                        'end': max(prev['end'], entry['end']),
                        'source': entry['source']
                    }
                else:
                    deduped[-1] = {
                        'word': prev['word'],
                        'start': min(prev['start'], entry['start']),
                        'end': max(prev['end'], entry['end']),
                        'source': prev['source']
                    }
            else:
                deduped.append(entry)
        merged = deduped

    return [(m['word'], m['start'], m['end']) for m in merged]


def check_subtitle_sync(audio_file, subtitle_file, threshold=0.5):
    """
    Check subtitle sync against audio using ffsubsync.
    If the offset exceeds threshold, replace subtitle_file with the corrected version.

    Args:
        audio_file: Path to the audio/video file
        subtitle_file: Path to the SRT subtitle file
        threshold: Maximum acceptable offset in seconds (default 0.5)

    Returns:
        The (possibly corrected) subtitle file path, or the original if ffsubsync
        is not installed or sync is already good.
    """
    # Check if ffsubsync is available
    try:
        result = subprocess.run(
            ['ffsubsync', '--version'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("##########\nffsubsync not available - skipping sync check\n##########")
            return subtitle_file
    except FileNotFoundError:
        print("##########\nffsubsync not installed - skipping sync check\n##########")
        return subtitle_file

    print(f"##########\nChecking subtitle sync against audio...\n##########")

    # Create a temp file for the synced output
    base, ext = os.path.splitext(subtitle_file)
    synced_file = base + "_synced" + ext

    try:
        cmd = [
            'ffsubsync',
            audio_file,
            '-i', subtitle_file,
            '-o', synced_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"##########\nffsubsync failed: {result.stderr.strip()}\n##########")
            if os.path.exists(synced_file):
                os.remove(synced_file)
            return subtitle_file

        # Parse offset from ffsubsync output
        # ffsubsync prints offset info to stderr
        offset = 0.0
        for line in result.stderr.split('\n'):
            if 'offset' in line.lower():
                # Try to extract numeric offset value
                import re as _re
                match = _re.search(r'[-+]?\d*\.?\d+', line)
                if match:
                    offset = abs(float(match.group()))
                    break

        print(f"##########\nSubtitle sync offset: {offset:.3f}s (threshold: {threshold}s)\n##########")

        if offset > threshold and os.path.exists(synced_file):
            # Replace original with corrected version
            import shutil
            shutil.move(synced_file, subtitle_file)
            print(f"##########\nSubtitles re-synced (offset was {offset:.3f}s)\n##########")
        else:
            # Sync is good enough, clean up temp file
            if os.path.exists(synced_file):
                os.remove(synced_file)
            print(f"##########\nSubtitles are in sync (offset {offset:.3f}s <= {threshold}s)\n##########")

    except subprocess.TimeoutExpired:
        print("##########\nffsubsync timed out - skipping sync check\n##########")
        if os.path.exists(synced_file):
            os.remove(synced_file)
    except Exception as e:
        print(f"##########\nffsubsync error: {e} - skipping sync check\n##########")
        if os.path.exists(synced_file):
            os.remove(synced_file)

    return subtitle_file


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

        print(f"  Analyzing window: {window_start:.2f}s - {window_end:.2f}s for '{mask_for_log(sub_item['word'])}'")

        try:
            # Transcribe just this segment with word timestamps
            segments, _ = model.transcribe(
                audio_file,
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,  # Disable VAD for small segments
                clip_timestamps=[window_start, window_end]
            )

            # Consume generator with timeout to prevent hangs
            def _collect_segments():
                return list(segments)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_collect_segments)
                try:
                    seg_list = future.result(timeout=30)
                except FuturesTimeoutError:
                    print(f"    Timeout during re-analysis - using subtitle timing as fallback")
                    refined_swears.append((
                        sub_item['word'],
                        max(0, sub_item['start'] - 0.2),
                        sub_item['end'] + 0.3
                    ))
                    continue

            segment_found = False
            for segment in seg_list:
                if segment.words:
                    for word in segment.words:
                        # Check if this word matches any profanity pattern
                        for pattern in compiled_patterns:
                            if pattern.search(word.word):
                                end_time = word.end + 0.1
                                refined_swears.append((word.word, word.start, end_time))
                                print(f"    Found precise timing: '{mask_for_log(word.word)}' at {word.start:.2f}s - {end_time:.2f}s")
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
DEFAULT_MODEL = "nyrahealth/faster_CrisperWhisper"

def load_whisper_model(model_name=None):
    """
    Load the Whisper model using faster-whisper with GPU support.
    Returns the model and device string.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
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

    print(f"##########\nLoading Whisper model: {model_name}\n##########")
    print(f"##########\nDevice: {device}, Compute type: {compute_type}\n##########")

    # Load the model
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    return model, device


def transcribe_audio(audio_file, subtitle_file=None, output_transcription=False, model_name=None):
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
    model, device = load_whisper_model(model_name=model_name)

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
                        print(f"  Whisper found: '{mask_for_log(word.word)}' at {word.start:.2f}s - {end_time:.2f}s")
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

                # Find which subtitle items weren't matched by whisper,
                # and track their index in final_swear_list for direct update
                whisper_times = [(s[1], s[2]) for s in whisper_swear_list]
                missed_subs = []
                for sub in subtitle_swears:
                    found = False
                    for w_start, w_end in whisper_times:
                        if (w_start <= sub['end'] + 2.0 and w_end >= sub['start'] - 2.0):
                            found = True
                            break
                    if not found:
                        # Find this entry's index in final_swear_list
                        target_idx = None
                        for i, (word, start, end) in enumerate(final_swear_list):
                            if (abs(start - sub['start']) < 1.0 and
                                    word.lower() == sub['word'].lower()):
                                target_idx = i
                                break
                        missed_subs.append((sub, target_idx))

                # Do targeted transcription on missed segments
                if missed_subs:
                    print(f"##########\nAttempting to refine {len(missed_subs)} subtitle-only detections...\n##########")
                    for missed, target_idx in missed_subs:
                        window_start = max(0, missed['start'] - 1.0)
                        window_end = missed['end'] + 1.0
                        print(f"  Re-analyzing: {window_start:.2f}s - {window_end:.2f}s for '{mask_for_log(missed['word'])}'")

                        try:
                            # Transcribe just this segment with word timestamps
                            segs, _ = model.transcribe(
                                audio_file,
                                beam_size=5,
                                word_timestamps=True,
                                vad_filter=False,
                                clip_timestamps=[window_start, window_end]
                            )

                            # Consume generator with timeout to prevent hangs
                            def _collect_segments():
                                return list(segs)

                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(_collect_segments)
                                try:
                                    seg_list = future.result(timeout=30)
                                except FuturesTimeoutError:
                                    print(f"    Timeout during re-analysis - keeping subtitle timing")
                                    continue

                            # Collect all profanity candidates from re-analysis
                            candidates = []
                            for seg in seg_list:
                                if seg.words:
                                    for w in seg.words:
                                        for pattern in compiled_patterns:
                                            if pattern.search(w.word):
                                                candidates.append(w)
                                                break

                            if candidates and target_idx is not None:
                                # Pick candidate closest to expected subtitle time
                                expected_time = missed['start']
                                best = min(candidates, key=lambda c: abs(c.start - expected_time))
                                final_swear_list[target_idx] = (best.word, best.start, best.end + 0.1)
                                print(f"    Refined: '{mask_for_log(best.word)}' at {best.start:.2f}s - {best.end + 0.1:.2f}s")
                            else:
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
        print(f"Muting: '{mask_for_log(swear[0])}' at {swear[1]:.2f}s - {swear[2]:.2f}s")
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
#                           MASK SUBTITLES                                    #
###############################################################################
def mask_subtitle_file(input_subtitle_file, output_subtitle_file):
    """
    Mask profanity in a subtitle file by replacing matches with asterisks.
    Returns the number of replacements made.
    """
    print(f"##########\nMasking profanity in subtitle file: {input_subtitle_file}\n##########")

    # Pattern to match fuck and variations
    profanity_pattern = re.compile(r'(f+u+c+k+)', re.IGNORECASE)

    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    content = None
    used_encoding = 'utf-8'

    for encoding in encodings:
        try:
            with open(input_subtitle_file, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"##########\nCould not decode subtitle file with any supported encoding\n##########")
        return 0

    # Count matches
    matches = profanity_pattern.findall(content)
    count = len(matches)

    if count == 0:
        print(f"##########\nNo profanity found in subtitle file\n##########")
        return 0

    # Replace matches with asterisks of same length
    def replace_match(match):
        return '*' * len(match.group(0))

    masked_content = profanity_pattern.sub(replace_match, content)

    # Write output
    with open(output_subtitle_file, 'w', encoding=used_encoding) as f:
        f.write(masked_content)

    print(f"##########\nMasked {count} instance(s) of profanity -> {output_subtitle_file}\n##########")
    return count


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
    parser.add_argument('--preserve-original', action='store_true',
                        help='Keep the original file (default is to delete it after creating clean version)')
    parser.add_argument('--no-sync-check', action='store_true',
                        help='Disable ffsubsync subtitle sync verification')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Whisper model to use (default: {DEFAULT_MODEL})')
    args = parser.parse_args()

    video_files = args.input
    ignore_subtitles = args.ignore_subtitles
    subtitle_only = args.subtitle_only
    output_transcription = args.output_transcription
    use_subtitle_enhance = not args.no_subtitle_enhance
    preserve_original = args.preserve_original
    sync_check = not args.no_sync_check

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

        # Run subtitle sync check if enabled and we have a subtitle file
        if subtitle_file and sync_check and not ignore_subtitles:
            subtitle_file = check_subtitle_sync(video_file, subtitle_file)

        # Determine subtitle file for enhancement
        enhance_subtitle_file = subtitle_file if use_subtitle_enhance and not ignore_subtitles else None

        # Extract audio track (using detected channel info)
        audio_only_file = extract_audio(video_file, audio_index, audio_codec, bit_rate, duration, channels)

        # Extract wav file for transcription only
        wav = extract_for_transcription(video_file, audio_index, duration)

        # Transcribe audio with optional subtitle enhancement
        swears = transcribe_audio(wav, subtitle_file=enhance_subtitle_file, output_transcription=output_transcription, model_name=args.model)

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

            # Mask subtitles if available
            if subtitle_file and os.path.exists(subtitle_file):
                clean_subtitle_file = os.path.join(directory, f"{base_name}-CLEAN.srt")
                mask_subtitle_file(subtitle_file, clean_subtitle_file)

            # Check if we need to clean up extracted subtitle file
            extracted_subtitle_file = os.path.join(directory, f"{base_name}_subtitles.srt")

            if preserve_original:
                print("##########\nPreserving original file as requested.\n##########")
                remove_int_files(defused_audio_file, audio_only_file, extracted_subtitle_file)
            else:
                remove_int_files(defused_audio_file, audio_only_file, video_file, extracted_subtitle_file)
        else:
            print(f"##########\nFailed to create clean file: {clean_video_file}. Keeping original.\n##########")
            extracted_subtitle_file = os.path.join(directory, f"{base_name}_subtitles.srt")
            remove_int_files(defused_audio_file, audio_only_file, extracted_subtitle_file)

if __name__ == "__main__":
    main()

