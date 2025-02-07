import subprocess
import whisper
import os
import argparse
import json
import time
import torch
import pysrt
import re
from tqdm import tqdm

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
#                           EXTRACT SUBTITLES                                 #
###############################################################################
def extract_subtitles(video_file, subtitles_exist, external_srt_exists):
    print("##########\nExtracting subtitles from video file...\n##########")
    base_name, _ = os.path.splitext(video_file)
    subtitle_file = base_name + ".srt"

    if external_srt_exists:
        with open(subtitle_file, 'r', encoding='utf-8') as file:
            subtitles = file.read()
    else:
        # Extract subtitles using ffmpeg
        subtitle_file = base_name + "_subtitles.srt"
        cmd = ['ffmpeg', '-i', video_file, '-map', '0:s:0', subtitle_file]
        subprocess.run(cmd, text=True)
        with open(subtitle_file, 'r', encoding='utf-8') as file:
            subtitles = file.read()

    # Look for instances of "fuck" (case-insensitive)
    matches = re.findall(r'fuck', subtitles, re.IGNORECASE)
    if matches:
        print(f"##########\nFound {len(matches)} instances of 'f**k' in subtitles.\n##########")
        subtitle_swears = True
    else:
        print("##########\nNo instances of 'f**k' found in subtitles.\n##########")
        subtitle_swears = False

    return subtitle_swears

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
#                           CONVERT TO MP3                                    #
###############################################################################
def convert_to_mp3(audio_file, duration):
    print("##########\nConverting audio to MP3 format...\n##########")
    mp3_audio_file = os.path.splitext(audio_file)[0] + ".mp3"

    # Force stereo for MP3 conversion to avoid ffmpeg guessing mono
    cmd = [
        'ffmpeg',
        '-i', audio_file,
        '-vn',
        '-ac', '2',
        '-acodec', 'libmp3lame',
        '-b:a', '256k',
        mp3_audio_file
    ]
    if duration:
        cmd.insert(-1, '-t')
        cmd.insert(-1, str(duration))
    subprocess.run(cmd, text=True)

    print("##########\nAudio conversion to MP3 completed.\n##########")
    return mp3_audio_file

###############################################################################
#                           TRANSCRIBE AUDIO                                  #
###############################################################################
def transcribe_audio(audio_file, output_transcription=False):
    print("##########\nTranscribing audio into text to find F-words...\n##########")
    start_time = time.time()

    if torch.cuda.is_available():
        print("##########\nCUDA is available: Using GPU!\n##########")
        device = "cuda"
    else:
        print("##########\nCUDA not available: Using CPU.\n##########")
        device = "cpu"

    with tqdm(total=100, desc="Loading Whisper model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        model = whisper.load_model("base", device=device)
        for _ in range(100):
            pbar.update(1)
            time.sleep(0.01)
        
    if hasattr(model, "device"):
        print(f"Model loaded on device: {model.device}")

    with tqdm(total=100, desc="Transcribing audio", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        result = model.transcribe(audio_file, word_timestamps=True)
        for _ in range(100):
            pbar.update(1)
            time.sleep(0.05)

    end_time = time.time()
    transcription_text = result['text']
    transcription_time = end_time - start_time
    print(f"##########\nTranscription Completed in {transcription_time:.2f} seconds\n##########")
    
    if output_transcription:
        transcription_file = os.path.splitext(audio_file)[0] + "_transcription.txt"
        with open(transcription_file, 'w', encoding='utf-8') as file:
            file.write(transcription_text)
        print(f"##########\nTranscription saved to: {transcription_file}\n##########")

    # Find swear words (add more terms as needed)
    swear_list = []
    for segment in result['segments']:
        for word_obj in segment['words']:
            word = word_obj['word']
            start = word_obj['start']
            end = word_obj['end'] + 0.1
            if any(swear in word.lower() for swear in ["fuck", "nigger"]):
                swear_list.append((word, start, end))

    print(f"##########\nTotal swear words: {len(swear_list)}\n##########")
    return swear_list

###############################################################################
#                           COMPARE WITH SUBTITLES                            #
###############################################################################
def compare_with_subtitles(transcribed_text, subtitle_file):
    print("##########\nComparing transcription with subtitles...\n##########")
    with open(subtitle_file, 'r') as file:
        subtitle_lines = file.readlines()

    missing_f_words = []
    current_dialogue = ""
    for line in subtitle_lines:
        if line.strip() == "":
            if any("fuck" in word.lower() for word in current_dialogue.split()):
                missing_f_words.append(current_dialogue)
            current_dialogue = ""
        else:
            current_dialogue += line.strip() + " "

    if any("fuck" in word.lower() for word in current_dialogue.split()):
        missing_f_words.append(current_dialogue)

    for dialogue in missing_f_words:
        dialogue_words = dialogue.split()
        for word in dialogue_words:
            if "fuck" in word.lower() and word not in transcribed_text:
                print(f"Missing F-word: {word}")

    print("##########\nComparison complete.\n##########")

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
    parser.add_argument('--ignore-subtitles', action='store_true', help='Ignore subtitles check')
    parser.add_argument('--output-transcription', action='store_true', help='Output transcription to a file')
    args = parser.parse_args()

    video_files = args.input
    ignore_subtitles = args.ignore_subtitles
    output_transcription = args.output_transcription

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

        if not ignore_subtitles and (subtitles_exist or external_srt_exists):
            subtitle_swears = extract_subtitles(video_file, subtitles_exist, external_srt_exists)
            if not subtitle_swears:
                print("##########\nNo F-words found in subtitles. Exiting.\n##########")
                continue

        # Extract audio track (using detected channel info)
        audio_only_file = extract_audio(video_file, audio_index, audio_codec, bit_rate, duration, channels)
        # Convert extracted audio to MP3 for faster Whisper transcription
        mp3_audio_file = convert_to_mp3(audio_only_file, duration)
        # Transcribe the MP3 to get swear word timestamps
        swears = transcribe_audio(mp3_audio_file, output_transcription)

        if not swears:
            print("##########\nNo F-words found in audio. Exiting.\n##########")
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
            remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file, video_file)
        else:
            print(f"##########\nFailed to create clean file: {clean_video_file}. Keeping original.\n##########")
            remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file)

if __name__ == "__main__":
    main()

