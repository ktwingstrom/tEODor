import subprocess
import whisper
import os
import argparse
import json
import time
import torch
import pysrt
import re
import time
from tqdm import tqdm


# Map out common extensions to their codec. Used in multiple functions
AUDIO_EXTENSION_MAP = {
    'aac':       'aac',
    'ac3':       'ac3',
    'eac3':      'eac3',
    'dts':       'dts',
    'mp3':       'mp3',
    'libmp3lame': 'mp3',
    'flac':      'flac',
    'opus':      'opus',
    'libopus':   'opus',
    'vorbis':    'wav',
    'libvorbis': 'wav',
    'wav':       'wav',
    'pcm_s16le': 'wav',
    'pcm_s24le': 'wav'
}

# Function to get info from file to figure out the input codec for the audio stream
def get_info(video_file):
    print("##########\nGetting audio and subtitle info from video file...\n##########")

    # 1) FFprobe to find all audio streams (and their tags)
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=index:stream_tags=language:stream=codec_name,bit_rate,duration',
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

    # 2) Parse the JSON
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            # data['streams'] should be a list of audio streams
            audio_streams = data.get('streams', [])

            # Try to find an English track first
            for stream in audio_streams:
                tags = stream.get('tags', {})
                language = tags.get('language', '').lower()
                if language == 'eng':
                    audio_stream_index = stream.get('index')
                    audio_codec = stream.get('codec_name', audio_codec)
                    bit_rate = stream.get('bit_rate', bit_rate)
                    duration = stream.get('duration', duration)
                    break

            # If we didn't find English, just grab the first audio stream
            if audio_stream_index is None and audio_streams:
                first_stream = audio_streams[0]
                audio_stream_index = first_stream.get('index')
                audio_codec = first_stream.get('codec_name', audio_codec)
                bit_rate = first_stream.get('bit_rate', bit_rate)
                duration = first_stream.get('duration', duration)

        except json.JSONDecodeError:
            print("Error: Failed to parse stream information JSON.")
    else:
        print("Error: Failed to get stream information via ffprobe.")

    # 3) Check for subtitle streams
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

    # 4) Print debugging info
    print(f"##########\nAudio Stream Index Chosen: {audio_stream_index}")
    print(f"Audio Codec: {audio_codec}")
    print(f"Bitrate: {bit_rate}")
    print(f"Duration: {duration} seconds\n##########")
    print(f"##########\nSubtitles Exist in Video File: {subtitles_exist}\n##########")

    # 5) Check for external SRT subtitle file
    base_name, _ = os.path.splitext(video_file)
    subtitle_file = base_name + ".srt"
    external_srt_exists = os.path.isfile(subtitle_file)
    print(f"##########\nExternal SRT Subtitle File Exists: {external_srt_exists}\n##########")

    # 6) Return everything needed
    return audio_stream_index, audio_codec, bit_rate, duration, subtitles_exist, external_srt_exists


####################################################################
#    HELPER FUNCTIONS FOR DEFINING THE FILENAMES AND EXTENSIONS.   #
####################################################################
def get_audio_extension(codec_name: str) -> str:
    """
    Given an audio codec name, return the proper file extension.
    If not in the map, we just return the codec_name itself.
    """
    return AUDIO_EXTENSION_MAP.get(codec_name, codec_name)

def get_extracted_filename(video_file: str, codec_name: str) -> str:
    """
    For the extracted audio, we simply want <base>.<ext>.
    E.g., "/path/video.mkv" -> "/path/video.ogg" if codec_name=libvorbis.
    """
    base_name, _ = os.path.splitext(video_file)
    ext = get_audio_extension(codec_name)
    return f"{base_name}.{ext}"

def get_defused_filename(audio_file: str, codec_name: str) -> str:
    """
    For the muted/defused audio, we want <base>-DEFUSED-AUDIO.<ext>.
    E.g., "/path/video.ogg" -> "/path/video-DEFUSED-AUDIO.ogg"
    """
    base_name, _ = os.path.splitext(audio_file)
    ext = get_audio_extension(codec_name)
    return f"{base_name}-DEFUSED-AUDIO.{ext}"

def get_ac3_or_copy(audio_file: str):
    """
    Decide how to encode the defused audio, 
    based on the input file extension or user logic.
    
    Returns: (out_codec, extra_args, defused_ext)
    """
    # Grab extension (lowercase, just in case)
    _, ext = os.path.splitext(audio_file)
    ext = ext.lower()
    
    # If the input is .wav => produce AC3
    if ext == '.wav':
        out_codec = 'ac3'
        extra_args = []
        defused_ext = '.ac3'
    else:
        # Otherwise, we’ll just keep it “copy” or revert to the old approach
        out_codec = 'copy'
        extra_args = []
        defused_ext = ext  
    
    return out_codec, extra_args, defused_ext

####################################################################

def extract_subtitles(video_file, subtitles_exist, external_srt_exists):
    print("##########\nExtracting subtitles from video file...\n##########")
    base_name, _ = os.path.splitext(video_file)
    subtitle_file = base_name + ".srt"

    if external_srt_exists:
        with open(subtitle_file, 'r', encoding='utf-8') as file:
            subtitles = file.read()
    else:
        # Extract subtitles from the video file using ffmpeg
        subtitle_file = base_name + "_subtitles.srt"
        cmd = ['ffmpeg', '-i', video_file, '-map', '0:s:0', subtitle_file]
        subprocess.run(cmd, text=True)
        with open(subtitle_file, 'r', encoding='utf-8') as file:
            subtitles = file.read()

    # Search for instances of the string "fuck" in a case-insensitive manner
    matches = re.findall(r'fuck', subtitles, re.IGNORECASE)
    if matches:
        print(f"##########\nFound {len(matches)} instances of 'f**k' in subtitles.\n##########")
        subtitle_swears = True
    else:
        print("##########\nNo instances of 'f**k' found in subtitles.\n##########")
        subtitle_swears = False

    return subtitle_swears

def extract_audio(video_file, audio_index, audio_codec, bit_rate, duration):
    print("##########\nExtracting audio from video file...\n##########")
    
    # A single lookup table unifying extension, ffmpeg_codec, and extra_args
    codec_map = {
        'dts': {
            'ext': '.wav',
            'ffmpeg_codec': 'pcm_s16le',
            'extra_args': ['-ar', '16000', '-ac', '1']
        },
        'aac': {
            'ext': '.aac',
            'ffmpeg_codec': 'copy',
            'extra_args': ['-strict', '-2']
        },
        # If the source is 'vorbis' or 'libvorbis', force to wav/pcm
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
        # Add more entries here if you want to handle EAC3, etc.
    }

    # Decide which codec settings to use
    if audio_codec in codec_map:
        chosen_ext = codec_map[audio_codec]['ext']
        chosen_codec = codec_map[audio_codec]['ffmpeg_codec']
        extra_args = codec_map[audio_codec]['extra_args']
    else:
        # Fallback: just copy with extension set to the codec name
        chosen_ext = f".{audio_codec}"
        chosen_codec = 'copy'
        extra_args = ['-strict', '-2']

    # Construct the output filename (e.g. "MyVideo.ogg")
    output_audio = os.path.splitext(video_file)[0] + chosen_ext

    # Build one FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_file,
        '-map', f'0:{audio_index}',  # The chosen audio track index
        '-vn',                       # No video
        '-acodec', chosen_codec,
        *extra_args
    ]

    # If a duration was found, limit to that duration
    if duration:
        cmd.extend(['-t', str(duration)])

    # Finally, add the output filename
    cmd.append(output_audio)

    # Debugging if desired:
    # print("FFmpeg command:", " ".join(cmd))

    # Execute the command
    subprocess.run(cmd, text=True)

    # Return the path to the extracted audio file
    return output_audio

def convert_to_mp3(audio_file, duration):
    print("##########\nConverting audio to MP3 format...\n##########")
    # Append ".mp3" extension to the audio file
    mp3_audio_file = os.path.splitext(audio_file)[0] + ".mp3"

    # Use ffmpeg command to convert audio to MP3 format at 256kbps
    cmd = ['ffmpeg', '-i', audio_file, '-vn', '-acodec', 'libmp3lame', '-b:a', '256k', mp3_audio_file]

    if duration:
        cmd.insert(-1, '-t')  # Add '-t' before the output file
        cmd.insert(-1, str(duration))

# Execute command
    subprocess.run(cmd, text=True)

    print("##########\nAudio conversion to MP3 completed.\n##########")
    return mp3_audio_file

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file, output_transcription=False):
    print("##########\nTranscribing audio into text to find F-words...\n##########")

    # Check if CUDA is available, and set it 
    if torch.cuda.is_available():
        print("##########\nCUDA is available: Using GPU!\n##########")
        device = "cuda"
    else:
        print("##########\nCUDA not available: Using CPU.\n##########")
        device = "cpu"

    # Load the Whisper model
    with tqdm(total=100, desc="Loading Whisper model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        model = whisper.load_model("base", device=device)
        for _ in range(100):
            pbar.update(1)
            time.sleep(0.01)
        
    if hasattr(model, "device"):
        print(f"Model loaded on device: {model.device}")

    # Transcribe the audio
    with tqdm(total=100, desc="Transcribing audio", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        result = model.transcribe(audio_file, word_timestamps=True)
        for _ in range(100):
            pbar.update(1)
            time.sleep(0.05)

    transcription_text = result['text']
    print("##########\nTranscription Completed\n##########")
    
    # If output_transcription is True, save to a file
    if output_transcription:
        transcription_file = os.path.splitext(audio_file)[0] + "_transcription.txt"
        with open(transcription_file, 'w', encoding='utf-8') as file:
            file.write(transcription_text)
        print(f"##########\nTranscription saved to: {transcription_file}\n##########")

    # Process swear words
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

def compare_with_subtitles(transcribed_text, subtitle_file):
    print("##########\nComparing transcription with subtitles...\n##########")

    # Read the subtitle file
    with open(subtitle_file, 'r') as file:
        subtitle_lines = file.readlines()

    # Initialize variables
    missing_f_words = []
    current_dialogue = ""

    # Iterate through subtitle lines
    for line in subtitle_lines:
        # If the line is empty, it's a new dialogue
        if line.strip() == "":
            # Check if any F-word is missing in the current dialogue
            if any("fuck" in word.lower() for word in current_dialogue.split()):
                missing_f_words.append(current_dialogue)
            current_dialogue = ""
        else:
            # Append the dialogue to the current dialogue
            current_dialogue += line.strip() + " "

    # Check if any F-word is missing in the last dialogue
    if any("fuck" in word.lower() for word in current_dialogue.split()):
        missing_f_words.append(current_dialogue)

    # Compare missing F-words with the transcribed text
    for dialogue in missing_f_words:
        dialogue_words = dialogue.split()
        for word in dialogue_words:
            if "fuck" in word.lower() and word not in transcribed_text:
                # Implement logic to find timestamps based on surrounding words
                print(f"Missing F-word: {word}")
                # Add logic here to find timestamps based on surrounding words

    print("##########\nComparison complete.\n##########")

# Function to mute audio at specified timestamps using FFmpeg
def mute_audio(audio_only_file, swears):
    """
    Mute the specified swear words in the given audio file by applying volume=0
    filters at the timestamps. If audio_only_file is .wav, final track is AC3;
    otherwise, we copy the codec.
    """
    # Initialize an empty list to store filter expressions for muting
    filter_expressions = []

    # Construct the filter expression for each swear word
    print("##########\nIterating through swear list and muting...\n##########")
    for swear in swears:
        print("Swear tuple:", swear)
        start = float(swear[1])
        end = float(swear[2])
        filter_expressions.append({'start': start, 'end': end})

    # Build the FFmpeg volume filters
    filter_string = ', '.join(
        f"volume=enable='between(t,{expr['start']},{expr['end']}):volume=0'"
        for expr in filter_expressions
    )

    print("##########\nMuting all F-words...\n##########")
    print(f"Filter String: {filter_string}")

    # Decide whether to encode as AC3 (if .wav) or copy otherwise
    out_codec, extra_args, defused_ext = get_ac3_or_copy(audio_only_file)

    # Construct our final DEFUSED filename
    base_name, _ = os.path.splitext(audio_only_file)
    defused_audio_file = f"{base_name}-DEFUSED-AUDIO{defused_ext}"

    # Build the ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', audio_only_file,
        '-vn',                 # No video
        '-af', filter_string,  # The volume filters
        '-c:a', out_codec,     # "ac3" or "copy"
        *extra_args,           # If AC3, you can add extra args like "-b:a 640k" etc. in get_ac3_or_copy
        defused_audio_file
    ]

    # Print or debug if you wish
    # print("FFmpeg command:", " ".join(cmd))

    # Execute the ffmpeg command
    subprocess.run(cmd, text=True)

    # Return the path to the defused audio file
    return defused_audio_file


def remove_int_files(*file_paths):
    # Remove intermediate audio files
    print("##########\nRemoving intermediate files...\n##########")
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"##########\nDeleted: {file_path}\n##########")
        else:
            print(f"##########\nFile not found: {file_path}\n##########")
def main():
    # Get user input for the video files
    parser = argparse.ArgumentParser(description='Process video files and mute profanity.')
    parser.add_argument('-i', '--input', nargs='+', help='Input video files', required=True)
    parser.add_argument('--ignore-subtitles', action='store_true', help='Ignore subtitles check')
    parser.add_argument('--output-transcription', action='store_true', help='Output transcription to a file')
    args = parser.parse_args()

    video_files = args.input
    ignore_subtitles = args.ignore_subtitles
    output_transcription = args.output_transcription

    # Loop through each input file
    for video_file in video_files:
        # Convert user input to absolute path
        video_file = os.path.abspath(video_file)

        # Check if the file exists
        if not os.path.isfile(video_file):
            print(f"##########\nError: File not found: {video_file}\n##########")
            continue

        # Get the directory and filename parts
        directory, filename = os.path.split(video_file)
        base_name, extension = os.path.splitext(filename)

        # Check if a file with '-CLEAN' appended exists already (indicating I've already cleaned it)
        clean_filename = f"{base_name}-CLEAN{extension}"
        clean_file_path = os.path.join(directory, clean_filename)
        if os.path.exists(clean_file_path):
            print(f"A defused file with the name '{clean_filename}' already exists in the directory, skipping: {video_file}")
            continue

        # Run a probe command on the video file to get all the codec, bitrate, and subtitle info we need first:
        audio_index, audio_codec, bit_rate, duration, subtitles_exist, external_srt_exists = get_info(video_file)

        # Extract subtitles if they exist, and if there are no swears exit. 
        if not ignore_subtitles and (subtitles_exist or external_srt_exists):
            subtitle_swears = extract_subtitles(video_file, subtitles_exist, external_srt_exists)
            if not subtitle_swears:
                print("##########\nNo F-words found in subtitles. Exiting gracefully.\n##########")
                continue

        # Extract audio from video
        audio_only_file = extract_audio(video_file, audio_index, audio_codec, bit_rate, duration) 

        # Convert audio to mp3 for better Whisper compatibility
        mp3_audio_file = convert_to_mp3(audio_only_file, duration)

        # Transcribe audio to text and obtain timestamps
        swears = transcribe_audio(mp3_audio_file, output_transcription)

        # Check if no F-words were found
        if not swears:
            print("##########\nNo F-words found. Exiting gracefully.\n##########")
            remove_int_files(audio_only_file)
            continue

        # Mute audio at specified timestamps to "defuse" the f-bombs
        defused_audio_file = mute_audio(audio_only_file, swears)

        # Compare the transcription with subtitles if they exist
        #if subtitles:
        #    compare_with_subtitles(swears, subtitles)

        # Append the desired suffix and the original extension to the base name
        directory, filename = os.path.split(video_file)
        base_name, extension = os.path.splitext(filename)
        clean_video_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")

        # Combine modified audio with original video
        print("##########\nAdding edited audio as a second audio stream to the original video file...\n##########")
        cmd = ['ffmpeg', '-i', video_file, '-i', defused_audio_file, '-c:v', 'copy', '-map', '0:v:0', '-map', '0:a:0', '-map', '1:a:0', '-metadata:s:a:1', 'language=eng', '-metadata:s:a:1', 'title=Defused (CLEAN) Track', clean_video_file]
        subprocess.run(cmd)

        if os.path.exists(clean_video_file):
            print(f"##########\nSuccessfully created clean video file: {clean_video_file}\n##########")
            remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file, video_file)
        else:
            print(f"##########\nFailed to create clean video file: {clean_video_file}. Skipping original file deletion.\n##########")
            remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file)  # Do not delete the original file

if __name__ == "__main__":
    main()
