import subprocess
import whisper
import os
import argparse
import json
import time
import torch
import pysrt
import re

# Function to get info from file to figure out the input codec for the audio stream
def get_info(video_file):
    print("##########\nGetting audio and subtitle info from video file...\n##########")
    # Run ffprobe command to get information about the audio and subtitle streams
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:s', '-show_entries', 'stream=index:stream_tags=NAME:stream=codec_name:stream=bit_rate', '-of', 'json', video_file]
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)

    audio_codec = 'aac'
    bit_rate = '320000'
    subtitles_exist = False

    # First check to see if the video file has been edited before:
    try:
        streams_info = json.loads(result.stdout)['streams']
        # Check if any stream has the specified name
        for stream in streams_info:
            if 'tags' in stream and 'NAME' in stream['tags'] and stream['tags']['NAME'] == 'Defused (CLEAN) Track':
                print("Error: Found an existing audio stream with the name 'Defused (CLEAN) Track'. Exiting the script.")
                exit()

        # Extract audio codec and bit rate
        for stream in streams_info:
            if 'codec_type' in stream and stream['codec_type'] == 'audio':
                audio_codec = stream.get('codec_name', audio_codec)
                bit_rate = stream.get('bit_rate', bit_rate)

            # Check for subtitle stream
            if 'codec_type' in stream and stream['codec_type'] == 'subtitle':
                subtitles_exist = True

    except json.JSONDecodeError:
        print("Error: Failed to parse stream information JSON.")

    if result.returncode != 0:
        print("Error: Failed to get stream information, defaulting to 'aac' codec at 320kbps.")
    
    print(f"##########\nAudio Codec & Bitrate from source:\nCodec: {audio_codec}\nBitrate: {bit_rate}\n##########")
    print(f"##########\nSubtitles Exist in Video File: {subtitles_exist}\n##########")

    # Check for external SRT subtitle file
    base_name, _ = os.path.splitext(video_file)
    subtitle_file = base_name + ".srt"
    external_srt_exists = os.path.isfile(subtitle_file)
    print(f"##########\nExternal SRT Subtitle File Exists: {external_srt_exists}\n##########")

    return audio_codec, bit_rate, subtitles_exist, external_srt_exists

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


def extract_audio(video_file, audio_codec, bit_rate):
    print("##########\nExtracting audio from video file...\n##########")
    # Use the ext that relates to the codec;
    # Append a dot before the audio codec value to create the extension
    audio_extension = f".{audio_codec}"

    # Change the extension of video_file to audio_extension
    base_name, _ = os.path.splitext(video_file)
    audio_file = base_name + "-AUDIO" + audio_extension

    # Use the determined audio codec in the ffmpeg command
    cmd = ['ffmpeg', '-i', video_file, '-vn', '-acodec', 'copy', '-strict', '-2', audio_file]
    #cmd = ['ffmpeg', '-i', video_file, '-vn', '-acodec', audio_codec, '-b:a', bit_rate, '-strict', '-2', audio_file]
    subprocess.run(cmd, text=True)
    return audio_file

def convert_to_mp3(audio_file):
    print("##########\nConverting audio to MP3 format...\n##########")
    # Append ".mp3" extension to the audio file
    mp3_audio_file = os.path.splitext(audio_file)[0] + ".mp3"

    # Use ffmpeg command to convert audio to MP3 format at 256kbps
    cmd = ['ffmpeg', '-i', audio_file, '-vn', '-acodec', 'libmp3lame', '-b:a', '256k', mp3_audio_file]
    subprocess.run(cmd, text=True)

    print("##########\nAudio conversion to MP3 completed.\n##########")
    return mp3_audio_file

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(mp3_audio_file):

    # Check if cuda is available
    print(f"##########\nCuda available? {torch.cuda.is_available()}\n##########")
    print("##########\nTranscribing audio into text to find F-words...\n##########")

    # Measure the start time
    start_time = time.time()

    model = whisper.load_model("base")
    result = model.transcribe(mp3_audio_file, word_timestamps="True")

    # Measure the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time
    print(f"##########\nTranscription completed in {duration:.2f} seconds\n##########")

    # Extract transcribed text and corresponding timestamps
    transcribed_text = result["text"]

    # Determine the filename for the transcription file
    base_name = os.path.basename(mp3_audio_file)
    filename, _ = os.path.splitext(base_name)
    filename_parts = filename.split('.')

    # This section is to name the transcription and segments file. Uncomment to use.
    # Find the index of the first occurrence of 'S' followed by a number
    season_index = next((i for i, part in enumerate(filename_parts) if part.startswith('S') and part[1:].isdigit()), None)
    if season_index is not None:
        filename_prefix = '.'.join(filename_parts[:season_index+1])
    else:
        filename_prefix = filename

    # Write transcription to a text file for troubleshooting
    #transcription_file = f"{filename_prefix}-TRANSCRIPTION.txt"
    #with open(transcription_file, 'w') as file:
    #    file.write(transcribed_text)

    # Write segments to a JSON file for troubleshooting
    #segments_file = f"{filename_prefix}-SEGMENTS.json"
    #with open(segments_file, 'w') as file:
    #    json.dump(result['segments'], file, indent=4)

    # pull segments from results
    segments = result['segments']

    # Instantiate empty list
    swear_list = []

    for segment in segments:
        # Access the 'words' list within the segment
        words_list = segment['words']

        # Iterate over the elements in the 'words' list
        for word_obj in words_list:
            # Access the 'word', 'start', and 'end' elements within each word object
            word = word_obj['word']
            start = word_obj['start']
            end = word_obj['end'] + 0.1

            # Do something with the word, start, and end values
            if "fuck" in word.lower():
                swear_list.append((word, start, end))
                print(f"Word: {word}, Start: {start}, End: {end}")
    print(f"##########\nTotal F-words: {len(swear_list)}\n##########")
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
def mute_audio(audio_only_file, swears, audio_codec, bit_rate):
    # Initialize an empty list to store filter expressions for muting
    filter_expressions = []

    # Construct the filter expression for each swear word
    print("##########\nIterating through swear list and muting...\n##########")
    for swear in swears:
        print("Swear tuple:", swear)
        start = float(swear[1])
        end = float(swear[2])

        # Define the filter expressions dynamically
        filter_expressions.append({'start': start, 'end': end})

    # Construct the filter string
    filter_string = ', '.join(
        f"volume=enable='between(t,{expr['start']},{expr['end']}):volume=0'"
        for expr in filter_expressions
    )

    # Set up filename for muted file
    base_name, _ = os.path.splitext(audio_only_file)
    defused_audio_file = base_name + "-DEFUSED-AUDIO" + "." + audio_codec

    # Construct ffmpeg command with a complex filtergraph
    print("##########\nMuting all F-words...\n##########")
    print(f"Filter String: {filter_string}")
    cmd = ['ffmpeg', '-i', audio_only_file, '-vn', '-af', filter_string, '-c:a', audio_codec, '-b:a', bit_rate, '-strict', 'experimental', defused_audio_file]

    # Execute the command
    subprocess.run(cmd)
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
    args = parser.parse_args()

    video_files = args.input

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
        audio_codec, bit_rate, subtitles_exist, external_srt_exists = get_info(video_file)

        # Extract subtitles if they exist, and if there are no swears exit. 
        if subtitles_exist or external_srt_exists:
            subtitle_swears = extract_subtitles(video_file, subtitles_exist, external_srt_exists)
            if not subtitle_swears:
                print("##########\nNo F-words found in subtitles. Exiting gracefully.\n##########")
                continue

        # Extract audio from video
        audio_only_file = extract_audio(video_file, audio_codec, bit_rate)

        # Convert audio to mp3 for better Whisper compatibility
        mp3_audio_file = convert_to_mp3(audio_only_file)

        # Transcribe audio to text and obtain timestamps
        swears = transcribe_audio(mp3_audio_file)

        # Check if no F-words were found
        if not swears:
            print("##########\nNo F-words found. Exiting gracefully.\n##########")
            remove_int_files(audio_only_file, mp3_audio_file)
            continue

        # Mute audio at specified timestamps to "defuse" the f-bombs
        defused_audio_file = mute_audio(audio_only_file, swears, audio_codec, bit_rate)

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

        # Remove all intermediate files
        remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file, video_file)


if __name__ == "__main__":
    main()
