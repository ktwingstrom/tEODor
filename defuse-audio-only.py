import subprocess
import whisper
import os
import argparse
import json
import time
import torch
import ffmpeg

# Function to get info from file to figure out the input codec for the audio stream
def get_audio_info(audio_file):
    print("##########\nGetting audio info from Video file...\n##########")
    # Run ffprobe command to get information about the audio stream
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index:stream_tags=NAME:stream=codec_name:stream=bit_rate', '-of', 'json', audio_file]
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)

    # First check to see if I've already edited this file:
    try:
        streams_info = json.loads(result.stdout)['streams']
        # Check if any stream has the specified name
        for stream in streams_info:
            if 'tags' in stream and 'NAME' in stream['tags'] and stream['tags']['NAME'] == 'Defused (CLEAN) Track':
                print("Error: Found an existing audio stream with the name 'Defused (CLEAN) Track'. Exiting the script.")
                exit()
    except json.JSONDecodeError:
        print("Error: Failed to parse audio information JSON.")
        return 'mp3', '128000'

    if result.returncode != 0:
        print("Error: Failed to get audio codec information, defualt to aac at 128kbps.")
        return 'mp3', '128000'
    # Parse the JSON output to get the audio codec
    try:
        codec_info = json.loads(result.stdout)

        # Try to get codec information
        try:
            audio_codec = codec_info['streams'][0]['codec_name']
        except (KeyError, IndexError):
            print("Error: Failed to parse audio codec information. Defaulting to 'mp3'.")
            audio_codec = 'mp3'

        # Try to get bitrate information
        try:
            bit_rate = codec_info['streams'][0]['bit_rate']
        except (KeyError, IndexError):
            print("Error: Failed to parse bitrate information. Defaulting to '128kbps'.")
            bit_rate = '128000'

        print(f"##########\nAudio Codec & Bitrate from source:\nCodec:{audio_codec}\nBitrate:{bit_rate}\n##########")
        return audio_codec, bit_rate
    
    except json.JSONDecodeError:
        print("Error: Failed to parse audio information JSON.")
        return 'mp3', '128000'

# Function to transcribe audio to text using OpenAI Whisper
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
    transcription_file = f"{filename_prefix}-TRANSCRIPTION.txt"
    #with open(transcription_file, 'w') as file:
    #    file.write(transcribed_text)

    # Write segments to a JSON file for troubleshooting
    segments_file = f"{filename_prefix}-SEGMENTS.json"
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
    # Get user input for the video file
    parser = argparse.ArgumentParser(description='Process audio file and mute profanity.')
    parser.add_argument('-i', '--input', help='Input audio file', required=True)
    args = parser.parse_args()

    audio_file = args.input

    # Uncomment hardcoded video file for testing
    #audio_file = '/Users/kevint/Downloads/Test/Its.Always.Sunny.in.Philadelphia.S16E07.720p.WEB.x265-MiNX.eztv.re.mkv'

    # Convert user input to absolute path
    audio_file = os.path.abspath(audio_file)

    # Check if the file exists
    if not os.path.isfile(audio_file):
        print("##########\nError: File not found.\n##########")
        exit()
    
    # Get the directory and filename parts
    directory, filename = os.path.split(audio_file)
    base_name, extension = os.path.splitext(filename)

    # Check if a file with '-CLEAN' appended exists already (indicating I've already cleaned it)
    clean_filename = f"{base_name}-CLEAN{extension}"
    clean_file_path = os.path.join(directory, clean_filename)
    if os.path.exists(clean_file_path):
        print(f"A defused file with the name '{clean_filename}' already exists in the directory, exiting")
        exit()

    # Run a probe command on the video file to get all the codec and bitrate info we need first:
    audio_codec, bit_rate = get_audio_info(audio_file)

    # Extract audio from video
    audio_only_file = extract_audio(audio_file, audio_codec, bit_rate)

    # Convert audio to mp3 for better Whisper compatibility
    mp3_audio_file = convert_to_mp3(audio_only_file)

    # Transcribe audio to text and obtain timestamps
    swears = transcribe_audio(mp3_audio_file)

     # Check if no F-words were found
    if not swears:
        print("##########\nNo F-words found. Exiting gracefully.\n##########")
        remove_int_files(audio_only_file)
        exit()
        
    # Mute audio at specified timestamps to "defuse" the f-bombs
    defused_audio_file = mute_audio(audio_only_file, swears, audio_codec, bit_rate)

    # Append the desired suffix and the original extension to the base name
    directory, filename = os.path.split(audio_file)
    base_name, extension = os.path.splitext(filename)
    clean_audio_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")

    # Combine modified audio with original video
    print("##########\nAdding edited audio as a second audio stream to the original video file...\n##########")
    cmd = ['ffmpeg', '-i', audio_file, '-i', defused_audio_file, '-c:v', 'copy', '-map', '0:v:0', '-map', '0:a:0', '-map', '1:a:0', '-metadata:s:a:1', 'language=eng', '-metadata:s:a:1', 'title=Defused (CLEAN) Track', clean_audio_file]
    subprocess.run(cmd)

    # Remove all intermediate files
    remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file, audio_file)

if __name__ == "__main__":    
    main()

