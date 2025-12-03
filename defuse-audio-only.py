import subprocess
import os
import argparse
import json
import time
import ffmpeg
from faster_whisper import WhisperModel

# Function to get info from file to figure out the input codec for the audio stream
def get_audio_info(audio_file):
    print("##########\nGetting audio info from file...\n##########")
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

# Function to transcribe audio to text using faster-whisper
def transcribe_audio(mp3_audio_file):
    print("##########\nTranscribing audio into text to find F-words...\n##########")

    # Measure the start time
    start_time = time.time()

    # Determine compute type and device
    # faster-whisper auto-detects CUDA, we just specify compute type
    # For CPU: use "int8" for best speed/accuracy tradeoff
    # For GPU: use "float16" when available
    compute_type = "int8"  # CPU optimized
    device = "cpu"

    print(f"##########\nLoading Whisper model (faster-whisper)...\n##########")
    print(f"##########\nDevice: {device}, Compute type: {compute_type}\n##########")

    # Load the Whisper model
    # model_size can be: tiny, base, small, medium, large-v2, large-v3
    model = WhisperModel("base", device=device, compute_type=compute_type)

    print("##########\nTranscribing audio...\n##########")

    # Transcribe the audio file
    # faster-whisper returns segments generator which is memory-efficient
    segments, info = model.transcribe(
        mp3_audio_file,
        beam_size=5,
        word_timestamps=True,  # Enable word-level timestamps
        vad_filter=True  # Voice activity detection to improve accuracy
    )

    # Measure the end time
    end_time = time.time()
    duration = end_time - start_time

    print(f"##########\nDetected language '{info.language}' with probability {info.language_probability}\n##########")
    print(f"##########\nTranscription completed in {duration:.2f} seconds\n##########")

    # Determine the filename for the transcription file
    base_name = os.path.basename(mp3_audio_file)
    filename, _ = os.path.splitext(base_name)
    filename_parts = filename.split('.')

    # Find the index of the first occurrence of 'S' followed by a number
    season_index = next((i for i, part in enumerate(filename_parts) if part.startswith('S') and part[1:].isdigit()), None)
    if season_index is not None:
        filename_prefix = '.'.join(filename_parts[:season_index+1])
    else:
        filename_prefix = filename

    # Instantiate empty lists
    swear_list = []
    transcribed_text_parts = []
    segments_data = []

    # Process segments and words
    print("##########\nProcessing segments for F-words...\n##########")
    for segment in segments:
        transcribed_text_parts.append(segment.text)

        # Store segment data for JSON output
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }

        # Process words in the segment if available
        if segment.words:
            segment_data["words"] = []
            for word in segment.words:
                segment_data["words"].append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })

                # Check if word contains profanity
                if "fuck" in word.word.lower():
                    # Add 0.1 second buffer to the end
                    end_time = word.end + 0.1
                    swear_list.append((word.word, word.start, end_time))
                    print(f"Word: {word.word}, Start: {word.start}, End: {end_time}")

        segments_data.append(segment_data)

    # Write transcription to a text file for troubleshooting
    transcription_file = f"{filename_prefix}-TRANSCRIPTION.txt"
    with open(transcription_file, 'w') as file:
        file.write(' '.join(transcribed_text_parts))

    # Write segments to a JSON file for troubleshooting
    segments_file = f"{filename_prefix}-SEGMENTS.json"
    with open(segments_file, 'w') as file:
        json.dump(segments_data, file, indent=4)

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

    # Map codec to file extension
    codec_to_extension = {
        'pcm_s16le': 'wav',
        'pcm_s24le': 'wav',
        'pcm_s32le': 'wav',
        'aac': 'aac',
        'ac3': 'ac3',
        'mp3': 'mp3',
        'opus': 'opus',
        'vorbis': 'ogg',
        'flac': 'flac'
    }

    # Get the proper file extension for the codec
    file_extension = codec_to_extension.get(audio_codec, os.path.splitext(audio_only_file)[1][1:])

    # Set up filename for muted file
    base_name, _ = os.path.splitext(audio_only_file)
    defused_audio_file = base_name + "-DEFUSED-AUDIO" + "." + file_extension

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

    # Transcribe audio to text and obtain timestamps
    swears = transcribe_audio(audio_file)

     # Check if no F-words were found
    if not swears:
        print("##########\nNo F-words found. Exiting gracefully.\n##########")
        exit()
        
    # Mute audio at specified timestamps to "defuse" the f-bombs
    defused_audio_file = mute_audio(audio_file, swears, audio_codec, bit_rate)

    # Append the desired suffix and the original extension to the base name
    directory, filename = os.path.split(audio_file)
    base_name, extension = os.path.splitext(filename)
    clean_audio_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")

    # Remove all intermediate files
    #remove_int_files(defused_audio_file, audio_only_file, mp3_audio_file, audio_file)

if __name__ == "__main__":    
    main()

