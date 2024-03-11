import subprocess
import whisper
import os
import argparse
import json

# Function to extract audio from video using FFmpeg
def extract_audio(video_file, audio_file):
    print("##########\nExtracting audio from video file...\n##########")
    cmd = ['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', audio_file]
    subprocess.run(cmd, text=True)
    return audio_file

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file):
    "Transcribing audio into text to find F-words..."
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps="True")
    # Extract transcribed text and corresponding timestamps
    transcribed_text = result["text"]

    # Determine the filename for the transcription file
    base_name = os.path.basename(audio_file)
    filename, _ = os.path.splitext(base_name)
    filename_parts = filename.split('.')

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

    return swear_list


# Function to mute audio at specified timestamps using FFmpeg
def mute_audio(audio_file, swears):
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
    base_name, _ = os.path.splitext(os.path.basename(audio_file))
    directory = os.path.dirname(audio_file)
    muted_audio_file = os.path.join(directory, f"{base_name}-MUTED.wav")

    # Construct ffmpeg command with a complex filtergraph
    print("##########\nMuting all F-words...\n##########")
    print(f"Filter String: {filter_string}")
    cmd = ['ffmpeg', '-i', audio_file, '-vn', '-af', filter_string, '-c:a', 'pcm_s16le', '-strict', 'experimental', muted_audio_file]
    
    # Execute the command
    subprocess.run(cmd)
    return muted_audio_file


def main():
    # Get user input for the video file
    parser = argparse.ArgumentParser(description='Process video file and mute profanity.')
    parser.add_argument('-i', '--input', help='Input video file', required=True)
    args = parser.parse_args()

    video_file = args.input

    # Convert user input to absolute path
    video_file = os.path.abspath(video_file)

    # Check if the file exists
    if not os.path.isfile(video_file):
        print("##########\nError: File not found.\n##########")
        exit()

    base_name, _ = os.path.splitext(os.path.basename(video_file))
    directory = os.path.dirname(video_file)
    extracted_audio_file = os.path.join(directory, f"{base_name}-AUDIO-ONLY.wav")
    print(f"##########\nVideo File: {video_file}, base_name: {base_name}, extracted_audio_file: {extracted_audio_file}\n##########")

    # Extract audio from video
    extract_audio(video_file, extracted_audio_file)

    # Transcribe audio to text and obtain timestamps
    swears = transcribe_audio(extracted_audio_file)

     # Check if no F-words were found
    if not swears:
        print("##########\nNo F-words found. Exiting gracefully.\n##########")
        os.remove(extracted_audio_file)
        exit()
        
    # Mute audio at specified timestamps
    muted_audio_file = mute_audio(extracted_audio_file, swears)

    # Compress WAV file down to AAC to preserve size of original video file
    print("##########\nConvert wav to aac file for size\n##########")
    aac_file = f"{os.path.splitext(muted_audio_file)[0]}.aac"
    cmd = ['ffmpeg', '-i', muted_audio_file, '-c:a', 'aac', '-b:a', '320k', aac_file]
    subprocess.run(cmd)

    # Combine modified audio with original video
    #ffmpeg -i video.mp4 -i muted-audio.aac -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4
    directory, filename = os.path.split(video_file)
    base_name, extension = os.path.splitext(filename)
    # Append the desired suffix and the original extension to the base name
    clean_video_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")

    print("##########\nCombining edited audio with original video file...\n##########")
    cmd = ['ffmpeg', '-i', video_file, '-i', aac_file, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', clean_video_file]
    subprocess.run(cmd)

    # Remove intermediate audio files
    print("##########\nRemoving intermediate files...\n##########")
    os.remove(extracted_audio_file)
    os.remove(muted_audio_file)
    os.remove(aac_file)

if __name__ == "__main__":
    main()