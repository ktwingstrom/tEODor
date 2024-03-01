import subprocess
import whisper
import os

# Function to extract audio from video using FFmpeg
def extract_audio(video_file, audio_file):
    cmd = ['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_file]
    subprocess.run(cmd)

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps="True")
    # Extract transcribed text and corresponding timestamps
    transcribed_text = result["text"]

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
            end = word_obj['end']

            # Do something with the word, start, and end values
            if "fuck" in word:
                swear_list.append((word, start, end))
                print(f"Word: {word}, Start: {start}, End: {end}")
    #timestamps = result["audio_time"]
    return swear_list

# Function to mute audio at specified timestamps using FFmpeg
def mute_audio(video_file, swears):
    for swear in swears:
        print("Swear tuple:", swear)
        start = float(swear[1])
        end = float(swear[2])
        # Construct ffmpeg command with properly formatted timestamps and escaped quotes
        cmd = ['ffmpeg', '-i', video_file, '-af', f"volume=enable='between(t,{start_time},{end_time})':volume=0", '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', f'muted_{video_file}']
        subprocess.run(cmd)


# Example usage
video_file = 'Schitts.Creek.S01E03.720p.WEB-DL.x265-HETeam.mkv'
extracted_audio_file = 'scs1e3.wav'

# Extract audio from video
extract_audio(video_file, extracted_audio_file)

# Transcribe audio to text and obtain timestamps
swears = transcribe_audio(extracted_audio_file)

# Mute audio at specified timestamps
mute_audio(video_file, swears)

# Combine modified audio with original video
#cmd = ['ffmpeg', '-i', f'muted_{extracted_audio_file}', '-i', video_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 'output_video.mp4']
#subprocess.run(cmd)