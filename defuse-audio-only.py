import subprocess
import os
import argparse
import json
import time
import re
import ffmpeg
from faster_whisper import WhisperModel

###############################################################################
#                           PROFANITY PATTERNS                                #
###############################################################################
# Profanity patterns to detect (can be extended)
# These patterns match both standalone words and compound words
PROFANITY_PATTERNS = [
    r'\w*f+u+c+k+\w*',     # fuck and variations (fucker, fucking, motherfucker, etc.)
    r'\w*n+i+g+g+e+r+\w*', # n-word and variations
    r'\w*s+h+i+t+\w*',     # shit and variations (bullshit, shitty, etc.)
]

###############################################################################
#                              AUDIO CHUNKING                                 #
###############################################################################

# Default chunk duration in seconds (2 hours)
CHUNK_DURATION = 7200
# Overlap between chunks to avoid cutting words (5 seconds)
CHUNK_OVERLAP = 5

def get_audio_duration(audio_file):
    """Get the duration of an audio file in seconds."""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Could not get duration, assuming file needs chunking")
        return float('inf')
    return float(result.stdout.strip())


def split_audio_into_chunks(audio_file, chunk_duration=CHUNK_DURATION, overlap=CHUNK_OVERLAP):
    """
    Split a long audio file into smaller chunks for processing.
    Returns a list of (chunk_file_path, start_time) tuples.
    """
    duration = get_audio_duration(audio_file)

    if duration <= chunk_duration:
        # No need to split
        return [(audio_file, 0.0)]

    print(f"##########\nAudio is {duration/3600:.1f} hours long, splitting into chunks...\n##########")

    chunks = []
    base_name = os.path.splitext(audio_file)[0]
    chunk_dir = base_name + "_chunks"

    # Create temporary directory for chunks
    os.makedirs(chunk_dir, exist_ok=True)

    start_time = 0.0
    chunk_num = 0

    while start_time < duration:
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_num:04d}.mp3")

        # Calculate chunk end time (with overlap for next chunk)
        chunk_end = min(start_time + chunk_duration, duration)
        chunk_length = chunk_end - start_time

        print(f"Creating chunk {chunk_num + 1}: {start_time/60:.1f}m - {chunk_end/60:.1f}m")

        # Extract chunk using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', audio_file,
            '-ss', str(start_time),
            '-t', str(chunk_length),
            '-acodec', 'mp3', '-ab', '64k',  # Lower bitrate for temp chunks
            '-v', 'fatal',  # Only show fatal errors, suppress ID3 tag warnings
            chunk_file
        ]
        subprocess.run(cmd, check=True)

        chunks.append((chunk_file, start_time))

        # Break if we've reached the end
        if chunk_end >= duration:
            break

        # Move to next chunk, accounting for overlap
        start_time = chunk_end - overlap
        chunk_num += 1

    print(f"##########\nCreated {len(chunks)} chunks\n##########")
    return chunks


def cleanup_chunks(audio_file):
    """Remove temporary chunk files and directory."""
    base_name = os.path.splitext(audio_file)[0]
    chunk_dir = base_name + "_chunks"

    if os.path.exists(chunk_dir):
        import shutil
        shutil.rmtree(chunk_dir)
        print(f"##########\nCleaned up chunk directory: {chunk_dir}\n##########")


###############################################################################
#                           LOAD WHISPER MODEL                                #
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

# Function to transcribe a single audio chunk
def transcribe_chunk(model, chunk_file, time_offset=0.0):
    """
    Transcribe a single audio chunk and return swear words with adjusted timestamps.
    """
    swear_list = []
    transcribed_text_parts = []
    segments_data = []

    # Compile profanity patterns
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in PROFANITY_PATTERNS]

    # Transcribe the chunk
    segments, info = model.transcribe(
        chunk_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )

    # Process segments and words
    for segment in segments:
        transcribed_text_parts.append(segment.text)

        # Store segment data (with adjusted timestamps)
        segment_data = {
            "start": segment.start + time_offset,
            "end": segment.end + time_offset,
            "text": segment.text
        }

        if segment.words:
            segment_data["words"] = []
            for word in segment.words:
                # Adjust timestamps by adding the chunk's start time
                adjusted_start = word.start + time_offset
                adjusted_end = word.end + time_offset

                segment_data["words"].append({
                    "word": word.word,
                    "start": adjusted_start,
                    "end": adjusted_end
                })

                # Check if word matches any profanity pattern
                for pattern in compiled_patterns:
                    if pattern.search(word.word):
                        end_with_buffer = adjusted_end + 0.1
                        swear_list.append((word.word, adjusted_start, end_with_buffer))
                        print(f"Word: {word.word}, Start: {adjusted_start:.2f}, End: {end_with_buffer:.2f}")
                        break  # Don't add the same word multiple times

        segments_data.append(segment_data)

    return swear_list, transcribed_text_parts, segments_data, info


# Function to transcribe audio to text using faster-whisper
def transcribe_audio(mp3_audio_file):
    print("##########\nTranscribing audio into text to find F-words...\n##########")

    # Measure the start time
    start_time = time.time()

    # Split audio into chunks if needed
    chunks = split_audio_into_chunks(mp3_audio_file)
    is_chunked = len(chunks) > 1

    # Load the model once (reuse for all chunks)
    model, device = load_whisper_model()

    # Determine the filename for output files
    base_name = os.path.basename(mp3_audio_file)
    filename, _ = os.path.splitext(base_name)
    filename_parts = filename.split('.')

    season_index = next((i for i, part in enumerate(filename_parts) if part.startswith('S') and part[1:].isdigit()), None)
    if season_index is not None:
        filename_prefix = '.'.join(filename_parts[:season_index+1])
    else:
        filename_prefix = filename

    # Collect results from all chunks
    all_swears = []
    all_text_parts = []
    all_segments = []
    detected_language = None

    # Process each chunk
    for i, (chunk_file, time_offset) in enumerate(chunks):
        if is_chunked:
            print(f"##########\nProcessing chunk {i + 1}/{len(chunks)} (offset: {time_offset/60:.1f} min)...\n##########")
        else:
            print("##########\nTranscribing audio...\n##########")

        swears, text_parts, segments_data, info = transcribe_chunk(model, chunk_file, time_offset)

        all_swears.extend(swears)
        all_text_parts.extend(text_parts)
        all_segments.extend(segments_data)

        if detected_language is None:
            detected_language = info.language
            print(f"##########\nDetected language '{info.language}' with probability {info.language_probability}\n##########")

        # Free up memory after each chunk
        if is_chunked:
            import gc
            gc.collect()

    # Clean up chunk files if we created them
    if is_chunked:
        cleanup_chunks(mp3_audio_file)

        # Remove duplicate swears from overlap regions
        all_swears = deduplicate_swears(all_swears)

    # Measure completion time
    end_time = time.time()
    duration = end_time - start_time
    print(f"##########\nTranscription completed in {duration:.2f} seconds\n##########")

    # Write transcription to a text file for troubleshooting
    transcription_file = f"{filename_prefix}-TRANSCRIPTION.txt"
    with open(transcription_file, 'w') as file:
        file.write(' '.join(all_text_parts))

    # Write segments to a JSON file for troubleshooting
    segments_file = f"{filename_prefix}-SEGMENTS.json"
    with open(segments_file, 'w') as file:
        json.dump(all_segments, file, indent=4)

    print(f"##########\nTotal profanity found: {len(all_swears)}\n##########")
    return all_swears


def deduplicate_swears(swears):
    """
    Remove duplicate swear detections that may occur in chunk overlap regions.
    If two swears are within 1 second of each other, keep only the first one.
    """
    if not swears:
        return swears

    # Sort by start time
    sorted_swears = sorted(swears, key=lambda x: x[1])
    deduplicated = [sorted_swears[0]]

    for swear in sorted_swears[1:]:
        last_swear = deduplicated[-1]
        # If this swear starts more than 1 second after the last one, it's unique
        if swear[1] - last_swear[1] > 1.0:
            deduplicated.append(swear)

    if len(deduplicated) < len(swears):
        print(f"##########\nRemoved {len(swears) - len(deduplicated)} duplicate detections from overlap regions\n##########")

    return deduplicated

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

    try:
        # Transcribe audio to text and obtain timestamps
        swears = transcribe_audio(audio_file)

        # Check if no profanity was found
        if not swears:
            print("##########\nNo profanity found. Exiting gracefully.\n##########")
            exit()

        # Mute audio at specified timestamps to "defuse" the f-bombs
        defused_audio_file = mute_audio(audio_file, swears, audio_codec, bit_rate)

        # Append the desired suffix and the original extension to the base name
        directory, filename = os.path.split(audio_file)
        base_name, extension = os.path.splitext(filename)
        clean_audio_file = os.path.join(directory, f"{base_name}-CLEAN{extension}")

        # Rename the defused file to the clean file
        print(f"##########\nRenaming {defused_audio_file} to {clean_audio_file}\n##########")
        os.rename(defused_audio_file, clean_audio_file)
        print(f"##########\nDone! Clean audio saved to: {clean_audio_file}\n##########")

    finally:
        # Always clean up chunks even if there's an error
        cleanup_chunks(audio_file)

if __name__ == "__main__":    
    main()

