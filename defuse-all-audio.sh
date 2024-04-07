#!/bin/bash

# Parse command-line options
while getopts ":i:" opt; do
  case ${opt} in
    i )
      # Remove trailing slashes from the directory path
      directory=$(echo "$OPTARG" | sed 's:/*$::')
      ;;
    \? )
      echo "Usage: $0 -i <directory>"
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument"
      echo "Usage: $0 -i <directory>"
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Check if directory option is provided
if [ -z "$directory" ]; then
  echo "Usage: $0 -i <directory>"
  exit 1
fi

# Find all audio files in the directory and its subdirectories
find "$directory" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" -o -iname "*.ogg" -o -iname "*.m4a" -o -iname "*.aac" -o -iname "*.wma" -o -iname "*.m4b" \) -exec sh -c '
    for audio_file do
        # Execute the defuse command for each video file
        echo "Audio File: $audio_file"
        /usr/bin/python3 /Users/kevint/Documents/scripts/tEODor/defuse-audio-only.py -i "$audio_file"
    done
' sh {} +
