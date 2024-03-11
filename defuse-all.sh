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

# Find all video files in the directory and its subdirectories
find "$directory" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov" \) -exec sh -c '
    for video_file do
        # Execute the defuse command for each video file
        echo "Video File: $video_file"
        python3 /Users/kevint/Documents/scripts/tEODor/defuse.py -i "$video_file"
    done
' sh {} +
