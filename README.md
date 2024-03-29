# tEODor
A tool for defusing (muting) F-bombs in video and audio content

USAGE:
run the defuse.py script and pass it a video file.  

Flags:

-i --> input file

ex:
python3 defuse.py -i /path/to/my_video.mp4

When the script runs it will extract the audio from the video file, then transcribe it to text using the whisper-python AI model.  It will locate all the f-bombs, and make a note of the exact timestamps of the beginning and ending of the word.  Then the audio file is fed into FFMPEG again and given a filter with the timestamps of each word we need to mute.  FFMPEG will mute JUST the f-words that were found.  Then it recombines the newly cleaned audio track with the original video file. The video file is not altered or transcoded at all. 

REQUIREMENTS:

openai-whisper (https://github.com/openai/whisper)

ffmpeg (https://formulae.brew.sh/formula/ffmpeg OR https://pypi.org/project/ffmpeg/)


#####################
BATCH FUNCTIONALITY #
#####################

Use defuse-all.sh -i /path/to/folder

This will look for all video files in the folder you specified recursively and perform the F-bomb defusion. 
PLEASE NOTE: you'll need to update the file location of the defuse.py script that is listed in the defuse-all.sh script. 
