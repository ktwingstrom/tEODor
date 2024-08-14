# tEODor
A tool for defusing (muting) F-bombs in video and audio content

USAGE:
run the defuse.py script and pass it a video file.  

Flags:

-i --> input file(s)
--ignore-subtitles --> this flag tells the script to skip checking embedded or external subtitles for f-bombs.  This is a default check to quickly assess whether filtering needs to be done. 

ex:
python3 defuse.py -i /path/to/my_video.mp4

#####################
Script Flow
#####################

When the script runs it will first check for subtitles, and quickly scan through them for the presence of any F words as a basic check to see if the audio even needs to be filtered.  If there are no subtitles found it will just continue as normal.  If there are subtitles and no swears are found it exits, otherwise it continues. 

Then it will extract the audio from the video file, then transcribe it to text using the whisper-python AI model.  

It will locate all the f-bombs, and make a note of the exact timestamps of the beginning and ending of the word. 

Then the audio file is fed into FFMPEG again and given a filter with the timestamps of each word we need to mute.  

FFMPEG will mute JUST the f-words that were found.  

Then it will add the newly cleaned audio track back into the original container (mp4, mkv, etc). The new audio track will have "(Defused)" appended to it. The video file is not altered or transcoded at all. 

REQUIREMENTS:

openai-whisper (https://github.com/openai/whisper)

ffmpeg (https://formulae.brew.sh/formula/ffmpeg OR https://pypi.org/project/ffmpeg/)


#####################
BATCH FUNCTIONALITY 
#####################

You may pass multiple filenames to the script and it will execute fiiltering on all files in sequence.  If you have a folder full of files you can just pass the /folder/location/* and it will try to filter all video files in the folder.  

This will look for all video files in the folder you specified recursively and perform the F-bomb defusing. 
PLEASE NOTE: you'll need to update the file location of the defuse.py script that is listed in the defuse-all.sh script. 
