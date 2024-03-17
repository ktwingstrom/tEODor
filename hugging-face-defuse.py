import torch
from transformers import pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    return_timestamps=True,
    model="openai/whisper-large-v3",
    chunk_length_s=30,
    device=device,
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code="True")
sample = '/Users/kevint/Downloads/SchittsCreek/Season04/Schitts.Creek.S04E10.720p.webrip.x264-tbs-AUDIO.ac3'

# we can also return timestamps for the predictions
prediction = pipe(sample, generate_kwargs={"language": "english"}, batch_size=8, return_timestamps="word")["chunks"]
print(prediction["chunks"])