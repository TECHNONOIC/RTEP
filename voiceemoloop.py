import os
import pyaudio
import wave
import csv
import time
from transformers import pipeline

# Pipeline model for audio classification
pipe = pipeline("audio-classification", model="DunnBC22/wav2vec2-base-Speech_Emotion_Recognition", from_pt=True)

# Function to record audio from the microphone
def record_audio(output_file, duration=7):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Create output directory if it doesn't exist
output_dir = "audio_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create CSV file for saving emotions
csv_file = "voiceemotions.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "voice emotion"])

# Loop for recording, processing, and saving output
for i in range(5):
    # Record audio
    output_file = os.path.join(output_dir, f"audio_{i}.wav")
    record_audio(output_file)

    # Process audio
    result = pipe(output_file)
    emotion = result[0]['label']

    # Save emotion and time to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), emotion])

print("Emotions saved to", csv_file)
