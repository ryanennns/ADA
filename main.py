import sounddevice as sd
import queue
import sys
import json
import numpy as np
import torch
import time
import requests
from vosk import Model, KaldiRecognizer
from silero_vad import load_silero_vad

# === Settings ===
device = 'pulse'
samplerate = 16000
blocksize = 512
silence_timeout = 1.0  # seconds of silence to stop recording
model_path = "vosk-model-en-us-0.22"
ollama_model = "gemma3:4b"

# === Initialize ===
q = queue.Queue()
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
vad = load_silero_vad()
recording = False
buffer = []
last_speech_time = None

def callback(indata, frames, time_info, status):
    q.put(bytes(indata))

chat_history = []

def query_ollama(prompt):
    user_prompt = "/no_think " + prompt
    chat_history.append({"role": "user", "content": user_prompt})
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": ollama_model,
            "messages": chat_history,
            "stream": True
        },
        stream=True
    )

    if not response.ok:
        return f"[Ollama Error] {response.status_code}: {response.text}"

    collected = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            collected += content
        except json.JSONDecodeError:
            continue

    print()  # final newline after stream
    chat_history.append({"role": "assistant", "content": collected.strip()})
    return collected.strip()

# === Start Listening ===
print("Listening...")
with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize,
                       device=device, dtype='int16', channels=1,
                       latency='low', callback=callback):
    while True:
        data = q.get()
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        is_speech = vad(torch.from_numpy(samples), samplerate).item() > 0.5

        if is_speech:
            if not recording:
                print("\n[Silero VAD] Speech started")
                print("[Vosk] Starting transcription...")
                recording = True
                buffer = []
                rec.Reset()
            last_speech_time = time.time()
            buffer.append(data)
        elif recording:
            if time.time() - last_speech_time > silence_timeout:
                print("[Silero VAD] Speech ended")
                print("[Vosk] Transcribing...")

                full_audio = b''.join(buffer)
                rec.AcceptWaveform(full_audio)
                result = json.loads(rec.FinalResult())
                transcript = result.get("text", "")
                print("[Vosk] Transcript:", transcript)

                if transcript:
                    print("[Ollama] Sending prompt...")
                    query_ollama(transcript)

                print("[Vosk] Ready.")
                recording = False
