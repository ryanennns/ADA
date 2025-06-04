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
from kokoro import KPipeline

# === Settings ===
device = 'pulse'
samplerate = 16000
blocksize = 512
silence_timeout = 1.0
model_path = "vosk-model-en-us-0.22"
ollama_model = "gemma3:4b"

# === Initialize speech components ===
q = queue.Queue()
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
vad = load_silero_vad()
recording = False
buffer = []
last_speech_time = None

# === Initialize Kokoro TTS ===
pipeline = KPipeline(lang_code='a')  # American English voice set

def speak_kokoro(text):
    generator = pipeline(
        text,
        voice='af_heart',
        speed=1.0,
        split_pattern=None
    )
    for _, _, audio in generator:
        sd.play(audio, samplerate=24000, blocking=True)

def callback(indata, frames, time_info, status):
    q.put(bytes(indata))

# === Ollama Chat with Context ===
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful, thoughtful assistant engaged in natural conversation. "
            "Speak like a human would in voice, avoiding markdown, formatting, or code blocks. "
            "Be concise, expressive, and natural. Do not use symbols, emojis, or special punctuation unless absolutely required."
        )
    }
]

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

# === Main Loop ===
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
        elif recording and (time.time() - last_speech_time > silence_timeout):
            print("[Silero VAD] Speech ended")
            print("[Vosk] Transcribing...")

            full_audio = b''.join(buffer)
            rec.AcceptWaveform(full_audio)
            result = json.loads(rec.FinalResult())
            transcript = result.get("text", "")
            print("[Vosk] Transcript:", transcript)

            if transcript:
                print("[Ollama] Sending prompt...")
                response = query_ollama(transcript)
                print("[Kokoro] Speaking...")
                speak_kokoro(response)

            print("[Vosk] Ready.")
            recording = False
