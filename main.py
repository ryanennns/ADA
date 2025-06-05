import base64
import json
import os
import queue
import re
import time

import cv2
import numpy as np
import requests
import sounddevice as sd
import torch
from dotenv import load_dotenv
from kokoro import KPipeline
from silero_vad import load_silero_vad
from vosk import Model, KaldiRecognizer

from tool_definitions import get_current_time

load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")
vosk_model = os.getenv("VOSK_MODEL")
kokoro_voice = os.getenv("KOKORO_VOICE")
ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE")) or 0.5
silence_timeout = float(os.getenv("SILENCE_TIMEOUT")) or 0.5

print("[System] Ollama model: " + ollama_model)
print("[System] Vosk model: " + vosk_model)
print("[System] Kokoro voice: " + kokoro_voice)
print("[System] Ollama temperature: " + str(ollama_temperature))

device = 'pulse'
samplerate = 16000
blocksize = 512

q = queue.Queue()
loaded_vosk_model = Model(vosk_model)
rec = KaldiRecognizer(loaded_vosk_model, samplerate)
vad = load_silero_vad()
recording = False
buffer = []
last_speech_time = None

pipeline = KPipeline(lang_code='a')

latest_frame = None

def capture_snapshot(filename="snapshot.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(filename, frame)
        return filename
    return None


def speak_kokoro(text):
    sd.stop()
    generator = pipeline(
        text,
        voice=kokoro_voice,
        speed=1.3,
        split_pattern=None
    )
    for _, _, audio in generator:
        sd.play(audio, samplerate=24000, blocking=True)

def callback(indata, frames, time_info, status):
    q.put(bytes(indata))

chat_history = [
    {
        "role": "system",
        "content": (
            "You are ADA — pronounced AY DAH — a helpful AI assistant. Speak like a human."
        )
    }
]

def query_ollama(prompt):
    user_prompt = "/no_think " + prompt
    chat_history.append({"role": "user", "content": user_prompt})

    image_path = capture_snapshot()
    if not image_path:
        return "[Vision] No webcam frame available."

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [encoded_image]
                }
            ],
            "stream": False,
            "options": {
                "temperature": ollama_temperature
            },
            # "tools": [
            #     tool_get_current_time,
            #     tool_toggle_lights
            # ]
        }
    )
    if not response.ok:
        return f"[Ollama Error] {response.status_code}: {response.text}"
    response_json = response.json()
    content = response_json.get("message", {}).get("content", "")
    tool_calls = response_json.get("message", {}).get("tool_calls", [])
    print(tool_calls)

    for tool_call in tool_calls:
        if tool_call["function"]["name"] == "get_current_time":
            timezone = tool_call["function"]["arguments"].get("timezone", "UTC")
            current_time = get_current_time(timezone)
            content += f"\nThe current time in {timezone} is {current_time}"
        elif tool_call["function"]["name"] == "toggle_lights":
            state = tool_call["function"]["arguments"].get("state", "off")
            content += f"\nLights turned {state}."

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    print(content)
    chat_history.append({"role": "assistant", "content": content})
    return content

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
