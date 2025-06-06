import base64

import cv2

from src.ollama_client import OllamaClient

client = OllamaClient()
client.provide_history = False


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


while True:
    snapshot = capture_snapshot()
    if snapshot:
        with open(snapshot, "rb") as f:
            snapshot = encoded_image = base64.b64encode(f.read()).decode("utf-8")
            content, tool_calls = client.chat_completion(
                'Describe in one sentence what is found in the image.',
                images=[snapshot]
            )
            print(content)
    else:
        print("Failed to capture snapshot")
