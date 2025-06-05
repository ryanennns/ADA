import os

from src.__llm_interface__ import LlmInterface


class OllamaClient(LlmInterface):
    temperature: float

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
        self.model = os.getenv("OLLAMA_MODEL", "llama2")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))

    def chat_completion(self, prompt: str, images: list = []):
        self.messages.append({"role": "user", "content": prompt, "images": images})

        import requests
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature
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

        self.messages.append({"role": "assistant", "content": content})
        return content, tool_calls
