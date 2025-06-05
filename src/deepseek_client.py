import os
from openai import OpenAI

from src.__llm_interface__ import LlmInterface


class DeepseekClient(LlmInterface):
    temperature: float
    api_key: str

    def __init__(self):
        self.base_url = os.getenv("DEEPKSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPKSEEK_MODEL", "deepseek-chat")
        self.temperature = float(os.getenv("DEEPKSEEK_TEMPERATURE", "0.5"))
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completion(self, prompt: str, images: list = []):
        self.messages.append({"role": "user", "content": prompt, "images": images})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False,
        )

        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})

        return content, []
