from abc import abstractmethod, ABC

class LlmInterface(ABC):
    base_url: str
    model: str
    messages: list = [
        {
            "role": "system",
            "content": (
                "You are ADA — pronounced AY DAH — a helpful AI assistant. Speak like a human."
            )
        }
    ]
