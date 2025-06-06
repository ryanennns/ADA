import os
system_prompt = os.getenv("SYSTEM_PROMPT", "")

class LlmInterface:
    provide_history: bool = True
    base_url: str
    model: str
    messages: list = [
        {
            "role": "system",
            "content": (
                "You are ADA — pronounced AY DAH — a helpful AI assistant."
                "" + system_prompt
            )
        }
    ]
