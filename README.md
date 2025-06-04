# ADA

A Python app combining Silero VAD, Vosk, Ollama, and Kokoro to create a voice-activated, streaming, low-latency AI companion that speaks naturally and listens continuously.

Named after ADA from *The Outer Worlds*.

## Quickstart Guide

### 1. **System Requirements**

- Linux (PulseAudio or PipeWire required)
- Python 3.12+
- Ollama (with supported LLMs downloaded)
- A working microphone
- At least 8GB RAM recommended

### 2. **Install System Dependencies**

```bash
sudo apt update
sudo apt install python3 python3-pip espeak-ng libsndfile1
```

### 3. Install Ollama 

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma:2b  # or your preferred model
```

### 4. **Clone and Install ADA**

```bash
git clone https://github.com/yourusername/ada-voice.git
cd ada-voice
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Download a Vosk model

```bash
mkdir -p vosk-model-en-us-0.22
curl -LO https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip -d .
```

### 6. Run ADA

```bash
python main.py
```

## Credits

- [Silero VAD](https://github.com/snakers4/silero-vad) — lightweight, real-time Voice Activity Detection from Snakers4
- [Vosk](https://github.com/alphacep/vosk) — efficient offline speech recognition toolkit
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — expressive, fast neural TTS engine with multilingual support
- [Ollama](https://github.com/ollama/ollama) — local LLM inference engine with easy model management
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - obligatory "Ollama is just a llama.cpp wrapper"


