# 🤖 LitBuddy

LitBuddy is your AI-powered reading companion that allows you to chat with PDFs, ask questions across multiple documents, and get smart answers — all locally and for free.

---

## 🚀 Features

- 🧠 Chat with documents using local LLMs (no OpenAI API needed)
- 📄 Load and query multiple PDFs simultaneously
- 💬 Session-based chat history saving and management
- 🪄 Lightweight and GPU-friendly (runs on consumer GPUs)
- 🎙️ Voice-to-text input support (optional)
- 🧩 Modular and easy to extend

---

## 📦 Tech Stack

- `Python 3.11`
- `Streamlit` for the UI
- `LangChain` for chaining LLM logic
- `CTransformers` for running local models
- `PyYAML` for config management
- `Torch` for model execution
- `Whisper / Librosa` (optional: audio input)

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/litbuddy.git
cd litbuddy

