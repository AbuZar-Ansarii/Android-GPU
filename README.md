# ⚡ Android-GPU: Turn Your Phone into a Local AI Accelerator

<div align="center">

![License](https://img.shields.io/badge/License-MIT-crimson.svg)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Windows%20%7C%20Linux%20%7C%20macOS-292c35.svg)
![Backend](https://img.shields.io/badge/Engine-llama.cpp%20%7C%20GGML-e63946.svg)
![API](https://img.shields.io/badge/API-OpenAI%20Compatible%20v1-informational.svg)

**Offload local LLM compute, neural execution, and RAM allocation directly to your Android smartphone over high-speed USB or Wi-Fi.**

[⚡ 1-Command Setup](#-1-command-phone-setup) • [Features](#-key-features) • [Architecture](#-system-architecture) • [Model Catalog](#-recommended-models) • [API Guide](#-openai-api-integration)

</div>

---

## ⚡ 1-Command Phone Setup

Open **Termux** on your phone, paste this single command, and press Enter:

```bash
curl -sL https://raw.githubusercontent.com/AbuZar-Ansarii/Android-GPU/main/setup_termux.sh | bash
```

> **What this one command does automatically:**
> 1. Clones the Android-GPU repository.
> 2. Acquires CPU wake-lock (keeps the model alive in the background).
> 3. Installs dependencies and compiles `llama.cpp` using all phone CPU cores.
> 4. Gives you an interactive menu to choose & download a model (135M to 3B).
> 5. Launches the Web UI & OpenAI-compatible server at `http://localhost:8080`.

*(Alternative without curl:* `pkg update -y && pkg install -y git && git clone https://github.com/AbuZar-Ansarii/Android-GPU.git ~/Android-GPU && cd ~/Android-GPU && bash setup_termux.sh`*)*

---

## 💻 1-Step PC Connection

Connect your phone to your PC via USB (with **USB Debugging** enabled in Developer Options) and run:

* **On Windows:** Double-click **`start_pc_bridge.bat`** (it forwards the port and opens the browser automatically).
* **On Linux / macOS:** Run:
  ```bash
  adb forward tcp:8080 tcp:8080
  ```
  Then open: **`http://localhost:8080`**

*(Or connect wirelessly over the same Wi-Fi using `http://<PHONE_IP>:8080`)*

---

## 📌 Why Android-GPU?

Most budget or older laptops are bottlenecked by low-core CPUs (e.g. dual-core processors) and limited RAM, making local Large Language Model (LLM) execution slow and causing laptop fans to throttle constantly.

Modern Android smartphones pack **8-core ARM CPUs, high-speed unified LPDDR4X/LPDDR5 RAM (8GB–16GB), and fast UFS storage**. 

**Android-GPU** turns your phone into an external AI accelerator: your PC handles the user interface, while your phone's processor and RAM do 100% of the heavy token generation.

---

## 🚀 Key Features

* **📱 Pure On-Device Execution:** 100% offline, zero cloud subscriptions, zero telemetry.
* **⚡ High-Speed USB Tunneling (ADB):** Connect via USB cable for zero-latency token streaming without depending on local network speeds.
* **🌐 Wireless Local Wi-Fi:** Roam freely and access the model on any browser across your local network.
* **🎨 Industrial Matte & Crimson UI:**
  * Engineered dark matte grey (`#111215`) with high-contrast needle crimson red (`#e63946`) aesthetic.
  * **Light & Dark Mode** toggle button with persistent state memory.
  * **Recent Conversations Sidebar** with persistent browser `localStorage` saving, auto-naming, and purge controls.
  * Real-time token streaming with Markdown formatting and 1-click **Copy Code** blocks.
* **🧠 Dynamic Model & Buffer Manager:**
  * **Free RAM & Unload:** Instantly kill the inference backend to free 100% of phone RAM for normal phone use.
  * **Phone Storage Scanner:** Scans `/sdcard/Download` with 1 click to mount any `.gguf` model downloaded through Chrome or mobile browsers.
  * **Built-in Presets Downloader:** Fetch verified compact models (135M to 3B) straight from Hugging Face with live progress meter.
* **🔌 Universal OpenAI API Compatibility:** Direct drop-in endpoint at `http://localhost:8080/v1` for VS Code extensions (Continue, Roo Code, Cline), Python scripts, LangChain, or Open WebUI.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------+
|                       YOUR COMPUTER                         |
|                                                             |
|  [Browser Web UI]     [VS Code / Continue]     [Python App]  |
|         │                      │                    │       |
|         └──────────────────────┼────────────────────┘       |
|                                │ http://localhost:8080/v1   |
+────────────────────────────────┼────────────────────────────+
                                 │
                   USB Cable (ADB Port Forward)
                     or Local Wi-Fi Network
                                 │
+────────────────────────────────┼────────────────────────────+
|                    ANDROID PHONE (Termux)                   |
|                                │                            |
|             ┌──────────────────┴──────────────────┐         |
|             ▼                                     ▼         |
|      [Local Web UI]                     [Python Hub Server] |
|   (Matte Grey & Crimson)                 (Port 8080, app.py)|
|             │                                     │         |
|             └──────────────────┬──────────────────┘         |
|                                │ Process Management         |
|                                ▼                            |
|                     [llama.cpp Engine Server]               |
|                      - Port 50053 (internal)                |
|                      - 8-Core ARM NEON Threading            |
|                      - Unified LPDDR RAM Buffer             |
+-------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
Android-GPU/
├── web/
│   └── index.html           # Modern responsive Web UI (Dark/Light, Sidebar, Model Manager)
├── app.py                   # Zero-dependency Python manager & OpenAI proxy
├── setup_termux.sh          # Self-bootstrapping setup, compiler & model downloader
├── install_to_phone.bat     # Windows 1-click script to push project files via ADB
├── start_pc_bridge.bat      # Windows 1-click USB port forwarding & browser launcher
├── .gitignore               # Model weights and temporary cache exclusion
└── README.md                # Comprehensive documentation
```

---

## 📊 Recommended Models

Always look for **`Q4_K_M`** (4-bit quantization) GGUF files for the optimal intelligence-to-memory ratio:

| Model | Parameters | Quantization | RAM Usage | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **SmolLM2** | 135M | Q4_K_M | ~95 MB | Ultra-compact, instant execution on any device |
| **SmolLM2** | 360M | Q4_K_M | ~240 MB | Fast structured parsing and reasoning |
| **Qwen 2.5** | 0.5B | Q4_K_M | ~390 MB | Multilingual & basic code generation |
| **Llama 3.2** | 1B | Q4_K_M | ~800 MB | Flagship lightweight general assistant |
| **Qwen 2.5** | 1.5B | Q4_K_M | ~1.0 GB | High reasoning, logic, and multi-language |
| **Llama 3.2** | 3B | Q4_K_M | ~2.0 GB | Flagship-grade reasoning on 6GB+ RAM phones |
| **Qwen 2.5** | 3B | Q4_K_M | ~2.1 GB | Outstanding Python/JS coding and math logic |

---

## 🔌 OpenAI API Integration

You can use your phone as the local AI engine inside your favorite development tools.

### VS Code (Continue / Roo Code / Cline)
Add this to your `config.json`:

```json
{
  "models": [
    {
      "title": "Phone as GPU",
      "provider": "openai",
      "model": "local-phone-model",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-phone-model",
    messages=[
        {"role": "system", "content": "You are an expert AI assistant running on a smartphone."},
        {"role": "user", "content": "Explain how distributed computing works in 2 sentences."}
    ]
)

print(response.choices[0].message.content)
```

---

## 🛠️ Model Management & Storage

* **Download through phone browser:** Open Hugging Face on Chrome on your phone, download any `.gguf` file, then open the Web UI $\rightarrow$ **Models** $\rightarrow$ **Scan Phone Storage** $\rightarrow$ **Load & Execute**.
* **Free RAM anytime:** Click **Free RAM & Unload** in the Web UI to terminate the inference engine and free up 100% of the phone's memory when not in use.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
