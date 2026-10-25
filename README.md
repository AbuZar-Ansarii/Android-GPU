# ⚡ Android-GPU: Turn Your Phone into a Local AI Accelerator

<div align="center">

![License](https://img.shields.io/badge/License-MIT-crimson.svg)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Windows%20%7C%20Linux%20%7C%20macOS-292c35.svg)
![Backend](https://img.shields.io/badge/Engine-llama.cpp%20%7C%20GGML-e63946.svg)
![API](https://img.shields.io/badge/API-OpenAI%20Compatible%20v1-informational.svg)

**Offload local LLM compute, neural execution, and RAM allocation directly to your Android smartphone over high-speed USB or Wi-Fi.**

[Features](#-key-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Model Catalog](#-recommended-models) • [API Guide](#-openai-api-integration)

</div>

---

## 📌 Overview

Most budget or older laptops are bottlenecked by low-core CPUs (e.g. dual-core processors) and limited RAM, making local Large Language Model (LLM) execution slow and causing fans to throttle constantly. 

Modern Android smartphones (such as devices powered by Snapdragon or MediaTek Dimensity processors) pack **8-core ARM CPUs, high-speed unified LPDDR4X/LPDDR5 RAM (8GB–16GB), and dedicated NPUs/GPUs**. 

**Android-GPU** turns your smartphone into a dedicated local AI hardware accelerator for your PC. It compiles and runs an optimized `llama.cpp` inference engine on the phone and exposes both an **aesthetic modern Web UI** and a standard **OpenAI-compatible REST API** to your PC over zero-latency USB tunneling or local Wi-Fi.

---

## 🚀 Key Features

* **📱 Pure On-Device Execution:** 100% offline, zero cloud subscriptions, zero telemetry.
* **⚡ High-Speed USB Tunneling (ADB):** Connect via USB cable with `adb forward` for zero-latency, rock-solid token streaming without depending on local network speeds.
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
├── setup_termux.sh          # Automated Termux setup, compiler & model downloader
├── install_to_phone.bat     # Windows 1-click script to push project files via ADB
├── start_pc_bridge.bat      # Windows 1-click USB port forwarding & browser launcher
├── .gitignore               # Model weights and temporary cache exclusion
└── README.md                # Comprehensive documentation
```

---

## ⚡ Quick Start

### Prerequisites
1. **Android Smartphone:** Android 8.0+ with **[Termux](https://f-droid.org/en/packages/com.termux/)** installed from F-Droid (do not use Google Play Store version).
2. **PC:** Windows, macOS, or Linux with `adb` (Android Debug Bridge) installed.

---

### Step 1: Enable USB Debugging on Phone
1. Go to **Settings** $\rightarrow$ **About Phone** and tap **Build Number** 7 times to unlock Developer Options.
2. Go to **Settings** $\rightarrow$ **System** $\rightarrow$ **Developer Options** and enable **USB Debugging**.
3. Connect your phone to your PC with a USB cable and tap **Allow** when prompted for authorization.

---

### Step 2: Transfer Files to Phone

#### On Windows:
Double-click **`install_to_phone.bat`**.

#### On Linux / macOS:
```bash
adb shell mkdir -p /sdcard/Download/PhoneAsGPU
adb push setup_termux.sh /sdcard/Download/PhoneAsGPU/
adb push app.py /sdcard/Download/PhoneAsGPU/
adb push web /sdcard/Download/PhoneAsGPU/
```

---

### Step 3: Run Setup in Termux
Open the **Termux** app on your phone and run:

```bash
termux-setup-storage
cp -r ~/storage/downloads/PhoneAsGPU ~/
cd ~/PhoneAsGPU && chmod +x setup_termux.sh && ./setup_termux.sh
```

1. The script automatically acquires an Android CPU wake-lock (to prevent sleeping), updates packages, and compiles `llama.cpp` using all available phone CPU cores.
2. When prompted, select a starting model to download (e.g. `1` for SmolLM2 135M or `4` for Llama 3.2 1B).
3. The server starts automatically on port `8080`.

---

### Step 4: Connect from PC

#### Option A: High-Speed USB Connection (Recommended)
On Windows, double-click **`start_pc_bridge.bat`**. 

*(Or manually run in terminal: `adb forward tcp:8080 tcp:8080`)*

Then open your PC browser:
```text
http://localhost:8080
```

#### Option B: Wireless Wi-Fi
Ensure your PC and phone are on the same Wi-Fi network. Open in your PC browser:
```text
http://<PHONE_IP>:8080
```
*(Your phone's Wi-Fi IP is printed directly in the Termux terminal output.)*

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
