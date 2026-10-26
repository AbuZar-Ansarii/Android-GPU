# ⚡ Android-GPU: Turn Your Phone into a Local AI Accelerator

<div align="center">

![License](https://img.shields.io/badge/License-MIT-crimson.svg)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Windows%20%7C%20Linux%20%7C%20macOS-292c35.svg)
![Backend](https://img.shields.io/badge/Engine-llama.cpp%20%7C%20GGML-e63946.svg)
![API](https://img.shields.io/badge/API-OpenAI%20Compatible%20v1-informational.svg)

**Offload local LLM compute, neural execution, and RAM allocation directly to your Android smartphone over high-speed USB or Wi-Fi.**

[⚡ 1-Command Setup](#-method-1-1-command-automatic-setup-recommended) • [🛠️ Manual Setup](#-method-2-manual-step-by-step-setup) • [Features](#-key-features) • [Architecture](#-system-architecture) • [Model Catalog](#-recommended-models) • [API Guide](#-openai-api-integration)

</div>

---

## ⚡ Method 1: 1-Command Automatic Setup (Recommended)

### 1. On Your Android Phone
Open the **Termux** app on your phone, paste this single command, and press **Enter**:

```bash
curl -sL https://raw.githubusercontent.com/AbuZar-Ansarii/Android-GPU/main/setup_termux.sh | bash
```
*(or via:* `bash -c "$(curl -fsSL https://raw.githubusercontent.com/AbuZar-Ansarii/Android-GPU/main/setup_termux.sh)"`*)*

> **What this one command handles automatically:**
> 1. Clones the Android-GPU repository.
> 2. Acquires CPU wake-lock (keeps the model running in the background with screen off).
> 3. Installs compilers and builds `llama.cpp` using all phone CPU cores.
> 4. Displays an interactive terminal menu to choose and download a model (135M to 3B).
> 5. Launches the Web UI and OpenAI API server at `http://localhost:8080`.

*(Alternative without curl:* `pkg update -y && pkg install -y git && git clone https://github.com/AbuZar-Ansarii/Android-GPU.git ~/Android-GPU && cd ~/Android-GPU && bash setup_termux.sh`*)*

### 2. Connect from Your PC
Plug your phone into your PC via USB (with **USB Debugging** enabled) and run:
* **On Windows:** Double-click **`start_pc_bridge.bat`** (forwards port 8080 and launches your browser automatically).
* **On Linux / macOS:** Run `adb forward tcp:8080 tcp:8080` and visit **`http://localhost:8080`**.

---

## 🛠️ Method 2: Manual Step-by-Step Setup

If you prefer full control or want to transfer files manually over USB without cloning from the internet:

### Step 1: Phone Preparation
1. Enable **Developer Options** and turn on **USB Debugging**:
   * Open **Settings** $\rightarrow$ **About Phone** $\rightarrow$ tap **Build Number** 7 times.
   * Go to **Settings** $\rightarrow$ **System** $\rightarrow$ **Developer Options** $\rightarrow$ enable **USB Debugging**.
2. Install **Termux**:
   * Download the latest APK from [F-Droid](https://f-droid.org/en/packages/com.termux/) or GitHub Releases (do not use the Google Play Store version).

---

### Step 2: Transfer Project Files to Your Phone
Connect your phone to your PC with a USB cable.

#### On Windows:
Double-click **`install_to_phone.bat`** in this folder. It will automatically push all project files (`setup_termux.sh`, `app.py`, and `web/`) to `/sdcard/Download/PhoneAsGPU/`.

#### On Linux / macOS:
```bash
adb shell mkdir -p /sdcard/Download/PhoneAsGPU
adb push setup_termux.sh /sdcard/Download/PhoneAsGPU/
adb push app.py /sdcard/Download/PhoneAsGPU/
adb push web /sdcard/Download/PhoneAsGPU/
```

---

### Step 3: Run the Setup in Termux
Open **Termux** on your phone and run:

```bash
termux-setup-storage
cp -r ~/storage/downloads/PhoneAsGPU ~/
cd ~/PhoneAsGPU && chmod +x setup_termux.sh && ./setup_termux.sh
```

1. It will install build tools (`clang`, `cmake`, `make`, `python`) and compile `llama.cpp`.
2. It will prompt you with a numbered list to pick a model to download (e.g. `1` for SmolLM2 135M or `4` for Llama 3.2 1B).
3. Once completed, the server starts running on port `8080`.

---

### Step 4: Access on PC & Phone

* **On Your Phone:** Open Chrome or any browser on your phone and visit:
  ```text
  http://localhost:8080
  ```
* **On Your PC over USB (Zero Latency):**
  * Run **`start_pc_bridge.bat`** (or `adb forward tcp:8080 tcp:8080`).
  * Open in your PC browser: **`http://localhost:8080`**.
* **On Your PC over Local Wi-Fi:**
  * Connect your PC and phone to the same Wi-Fi network.
  * Open in your PC browser: **`http://<PHONE_IP>:8080`** *(Phone IP is displayed in the Termux terminal output)*.

---

## 📌 Why Android-GPU?

Most budget or older laptops are bottlenecked by low-core CPUs (e.g. dual-core processors) and limited RAM, making local Large Language Model (LLM) execution slow and causing laptop fans to throttle constantly.

Modern Android smartphones pack **8-core ARM CPUs, high-speed unified LPDDR4X/LPDDR5 RAM (8GB–16GB), and fast UFS storage**. 

**Android-GPU** turns your phone into an external AI accelerator: your PC handles the user interface, while your phone's processor and RAM do 100% of the heavy token generation.

---

## 🚀 Key Features

* **📱 Pure On-Device Execution:** 100% offline, zero cloud subscriptions, zero telemetry.
* **⚡ Hardware Acceleration Engine:**
  * **Vulkan Mobile GPU Offloading:** Offload model layers directly to Snapdragon Adreno or MediaTek Mali mobile GPUs for 2x–3x faster generation.
  * **ARMv8.2-A Dot Product Vector Math:** Leverages hardware `sdot` / `udot` instructions for ultra-fast 4-bit integer matrix multiplication.
  * **Big.LITTLE Core Pinning (`taskset`):** Prevents slow efficiency-core stalls by locking compute threads exclusively to the high-performance CPU cores.
  * **Flash Attention & 8-bit KV Cache:** Reduces memory bandwidth consumption by 50% for high-speed, long-context generation.
  * **Live Performance Metrics:** Real-time tokens/second ($t/s$), elapsed generation time, and Time-To-First-Token (TTFT) benchmarking badges on every assistant response.
  * **On-Device Hardware Telemetry:** Real-time phone battery % and thermal temperature monitoring to keep your device running cool.
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
  * **Built-in Presets Downloader:** Fetch verified compact models (135M to 3B, with Q4_0 and Q4_K_M quants) straight from Hugging Face with live progress meter.
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

## 📊 Recommended Models by Phone RAM Tier

Choose a model matched to your phone's total RAM capacity. Using 4-bit (`Q4_K_M`) quantization leaves plenty of headroom for Android OS background services:

### 🟢 Tier 1: For Phones with 3 GB – 4 GB RAM
*Best for ultra-fast token generation, basic assistant tasks, and minimal memory usage. Note: `Q4_0` quantizations leverage ARM Dot-Product vector instructions directly with zero unpacking overhead for maximum tokens/second.*

| Model | Parameters | Quantization | RAM Footprint | Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **SmolLM2** | 135M | Q4_0 | ~80 MB | **[Fastest]** Direct ARM NEON DotProd execution |
| **SmolLM2** | 135M | Q4_K_M | ~95 MB | Ultra-compact, instant execution on any phone |
| **SmolLM2** | 360M | Q4_K_M | ~240 MB | Structured reasoning & fast generation |
| **Qwen 2.5** | 0.5B | Q4_K_M | ~390 MB | Multilingual & basic code generation |
| **Llama 3.2** | 1B | Q4_0 | ~700 MB | **[Fastest 1B]** Meta lightweight model with fastest SIMD math |
| **Llama 3.2** | 1B | Q4_K_M | ~800 MB | Meta's flagship lightweight general assistant |
| **Qwen 2.5** | 1.5B | Q4_K_M | ~1.0 GB | High reasoning, logic, and multi-language |

---

### 🟡 Tier 2: For Phones with 6 GB RAM
*The ideal sweet-spot for daily chat, long-form writing, and problem-solving without slowing down your phone.*

| Model | Parameters | Quantization | RAM Footprint | Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **Google Gemma 2** | 2B | Q4_K_M | ~1.6 GB | High-intelligence 2B model derived from Gemini |
| **Llama 3.2** | 3B | Q4_K_M | ~2.0 GB | Recommended daily driver for chat & summaries |
| **Qwen 2.5** | 3B | Q4_K_M | ~2.1 GB | Exceptional Python/JS code synthesis & math |

---

### 🔴 Tier 3: For Phones with 8 GB – 12 GB+ RAM
*Heavy software engineering, advanced reasoning, and deep comprehension for flagship smartphones (e.g. Motorola Edge 40, Galaxy S series).*

| Model | Parameters | Quantization | RAM Footprint | Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder** | 3B | Q4_K_M | ~2.1 GB | Specialist dedicated to Python, JavaScript & algorithms |
| **Microsoft Phi-3.5 mini** | 3.8B | Q4_K_M | ~2.3 GB | State-of-the-art math and multi-step reasoning |
| **Qwen 2.5** | 7B | Q3_K_M | ~3.8 GB | Near-flagship intelligence (requires 8GB+ RAM) |

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
