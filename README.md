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
| **Liquid AI LFM3** | 1.7B | QAD-Q4_0 | ~1.00 GB | **[Liquid AI]** Tri-hybrid Conv/Attn with QAD 4-bit distillation |
| **Liquid AI LFM3** | 1.7B | Q4_K_M | ~1.05 GB | **[Liquid AI]** 256K context tri-hybrid edge reasoning model |
| **Qwen 2.5** | 1.5B | Q4_K_M | ~1.0 GB | High reasoning, logic, and multi-language |

---

### 🟡 Tier 2: For Phones with 6 GB RAM
*The ideal sweet-spot for daily chat, long-form writing, and problem-solving without slowing down your phone.*

| Model | Parameters | Quantization | RAM Footprint | Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **Liquid AI LFM3 Omni** | 2B | Q4_K_M | ~1.05 GB | **[Omni-Modal]** End-to-end multimodal (text, vision & audio offload) |
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

You can use your phone as a private, local OpenAI-compatible API endpoint directly on the **same phone** or connected across your PC.

### 📍 Available Base URLs

| Location / Client | Base URL | Network / Requirements |
| :--- | :--- | :--- |
| **📱 On the Same Phone** | `http://127.0.0.1:8080/v1` *(or `http://localhost:8080/v1`)* | 100% offline, zero Wi-Fi required |
| **💻 On PC via USB (ADB)** | `http://localhost:8080/v1` | Run `start_pc_bridge.bat` (zero latency) |
| **🌐 Over Local Wi-Fi** | `http://<PHONE_IP>:8080/v1` | Same Wi-Fi network (displayed in app) |

> **Note on API Keys:** The endpoint is 100% local and open. If your client app (like Chatbox, ChatterUI, or Continue) requires an API key, you can enter any string (e.g. `phone-gpu` or `not-needed`).

### 📱 Using on the Same Phone (Android Apps / Scripts)
You can point mobile AI chat apps (like **Chatbox**, **ChatterUI**, **NextChat**, **LibreChat**, or **Termux Python scripts**) directly to the engine running in the background:
* **API Host / Base URL:** `http://127.0.0.1:8080/v1` (or `http://localhost:8080/v1`)
* **API Key:** `phone-gpu`
* **Models:** Call `GET http://127.0.0.1:8080/v1/models` or pick your loaded `.gguf` model from the dropdown.

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
      "apiKey": "phone-gpu"
    }
  ]
}
```

### Python (OpenAI SDK on Phone or PC)
```python
from openai import OpenAI

# Works in Termux on the phone (127.0.0.1:8080/v1) or on PC (localhost:8080/v1)
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",  # Use http://localhost:8080/v1 on PC
    api_key="phone-gpu"
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

## 👁️ Multimodal Vision & Document Intelligence (Images & PDFs)

Android-GPU includes native multimodal vision support designed for **Liquid AI LFM3 Omni 2B**, LLaVA, and vision-enabled models.

### 🖼️ In the Web UI
* **📎 Attach Any Media:** Click the attachment button, drag & drop files onto the prompt box, or paste directly from your clipboard (`Ctrl+V` / `Cmd+V`).
* **📸 Images:** Supports PNG, JPEG, WEBP, and GIF. Images are automatically optimized and resized client-side to prevent network bottlenecks over mobile loopback.
* **📄 PDFs with Zero Server Dependencies:** Upload multi-page PDF documents. The in-browser engine (PDF.js) automatically renders the pages as high-resolution visual canvases for the vision model to inspect (charts, diagrams, tables, and document layout) while extracting text for 100% precision context injection.
* **🔍 Full-Resolution Lightbox:** Click on any attached thumbnail in the chat thread to inspect it in full high resolution.

---

### 🤖 Autonomous AI Agents (Vision via Base URL)

If you are building an AI agent (with **LangChain**, **AutoGen**, **CrewAI**, **LlamaIndex**, or the **OpenAI SDK**) that needs visual perception (e.g. inspecting web screenshots, analyzing diagrams, reading documents), you can pass images directly to the OpenAI-compatible Base URL:

* **On the Same Phone:** `http://127.0.0.1:8080/v1`
* **On PC over USB Tunnel:** `http://localhost:8080/v1`

#### How the Engine Handles Vision & Ollama Agent Payloads:
1. **Full Ollama API & Format Support:** In addition to OpenAI (`/v1/chat/completions`), Android-GPU natively supports Ollama endpoints (`/api/chat`, `/api/generate`, `/api/tags`, `/api/version`).
2. **Raw Base64 & `"images"` Arrays:** AI agents can send images using Ollama's `"images": ["<raw_base64>"]` array or OpenAI's `{"type": "image_url", ...}` format. Raw base64 strings and remote HTTP/HTTPS URLs are automatically normalized into data URIs.
3. **Automatic Action JSON Cleaning:** Autonomous phone-controlling agents (which take screenshots to decide where to tap and parse the output with `json.loads`) frequently fail if the model wraps output in markdown code blocks (` ```json ... ``` `) or conversational filler. Android-GPU automatically sanitizes the response, extracting pure, valid JSON so the agent's action parser never crashes.
4. **Paired Projector Auto-Download:** Multimodal models like Liquid AI LFM3 Omni 2B require both the model weights and the visual projector (`mmproj-LFM3-Omni-2B-F16.gguf`). The engine automatically downloads and mounts the matching projector.
5. **Dynamic Vision Context Scaling:** When an image is detected, the context buffer automatically scales to 4096+ tokens to accommodate visual projector tokens (~576–1152 tokens per image).

#### Ollama-Style Phone Automation Agent Example (Take Screenshot -> Output Action JSON)
```python
import base64
import requests
import json

# 1. Take phone screenshot and encode to base64
with open("phone_screen.png", "rb") as f:
    screen_base64 = base64.b64encode(f.read()).decode("utf-8")

# 2. Call local endpoint (using Ollama /api/chat or OpenAI /v1/chat/completions)
url = "http://127.0.0.1:8080/api/chat"
payload = {
    "model": "lfm3-omni",
    "messages": [
        {
            "role": "user",
            "content": "Look at this phone screenshot. Where should I tap to open Settings? Respond ONLY with an action JSON.",
            "images": [screen_base64]  # Raw base64 supported just like Ollama!
        }
    ],
    "format": "json",
    "stream": False
}

resp = requests.post(url, json=payload)
data = resp.json()

# 3. Clean JSON guaranteed! No markdown code fences or conversational errors:
action_json = json.loads(data["message"]["content"])
print("Action JSON parsed successfully:", action_json)
# Example output: {"action": "tap", "coordinate": [540, 960]}
```

#### Python AI Agent Example (Multimodal Vision)
```python
import base64
from openai import OpenAI

# 1. Initialize client pointing to your phone
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",  # or http://localhost:8080/v1 on PC
    api_key="phone-gpu"
)

# 2. Encode local image to base64
with open("screenshot.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# 3. Request visual analysis
response = client.chat.completions.create(
    model="LFM3-Omni-2B-Q4_K_M.gguf",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this screenshot? Extract any key metrics and summarize the chart."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                        # Remote URLs like "https://example.com/chart.jpg" are also supported!
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### cURL Agent Example
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer phone-gpu" \
  -d '{
    "model": "LFM3-Omni-2B-Q4_K_M.gguf",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this architecture diagram in detail."},
          {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/AbuZar-Ansarii/Android-GPU/main/web/favicon.png"}}
        ]
      }
    ]
  }'
```

---

## 🛠️ Model Management & Storage

* **Download through phone browser:** Open Hugging Face on Chrome on your phone, download any `.gguf` file, then open the Web UI $\rightarrow$ **Models** $\rightarrow$ **Scan Phone Storage** $\rightarrow$ **Load & Execute**.
* **Free RAM anytime:** Click **Free RAM & Unload** in the Web UI to terminate the inference engine and free up 100% of the phone's memory when not in use.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
