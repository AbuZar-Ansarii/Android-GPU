#!/usr/bin/env python3
"""
Phone as GPU: Local AI Hub Server
Manages llama-server with hardware acceleration, serves the Web UI,
and proxies OpenAI-compatible requests.
Zero external dependencies - runs on standard Python 3 in Termux.
"""

import os
import sys
import json
import time
import socket
import signal
import ssl
import shutil
import base64
import urllib.request
import urllib.error
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure UTF-8 output across Termux, Linux, macOS, and Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configuration
PORT = int(os.environ.get("PORT", 8080))
LLAMA_PORT = 50053
HOME_DIR = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME_DIR, "models")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# State
state_lock = threading.Lock()
current_model_path = None
current_model_name = None
current_llama_port = LLAMA_PORT
llama_process = None
active_runtime_config = {}

download_state = {
    "active": False,
    "model": "",
    "percent": 0.0,
    "speed": "",
    "error": None
}

_binary_features_cache = {}

def get_local_ip():
    """Detect local Wi-Fi IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_binary_env(binary_path=None):
    """
    Construct environment with LD_LIBRARY_PATH pointing to any shared libraries
    required by llama-server (e.g., libllama.so, libggml-*.so, OpenCL/Vulkan drivers).
    """
    env = os.environ.copy()
    ld_paths = []
    if binary_path:
        bin_dir = os.path.dirname(os.path.abspath(binary_path))
        for d in [bin_dir, os.path.join(bin_dir, "..", "lib"), os.path.join(bin_dir, "lib")]:
            if os.path.isdir(d) and d not in ld_paths:
                ld_paths.append(d)
    
    # Automatically locate all directories in ~/llama.cpp that contain .so files
    base_llama = os.path.join(HOME_DIR, "llama.cpp")
    if os.path.isdir(base_llama):
        for root, _, files in os.walk(base_llama):
            if any(f.endswith(".so") for f in files) and root not in ld_paths:
                ld_paths.append(root)

    default_dirs = [
        os.path.join(HOME_DIR, "llama.cpp", "build", "bin"),
        os.path.join(HOME_DIR, "llama.cpp", "lib"),
        os.path.join(HOME_DIR, "llama.cpp", "bin"),
        os.path.join(HOME_DIR, "llama.cpp"),
        "/data/data/com.termux/files/usr/lib"
    ]
    for d in default_dirs:
        if os.path.isdir(d) and d not in ld_paths:
            ld_paths.append(d)
            
    cur_ld = env.get("LD_LIBRARY_PATH", "")
    if cur_ld:
        ld_paths.append(cur_ld)
    
    env["LD_LIBRARY_PATH"] = ":".join([d for d in ld_paths if d and os.path.isdir(d)])
    return env

# Setup process-wide LD_LIBRARY_PATH at startup
_env_ld = get_binary_env().get("LD_LIBRARY_PATH")
if _env_ld:
    os.environ["LD_LIBRARY_PATH"] = _env_ld

def find_llama_server_binary():
    """Locate the compiled or pre-installed llama-server executable."""
    candidates = [
        os.path.join(HOME_DIR, "llama.cpp", "build", "bin", "llama-server"),
        os.path.join(HOME_DIR, "llama.cpp", "bin", "llama-server"),
        os.path.join(HOME_DIR, "llama.cpp", "llama-server"),
        os.path.join(HOME_DIR, "llama.cpp", "build", "bin", "server"),
        shutil.which("llama-server"),
        shutil.which("server"),
        "/data/data/com.termux/files/usr/bin/llama-server",
        "/data/data/com.termux/files/usr/bin/server",
        os.path.join(HOME_DIR, "bin", "llama-server"),
        os.path.join(HOME_DIR, "llama-server")
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    
    # Dynamic search across ~/llama.cpp and system directories
    search_dirs = [
        os.path.join(HOME_DIR, "llama.cpp"),
        "/data/data/com.termux/files/usr/bin"
    ]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for root, _, files in os.walk(sdir):
                for f in files:
                    if f in ("llama-server", "server"):
                        fp = os.path.join(root, f)
                        if os.access(fp, os.X_OK):
                            return fp
    return None

def get_binary_features(binary_path):
    """
    Inspect llama-server --help to dynamically discover supported acceleration flags.
    Prevents crash if an older binary does not support specific arguments.
    """
    if not binary_path:
        return {}
    if binary_path in _binary_features_cache:
        return _binary_features_cache[binary_path]

    features = {
        "flash_attn": False,
        "kv_quant": False,
        "gpu_layers": False,
        "vulkan": False,
        "mmproj": False
    }
    try:
        env = get_binary_env(binary_path)
        res = subprocess.run([binary_path, "--help"], capture_output=True, text=True, timeout=5, env=env)
        help_text = res.stdout + res.stderr
        if "--flash-attn" in help_text or "-fa" in help_text:
            if any(term in help_text for term in ["[on|off|auto]", "<on|off|auto>", "set Flash Attention use", "LLAMA_ARG_FLASH_ATTN"]):
                features["flash_attn"] = "value"
            else:
                features["flash_attn"] = "flag"
        if "--cache-type-k" in help_text or "-ctk" in help_text:
            features["kv_quant"] = True
        if "--n-gpu-layers" in help_text or "-ngl" in help_text:
            features["gpu_layers"] = True
        if any(term in help_text.lower() for term in ["vulkan", "ggml_vulkan", "opencl", "qualcomm", "snapdragon", "qnn"]):
            features["vulkan"] = True
        if "--mmproj" in help_text or "-mmp" in help_text:
            features["mmproj"] = True
    except Exception:
        pass

    _binary_features_cache[binary_path] = features
    return features

# Paired multimodal projectors for vision models (auto-downloaded if model is fetched)
PAIRED_PROJECTORS = {
    "lfm3-omni": {
        "filename": "mmproj-LFM3-Omni-2B-F16.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/mmproj-LFM3-Omni-2B-F16.gguf"
    },
    "lfm3-omni-2b": {
        "filename": "mmproj-LFM3-Omni-2B-F16.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/mmproj-LFM3-Omni-2B-F16.gguf"
    }
}

def find_matching_mmproj(model_path):
    """
    Find matching multimodal projector (mmproj-*.gguf) for vision/omni/audio models.
    Searches model directory, internal models directory, and all phone download directories.
    """
    if not model_path:
        return None
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_base = os.path.basename(model_path)
    model_stem = os.path.splitext(model_base)[0].lower()

    search_dirs = [
        model_dir,
        MODELS_DIR,
        "/sdcard/Download",
        "/sdcard/Downloads",
        "/storage/emulated/0/Download",
        "/storage/emulated/0/Downloads",
        os.path.expanduser("~/storage/downloads"),
        os.path.expanduser("~/storage/shared/Download")
    ]
    seen_dirs = set()
    valid_dirs = []
    for d in search_dirs:
        if d and os.path.isdir(d):
            rp = os.path.realpath(d)
            if rp not in seen_dirs:
                seen_dirs.add(rp)
                valid_dirs.append(d)

    # 1. Direct name variations for LFM3 Omni
    if "lfm3" in model_stem and "omni" in model_stem:
        for sdir in valid_dirs:
            for fname in [
                "mmproj-LFM3-Omni-2B-F16.gguf",
                "mmproj-LFM3-Omni-2B-f16.gguf",
                "mmproj-LFM3-Omni-2B-Q4_K_M.gguf",
                "mmproj-LFM3-Omni-2B.gguf"
            ]:
                p = os.path.join(sdir, fname)
                if os.path.isfile(p):
                    return p

    # 2. Direct candidate: mmproj-<model_filename>
    for sdir in valid_dirs:
        for c in [
            os.path.join(sdir, f"mmproj-{model_base}"),
            os.path.join(sdir, model_base.replace("LFM3-Omni-2B-", "mmproj-LFM3-Omni-2B-")),
            os.path.join(sdir, "mmproj-model-f16.gguf"),
            os.path.join(sdir, "mmproj-model-q4_0.gguf")
        ]:
            if os.path.isfile(c):
                return c

    # 3. Check all files in valid directories starting with mmproj
    for sdir in valid_dirs:
        try:
            for f in os.listdir(sdir):
                fl = f.lower()
                if fl.startswith("mmproj") and fl.endswith(".gguf"):
                    # Match family keywords
                    if any(kw in model_stem and kw in fl for kw in ["omni", "lfm", "llava", "minicpm", "vision", "audio", "moondream", "smolvlm", "qwen"]):
                        return os.path.join(sdir, f)
                    if any(kw in model_stem for kw in ["omni", "vision", "vl"]):
                        return os.path.join(sdir, f)
        except Exception:
            pass
    return None

def resolve_image_url_to_base64(url):
    """
    Resolve remote HTTP/HTTPS image URL or raw base64 string to a standardized data URI.
    Ensures llama-server on Android/Termux processes images reliably without network lookup failures.
    """
    if not url or not isinstance(url, str):
        return url
    url = url.strip()
    if url.startswith("data:"):
        return url
    # If raw base64 string without data: prefix
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"data:image/jpeg;base64,{url}"
    # Remote HTTP/HTTPS URL: pre-fetch to base64
    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "PhoneAsGPU/2.0 (Multimodal Agent)"})
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            if not content_type.startswith("image/"):
                content_type = "image/jpeg"
            img_bytes = resp.read()
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:{content_type};base64,{b64_str}"
    except Exception as e:
        print(f"[!] Warning: Could not pre-fetch remote image URL {url}: {e}")
        return url

def extract_clean_json(text):
    """
    Sanitize AI response to extract pure parseable JSON.
    Essential for autonomous phone/screen control agents where json.loads() fails
    if the model wrapped JSON in markdown code fences (```json ... ```) or conversational intro.
    """
    if not text or not isinstance(text, str):
        return text
    text_stripped = text.strip()

    # 1. Strip markdown code fences ```json ... ``` or ``` ... ```
    if text_stripped.startswith("```"):
        lines = text_stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text_stripped = "\n".join(lines).strip()

    # 2. Check if text_stripped is directly valid JSON
    try:
        json.loads(text_stripped)
        return text_stripped
    except Exception:
        pass

    # 3. Search for outermost JSON object { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 4. Search for outermost JSON array [ ... ]
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = text[first_bracket:last_bracket + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return text_stripped

def normalize_multimodal_payload(payload):
    """
    Scan payload for vision / image inputs from:
    1. Standard OpenAI format: content=[{"type": "image_url", "image_url": {"url": "..."}}]
    2. Ollama message format: msg["images"] = ["<base64>", ...] or msg["image"] = "<base64>"
    3. Ollama root format: payload["images"] = ["<base64>", ...] or payload["image"] = "<base64>"
    
    Converts raw base64 and remote URLs to standardized data:image/...;base64 URIs
    and ensures llama-server receives OpenAI-compliant image_url objects.
    Returns (modified_payload, has_images).
    """
    if not isinstance(payload, dict):
        return payload, False

    has_images = False

    # Check for root-level images (Ollama /api/generate or /api/chat)
    root_images = []
    if "images" in payload and isinstance(payload["images"], list):
        root_images.extend(payload["images"])
    elif "image" in payload and isinstance(payload["image"], str):
        root_images.append(payload["image"])

    # If payload is /api/generate style (has 'prompt' but no 'messages')
    if "prompt" in payload and "messages" not in payload:
        prompt_text = payload.get("prompt", "")
        payload["messages"] = [{"role": "user", "content": prompt_text}]

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        # Check message-level images (Ollama style)
        msg_images = []
        if "images" in msg and isinstance(msg["images"], list):
            msg_images.extend(msg["images"])
            del msg["images"]
        elif "image" in msg and isinstance(msg["image"], str):
            msg_images.append(msg["image"])
            del msg["image"]

        # If this is the last user message, attach any root-level images
        if root_images and msg.get("role") == "user" and idx == len(messages) - 1:
            msg_images.extend(root_images)
            root_images = []

        content = msg.get("content")

        # Normalize content to list of parts if images are present
        if msg_images:
            has_images = True
            parts = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                parts.extend(content)

            for img_item in msg_images:
                if not isinstance(img_item, str) or not img_item.strip():
                    continue
                url = resolve_image_url_to_base64(img_item)
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            msg["content"] = parts
        elif isinstance(content, list):
            # OpenAI format: inspect existing parts
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_images = True
                    img_data = part.get("image_url")
                    url = ""
                    if isinstance(img_data, dict) and "url" in img_data:
                        url = img_data["url"]
                    elif isinstance(img_data, str):
                        url = img_data
                    if url:
                        url = resolve_image_url_to_base64(url)
                    new_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
                else:
                    new_parts.append(part)
            msg["content"] = new_parts

    return payload, has_images

def detect_cpu_topology():
    """
    Detect ARM CPU topology on Android/Linux.
    Modern 8-core mobile chips (Snapdragon, Dimensity, Exynos) use big.LITTLE architectures
    (e.g., 4 little cores @ 1.8GHz, 3 big cores @ 2.4GHz, 1 prime core @ 2.8GHz).
    
    If all 7 or 8 cores are used, matrix multiplication stalls waiting for the slow 1.8GHz efficiency cores!
    Running on only the big/performance cores (usually 4 to 6 threads, pinned via taskset)
    is up to 40% faster in real-world token generation.
    """
    total_cores = os.cpu_count() or 4
    big_cores = []

    # Attempt to read sysfs CPU max frequencies
    try:
        freqs = []
        for i in range(total_cores):
            freq_file = f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq"
            if not os.path.exists(freq_file):
                freq_file = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq"
            if os.path.exists(freq_file):
                with open(freq_file, "r") as f:
                    freqs.append((i, int(f.read().strip())))
        
        if freqs and len(freqs) == total_cores:
            max_freq = max(f[1] for f in freqs)
            min_freq = min(f[1] for f in freqs)
            # If there's a difference between big and little cores (> 15% clock difference)
            if max_freq > min_freq * 1.15:
                # Big cores are those significantly faster than min_freq
                threshold = min_freq + (max_freq - min_freq) * 0.35
                big_cores = [f[0] for f in freqs if f[1] >= threshold]
    except Exception:
        pass

    # Fallback heuristics for mobile ARM chips if sysfs is restricted
    if not big_cores:
        if total_cores >= 8:
            # 8-core Android chips typically have 4 big cores (cores 4, 5, 6, 7)
            big_cores = list(range(4, total_cores))
            recommended_threads = 4
        elif total_cores >= 6:
            big_cores = list(range(2, total_cores))
            recommended_threads = 4
        else:
            big_cores = list(range(total_cores))
            recommended_threads = max(1, total_cores - 1)
    else:
        recommended_threads = len(big_cores)

    # Sanity checks
    recommended_threads = max(1, min(recommended_threads, total_cores))
    affinity_str = ",".join(str(c) for c in big_cores) if big_cores else None

    return {
        "total_cores": total_cores,
        "big_cores": big_cores,
        "recommended_threads": recommended_threads,
        "affinity_str": affinity_str
    }

def get_hardware_telemetry():
    """Read phone battery %, charging state, and thermal temperature."""
    telemetry = {
        "battery_level": None,
        "battery_temp": None,
        "is_charging": None
    }
    try:
        cap_file = "/sys/class/power_supply/battery/capacity"
        if os.path.exists(cap_file):
            with open(cap_file, "r") as f:
                telemetry["battery_level"] = int(f.read().strip())
        
        temp_file = "/sys/class/power_supply/battery/temp"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                raw_temp = int(f.read().strip())
                # Temp is usually reported in tenths of a degree (e.g. 325 = 32.5°C)
                telemetry["battery_temp"] = round(raw_temp / 10.0, 1) if raw_temp > 100 else float(raw_temp)

        status_file = "/sys/class/power_supply/battery/status"
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                telemetry["is_charging"] = f.read().strip().lower() in ("charging", "full")
    except Exception:
        pass
    return telemetry

def kill_all_llama_processes():
    """
    Forcefully terminate any existing llama-server or server instances,
    including zombie processes from previous crashes, unmanaged process groups,
    or earlier Termux sessions.
    """
    global llama_process
    # 1. Terminate tracked process if alive
    if llama_process is not None:
        try:
            pid = llama_process.pid
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    llama_process.kill()
            else:
                llama_process.kill()
            llama_process.wait(timeout=1.5)
        except Exception:
            pass
        llama_process = None

    # 2. System-level cleanup for untracked / orphaned processes on Termux/Linux
    if sys.platform != "win32":
        for cmd in [
            ["pkill", "-9", "-f", "llama-server"],
            ["pkill", "-9", "-x", "llama-server"],
            ["pkill", "-9", "-f", "server"]
        ]:
            try:
                subprocess.run(cmd, capture_output=True, timeout=1.5)
            except Exception:
                pass
    else:
        try:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, timeout=1.5)
        except Exception:
            pass

    # 3. Allow sockets to leave TIME_WAIT
    time.sleep(0.25)

def find_free_port(start_port=50053, max_attempts=25):
    """
    Find a guaranteed bindable TCP port on localhost, starting with start_port.
    Prevents 'couldn't bind HTTP server socket' crashes when ports are locked.
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    
    # Fallback: Let kernel assign an ephemeral port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except Exception:
        return start_port

def stop_llama_server():
    """Stop the running llama-server process and purge any orphan instances to free all RAM."""
    global llama_process, current_model_path, current_model_name, active_runtime_config
    with state_lock:
        print("[*] Unloading model and stopping llama-server...")
        kill_all_llama_processes()
        llama_process = None
        current_model_path = None
        current_model_name = None
        active_runtime_config = {}
        print("[✓] Model unloaded. Memory freed.")

def _poll_server_health(proc, timeout_sec=35, port=None):
    """
    Poll llama-server /health endpoint until it reports ready or process exits.
    Handles HTTP 503 (loading model) gracefully so slow storage devices don't fail prematurely.
    """
    if port is None:
        port = current_llama_port
    iterations = int(timeout_sec * 2)
    for _ in range(iterations):
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") in ("ok", "ready"):
                        return True
        except urllib.error.HTTPError as he:
            # During model loading, llama-server responds with HTTP 503: {"status": "loading model"}
            # This confirms the HTTP socket bound successfully and the server is actively reading weights!
            try:
                body = json.loads(he.read().decode("utf-8"))
                if body.get("status") in ("ok", "ready"):
                    return True
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(0.5)
    return False

def start_llama_server(model_path, threads=None, ctx_size=2048, flash_attn=True, kv_quant="f16", gpu_layers=0, pin_cores=True, mmproj_path=None):
    """
    Start llama-server with hardware acceleration flags:
    - Big-core CPU affinity (prevents slow efficiency core stalls)
    - Flash Attention (reduces RAM usage & attention computational complexity)
    - 8-bit Key-Value cache quantization (halves KV memory bandwidth bottleneck)
    - Mobile GPU offload via Vulkan (if compiled with Vulkan support)
    - Multimodal projector support (for LFM3 Omni, vision, and audio models)
    - Optimized batch & micro-batch sizes for mobile CPU caches
    """
    global llama_process, current_model_path, current_model_name, current_llama_port, active_runtime_config

    stop_llama_server()

    binary = find_llama_server_binary()
    if not binary:
        raise RuntimeError("llama-server binary not found! Please run setup_termux.sh to build it.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cpu_topo = detect_cpu_topology()
    features = get_binary_features(binary)

    if threads is None or threads <= 0:
        threads = cpu_topo["recommended_threads"]

    # Detect multimodal projector (for LFM3 Omni, LLaVA, etc.)
    if not mmproj_path:
        mmproj_path = find_matching_mmproj(model_path)

    # Multimodal vision models require higher context window (minimum 4096 tokens) for image tokens
    if mmproj_path and os.path.isfile(mmproj_path) and ctx_size < 4096:
        ctx_size = 4096

    # Acquire guaranteed free port
    active_port = find_free_port(LLAMA_PORT)
    current_llama_port = active_port

    # Base universal command arguments
    base_args = [
        binary,
        "-m", model_path,
        "-c", str(ctx_size),
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(active_port)
    ]

    # Attach multimodal projector if found and supported
    if mmproj_path and os.path.isfile(mmproj_path) and features.get("mmproj"):
        base_args.extend(["--mmproj", mmproj_path])

    # Build accelerated command
    accel_args = list(base_args)

    # Flash Attention (reduces memory bandwidth and attention overhead)
    fa_style = features.get("flash_attn")
    if fa_style == "value":
        accel_args.extend(["--flash-attn", "on" if flash_attn else "off"])
    elif fa_style == "flag" and flash_attn:
        accel_args.append("--flash-attn")

    # KV Cache Quantization (reduces memory bandwidth bottleneck by 50% for faster generation)
    if kv_quant and kv_quant != "f16" and features.get("kv_quant"):
        accel_args.extend(["-ctk", kv_quant, "-ctv", kv_quant])

    # Mobile GPU layer offloading (Vulkan / OpenCL / Hexagon)
    if gpu_layers > 0 and features.get("gpu_layers"):
        accel_args.extend(["-ngl", str(gpu_layers)])

    print("=" * 60)
    print("      🚀 STARTING ACCELERATED LLAMA-SERVER")
    print("=" * 60)
    print(f"[*] Model       : {os.path.basename(model_path)}")
    if mmproj_path and os.path.isfile(mmproj_path):
        print(f"[*] Multimodal  : {os.path.basename(mmproj_path)} (Omni/Vision Active)")
    print(f"[*] Port        : {active_port}")
    print(f"[*] Threads     : {threads} (Optimal ARM core allocation)")
    print(f"[*] Context     : {ctx_size} tokens")
    print(f"[*] Flash-Attn  : {'Enabled' if (flash_attn and fa_style) else 'Disabled'}")
    print(f"[*] KV Cache    : {kv_quant if (kv_quant != 'f16' and features.get('kv_quant')) else 'f16 (standard)'}")
    print(f"[*] GPU Offload : {gpu_layers if features.get('gpu_layers') else 0} layers")
    print("=" * 60)

    server_log_path = os.path.join(HOME_DIR, "llama_server.log")
    env = get_binary_env(binary)
    if flash_attn:
        env["LLAMA_ARG_FLASH_ATTN"] = "on"

    # Try 1: Launch with requested acceleration flags
    launch_success = False
    proc = None
    used_cmd = accel_args

    try:
        with open(server_log_path, "a") as log_file:
            log_file.write(f"\n--- Launching llama-server at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_file.write(f"Command: {' '.join(accel_args)}\n")
            log_file.flush()

        log_out = open(server_log_path, "a")
        proc = subprocess.Popen(
            accel_args,
            stdout=log_out,
            stderr=log_out,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None
        )
        launch_success = _poll_server_health(proc, timeout_sec=35, port=active_port)
    except Exception as e:
        print(f"[!] Accelerated launch error: {e}")

    # Try 2: If acceleration flags failed, forcefully terminate Try 1 and fall back to universal baseline
    if (not launch_success or (proc and proc.poll() is not None)) and accel_args != base_args:
        print("[!] Accelerated startup failed or timed out. Purging orphan processes and recovering with universal baseline...")
        kill_all_llama_processes()
        
        # Verify fresh bindable port for baseline fallback
        fallback_port = find_free_port(active_port)
        current_llama_port = fallback_port

        base_cmd = list(base_args)
        if "--port" in base_cmd:
            port_idx = base_cmd.index("--port")
            base_cmd[port_idx + 1] = str(fallback_port)

        used_cmd = base_cmd
        try:
            with open(server_log_path, "a") as log_file:
                log_file.write(f"\n--- Fallback to Universal Baseline at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                log_file.write(f"Command: {' '.join(base_cmd)}\n")
                log_file.flush()

            log_out = open(server_log_path, "a")
            proc = subprocess.Popen(
                base_cmd,
                stdout=log_out,
                stderr=log_out,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
            launch_success = _poll_server_health(proc, timeout_sec=35, port=fallback_port)
        except Exception as e:
            print(f"[!] Baseline launch error: {e}")

    if not launch_success or proc is None or proc.poll() is not None:
        err_msg = ""
        if os.path.isfile(server_log_path):
            try:
                with open(server_log_path, "r") as f:
                    lines = f.readlines()
                    err_msg = "".join(lines[-15:]).strip()
            except Exception:
                pass
        stop_llama_server()
        if not err_msg:
            err_msg = f"llama-server exited with code {proc.returncode if proc and proc.returncode is not None else 'timeout'}"
        print(f"[ERROR] Failed to start llama-server:\n{err_msg}")
        raise RuntimeError(f"Engine failed to load model: {err_msg}")

    print(f"[✓] Model successfully loaded and online: {os.path.basename(model_path)}")

    with state_lock:
        llama_process = proc
        current_model_path = model_path
        current_model_name = os.path.basename(model_path)
        is_accel = (used_cmd == accel_args)
        active_runtime_config = {
            "threads": threads,
            "ctx_size": ctx_size,
            "port": current_llama_port,
            "flash_attn": bool(flash_attn and is_accel and fa_style),
            "kv_quant": kv_quant if (kv_quant != "f16" and is_accel and features.get("kv_quant")) else "f16",
            "gpu_layers": gpu_layers if (is_accel and features.get("gpu_layers")) else 0,
            "pinned_cores": cpu_topo.get("affinity_str") if pin_cores else None,
            "mmproj": os.path.basename(mmproj_path) if (mmproj_path and os.path.isfile(mmproj_path)) else None
        }

def scan_models(include_downloads=True):
    """Scan ~/models and Android download directories for GGUF files."""
    results = []
    seen_real_paths = set()
    
    # 1. Scan internal ~/models
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.lower().endswith(".gguf"):
                path = os.path.join(MODELS_DIR, f)
                try:
                    real = os.path.realpath(path)
                    if real in seen_real_paths:
                        continue
                    size_bytes = os.path.getsize(path)
                    # Filter out empty or incomplete files (< 1MB)
                    if size_bytes < 1024 * 1024:
                        continue
                    seen_real_paths.add(real)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1000 else f"{size_mb:.1f} MB"
                    results.append({
                        "name": f,
                        "path": path,
                        "size": size_str,
                        "location": "internal"
                    })
                except Exception:
                    pass

    # 2. Scan Phone Downloads & Storage (deduplicating symlinked paths)
    if include_downloads:
        dl_dirs = [
            "/sdcard/Download",
            "/sdcard/Downloads",
            "/storage/emulated/0/Download",
            "/storage/emulated/0/Downloads",
            os.path.expanduser("~/storage/downloads"),
            os.path.expanduser("~/storage/shared/Download")
        ]
        for d in dl_dirs:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        if f.lower().endswith(".gguf"):
                            path = os.path.join(d, f)
                            real = os.path.realpath(path)
                            if real in seen_real_paths:
                                continue
                            size_bytes = os.path.getsize(path)
                            if size_bytes < 1024 * 1024:
                                continue
                            seen_real_paths.add(real)
                            size_mb = size_bytes / (1024 * 1024)
                            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1000 else f"{size_mb:.1f} MB"
                            results.append({
                                "name": f,
                                "path": path,
                                "size": size_str,
                                "location": "downloads"
                            })
                except Exception:
                    pass

    return results

def download_worker(url, dest_path):
    """Download model in background thread with progress and error tracking."""
    global download_state
    try:
        download_state["active"] = True
        download_state["model"] = os.path.basename(dest_path)
        download_state["percent"] = 0.0
        download_state["error"] = None

        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Android; Mobile)"})
        with urllib.request.urlopen(req, context=ctx) as resp, open(dest_path, "wb") as f:
            total_length = resp.headers.get("Content-Length")
            total_size = int(total_length) if total_length else 0
            downloaded = 0
            start_time = time.time()

            while True:
                chunk = resp.read(1024 * 512) # 512 KB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    elapsed = max(0.1, time.time() - start_time)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed
                    download_state["percent"] = pct
                    download_state["speed"] = f"{speed_mb:.1f} MB/s"

        download_state["percent"] = 100.0

        # Check if downloaded model has a paired multimodal projector (e.g. LFM3 Omni)
        base_name_lower = os.path.basename(dest_path).lower()
        paired_mmproj = None
        for pkey, pinfo in PAIRED_PROJECTORS.items():
            if pkey in base_name_lower and not base_name_lower.startswith("mmproj"):
                proj_file = pinfo["filename"]
                proj_url = pinfo["url"]
                proj_dest = os.path.join(MODELS_DIR, proj_file)
                if not os.path.isfile(proj_dest) and not find_matching_mmproj(dest_path):
                    print(f"[*] Auto-fetching paired visual projector: {proj_file}...")
                    download_state["model"] = f"{proj_file} (Vision Projector)"
                    download_state["percent"] = 0.0
                    try:
                        req_p = urllib.request.Request(proj_url, headers={"User-Agent": "Mozilla/5.0 (Android; Mobile)"})
                        with urllib.request.urlopen(req_p, context=ctx) as p_resp, open(proj_dest, "wb") as pf:
                            p_len = p_resp.headers.get("Content-Length")
                            p_total = int(p_len) if p_len else 0
                            p_dl = 0
                            p_start = time.time()
                            while True:
                                chunk = p_resp.read(1024 * 512)
                                if not chunk:
                                    break
                                pf.write(chunk)
                                p_dl += len(chunk)
                                if p_total > 0:
                                    pct = (p_dl / p_total) * 100.0
                                    elapsed = max(0.1, time.time() - p_start)
                                    speed_mb = (p_dl / (1024 * 1024)) / elapsed
                                    download_state["percent"] = pct
                                    download_state["speed"] = f"{speed_mb:.1f} MB/s"
                            download_state["percent"] = 100.0
                            print(f"[✓] Successfully downloaded paired projector: {proj_file}")
                            paired_mmproj = proj_dest
                    except Exception as pe:
                        print(f"[!] Warning: Could not auto-download projector {proj_file}: {pe}")
                elif os.path.isfile(proj_dest):
                    paired_mmproj = proj_dest
                break

        # Automatically load downloaded weights if no model is currently running
        if llama_process is None or llama_process.poll() is not None:
            print(f"[*] Auto-mounting freshly downloaded weights: {os.path.basename(dest_path)}")
            try:
                start_llama_server(dest_path, mmproj_path=paired_mmproj)
            except Exception as e:
                print(f"[!] Auto-mount failed: {e}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        download_state["error"] = str(e)
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
    finally:
        download_state["active"] = False

class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        clean_path = path.rstrip("/") if path != "/" else "/"

        if clean_path in ("/", "/index.html"):
            index_path = os.path.join(WEB_DIR, "index.html")
            if os.path.isfile(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Web UI not found")
            return

        elif clean_path in ("/api/status", "/status"):
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                model = current_model_name if loaded else None
                config = dict(active_runtime_config) if loaded else {}
            
            binary = find_llama_server_binary()
            features = get_binary_features(binary) if binary else {}
            cpu_topo = detect_cpu_topology()
            telemetry = get_hardware_telemetry()
            local_ip = get_local_ip()

            self.send_json(200, {
                "loaded": loaded,
                "current_model": model,
                "ip": local_ip,
                "endpoints": {
                    "local_phone": f"http://127.0.0.1:{PORT}/v1",
                    "local_phone_alt": f"http://localhost:{PORT}/v1",
                    "wifi": f"http://{local_ip}:{PORT}/v1",
                    "pc_usb": f"http://localhost:{PORT}/v1"
                },
                "config": config,
                "cpu": cpu_topo,
                "features": features,
                "telemetry": telemetry
            })
            return

        elif clean_path in ("/api/models", "/models_local"):
            query = parse_qs(parsed.query)
            include_dl = "scan_downloads" in query
            models = scan_models(include_downloads=include_dl)
            self.send_json(200, models)
            return

        elif clean_path == "/api/download_status":
            self.send_json(200, download_state)
            return

        elif clean_path in ("/v1", "/health", "/v1/health"):
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                model = current_model_name if loaded else None
            self.send_json(200, {
                "status": "ok",
                "loaded": loaded,
                "model": model,
                "message": "Phone as GPU OpenAI API is active and ready",
                "endpoints": {
                    "chat": "/v1/chat/completions",
                    "completions": "/v1/completions",
                    "models": "/v1/models"
                }
            })
            return

        elif clean_path in ("/v1/models", "/models"):
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                active_model = current_model_name if loaded else None
            
            scanned = scan_models(include_downloads=True)
            models_data = []
            seen = set()

            if active_model:
                models_data.append({
                    "id": active_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "phone-as-gpu",
                    "active": True
                })
                seen.add(active_model)

            for m in scanned:
                if m["name"] not in seen:
                    models_data.append({
                        "id": m["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "phone-as-gpu",
                        "active": False
                    })
                    seen.add(m["name"])

            if not models_data:
                models_data.append({
                    "id": "default",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "phone-as-gpu",
                    "active": False
                })

            self.send_json(200, {
                "object": "list",
                "data": models_data
            })
            return

        elif clean_path.startswith("/v1/models/") or clean_path.startswith("/models/"):
            mid = clean_path.split("/")[-1]
            self.send_json(200, {
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "phone-as-gpu"
            })
            return

        elif clean_path in ("/api/tags", "/tags"):
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                active_model = current_model_name if loaded else None

            scanned = scan_models(include_downloads=True)
            models_list = []
            seen = set()

            if active_model:
                clean_name = active_model.replace(".gguf", "")
                models_list.append({
                    "name": f"{clean_name}:latest",
                    "model": f"{clean_name}:latest",
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "size": 1024 * 1024 * 1024,
                    "digest": "sha256:phonegpu",
                    "details": {
                        "format": "gguf",
                        "family": "lfm" if "lfm" in clean_name.lower() else "llama",
                        "parameter_size": "2B",
                        "quantization_level": "Q4_K_M"
                    }
                })
                seen.add(active_model)

            for m in scanned:
                m_name = m["name"]
                if m_name in seen:
                    continue
                clean_name = m_name.replace(".gguf", "")
                models_list.append({
                    "name": f"{clean_name}:latest",
                    "model": f"{clean_name}:latest",
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "size": 1024 * 1024 * 1024,
                    "digest": "sha256:phonegpu",
                    "details": {
                        "format": "gguf",
                        "family": "lfm" if "lfm" in clean_name.lower() else "llama",
                        "parameter_size": "2B",
                        "quantization_level": "Q4_K_M"
                    }
                })
                seen.add(m_name)

            self.send_json(200, {"models": models_list})
            return

        elif clean_path in ("/api/version", "/version"):
            self.send_json(200, {"version": "0.5.7"})
            return

        elif clean_path == "/api/show":
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                active_model = current_model_name if loaded else "phone-gpu"
            clean_name = active_model.replace(".gguf", "")
            self.send_json(200, {
                "details": {
                    "format": "gguf",
                    "family": "lfm" if "lfm" in clean_name.lower() else "llama",
                    "parameter_size": "2B",
                    "quantization_level": "Q4_K_M"
                }
            })
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        clean_path = path.rstrip("/") if path != "/" else "/"

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if clean_path == "/api/load":
            model_path = payload.get("path")
            if not model_path:
                self.send_json(400, {"error": "Missing model path"})
                return
            try:
                threads = payload.get("threads")
                if threads is not None:
                    threads = int(threads)
                ctx_size = int(payload.get("ctx_size", 2048))
                flash_attn = bool(payload.get("flash_attn", True))
                kv_quant = str(payload.get("kv_quant", "f16"))
                gpu_layers = int(payload.get("gpu_layers", 0))
                pin_cores = bool(payload.get("pin_cores", True))
                mmproj_path = payload.get("mmproj_path")

                start_llama_server(
                    model_path=model_path,
                    threads=threads,
                    ctx_size=ctx_size,
                    flash_attn=flash_attn,
                    kv_quant=kv_quant,
                    gpu_layers=gpu_layers,
                    pin_cores=pin_cores,
                    mmproj_path=mmproj_path
                )
                self.send_json(200, {
                    "status": "ok",
                    "model": os.path.basename(model_path),
                    "config": active_runtime_config
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif clean_path == "/api/unload":
            stop_llama_server()
            self.send_json(200, {"status": "ok", "message": "Model unloaded"})
            return

        elif clean_path == "/api/download":
            url = payload.get("url")
            filename = payload.get("filename")
            if not url or not filename:
                self.send_json(400, {"error": "Missing url or filename"})
                return

            dest = os.path.join(MODELS_DIR, filename)
            t = threading.Thread(target=download_worker, args=(url, dest), daemon=True)
            t.start()
            self.send_json(200, {"status": "started", "destination": dest})
            return

        elif clean_path in (
            "/v1/chat/completions", "/chat/completions",
            "/v1/completions", "/completions",
            "/v1/embeddings", "/embeddings",
            "/api/chat", "/api/generate"
        ):
            is_ollama_chat = (clean_path == "/api/chat")
            is_ollama_generate = (clean_path == "/api/generate")
            is_stream = bool(payload.get("stream", False))
            wants_json = bool(
                payload.get("format") == "json"
                or (isinstance(payload.get("response_format"), dict) and payload.get("response_format", {}).get("type") == "json_object")
            )

            # Normalize and resolve any multimodal image inputs (Ollama images arrays, raw base64, data URIs, remote URLs)
            payload, has_images = normalize_multimodal_payload(payload)
            if wants_json and "response_format" not in payload:
                payload["response_format"] = {"type": "json_object"}
            body = json.dumps(payload).encode("utf-8")

            # 1. Verify or auto-load model if none currently active
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                active_has_mmproj = bool(active_runtime_config.get("mmproj"))
                active_path = current_model_path

            if not loaded:
                available = scan_models(include_downloads=True)
                if not available:
                    self.send_json(400, {
                        "error": {
                            "message": "No model loaded on phone. Please load or download a model first from the Web UI.",
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    })
                    return
                req_model = payload.get("model") if isinstance(payload, dict) else None
                target_path = None
                if req_model:
                    for m in available:
                        if req_model.lower() in m["name"].lower():
                            target_path = m["path"]
                            break
                if not target_path:
                    target_path = available[0]["path"]

                mm_proj = find_matching_mmproj(target_path) if has_images else None
                try:
                    start_llama_server(target_path, mmproj_path=mm_proj, ctx_size=4096 if has_images else 2048)
                    with state_lock:
                        loaded = (llama_process is not None and llama_process.poll() is None)
                        active_has_mmproj = bool(active_runtime_config.get("mmproj"))
                        active_path = current_model_path
                except Exception as e:
                    print(f"[!] Auto-load error: {e}")

            if not loaded:
                self.send_json(400, {
                    "error": {
                        "message": "No model currently loaded in phone RAM. Please load a model from the Models tab.",
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })
                return

            # 2. If request contains visual media but running model lacks multimodal projector
            if has_images and not active_has_mmproj:
                mm_proj = find_matching_mmproj(active_path)
                if mm_proj:
                    print(f"[*] Multimodal media provided: Mounting visual projector {os.path.basename(mm_proj)}...")
                    try:
                        start_llama_server(active_path, mmproj_path=mm_proj, ctx_size=4096)
                        with state_lock:
                            active_has_mmproj = bool(active_runtime_config.get("mmproj"))
                    except Exception as e:
                        print(f"[!] Projector mount error: {e}")
                else:
                    # Search for any other installed model that has a paired mmproj
                    available = scan_models(include_downloads=True)
                    switched = False
                    for m in available:
                        mp = find_matching_mmproj(m["path"])
                        if mp:
                            print(f"[*] Auto-switching to multimodal model {m['name']} with {os.path.basename(mp)}...")
                            try:
                                start_llama_server(m["path"], mmproj_path=mp, ctx_size=4096)
                                switched = True
                                break
                            except Exception:
                                pass
                    if not switched:
                        model_disp = os.path.basename(active_path) if active_path else "Loaded model"
                        self.send_json(400, {
                            "error": {
                                "message": (
                                    f"Vision Projector Missing: Model '{model_disp}' cannot analyze images without its "
                                    f"multimodal projector (mmproj-LFM3-Omni-2B-F16.gguf). "
                                    f"Please download the projector from the Models tab in the Web UI."
                                ),
                                "type": "missing_projector_error",
                                "code": "mmproj_missing"
                            }
                        })
                        return

            # 3. Determine proxy route for llama-server
            if is_ollama_chat or is_ollama_generate or clean_path in ("/v1/chat/completions", "/chat/completions"):
                proxy_subpath = "/v1/chat/completions"
            elif clean_path in ("/v1/completions", "/completions"):
                proxy_subpath = "/v1/completions"
            elif clean_path in ("/v1/embeddings", "/embeddings"):
                proxy_subpath = "/v1/embeddings"
            else:
                proxy_subpath = clean_path

            try:
                llama_url = f"http://127.0.0.1:{current_llama_port}{proxy_subpath}"
                headers = {"Content-Type": "application/json"}
                auth = self.headers.get("Authorization")
                if auth:
                    headers["Authorization"] = auth

                req = urllib.request.Request(
                    llama_url,
                    data=body,
                    headers=headers
                )

                with urllib.request.urlopen(req) as resp:
                    # Non-streaming responses (most AI agents use stream: false to parse action JSON)
                    if not is_stream:
                        raw_bytes = resp.read()
                        try:
                            resp_json = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                        except Exception:
                            resp_json = {}

                        content_str = ""
                        choices = resp_json.get("choices")
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            msg_node = choices[0].get("message")
                            if msg_node and isinstance(msg_node, dict):
                                content_str = msg_node.get("content", "")
                                # Clean JSON output for agents expecting raw action JSON
                                if wants_json or "{" in content_str or "action" in content_str.lower() or "coordinate" in content_str.lower():
                                    clean_content = extract_clean_json(content_str)
                                    msg_node["content"] = clean_content
                                    content_str = clean_content

                        # If request came from Ollama /api/chat
                        if is_ollama_chat:
                            self.send_json(200, {
                                "model": payload.get("model", current_model_name or "phone-gpu"),
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                                "message": {
                                    "role": "assistant",
                                    "content": content_str
                                },
                                "done": True,
                                "total_duration": 1200000000,
                                "load_duration": 1000000
                            })
                            return

                        # If request came from Ollama /api/generate
                        if is_ollama_generate:
                            self.send_json(200, {
                                "model": payload.get("model", current_model_name or "phone-gpu"),
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                                "response": content_str,
                                "done": True
                            })
                            return

                        # OpenAI standard response
                        self.send_json(200, resp_json)
                        return

                    # Streaming responses
                    self.send_response(resp.status)
                    is_sse = False
                    for k, v in resp.headers.items():
                        kl = k.lower()
                        if kl == "content-type":
                            if is_ollama_chat or is_ollama_generate:
                                self.send_header("Content-Type", "application/x-ndjson")
                            else:
                                self.send_header(k, v)
                            if "text/event-stream" in v:
                                is_sse = True
                        elif kl in ("cache-control", "x-accel-buffering", "content-length"):
                            self.send_header(k, v)

                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, HEAD")
                    self.send_header("Access-Control-Allow-Headers", "*")
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                    self.end_headers()

                    # Stream line-by-line
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        line_str = line.decode("utf-8", errors="replace").strip()
                        if is_ollama_chat or is_ollama_generate:
                            if line_str.startswith("data:"):
                                data_part = line_str[5:].strip()
                                if data_part == "[DONE]":
                                    fin_chunk = {"model": payload.get("model", current_model_name or "phone-gpu"), "done": True}
                                    if is_ollama_chat:
                                        fin_chunk["message"] = {"role": "assistant", "content": ""}
                                    else:
                                        fin_chunk["response"] = ""
                                    self.wfile.write((json.dumps(fin_chunk) + "\n").encode("utf-8"))
                                    self.wfile.flush()
                                else:
                                    try:
                                        c_data = json.loads(data_part)
                                        delta = c_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if delta:
                                            chunk_out = {"model": payload.get("model", current_model_name or "phone-gpu"), "done": False}
                                            if is_ollama_chat:
                                                chunk_out["message"] = {"role": "assistant", "content": delta}
                                            else:
                                                chunk_out["response"] = delta
                                            self.wfile.write((json.dumps(chunk_out) + "\n").encode("utf-8"))
                                            self.wfile.flush()
                                    except Exception:
                                        pass
                        else:
                            self.wfile.write(line)
                            self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError):
                pass
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                print(f"[!] llama-server HTTP {he.code}: {err_body}")
                try:
                    err_json = json.loads(err_body)
                    self.send_json(he.code, err_json)
                except Exception:
                    self.send_json(he.code, {"error": {"message": err_body, "code": he.code}})
            except Exception as e:
                try:
                    self.send_json(502, {"error": f"Llama-server communication failed: {str(e)}"})
                except Exception:
                    pass
            return

        self.send_error(404, "Not Found")

def cleanup(*args):
    stop_llama_server()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Forcefully clean up any stale or orphaned llama-server processes on startup
    kill_all_llama_processes()

    ip = get_local_ip()
    cpu_topo = detect_cpu_topology()
    binary = find_llama_server_binary()
    features = get_binary_features(binary) if binary else {}

    print("=" * 65)
    print("       [*] PHONE AS GPU - LOCAL AI HUB RUNNING")
    print("=" * 65)
    print(f"[*] Phone Web UI (Browser)     : http://localhost:{PORT}")
    print(f"[*] Phone Base URL (Same Phone): http://127.0.0.1:{PORT}/v1  <-- IN-APP API")
    print(f"[*] Phone Base URL (Alt)       : http://localhost:{PORT}/v1")
    print(f"[*] PC Wi-Fi Web UI            : http://{ip}:{PORT}")
    print(f"[*] PC Wi-Fi Base URL          : http://{ip}:{PORT}/v1")
    print(f"[*] PC USB Port Forward        : adb forward tcp:{PORT} tcp:{PORT}")
    print(f"[*] PC USB Base URL            : http://localhost:{PORT}/v1")
    print(f"[*] Detected CPU               : {cpu_topo['total_cores']} cores (Optimal Threads: {cpu_topo['recommended_threads']})")
    if cpu_topo.get("affinity_str"):
        print(f"[*] Performance Cores          : {cpu_topo['affinity_str']} (Big Core Pinning Ready)")
    if features.get("vulkan"):
        print(f"[*] Hardware Engine            : Vulkan Mobile GPU Acceleration ACTIVE")
    elif features.get("flash_attn"):
        print(f"[*] Hardware Engine            : ARM NEON + Flash Attention + 8-bit KV")
    if features.get("mmproj"):
        print(f"[*] Multimodal Support         : Active (LFM3 Omni, Vision & Audio Supported)")
    print("=" * 65)

    # Auto-load existing model at startup if available
    available_models = scan_models(include_downloads=True)
    if available_models and binary:
        preferred = available_models[0]["path"]
        print(f"[*] Auto-loading model on startup: {os.path.basename(preferred)} ...")
        try:
            start_llama_server(preferred)
        except Exception as e:
            print(f"[!] Startup auto-load failed: {e}")

    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()
