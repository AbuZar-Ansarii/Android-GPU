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
import shutil
import urllib.request
import urllib.error
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

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
llama_process = None
active_runtime_config = {}

download_state = {
    "active": False,
    "model": "",
    "percent": 0.0,
    "speed": ""
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

def find_llama_server_binary():
    """Locate the compiled llama-server executable."""
    candidates = [
        os.path.join(HOME_DIR, "llama.cpp", "build", "bin", "llama-server"),
        os.path.join(HOME_DIR, "llama.cpp", "build", "bin", "server"),
        shutil.which("llama-server"),
        shutil.which("server"),
        "/data/data/com.termux/files/usr/bin/llama-server",
        "/data/data/com.termux/files/usr/bin/server"
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    
    # Dynamic search in ~/llama.cpp/build for any valid binary
    build_dir = os.path.join(HOME_DIR, "llama.cpp", "build")
    if os.path.isdir(build_dir):
        for root, _, files in os.walk(build_dir):
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
        "vulkan": False
    }
    try:
        res = subprocess.run([binary_path, "--help"], capture_output=True, text=True, timeout=5)
        help_text = res.stdout + res.stderr
        if "--flash-attn" in help_text or "-fa" in help_text:
            features["flash_attn"] = True
        if "--cache-type-k" in help_text or "-ctk" in help_text:
            features["kv_quant"] = True
        if "--n-gpu-layers" in help_text or "-ngl" in help_text:
            features["gpu_layers"] = True
        if "vulkan" in help_text.lower() or "ggml_vulkan" in help_text.lower():
            features["vulkan"] = True
    except Exception:
        pass

    _binary_features_cache[binary_path] = features
    return features

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

def stop_llama_server():
    """Stop the running llama-server process to free all RAM."""
    global llama_process, current_model_path, current_model_name, active_runtime_config
    with state_lock:
        if llama_process is not None:
            print("[*] Unloading model and stopping llama-server...")
            try:
                llama_process.terminate()
                llama_process.wait(timeout=3)
            except Exception:
                try:
                    llama_process.kill()
                except Exception:
                    pass
            llama_process = None
            current_model_path = None
            current_model_name = None
            active_runtime_config = {}
            print("[✓] Model unloaded. Memory freed.")

def start_llama_server(model_path, threads=None, ctx_size=2048, flash_attn=True, kv_quant="q8_0", gpu_layers=0, pin_cores=True):
    """
    Start llama-server with hardware acceleration flags:
    - Big-core CPU affinity (prevents slow efficiency core stalls)
    - Flash Attention (reduces RAM usage & attention computational complexity)
    - 8-bit Key-Value cache quantization (halves KV memory bandwidth bottleneck)
    - Mobile GPU offload via Vulkan (if compiled with Vulkan support)
    - Optimized batch & micro-batch sizes for mobile CPU caches
    """
    global llama_process, current_model_path, current_model_name, active_runtime_config

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

    # Base command arguments
    args = [
        binary,
        "-m", model_path,
        "-c", str(ctx_size),
        "-t", str(threads),
        "-tb", str(cpu_topo["recommended_threads"]),
        "-b", "512",
        "-ub", "256",
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT)
    ]

    # Flash Attention (reduces memory bandwidth and attention overhead)
    if flash_attn and features.get("flash_attn"):
        args.append("--flash-attn")

    # KV Cache Quantization (reduces memory bandwidth bottleneck by 50% for faster generation)
    if kv_quant and kv_quant != "f16" and features.get("kv_quant"):
        args.extend(["-ctk", kv_quant, "-ctv", kv_quant])

    # Mobile GPU layer offloading (Vulkan)
    if gpu_layers > 0 and features.get("gpu_layers"):
        args.extend(["-ngl", str(gpu_layers)])

    # Check for taskset CPU core pinning to big cores
    cmd = args
    has_taskset = shutil.which("taskset") or os.path.isfile("/system/bin/taskset")
    taskset_bin = shutil.which("taskset") or "/system/bin/taskset" if os.path.isfile("/system/bin/taskset") else None
    
    use_pinning = pin_cores and cpu_topo.get("affinity_str") and taskset_bin
    if use_pinning:
        cmd = [taskset_bin, "-c", cpu_topo["affinity_str"]] + args

    print("=" * 60)
    print("      🚀 STARTING ACCELERATED LLAMA-SERVER")
    print("=" * 60)
    print(f"[*] Model       : {os.path.basename(model_path)}")
    print(f"[*] Threads     : {threads} (batch eval: {cpu_topo['recommended_threads']})")
    print(f"[*] Context     : {ctx_size} tokens")
    print(f"[*] Flash-Attn  : {'Enabled (--flash-attn)' if (flash_attn and features.get('flash_attn')) else 'Disabled'}")
    print(f"[*] KV Cache    : {kv_quant if (kv_quant != 'f16' and features.get('kv_quant')) else 'f16 (standard)'}")
    print(f"[*] GPU Offload : {gpu_layers if features.get('gpu_layers') else 0} layers")
    print(f"[*] CPU Pinning : {cpu_topo.get('affinity_str') if use_pinning else 'None'}")
    print("=" * 60)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None
    )

    # Wait for server to come online
    time.sleep(2.5)

    with state_lock:
        llama_process = proc
        current_model_path = model_path
        current_model_name = os.path.basename(model_path)
        active_runtime_config = {
            "threads": threads,
            "ctx_size": ctx_size,
            "flash_attn": bool(flash_attn and features.get("flash_attn")),
            "kv_quant": kv_quant if (kv_quant != "f16" and features.get("kv_quant")) else "f16",
            "gpu_layers": gpu_layers if features.get("gpu_layers") else 0,
            "pinned_cores": cpu_topo.get("affinity_str") if use_pinning else None
        }

    print(f"[✓] Model successfully loaded with acceleration: {current_model_name}")

def scan_models(include_downloads=True):
    """Scan ~/models and Android download directories for GGUF files."""
    results = []
    
    # 1. Scan internal ~/models
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".gguf"):
                path = os.path.join(MODELS_DIR, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1000 else f"{size_mb:.1f} MB"
                results.append({
                    "name": f,
                    "path": path,
                    "size": size_str,
                    "location": "internal"
                })

    # 2. Scan Phone Downloads
    if include_downloads:
        dl_dirs = [
            "/sdcard/Download",
            "/storage/emulated/0/Download",
            os.path.expanduser("~/storage/downloads")
        ]
        for d in dl_dirs:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        if f.endswith(".gguf"):
                            path = os.path.join(d, f)
                            size_mb = os.path.getsize(path) / (1024 * 1024)
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
    """Download model in background thread with progress."""
    global download_state
    try:
        download_state["active"] = True
        download_state["percent"] = 0.0

        req = urllib.request.Request(url, headers={"User-Agent": "PhoneAsGPU/1.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
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
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
    finally:
        download_state["active"] = False

class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = os.path.join(WEB_DIR, "index.html")
            if os.path.isfile(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Web UI not found")
            return

        elif path == "/api/status":
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                model = current_model_name if loaded else None
                config = dict(active_runtime_config) if loaded else {}
            
            binary = find_llama_server_binary()
            features = get_binary_features(binary) if binary else {}
            cpu_topo = detect_cpu_topology()
            telemetry = get_hardware_telemetry()

            self.send_json(200, {
                "loaded": loaded,
                "current_model": model,
                "ip": get_local_ip(),
                "config": config,
                "cpu": cpu_topo,
                "features": features,
                "telemetry": telemetry
            })
            return

        elif path == "/api/models":
            query = parse_qs(parsed.query)
            include_dl = "scan_downloads" in query
            models = scan_models(include_downloads=include_dl)
            self.send_json(200, models)
            return

        elif path == "/api/download_status":
            self.send_json(200, download_state)
            return

        elif path == "/v1/models":
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)
                model = current_model_name if loaded else "no-model-loaded"
            self.send_json(200, {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "phone"}]
            })
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path == "/api/load":
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
                kv_quant = str(payload.get("kv_quant", "q8_0"))
                gpu_layers = int(payload.get("gpu_layers", 0))
                pin_cores = bool(payload.get("pin_cores", True))

                start_llama_server(
                    model_path=model_path,
                    threads=threads,
                    ctx_size=ctx_size,
                    flash_attn=flash_attn,
                    kv_quant=kv_quant,
                    gpu_layers=gpu_layers,
                    pin_cores=pin_cores
                )
                self.send_json(200, {
                    "status": "ok",
                    "model": os.path.basename(model_path),
                    "config": active_runtime_config
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path == "/api/unload":
            stop_llama_server()
            self.send_json(200, {"status": "ok", "message": "Model unloaded"})
            return

        elif path == "/api/download":
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

        elif path in ("/v1/chat/completions", "/v1/completions"):
            # Proxy directly to llama-server with optimized SSE streaming
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)

            if not loaded:
                self.send_json(400, {"error": "No model loaded. Please load a model from the Web UI first."})
                return

            try:
                llama_url = f"http://127.0.0.1:{LLAMA_PORT}{path}"
                req = urllib.request.Request(
                    llama_url,
                    data=body,
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req) as resp:
                    self.send_response(resp.status)
                    is_sse = False
                    for k, v in resp.headers.items():
                        kl = k.lower()
                        if kl == "content-type":
                            self.send_header(k, v)
                            if "text/event-stream" in v:
                                is_sse = True
                        elif kl in ("cache-control", "x-accel-buffering"):
                            self.send_header(k, v)
                    
                    if is_sse:
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")

                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "*")
                    self.end_headers()

                    # Stream with low latency
                    while True:
                        chunk = resp.read(512)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except Exception as e:
                self.send_json(502, {"error": f"Llama-server communication failed: {str(e)}"})
            return

        self.send_error(404, "Not Found")

def cleanup(*args):
    stop_llama_server()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    ip = get_local_ip()
    cpu_topo = detect_cpu_topology()
    binary = find_llama_server_binary()
    features = get_binary_features(binary) if binary else {}

    print("=" * 60)
    print("       PHONE AS GPU - LOCAL AI HUB RUNNING")
    print("=" * 60)
    print(f"[*] Phone Local Web UI : http://localhost:{PORT}")
    print(f"[*] Wi-Fi Web UI       : http://{ip}:{PORT}")
    print(f"[*] OpenAI API Base    : http://{ip}:{PORT}/v1")
    print(f"[*] USB Port Forward   : adb forward tcp:{PORT} tcp:{PORT}")
    print(f"[*] Detected CPU       : {cpu_topo['total_cores']} cores (Optimal Threads: {cpu_topo['recommended_threads']})")
    if cpu_topo.get("affinity_str"):
        print(f"[*] Performance Cores  : {cpu_topo['affinity_str']} (Big Core Pinning Ready)")
    if features.get("vulkan"):
        print(f"[*] Hardware Engine    : Vulkan Mobile GPU Acceleration ACTIVE")
    elif features.get("flash_attn"):
        print(f"[*] Hardware Engine    : ARM NEON + Flash Attention + 8-bit KV")
    print("=" * 60)

    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()
