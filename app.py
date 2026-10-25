#!/usr/bin/env python3
"""
Phone as GPU: Local AI Hub Server
Manages llama-server, serves the Web UI, and proxies OpenAI-compatible requests.
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

download_state = {
    "active": False,
    "model": "",
    "percent": 0.0,
    "speed": ""
}

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
        shutil.which("server")
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None

def stop_llama_server():
    """Stop the running llama-server process to free all RAM."""
    global llama_process, current_model_path, current_model_name
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
            print("[✓] Model unloaded. Memory freed.")

def start_llama_server(model_path):
    """Start llama-server with the specified model."""
    global llama_process, current_model_path, current_model_name

    stop_llama_server()

    binary = find_llama_server_binary()
    if not binary:
        raise RuntimeError("llama-server binary not found! Please run setup_termux.sh to build it.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cores = os.cpu_count() or 4
    # Reserve 1 core for OS/UI responsiveness
    threads = max(1, cores - 1)

    cmd = [
        binary,
        "-m", model_path,
        "-c", "2048",
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT)
    ]

    print(f"[*] Starting llama-server with model: {os.path.basename(model_path)} (threads: {threads})...")
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

    print(f"[✓] Model successfully loaded: {current_model_name}")

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
            self.send_json(200, {
                "loaded": loaded,
                "current_model": model,
                "ip": get_local_ip()
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
                start_llama_server(model_path)
                self.send_json(200, {"status": "ok", "model": os.path.basename(model_path)})
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

        elif path == "/v1/chat/completions":
            # Proxy directly to llama-server
            with state_lock:
                loaded = (llama_process is not None and llama_process.poll() is None)

            if not loaded:
                self.send_json(400, {"error": "No model loaded. Please load a model from the Web UI first."})
                return

            try:
                llama_url = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
                req = urllib.request.Request(
                    llama_url,
                    data=body,
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req) as resp:
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in ("content-type", "access-control-allow-origin"):
                            self.send_header(k, v)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    while True:
                        chunk = resp.read(1024)
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
    print("=" * 60)
    print("       PHONE AS GPU - LOCAL AI HUB RUNNING")
    print("=" * 60)
    print(f"[*] Phone Local Web UI : http://localhost:{PORT}")
    print(f"[*] Wi-Fi Web UI       : http://{ip}:{PORT}")
    print(f"[*] OpenAI API Base    : http://{ip}:{PORT}/v1")
    print(f"[*] USB Port Forward   : adb forward tcp:{PORT} tcp:{PORT}")
    print("=" * 60)

    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()
