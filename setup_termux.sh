#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Phone as GPU: Complete Termux AI Hub Setup
# Builds llama.cpp with Hardware Acceleration (Vulkan Mobile GPU / ARM DotProd),
# sets up model catalog, starts Web UI + OpenAI API Server
# ==============================================================================

set -e

# Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================${NC}"
echo -e "${GREEN}${BOLD}   📱 Phone as GPU: Local AI Hub Automated Setup    ${NC}"
echo -e "${CYAN}====================================================${NC}"

# 0. Auto-Bootstrap: If executed outside repo (e.g. via curl), clone the repo automatically
REPO_DIR="$HOME/Android-GPU"
if [ ! -f "app.py" ] || [ ! -d "web" ]; then
    echo -e "${YELLOW}[*] Preparing environment and cloning Android-GPU repository...${NC}"
    if ! command -v git >/dev/null 2>&1; then
        pkg update -y 2>/dev/null || true
        pkg install -y git
    fi
    if [ ! -d "$REPO_DIR" ]; then
        git clone https://github.com/AbuZar-Ansarii/Android-GPU.git "$REPO_DIR"
    else
        cd "$REPO_DIR" && git pull origin main || true
    fi
    cd "$REPO_DIR"
    chmod +x setup_termux.sh
    exec bash setup_termux.sh </dev/tty
fi

# 1. Acquire Wake-Lock to keep phone CPU active in background
if command -v termux-wake-lock >/dev/null 2>&1; then
    echo -e "${BLUE}[*] Acquiring CPU wake-lock (prevents sleeping)...${NC}"
    termux-wake-lock
fi

# Request storage permission if not already granted
if [ ! -d "$HOME/storage" ]; then
    echo -e "${BLUE}[*] Setting up phone storage access...${NC}"
    termux-setup-storage || true
fi

# 2. Check if llama-server is already compiled
INSTALL_DIR="$HOME/llama.cpp"
BUILD_DIR="$INSTALL_DIR/build"
MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR"

SERVER_BIN=""
if [ -f "$BUILD_DIR/bin/llama-server" ]; then
    SERVER_BIN="$BUILD_DIR/bin/llama-server"
elif [ -f "$BUILD_DIR/bin/server" ]; then
    SERVER_BIN="$BUILD_DIR/bin/server"
fi

if [ -n "$SERVER_BIN" ]; then
    echo -e "${GREEN}[✓] llama-server is already compiled!${NC}"
    read -r -p "Do you want to recompile with maximum speed/Vulkan acceleration? (y/N): " REBUILD </dev/tty || REBUILD="N"
    if [[ ! "$REBUILD" =~ ^[Yy]$ ]]; then
        SKIP_BUILD=1
    fi
fi

if [ "$SKIP_BUILD" != "1" ]; then
    echo -e "\n${YELLOW}[1/4] Installing necessary packages, OpenMP, and GPU build tools...${NC}"
    pkg update -y
    pkg install -y git cmake clang make termux-tools python iproute2 curl openmp
    # Install Vulkan libraries for mobile GPU acceleration
    pkg install -y vulkan-headers vulkan-loader-android 2>/dev/null || true

    echo -e "\n${YELLOW}[2/4] Cloning / Updating llama.cpp repository...${NC}"
    if [ ! -d "$INSTALL_DIR" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git "$INSTALL_DIR"
    else
        echo "Updating existing llama.cpp repository..."
        cd "$INSTALL_DIR"
        git pull origin master || true
    fi

    cd "$INSTALL_DIR"

    # Detect hardware capabilities
    echo -e "\n${YELLOW}[3/4] Probing ARM CPU Vector & Hardware Capabilities...${NC}"
    CPU_INFO=$(cat /proc/cpuinfo 2>/dev/null || true)
    ARCH_FLAGS=""
    if echo "$CPU_INFO" | grep -q -i -E "asimddp|dotprod"; then
        echo -e "${GREEN}[✓] Hardware ARM Dot Product (sdot/udot) detected! Turbo 4-bit GEMM enabled.${NC}"
        ARCH_FLAGS="-DCMAKE_C_FLAGS=\"-O3 -march=armv8.2-a+fp16+dotprod\" -DCMAKE_CXX_FLAGS=\"-O3 -march=armv8.2-a+fp16+dotprod\""
    elif echo "$CPU_INFO" | grep -q -i -E "fp16|half"; then
        echo -e "${GREEN}[✓] Hardware ARM FP16 vector instructions detected.${NC}"
        ARCH_FLAGS="-DCMAKE_C_FLAGS=\"-O3 -march=armv8.2-a+fp16\" -DCMAKE_CXX_FLAGS=\"-O3 -march=armv8.2-a+fp16\""
    fi

    CORES=$(nproc 2>/dev/null || echo 4)
    echo -e "${BLUE}Detected $CORES CPU cores for compilation.${NC}"

    # Acceleration selection
    echo -e "\n${MAGENTA}====================================================${NC}"
    echo -e "${GREEN}${BOLD}   ⚡ SELECT ACCELERATION BACKEND                  ${NC}"
    echo -e "${MAGENTA}====================================================${NC}"
    echo -e " 1) ${CYAN}Vulkan Mobile GPU + ARM NEON${NC}  (Fastest! Uses Adreno/Mali GPU + CPU)"
    echo -e " 2) ${CYAN}Turbo ARM CPU (NEON+DotProd)${NC}  (Pure CPU with hardware vector math)"
    read -r -p "Enter choice [1-2] (Default: 1): " ACCEL_CHOICE </dev/tty || ACCEL_CHOICE="1"
    ACCEL_CHOICE=${ACCEL_CHOICE:-1}

    echo -e "\n${YELLOW}[4/4] Configuring & Compiling llama-server (this may take a few minutes)...${NC}"
    rm -rf build
    mkdir -p build

    BUILD_SUCCESS=0
    if [ "$ACCEL_CHOICE" -eq 1 ]; then
        echo -e "${BLUE}[*] Attempting Vulkan GPU + OpenMP build...${NC}"
        if cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release $ARCH_FLAGS 2>/dev/null; then
            if cmake --build build --target llama-server rpc-server -j"$CORES" 2>/dev/null || cmake --build build -j"$CORES" 2>/dev/null; then
                if [ -f "$BUILD_DIR/bin/llama-server" ] || [ -f "$BUILD_DIR/bin/server" ]; then
                    echo -e "${GREEN}[✓] Compiled successfully with Vulkan Mobile GPU acceleration!${NC}"
                    BUILD_SUCCESS=1
                fi
            fi
        fi

        if [ "$BUILD_SUCCESS" -ne 1 ]; then
            echo -e "${YELLOW}[!] Vulkan GPU build unavailable on this device ROM. Falling back to Turbo ARM CPU build...${NC}"
            rm -rf build && mkdir -p build
        fi
    fi

    if [ "$BUILD_SUCCESS" -ne 1 ]; then
        echo -e "${BLUE}[*] Building Turbo ARM CPU (NEON + Hardware Vector + OpenMP)...${NC}"
        cmake -B build -DGGML_RPC=ON -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release $ARCH_FLAGS
        cmake --build build --target llama-server rpc-server -j"$CORES" 2>/dev/null || cmake --build build -j"$CORES"
    fi

    if [ -f "$BUILD_DIR/bin/llama-server" ]; then
        SERVER_BIN="$BUILD_DIR/bin/llama-server"
    elif [ -f "$BUILD_DIR/bin/server" ]; then
        SERVER_BIN="$BUILD_DIR/bin/server"
    else
        echo -e "${RED}[ERROR] Compilation completed but llama-server was not found!${NC}"
        exit 1
    fi

    echo -e "${GREEN}[✓] llama-server successfully compiled and ready for high-speed inference!${NC}"
fi

# 3. Interactive Model Selection by Phone RAM Tier
EXISTING_MODELS=$(ls "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l)
if [ "$EXISTING_MODELS" -eq 0 ]; then
    # Detect Phone RAM
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "4000000")
    RAM_GB=$(( (TOTAL_RAM_KB + 500000) / 1024 / 1024 ))
    
    echo -e "\n${MAGENTA}====================================================${NC}"
    echo -e "${GREEN}${BOLD}   🧠 SELECT A MODEL MATCHED TO YOUR PHONE RAM     ${NC}"
    echo -e "${MAGENTA}====================================================${NC}"
    echo -e "Detected Phone RAM: ${YELLOW}${BOLD}${RAM_GB} GB${NC}"
    if [ "$RAM_GB" -le 4 ]; then
        echo -e "${CYAN}👉 Recommended tier for your phone: TIER 1 (4GB RAM) [Options 1-6]${NC}"
    elif [ "$RAM_GB" -le 6 ]; then
        echo -e "${CYAN}👉 Recommended tier for your phone: TIER 2 (6GB RAM) [Options 7-9]${NC}"
    else
        echo -e "${CYAN}👉 Recommended tier for your phone: TIER 2 or TIER 3 (8GB+ RAM) [Options 7-12]${NC}"
    fi
    echo -e "${MAGENTA}----------------------------------------------------${NC}"

    echo -e "${GREEN}${BOLD}🟢 TIER 1: FOR 4 GB RAM PHONES (Ultra-Fast & Compact)${NC}"
    echo -e " 1) ${CYAN}SmolLM2 135M Instruct (Q4_0)${NC}   (~80 MB)  - [FASTEST] Instant token generation"
    echo -e " 2) ${CYAN}SmolLM2 360M Instruct (Q4_K_M)${NC} (~240 MB) - Very small & structured"
    echo -e " 3) ${CYAN}Qwen 2.5 0.5B Instruct (Q4_K_M)${NC}(~390 MB) - Compact multilingual powerhouse"
    echo -e " 4) ${CYAN}Llama 3.2 1B Instruct (Q4_0)${NC}   (~700 MB) - [FASTEST 1B] Meta lightweight model"
    echo -e " 5) ${CYAN}Llama 3.2 1B Instruct (Q4_K_M)${NC} (~800 MB) - Meta high-accuracy 1B quant"
    echo -e " 6) ${CYAN}Qwen 2.5 1.5B Instruct (Q4_K_M)${NC}(~1.0 GB) - Great logic & multi-language"

    echo -e "\n${YELLOW}${BOLD}🟡 TIER 2: FOR 6 GB RAM PHONES (Smart & Balanced)${NC}"
    echo -e " 7) ${CYAN}Google Gemma 2 2B IT (Q4_K_M)${NC}  (~1.6 GB) - High intelligence 2B by Google"
    echo -e " 8) ${CYAN}Llama 3.2 3B Instruct (Q4_K_M)${NC} (~2.0 GB) - Recommended sweet-spot for daily chat"
    echo -e " 9) ${CYAN}Qwen 2.5 3B Instruct (Q4_K_M)${NC}  (~2.1 GB) - Exceptional code & math logic"

    echo -e "\n${RED}${BOLD}🔴 TIER 3: FOR 8 GB+ RAM PHONES (Heavy Reasoning & Coding)${NC}"
    echo -e " 10) ${CYAN}Qwen 2.5 Coder 3B (Q4_K_M)${NC}    (~2.1 GB) - Dedicated Python/JS coding specialist"
    echo -e " 11) ${CYAN}Phi-3.5 mini 3.8B (Q4_K_M)${NC}    (~2.3 GB) - Microsoft's reasoning champion"
    echo -e " 12) ${CYAN}Qwen 2.5 7B (Q3_K_M)${NC}          (~3.8 GB) - Flagship intelligence (high memory)"

    echo -e "\n 13) ${MAGENTA}Skip download${NC} (Use existing models or download later in Web UI)"
    echo -e "${MAGENTA}====================================================${NC}"

    MODEL_CHOICE=""
    while true; do
        read -r -p "Enter model choice [1-13] (or enter 13 to skip download): " MODEL_CHOICE </dev/tty || MODEL_CHOICE="13"
        MODEL_CHOICE=$(echo "$MODEL_CHOICE" | tr -d '[:space:]')
        if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] && [ "$MODEL_CHOICE" -ge 1 ] && [ "$MODEL_CHOICE" -le 13 ]; then
            break
        fi
        echo -e "${RED}Invalid input.${NC} Please enter a number between 1 and 13 (or 13 to skip)."
    done

    MODEL_URL=""
    MODEL_FILE=""

    case "$MODEL_CHOICE" in
        1)
            MODEL_FILE="SmolLM2-135M-Instruct-Q4_0.gguf"
            MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_0.gguf"
            ;;
        2)
            MODEL_FILE="SmolLM2-360M-Instruct-Q4_K_M.gguf"
            MODEL_URL="https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
            ;;
        3)
            MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            ;;
        4)
            MODEL_FILE="Llama-3.2-1B-Instruct-Q4_0.gguf"
            MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf"
            ;;
        5)
            MODEL_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
            MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
            ;;
        6)
            MODEL_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            ;;
        7)
            MODEL_FILE="gemma-2-2b-it-Q4_K_M.gguf"
            MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
            ;;
        8)
            MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            ;;
        9)
            MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
            ;;
        10)
            MODEL_FILE="qwen2.5-coder-3b-instruct-q4_k_m.gguf"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
            ;;
        11)
            MODEL_FILE="Phi-3.5-mini-instruct-Q4_K_M.gguf"
            MODEL_URL="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
            ;;
        12)
            MODEL_FILE="qwen2.5-7b-instruct-q3_k_m.gguf"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"
            ;;
        *)
            echo -e "${YELLOW}[*] Skipping download. You can load or download models from the Web UI.${NC}"
            ;;
    esac

    if [ -n "$MODEL_URL" ]; then
        echo -e "\n${BLUE}[*] Downloading $MODEL_FILE...${NC}"
        curl -L -# -o "$MODELS_DIR/$MODEL_FILE" "$MODEL_URL"
        echo -e "${GREEN}[✓] Download complete: $MODELS_DIR/$MODEL_FILE${NC}"
    fi
fi

# 4. Start the Web UI & API Server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHONE_IP=$(ip -4 addr show wlan0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "127.0.0.1")

echo -e "\n${CYAN}====================================================${NC}"
echo -e "${GREEN}   🚀 STARTING ACCELERATED AI HUB (WEB UI + API)    ${NC}"
echo -e "${CYAN}====================================================${NC}"
echo -e "Phone Web UI : ${YELLOW}http://localhost:8080${NC}"
echo -e "PC Wi-Fi     : ${YELLOW}http://$PHONE_IP:8080${NC}"
echo -e "PC USB (ADB) : ${CYAN}adb forward tcp:8080 tcp:8080${NC} -> ${YELLOW}http://localhost:8080${NC}"
echo -e "${CYAN}====================================================${NC}\n"

exec python3 "$SCRIPT_DIR/app.py"
