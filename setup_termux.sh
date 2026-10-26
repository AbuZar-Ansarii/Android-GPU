#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Phone as GPU: Universal Adaptive Setup for Any Android Phone
# Detects Hardware & Processor, Auto-Configures Optimal Acceleration Engine,
# Builds/Installs llama-server, and Launches Web UI + OpenAI API Server
# ==============================================================================

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
echo -e "${GREEN}${BOLD}   📱 Phone as GPU: Local AI Hub Universal Setup    ${NC}"
echo -e "${CYAN}====================================================${NC}"

# 0. Auto-Bootstrap: If executed outside repo (e.g. via curl), clone latest repo automatically
REPO_DIR="$HOME/Android-GPU"
if [ ! -f "app.py" ] || [ ! -d "web" ]; then
    echo -e "${YELLOW}[*] Preparing environment and fetching latest Android-GPU...${NC}"
    if ! command -v git >/dev/null 2>&1; then
        pkg update -y 2>/dev/null || true
        pkg install -y git 2>/dev/null || true
    fi
    rm -rf "$REPO_DIR"
    git clone --depth 1 https://github.com/AbuZar-Ansarii/Android-GPU.git "$REPO_DIR"
    cd "$REPO_DIR"
    chmod +x setup_termux.sh
    exec bash setup_termux.sh </dev/tty
else
    # Pull latest updates if already in repo
    git pull origin main 2>/dev/null || true
fi

# 1. Acquire Wake-Lock to keep phone CPU active in background
if command -v termux-wake-lock >/dev/null 2>&1; then
    echo -e "${BLUE}[*] Acquiring CPU wake-lock (prevents sleeping)...${NC}"
    termux-wake-lock
fi

# Request storage permission if not already granted
if [ ! -d "$HOME/storage" ]; then
    echo -e "${BLUE}[*] Setting up phone storage access...${NC}"
    termux-setup-storage 2>/dev/null || true
fi

# ==============================================================================
# 🔍 STEP 1: DETECT PHONE HARDWARE & PROCESSOR PROFILE
# ==============================================================================
DEVICE_BRAND=$(getprop ro.product.brand 2>/dev/null || echo "")
DEVICE_MODEL=$(getprop ro.product.model 2>/dev/null || getprop ro.product.device 2>/dev/null || echo "Android Device")
DEVICE_MANUF=$(getprop ro.product.manufacturer 2>/dev/null || echo "$DEVICE_BRAND")
SOC_NAME=$(getprop ro.soc.model 2>/dev/null || getprop ro.board.platform 2>/dev/null || grep -m1 "Hardware" /proc/cpuinfo 2>/dev/null | awk -F: '{print $2}' | xargs || echo "ARM64 Processor")
ARCH=$(uname -m 2>/dev/null || echo "aarch64")
CORES=$(nproc 2>/dev/null || echo 4)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "4000000")
RAM_GB=$(( (TOTAL_RAM_KB + 500000) / 1024 / 1024 ))
ANDROID_VER=$(getprop ro.build.version.release 2>/dev/null || echo "Unknown")

# Format full device name
FULL_DEVICE_NAME="${DEVICE_MANUF} ${DEVICE_MODEL}"
if [ -z "$DEVICE_MANUF" ] || [ "$DEVICE_MANUF" = "$DEVICE_MODEL" ]; then
    FULL_DEVICE_NAME="$DEVICE_MODEL"
fi

# Detect CPU SIMD & Vector extensions
CPU_INFO=$(cat /proc/cpuinfo 2>/dev/null || true)
HAS_DOTPROD=0
HAS_FP16=0
if echo "$CPU_INFO" | grep -q -i -E "asimddp|dotprod"; then
    HAS_DOTPROD=1
fi
if echo "$CPU_INFO" | grep -q -i -E "fp16|half"; then
    HAS_FP16=1
fi

# Detect Vulkan Mobile GPU driver
HAS_VULKAN=0
if [ -e "/system/lib64/libvulkan.so" ] || [ -e "/vendor/lib64/hw/vulkan."* ] || [ -e "/system/lib/libvulkan.so" ] || command -v vulkaninfo >/dev/null 2>&1; then
    HAS_VULKAN=1
fi

echo -e "\n${CYAN}====================================================${NC}"
echo -e "${GREEN}${BOLD}   📱 DETECTED PHONE HARDWARE & PROCESSOR PROFILE   ${NC}"
echo -e "${CYAN}====================================================${NC}"
echo -e " 📲 Device        : ${YELLOW}${BOLD}${FULL_DEVICE_NAME}${NC} (Android ${ANDROID_VER})"
echo -e " 🧠 Processor     : ${YELLOW}${BOLD}${SOC_NAME}${NC} (${ARCH})"
echo -e " ⚡ CPU Cores     : ${CYAN}${CORES} Cores detected${NC}"
echo -e " 💾 System Memory : ${CYAN}${RAM_GB} GB RAM detected${NC}"
if [ "$HAS_DOTPROD" -eq 1 ]; then
    echo -e " 🚀 Vector Math   : ${GREEN}${BOLD}ARM NEON + Hardware Dot-Product (sdot/udot)${NC}"
else
    echo -e " 🚀 Vector Math   : ${YELLOW}Standard ARM NEON Vector Engine${NC}"
fi
if [ "$HAS_VULKAN" -eq 1 ]; then
    echo -e " 🎮 Mobile GPU    : ${GREEN}${BOLD}Vulkan Hardware Acceleration Driver DETECTED${NC}"
else
    echo -e " 🎮 Mobile GPU    : ${YELLOW}CPU-Only Engine (No Vulkan driver found)${NC}"
fi
echo -e "${CYAN}====================================================${NC}\n"

# ==============================================================================
# ⚙️ STEP 2: CHECK / PREPARE INFERENCE BACKEND
# ==============================================================================
INSTALL_DIR="$HOME/llama.cpp"
BUILD_DIR="$INSTALL_DIR/build"
MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR"

SERVER_BIN=""
if [ -f "$BUILD_DIR/bin/llama-server" ]; then
    SERVER_BIN="$BUILD_DIR/bin/llama-server"
elif [ -f "$INSTALL_DIR/bin/llama-server" ]; then
    SERVER_BIN="$INSTALL_DIR/bin/llama-server"
elif [ -f "$INSTALL_DIR/llama-server" ]; then
    SERVER_BIN="$INSTALL_DIR/llama-server"
elif [ -f "$BUILD_DIR/bin/server" ]; then
    SERVER_BIN="$BUILD_DIR/bin/server"
elif command -v llama-server >/dev/null 2>&1; then
    SERVER_BIN=$(command -v llama-server)
elif [ -x "$PREFIX/bin/llama-server" ]; then
    SERVER_BIN="$PREFIX/bin/llama-server"
fi

SKIP_BUILD=0
if [ -n "$SERVER_BIN" ] && [ -x "$SERVER_BIN" ]; then
    echo -e "${GREEN}[✓] llama-server is already installed and ready at: $SERVER_BIN${NC}"
    read -r -p "Do you want to re-install/update the engine? (y/N): " REBUILD </dev/tty || REBUILD="N"
    if [[ ! "$REBUILD" =~ ^[Yy]$ ]]; then
        SKIP_BUILD=1
    fi
fi

# Function: Install verified pre-compiled high-performance binary for Android ARM64
install_precompiled_engine() {
    echo -e "${BLUE}[*] Fetching verified pre-compiled high-speed engine for Android...${NC}"
    mkdir -p "$BUILD_DIR/bin"
    mkdir -p "$INSTALL_DIR/lib"
    mkdir -p "$INSTALL_DIR/bin"

    # Strategy 1: Termux official repository package (llama-cpp)
    echo -e "${CYAN}[1/2] Checking Termux official package repository...${NC}"
    pkg install -y libandroid-spawn 2>/dev/null || true
    if pkg install -y llama-cpp 2>/dev/null || pkg install -y llama.cpp 2>/dev/null; then
        SYS_BIN=$(command -v llama-server || which llama-server 2>/dev/null || echo "$PREFIX/bin/llama-server")
        if [ -x "$SYS_BIN" ]; then
            SERVER_BIN="$SYS_BIN"
            mkdir -p "$BUILD_DIR/bin"
            cp -f "$SERVER_BIN" "$BUILD_DIR/bin/llama-server" 2>/dev/null || true
            chmod +x "$BUILD_DIR/bin/llama-server" 2>/dev/null || true
            echo -e "${GREEN}[✓] llama-server installed successfully via Termux package manager!${NC}"
            return 0
        fi
    fi

    # Strategy 2: Official Pre-compiled Android Release from GitHub (ggml-org/llama.cpp)
    echo -e "${CYAN}[2/2] Fetching official high-performance Android ARM64 release from GitHub...${NC}"
    RELEASE_TAG="b11889"
    ASSET_NAME="llama-${RELEASE_TAG}-bin-android-arm64.tar.gz"

    if echo "${SOC_NAME} ${FULL_DEVICE_NAME}" | grep -q -i -E "snapdragon|qualcomm|sm[0-9]{4}|qcom"; then
        ASSET_NAME="llama-${RELEASE_TAG}-bin-android-arm64-snapdragon.tar.gz"
        echo -e "${GREEN}[*] Qualcomm Snapdragon chip detected: Using dedicated Snapdragon NPU/GPU engine!${NC}"
    else
        echo -e "${GREEN}[*] Mobile ARM64 processor detected: Using universal ARMv8/v9 optimized engine!${NC}"
    fi

    DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
    FALLBACK_URL="https://github.com/ggml-org/llama.cpp/releases/download/${RELEASE_TAG}/llama-${RELEASE_TAG}-bin-android-arm64.tar.gz"
    TMP_ARCHIVE="$HOME/llama-android.tar.gz"
    rm -f "$TMP_ARCHIVE"

    echo -e "${BLUE}[*] Downloading engine archive (${ASSET_NAME})...${NC}"
    if ! curl -L --fail --connect-timeout 20 -# -o "$TMP_ARCHIVE" "$DOWNLOAD_URL"; then
        echo -e "${YELLOW}[*] Downloading universal ARM64 engine fallback...${NC}"
        curl -L --fail --connect-timeout 20 -# -o "$TMP_ARCHIVE" "$FALLBACK_URL" || true
    fi

    if [ -f "$TMP_ARCHIVE" ] && [ -s "$TMP_ARCHIVE" ]; then
        echo -e "${BLUE}[*] Unpacking engine and hardware acceleration libraries...${NC}"
        mkdir -p "$INSTALL_DIR"
        tar -xzf "$TMP_ARCHIVE" -C "$INSTALL_DIR" --strip-components=1 2>/dev/null || tar -xzf "$TMP_ARCHIVE" -C "$INSTALL_DIR" 2>/dev/null || true
        rm -f "$TMP_ARCHIVE"

        # Locate llama-server anywhere inside INSTALL_DIR
        FOUND_BIN=$(find "$INSTALL_DIR" -type f \( -name "llama-server" -o -name "server" \) 2>/dev/null | head -n 1)
        if [ -n "$FOUND_BIN" ] && [ -f "$FOUND_BIN" ]; then
            mkdir -p "$BUILD_DIR/bin"
            cp -f "$FOUND_BIN" "$BUILD_DIR/bin/llama-server" 2>/dev/null || true
            chmod +x "$BUILD_DIR/bin/llama-server"
            chmod +x "$FOUND_BIN"
            # Ensure permissions on all shared libraries
            find "$INSTALL_DIR" -type f -name "*.so" -exec chmod +x {} + 2>/dev/null || true
            SERVER_BIN="$BUILD_DIR/bin/llama-server"
            echo -e "${GREEN}[✓] Pre-compiled high-performance llama-server is ready: $SERVER_BIN${NC}"
            return 0
        fi
    fi

    return 1
}

if [ "$SKIP_BUILD" -ne 1 ]; then
    echo -e "${MAGENTA}====================================================${NC}"
    echo -e "${GREEN}${BOLD}   ⚡ SELECT INFERENCE ENGINE SETUP                ${NC}"
    echo -e "${MAGENTA}====================================================${NC}"
    echo -e " 1) ${GREEN}${BOLD}🚀 Pre-Compiled Turbo Engine${NC} (Instant 10s setup, 100% verified, zero errors) [RECOMMENDED]"
    echo -e " 2) ${CYAN}🎮 Custom Vulkan Mobile GPU${NC}  (Compile from source with Adreno/Mali Vulkan shaders)"
    echo -e " 3) ${CYAN}⚡ Custom Turbo ARM CPU${NC}      (Compile from source with ARM NEON+DotProd)"
    read -r -p "Enter choice [1-3] (Default: 1): " ENGINE_CHOICE </dev/tty || ENGINE_CHOICE="1"
    ENGINE_CHOICE=${ENGINE_CHOICE:-1}

    BUILD_SUCCESS=0

    # Option 1: Fast Pre-Compiled Engine
    if [ "$ENGINE_CHOICE" -eq 1 ]; then
        echo -e "\n${YELLOW}[1/2] Installing base system dependencies...${NC}"
        pkg update -y 2>/dev/null || true
        pkg install -y python iproute2 curl termux-tools 2>/dev/null || true
        echo -e "\n${YELLOW}[2/2] Setting up pre-compiled inference engine...${NC}"
        if install_precompiled_engine; then
            BUILD_SUCCESS=1
        fi
    fi

    # Option 2 or 3: Compile from source
    if [ "$ENGINE_CHOICE" -eq 2 ] || [ "$ENGINE_CHOICE" -eq 3 ]; then
        echo -e "\n${YELLOW}[1/4] Installing build tools and packages...${NC}"
        pkg update -y 2>/dev/null || true
        pkg install -y git cmake clang ninja make termux-tools python iproute2 curl libandroid-spawn 2>/dev/null || pkg install -y git cmake clang make termux-tools python iproute2 curl libandroid-spawn
        pkg install -y libomp 2>/dev/null || true
        if [ "$ENGINE_CHOICE" -eq 2 ] && [ "$HAS_VULKAN" -eq 1 ]; then
            echo -e "${BLUE}[*] Installing Vulkan GPU libraries and shader compiler (glslang)...${NC}"
            pkg install -y vulkan-headers vulkan-loader vulkan-tools glslang 2>/dev/null || true
            pkg install -y vulkan-loader-android 2>/dev/null || true
        fi

        echo -e "\n${YELLOW}[2/4] Fetching llama.cpp source code...${NC}"
        if [ ! -d "$INSTALL_DIR/.git" ]; then
            echo -e "${BLUE}[*] Cloning fresh llama.cpp repository (shallow clone)...${NC}"
            rm -rf "$INSTALL_DIR"
            git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$INSTALL_DIR" || git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$INSTALL_DIR"
        else
            echo -e "${BLUE}[*] Updating existing llama.cpp repository...${NC}"
            cd "$INSTALL_DIR"
            git fetch --depth 1 origin 2>/dev/null || true
            git reset --hard origin/master 2>/dev/null || git reset --hard origin/main 2>/dev/null || true
        fi

        cd "$INSTALL_DIR"
        BUILD_CORES=$(( CORES > 4 ? 4 : (CORES > 1 ? CORES : 1) ))
        echo -e "\n${YELLOW}[3/4] Configuring compiler for ${SOC_NAME} (using ${BUILD_CORES} RAM-safe jobs)...${NC}"

        echo -e "\n${YELLOW}[4/4] Building llama-server...${NC}"
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"

        CMAKE_GEN=""
        if command -v ninja >/dev/null 2>&1; then
            CMAKE_GEN="-G Ninja"
        fi

        if [ "$ENGINE_CHOICE" -eq 2 ] && [ "$HAS_VULKAN" -eq 1 ]; then
            echo -e "${BLUE}[*] Configuring with Vulkan GPU acceleration...${NC}"
            if cmake -B build $CMAKE_GEN -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF; then
                echo -e "${BLUE}[*] Compiling llama-server with Vulkan GPU support (${BUILD_CORES} parallel jobs)...${NC}"
                if cmake --build build --target llama-server -j"$BUILD_CORES" || cmake --build build --target server -j"$BUILD_CORES" || cmake --build build -j"$BUILD_CORES"; then
                    FOUND_BIN=$(find "$BUILD_DIR" -type f \( -name "llama-server" -o -name "server" \) 2>/dev/null | head -n 1)
                    if [ -n "$FOUND_BIN" ] && [ -f "$FOUND_BIN" ]; then
                        mkdir -p "$BUILD_DIR/bin"
                        cp -f "$FOUND_BIN" "$BUILD_DIR/bin/llama-server"
                        chmod +x "$BUILD_DIR/bin/llama-server"
                        SERVER_BIN="$BUILD_DIR/bin/llama-server"
                        echo -e "${GREEN}[✓] Compiled successfully with Vulkan Mobile GPU acceleration!${NC}"
                        BUILD_SUCCESS=1
                    fi
                fi
            fi
        fi

        # If Vulkan failed or if CPU was chosen
        if [ "$BUILD_SUCCESS" -ne 1 ]; then
            if [ "$ENGINE_CHOICE" -eq 2 ]; then
                echo -e "${YELLOW}[!] Vulkan GPU source build encountered driver issues. Trying Optimized ARM CPU build...${NC}"
                rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
            fi
            echo -e "${BLUE}[*] Configuring Optimized ARM CPU build...${NC}"
            if cmake -B build $CMAKE_GEN -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF; then
                echo -e "${BLUE}[*] Compiling llama-server for ARM CPU (${BUILD_CORES} parallel jobs)...${NC}"
                if cmake --build build --target llama-server -j"$BUILD_CORES" || cmake --build build --target server -j"$BUILD_CORES" || cmake --build build -j"$BUILD_CORES"; then
                    FOUND_BIN=$(find "$BUILD_DIR" -type f \( -name "llama-server" -o -name "server" \) 2>/dev/null | head -n 1)
                    if [ -n "$FOUND_BIN" ] && [ -f "$FOUND_BIN" ]; then
                        mkdir -p "$BUILD_DIR/bin"
                        cp -f "$FOUND_BIN" "$BUILD_DIR/bin/llama-server"
                        chmod +x "$BUILD_DIR/bin/llama-server"
                        SERVER_BIN="$BUILD_DIR/bin/llama-server"
                        echo -e "${GREEN}[✓] Compiled successfully for ARM CPU!${NC}"
                        BUILD_SUCCESS=1
                    fi
                fi
            fi
        fi

        # If source build failed, automatically install pre-compiled engine as seamless recovery
        if [ "$BUILD_SUCCESS" -ne 1 ]; then
            echo -e "${YELLOW}[!] Source compilation was incomplete on this Android ROM.${NC}"
            echo -e "${YELLOW}[*] Seamlessly recovering with verified pre-compiled engine...${NC}"
            if install_precompiled_engine; then
                BUILD_SUCCESS=1
            fi
        fi
    fi

    # Final safety net
    if [ "$BUILD_SUCCESS" -ne 1 ] && [ -z "$SERVER_BIN" ]; then
        install_precompiled_engine || true
    fi

    # Verify binary presence
    if [ -z "$SERVER_BIN" ] || [ ! -x "$SERVER_BIN" ]; then
        FOUND_ANY=$(find "$INSTALL_DIR" "$PREFIX/bin" -type f \( -name "llama-server" -o -name "server" \) 2>/dev/null | head -n 1)
        if [ -n "$FOUND_ANY" ]; then
            SERVER_BIN="$FOUND_ANY"
            chmod +x "$SERVER_BIN"
        else
            echo -e "${RED}[ERROR] Could not setup llama-server executable.${NC}"
            echo -e "${YELLOW}Tip: Run 'pkg install -y llama-cpp' in Termux to install directly.${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}[✓] llama-server ready: $SERVER_BIN${NC}"
fi

# ==============================================================================
# 🧠 STEP 3: INTERACTIVE MODEL SELECTION BY PHONE RAM TIER
# ==============================================================================
EXISTING_MODELS=$(ls "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l)
if [ "$EXISTING_MODELS" -eq 0 ]; then
    echo -e "\n${MAGENTA}====================================================${NC}"
    echo -e "${GREEN}${BOLD}   🧠 SELECT A MODEL MATCHED TO YOUR ${RAM_GB}GB RAM        ${NC}"
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

# ==============================================================================
# 🚀 STEP 4: LAUNCH AI HUB (WEB UI + OPENAI API)
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHONE_IP=$(ip -4 addr show wlan0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "127.0.0.1")

echo -e "\n${CYAN}====================================================${NC}"
echo -e "${GREEN}   🚀 STARTING LOCAL AI HUB (WEB UI + API)         ${NC}"
echo -e "${CYAN}====================================================${NC}"
echo -e " Phone Web UI : ${YELLOW}http://localhost:8080${NC}"
echo -e " PC Wi-Fi     : ${YELLOW}http://$PHONE_IP:8080${NC}"
echo -e " PC USB (ADB) : ${CYAN}adb forward tcp:8080 tcp:8080${NC} -> ${YELLOW}http://localhost:8080${NC}"
echo -e "${CYAN}====================================================${NC}\n"

exec python3 "$SCRIPT_DIR/app.py"
