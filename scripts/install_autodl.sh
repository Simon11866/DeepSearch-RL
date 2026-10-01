#!/usr/bin/env bash
# =============================================================================
# DeepSearch-RL AutoDL 环境一键安装（Ubuntu + 8×RTX 4090 24GB / sm_89）
#
# 目标：Python 3.12 + CUDA 12.6 + PyTorch 2.8.0+cu126
#       + verl v0.6.0 + sglang 0.5.2 + flashinfer 0.3.1 + flash-attn 2.8.3
#
# 用法（在 AutoDL 实例内，工程根目录执行）：
#   bash scripts/install_autodl.sh
#
# 为什么 4090 用 cu126 而不是 cu128：
# - 4090 是 Ada sm_89（不是 Blackwell），CUDA 12.4/12.6 生态最成熟、轮子最全；
# - 但 verl v0.6.0 强制要求 torch==2.8.0（见其 setup.py 的 sglang extra），
#   因此 torch 版本不能降，只把「CUDA 车道」从 cu128 换成同样支持 torch2.8 的 cu126；
# - 4090 上 flash-attn 有官方预编译 wheel，无需再用 liger-kernel 替代。
#
# 设计：
# - set -e：任一步失败立即停；
# - 可重复执行：已存在的目录/包会跳过或覆盖安装，不会把环境搞坏；
# - 不真正下载模型/数据，只搭训练环境。
# =============================================================================
set -euo pipefail

# 切到工程根目录（本脚本位于 scripts/ 下）
cd "$(dirname "$0")/.."

# 允许通过环境变量覆盖 CUDA 车道（默认 cu126；5090 可改 cu128）
CUDA_TAG=${CUDA_TAG:-cu126}

echo "======================================================================"
echo "[1/7] 基础环境自检（python / pip / nvcc）"
echo "======================================================================"
PY=${PYTHON:-python}
echo "[info] 使用解释器： $($PY --version 2>&1)"
echo "[info] pip：       $($PY -m pip --version 2>&1)"
echo "[info] CUDA 车道： $CUDA_TAG（可用 CUDA_TAG=cu128 bash ... 覆盖）"

# 编译 flash-attn 源码兜底时需要 nvcc；有预编译 wheel 时可忽略
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[提示] 未检测到 nvcc（仅从源码编译 flash-attn 时需要；本流程优先用预编译 wheel）"
fi

# 升级 pip，避免老 pip 解析不了新 wheel
$PY -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# [2/7] 安装 PyTorch 2.8.0 + 选定 CUDA 车道（verl v0.6.0 强制 torch==2.8.0）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[2/7] 安装 torch==2.8.0 $CUDA_TAG"
echo "======================================================================"
$PY -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/$CUDA_TAG

# ---------------------------------------------------------------------------
# [3/7] 安装通用 python 依赖（对齐 requirements.txt 的版本口径）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[3/7] 安装通用依赖（modelscope / datasets / ray / swanlab 等）"
echo "======================================================================"
if [[ -f requirements.txt ]]; then
    $PY -m pip install -r requirements.txt
else
    # 与 requirements.txt 对齐的最小集合（防 requirements.txt 缺失时脚本仍可跑）
    $PY -m pip install \
        "modelscope==1.23.1" \
        "datasets>=2.20.0" "huggingface_hub>=0.25.0" "pyarrow>=15.0.0" \
        "ray[data,train,tune,serve]>=2.45.0,<2.49" \
        "hydra-core>=1.3.2" \
        "transformers>=4.51.0" "accelerate>=0.34.0" "peft>=0.12.0" \
        "numpy<2.0" pandas dill pylatexenc tiktoken codetiming \
        "swanlab==0.9.0" \
        "openai>=1.40.0" "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
        "tenacity>=2.3.0" "aiohttp>=3.10.0" "requests>=2.32.0" \
        "trafilatura>=1.12.0" "beautifulsoup4>=4.12.0" "lxml>=5.3.0"
fi

# ---------------------------------------------------------------------------
# [4/7] clone verl 并 checkout v0.6.0（必须 pin，main 已迁 CUDA13/torch 更高版本）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[4/7] 安装 verl v0.6.0（源码 editable，含 sglang extra）"
echo "======================================================================"
VERL_DIR=${VERL_DIR:-$HOME/verl}
if [[ ! -d "$VERL_DIR" ]]; then
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
else
    echo "[info] 已存在 $VERL_DIR，跳过 clone"
fi
cd "$VERL_DIR"
git fetch --tags --force
git checkout v0.6.0
# 装 verl 本体 + sglang extra（其 setup.py 会 pin sglang==0.5.2、torch==2.8.0）
$PY -m pip install -e ".[sglang]"
cd - >/dev/null

# ---------------------------------------------------------------------------
# [5/7] sglang 版本二次固定 + flashinfer 0.3.1（预编译优先，PyPI 源码兜底）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[5/7] 固定 sglang==0.5.2 并安装 flashinfer 0.3.1"
echo "======================================================================"
# verl 的 setup.py 已 pin sglang==0.5.2；这里再显式固定一次，防止 pip 自动升新
$PY -m pip install "sglang[srt,openai]==0.5.2"

# flashinfer 0.3.1：优先用匹配 CUDA 车道的预编译 wheel（免编译、即装即用）
FLASHINFER_IDX="https://flashinfer.ai/whl/$CUDA_TAG/torch2.8/flashinfer-python"
if ! $PY -m pip install "flashinfer_python==0.3.1" --find-links "$FLASHINFER_IDX"; then
    echo "[提示] 未找到 $CUDA_TAG/torch2.8 的预编译 flashinfer，改用 PyPI 源码安装（首次运行会 JIT 编译）"
    $PY -m pip install --no-build-isolation "flashinfer_python==0.3.1"
fi

# sgl-kernel 由 sglang[srt] 自动 pin 到 0.3.9.post2（PyPI 稳定 ABI wheel，兼容 4090）

# ---------------------------------------------------------------------------
# [6/7] flash-attn（4090 sm_89 有官方预编译 wheel，正常安装）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[6/7] 安装 flash-attn 2.8.3（预编译 wheel，4090 可用）"
echo "======================================================================"
# 解析 python 版本标签，如 cp312
PY_TAG=$($PY - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)
FA_VER=${FA_VER:-2.8.3}
FA_OK=0
# 官方 PyTorch manylinux wheel 用新 C++ ABI（cxx11abiTRUE）；先试 TRUE 再试 FALSE
for ABI in TRUE FALSE; do
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12torch2.8cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    echo "[info] 尝试 flash-attn wheel: cxx11abi${ABI}"
    if $PY -m pip install "$FA_URL"; then
        FA_OK=1
        break
    fi
done
if [[ "$FA_OK" -ne 1 ]]; then
    echo "[提示] 预编译 wheel 未命中（可能 python 版本非 3.10~3.12），改用源码编译（需 nvcc，约 10-30 分钟）"
    MAX_JOBS=${MAX_JOBS:-4} $PY -m pip install "flash-attn==${FA_VER}" --no-build-isolation
fi

# ---------------------------------------------------------------------------
# [7/7] 自检
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[7/7] 安装完成，自检关键 import"
echo "======================================================================"
$PY - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "ray", "swanlab", "omegaconf", "hydra",
        "verl", "sglang", "flashinfer", "flash_attn", "modelscope", "datasets"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  [ok] {m:14s} {v}")
    except Exception as e:
        print(f"  [FAIL] {m:14s} {type(e).__name__}: {e}")
print("python:", sys.version.split()[0])
try:
    import torch
    print("cuda available:", torch.cuda.is_available(), "| gpu count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda:", torch.version.cuda, "| cap:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch cuda check failed:", e)
PY

echo "======================================================================"
echo "下一步："
echo "  1) bash scripts/download_model.sh   # 下 Qwen3-8B"
echo "  2) export MODEL_PATH=\$HOME/models/Qwen3-8B"
echo "  3) bash scripts/start_retrieval.sh  # 终端 A：检索服务"
echo "  4) bash scripts/start_judge.sh       # 终端 B：Judge 服务"
echo "  5) bash scripts/train.sh             # 终端 C：训练"
echo "======================================================================"
