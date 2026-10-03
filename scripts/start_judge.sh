#!/usr/bin/env bash
# 启动 Judge。默认走 DeepSeek 云端 API，无需本机 vLLM。
# 若 JUDGE_BASE_URL 指向 127.0.0.1，则仍拉起本地 vLLM。
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

if [[ "${JUDGE_BASE_URL}" == *"deepseek.com"* ]] || [[ "${JUDGE_BASE_URL}" != *"127.0.0.1"* && "${JUDGE_BASE_URL}" != *"localhost"* ]]; then
  echo "[judge] 使用云端 API，无需启动本机 vLLM。"
  echo "[judge] JUDGE_BASE_URL=${JUDGE_BASE_URL}"
  echo "[judge] JUDGE_MODEL=${JUDGE_MODEL}"
  if [[ -z "${JUDGE_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[警告] 未设置 JUDGE_API_KEY / DEEPSEEK_API_KEY，奖励将降级为规则分。"
    exit 2
  fi
  echo "[judge] API key 已就绪（不打印）。训练进程会直接 HTTP 调用 DeepSeek。"
  exit 0
fi

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_GPU_MEM="${JUDGE_GPU_MEM:-0.4}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"

exec python -m deepsearch_rl.judge.judge_server \
  --model "${JUDGE_MODEL}" \
  --port "${JUDGE_PORT}" \
  --tensor-parallel-size "${JUDGE_TP}" \
  --gpu-memory-utilization "${JUDGE_GPU_MEM}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}"
