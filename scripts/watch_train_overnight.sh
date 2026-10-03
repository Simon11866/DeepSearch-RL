#!/usr/bin/env bash
# Overnight watcher: print a wake line on crash / complete / 10-min heartbeat.
set -u
LOG=/root/autodl-tmp/DeepSearch-RL/logs/train.log
ST=/root/autodl-tmp/DeepSearch-RL/logs/watchdog_state.txt
last_wake=0
last_tb=0
was_alive=1
while true; do
  sleep 30
  now=$(date +%s)
  pid=$(pgrep -f 'python -m deepsearch_rl.train.train_grpo' | head -1 || true)
  alive=0
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    alive=1
  fi
  tb=0
  prog=""
  if [[ -f "${LOG}" ]]; then
    tb=$(grep -a -c Traceback "${LOG}" 2>/dev/null || echo 0)
    prog=$(grep -a 'Training Progress:' "${LOG}" 2>/dev/null | tail -1 | tr -d '\000' | tail -c 180 || true)
  fi
  reason=""
  if [[ "${alive}" != 1 ]]; then
    if [[ "${was_alive}" == 1 ]]; then
      reason=DEAD
    fi
    was_alive=0
  else
    was_alive=1
    if [[ "${tb}" =~ ^[1-9] ]]; then
      if [[ "${tb}" != "${last_tb}" ]]; then
        reason=TRACEBACK
        last_tb=${tb}
      fi
    elif echo "${prog}" | grep -q '36/36'; then
      reason=COMPLETE
    elif [[ $((now - last_wake)) -ge 600 ]]; then
      reason=HEARTBEAT
    fi
  fi
  if [[ -n "${reason}" ]]; then
    last_wake=${now}
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
    echo "AGENT_LOOP_WAKE_train {\"prompt\":\"Overnight DeepSearch-RL GRPO: check logs/train.log and trainer PID. If DEAD or TRACEBACK, diagnose, patch, ray stop, restart train.sh (resume_mode auto, do not kill a healthy job). If COMPLETE 36/36, stop this loop. Keep going until 36 steps.\",\"reason\":\"${reason}\",\"pid\":\"${pid}\",\"alive\":${alive},\"progress\":\"${prog}\",\"gpu\":\"${gpu}\",\"time\":\"$(date -Iseconds)\"}"
    echo "${reason} $(date -Iseconds) pid=${pid} alive=${alive} ${prog}" >> "${ST}"
    if [[ "${reason}" == COMPLETE ]]; then
      echo "AGENT_LOOP_WAKE_train complete; watcher exiting"
      exit 0
    fi
  fi
done
