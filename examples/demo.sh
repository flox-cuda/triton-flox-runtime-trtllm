#!/usr/bin/env bash
# Phi-4-mini TRT-LLM demo — showcases the Triton + TensorRT-LLM inference pipeline.
#
# This script sends a series of prompts to a running Triton server via the
# OpenAI-compatible frontend (port 9000). It demonstrates basic Q&A, streaming,
# reasoning, code generation, and creative writing.
#
# Prerequisites:
#   1. You need a running Triton server with the OpenAI frontend enabled.
#      Open a terminal in the triton-runtime directory and run:
#
#        flox activate --start-services
#
#      Wait until you see "Started HTTPService at 0.0.0.0:9000" in the logs.
#      The manifest already sets TRITON_OPENAI_FRONTEND=true by default.
#
#   2. This script must be run from a second terminal, also inside the
#      triton-runtime directory. You need to be in the Flox environment
#      (for python3 and curl):
#
#        flox activate
#        ./examples/demo.sh
#
#      Or in one line:
#
#        flox activate -- ./examples/demo.sh
#
# Usage:
#   ./examples/demo.sh              # run the full demo (5 prompts)
#   ./examples/demo.sh --quick      # shorter version (3 prompts)
#
# Environment:
#   TRITON_OPENAI_BASE   API base URL   (default: http://localhost:9000/v1)
#   TRITON_MODEL         Model name     (default: phi4_mini_trtllm)

set -euo pipefail

base="${TRITON_OPENAI_BASE:-http://localhost:9000/v1}"
model="${TRITON_MODEL:-phi4_mini_trtllm}_ensemble"
quick=false
[[ "${1:-}" == "--quick" ]] && quick=true

# --- Helpers ---

bold=$'\033[1m'
dim=$'\033[2m'
cyan=$'\033[36m'
green=$'\033[32m'
yellow=$'\033[33m'
reset=$'\033[0m'

banner() { printf '\n%s── %s ──%s\n\n' "$bold" "$1" "$reset"; }
prompt() { printf '%s> %s%s' "$cyan" "$1" "$reset"; }
info()   { printf '%s%s%s\n' "$dim" "$1" "$reset"; }
ok()     { printf '%s✓ %s%s\n' "$green" "$1" "$reset"; }
warn()   { printf '%s⚠ %s%s\n' "$yellow" "$1" "$reset"; }
pause()  { if ! $quick; then sleep "${1:-1}"; fi; }

json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")'
}

chat_request() {
  local user_msg="$1"
  local max_tokens="${2:-128}"
  local escaped
  escaped=$(json_escape "$user_msg")
  curl -s "${base}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${model}\",
      \"messages\": [{\"role\": \"user\", \"content\": ${escaped}}],
      \"max_tokens\": ${max_tokens}
    }" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

chat_stream() {
  local user_msg="$1"
  local max_tokens="${2:-128}"
  local escaped
  escaped=$(json_escape "$user_msg")
  curl -sN "${base}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${model}\",
      \"messages\": [{\"role\": \"user\", \"content\": ${escaped}}],
      \"max_tokens\": ${max_tokens},
      \"stream\": true
    }" | python3 -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: {"):
        continue
    chunk = json.loads(line[6:])
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
print()
'
}

# --- Preflight ---

banner "Phi-4-mini TRT-LLM Demo"
info "Model:    ${model}"
info "Endpoint: ${base}"
echo

printf "Checking server... "
if ! curl -sf "${base}/models" >/dev/null 2>&1; then
  warn "Server not reachable at ${base}"
  echo "Start the server first:"
  echo "  flox activate --start-services"
  exit 1
fi
ok "Server is running"
echo

# --- Demo 1: Basic Q&A ---

banner "1. Basic Question Answering"
prompt "What is the theory of relativity in simple terms?"
echo
pause
chat_request "What is the theory of relativity in simple terms? Explain in 2-3 sentences." 512
pause 2

# --- Demo 2: Streaming ---

banner "2. Streaming Response"
info "(tokens appear as they are generated)"
echo
prompt "Write a haiku about programming."
echo
pause
chat_stream "Write a single haiku about programming. Output only the haiku, nothing else." 512
pause 2

# --- Demo 3: Reasoning ---

banner "3. Reasoning & Math"
prompt "If a train travels 120 miles in 2 hours, what is its speed?"
echo
pause
chat_request "If a train travels 120 miles in 2 hours, what is its average speed in mph? Show your work briefly." 512
pause 2

if ! $quick; then

# --- Demo 4: Code generation ---

banner "4. Code Generation"
prompt "Write a Python function to check if a number is prime."
echo
pause
chat_stream "Write a short Python function called is_prime(n) that returns True if n is prime. Include a brief docstring. Output only the code." 768
pause 2

# --- Demo 5: Creative writing ---

banner "5. Creative Writing (Streaming)"
prompt "Tell me a very short story about a robot learning to paint."
echo
pause
chat_stream "Write a very short story (3-4 sentences) about a robot that learns to paint. Be creative and vivid." 768
pause 2

fi

# --- Done ---

banner "Demo Complete"
info "Model: microsoft/Phi-4-mini-instruct (3.8B params, INT4 AWQ)"
info "Engine: TensorRT-LLM on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'GPU')"
echo
info "Try the interactive chat:"
info "  ./examples/chat.sh"
echo
