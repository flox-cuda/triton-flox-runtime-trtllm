#!/usr/bin/env bash
# Interactive chat with the Triton-served model via the OpenAI-compatible API.
#
# Usage:
#   ./examples/chat.sh                          # interactive mode
#   ./examples/chat.sh "What is 2+2?"           # single prompt
#   ./examples/chat.sh --no-stream "Hello"      # disable streaming
#
# Environment:
#   TRITON_OPENAI_BASE   API base URL   (default: http://localhost:9000/v1)
#   TRITON_MODEL         Model name     (default: qwen2_5_05b_trtllm_ensemble)
#   MAX_TOKENS           Max tokens     (default: 64)

set -euo pipefail

base="${TRITON_OPENAI_BASE:-http://localhost:9000/v1}"
model="${TRITON_MODEL:-qwen2_5_05b_trtllm}_ensemble"
max_tokens="${MAX_TOKENS:-64}"
stream=true

# Parse flags
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --no-stream) stream=false; shift ;;
    --model)     model="$2"; shift 2 ;;
    --max-tokens) max_tokens="$2"; shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

send_chat() {
  local user_msg="$1"
  # Escape for JSON
  user_msg=$(printf '%s' "$user_msg" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")')

  if [[ "$stream" == "true" ]]; then
    curl -sN "${base}/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${model}\",
        \"messages\": [{\"role\": \"user\", \"content\": ${user_msg}}],
        \"max_tokens\": ${max_tokens},
        \"stream\": true
      }" | while IFS= read -r line; do
        # Skip empty lines and [DONE]
        [[ "$line" =~ ^data:\ \{  ]] || continue
        chunk="${line#data: }"
        delta=$(printf '%s' "$chunk" | python3 -c 'import json,sys; d=json.load(sys.stdin)["choices"][0]["delta"]; print(d.get("content",""), end="")')
        printf '%s' "$delta"
      done
    echo
  else
    curl -s "${base}/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${model}\",
        \"messages\": [{\"role\": \"user\", \"content\": ${user_msg}}],
        \"max_tokens\": ${max_tokens}
      }" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
  fi
}

# Single prompt mode
if [[ $# -gt 0 ]]; then
  send_chat "$*"
  exit 0
fi

# Interactive mode
echo "Triton Chat  (model: ${model}, streaming: ${stream})"
echo "Type your message, or 'quit' to exit."
echo
while true; do
  printf '\033[1m> \033[0m'
  IFS= read -r prompt || break
  [[ -n "$prompt" ]] || continue
  [[ "$prompt" != "quit" && "$prompt" != "exit" ]] || break
  send_chat "$prompt"
  echo
done
