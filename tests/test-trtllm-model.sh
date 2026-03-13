#!/usr/bin/env bash
# test-trtllm-model.sh — End-to-end test of qwen2_5_05b_trtllm via Triton
#
# Usage:
#   # From inside flox activate:
#   ./tests/test-trtllm-model.sh
#
#   # Or launch the server and test in one shot:
#   ./tests/test-trtllm-model.sh --start-server
#
# Prerequisites:
#   - flox activate (sets up backends, models, env vars)
#   - tritonserver running (or use --start-server)

set -euo pipefail

MODEL="qwen2_5_05b_trtllm"
HOST="127.0.0.1"
PORT="${TRITON_HTTP_PORT:-8000}"
BASE="http://${HOST}:${PORT}"
MAX_WAIT=120  # seconds to wait for server readiness
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    echo "Stopping tritonserver (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Start server if requested ────────────────────────────────────────
if [[ "${1:-}" == "--start-server" ]]; then
  echo "Starting tritonserver with --model-control-mode=explicit --load-model=${MODEL} ..."

  tritonserver \
    --model-repository="${TRITON_MODEL_REPOSITORY}" \
    --backend-directory="${TRITON_BACKEND_DIR}" \
    --model-control-mode=explicit \
    --load-model="${MODEL}" \
    --http-port="${PORT}" \
    --grpc-port="${TRITON_GRPC_PORT:-8001}" \
    --metrics-port="${TRITON_METRICS_PORT:-8002}" \
    --log-verbose=0 &
  SERVER_PID=$!
  echo "tritonserver pid: $SERVER_PID"
fi

# ── Wait for readiness ───────────────────────────────────────────────
echo "Waiting for server at ${BASE} ..."
elapsed=0
while ! curl -sf "${BASE}/v2/health/ready" >/dev/null 2>&1; do
  if [[ $elapsed -ge $MAX_WAIT ]]; then
    echo "FAIL: server not ready after ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  printf "  %ds ...\r" "$elapsed"
done
echo "Server ready (${elapsed}s)."

# ── Verify model is loaded ───────────────────────────────────────────
echo ""
echo "=== Model metadata ==="
model_meta=$(curl -sf "${BASE}/v2/models/${MODEL}" || true)
if [[ -z "$model_meta" ]]; then
  echo "FAIL: model ${MODEL} not found. Available models:"
  curl -sf "${BASE}/v2/models" | python3 -m json.tool 2>/dev/null || echo "(could not list models)"
  exit 1
fi
echo "$model_meta" | python3 -m json.tool

# ── Send inference request ───────────────────────────────────────────
echo ""
echo "=== Inference test ==="

python3 - "${BASE}" "${MODEL}" <<'PYEOF'
import json, sys, urllib.request

base_url, model = sys.argv[1], sys.argv[2]

# ── Tokenize ─────────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
except Exception as e:
    print(f"WARN: could not load tokenizer ({e}), using pre-computed tokens")
    tokenizer = None

prompt = "The capital of Westeros is"
if tokenizer:
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt!r}")
    print(f"Tokens ({len(input_ids)}): {input_ids}")
else:
    # Pre-computed Qwen2.5 token IDs for "The capital of France is"
    input_ids = [785, 6864, 315, 9822, 374]
    print(f"Prompt: {prompt!r}  (pre-computed tokens)")

max_new_tokens = 32

# ── Build Triton v2 inference request ────────────────────────────
request_body = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": [1, len(input_ids)],
            "datatype": "INT32",
            "data": input_ids,
        },
        {
            "name": "input_lengths",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [len(input_ids)],
        },
        {
            "name": "request_output_len",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [max_new_tokens],
        },
        {
            "name": "end_id",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [151643],  # Qwen2.5 eos_token_id
        },
        {
            "name": "pad_id",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [151643],
        },
    ],
    "outputs": [
        {"name": "output_ids"},
        {"name": "sequence_length"},
    ],
}

url = f"{base_url}/v2/models/{model}/infer"
print(f"\nPOST {url}")

req = urllib.request.Request(
    url,
    data=json.dumps(request_body).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"FAIL: HTTP {e.code}")
    print(body[:2000])
    sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# ── Parse response ───────────────────────────────────────────────
outputs = {o["name"]: o for o in result["outputs"]}
output_ids = outputs["output_ids"]["data"]
seq_len = outputs["sequence_length"]["data"][0]

# output_ids includes input; strip it
generated_ids = output_ids[len(input_ids):seq_len]

if tokenizer:
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
else:
    generated_text = f"(raw token ids: {generated_ids})"

print(f"\nGenerated ({len(generated_ids)} tokens): {generated_text}")
print(f"Full output_ids ({seq_len} total): {output_ids[:seq_len]}")
print("\nPASS")
PYEOF

echo ""
echo "=== Server metrics sample ==="
curl -sf "${BASE//$PORT/${TRITON_METRICS_PORT:-8002}}/metrics" \
  | grep -E "^nv_inference_(count|request_success)" \
  | head -5

echo ""
echo "Done."
