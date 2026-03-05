# tests/test_helper.bash — Shared setup/teardown and mock builders for BATS tests.

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"

# --- Common setup/teardown ---

common_setup() {
  TEST_TMPDIR="$(mktemp -d)"
  mkdir -p "$TEST_TMPDIR/model-repo/my-model/1"
  mkdir -p "$TEST_TMPDIR/resolved"
  mkdir -p "$TEST_TMPDIR/cache"
  mkdir -p "$TEST_TMPDIR/bin"
  export PATH="$TEST_TMPDIR/bin:$PATH"
  export TEST_TMPDIR
}

common_teardown() {
  rm -rf "$TEST_TMPDIR"
}

# --- Mock builders ---

# Creates a minimal env file. Extra key=value args are appended.
# Prints the file path to stdout.
make_env_file() {
  local f="$TEST_TMPDIR/cache/test-model.env"
  cat > "$f" <<EOF
export _TRITON_RESOLVED_PATH='$TEST_TMPDIR/resolved'
export TRITON_MODEL_REPOSITORY='$TEST_TMPDIR/model-repo'
EOF
  local pair
  for pair in "$@"; do
    echo "export $pair" >> "$f"
  done
  echo "$f"
}

# Creates a no-op tritonserver stub in $TEST_TMPDIR/bin/.
make_mock_tritonserver() {
  cat > "$TEST_TMPDIR/bin/tritonserver" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
  chmod +x "$TEST_TMPDIR/bin/tritonserver"
}

# Creates a stub main.py at the given path (default: $TEST_TMPDIR/openai/main.py).
# Prints the path to stdout.
make_mock_openai_main() {
  local dest="${1:-$TEST_TMPDIR/openai/main.py}"
  mkdir -p "$(dirname "$dest")"
  cat > "$dest" <<'STUB'
#!/usr/bin/env python3
import sys; sys.exit(0)
STUB
  chmod +x "$dest"
  echo "$dest"
}

# Sets all required env vars to valid defaults for triton-serve tests.
# Tests override individual vars after calling this.
setup_serve_env() {
  local env_file
  env_file="$(make_env_file)"
  export TRITON_MODEL_ENV_FILE="$env_file"
  export TRITON_HTTP_PORT=8000
  export TRITON_GRPC_PORT=8001
  export TRITON_METRICS_PORT=8002
  export TRITON_MODEL_CONTROL_MODE=none
  export TRITON_LOG_VERBOSE=0
  export TRITON_STRICT_READINESS=true
  export TRITON_ALLOW_HTTP=true
  export TRITON_ALLOW_GRPC=true
  export TRITON_ALLOW_METRICS=true
  export TRITON_BACKEND_CONFIG=""
  export TRITON_ENV_FILE_TRUSTED=true
  export TRITON_OPENAI_FRONTEND=false
  export TRITON_OPENAI_PORT=9000
  export TRITON_OPENAI_MAIN=""
  export TRITON_OPENAI_TOKENIZER=""
  export TRITON_HOST=0.0.0.0
}
