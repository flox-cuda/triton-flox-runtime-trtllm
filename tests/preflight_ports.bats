#!/usr/bin/env bats
# tests/preflight_ports.bats — Port list construction tests for triton-preflight.
#
# Strategy: Create a python3 shim that intercepts the port-scanning heredoc
# (identified by 'find_listen_inodes_multi' in stdin) and prints the ports
# that were passed as arguments. All other python3 calls are forwarded.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup

  # Find the real python3 path before we shadow it
  REAL_PYTHON3="$(command -v python3)"
  export REAL_PYTHON3

  # Create a python3 shim that intercepts the port-scanning heredoc
  cat > "$TEST_TMPDIR/bin/python3" <<'SHIM'
#!/usr/bin/env bash
# Read stdin to determine if this is the port-scanning heredoc
stdin_content="$(cat)"
if [[ "$stdin_content" == *"find_listen_inodes_multi"* ]]; then
  # This is the port-scanning call. Extract the ports from args.
  # Args: python3 - <host> <euid> <port1> <port2> ...
  shift  # skip the '-'
  shift  # skip host
  shift  # skip euid
  echo "INTERCEPTED_PORTS: $*" >&2
  # Exit cleanly (no listeners found = exit 0)
  exit 0
else
  # Forward to real python3
  echo "$stdin_content" | exec "$REAL_PYTHON3" "$@"
fi
SHIM
  chmod +x "$TEST_TMPDIR/bin/python3"

  # Preflight needs flock
  command -v flock >/dev/null 2>&1 || skip "flock not available"

  export TRITON_HTTP_PORT=8000
  export TRITON_GRPC_PORT=8001
  export TRITON_METRICS_PORT=8002
  export TRITON_OPENAI_PORT=9000
  export TRITON_HOST=0.0.0.0
  export TRITON_GPU_WARN_PCT=50
  export TRITON_SKIP_GPU_CHECK=1
  export TRITON_DRY_RUN=0
  export TRITON_PREFLIGHT_JSON=0
  export TRITON_OWNER_REGEX=""
  export TRITON_ALLOW_KILL_OTHER_UID=0
  export TRITON_TERM_GRACE=3
  export TRITON_PORT_FREE_TIMEOUT=10
  export TRITON_PREFLIGHT_LOCKFILE="$TEST_TMPDIR/preflight.lock"
}

teardown() {
  common_teardown
}

@test "preflight ports: standard mode has 3 ports" {
  export TRITON_OPENAI_FRONTEND=false
  run "$SCRIPTS_DIR/triton-preflight"
  [ "$status" -eq 0 ]
  # Extract the intercepted ports line
  intercepted_line="$(echo "$output" | grep 'INTERCEPTED_PORTS:' || true)"
  [ -n "$intercepted_line" ]
  # Count the ports (space-separated after the label)
  port_str="${intercepted_line#*INTERCEPTED_PORTS: }"
  read -ra ports <<< "$port_str"
  [ "${#ports[@]}" -eq 3 ]
}

@test "preflight ports: openai mode has 4 ports" {
  export TRITON_OPENAI_FRONTEND=true
  run "$SCRIPTS_DIR/triton-preflight"
  [ "$status" -eq 0 ]
  intercepted_line="$(echo "$output" | grep 'INTERCEPTED_PORTS:' || true)"
  [ -n "$intercepted_line" ]
  port_str="${intercepted_line#*INTERCEPTED_PORTS: }"
  read -ra ports <<< "$port_str"
  [ "${#ports[@]}" -eq 4 ]
}

@test "preflight ports: TRITON_OPENAI_FRONTEND=banana fails validation" {
  export TRITON_OPENAI_FRONTEND=banana
  run "$SCRIPTS_DIR/triton-preflight"
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be true/false/1/0/yes/no"* ]]
}

@test "preflight ports: openai port value present in 4-port case" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_PORT=9000
  run "$SCRIPTS_DIR/triton-preflight"
  [ "$status" -eq 0 ]
  intercepted_line="$(echo "$output" | grep 'INTERCEPTED_PORTS:' || true)"
  [[ "$intercepted_line" == *"9000"* ]]
}
