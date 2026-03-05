#!/usr/bin/env bats
# tests/serve_validation.bats — Error paths and validation tests.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  make_mock_tritonserver
  setup_serve_env
}

teardown() {
  common_teardown
}

@test "validation: TRITON_OPENAI_FRONTEND=banana fails" {
  export TRITON_OPENAI_FRONTEND=banana
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be true/false/1/0/yes/no"* ]]
}

@test "validation: TRITON_OPENAI_PORT=0 in openai mode fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_PORT=0
  TRITON_OPENAI_MAIN="$(make_mock_openai_main)"
  export TRITON_OPENAI_MAIN
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be > 0"* ]]
}

@test "validation: TRITON_OPENAI_PORT=abc in openai mode fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_PORT=abc
  TRITON_OPENAI_MAIN="$(make_mock_openai_main)"
  export TRITON_OPENAI_MAIN
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be a positive integer"* ]]
}

@test "validation: auto-discovery finds main.py relative to mock tritonserver" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN=""
  # Create main.py relative to mock tritonserver: bin/../python/openai/main.py
  mkdir -p "$TEST_TMPDIR/python/openai"
  cat > "$TEST_TMPDIR/python/openai/main.py" <<'STUB'
#!/usr/bin/env python3
import sys; sys.exit(0)
STUB
  chmod +x "$TEST_TMPDIR/python/openai/main.py"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"main.py"* ]]
}

@test "validation: auto-discovery fails with clear error when main.py absent" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN=""
  # tritonserver is on PATH but no main.py exists relative to it
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"not found"* ]]
}

@test "validation: TRITON_OPENAI_MAIN=/nonexistent/path fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN="/nonexistent/path/main.py"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"not found at"* ]]
}

@test "validation: standard mode requires tritonserver on PATH" {
  # Remove mock tritonserver
  rm -f "$TEST_TMPDIR/bin/tritonserver"
  # Also clear bash hash table
  hash -d tritonserver 2>/dev/null || true
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"tritonserver"* ]]
  [[ "$output" == *"not found"* ]]
}
