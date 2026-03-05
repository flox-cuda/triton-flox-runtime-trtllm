#!/usr/bin/env bats
# tests/serve_openai.bats — OpenAI frontend mode tests.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  setup_serve_env
  export TRITON_OPENAI_FRONTEND=true
  TRITON_OPENAI_MAIN="$(make_mock_openai_main)"
  export TRITON_OPENAI_MAIN
}

teardown() {
  common_teardown
}

@test "openai: dry-run produces python3 main.py with core flags" {
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"CMD:"* ]]
  [[ "$output" == *"python3"* ]]
  [[ "$output" == *"main.py"* ]]
  [[ "$output" == *"--model-repository="* ]]
  [[ "$output" == *"--openai-port=9000"* ]]
  [[ "$output" == *"--host=0.0.0.0"* ]]
}

@test "openai: --tokenizer present when TRITON_OPENAI_TOKENIZER set" {
  export TRITON_OPENAI_TOKENIZER="meta-llama/Llama-3-8B"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--tokenizer=meta-llama/Llama-3-8B"* ]]
}

@test "openai: --tritonserver-log-verbose-level present when LOG_VERBOSE > 0" {
  export TRITON_LOG_VERBOSE=2
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--tritonserver-log-verbose-level=2"* ]]
}

@test "openai: kserve frontends enabled when HTTP or gRPC truthy" {
  export TRITON_ALLOW_HTTP=true
  export TRITON_ALLOW_GRPC=true
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--enable-kserve-frontends"* ]]
  [[ "$output" == *"--kserve-http-port=8000"* ]]
  [[ "$output" == *"--kserve-grpc-port=8001"* ]]
}

@test "openai: kserve frontends absent when both HTTP and gRPC false" {
  export TRITON_ALLOW_HTTP=false
  export TRITON_ALLOW_GRPC=false
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" != *"--enable-kserve-frontends"* ]]
  [[ "$output" != *"--kserve-http-port"* ]]
  [[ "$output" != *"--kserve-grpc-port"* ]]
}

@test "openai: --backend-config NEVER present (regression)" {
  export TRITON_BACKEND_CONFIG="tensorrt:coalesced-memory-size=256"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" != *"--backend-config"* ]]
}

@test "openai: tritonserver not required on PATH" {
  # Do NOT call make_mock_tritonserver — tritonserver is absent
  # Remove any tritonserver that may be on the real PATH
  hash -d tritonserver 2>/dev/null || true
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "openai: extra passthrough args work" {
  run "$SCRIPTS_DIR/triton-serve" --dry-run -- --extra-flag val
  [ "$status" -eq 0 ]
  [[ "$output" == *"--extra-flag"* ]]
  [[ "$output" == *"val"* ]]
}
