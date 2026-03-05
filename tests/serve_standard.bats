#!/usr/bin/env bats
# tests/serve_standard.bats — Standard tritonserver mode tests.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  make_mock_tritonserver
  setup_serve_env
}

teardown() {
  common_teardown
}

@test "standard: dry-run produces correct tritonserver argv" {
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"CMD:"* ]]
  [[ "$output" == *"tritonserver"* ]]
  [[ "$output" == *"--model-repository="* ]]
  [[ "$output" == *"--http-port=8000"* ]]
  [[ "$output" == *"--grpc-port=8001"* ]]
  [[ "$output" == *"--metrics-port=8002"* ]]
  [[ "$output" == *"--model-control-mode=none"* ]]
  [[ "$output" == *"--strict-readiness=true"* ]]
  [[ "$output" == *"--log-verbose=0"* ]]
}

@test "standard: --allow-http=false when TRITON_ALLOW_HTTP=false" {
  export TRITON_ALLOW_HTTP=false
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--allow-http=false"* ]]
}

@test "standard: --allow-grpc=false when TRITON_ALLOW_GRPC=false" {
  export TRITON_ALLOW_GRPC=false
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--allow-grpc=false"* ]]
}

@test "standard: --allow-metrics=false when TRITON_ALLOW_METRICS=false" {
  export TRITON_ALLOW_METRICS=false
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--allow-metrics=false"* ]]
}

@test "standard: backend config entries expanded" {
  export TRITON_BACKEND_CONFIG="tensorrt:coalesced-memory-size=256,python:shm-default-byte-size=1048576"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"--backend-config=tensorrt:coalesced-memory-size=256"* ]]
  [[ "$output" == *"--backend-config=python:shm-default-byte-size=1048576"* ]]
}

@test "standard: extra passthrough args appear in command" {
  run "$SCRIPTS_DIR/triton-serve" --dry-run -- --buffer-manager-thread-count 8
  [ "$status" -eq 0 ]
  [[ "$output" == *"--buffer-manager-thread-count"* ]]
  [[ "$output" == *"8"* ]]
}
