# Triton Inference Server Runtime

Production NVIDIA Triton Inference Server deployment as a Flox environment. Ships with five backends: **Python**, **ONNX Runtime**, **vLLM**, **TensorRT**, and **TensorRT-LLM**. GPU-accelerated multi-port serving (HTTP, gRPC, metrics).

- **Triton Inference Server**: v2.66.0 (built from source via Nix)
- **CUDA**: requires NVIDIA driver with CUDA support
- **Platform**: Linux only (`/proc` required for preflight)

Triton serves model *repositories* -- directories containing versioned subdirectories with backend-specific artifacts and optional `config.pbtxt` files. It exposes HTTP, gRPC, and Prometheus metrics APIs; this runtime handles operational lifecycle: port reclaim, model provisioning, environment validation, and process management.

## Quick start

```bash
# Activate and start the tritonserver service
flox activate --start-services

# Override the model at activation time
TRITON_MODEL=my-onnx-model \
TRITON_MODEL_REPOSITORY=/data/models \
TRITON_MODEL_BACKEND=onnx \
  flox activate --start-services

# Launch with OpenAI-compatible frontend (port 9000)
TRITON_OPENAI_FRONTEND=true \
TRITON_OPENAI_TOKENIZER=meta-llama/Llama-3-8B \
TRITON_MODEL=llama \
  flox activate --start-services

# Serve a local TRT-LLM model (engine + tokenizer only, ensemble auto-synthesized)
TRITON_MODEL=my_llm_trtllm \
TRITON_LOCAL_MODELS=/data/trtllm-models \
  flox activate --start-services
```

### Verify it is running

```bash
# HTTP health check
curl http://127.0.0.1:8000/v2/health/ready

# Server metadata
curl http://127.0.0.1:8000/v2

# Model metadata
curl http://127.0.0.1:8000/v2/models/my-onnx-model

# Prometheus metrics
curl http://127.0.0.1:8002/metrics

# OpenAI-compatible endpoint (when TRITON_OPENAI_FRONTEND=true)
curl http://127.0.0.1:9000/v1/models
```

gRPC health checks require `grpcurl` or a gRPC client on port 8001. See the [Triton Inference Server documentation](https://github.com/triton-inference-server/server) for full API details.

### Client testing

[triton-api-client](https://github.com/barstoolbluz/triton-api-client) is a companion Flox environment with tools and examples for all four Triton client interfaces:

| Tool | Description |
|------|-------------|
| `triton-infer` | Universal inference CLI -- auto-detects model type (generate vs TRT-LLM) and routes accordingly |
| `triton-chat` | Interactive multi-turn chat REPL via OpenAI-compatible frontend (port 9000) |
| `triton-test` | Health check, smoke test, and benchmark tool |
| `examples/openai/` | Chat, streaming, and batch completions via OpenAI SDK |
| `examples/generate/` | Text generation via Triton's generate extension |
| `examples/kserve/` | KServe v2 tensor inference (HTTP, gRPC, async) and server metadata |
| `examples/trtllm/` | TRT-LLM raw tensor inference with client-side HuggingFace tokenization |

```bash
cd ~/dev/triton-api-client && flox activate

# Universal inference (auto-detects model type)
TRITON_MODEL=qwen2_5_05b_trtllm triton-infer "The capital of France is"

# Interactive chat (OpenAI frontend, port 9000)
TRITON_MODEL=my-llm triton-chat

# Health + smoke test + benchmark
TRITON_MODEL=my-llm triton-test bench -n 50 --concurrent 5
```

### Local dev vs production

| Setting | Local dev | Production |
|---------|-----------|------------|
| `TRITON_HOST` | `127.0.0.1` for local-only access | `0.0.0.0` (default) to accept remote connections |
| `TRITON_MODEL_CONTROL_MODE` | `poll` for hot-reload | `none` (default) for stability |
| `TRITON_LOG_VERBOSE` | `1` or higher for debugging | `0` (default) |
| `TRITON_MODEL_SOURCES` | `local` for pre-staged models | `flox,local,r2,hf-hub` (default) |
| `TRITON_STRICT_READINESS` | `false` during iteration | `true` (default) |
| `TRITON_ALLOW_HTTP` | `true` (default) | Disable unused protocols |
| `TRITON_ALLOW_GRPC` | `true` (default) | Disable unused protocols |
| `TRITON_ALLOW_METRICS` | `true` (default) | `true` for observability |
| `TRITON_OPENAI_FRONTEND` | `true` for OpenAI API testing | `true` when OpenAI-compatible API is needed |

Production example:

```bash
TRITON_MODEL_CONTROL_MODE=none \
TRITON_LOG_VERBOSE=0 \
TRITON_STRICT_READINESS=true \
  flox activate --start-services
```

## Architecture

The service command chains two scripts:

```
triton-resolve-model && triton-serve
```

`triton-serve` runs `triton-preflight` internally before launching the server (controlled
by `TRITON_SERVE_RUN_PREFLIGHT`, default `true`), so preflight does not need to be
chained separately.

```
┌──────────────────────────────────────────────────────────┐
│  Environment (.flox/env/manifest.toml)                    │
│                                                          │
│  [install]                                               │
│    triton-server (flox)         # server + scripts       │
│    triton-python-backend        # Python backend .so     │
│    triton-onnxruntime-backend   # ONNX Runtime backend   │
│    triton-tensorrt-backend      # TensorRT backend       │
│    triton-tensorrtllm-backend   # TensorRT-LLM backend   │
│    triton-model-*               # Model packages (opt.)  │
│    util-linux                   # flock (preflight)      │
│    iproute2                     # ss (port scanning)     │
│    vllm, torch, numpy, ...      # Python ML packages     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  on-activate (flox activate)                       │  │
│  │    triton-setup-backends  → $FLOX_ENV_CACHE/backends│  │
│  │    triton-setup-models    → $FLOX_ENV_CACHE/models  │  │
│  │      (Tier 1/2/3 + TRT-LLM ensemble synthesis)     │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  triton-resolve-model                              │  │
│  │    Sources: flox → local → r2 → hf-hub             │  │
│  │    Layout validation: version dirs + artifacts     │  │
│  │    Output: per-model .env file (mode 600)          │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  triton-serve                                      │  │
│  │    Loads .env → validates args                      │  │
│  │    Runs triton-preflight (port reclaim + GPU check) │  │
│  │    → exec tritonserver  (default)                   │  │
│  │    → exec python3 main.py  (OPENAI_FRONTEND=true)  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

1. **triton-resolve-model** -- Provisions the model repository from configured sources with per-model locking, staging directories, atomic swaps, and layout validation. Writes a per-model env file.
2. **triton-serve** -- Loads the env file (safe or trusted mode), validates all required vars, runs `triton-preflight` for port reclaim and GPU health checks, then `exec`s either `tritonserver` (default) or `python3 main.py` (when `TRITON_OPENAI_FRONTEND=true`).

`triton-preflight` can also be run standalone for diagnostics (see [Pre-flight](#pre-flight-triton-preflight)).

Scripts are bundled in the `triton-server` package at `$FLOX_ENV/bin/` and available on `PATH` after `flox activate`.

## Model repository layout

Triton requires models to follow a specific directory structure within the model repository.

### Required structure

```
$TRITON_MODEL_REPOSITORY/
  $TRITON_MODEL/
    [config.pbtxt]              # optional for many backends
    1/                          # at least one numeric version directory
      model.plan                # backend-specific artifact
    2/                          # additional versions optional
      model.plan
```

### Supported backends

| Backend | Artifact | Notes |
|---------|----------|-------|
| `tensorrt` | `model.plan` | Pre-compiled TensorRT engine |
| `tensorrtllm` | `model` (engine dir) | TensorRT-LLM engine (LLM inference with inflight batching) |
| `onnx` | `model.onnx` | ONNX Runtime |
| `pytorch` | `model.pt` | TorchScript model |
| `tensorflow` | `model.savedmodel/` | Directory; must contain `saved_model.pb` |
| `python` | `model.py` | Python backend script |
| `vllm` | `model.json` | vLLM configuration file |

**TRT-LLM model preparation workflow:** TRT-LLM models require a two-stage pipeline
before they can be served. This runtime handles *serving only* — preparation happens
in separate environments:

1. **Build engines** ([triton-trtllm-tools](../triton-trtllm-tools/)): Convert
   HuggingFace models to TRT-LLM engine format using the `tensorrt_llm` Python package
   (Python 3.12). This produces an `engine/` directory and a `tokenizer/` directory.

2. **Consume engines** (this runtime): Serve the prepared engines via one of two paths:

   - **Direct** — Point `TRITON_LOCAL_MODELS` at a directory containing your engine
     builds. `triton-setup-models` assembles them as Tier 3 and auto-synthesizes the
     ensemble pipeline (preprocessing, postprocessing, BLS orchestrator). No packaging
     step required.
   - **Packaged** — Build a Nix model package in the
     [build-trtllm-models](../builds/build-trtllm-models/) repo, publish it to the Flox
     catalog, and install it in this environment's manifest. The model appears as Tier 1
     with its ensemble bundled. This path gives you pinned versions, reproducibility, and
     shared catalog distribution.

The direct path is faster for iteration; the packaged path is better for production.

### Version directories

At least one numeric version directory (e.g., `1/`) is required. Multiple versions are supported; Triton serves the latest by default. Version directories must contain a recognized artifact for the model's backend.

### config.pbtxt

The `config.pbtxt` file is optional for many backends -- Triton can auto-generate minimal configurations. For production deployments, an explicit `config.pbtxt` is recommended to control instance groups, dynamic batching, and input/output tensor specifications. See the [Triton Model Configuration documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md).

### Backend hint

Set `TRITON_MODEL_BACKEND` to restrict validation to a single backend's artifact type. When unset, the validation checks for any recognized artifact. When set, only the specified backend's artifact is checked in each version directory.

```bash
# Only validate for ONNX artifacts
TRITON_MODEL_BACKEND=onnx flox activate --start-services
```

## Configuration reference

All settings are runtime environment variables with `${VAR:-default}` fallbacks. Override any var at activation time:

```bash
TRITON_HTTP_PORT=9000 TRITON_LOG_VERBOSE=1 flox activate --start-services
```

### Global settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_VERBOSITY` | `1` | Script log verbosity. `0` = quiet, `1` = normal, `2` = verbose |

### Network settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_HOST` | `0.0.0.0` | Bind address for preflight port checks. Passed to OpenAI frontend via `--host` when `TRITON_OPENAI_FRONTEND=true`. Not passed to `tritonserver` in standard mode (use `--` passthrough for `--http-address` / `--grpc-address`) |
| `TRITON_HTTP_PORT` | `8000` | HTTP API port. Must be 1-65535 |
| `TRITON_GRPC_PORT` | `8001` | gRPC API port. Must be 1-65535 |
| `TRITON_METRICS_PORT` | `8002` | Prometheus metrics port. Must be 1-65535 |

### Model settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_MODEL` | _(required)_ | Model name (directory name within the repository). Controls which model `triton-resolve-model` provisions — does **not** restrict which models tritonserver loads (see `TRITON_MODEL_CONTROL_MODE`). Must not contain `/`, `\`, or be `.`/`..` |
| `TRITON_MODEL_REPOSITORY` | _(required)_ | Base model repository path. Created automatically if missing |
| `TRITON_MODEL_ID` | _(unset)_ | Explicit HuggingFace model ID (`org/repo`) for hf-hub source |
| `TRITON_MODEL_ORG` | _(unset)_ | HF org prefix. Used to derive model ID as `$TRITON_MODEL_ORG/$TRITON_MODEL` |
| `TRITON_MODEL_BACKEND` | _(unset)_ | Backend hint: `tensorrt`, `tensorrtllm`, `onnx`, `pytorch`, `tensorflow`, `python`, `vllm`. Restricts artifact validation |
| `TRITON_MODEL_SOURCES` | `flox,local,r2,hf-hub` | Comma-separated source chain. Available: `flox`, `local`, `hf-cache`, `r2`, `hf-hub` |
| `TRITON_LOCAL_MODELS` | _(unset)_ | Path to a directory of external TRT-LLM models. Each subdirectory should contain `engine/` and `tokenizer/`. Assembled as Tier 3 by `triton-setup-models`; ensemble pipelines are synthesized automatically |
| `TRITON_MODEL_ENV_FILE` | _(derived)_ | Override env file path. Default: `$FLOX_ENV_CACHE/triton-model.<slug>.<hash>.env` |

### Server settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_MODEL_CONTROL_MODE` | `none` | How tritonserver manages models at startup. `none` (default): loads **all** subdirectories in the model repository. `explicit`: loads nothing until `--load-model=<name>` is passed as an extra arg to `triton-serve`. `poll`: loads all initially, then watches for additions/changes |
| `TRITON_LOG_VERBOSE` | `0` | Tritonserver log verbosity level. Non-negative integer |
| `TRITON_STRICT_READINESS` | `true` | Require all models ready for health check. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_HTTP` | `true` | Enable HTTP endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_GRPC` | `true` | Enable gRPC endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_METRICS` | `true` | Enable Prometheus metrics endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_BACKEND_DIR` | _(set by on-activate hook)_ | Backend library directory. Automatically set to `$FLOX_ENV_CACHE/backends` by the `triton-setup-backends` hook. Passed as `--backend-directory` to tritonserver. Must exist as a directory |
| `TRITON_BACKEND_CONFIG` | _(unset)_ | Comma-separated backend configs. Format: `backend:key=val,backend:key=val` |

### OpenAI frontend settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_OPENAI_FRONTEND` | `false` | Enable the OpenAI-compatible frontend mode. When `true`, `triton-serve` execs `python3 main.py` instead of `tritonserver`. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_OPENAI_PORT` | `9000` | Port for the OpenAI-compatible frontend. Must be a positive integer |
| `TRITON_OPENAI_MAIN` | _(auto-discovered)_ | Path to `main.py`. Auto-searches `/opt/tritonserver/python/openai/main.py` and relative to the `tritonserver` binary. Set explicitly for non-standard installs |
| `TRITON_OPENAI_TOKENIZER` | _(unset)_ | HuggingFace tokenizer for chat template rendering (e.g., `meta-llama/Llama-3-8B`). Required for chat completions |

### Pre-flight settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_DRY_RUN` | `0` | Report what would happen without sending signals. `0` or `1` |
| `TRITON_PREFLIGHT_JSON` | `0` | Machine-readable JSON on stdout. Incompatible with downstream command. `0` or `1` |
| `TRITON_OWNER_REGEX` | _(built-in heuristic)_ | Regex to identify tritonserver processes. Matched against cmdline and exe |
| `TRITON_ALLOW_KILL_OTHER_UID` | `0` | Allow killing tritonserver owned by other UIDs. `0` or `1` |
| `TRITON_SKIP_GPU_CHECK` | `0` | Skip all GPU checks. `0` or `1` |
| `TRITON_GPU_WARN_PCT` | `50` | Warn if GPU memory usage exceeds this percentage. Numeric, 0-100 |
| `TRITON_TERM_GRACE` | `3` | Seconds to wait after SIGTERM before SIGKILL. Numeric, >= 0 |
| `TRITON_PORT_FREE_TIMEOUT` | `10` | Seconds to wait for ports to free after killing. Numeric, >= 0 |
| `TRITON_PREFLIGHT_LOCKFILE` | `/tmp/triton-preflight.lock` | Lock file path |

### Resolve settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_RESOLVE_LOCK_TIMEOUT` | `300` | Seconds to wait for the per-model lock |
| `TRITON_RESOLVE_NETWORK_RETRIES` | `3` | Retry attempts per network operation |
| `TRITON_RESOLVE_NETWORK_TIMEOUT` | `900` | Per-attempt timeout in seconds |
| `TRITON_RESOLVE_RETRY_SLEEP` | `2` | Sleep seconds between retries |
| `TRITON_KEEP_LOGS` | `0` | `1` to keep download logs on success (always kept on failure) |
| `TRITON_MODEL_STATE_DIR` | _(auto)_ | State directory for env/lock/provenance files. Fallback chain: env file directory → `$FLOX_ENV_CACHE` → `${XDG_CACHE_HOME:-$HOME/.cache}/triton-resolve` |
| `TRITON_DEEP_VALIDATE` | `1` | Run deeper validation (ONNX integrity, PyTorch format, TensorFlow structure, Python syntax, vLLM JSON) |
| `TRITON_STRICT_DEEP_VALIDATION` | `0` | Fail if a deep validator is unavailable (e.g., `onnx` Python module not installed) |
| `TRITON_PUBLISH_STRATEGY` | `symlink-store` | Publishing strategy: `symlink-store` or `replace-dir` |
| `TRITON_PUBLISH_MIGRATE_TARGET_DIR` | `0` | Allow one-time migration from a plain directory target to symlink-store |
| `TRITON_HF_CACHE_DIR` | _(unset)_ | Explicit HF cache root for the `hf-cache` source |
| `TRITON_MODEL_LOADABILITY_CHECK` | _(unset)_ | Shell command for a custom loadability probe after validation |
| `TRITON_STRICT_LOADABILITY_CHECK` | `0` | Fail if the loadability probe fails (`0` = warn only) |

### Env file settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_ENV_FILE_TRUSTED` | `false` | Skip safe-mode parsing and `source` the file directly. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `FLOX_ENV_CACHE` | _(set by Flox)_ | Cache directory for env files. Required when `TRITON_MODEL_ENV_FILE` is not set |
| `FLOX_ENV` | _(set by Flox)_ | Flox environment path. Required for `flox` source |

### R2 (S3-compatible) settings

| Variable | Default | Description |
|----------|---------|-------------|
| `R2_BUCKET` | _(unset)_ | Cloudflare R2 / S3-compatible bucket name |
| `R2_MODELS_PREFIX` | _(unset)_ | Key prefix for models within the bucket |
| `R2_ENDPOINT_URL` | _(unset)_ | AWS CLI endpoint URL for R2 / S3-compatible storage |

## Model provisioning (triton-resolve-model)

Searches configured sources in order, validates the model repository layout, and writes an env file that `triton-serve` loads. The first source that produces a valid model directory wins.

### Source chain

Sources are tried in the order specified by `TRITON_MODEL_SOURCES`. The default chain is `flox,local,r2,hf-hub`. The `hf-cache` source is available but not in the default chain -- add it explicitly if your models are cached from previous HuggingFace Hub downloads.

### Source table

| Source | What it checks | Skip condition | Resolution |
|--------|---------------|----------------|------------|
| `flox` | `$FLOX_ENV/share/models/$TRITON_MODEL/` | `FLOX_ENV` not set | Sets repository to `$FLOX_ENV/share/models` |
| `local` | `$TRITON_MODEL_REPOSITORY/$TRITON_MODEL/` | Missing or fails layout validation | Sets path to existing local directory |
| `r2` | Downloads from `s3://$R2_BUCKET/$R2_MODELS_PREFIX/$TRITON_MODEL/` | `aws` CLI missing, R2 vars not set, credentials invalid | Stages to temp dir, validates layout, atomic-swaps into repository |
| `hf-hub` | Downloads from HuggingFace Hub | No model ID derivable, no download tool | Stages to temp dir, validates layout, atomic-swaps into repository |
| `hf-cache` | Scans standard HF cache locations for `models--<slug>/snapshots/` (see [HF cache source](#hf-cache-source)) | No model ID derivable, no usable snapshot | Sets path to newest valid snapshot |

### Model repository validation

The `_validate_model_repo` function checks every candidate directory:

1. Model directory exists and is listable.
2. At least one numeric version directory (e.g., `1/`) is present.
3. Every version directory contains a recognized artifact for the target backend.
4. When `TRITON_MODEL_BACKEND` is set, only that backend's artifact is checked.
5. TensorFlow `model.savedmodel/` must contain `saved_model.pb`.

The function returns a JSON object with fields: `valid`, `versions`, `backends_detected`, `has_config`, and `error` (on failure).

### Deep validation

When `TRITON_DEEP_VALIDATE=1` (the default), the resolver runs backend-specific integrity checks on each artifact after basic layout validation passes:

| Backend | Check |
|---------|-------|
| `onnx` | `onnx.load()` + `onnx.checker.check_model()`. If the `onnx` Python module is not installed the check is skipped with a warning (or fails if `TRITON_STRICT_DEEP_VALIDATION=1`) |
| `pytorch` | Reads the first bytes to identify zip (TorchScript) or pickle (`\x80`) format. Warns if the format is unrecognized but does not fail |
| `tensorflow` | Verifies `saved_model.pb` exists and is non-empty inside `model.savedmodel/` |
| `python` | `compile(source, path, 'exec')` syntax check and UTF-8 validation |
| `vllm` | `json.load()` parses `model.json` and checks that the top-level value is a JSON object |
| `tensorrt` | Verifies `model.plan` is non-empty. No deeper check (final loadability depends on runtime TensorRT compatibility) |

Set `TRITON_STRICT_DEEP_VALIDATION=1` to fail resolution when a requested deep validator is unavailable (currently only affects the ONNX check). With the default `0`, an unavailable validator emits a warning and the artifact is accepted.

### Loadability checks

After validation, an optional custom probe can be run via `TRITON_MODEL_LOADABILITY_CHECK`. Set it to a shell command that will be executed with:

| Env var passed to probe | Value |
|-------------------------|-------|
| `TRITON_VALIDATE_MODEL_DIR` | Path to the validated model directory |
| `TRITON_VALIDATE_MODEL` | Model name (`$TRITON_MODEL`) |
| `TRITON_VALIDATE_BACKEND` | Comma-separated detected backends |

If the command exits `0`, the model is accepted. If it exits non-zero:

- **`TRITON_STRICT_LOADABILITY_CHECK=0`** (default): a warning is logged and resolution continues.
- **`TRITON_STRICT_LOADABILITY_CHECK=1`**: resolution fails with the probe's stderr/stdout as the error message.

```bash
# Example: reject models larger than 10 GB
TRITON_MODEL_LOADABILITY_CHECK='test $(du -sb "$TRITON_VALIDATE_MODEL_DIR" | cut -f1) -lt 10737418240' \
  flox activate --start-services
```

### HuggingFace download

The download tool cascade tries three methods in order:

1. `hf` CLI (`hf download <repo_id> --local-dir <dir>`)
2. `huggingface-cli` (`huggingface-cli download <repo_id> --local-dir <dir>`)
3. Python `huggingface_hub` (`snapshot_download()`)

If none are available, the source fails with exit code 127.

### Env file output

Written atomically (mktemp + mv) with mode `600` (umask `077`). Contains:

```bash
# generated by triton-resolve-model
export TRITON_MODEL='my-onnx-model'
export TRITON_MODEL_REPOSITORY='/data/models'
export _TRITON_RESOLVED_PATH='/data/models/my-onnx-model'
export _TRITON_RESOLVED_VIA='local'
export _TRITON_BACKENDS_DETECTED='onnx'
export _TRITON_VERSIONS='1,2'
```

### Offline operation

Restrict sources to avoid network access:

```bash
TRITON_MODEL_SOURCES=local flox activate --start-services           # local only
TRITON_MODEL_SOURCES=local,hf-cache flox activate --start-services  # local + cached
```

### Locking and atomic swap

- **Per-model lock**: acquired before any source search. Lock file: `$TRITON_MODEL_STATE_DIR/triton-model.<slug>.<hash>.lock`. Timeout: `TRITON_RESOLVE_LOCK_TIMEOUT` seconds (default 300). Locking is performed by a background Python helper that uses `fcntl.flock()` with bounded polling (50 ms intervals) and sets `PR_SET_PDEATHSIG` so the lock is released if the parent shell dies. Symlink and regular-file checks are enforced before opening.
- **Atomic swap** (r2 and hf-hub only): downloads stage into a temp directory under `$TRITON_MODEL_STATE_DIR/.staging/`. After layout validation, the staged directory is published via the configured `TRITON_PUBLISH_STRATEGY` (see [Publishing strategies](#publishing-strategies)). If a `replace-dir` swap is interrupted, `lib::restore_backup` recovers the most recent backup on the next run.
- **Staging cleanup**: staging directories and download logs are cleaned up on success. On failure, logs are preserved for debugging.

### Publishing strategies

After a model is downloaded and validated, it is published into the target directory using one of two strategies controlled by `TRITON_PUBLISH_STRATEGY`:

**`symlink-store`** (default): model content is stored by content manifest SHA under `$TRITON_MODEL_STATE_DIR/.published/<model>/<sha>/`. The target directory (`$TRITON_MODEL_REPOSITORY/$TRITON_MODEL`) becomes a relative symlink pointing into the store. Benefits:

- **Deduplication**: identical content (same SHA) is stored once regardless of how many times it is downloaded.
- **Atomic symlink swap**: updating the target is a single `mv -T` of a temporary symlink, so readers never see a partially-written directory.
- **No same-device constraint**: the store and target can be on different filesystems because the swap operates on a symlink, not a directory rename.

**`replace-dir`**: the staged directory replaces the target directly via `mv -T` with an automatic backup/rollback mechanism. Requirements:

- Staged and target directories must be on the **same filesystem** (`mv -T` requires it).
- Benefits: no symlink indirection, simpler directory layout.

**Migration**: if you switch an existing deployment from `replace-dir` to `symlink-store`, the target may already be a plain directory. Set `TRITON_PUBLISH_MIGRATE_TARGET_DIR=1` for a one-time run to move the existing directory aside and replace it with a symlink. After migration, unset the variable.

### Provenance and no-op detection

Each successful resolution writes a provenance file at `$TRITON_MODEL_STATE_DIR/triton-model.<slug>.<hash>.provenance.json` containing:

| Field | Description |
|-------|-------------|
| `source` | Source that resolved the model (`flox`, `local`, `r2`, `hf-hub`, `hf-cache`) |
| `source_identity` | Model ID, path, or bucket key used |
| `requested_revision` | Revision requested (branch/tag), if any |
| `resolved_revision` | Actual commit or revision resolved, if any |
| `remote_manifest_sha` | Content SHA of the remote listing (R2), if available |
| `local_manifest_sha` | Content manifest SHA of the local directory tree |
| `resolved_repository` | Model repository base path |
| `resolved_path` | Full path to the resolved model directory |
| `backends_detected` | List of detected backends |
| `versions` | List of numeric version directories found |
| `recorded_at` | ISO 8601 UTC timestamp |

The content manifest SHA is computed by walking the directory tree and hashing every file's content, size, and relative path.

**No-op detection**: on re-run, if the provenance file exists and the current local manifest SHA matches the recorded provenance (plus source, identity, and revision where applicable), the remote download is skipped entirely. This makes repeated `flox activate --start-services` cycles fast when the model has not changed.

### R2 (S3-compatible storage)

R2 downloads use `aws s3 sync` to fetch the model directory from `s3://$R2_BUCKET/$R2_MODELS_PREFIX/$TRITON_MODEL/` into a staging directory. Requirements:

- `aws` CLI must be on PATH.
- `R2_BUCKET` and `R2_MODELS_PREFIX` must both be set.
- `R2_ENDPOINT_URL` is passed as `--endpoint-url` when set.
- AWS credentials must be valid (`aws sts get-caller-identity` is checked before download).

The download is staged, validated, and then atomically swapped into the target directory.

### HF cache source

When `hf-cache` is in the source chain, the script searches standard HuggingFace cache locations for a usable snapshot. Cache roots are tried in order:

1. `$TRITON_HF_CACHE_DIR` (explicit override)
2. `$HF_HUB_CACHE`
3. `$HUGGINGFACE_HUB_CACHE`
4. `$HF_HOME/hub`
5. `$XDG_CACHE_HOME/huggingface/hub`
6. `~/.cache/huggingface/hub`

Within each cache root, the script scans `models--<slug>/snapshots/` for valid model layouts. Snapshots are checked newest-first (by modification time). The slug is derived from the model ID by replacing `/` with `--`. This source requires `TRITON_MODEL_ID` or `TRITON_MODEL_ORG` to be set.

## Pre-flight (triton-preflight)

Multi-port reclaim, GPU health check, and optional downstream command execution. Linux only (requires `/proc`).

### Usage

```bash
triton-preflight                              # checks only
triton-preflight ./start.sh arg1 arg2         # checks, then exec command
triton-preflight -- triton-serve --print-cmd  # checks, then exec command (after --)
```

### Exit codes

Stable contract -- safe to match on programmatically.

| Code | Meaning | When |
|------|---------|------|
| `0` | Success | Ports free (or reclaimed), GPU OK, downstream command exec'd |
| `1` | Validation error | Bad env var, GPU hard failure (no CUDA), bad config, missing `python3` |
| `2` | Port owned by non-Triton process | A non-tritonserver listener holds one or more ports. Will not kill |
| `3` | Different UID | Tritonserver on the port belongs to another user. Will not kill (unless `TRITON_ALLOW_KILL_OTHER_UID=1`) |
| `4` | Not attributable | Listener found but cannot map socket inodes to PIDs (permissions / `hidepid`) |
| `5` | Stop failed | Sent SIGTERM/SIGKILL but port(s) still listening after timeout |
| `6` | Partial port reclaim | Some ports reclaimable (Triton), others blocked (non-Triton). Mixed ownership |

In dry-run mode (`TRITON_DRY_RUN=1`), exit code `5` cannot occur since no processes are killed. Exit code `6` can still occur (it is a classification result, not a kill action).

### Multi-port reclaim behavior

1. **Single-pass scan**: Parses `/proc/net/tcp` and `/proc/net/tcp6` for LISTEN-state sockets matching all configured ports (HTTP, gRPC, metrics, and OpenAI when `TRITON_OPENAI_FRONTEND=true`) simultaneously.
2. **Target resolution**: Resolves the bind address to IPv4/IPv6 targets, including wildcard (`0.0.0.0`/`::`) catchall matching.
3. **Inode mapping**: Maps socket inodes to PIDs via `/proc/<pid>/fd/` symlink scanning.
4. **Unmappable inodes**: If any inodes cannot be mapped, exits with code 4 and reports affected ports.
5. **Process classification**: Reads `/proc/<pid>/cmdline` and `/proc/<pid>/exe` for each listener PID. Matches against `tritonserver` (built-in heuristic) or `TRITON_OWNER_REGEX` (custom).
6. **Non-Triton listener**: If non-tritonserver processes hold ports exclusively, exits with code 2. If mixed (some ports Triton, some not), exits with code 6.
7. **UID check**: If tritonserver belongs to a different UID, exits with code 3 unless `TRITON_ALLOW_KILL_OTHER_UID=1`.
8. **Kill tree**: Walks the process tree (children via `/proc/<pid>/stat`) in post-order. Sends SIGTERM, waits `TRITON_TERM_GRACE` seconds, then SIGKILL survivors.
9. **Port wait**: Polls until all reclaimed ports are free or `TRITON_PORT_FREE_TIMEOUT` expires. On timeout, exits with code 5.

### GPU health check

Runs after port reclaim. Two detection paths:

1. **PyTorch** (preferred): If `import torch` succeeds, uses `torch.cuda.is_available()`, `torch.cuda.device_count()`, and `torch.cuda.mem_get_info()` to report per-GPU status. Exits with code 1 if no CUDA GPUs are available.
2. **nvidia-smi** (fallback): If PyTorch is unavailable but `nvidia-smi` is on PATH, queries GPU name, total memory, and free memory. Soft-skips with a warning on failure.
3. **Neither available**: Logs a warning and continues.

In all cases, a warning is emitted if any GPU's memory usage exceeds `TRITON_GPU_WARN_PCT` (default 50%).

### JSON output mode

When `TRITON_PREFLIGHT_JSON=1`, a single JSON object is printed to stdout. Human-readable logs still go to stderr. Incompatible with downstream command execution.

Examples:

```json
{"status":"ok","action":"noop","dry_run":false,"ports":[8000,8001,8002]}
{"status":"ok","action":"stopped","dry_run":false,"pids":[12345],"ports":[8000,8001,8002]}
{"status":"ok","action":"would_stop","dry_run":true,"pids":[12345]}
```

### Downstream command execution

When positional arguments are provided (with or without a leading `--`), they are executed via `exec` after all checks pass. `triton-serve` uses this internally to delegate to `triton-preflight` before launching the server.

```bash
# Standalone usage with a downstream command:
triton-preflight -- ./start.sh arg1 arg2
```

`TRITON_PREFLIGHT_JSON=1` is incompatible with downstream commands because stdout must remain JSON-only.

### Locking

Acquired via `flock` (from `util-linux`, installed in the manifest) on `TRITON_PREFLIGHT_LOCKFILE` (default `/tmp/triton-preflight.lock`) with a 10-second timeout. Prevents concurrent preflight runs from racing. The lockfile is validated: symlinks are rejected, and only regular files are accepted. Port scanning uses `ss` (from `iproute2`, also in the manifest) for fast PID-to-port mapping, with a `/proc/net/tcp` fallback.

## Serving (triton-serve)

Loads the resolved model env file, validates configuration, runs `triton-preflight` for port reclaim and GPU health checks, then executes `tritonserver` (default) or the OpenAI-compatible frontend (`python3 main.py`) when `TRITON_OPENAI_FRONTEND=true`. The built-in preflight step is controlled by `TRITON_SERVE_RUN_PREFLIGHT` (default `true`).

### Usage

```bash
triton-serve                           # standard launch
triton-serve --print-cmd               # print the tritonserver argv to stderr, then exec
triton-serve --dry-run                 # print the argv and exit 0 (no exec)
triton-serve -h                        # show help
triton-serve -- --extra-flag val       # pass extra args through to tritonserver
```

### Env file loading

Two modes:

**Safe mode** (default): Parsed by a Python script enforcing a restricted `.env` subset -- `KEY=VALUE` or `export KEY=VALUE`, optional single/double quotes, escape sequences in double quotes. No shell interpolation or command substitution. Requires `python3` on PATH.

**Trusted mode** (`TRITON_ENV_FILE_TRUSTED=true`): `source`d directly as shell code. Only enable this for env files you control.

The env file must define `_TRITON_RESOLVED_PATH` or `triton-serve` exits with an error. Both the model repository and the resolved path must exist as directories.

**Legacy path fallback**: If the hashed env file path does not exist, `triton-serve` falls back to a legacy slug-only path (`triton-model.<slug>.env`).

### Command construction

#### Standard mode (default)

`triton-serve` builds the final argv as:

```bash
tritonserver \
  --model-repository=<TRITON_MODEL_REPOSITORY> \
  --http-port=<TRITON_HTTP_PORT> \
  --grpc-port=<TRITON_GRPC_PORT> \
  --metrics-port=<TRITON_METRICS_PORT> \
  --model-control-mode=<TRITON_MODEL_CONTROL_MODE> \
  --strict-readiness=<TRITON_STRICT_READINESS> \
  --log-verbose=<TRITON_LOG_VERBOSE> \
  [--backend-directory=<TRITON_BACKEND_DIR>] # if TRITON_BACKEND_DIR is set
  [--allow-http=false]                    # if TRITON_ALLOW_HTTP is falsy
  [--allow-grpc=false]                    # if TRITON_ALLOW_GRPC is falsy
  [--allow-metrics=false]                 # if TRITON_ALLOW_METRICS is falsy
  [--backend-config=<spec> ...]           # for each entry in TRITON_BACKEND_CONFIG
  [extra args...]                         # anything after -- on the triton-serve command line
```

The env-var-to-CLI-flag mapping:

| Env var | CLI flag | Condition |
|---------|----------|-----------|
| `TRITON_MODEL_REPOSITORY` | `--model-repository` | Always |
| `TRITON_HTTP_PORT` | `--http-port` | Always |
| `TRITON_GRPC_PORT` | `--grpc-port` | Always |
| `TRITON_METRICS_PORT` | `--metrics-port` | Always |
| `TRITON_MODEL_CONTROL_MODE` | `--model-control-mode` | Always |
| `TRITON_STRICT_READINESS` | `--strict-readiness` | Always |
| `TRITON_LOG_VERBOSE` | `--log-verbose` | Always |
| `TRITON_BACKEND_DIR` | `--backend-directory` | When set |
| `TRITON_ALLOW_HTTP` | `--allow-http=false` | When falsy |
| `TRITON_ALLOW_GRPC` | `--allow-grpc=false` | When falsy |
| `TRITON_ALLOW_METRICS` | `--allow-metrics=false` | When falsy |
| `TRITON_BACKEND_CONFIG` | `--backend-config` | When set (one flag per entry) |

#### OpenAI frontend mode (`TRITON_OPENAI_FRONTEND=true`)

When the OpenAI-compatible frontend is enabled, `triton-serve` execs `python3 main.py` instead of `tritonserver`. The OpenAI frontend is a FastAPI/Uvicorn application that embeds Triton in-process via Python bindings -- it replaces the standalone `tritonserver` binary. It ships in official Triton Docker containers at `/opt/tritonserver/python/openai/`.

```bash
python3 <TRITON_OPENAI_MAIN> \
  --model-repository=<TRITON_MODEL_REPOSITORY> \
  --openai-port=<TRITON_OPENAI_PORT> \
  --host=<TRITON_HOST> \
  [--tokenizer=<TRITON_OPENAI_TOKENIZER>]                   # when set
  [--tritonserver-log-verbose-level=<TRITON_LOG_VERBOSE>]    # when > 0
  [--enable-kserve-frontends]                                # when HTTP or gRPC enabled
  [--kserve-http-port=<TRITON_HTTP_PORT>]                    # with kserve frontends
  [--kserve-grpc-port=<TRITON_GRPC_PORT>]                    # with kserve frontends
  [extra args...]                                            # anything after --
```

| Env var | CLI flag | Condition |
|---------|----------|-----------|
| `TRITON_MODEL_REPOSITORY` | `--model-repository` | Always |
| `TRITON_OPENAI_PORT` | `--openai-port` | Always |
| `TRITON_HOST` | `--host` | Always |
| `TRITON_OPENAI_TOKENIZER` | `--tokenizer` | When set |
| `TRITON_LOG_VERBOSE` | `--tritonserver-log-verbose-level` | When > 0 |
| `TRITON_ALLOW_HTTP` / `TRITON_ALLOW_GRPC` | `--enable-kserve-frontends` | When either is truthy |
| `TRITON_HTTP_PORT` | `--kserve-http-port` | With kserve frontends |
| `TRITON_GRPC_PORT` | `--kserve-grpc-port` | With kserve frontends |

When `--enable-kserve-frontends` is passed, the OpenAI frontend also serves KServe HTTP and gRPC, so all three interfaces (OpenAI port 9000, KServe HTTP port 8000, KServe gRPC port 8001) run from a single process.

The `tritonserver` binary is not required on PATH in OpenAI frontend mode.

### Backend configuration

`TRITON_BACKEND_CONFIG` accepts a comma-separated list of `backend:key=val` entries. Each entry becomes a separate `--backend-config` flag.

```bash
# Configure TensorRT and Python backends
TRITON_BACKEND_CONFIG="tensorrt:coalesced-memory-size=256,python:shm-default-byte-size=1048576" \
  flox activate --start-services

# Results in:
#   --backend-config=tensorrt:coalesced-memory-size=256
#   --backend-config=python:shm-default-byte-size=1048576
```

### Validation

All checks performed before exec:

- `tritonserver` must be on PATH (skipped when `TRITON_OPENAI_FRONTEND=true`).
- Env file must exist, be readable, and set `_TRITON_RESOLVED_PATH`.
- `TRITON_MODEL_REPOSITORY` must be set and exist as a directory.
- `_TRITON_RESOLVED_PATH` must exist as a directory.
- `TRITON_HTTP_PORT`, `TRITON_GRPC_PORT`, `TRITON_METRICS_PORT` must be positive integers.
- `TRITON_LOG_VERBOSE` must be a non-negative integer.
- `TRITON_MODEL_CONTROL_MODE` must be `none`, `explicit`, or `poll`.
- `TRITON_STRICT_READINESS`, `TRITON_ALLOW_HTTP`, `TRITON_ALLOW_GRPC`, `TRITON_ALLOW_METRICS` must be valid boolean values.
- `TRITON_OPENAI_FRONTEND` must be a valid boolean value.
- When `TRITON_OPENAI_FRONTEND=true`: `TRITON_OPENAI_PORT` must be a positive integer, and `TRITON_OPENAI_MAIN` must point to an existing file (auto-discovered if not set).

## Multi-GPU

Triton handles GPU assignment through model `config.pbtxt` instance groups, not through runtime env vars. To restrict which GPUs are visible:

```bash
CUDA_VISIBLE_DEVICES=0,1 flox activate --start-services
```

For per-model GPU placement, configure `instance_group` in `config.pbtxt`:

```
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

See the [Triton Model Configuration documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md) for instance group details.

## Swapping models

Override the model at activation time:

```bash
TRITON_MODEL=resnet50 \
TRITON_MODEL_REPOSITORY=/data/models \
TRITON_MODEL_BACKEND=onnx \
  flox activate --start-services
```

**Important:** With the default `TRITON_MODEL_CONTROL_MODE=none`, tritonserver
loads every model subdirectory in `TRITON_MODEL_REPOSITORY` at startup — not
just the one named by `TRITON_MODEL`. To load **only** the specified model, use
`explicit` mode with `--load-model`:

```bash
TRITON_MODEL=qwen3_8b \
TRITON_MODEL_CONTROL_MODE=explicit \
TRITON_MODEL_SOURCES=local \
  flox activate -- bash -c 'triton-resolve-model && triton-serve -- --load-model=qwen3_8b'
```

For hot-swapping without restart, use `poll` mode:

```bash
TRITON_MODEL_CONTROL_MODE=poll flox activate --start-services
# Now copy new model versions into the repository; Triton picks them up automatically
```

To restart with a different model:

```bash
flox services restart tritonserver
```

## Service management

```bash
flox services status                  # check service state
flox services logs tritonserver       # tail service logs
flox services logs tritonserver -f    # follow logs
flox services restart tritonserver    # restart the tritonserver service
flox services stop                    # stop all services
flox activate --start-services        # activate and start in one step
```

## Troubleshooting

Common issues and their solutions. Exit codes refer to `triton-preflight`.

### Port conflict (exit code 2)

`triton-preflight` automatically reclaims ports from stale tritonserver processes. If it exits with code 2, a non-tritonserver process is using one or more of the configured ports.

```bash
# Find what is on the ports
ss -tlnp | grep -E ':(8000|8001|8002)\b'

# Either stop that process or change the ports
TRITON_HTTP_PORT=9000 \
TRITON_GRPC_PORT=9001 \
TRITON_METRICS_PORT=9002 \
  flox activate --start-services
```

### Partial port reclaim (exit code 6)

Some ports are held by tritonserver (reclaimable) and others by non-Triton processes (blocked). This mixed-ownership situation requires manual intervention: stop the non-Triton processes or change ports to avoid the conflict.

### Different UID (exit code 3)

Another user's tritonserver holds one or more ports:

```bash
TRITON_ALLOW_KILL_OTHER_UID=1 flox activate --start-services
```

### Unattributable listener (exit code 4)

A listener was found but the script could not map socket inodes to PIDs. This typically happens when `/proc/<pid>/fd` visibility is restricted (e.g., `hidepid=2` mount option on `/proc`).

Solutions:
- Run as the same user that owns the listener.
- Adjust `/proc` mount options (`hidepid`).
- Run with elevated permissions.

### Stop failed (exit code 5)

Tritonserver was identified and signaled but the ports are still listening after `TRITON_PORT_FREE_TIMEOUT` seconds.

```bash
# Increase timeouts
TRITON_TERM_GRACE=10 TRITON_PORT_FREE_TIMEOUT=30 flox activate --start-services
```

If the process is a zombie or unkillable, manual intervention is required (`kill -9 <pid>`).

### GPU not detected

Verify GPU visibility:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

To skip the GPU check entirely:

```bash
TRITON_SKIP_GPU_CHECK=1 flox activate --start-services
```

### Model validation failure

Common layout mistakes:

- Missing numeric version directory (e.g., model files placed directly in the model directory instead of `1/`).
- Wrong artifact filename (e.g., `model.onnx` for a PyTorch model).
- TensorFlow `model.savedmodel/` missing `saved_model.pb` inside it.
- `TRITON_MODEL_BACKEND` set to the wrong backend.

Diagnostic steps:

```bash
# Check the directory structure
find $TRITON_MODEL_REPOSITORY/$TRITON_MODEL -type f

# Run resolve with verbose logging
TRITON_VERBOSITY=2 triton-resolve-model
```

### Gated model 401

Gated HuggingFace models require authentication:

```bash
HF_TOKEN=hf_... flox activate --start-services
```

### R2 download failure

Common R2 issues:

- `aws` CLI not installed or not on PATH.
- `R2_BUCKET` or `R2_MODELS_PREFIX` not set.
- Invalid AWS/R2 credentials (`aws sts get-caller-identity` fails).
- Wrong `R2_ENDPOINT_URL` for the storage provider.

Check staging logs (preserved on failure) at `$TRITON_MODEL_STATE_DIR/.staging/`.

### OpenAI frontend dependency installation

The OpenAI frontend's Python dependencies (FastAPI, Uvicorn, etc.) are installed automatically on first `flox activate` into `$FLOX_ENV_CACHE/openai-deps/` using `uv pip install --target`. A sentinel file (`.installed-v1`) prevents re-installation on subsequent activations. To force a reinstall:

```bash
rm -rf "$FLOX_ENV_CACHE/openai-deps"
```

The `tritonserver` and `tritonfrontend` wheels shipped in the `triton-server` package are also installed into this target directory.

### OpenAI frontend main.py not found

The OpenAI frontend auto-discovers `main.py` relative to the `tritonserver` binary (`$(dirname $(which tritonserver))/../python/openai/main.py`). The `triton-server` package bundles the frontend source at this path. If discovery fails, set the path explicitly:

```bash
TRITON_OPENAI_MAIN=/path/to/openai/main.py flox activate --start-services
```

### Chat completions return empty

`TRITON_OPENAI_TOKENIZER` is required for chat completions. Set it to the HuggingFace tokenizer that matches your model:

```bash
TRITON_OPENAI_TOKENIZER=meta-llama/Llama-3-8B flox activate --start-services
```

### Connection refused on port 9000

Verify that `TRITON_OPENAI_FRONTEND=true` is set. Without it, `triton-serve` launches `tritonserver` which does not serve the OpenAI-compatible API on port 9000.

### Stale lock

If a previous run was killed mid-operation:

```bash
# For triton-preflight
rm -f /tmp/triton-preflight.lock

# For triton-resolve-model (per-model lock)
# Default state dir is $FLOX_ENV_CACHE, or ${XDG_CACHE_HOME:-$HOME/.cache}/triton-resolve
rm -f "$TRITON_MODEL_STATE_DIR"/triton-model.*.lock
```

### Inspecting the generated command

```bash
triton-serve --print-cmd   # print the tritonserver argv to stderr, then run it
triton-serve --dry-run     # print the argv and exit without running
```

### Passing extra tritonserver flags

Any flags not covered by env vars can be passed through:

```bash
triton-serve -- --buffer-manager-thread-count 8 --pinned-memory-pool-byte-size 268435456
```

## Backends

Triton backends are loaded from the directory specified by `TRITON_BACKEND_DIR`. Each
backend is a subdirectory containing a shared library (`libtriton_<name>.so`).

### Built-from-source backends

The server and compiled backends (Python, ONNX Runtime, TensorRT, TensorRT-LLM) are
built from source (or extracted from NGC containers) via Nix expressions in a separate
build repository ([build-triton-server](../builds/build-triton-server/)). The resulting
packages are published to the `flox` Flox catalog and referenced in `manifest.toml`
via `pkg-path`:

```toml
# .flox/env/manifest.toml
[install]
triton-server.pkg-path = "flox/triton-server"
triton-python-backend.pkg-path = "flox/triton-python-backend"
triton-onnxruntime-backend.pkg-path = "flox/triton-onnxruntime-backend"
triton-tensorrt-backend.pkg-path = "flox/triton-tensorrt-backend"
triton-tensorrtllm-backend.pkg-path = "flox/triton-tensorrtllm-backend"

# Model packages (optional — Tier 1 model discovery by triton-setup-models)
triton-model-qwen2-5-05b-trtllm.pkg-path = "flox/triton-model-qwen2-5-05b-trtllm"
triton-model-qwen2-5-05b-trtllm.pkg-group = "trtllm-models"
```

### Nixpkgs-provided packages

Some backends and their dependencies are available pre-built from nixpkgs (via the
`flox-cuda` channel) and require no custom Nix build expressions:

```toml
# .flox/env/manifest.toml
[install]
vllm.pkg-path = "flox-cuda/python3Packages.vllm"
vllm.systems = ["x86_64-linux"]
vllm.pkg-group = "vllm"
```

The vLLM engine (v0.15.1) is installed this way. The vLLM *backend* itself is pure
Python — source files from the
[vllm_backend](https://github.com/triton-inference-server/vllm_backend) repo are
checked directly into `backends/vllm/` with no compilation step.

### Backend directory setup

Backend assembly is fully automated. The `triton-setup-backends` script (bundled in the
`triton-server` package) runs during `flox activate` via the on-activate hook and builds
a unified backend directory at `$FLOX_ENV_CACHE/backends/`:

- **Tier 1** (package-provided): Each subdirectory in `$FLOX_ENV/backends/` is symlinked
  wholesale into the cache. This covers compiled backends installed via the Flox catalog
  (python, onnxruntime, tensorrt, tensorrtllm).
- **Tier 2** (repo-local): Each real directory in `$FLOX_ENV_PROJECT/backends/` that was
  not already handled by Tier 1 is assembled with per-file symlinks. Python-based backends
  (detected by the presence of `model.py` and absence of `libtriton_*.so`) automatically
  get `triton_python_backend_stub` and `triton_python_backend_utils.py` injected from the
  python backend package.

The hook also exports `TRITON_BACKEND_DIR`, so `triton-serve` passes
`--backend-directory` to tritonserver automatically. No manual symlink creation or env
var setup is needed.

### Available backends

| Backend | Package | Library |
|---------|---------|---------|
| Python | `flox/triton-python-backend` | `backends/python/libtriton_python.so` |
| ONNX Runtime | `flox/triton-onnxruntime-backend` | `backends/onnxruntime/libtriton_onnxruntime.so` |
| TensorRT | `flox/triton-tensorrt-backend` | `backends/tensorrt/libtriton_tensorrt.so` |
| TensorRT-LLM | `flox/triton-tensorrtllm-backend` | `backends/tensorrtllm/libtriton_tensorrtllm.so` |
| vLLM | (pure Python, repo-local) | `backends/vllm/model.py` + Python backend stub |

The ONNX Runtime backend loads `libonnxruntime.so` (ORT 1.24.2) from its Nix store
RPATH automatically -- no need to copy ORT libraries into the backend directory.
The TensorRT backend similarly loads the TRT SDK via RPATH. The TensorRT-LLM backend
bundles its own runtime libraries (TRT-LLM, CUDA 13.x, NCCL) with `$ORIGIN`-relative
RPATHs for SONAME isolation from the system's CUDA 12.x. The TRT-LLM backend also
requires MPI — the on-activate hook auto-resolves `OPAL_PREFIX` from the backend's
`hpcx/ompi/` directory and prepends its `lib/` to `LD_LIBRARY_PATH`.

### Model directory setup

Model assembly is handled by `triton-setup-models` (bundled in the `triton-server`
package), which runs during `flox activate` alongside `triton-setup-backends`. It builds
a model directory at `$FLOX_ENV_CACHE/models/`:

- **Tier 1** (package-provided): Each model directory under `$FLOX_ENV/share/models/` is
  copied into the cache. These are Nix-store model packages (e.g.,
  `triton-model-qwen2-5-05b-trtllm`) installed via the Flox catalog. If a model contains
  `config.pbtxt.template`, token placeholders are expanded and the result is written as
  `config.pbtxt`.
- **Tier 2** (repo-local): Each directory in `$FLOX_ENV_PROJECT/models/` that was not
  already handled by Tier 1 is symlinked into the cache.
- **Tier 3** (external local): Each directory under `$TRITON_LOCAL_MODELS/` that was not
  already handled by Tier 1 or Tier 2 is assembled into the cache. Intended for TRT-LLM
  models with pre-built engines stored outside the repo.

**Template token expansion**: `config.pbtxt.template` files support four tokens:

| Token | Expanded to |
|-------|-------------|
| `@EXECUTOR_WORKER_PATH@` | `$TRITON_BACKEND_DIR/tensorrtllm/trtllmExecutorWorker` |
| `@GPT_MODEL_PATH@` | `$FLOX_ENV_CACHE/models/<name>/engine` |
| `@TOKENIZER_DIR@` | `$FLOX_ENV_CACHE/models/<name>/tokenizer` |
| `@MODEL_NAME@` | Model directory name (e.g., `my_llm_trtllm`) |

**TRT-LLM ensemble synthesis**: After all tiers are assembled, `triton-setup-models`
checks each model for ensemble synthesis eligibility. A model qualifies if it has an
`engine/` directory but no sibling `_preprocessing` directory. For qualifying models,
three directories are synthesized from templates bundled in the `triton-server` package:

- `{name}_preprocessing/` — tokenizer (Python backend)
- `{name}_postprocessing/` — detokenizer (Python backend)
- `{name}_ensemble/` — BLS orchestrator with streaming support

The raw model also gets a `config.pbtxt` and `1/` version directory if missing. Synthesis
is skipped for models that already ship with an ensemble (e.g., Tier 1 packages).

The hook exports `TRITON_MODEL_REPOSITORY` pointing to the assembled model directory
if model packages are present.

### ONNX Runtime backend details

The ORT backend uses the **CUDA execution provider** by default when `instance_group`
is set to `KIND_GPU`. Models are loaded into an ORT inference session and served through
Triton's standard HTTP/gRPC/metrics endpoints.

Example `config.pbtxt` for an ONNX model:

```
name: "my_onnx_model"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [{
  name: "INPUT0"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}]

output [{
  name: "OUTPUT0"
  data_type: TYPE_FP32
  dims: [ 1, 4 ]
}]

instance_group [{
  kind: KIND_GPU
}]
```

Key fields:
- **`platform`**: Must be `"onnxruntime_onnx"` (tells Triton to use the ORT backend)
- **`instance_group.kind`**: `KIND_GPU` for CUDA execution, `KIND_CPU` for CPU-only
- **`max_batch_size: 0`**: Disables dynamic batching (set > 0 if your model supports it)
- Model artifact must be named `model.onnx` in each version directory

Verified working: ORT backend loads the CUDA execution provider, creates an inference
session, and serves models through Triton on this system (RTX 5090, driver 590.48.01).

### vLLM backend details

The vLLM backend serves large language models using [vLLM](https://github.com/vllm-project/vllm)'s
high-performance async engine. Unlike compiled backends, vLLM is a **pure Python backend** that
runs on top of Triton's Python backend infrastructure (`TritonPythonModel`).

Source files come from [triton-inference-server/vllm_backend](https://github.com/triton-inference-server/vllm_backend)
at tag `r26.02`. The vLLM engine itself is installed via `flox-cuda/python3Packages.vllm`.

**Repo source files** (checked into `backends/vllm/`):

```
backends/vllm/
  model.py                         # Main TritonPythonModel (from vllm_backend repo)
  utils/
    __init__.py
    metrics.py                     # TritonMetrics, VllmStatLogger
    request.py                     # GenerateRequest, EmbedRequest
    vllm_backend_utils.py          # TritonSamplingParams, engine client builder
```

At activation time, `triton-setup-backends` assembles the runtime directory in
`$FLOX_ENV_CACHE/backends/vllm/` with per-file symlinks to the repo source plus
`triton_python_backend_stub` and `triton_python_backend_utils.py` injected from the
python backend package.

**Model configuration** requires two files per model:

`config.pbtxt` -- Triton model config:

```
backend: "vllm"

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]

model_transaction_policy {
  decoupled: True
}

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  },
  {
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  },
  {
    name: "exclude_input_in_output"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
```

`1/model.json` -- vLLM engine arguments:

```json
{
  "model": "facebook/opt-125m",
  "enable_log_requests": false,
  "gpu_memory_utilization": 0.3,
  "enforce_eager": true
}
```

Key `model.json` fields:
- **`model`**: HuggingFace model name or local path
- **`gpu_memory_utilization`**: fraction of GPU memory to use (0.0-1.0)
- **`tensor_parallel_size`**: number of GPUs for tensor parallelism
- **`enable_log_requests`**: enable per-request logging (default: true, set false to suppress)
- **`max_model_len`**: override maximum sequence length
- **`quantization`**: quantization method (`awq`, `gptq`, `squeezellm`)
- **`enforce_eager`**: disable CUDA graphs (useful for debugging)

See the [vLLM engine arguments documentation](https://docs.vllm.ai/en/latest/serving/engine_args.html)
for the full list.

**Example inference request:**

```bash
curl -X POST http://127.0.0.1:8000/v2/models/vllm_test/generate \
  -H "Content-Type: application/json" \
  -d '{"text_input": "What is machine learning?", "parameters": {"max_tokens": 64, "stream": false}}'
```

## File structure

```
triton-runtime/
  .flox/env/manifest.toml      # Flox manifest (packages, hook, service)
  backends/                     # Repo-local backend sources
    vllm/                       # Pure Python backend (vllm_backend r26.02 sources)
      model.py                  # Main TritonPythonModel
      utils/                    # vLLM backend utilities
  # At activation, triton-setup-backends assembles $FLOX_ENV_CACHE/backends/:
  #   python/       -> $FLOX_ENV/backends/python/        (Tier 1, from catalog)
  #   onnxruntime/  -> $FLOX_ENV/backends/onnxruntime/   (Tier 1, from catalog)
  #   tensorrt/     -> $FLOX_ENV/backends/tensorrt/      (Tier 1, from catalog)
  #   tensorrtllm/  -> $FLOX_ENV/backends/tensorrtllm/   (Tier 1, from catalog)
  #   vllm/         -> per-file symlinks + python stub   (Tier 2, assembled)
  # At activation, triton-setup-models assembles $FLOX_ENV_CACHE/models/:
  #   phi4_mini_trtllm/*          -> Tier 1 (from catalog, with ensemble)
  #   vllm_test/                  -> Tier 2 (symlinked from $FLOX_ENV_PROJECT/models/)
  #   my_model_trtllm/            -> Tier 3 (from $TRITON_LOCAL_MODELS/)
  #   my_model_trtllm_preprocessing/  -> synthesized from trtllm-templates
  #   my_model_trtllm_postprocessing/ -> synthesized from trtllm-templates
  #   my_model_trtllm_ensemble/       -> synthesized from trtllm-templates
  scripts/                      # Runtime script sources (also bundled in triton-server package)
    _lib.sh                     # Shared library sourced by the other scripts
    triton-preflight            # Pre-flight validation
    triton-resolve-model        # Multi-source model provisioning
    triton-serve                # Server launcher
    triton-setup-backends       # Backend directory assembler (activation-time)
    triton-setup-models         # Model directory assembler (activation-time)
  models/                       # Model repository
    vllm_test/                  # Example vLLM model (facebook/opt-125m)
      config.pbtxt
      1/
        model.json
    qwen3_8b/                   # Qwen3-8B via vLLM backend
      config.pbtxt
      1/
        model.json
    qwen2_5_05b/                # Qwen2.5-0.5B via Python backend (transformers)
      config.pbtxt
      1/
        model.py
    onnx_identity/              # ONNX identity test model
      config.pbtxt
      1/
        model.onnx
    identity_fp32/              # Python backend identity test model
      config.pbtxt
      1/
        model.py
    tensorrt_identity/          # TensorRT identity test model
      config.pbtxt
      1/
        model.plan
  tests/                        # Bats test suite
  README.md
```

Scripts (including `triton-setup-backends` and `triton-setup-models`) are bundled in the `triton-server` package at `$out/bin/` and available on `PATH` after `flox activate`. Source copies also live in `scripts/` in the [build-triton-server](../builds/build-triton-server/) repo (which is what the Nix build packages).

## Security notes

The runtime scripts handle untrusted input (model names, env files, lock files) and apply defense-in-depth.

### Env file trust model

The model env file is a trust boundary. In safe mode (default), `triton-serve` parses the file with a restrictive Python parser that accepts only `KEY=VALUE` or `export KEY=VALUE` lines with optional quotes. No shell interpolation, no command substitution, no variable expansion. In trusted mode, the file is `source`d directly -- only enable this for env files you control.

Even in safe mode, the env file can set arbitrary environment variables, so protect its location and permissions.

### File permissions

- **Env files**: written with `umask 077` and `chmod 600` -- readable only by the owning user.
- **Lock files**: created with `umask 077`. Symlink safety is checked before opening (symlinks are rejected; only regular files accepted).
- **Staging directories**: created under `$TRITON_MODEL_STATE_DIR/.staging/` with restricted permissions.

### Model name validation

`TRITON_MODEL` is validated by `lib::validate_model_name`:

- Must not be empty.
- Must not contain `/` or `\`.
- Must not be `.` or `..`.
- Must not contain control whitespace (newline, carriage return, tab).

This prevents path traversal and injection attacks in directory and file operations.

### Lock file safety

All lock files are validated before use:

- Symlinks are rejected (`[[ ! -L "$lockfile" ]]`).
- Only regular files are accepted (`[[ -f "$lockfile" ]]`).
- Created with `umask 077` to restrict access.
- Acquired via a background Python helper using `fcntl.flock()` with bounded polling and `TRITON_RESOLVE_LOCK_TIMEOUT` to prevent indefinite hangs. The helper sets `PR_SET_PDEATHSIG` so the lock is automatically released if the parent process dies.
