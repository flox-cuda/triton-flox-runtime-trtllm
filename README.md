# Triton Inference Server Runtime

Production NVIDIA Triton Inference Server deployment as a Flox environment. Serves model repositories (TensorRT, ONNX, PyTorch, TensorFlow, Python, vLLM backends) via `tritonserver` with GPU acceleration and multi-port serving (HTTP, gRPC, metrics).

- **Triton Inference Server**: latest (via Flox package)
- **CUDA**: requires NVIDIA driver with CUDA support
- **Platform**: Linux only (`/proc` required for preflight)

Unlike the llama.cpp runtime (which serves single GGUF files), Triton serves model *repositories* -- directories containing versioned subdirectories with backend-specific artifacts and optional `config.pbtxt` files. Triton exposes its own HTTP, gRPC, and Prometheus metrics APIs; this runtime handles operational lifecycle: port reclaim, model provisioning, environment validation, and process management.

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

The service command chains three scripts in a pipeline:

```
triton-preflight && triton-resolve-model && triton-serve
```

```
┌──────────────────────────────────────────────────────────┐
│  Consuming Environment (.flox/env/manifest.toml)         │
│                                                          │
│  [install]                                               │
│    flox/triton-runtime          # 3-script pipeline      │
│    python312Packages.huggingface-hub  # HF downloads     │
│                                                          │
│  [services]                                              │
│    triton → triton-preflight                             │
│             && triton-resolve-model                      │
│             && triton-serve                              │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  triton-preflight                                  │  │
│  │    Multi-port reclaim ← /proc/net/tcp + /proc/     │  │
│  │    GPU health         ← PyTorch or nvidia-smi      │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  triton-resolve-model                              │  │
│  │    Sources: flox → local → r2 → hf-hub             │  │
│  │    Layout validation: version dirs + artifacts     │  │
│  │    Output: per-model .env file (mode 600)          │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  triton-serve                                      │  │
│  │    Loads .env → validates args                      │  │
│  │    → exec tritonserver  (default)                   │  │
│  │    → exec python3 main.py  (OPENAI_FRONTEND=true)  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

1. **triton-preflight** -- Reclaims HTTP, gRPC, metrics, and (when `TRITON_OPENAI_FRONTEND=true`) OpenAI ports if occupied by stale tritonserver or OpenAI frontend processes, checks GPU health via PyTorch or nvidia-smi fallback, optionally executes a downstream command.
2. **triton-resolve-model** -- Provisions the model repository from configured sources with per-model locking, staging directories, atomic swaps, and layout validation. Writes a per-model env file.
3. **triton-serve** -- Loads the env file (safe or trusted mode), validates all required vars, and `exec`s either `tritonserver` (default) or `python3 main.py` (when `TRITON_OPENAI_FRONTEND=true`).

Scripts are provided by the `flox/triton-runtime` package and available on `PATH` after activation.

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
| `onnx` | `model.onnx` | ONNX Runtime |
| `pytorch` | `model.pt` | TorchScript model |
| `tensorflow` | `model.savedmodel/` | Directory; must contain `saved_model.pb` |
| `python` | `model.py` | Python backend script |
| `vllm` | `model.json` | vLLM configuration file |

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
| `TRITON_MODEL` | _(required)_ | Model name (directory name within the repository). Must not contain `/`, `\`, or be `.`/`..` |
| `TRITON_MODEL_REPOSITORY` | _(required)_ | Base model repository path. Created automatically if missing |
| `TRITON_MODEL_ID` | _(unset)_ | Explicit HuggingFace model ID (`org/repo`) for hf-hub source |
| `TRITON_MODEL_ORG` | _(unset)_ | HF org prefix. Used to derive model ID as `$TRITON_MODEL_ORG/$TRITON_MODEL` |
| `TRITON_MODEL_BACKEND` | _(unset)_ | Backend hint: `tensorrt`, `onnx`, `pytorch`, `tensorflow`, `python`, `vllm`. Restricts artifact validation |
| `TRITON_MODEL_SOURCES` | `flox,local,r2,hf-hub` | Comma-separated source chain. Available: `flox`, `local`, `hf-cache`, `r2`, `hf-hub` |
| `TRITON_MODEL_ENV_FILE` | _(derived)_ | Override env file path. Default: `$FLOX_ENV_CACHE/triton-model.<slug>.<hash>.env` |

### Server settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_MODEL_CONTROL_MODE` | `none` | Model control mode: `none`, `explicit`, `poll` |
| `TRITON_LOG_VERBOSE` | `0` | Tritonserver log verbosity level. Non-negative integer |
| `TRITON_STRICT_READINESS` | `true` | Require all models ready for health check. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_HTTP` | `true` | Enable HTTP endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_GRPC` | `true` | Enable gRPC endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `TRITON_ALLOW_METRICS` | `true` | Enable Prometheus metrics endpoint. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
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
| `TRITON_KEEP_LOGS` | `0` | `1` to keep download logs on success (always kept on failure) |

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
| `hf-cache` | Scans `$TRITON_MODEL_REPOSITORY/hub/models--<slug>/snapshots/` | No model ID derivable, no usable snapshot | Sets path to newest valid snapshot |

### Model repository validation

The `_validate_model_repo` function checks every candidate directory:

1. Model directory exists and is listable.
2. At least one numeric version directory (e.g., `1/`) is present.
3. Every version directory contains a recognized artifact for the target backend.
4. When `TRITON_MODEL_BACKEND` is set, only that backend's artifact is checked.
5. TensorFlow `model.savedmodel/` must contain `saved_model.pb`.

The function returns a JSON object with fields: `valid`, `versions`, `backends_detected`, `has_config`, and `error` (on failure).

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

- **Per-model lock**: acquired via `flock` before any source search. Lock file: `<env_file>.lock`. Timeout: `TRITON_RESOLVE_LOCK_TIMEOUT` seconds (default 300). Symlink and regular-file checks are enforced before opening.
- **Atomic swap** (r2 and hf-hub only): downloads stage into a temp directory under `$TRITON_MODEL_REPOSITORY/.staging/`. After layout validation, the staged directory replaces the target via backup+rename. If the swap is interrupted, `lib::restore_backup` recovers the most recent backup on the next run.
- **Staging cleanup**: staging directories and download logs are cleaned up on success. On failure, logs are preserved for debugging.

### R2 (S3-compatible storage)

R2 downloads use `aws s3 sync` to fetch the model directory from `s3://$R2_BUCKET/$R2_MODELS_PREFIX/$TRITON_MODEL/` into a staging directory. Requirements:

- `aws` CLI must be on PATH.
- `R2_BUCKET` and `R2_MODELS_PREFIX` must both be set.
- `R2_ENDPOINT_URL` is passed as `--endpoint-url` when set.
- AWS credentials must be valid (`aws sts get-caller-identity` is checked before download).

The download is staged, validated, and then atomically swapped into the target directory.

### HF cache source

When `hf-cache` is in the source chain, the script scans `$TRITON_MODEL_REPOSITORY/hub/models--<slug>/snapshots/` for valid model layouts. Snapshots are checked newest-first (by modification time). The slug is derived from the model ID by replacing `/` with `--`. This source requires `TRITON_MODEL_ID` or `TRITON_MODEL_ORG` to be set.

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

When positional arguments are provided (with or without a leading `--`), they are executed via `exec` after all checks pass. This is how the service pipeline chains preflight into resolve and serve.

```bash
# In the service definition:
triton-preflight -- triton-resolve-model && triton-serve
```

`TRITON_PREFLIGHT_JSON=1` is incompatible with downstream commands because stdout must remain JSON-only.

### Locking

Acquired via `flock` on `TRITON_PREFLIGHT_LOCKFILE` (default `/tmp/triton-preflight.lock`) with a 10-second timeout. Prevents concurrent preflight runs from racing. The lockfile is validated: symlinks are rejected, and only regular files are accepted.

## Serving (triton-serve)

Loads the resolved model env file, validates configuration, and executes `tritonserver` (default) or the OpenAI-compatible frontend (`python3 main.py`) when `TRITON_OPENAI_FRONTEND=true`.

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

For hot-swapping without restart, use `poll` mode:

```bash
TRITON_MODEL_CONTROL_MODE=poll flox activate --start-services
# Now copy new model versions into the repository; Triton picks them up automatically
```

To restart with a different model:

```bash
flox services restart triton
```

## Service management

```bash
flox services status              # check service state
flox services logs triton         # tail service logs
flox services logs triton -f      # follow logs
flox services restart triton      # restart the tritonserver service
flox services stop                # stop all services
flox activate --start-services    # activate and start in one step
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

Check staging logs (preserved on failure) at `$TRITON_MODEL_REPOSITORY/.staging/`.

### OpenAI frontend main.py not found

The OpenAI frontend auto-discovers `main.py` from standard locations (`/opt/tritonserver/python/openai/main.py` or relative to the `tritonserver` binary). If it fails, set the path explicitly:

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
rm -f "$FLOX_ENV_CACHE"/triton-model.*.env.lock
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

## File structure

```
triton-runtime/
  .flox/env/manifest.toml      # Flox manifest (packages, on-activate hook, service)
  models/                       # Model repository (created at runtime)
    my-onnx-model/
      config.pbtxt
      1/
        model.onnx
    .staging/                   # Temp dirs during downloads (cleaned up)
  README.md
```

Scripts (`triton-preflight`, `triton-resolve-model`, `triton-serve`, `_lib.sh`) are provided by the `flox/triton-runtime` package and available on `PATH` after activation. They are not stored in this directory.

## Security notes

The runtime scripts handle untrusted input (model names, env files, lock files) and apply defense-in-depth.

### Env file trust model

The model env file is a trust boundary. In safe mode (default), `triton-serve` parses the file with a restrictive Python parser that accepts only `KEY=VALUE` or `export KEY=VALUE` lines with optional quotes. No shell interpolation, no command substitution, no variable expansion. In trusted mode, the file is `source`d directly -- only enable this for env files you control.

Even in safe mode, the env file can set arbitrary environment variables, so protect its location and permissions.

### File permissions

- **Env files**: written with `umask 077` and `chmod 600` -- readable only by the owning user.
- **Lock files**: created with `umask 077`. Symlink safety is checked before opening (symlinks are rejected; only regular files accepted).
- **Staging directories**: created under `$TRITON_MODEL_REPOSITORY/.staging/` with restricted permissions.

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
- Acquired with `flock -w <timeout>` to prevent indefinite hangs.
