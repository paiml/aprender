# apr-cli — The Primary Interface

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §3
**Crate**: `apr-cli` (binary: `apr`)

---

## 1. Overview

`apr` is the primary interface to the Aprender ecosystem. It is not a thin
wrapper around library functions — it is the **product**. Every capability
ships as an `apr` subcommand first. Library APIs without CLI exposure are
considered incomplete.

**Framework**: Clap 4.5 with `#[derive(Parser)]` + `#[derive(Subcommand)]`

**Entry point**: `crates/apr-cli/src/main.rs` → `execute_command()`

---

## 2. CLI-First Development Rule

Every feature follows this order:

```
1. Design `apr <command>` UX — flags, output format, exit codes, --json
2. Write provable contract for the underlying operation
3. Implement the library function (aprender/realizar/entrenar)
4. Wire the CLI dispatch
5. Test via `apr probar` (golden regression) + unit tests
6. Document in `apr <command> --help`
```

If a library function has no `apr` subcommand → file a ticket. The CLI is
the source of truth for what the project can do.

---

## 3. Command Architecture

### 3.1 Dispatch Hierarchy

```
execute_command()
├── dispatch_runtime_commands()     → Run, Chat, Serve
├── dispatch_inspection_commands()  → Inspect, Debug, Validate, Lint, Explain
├── dispatch_diagnostic_commands()  → Trace, Tensors, Diff
├── dispatch_format_commands()      → Export, Import, Convert, Quantize
├── dispatch_model_commands()       → Merge, Finetune, Prune, Distill, Pull
├── dispatch_analysis()             → Monitor, Cbtop, Probar, Profile, Bench
└── dispatch_tool_commands()        → Tokenize, Compile, Data, Hex, Tree
```

Each dispatcher is in its own module (`dispatch.rs`, `dispatch_analysis.rs`,
`dispatch_run.rs`) to keep cyclomatic complexity ≤ 10.

### 3.2 Full Command Index

| Command | Group | Description |
|---------|-------|-------------|
| `run` | Runtime | Single inference with streaming output |
| `chat` | Runtime | Interactive chat session |
| `serve` | Runtime | OpenAI-compatible API server |
| `pull` | Runtime | Download models (hf:// URIs) |
| `finetune` | Model Ops | LoRA/QLoRA/full fine-tuning |
| `prune` | Model Ops | Magnitude/structured/Wanda/SparseGPT pruning |
| `distill` | Model Ops | Teacher → student knowledge transfer |
| `merge` | Model Ops | Weight-space interpolation (SLERP, TIES, DARE) |
| `quantize` | Model Ops | fp16/int8/int4/Q4K quantization |
| `train` | Training | Full training with plan/apply/watch/sweep |
| `tune` | Training | HPO (TPE/grid/random + ASHA scheduler) |
| `monitor` | Training | Training run monitoring |
| `runs` | Training | List/compare training runs |
| `inspect` | Analysis | Model metadata and structure |
| `debug` | Analysis | Drama mode, hex dump, ASCII extraction |
| `validate` | Analysis | Integrity check, 100-point quality score |
| `diff` | Analysis | Two-model comparison |
| `tensors` | Analysis | Tensor names, shapes, statistics |
| `trace` | Observe | Layer-by-layer state machine tracing |
| `profile` | Observe | Roofline, flamegraph, brick profiling |
| `cbtop` | Observe | Live ComputeBrick pipeline monitor |
| `probar` | Observe | Visual regression of layer activations |
| `qa` | Evaluation | Falsifiable QA gates (8+ checks) |
| `eval` | Evaluation | Perplexity, classification, pass@k |
| `bench` | Evaluation | Throughput benchmarking |
| `parity` | Evaluation | Cross-backend numerical validation |
| `lint` | Analysis | Best practices checking |
| `explain` | Analysis | Error codes, tensor, kernel explanation |
| `export` | Format | APR → SafeTensors/GGUF/MLX/ONNX |
| `import` | Format | HF/SafeTensors/GGUF → APR |
| `convert` | Format | Format conversion with quantization |
| `compile` | Format | Model → standalone executable |
| `tokenize` | Tools | BPE training, token breakdown |
| `data` | Tools | Load, validate, transform, statistics |
| `tui` | Viz | Terminal UI for model interaction |
| `cbtop` | Viz | ComputeBrick top (live monitoring) |
| `ptx` | Viz | PTX kernel inspector |
| `ptx-map` | Viz | PTX instruction mapping |
| `hex` | Viz | Format-aware binary forensics |
| `tree` | Viz | Architecture tree visualization |
| `flow` | Viz | Data flow visualization |

---

## 4. Global Flags

Every command respects these top-level flags:

| Flag | Effect |
|------|--------|
| `--json` | Machine-readable JSON output |
| `--verbose` | Detailed output |
| `-q` / `--quiet` | Suppress non-essential output |
| `--offline` | Block all network access (sovereign mode) |
| `--skip-contract` | Skip contract validation (dev only) |
| `--trace` | Enable layer tracing (renacer) |
| `--trace-level` | none / basic / layer / payload |
| `--trace-steps` | Comma-separated step filter |
| `--trace-output` | Export trace to JSON file |
| `--profile` | Enable brick profiling (BrickProfiler) |

---

## 5. Feature Flags

Conditional compilation controls which commands are available:

| Feature | Commands Enabled |
|---------|-----------------|
| `inference` (default) | run, chat, serve, profile, trace |
| `training` | finetune, train, tune, monitor, runs, experiment |
| `cuda` | GPU-accelerated inference and training |
| `cuda-batch` | Batched CUDA prefill |
| `visualization` | renacer tracing, trueno-viz charts, cbtop |
| `zram` | Compressed model loading |

---

## 6. Output Conventions

### 6.1 Human Output (default)

Readable tables, progress bars, colored status. Respects `NO_COLOR` and
terminal width.

### 6.2 JSON Output (`--json`)

Every command with `--json` emits a single JSON object to stdout. Exit
code 0 = success, non-zero = error. Errors include structured JSON with
`error_code` and `message` fields.

### 6.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Model not found / format error |
| 4 | Contract violation |
| 5 | QA gate failure |
| 6 | Parity check failure |

---

## 7. Tracing and Profiling Integration

Every inference command (`run`, `chat`, `serve`, `bench`) supports:

```bash
# Layer trace with tensor stats
apr run model.apr --prompt "test" --trace --trace-level layer

# Brick profiling with flamegraph
apr run model.apr --prompt "test" --profile --format flamegraph

# Both simultaneously
apr run model.apr --prompt "test" --trace --profile
```

Training commands (`train`, `finetune`, `tune`) support:

```bash
# BrickTracer on backward pass
apr train apply config.yaml --profile

# Anomaly detection with auto-escalation
apr finetune model.apr --data train.jsonl --trace
```

See [tracing.md](tracing.md) for the full tracing architecture.

---

## 8. Probar Integration

Visual regression testing is a first-class `apr` command:

```bash
# Capture golden snapshots
apr probar model.apr --golden golden/ --format png

# Validate against golden
apr probar model.apr --golden golden/ --assert

# Per-layer comparison
apr probar model.apr --golden golden/ --layer "transformer.0.*"

# Combined with profiling
apr probar model.apr --golden golden/ --profile --assert
```

Every model operation that modifies weights runs probar in CI:
merge → probar, finetune → probar, prune → probar, quantize → probar.

See [testing.md](testing.md) for the full probar specification.

---

## 9. Debugging Workflow

**Always use `apr` tools before reading code** (GH-202 lesson):

```
1. apr qa model.apr              ← Catches 80% of issues
2. apr trace model.apr           ← Layer-by-layer state machine
3. apr profile model.apr         ← Brick-level bottleneck
4. apr cbtop model.apr           ← Live anomaly detection
5. apr probar model.apr --golden ← Activation regression
6. apr diff model.apr ref.gguf   ← Two-model comparison
7. Read code                     ← Only after all tools fail
```
