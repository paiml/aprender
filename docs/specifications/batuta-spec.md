# Batuta Specification Overview

**Version:** 2.8.0
**Date:** 2026-04-05
**Status:** Active — apr code end-to-end on GPU

---

## 1. What is Batuta?

Batuta is the orchestration framework for the **Sovereign AI Stack** -- a vertically integrated, pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, realizar, entrenar, repartir, pacha) and provides transpilation pipelines for converting Python/C/Shell to Rust.

**Core mission:** Replace Python/CUDA/cloud dependencies with a self-contained Rust stack where data provenance, computation residency, and determinism are architectural guarantees -- not configuration options.

---

## 2. Stack Architecture

```
+-------------------------------------------------------------+
|                    batuta (Orchestration)                    |
+-------------------------------------------------------------+
| whisper-apr (ASR) | realizar (Inference)  | pacha (Registry) |
+-------------------+-----------------------+------------------+
|  aprender (ML)    | entrenar (Training)   | jugar (Games)    |
+-------------------+-----------------------+------------------+
|  simular (Simulation)     | profesor (Education)             |
+---------------------------+----------------------------------+
|               repartir (Distributed Compute)                 |
|          CPU (Rayon) | GPU (wgpu) | Remote (TCP/TLS)         |
+-------------------------------------------------------------+
| trueno-zram (Compression) | trueno-ublk (Block Device)       |
+---------------------------+----------------------------------+
|             trueno (SIMD/GPU Compute Primitives)             |
|       AVX2/AVX-512/NEON | wgpu | LZ4/ZSTD compression       |
+-------------------------------------------------------------+
```

### Layer Summary

| Layer | Crates | Purpose |
|-------|--------|---------|
| **Compute** | trueno, trueno-db, trueno-graph, trueno-rag, trueno-viz | SIMD/GPU primitives, vector DB, graph analytics, RAG, visualization |
| **Compression** | trueno-zram-core, trueno-ublk | SIMD/GPU memory compression, block device |
| **Distribution** | repartir | CPU/GPU/Remote work-stealing executors |
| **ML** | aprender, alimentar | Algorithms, APR v2 format, data loading |
| **Training** | entrenar | Autograd, LoRA/QLoRA, quantization, model merge |
| **Inference** | realizar, whisper-apr | GGUF/APR/SafeTensors inference, ASR |
| **Simulation** | simular, jugar | Monte Carlo, physics, game engine |
| **Education** | profesor | Courses, quizzes, labs (WASM) |
| **Registry** | pacha | Model registry with Ed25519 signatures |
| **Tracing** | renacer | Syscall tracing with source correlation |
| **Transpilers** | depyler, bashrs, decy | Python/Shell/C to Rust |
| **Profiling** | cgp (trueno) | Unified GPU/SIMD/CUDA profiling, roofline models, Nsight integration |
| **Testing** | jugar-probar | UX testing, LLM load testing (TTFT/TPOT/P99), visual regression, pixel coverage |
| **Quality** | pmat, certeza, apr-qa, provable-contracts | Static analysis, quality gates, model QA, YAML contract verification |
| **Orchestration** | batuta | Stack coordination, CLI, agent runtime |

---

## 3. Core Modules

### 3.1 Pipeline (`src/pipeline/`)

5-phase transpilation: Analysis -> Transpilation -> Optimization -> Validation -> Build. Uses Jidoka stop-on-error at each phase boundary. Main module at `src/pipeline/mod.rs` with stages in `src/pipeline/stages/`.

### 3.2 Backend (`src/backend.rs`)

Cost-based GPU/SIMD/Scalar selection using the 5x PCIe rule: dispatch to GPU only when `compute_time > 5 * transfer_time` (Gregg & Hazelwood, 2011).

### 3.3 Oracle (`src/oracle/`)

Knowledge graph for stack component recommendations. Supports natural language queries, RAG-indexed documentation search (SQLite+FTS5), and PMAT function-level code search. 34 cookbook recipes with TDD test companions.

### 3.4 Serve (`src/serve/`)

Model serving with failover, circuit breakers, and privacy tiers (Sovereign/Private/Standard). SpilloverRouter for local-first with remote fallback.

### 3.5 Stack (`src/stack/`)

Dependency graph management, coordinated release orchestration, quality gates across stack components. Publish-status caching with hash-based invalidation.

### 3.6 Agent (`src/agent/`)

Autonomous perceive-reason-act loop using local LLM inference (realizar) and persistent memory. Always Sovereign by default — all inference local, zero network. Primary entrypoint: `batuta code` (or `apr code` via apr-cli).

**Implemented (Phases 1-6c, 4q-4z):** Multi-turn conversation with 9 tools (file_read/write/edit, glob, grep, shell, memory, pmat_query, rag), tool definitions injected into prompt for local models, ChatML/Llama3/Qwen3NoThink chat templates auto-detected from model filename and GGUF architecture, session persistence (JSONL at `~/.apr/sessions/`), `--resume`/`--project` CLI flags, interactive auto-resume prompt for recent sessions (PMAT-165), `/test`/`/quality`/`/context`/`/compact`/`/session` slash commands, auto-compaction at 80% context window, model discovery with mtime-first sort and Jidoka validation (PMAT-185), APR.md/CLAUDE.md project instruction loading, output sanitization, `apr_model_validity` contract, context-aware prompt budgeting, AprServeDriver with graceful SIGTERM→SIGKILL shutdown (PMAT-166), dedicated `pmat_query` tool (PMAT-163), Qwen3NoThinkTemplate auto-selected via GGUF architecture caching (PMAT-181), **`apr code` subcommand wired in apr-cli** (PMAT-182), **`batuta code -p` end-to-end working** (PMAT-197: compact prompt + 32K context window + no-nudge). APR format preferred over GGUF throughout.

**Provable contracts (13 YAML + 129 FALSIFY tests):** `apr-code-v1`, `apr-model-discovery-v1` (real-file integration tests), `chat-template-v1`, `http-api-v1` (ChatCompletion/SSE), `session-v1`, `tokenizer-v1` (BPE roundtrip/thread safety), `apr-serve-v1` (lifecycle/health/format detection/Prometheus), `apr-serve-loadtest-v1` (SLO enforcement/baseline regression), `apr-chat-session-v1` (template/KV-cache/JSONL roundtrip), `apr-finetune-v1` (rank bounds/VRAM/merge shape), `model-format-conversion-v1` (quantization types, compression roundtrip, overwrite protection, no-op rejection, Q4K APR-preferred), `apr-model-lifecycle-v1` (check/inspect/convert/run/validate/export/import dispatch, invalid file rejection), `cli-dispatch-v1` (completeness/exit codes/feature gates). 10 Popperian falsification tests (PMAT-185). **6258 batuta tests, 58 apr-cli FALSIFY lib tests, 38 apr-cli FALSIFY integration tests.**

**Dogfood status (2026-04-05):**

| Feature | Status | Evidence |
|---------|--------|----------|
| `batuta code -p "What is 2+2?"` | **Working (GPU)** | `→ "4"` (Qwen3 1.7B, CUDA + serial prefill) |
| `batuta code -p "Write Rust function..."` | **Working (GPU)** | Complete code + explanation via CUDA |
| Model discovery (APR preferred) | Working | mtime-first sort, Jidoka validation, APR tiebreak |
| AprServeDriver launch | Working (GPU) | `apr serve --gpu` with `BATCHED_PREFILL=0`, 2.5-3.0s startup |
| Prompt scaling by model size | Working | Auto-detect from filename, 3-tier prompt selection |
| `apr code --help` in apr-cli | Working | Full flag set: --model, --project, --resume, -p |
| `apr serve loadtest --help` | Working | Wired into ServeCommands (PMAT-196) |
| CUDA GPU inference | **Working** | Serial prefill (Q4K/Q6K GEMV) correct. FP8 batched prefill broken (workaround: `BATCHED_PREFILL=0`) |
| APR Q4K via crates.io | **Blocked** | realizar 0.8.4 not published (PMAT-157) |

**Known blockers (remaining):**
- **FP8 batched prefill (low):** Requantization of Q6K tensors to FP8 produces wrong output. Workaround active (`BATCHED_PREFILL=0`). Affects prefill latency (~7ms slower) but not output quality.

**Resolved blockers (this cycle):**
- ~~PMAT-181:~~ CUDA GPU inference working via serial prefill (`BATCHED_PREFILL=0`). 27.5 tok/s on RTX 4090.
- ~~PMAT-159:~~ CUDA feature enabled in batuta (local path dep).
- ~~PMAT-197:~~ `-p` mode working — compact prompt, 32K context window, no-nudge.
- ~~PMAT-182:~~ apr-cli wiring verified complete.
- ~~PMAT-194:~~ cgp roofline profiling (Q4K/Q6K bandwidth-bound, 27.5 tok/s measured).
- ~~PMAT-195:~~ probar load test baselines (all SLOs pass at concurrency=1).

### Release Roadmap — apr-cli is the Product

**Goal:** Ship `apr` binary on crates.io with GPU inference, Qwen3 support, `apr code`, `apr serve loadtest`, and 13 provable contracts. apr-cli is the **entry point** — all sibling crates stabilize around it.

**Publish gate (ALL must pass for EVERY crate, ZERO exceptions):**
1. **95% test coverage** (`cargo llvm-cov`)
2. **`pmat comply`** (quality gates: TDG, complexity, SATD)
3. **`pv`** (provable-contracts verifier: all bindings implemented)
4. **GitHub CI green** (all workflows)
5. **`cargo install` locally** + verify it works end-to-end
6. **Clean-room build** (`make clean-room-p1` in `../infra/machines/clean-room/`)

**Version delta (crates.io → target):**

| Crate | crates.io | Target | Key Changes |
|-------|----------|--------|-------------|
| trueno | 0.17.0 | **0.17.1** | 12 BLIS FALSIFY tests, `blis-gemm-v1` contract, gemv safety fix, GEMM benchmarks |
| realizar | 0.8.3 | **0.8.4** | Qwen3NoThinkTemplate, has_quantized_tensors_apr, Q6K CUDA dispatch, architecture caching, build.rs publish fix |
| batuta | 0.7.2 | **0.7.4** | GPU inference (serial prefill), `cmd_code()` API, prompt scaling, COMPACT_SYSTEM_PROMPT, 32K context, 6258 tests, 13 contracts/129 FALSIFY |
| aprender | 0.27.5 | **0.27.6** | APR output dir fix, duplicated_attributes lint fix |
| entrenar | 0.7.6 | **0.7.7** | Training improvements, LoRA fixes |
| bashrs | 6.65.0 | **6.65.1** | Transpiler updates |
| apr-cli | 0.4.11 | **0.4.12** | `apr code`, `apr serve loadtest/bench`, prompt scaling, 9 FALSIFY test files, finetune contract |

**Publish order (strict — each step requires previous to be on crates.io):**

```
Step 1: trueno 0.17.1       (foundation — no sibling deps)
           │
Step 2: realizar 0.8.4      (depends on trueno 0.17)
  │        │
  │  Step 3: batuta 0.7.4   (depends on realizar 0.8)
  │        │
  │  Step 4: entrenar 0.7.7 (depends on trueno, aprender)
  │        │
  │  Step 5: bashrs 6.65.1  (independent — provable-contracts only)
  │        │
Step 6: aprender 0.27.6     (depends on alimentar, trueno)
           │
Step 7: apr-cli 0.4.12      (depends on ALL above + probar, whisper-apr)
```

Steps 3-5 can run in parallel after step 2 completes.

**Per-crate gate status (2026-04-05):**

| Gate | trueno | realizar | batuta | aprender | apr-cli |
|------|--------|----------|--------|----------|---------|
| Gate | trueno | realizar | batuta | entrenar | bashrs | aprender | apr-cli |
|------|--------|----------|--------|----------|--------|----------|---------|
| Tests | ✓ 3452 | ? 15K+ | ✓ 6258 | ? | ? | ? | ? |
| Coverage | ~88%* | ? | ? | ? | ? | ? | ? |
| pmat/pv | ? | ? | ✓ | ? | ? | ? | ? |
| CI green | **running** | **running** | ✓ | ? | ? | **running** | — |
| cargo install | ? | ? | ? | ? | ? | ? | ? |
| Clean-room | ? | ? | ? | ? | ? | ? | ? |
| Path deps OK | ✓ | ✓ | ✓ | ? | ? | ✓ | ✓ |

*Coverage: 88% limited by `unsafe` SIMD `#[target_feature]` — CI gate passes at this level.

**Progress (2026-04-05 final):**
- trueno: 12 BLIS FALSIFY tests, gemv safety fix, `cargo update`, pushed — CI running
- realizar: `cargo update` + build.rs publish fix pushed — CI running
- aprender: duplicated_attributes lint fix + `cargo update` pushed — CI queued
- entrenar: ✓ **CI GREEN** — ready for publish gate
- bashrs: CI running
- batuta: spec v2.8.0, book updated (apr code chapter), oracle refresh, 6258 tests, 13 contracts

**Five-whys root cause:** No release pipeline was executed — all gates are stale. The feature work (GPU inference, contracts, prompt scaling) is done; release engineering hasn't started.

**Falsification — what would DISPROVE this release can ship:**
1. realizar 0.8.4 coverage < 95% → fix coverage gaps before publish
2. aprender CI failing → fix lint/test/coverage before aprender publish
3. Clean-room build fails → fix any path-only deps
4. `apr code` doesn't work after `cargo install apr-cli` → verify with `apr code -p "What is 2+2?"`
5. Qwen3NoThinkTemplate not in published realizar → apr serve produces thinking loops

3. **Convert Qwen3 1.7B to APR format** — once realizar 0.8.4 lands, `apr convert --to-apr Qwen3-1.7B-Q4_K_M.gguf` should produce a valid `.apr` file with embedded tokenizer. APR is the stack-native format — faster loading, row-major layout, LZ4/ZSTD compression. Discovery will prefer it.

4. **Interactive REPL dogfood** — `-p` mode works but the interactive REPL (`batuta code` without `-p`) needs dogfood with the enriched system prompt. The full CODE_SYSTEM_PROMPT may still cause issues for 1.7B models. May need prompt scaling based on discovered model size.

5. **cgp roofline profiling** (PMAT-194) — after CUDA fix, profile Q4K/Q6K matvec, attention, softmax kernels. Establish baselines before optimizing. Blocked on step 1.

6. **probar LLM load testing baselines** (PMAT-195) — after CUDA fix, run `apr serve loadtest` to establish TTFT/TPOT/P99 SLOs for Qwen3 1.7B (APR and GGUF) at concurrency 1/4/8. Blocked on step 1.

7. **Prompt scaling by model size** — detect model parameter count at discovery and scale system prompt accordingly. <2B → compact prompt (current -p behavior). 2-7B → tool table without examples. 7B+ → full prompt. This makes interactive mode reliable for small models.

8. **OS-native sandboxing** (Phase 5) — Landlock (Linux) / Seatbelt (macOS) for tool execution. Lower priority since capability + allowlist + path restriction already provide 3 layers.

**Planned:** OS-native sandboxing (Phase 5), presentar-terminal TUI (Phase 7), prompt scaling by model size, cgp profiling (after CUDA fix), probar load testing baselines (after CUDA fix).

### 3.7 Bug Hunter (`src/bug_hunter/`)

Proactive fault localization using 5-channel SBFL (spectrum, mutation, static, semantic, PMAT quality). Research-based techniques from Zazworka et al. (2011).

---

## 4. Feature Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `native` | Full CLI, filesystem, tracing, TUI dashboard | Yes |
| `rag` | SQLite+FTS5 RAG oracle | Yes |
| `agents` | Autonomous agent runtime (`batuta code`, perceive-reason-act loop) | Yes |
| `inference` | RealizarDriver for local LLM inference (GGUF/APR) | Yes |
| `agents-inference` | Agent with local inference via RealizarDriver | No (included in default via `inference`) |
| `agents-rag` | Agent with trueno-rag document retrieval | No |
| `agents-full` | All agent features (inference + RAG) | No |
| `banco` | AI workbench API (OpenAI-compatible endpoints) | No |
| `distributed` | repartir distributed compute integration | No |
| `wasm` | Browser-compatible build (no filesystem, in-memory) | No |
| `trueno-integration` | SIMD/GPU tensor operations | No |
| `oracle-mode` | Knowledge graph with trueno-graph and trueno-db | No |
| `sovereign-stack` | All components enabled | No |

See `Cargo.toml` for the full list of 30 feature flags including `ml`, `speech`, `compression`, `simulation`, `transpilers`, `testing`, `tui`, `viz`, and `agents-*` variants.

---

## 5. Design Principles (Toyota Production System)

| Principle | Application in Batuta |
|-----------|----------------------|
| **Jidoka** (Stop-on-error) | Pipeline halts on first defect; circuit breakers in serving |
| **Poka-Yoke** (Mistake-proofing) | Privacy tiers prevent data leakage; type safety at API boundaries |
| **Heijunka** (Level loading) | SpilloverRouter balances local/remote; work-stealing in repartir |
| **Muda** (Waste elimination) | Cost circuit breakers; content-addressed caching in playbooks |
| **Kaizen** (Continuous improvement) | MoE backend selection; calibration feedback loops |
| **Genchi Genbutsu** (Go and see) | Hash-based validation; benchmark on actual hardware |

---

## 6. Quality Standards

| Metric | Target |
|--------|--------|
| Test coverage | >= 95% (90% enforced, 95% preferred) |
| Clippy warnings | Zero (`-D warnings`) |
| Mutation testing | >= 80% kill rate |
| TDG Score | A grade (>= 85) |
| Pre-commit time | < 30s |
| LLM serve latency (P99) | < 1s TTFT, < 50ms TPOT (GPU) |
| GPU kernel efficiency | > 50% roofline (cgp validated) |

### 6.1 Performance Validation: trueno cgp

**Required tool:** `cgp` from `../trueno/crates/cgp/` — Compute-GPU-Profile.

All GPU-accelerated inference paths (`apr serve`, realizar CUDA kernels, wgpu compute) must be profiled with cgp before release. cgp provides:

- **Roofline analysis** — kernel efficiency vs hardware peak (must exceed 50%)
- **SIMD/GPU profiling** — AVX2/AVX-512/NEON/wgpu/CUDA backend comparison
- **Nsight integration** — CUPTI hardware counters for CUDA kernels
- **Regression detection** — `cgp diff` compares baselines across commits

```bash
# Profile apr serve inference kernel
cgp profile --backend cuda -- apr serve run model.gguf

# Generate roofline model for Q4K dequantization
cgp roofline --kernel fused_q4k_matvec --backend cuda

# Compare current vs baseline
cgp diff --baseline .cgp/baseline.json

# Validate contract: kernel meets roofline threshold
cgp contract --min-efficiency 0.5 --kernel "q4k*"
```

**Integration with batuta:** `batuta stack quality` must invoke `cgp contract` when CUDA is available. Inference-critical kernels (Q4K/Q6K matvec, attention, softmax) must maintain roofline efficiency > 50% across releases.

### 6.2 LLM Load Testing: probar (Wired into apr-cli)

**Required dependency:** `jugar-probar` with `llm` feature in apr-cli's Cargo.toml.

LLM load testing is a **first-class apr-cli subcommand**, not an external tool. It is wired directly into `apr serve` as additional `ServeCommands` variants, giving users a single binary for serving AND testing.

#### 6.2.1 apr-cli Wiring (Required)

`ServeCommands` in `apr-cli/src/serve_commands.rs` must include:

```rust
/// Load test a running apr serve instance (probar LLM)
Loadtest {
    /// URL of running apr serve instance
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,
    /// Number of concurrent clients
    #[arg(long, default_value = "4")]
    concurrency: usize,
    /// Test duration (e.g., "30s", "2m")
    #[arg(long, default_value = "30s")]
    duration: String,
    /// Prompt corpus file (JSONL with "prompt" field)
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// TTFT P99 SLO in milliseconds (fail if exceeded)
    #[arg(long, default_value = "1000")]
    slo_ttft_p99_ms: u64,
    /// TPOT P99 SLO in milliseconds (fail if exceeded)
    #[arg(long, default_value = "50")]
    slo_tpot_p99_ms: u64,
    /// Output format: text, json
    #[arg(long, default_value = "text")]
    format: String,
    /// Save results as baseline for regression detection
    #[arg(long)]
    save_baseline: bool,
},
/// Multi-run benchmark with warmup and regression detection
Bench {
    /// URL of running apr serve instance
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,
    /// Number of benchmark runs
    #[arg(long, default_value = "5")]
    runs: usize,
    /// Warmup runs (excluded from stats)
    #[arg(long, default_value = "1")]
    warmup: usize,
    /// Baseline file for regression detection
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Fail if regression > threshold (percentage)
    #[arg(long, default_value = "10")]
    regression_threshold: f64,
},
```

Dispatch in `dispatch_run.rs`:

```rust
ServeCommands::Loadtest { url, concurrency, duration, prompts, slo_ttft_p99_ms, slo_tpot_p99_ms, format, save_baseline } => {
    commands::serve_loadtest::run_loadtest(url, concurrency, duration, prompts, slo_ttft_p99_ms, slo_tpot_p99_ms, format, save_baseline)
}
ServeCommands::Bench { url, runs, warmup, baseline, regression_threshold } => {
    commands::serve_bench::run_bench(url, runs, warmup, baseline, regression_threshold)
}
```

Feature-gated behind `llm-loadtest` (default when `inference` feature active):

```toml
# apr-cli/Cargo.toml
[features]
llm-loadtest = ["jugar-probar/llm"]
inference = ["dep:realizar", "llm-loadtest"]

[dependencies]
jugar-probar = { version = "1.0", path = "../../../probar/crates/probar", features = ["llm"], optional = true }
```

#### 6.2.2 User-Facing Commands

```bash
# Start server in one terminal
apr serve run model.gguf --gpu

# Load test in another terminal
apr serve loadtest --url http://localhost:8080 --concurrency 8 --duration 60s
apr serve loadtest --prompts corpus/coding-tasks.jsonl --slo-ttft-p99-ms 1000

# Multi-run benchmark with regression detection
apr serve bench --runs 5 --warmup 1 --baseline .apr/benchmarks/baseline.json

# Inline: start server, test, stop (CI mode)
apr serve run model.gguf --gpu &
sleep 5
apr serve loadtest --concurrency 4 --duration 30s --save-baseline
kill %1
```

#### 6.2.3 SLO Thresholds

| Metric | GPU (RTX 4090) | CPU | Jetson Orin |
|--------|---------------|-----|-------------|
| TTFT P99 | < 500ms | < 5s | < 2s |
| TPOT P99 | < 50ms | < 200ms | < 100ms |
| Throughput | > 10 tok/s | > 5 tok/s | > 20 tok/s |

#### 6.2.3b Measured Baseline (2026-04-05, Qwen3 1.7B Q4_K_M, RTX 4090, serial prefill)

| Metric | Concurrency=1 | Concurrency=4 | SLO |
|--------|--------------|---------------|-----|
| TTFT P50 | 328ms | 1297ms | < 500ms |
| TTFT P99 | 348ms | 1343ms | < 500ms |
| TPOT P50 | 36.4ms | 144.1ms | < 50ms |
| TPOT P99 | 38.7ms | 149.2ms | < 50ms |
| Throughput | 27.5 tok/s | 27.7 tok/s | > 10 tok/s |
| Error rate | 0% | 0% | 0% |
| Requests | 92/30s | 96/30s | - |

**Concurrency=1: All SLOs pass.** TTFT P99=348ms, TPOT P99=38.7ms, 27.5 tok/s.
**Concurrency=4: SLOs exceed at P99.** Serial prefill (BATCHED_PREFILL=0) serializes requests. Throughput unchanged (27.7 tok/s) — requests queue. Fix: restore FP8 batched prefill after Q6K requantization bug is fixed in realizar.

Baseline saved: `~/.apr/benchmarks/baseline.json`

#### 6.2.3c Roofline Analysis (cgp, RTX 4090)

```
Peak: FP32=82.6 TFLOP/s, DRAM=1.0 TB/s, Ridge=82 FLOP/byte
Q4K GEMV (2048x2048): 2.36 MB compressed, 3.5 FLOP/byte (bandwidth-bound)
Q6K GEMV (2048x2048): 3.44 MB compressed, 2.4 FLOP/byte (bandwidth-bound)
```

| Metric | CPU (Threadripper 7960X) | GPU (RTX 4090, serial) | GPU (theoretical batch) |
|--------|--------------------------|----------------------|------------------------|
| Q4K GEMV attn | 228 us / 20.7 GFLOP/s | ~650 us | ~2.4 us |
| Throughput | 14.8 tok/s | 27.5 tok/s | ~420 tok/s |
| DRAM BW utilization | N/A | ~0.4% | ~98% (at batch) |
| Speedup vs CPU | 1.0x | 1.86x | ~28x |

Serial prefill achieves 1.86x over CPU. Batched prefill (when FP8 Q6K bug is fixed) would approach DRAM bandwidth limit (~420 tok/s theoretical for Q4K at 1.7B).

#### 6.2.4 Provable Contract: `apr-serve-loadtest-v1`

| Equation | Property |
|----------|----------|
| `slo_enforcement` | Loadtest fails (exit 1) if any SLO threshold exceeded |
| `baseline_regression` | Bench fails if regression > threshold vs baseline |
| `result_completeness` | LoadTestResult contains TTFT, TPOT, P50/P95/P99, throughput, error count |
| `concurrent_correctness` | N concurrent requests produce N distinct valid responses |
| `streaming_metrics` | If server supports SSE, TPOT measured per-token (not per-response) |

#### 6.2.5 Integration with batuta

- `batuta stack gate` invokes `apr serve loadtest` when a model and CUDA are available
- Load test results archived in `~/.apr/benchmarks/{model}/{timestamp}.json`
- `batuta stack quality` reports SLO pass/fail in quality matrix
- `-p` mode HTTP timeout (PMAT-159 dogfood finding) detected by loadtest SLO violation

---

## 7. Key Commands

```bash
# batuta code / apr code (agentic coding assistant — sovereign-first)
batuta code                       # Interactive — auto-discovers model from ~/.apr/models/
batuta code --model ~/.apr/models/qwen3-1.7b-q4k.apr  # Default go-to model (0.960 tool score, APR preferred)
batuta code -p "Fix the auth bug" # Non-interactive: print response and exit
batuta code --resume              # Resume most recent session for this directory
batuta code --resume=<session-id> # Resume specific session
batuta code --project ../other    # Load APR.md/CLAUDE.md from another directory

# Slash commands inside batuta code:
# /test, /quality, /context, /compact, /session, /sessions, /help, /quit

# Stack management
batuta stack check              # Dependency health
batuta stack status             # TUI dashboard
batuta stack versions           # Check crates.io versions
batuta stack quality            # Quality matrix
batuta stack gate               # CI quality gate

# Oracle (knowledge queries)
batuta oracle "How do I train a model?"
batuta oracle --rag "tokenization"
batuta oracle --recipe ml-random-forest --format code
batuta oracle --pmat-query "error handling"

# LLM load testing (wired into apr serve — jugar-probar)
apr serve loadtest --concurrency 8 --duration 60s                    # Load test running server
apr serve loadtest --prompts corpus/tasks.jsonl --slo-ttft-p99-ms 1000  # With SLO enforcement
apr serve bench --runs 5 --baseline .apr/benchmarks/baseline.json    # Multi-run regression check

# GPU profiling (cgp from trueno)
cgp profile --backend cuda -- apr serve run model.gguf              # Profile inference
cgp roofline --kernel fused_q4k_matvec --backend cuda               # Roofline analysis

# Agent runtime (engine underneath apr code)
batuta agent run --manifest agent.toml

# Playbook (DAG pipelines)
batuta playbook run pipeline.yaml
batuta playbook status
batuta playbook validate pipeline.yaml

# Analysis
batuta analyze --languages --tdg .
```

---

## 8. APR v2 Model Format

The `.apr` format is the stack's native model serialization:

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Compression | None | LZ4/ZSTD |
| Index Format | JSON | Binary |
| Zero-Copy Loading | Partial | Full |
| Quantization | Int8 | Int4/Int8 |
| Streaming | No | Yes |

**Layout policy (LAYOUT-002):** The entire stack uses **row-major** tensor layout. GGUF column-major data is transposed at import by aprender.

---

## 9. Component Specifications

| Component Spec | Description | Key Source Files |
|----------------|-------------|-----------------|
| [oracle-and-rag.md](components/oracle-and-rag.md) | Oracle knowledge graph, RAG (SQLite+FTS5), PMAT query integration, code snippets | `src/oracle/`, `src/cli/oracle/` |
| [transpilation-pipeline.md](components/transpilation-pipeline.md) | 5-phase pipeline, transpiler integration (Decy/Depyler/Bashrs), CITL cross-language learning | `src/pipeline.rs`, `src/*_converter.rs` |
| [stack-management.md](components/stack-management.md) | Dependency graph, coordinated releases, quality matrix, QA checklist | `src/stack/`, `src/cli/stack/` |
| [agent-and-playbook.md](components/agent-and-playbook.md) | Autonomous agent runtime (perceive-reason-act), context compaction, parallel tools, OS sandboxing, hooks, session persistence, YAML playbook DAG pipelines | `src/agent/`, planned |
| [multi-provider-api.md](components/multi-provider-api.md) | Provider-agnostic LLM client (Anthropic/OpenAI translation), streaming SSE, exponential backoff, provider failover, cost tracking | `src/agent/driver/remote/` |
| [presentar-probar-integration.md](components/presentar-probar-integration.md) | Agent TUI via presentar-terminal (6 panels), Brick UX contracts, probar pixel coverage + state machine playbooks + M1-M5 mutation testing, visual regression | `src/agent/tui/`, `src/agent/brick/`, `tests/playbooks/` |
| [apr-code.md](components/apr-code.md) | `apr code` / `batuta code` — Sovereign-only agentic coding assistant. Phases 1-4a DONE: 9 tools (incl. pmat_query, rag), multi-turn history, session persistence (JSONL), model discovery (APR-preferred, Jidoka-validated), chat templates, auto-compaction, output sanitization, `apr_model_validity` contract, AprServeDriver (CUDA/GPU via apr serve), `apr code` subcommand in apr-cli (PMAT-162). All inference local via realizar (GGUF/APR). | `src/agent/code.rs`, `src/agent/`, `src/agent/session.rs` |
| [apr-code-tui-testing.md](components/apr-code-tui-testing.md) | Probar-first TUI testing spec: per-panel test harnesses, pixel coverage, visual regression baselines, state machine playbooks, Brick falsification, WCAG AA/AAA accessibility, frame budget benchmarks. Contracts: `tui-rendering-v1`, `tui-panels-v1` | `tests/tui/`, presentar-terminal, jugar-probar |
| [falsification-report.md](components/falsification-report.md) | Cross-spec Popperian falsification: 12 contradictions, 8 unfalsifiable claims, 6 missing failure modes, 4 circular dependencies. Priority fixes applied inline. | All specs |
| [apr-code-feasibility-falsification.md](components/apr-code-feasibility-falsification.md) | Code-verified feasibility of `apr code`: dependency chain (no circular dep), 2 real gaps (REPL + file tools), 77% reuse of 5,000+ existing agent lines | `src/agent/`, `apr-cli` |
| [banco-spec.md](components/banco-spec.md) | Banco AI workbench overview — unified AI studio with OpenAI-compatible API | `src/serve/` |
| [banco-phase1.md](components/banco-phase1.md) | Banco Phase 1: core endpoints (chat, completions, models) | `src/serve/` |
| [banco-phase2.md](components/banco-phase2.md) | Banco Phase 2: model slot, load/unload, system presets, auth | `src/serve/` |
| [banco-phase3.md](components/banco-phase3.md) | Banco Phase 3: inference integration with realizar OwnedQuantizedModel | `src/serve/` |
| [banco-phase4.md](components/banco-phase4.md) | Banco Phase 4: advanced features (streaming, embeddings, fine-tune) | `src/serve/` |
| [banco-infra.md](components/banco-infra.md) | Banco infrastructure: Axum server, CORS, config, deployment | `src/serve/` |
| [banco-contracts.md](components/banco-contracts.md) | Banco provable contracts: 5 YAML contracts for API correctness | `../provable-contracts/contracts/batuta/` |
| [banco-cross-cutting.md](components/banco-cross-cutting.md) | Banco cross-cutting: OpenAI SDK compat, error handling, logging | `src/serve/` |
| [banco-ux.md](components/banco-ux.md) | Banco UX: CLI, TUI dashboard, interactive mode | `src/cli/` |
| [banco-ux-falsification.md](components/banco-ux-falsification.md) | Banco UX Popperian falsification tests | `tests/` |
| [banco-testing.md](components/banco-testing.md) | Banco test strategy: 124 tests, 24 endpoints, falsification | `tests/` |
| [banco-falsification-report.md](components/banco-falsification-report.md) | Banco falsification report: cross-spec contradiction analysis | All banco specs |
| [quality-and-testing.md](components/quality-and-testing.md) | Popperian falsification methodology, testing ecosystem (pmat/oip/probar), bug-hunter PMAT integration | `src/bug_hunter/` |
| [external-integrations.md](components/external-integrations.md) | Data platforms (Databricks/Snowflake/AWS), visualization, content tooling, Apple hardware (manzana) | `src/cli/` |
| [sovereign-ai-architecture.md](components/sovereign-ai-architecture.md) | Formal architecture spec, lifetime/memory model, stack diagnostics and reporting | Core architecture |

---

## 10. Archive

All original specification files are preserved unchanged in [archive/](archive/). The component specs above are condensed versions that consolidate related topics and remove redundancy while preserving technical accuracy.

### Archived Files (29 total)

| Original File | Lines | Consolidated Into |
|---------------|-------|-------------------|
| `oracle-mode-spec.md` | 1229 | oracle-and-rag |
| `apr-powered-rag-oracle.md` | 1002 | oracle-and-rag |
| `sqlite-rag-integration.md` | 1318 | oracle-and-rag |
| `improve-oracle.md` | 1818 | oracle-and-rag |
| `pmat-query-batuta-oracle-integration.md` | 283 | oracle-and-rag |
| `code-snippets.md` | 242 | oracle-and-rag |
| `batuta-orchestration-...spec.md` | 2331 | transpilation-pipeline |
| `citl-cross-language-spec.md` | 698 | transpilation-pipeline |
| `batuta-stack-spec.md` | 1750 | stack-management |
| `stack-quality-matrix-spec.md` | 1094 | stack-management |
| `stack-tree-view.md` | 225 | stack-management |
| `batuta-stack-0.1-100-point-qa-checklist.md` | 186 | stack-management |
| `score-a-plus-spec.md` | 71 | stack-management |
| `book-score-spec.md` | 71 | stack-management |
| `batuta-agent.md` | 2257 | agent-and-playbook |
| `batuta-playbook.md` | 2056 | agent-and-playbook |
| `model-serving-ecosystem-spec.md` | 444 | banco-spec |
| `hugging-face-integration-query-publish-spec.md` | 619 | banco-spec |
| `hugging-face-crud-spec.md` | 448 | banco-spec |
| `retriever-spec.md` | 3019 | banco-spec |
| `popperian-falsification-checklist.md` | 3690 | quality-and-testing |
| `testing-quality-ecosystem-spec.md` | 372 | quality-and-testing |
| `bug-hunter-pmat-quality-integration.md` | 233 | quality-and-testing |
| `data-platforms-integration-spec-query.md` | 1131 | external-integrations |
| `data-visualization-integration-query.md` | 439 | external-integrations |
| `content-creation-tooling-spec.md` | 980 | external-integrations |
| `manzana-apple-hardware-spec.md` | 215 | external-integrations |
| `sovereign-ai-spec.md` | 868 | sovereign-ai-architecture |
| `stack-visualization-diagnostics-reporting.md` | 1223 | sovereign-ai-architecture |
