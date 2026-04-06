# Banco Specification: Local-First AI Workbench

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: model-serving-ecosystem-spec, hugging-face-integration-query-publish-spec, hugging-face-crud-spec, retriever-spec
> Sub-specs: [banco-phase1.md](banco-phase1.md), [banco-phase2.md](banco-phase2.md), [banco-phase3.md](banco-phase3.md), [banco-phase4.md](banco-phase4.md), [banco-cross-cutting.md](banco-cross-cutting.md), [banco-ux.md](banco-ux.md), [banco-testing.md](banco-testing.md), [banco-infra.md](banco-infra.md), [banco-contracts.md](banco-contracts.md)
> Falsification: [banco-falsification-report.md](banco-falsification-report.md), [banco-ux-falsification.md](banco-ux-falsification.md)

---

## 1. Model Serving Ecosystem

### Local Serving Landscape

| Tool | Language | Format | Key Feature | PAIML Integration |
|------|----------|--------|-------------|-------------------|
| **realizar** | Rust | GGUF/SafeTensors/APR | MoE, circuit breakers | Native |
| Ollama | Go | GGUF | Docker-like UX | API compatible |
| llamafile | C | GGUF | Single executable | Format compatible |
| llama.cpp | C++ | GGUF | Reference impl | Format compatible |
| vLLM | Python | PT/SafeTensors | PagedAttention | API bridgeable |
| TGI | Rust/Python | SafeTensors | HF official | Pattern reference |

### Remote Serving Landscape

| Service | Provider | Characteristics |
|---------|----------|----------------|
| HF Inference API | HuggingFace | Serverless, pay-per-token |
| HF Endpoints | HuggingFace | Dedicated, auto-scaling |
| Together.ai | Together | Fast inference, open models |
| AWS SageMaker | AWS | Managed, GPU instances |
| Replicate | Replicate | Docker-based, pay-per-second |

### Batuta Serve Architecture

```
src/serve/
  +-- mod.rs             # Module root, re-exports
  +-- backends.rs        # PrivacyTier (Sovereign/Private/Standard), BackendSelector
  +-- router.rs          # SpilloverRouter (local-first, remote fallback)
  +-- failover.rs        # FailoverManager (streaming recovery)
  +-- circuit_breaker.rs # CostCircuitBreaker (budget enforcement)
  +-- context.rs         # ContextManager (token counting, truncation)
  +-- templates.rs       # ChatTemplateEngine (ChatML/Llama/Mistral/Alpaca/Vicuna/Raw)
  +-- banco/             # Banco HTTP API (feature-gated, see §5)
```

### Privacy Tiers

| Tier | Data Location | Allowed Backends | Use Case |
|------|--------------|------------------|----------|
| **Sovereign** | Local only | realizar, ollama, llamacpp, llamafile, candle, vllm, tgi, localai | GDPR, classified data |
| **Private** | VPC/dedicated | local + AzureOpenAI, AwsBedrock, GoogleVertex, AwsLambda | Enterprise |
| **Standard** | Any | All backends | General use |

### Failover Strategy

```
Request -> Primary Backend (realizar local)
              |
              +-- Success -> Return
              |
              +-- Failure -> Circuit Breaker check
                               |
                               +-- Budget OK -> Spillover to remote
                               |
                               +-- Budget exceeded -> Reject
```

---

## 2. HuggingFace Integration

### CLI Interface

```bash
# Search
batuta hf search models "llama 7b" --task text-generation
batuta hf search datasets "common voice" --language en --size ">1GB"
batuta hf search spaces "gradio" --sdk gradio

# Info
batuta hf info model "meta-llama/Llama-2-7b-hf"
batuta hf info dataset "mozilla-foundation/common_voice_13_0"

# Pull
batuta hf pull model "TheBloke/Llama-2-7B-GGUF" --quantization Q4_K_M
batuta hf pull model "mistralai/Mistral-7B-v0.1" --format gguf --output ./models/
batuta hf pull dataset "squad" --split train --output ./data/

# Push
batuta hf push model ./my-model --repo "myorg/my-classifier" --format safetensors
batuta hf push dataset ./data/processed --repo "myorg/my-dataset"

# Ecosystem view
batuta hf tree                    # Full ecosystem
batuta hf tree --integration      # PAIML integration mapping
batuta hf tree --platform models  # Filter by category
```

### Ecosystem Taxonomy (50+ Components)

| Category | Examples | PAIML Mapping |
|----------|---------|---------------|
| Libraries | transformers, datasets, tokenizers | aprender, alimentar |
| Inference | TGI, Inference API, Optimum | realizar |
| Training | Accelerate, PEFT, TRL | entrenar |
| Formats | SafeTensors, GGUF | APR v2, GGUF |
| Tools | Hub, Evaluate, Gradio | pacha, pmat, presentar |

### PAIML Integration Mapping

| HuggingFace | PAIML Equivalent | Notes |
|-------------|-----------------|-------|
| `transformers` | `aprender` + `realizar` | Training + inference split |
| `datasets` | `alimentar` | Zero-copy Parquet/Arrow |
| `tokenizers` | `aprender::tokenizer` | Rust-native |
| `safetensors` | `aprender::apr` | APR v2 preferred |
| `accelerate` | `repartir` | Distributed compute |
| `peft` (LoRA) | `entrenar::lora` | LoRA/QLoRA |
| `evaluate` | `pmat` | Quality metrics |
| `gradio` | `presentar` | WASM-first UI |

### Coursera Specialization Support

The HF integration supports "Next-Gen AI Development with Hugging Face" (5 courses, 60 hours):

| Course | HF Components | Batuta Commands |
|--------|--------------|-----------------|
| Foundations | Hub, Transformers | `batuta hf tree`, `batuta hf info` |
| NLP Deep Dive | Tokenizers, Datasets | `batuta hf search`, `batuta hf pull` |
| Computer Vision | ViT, CLIP | `batuta hf pull model` |
| Audio/Multimodal | Whisper, BLIP | `batuta hf pull model --format gguf` |
| Production | TGI, Endpoints | `batuta hf push`, `batuta serve` |

---

## 3. Int8 Rescoring Retriever

### Design

Two-stage scalar int8 rescoring retriever achieving **99% accuracy retention** with **3.66x speedup** and **4x memory reduction**.

### Architecture

```
Stage 1: Fast Retrieval (Int8)
  Query -> Int8 Quantize -> ANN Search (HNSW) -> Candidate Pool (k * multiplier)

Stage 2: Precise Rescoring (Mixed Precision)
  f32 Query -> Dot Product (f32 x i8) -> Rerank Top-K -> Final Results
```

### Mathematical Foundation

**Stage 1 (Int8 Approximate):**
- Quantize: `Q(x) = round(x / scale) + zero_point`
- Scale: `(max(x) - min(x)) / 255`
- Compute approximate scores: `s_approx(q, d) = q_i8 . d_i8`
- Retrieve top-m candidates where m = k * multiplier

**Stage 2 (Float32 Rescoring):**
- Precise scores: `s_precise(q, c) = q . dequant(c_i8)`
- Return top-k from candidates

### Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Accuracy retention | >= 99% vs full f32 | NDCG@10 comparison |
| Speed improvement | >= 3x | Throughput benchmark |
| Memory reduction | >= 4x (f32 -> i8) | RSS measurement |
| Quantization error | < 2% MSE | Error bound validation |

### SIMD Implementation

Int8 dot product uses SIMD acceleration via trueno:
- **AVX-512 VNNI**: `_mm512_dpbusd_epi32` (64 int8 ops per cycle)
- **AVX2**: `_mm256_maddubs_epi16` fallback
- **NEON**: `vdotq_s32` on ARM

### Calibration

Per-vector symmetric quantization with calibration dataset:
1. Compute per-dimension statistics on representative sample
2. Select scale/zero_point to minimize quantization error
3. Validate error bounds (Poka-Yoke: reject if MSE > threshold)
4. Monitor runtime accuracy drift (Kaizen: recalibrate if NDCG drops)

### Key Citations

| Topic | Reference |
|-------|-----------|
| Quantization theory | Jacob et al. (2018), CVPR |
| Embedding quantization | Xiao et al. (2023), SmoothQuant, ICML |
| Two-stage retrieval | Nogueira & Cho (2019), BERT reranking |
| HNSW index | Malkov & Yashunin (2020), IEEE TPAMI |
| RRF fusion | Cormack et al. (2009), SIGIR |
| BM25 | Robertson & Zaragoza (2009) |

---

## 4. Serving CLI

```bash
# Realizar model server (existing)
batuta serve --model ./model.gguf --port 8080
batuta serve --model ./model.apr --privacy sovereign

# Banco AI workbench (new, Phase 1 complete)
batuta serve --banco --port 8090
batuta serve --banco --host 0.0.0.0 --port 8090

# Health check
batuta serve health

# Benchmark
batuta serve bench --tokens 100 --batch-size 32
```

---

## 5. Banco: Local-First AI Workbench

### Vision

Banco is a self-contained AI studio that ships as a single command: `batuta serve --banco`. It replaces the patchwork of cloud dashboards, Jupyter notebooks, and CLI scripts with one local-first HTTP service backed by the entire Sovereign AI Stack.

**What it replaces:**

| Today | Banco |
|-------|-------|
| OpenAI API + API keys + cloud billing | Local realizar inference, zero egress |
| Jupyter + MLflow + W&B | Integrated training, experiments, data management |
| Multiple tools, multiple accounts | One binary, one port, one privacy policy |
| Data leaves your machine | Sovereign tier: nothing leaves localhost |

**Why it matters:** Every other AI workbench is cloud-first with a local afterthought. Banco is local-first with cloud spillover only when the user explicitly opts in via privacy tiers. The same binary that serves a solo developer on a laptop scales to a team via Standard tier with remote backend spillover.

**End state:** Run `batuta serve --banco`, open a browser, and you have chat, training, data management, experiment tracking, and model registry — all Rust, all local, all sovereign.

### Phase Roadmap

| Phase | Scope | Ticket | Status | Sub-spec |
|-------|-------|--------|--------|----------|
| **Phase 1** | HTTP API skeleton: health, models, chat completions (echo+SSE), system info | PMAT-057 | **Complete** | [banco-phase1.md](banco-phase1.md) |
| **Phase 2a** | Model slot, load/unload/status, inference params, GGUF metadata | PMAT-069..074 | **Complete** | [banco-phase2.md](banco-phase2.md) |
| **Phase 2b** | Inference loop, real tokens, streaming, tokenizer, embeddings, Ollama generate | PMAT-077..082 | **Complete** | [banco-phase2.md](banco-phase2.md) |
| **Phase 3** | Files, recipes, RAG, eval, training, merge, experiments, batch (63 endpoints) | PMAT-083..104 | **Complete** | [banco-phase3.md](banco-phase3.md) |
| **Phase 4** | API: MCP, tools, audio, metrics, auth, probes (82 endpoints) | PMAT-105..115 | **API Complete, UI NOT DONE** | [banco-phase4.md](banco-phase4.md) |
| **Phase 5a** | Fix: APR loading, BPE tokenizer, L2+L4 tests, honest labeling, zero-JS UI | PMAT-116..125 | **Complete** (432 tests, zero-JS, batteries-included) | [banco-falsification-report.md](banco-falsification-report.md) |
| **Phase 5b** | Build: presentar WASM UI (7 screens), TUI dashboard | — | **BLOCKED — P1** | [banco-ux.md](banco-ux.md) |
| Phase 6 | Media (rmedia), simulation (simular), games (jugar), education (profesor) | — | Planned | — |

### Architecture

```
batuta serve --banco --port 8090
  │
  ├── axum Router (82 endpoints, 4 protocol layers)
  │     ├── Core:     /health /models /system
  │     ├── Chat:     /chat/completions (sync+SSE, real inference)
  │     ├── Data:     /tokenize /detokenize /embeddings
  │     ├── Models:   /models/load|unload|status
  │     ├── Params:   /chat/parameters (GET/PUT)
  │     ├── Convos:   /conversations (CRUD + export/import)
  │     ├── Presets:  /prompts (CRUD + @preset: expansion)
  │     ├── Files:    /data/upload|files (content-addressable)
  │     ├── Recipes:  /data/recipes (chunk/filter/format/dedup)
  │     ├── Datasets: /data/datasets (preview)
  │     ├── RAG:      /rag/index|status (BM25 retrieval)
  │     ├── OpenAI:   /v1/* (SDK compatible)
  │     └── Ollama:   /api/generate|chat|tags|show
  │
  ├── Middleware: audit logging → API key auth → privacy+CORS
  │
  └── BancoState = Arc<BancoStateInner> (19 fields)
        ├── BackendSelector      Privacy-aware routing
        ├── SpilloverRouter      Heijunka load leveling
        ├── CostCircuitBreaker   Muda budget enforcement
        ├── ContextManager       Token counting + truncation
        ├── ChatTemplateEngine   6 template formats
        ├── ConversationStore    In-memory + disk JSONL
        ├── PromptStore          3 built-in + custom presets
        ├── AuthStore            Local/ApiKey modes
        ├── ModelSlot            Arc<OwnedQuantizedModel> + vocab + BPE tokenizer
        ├── InferenceParams      Temperature, top_k, max_tokens (RwLock)
        ├── FileStore            Content-hash dedup file storage
        ├── RecipeStore          Declarative data pipelines
        ├── RagIndex             BM25 inverted index for doc search
        ├── EvalStore            Perplexity evaluation runs
        ├── TrainingStore        Training runs + metrics
        ├── ExperimentStore      Experiment tracking + comparison
        ├── BatchStore           Multi-prompt batch processing
        ├── ToolRegistry         Built-in + custom tools
        ├── MetricsCollector     Prometheus-compatible metrics
        ├── AuditLog             Request audit trail (disk-backed)
        └── EventBus             WebSocket event broadcasting
```

### Design Principles

1. **Reuse over recreate**: All orchestration logic comes from the existing serve module. Banco is a thin HTTP layer on top.
2. **Feature-gated**: The `banco` Cargo feature keeps axum/tower deps optional. Default builds unaffected.
3. **Submodule of serve**: `src/serve/banco/` not a top-level module. Banco IS serving.
4. **Arc wrapping**: SpilloverRouter and CostCircuitBreaker use atomics — not Clone. Arc is required for axum state.
5. **Test at every level**: L1 unit tests use `tower::ServiceExt::oneshot()` for speed. L2 API tests use real TCP servers. L4 browser tests use probar (jugar-probar) CDP automation to validate the full user experience. See [banco-testing.md](banco-testing.md).
6. **Zero JavaScript — Rust only**: All UI rendering via presentar (WASM-first framework) and all testing via probar (Playwright replacement). No `.js`/`.ts` files, no npm, no Node.js. See §Zero-JS Policy below.
7. **apr-cli as orchestration entry point**: `apr-cli` (from aprender) is the single CLI entry point for all model operations — import, convert, profile, bench, serve. Banco delegates model management (format conversion, quantization, tokenizer loading) to apr-cli's proven pipelines rather than reimplementing them. The `batuta serve --banco` path uses the same `BpeTokenizer::from_huggingface()`, `OwnedQuantizedModel::from_apr()`, and `MappedGGUFModel::from_path()` APIs that apr-cli orchestrates.

### Zero-JavaScript Policy

```
❌ FORBIDDEN                              ✅ REQUIRED
────────────────────────────────────────────────────────────
• .js / .ts files                         • Pure Rust → wasm32-unknown-unknown
• npm / node_modules / package.json       • presentar (WASM-first UI framework)
• Inline <script> tags with JS            • probar (CDP browser testing, zero JS)
• Any JavaScript bundler                  • web-sys Rust bindings where needed
• Playwright/Puppeteer/Selenium           • jugar-probar Locator + expect API
```

**Status:** `src/serve/banco/ui.rs` inline JavaScript **ELIMINATED**. The browser UI now uses server-rendered HTML with `<form method="POST" action="/ui/chat">` — zero `<script>` tags. Full presentar WASM widget is Phase 5b.

**Replacement plan:**
1. ~~Delete inline `<script>` from `ui.rs`~~ **DONE** — replaced with SSR `<form>` (zero JS)
2. Build banco chat widget using `presentar-core` + `presentar-widgets` (Rust) — Phase 5b
3. Compile to `wasm32-unknown-unknown` via `presentar-cli bundle` — Phase 5b
4. Embed `.wasm` binary in banco binary via `include_bytes!` — Phase 5b
5. Serve from `/assets/banco.wasm` + minimal HTML loader — Phase 5b

**Stack alignment:**
- **presentar** (0.3.x): WASM-first UI framework, YAML-driven, 60fps WebGPU rendering
- **presentar-terminal** (0.3.x): TUI surface for `batuta banco --tui`
- **probar** / **jugar-probar** (1.0.x): Playwright replacement — CDP browser automation, zero JS
- **probar** features: `browser` (CDP), `tui` (MockTty), `runtime` (wasmtime)

**Rationale** (from probar CLAUDE.md): "JavaScript introduces non-determinism and GC pauses." The Sovereign AI Stack is deterministic Rust all the way down — the UI surface must not be an exception.

### Sovereign Stack Integration Map

Banco is the **HTTP surface** for the entire Sovereign AI Stack. Every stack crate maps to a Banco feature:

| Stack Crate | Banco Feature | Endpoints | Status |
|-------------|--------------|-----------|--------|
| **realizar** | Inference | `/api/v1/chat/completions`, `/api/v1/models/*` | **Complete** — GGUF + APR loading |
| **aprender** | BPE Tokenizer | tokenize/detokenize, eval, embeddings | **Complete** — proper merge rules, default in banco |
| **entrenar** | Training + Merge | `/api/v1/train/*`, `/api/v1/models/merge` | **API Complete** — real merge API, real loss via forward pass (first metric), remaining steps simulated |
| **alimentar** | Data loading | `/api/v1/data/upload`, recipes, schema detection | **Complete** — Arrow parsing, default in banco |
| **trueno** | SIMD compute | Tensor ops underlying all inference/training | Implicit |
| **trueno-rag** | RAG pipeline | `/api/v1/rag/*`, chat with `rag: true` | **Complete** |
| **trueno-db** | Analytics | Experiment tracking, metrics storage | Not wired |
| **repartir** | Distributed | Multi-GPU training, batch inference | Not wired |
| **pacha** | Registry | `/api/v1/models/pull` (pacha:// URIs) | **Complete** |
| **whisper-apr** | Speech | `/api/v1/audio/transcriptions` | **Complete** |
| **presentar** | UI | Browser WASM workbench | **Zero-JS SSR** (inline JS eliminated, presentar WASM is Phase 5b) |
| **forjar** | IaC | Provisioning, deployment | Not wired |
| **probar** | Testing | Playwright-replacement: CDP browser E2E, load testing | **L1 (358) + L2 (70) + L4 (2) complete** |
| **pforge** | MCP | Model Context Protocol server | **Complete** |
| **rmedia** | Video | Media processing pipeline | Not wired |
| **simular** | Simulation | Monte Carlo, optimization | Not wired |
| **provable-contracts** | Verification | Kani harnesses, formal proofs | Cross-cutting |

**Sovereignty principle:** In Sovereign mode, Banco uses ONLY local crates. No cloud API, no telemetry, no data egress. The full stack runs on a single machine with zero network dependency.

### Current Status (Phase 5a In Progress — PMAT-123)

- **82 endpoints**, 4 protocols (native, OpenAI, Ollama, MCP)
- **356 L1 + 73 L2 + 2 L4 = 431 tests** passing, 0 failures (74/74 routes, **100% L2 coverage**)
- **`banco` feature is batteries-included**: realizar (inference) + aprender (BPE) + alimentar (Arrow) + entrenar (training/merge). Just `--features banco` for everything.
- **Honest labeling**: training and merge responses include `simulated: true`
- **Tokenizer status**: reported in `/models/status`, `/system`, and startup banner
- Zero clippy warnings, all files under 500 lines
- **82 endpoints** (78 routes) across 24 handler files, 53 source files, 27 test modules
- **4 protocol layers**: Banco native, OpenAI, MCP (JSON-RPC), Ollama
- Browser UI: embedded chat SPA at `/`, WebSocket real-time events (9 event types)
- MCP: JSON-RPC 2.0 at `/api/v1/mcp` (Claude Desktop/Cursor)
- OpenAI compat: `/v1/models`, `/v1/models/:id`, `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`
- Tool calling: calculator, code_execution, web_search + custom + self-healing retry
- Audio: whisper-apr speech-to-text
- Chat: file attachments, OpenAI tool calling, RAG context injection
- Model: merge (TIES/DARE/SLERP), registry (pacha pull/list/cache)
- Training: entrenar LoRA (5 presets, SSE metrics, export, cosine LR)
- Data: alimentar CSV/JSONL, 7 recipe steps, RAG (trueno-rag BM25)
- Inference: `forward_single_with_cache()` with greedy/top-k sampling
- Persistence: conversations, files, experiments, audit to `~/.banco/`
- Graceful shutdown: Ctrl+C / SIGTERM with connection draining

See [banco-phase3.md](banco-phase3.md) for details. 7 cookbook recipes in `../batuta-cookbook`.
