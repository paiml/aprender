# Banco Comprehensive Falsification Report

> Date: 2026-03-21
> Methodology: Attempt to break every claim in banco-spec.md against source code
> Verdict: **API excellent. Everything else has critical gaps.**

---

## What Is Real (Verified)

| Component | Claim | Verified | Evidence |
|-----------|-------|----------|----------|
| HTTP API | 82 endpoints, 4 protocols | **TRUE** | `grep .route() router.rs` = 82 method bindings |
| Tests L1 | 345 passing | **TRUE** | `cargo test --lib banco` = 345 passed |
| Tests L2 | 25 tests (5 LLM + 20 endpoints) | **TRUE** | `tests/banco_llm.rs` + `tests/banco_endpoints.rs` |
| Stack crates | 10 wired | **TRUE** | realizar, aprender, entrenar, alimentar, trueno-rag, pacha, whisper-apr, pforge/MCP, trueno (2) |
| APR loading | `from_apr()` wired | **TRUE** | `extract_apr_metadata()` in model_slot.rs |
| BPE tokenizer | Proper merge rules, default in banco | **TRUE** | `banco` feature includes `aprender`, `encode_text()` prefers BPE |
| Tokenizer status | Reported in API + banner | **TRUE** | `/models/status`, `/system`, startup banner show bpe/greedy |
| BPE decode | Fixed | **TRUE** | `decode_bpe_text()` converts U+01XX → bytes |
| Clippy | Zero warnings | **TRUE** | `cargo clippy -- -D warnings` clean |
| Files | All < 500 lines | **TRUE** | max 447 (model_slot.rs) |
| SATD | Zero markers | **TRUE** | `grep TODO/FIXME` = 0 in prod code |
| Persistence | Conversations, files, experiments, audit | **TRUE** | `~/.banco/` with disk reload |
| Graceful shutdown | Ctrl+C/SIGTERM | **TRUE** | `shutdown_signal()` in server.rs |

---

## What Is Broken (Falsified)

### 1. APR Native Format — ~~DEAD PATH~~ FIXED (PMAT-117)

| Claim | Reality | Evidence |
|-------|---------|---------|
| "APR v2 preferred" (spec line 116) | **FIXED** — APR models load via `extract_apr_metadata()` | `model_slot.rs:285-329` calls `OwnedQuantizedModel::from_apr()` |
| `batuta serve --model ./model.apr` | **FIXED** — Loads metadata + quantized model for inference | `extract_model_metadata()` dispatches by format |
| Export format `apr` in training | Export path generated but no APR writer wired | `handlers_train.rs` produces path string only |
| APR in merge output | Merge UI shows APR option but no APR serializer | `handlers_merge.rs` — format string only |

APR loading wired in PMAT-117. Export/merge APR serialization still missing.

### 2. Browser UI — CHAT BUBBLE, NOT AI STUDIO

| Claim | Reality |
|-------|---------|
| 7 screens (Chat, Arena, Models, Data, Training, Experiments, System) | 1 screen (Chat only) |
| 80+ presentar widgets | 0 presentar widgets, zero-JS SSR form (inline JS eliminated) |
| YAML-driven layout (`banco-wasm.prs`) | No `.prs` files exist |
| Theming (`~/.banco/theme.yaml`) | Hardcoded CSS variables |
| WCAG AA accessibility | Zero ARIA roles, zero keyboard nav beyond Enter |
| TUI dashboard (`batuta banco --tui`) | Does not exist |
| presentar compiled to wasm32 | No WASM binary built |

See `banco-ux-falsification.md` for full 50+ row breakdown.

### 3. Testing — L1+L2 Done, L3-L4 MISSING

| Level | Spec Claims | Exists | What's Missing |
|-------|-------------|--------|----------------|
| L1 Unit | tower::oneshot | 345 tests | None — but can't catch browser/TCP bugs |
| L2 API | Real TCP + probar LlmClient | **13 tests** (PMAT-116) | Full coverage of all endpoints |
| L3 TUI | probar MockTty + FrameAssertion | 0 tests | No TUI exists to test |
| L4 Browser | probar CDP + Locators | 0 tests | Would have caught "nothing works" |
| Load | probar loadtest | 0 tests | Performance unknown |
| Fuzz | probar InputFuzzer | 0 tests | Crash robustness unknown |
| a11y | probar AccessibilityValidator | 0 tests | WCAG compliance unknown |
| Visual | probar VisualRegressionTester | 0 tests | No baselines |

### 3a. probar LLM Testing Module — ~~NOT USED~~ PARTIALLY USED (PMAT-116)

probar `jugar_probar::llm` is now used in `tests/banco_llm.rs` with 13 L2 integration tests. These validate real TCP connections, chat completion structure, SSE streaming, latency budgets, and more.

**What probar::llm provides:**

| Component | What It Does | Status in Banco |
|-----------|-------------|-----------------|
| `LlmClient` | HTTP client for `/v1/chat/completions` (sync + SSE streaming) | **NOT USED** |
| `LlmClient::health_check()` | Verify server is up before tests | **NOT USED** |
| `LlmClient::wait_ready()` | Poll until server responds | **NOT USED** |
| `LlmClient::chat_completion()` | Send chat request, get timed response | **NOT USED** |
| `LlmClient::chat_completion_stream()` | SSE streaming with per-token timestamps + TTFT | **NOT USED** |
| `LlmAssertion::assert_response_valid()` | Check id, choices, non-empty content | **NOT USED** |
| `LlmAssertion::assert_contains("text")` | Verify response contains substring | **NOT USED** |
| `LlmAssertion::assert_latency_under(5s)` | Latency budget enforcement | **NOT USED** |
| `LlmAssertion::assert_token_count(min, max)` | Token count validation | **NOT USED** |
| `LlmAssertion::assert_matches_pattern(regex)` | Regex pattern matching on output | **NOT USED** |
| `assert_deterministic(responses)` | Same prompt → same output (temperature=0) | **NOT USED** |
| `LoadTest` | Concurrent load test with Poisson/constant rate | **NOT USED** |
| `LoadTestConfig` | concurrency, duration, warmup, SLOs (TTFT, TPOT, latency) | **NOT USED** |
| `LoadTestResult` | p50/p95/p99 latency, throughput RPS, tokens/sec | **NOT USED** |
| `ValidationMode::Basic` | Inline correctness checks during load | **NOT USED** |
| `Scorecard` | Multi-dimensional scoring (runtime, correctness, memory, cold start) | **NOT USED** |
| `to_markdown_table()` | Report generation | **NOT USED** |
| `compare_to_baseline()` | Performance regression detection | **NOT USED** |
| `BenchmarkReport` | Aggregate stats across runs | **NOT USED** |
| `GpuTelemetryCollector` | GPU memory/utilization during load | **NOT USED** |
| `PromptProfile` | Categorized prompt datasets for systematic testing | **NOT USED** |
| `StreamedChatResponse` | TTFT, per-token timestamps, finish_reason | **NOT USED** |

**This is the test framework that would validate Banco end-to-end.** It speaks the same OpenAI protocol Banco serves. A single test using `LlmClient` pointed at a real Banco server would have caught every UAT failure.

### 4. ~~Zero-JavaScript Policy — VIOLATED~~ FIXED (PMAT-125)

| Claim | Reality |
|-------|---------|
| "All UI is Rust" | **FIXED** — `ui.rs` inline JS eliminated, SSR via `handlers_ui.rs` |
| "No .js files" | **TRUE** — zero `<script>` tags in any response |
| presentar WASM | Zero presentar integration (Phase 5b) |
| probar for browser tests | **FIXED** — 2 L4 CDP tests + 72 L2 tests |

### 5. ~~Tokenizer — GREEDY MISMATCH~~ FIXED (PMAT-118/119)

**RESOLVED:** BPE tokenizer from aprender now wired via `ModelSlot::encode_text()`. All handler paths (chat, streaming, eval, embeddings, tokenize, batch, Ollama) prefer BPE merge rules when `tokenizer.json` found. Greedy remains as fallback. `banco` feature includes `aprender` by default. Status reported in `/models/status`, `/system`, and startup banner.

### 6. Integration Map — Updated

| Crate | Status | Evidence |
|-------|--------|----------|
| aprender | **Complete** — BPE tokenizer default in banco | `load_bpe_tokenizer()`, `encode_text()` |
| entrenar | **Partial** — LoRA config wired, metrics **simulated** (marked `simulated: true`) | `training_engine.rs`, `handlers_train.rs` |
| alimentar | **Complete** — Arrow CSV/JSON parsing default in banco | `handlers_data.rs` with `cfg(alimentar)` |
| realizarr | **Complete** — inference via `OwnedQuantizedModel` | GGUF + APR loading |
| presentar | Not wired — zero presentar | Correct, blocked on availability |
| trueno-db | Not wired — JSONL persistence | Never wired |
| repartir | Not wired | Never wired |
| forjar | Not wired | Never wired |

---

## Reprioritized Work (Honest)

### P0 — Fix What's Broken (blocks real usage)

| # | Item | Status | Uses |
|---|------|--------|------|
| 1 | **Write probar L2 LLM tests** | **DONE** (PMAT-116) — 13 tests in `tests/banco_llm.rs` | `jugar_probar::llm::{LlmClient, LlmAssertion}` |
| 2 | **Write probar L4 browser tests** — CDP headless Chrome | TODO — needs Chrome/CDP | `jugar_probar::browser::{Browser, Page, Locator}` |
| 3 | **Wire APR model loading** | **DONE** (PMAT-117) — `extract_apr_metadata()` in model_slot.rs | `realizar::gguf::OwnedQuantizedModel::from_apr()` |
| 4 | **Wire proper BPE tokenizer** from aprender | **DONE** (PMAT-118) — `load_bpe_tokenizer()` + `encode_text()` | `aprender::text::bpe::BpeTokenizer` |
| 5 | ~~Fix browser UI chat~~ | **DONE** (PMAT-125) — SSR form, zero `<script>` tags | `handlers_ui.rs` |

### P1 — Build Real UI (blocks "AI studio" claim)

| # | Item | Why P1 |
|---|------|--------|
| 6 | ~~Replace inline JS~~ → presentar WASM (Phase 5b) | JS eliminated, WASM deferred |
| 7 | **Build Training screen** (loss curve, config) | Training API exists, no UI |
| 8 | **Build Models screen** (load/unload, VRAM) | Model API exists, no UI |
| 9 | **Build Data screen** (upload, recipes, RAG) | Data API exists, no UI |
| 10 | **Build System screen** (metrics, probes) | Metrics API exists, no UI |

### P2 — Production Hardening

| # | Item | Why P2 |
|---|------|--------|
| 11 | **Wire real entrenar training** (not simulated) | Training metrics are fake |
| 12 | **Wire alimentar Arrow parsing** (not fallback) | CSV parsing uses line splitter |
| 13 | **Build TUI dashboard** via presentar-terminal | Spec promises, doesn't exist |
| 14 | **Probar load testing** | Performance unknown |
| 15 | **Probar visual regression baselines** | No visual QA |

### P3 — Remaining Stack Integration

| # | Item |
|---|------|
| 16 | trueno-db for experiment persistence (replace JSONL) |
| 17 | repartir for multi-GPU training |
| 18 | forjar for IaC deployment |
| 19 | Accessibility (ARIA, keyboard nav, WCAG AA) |
| 20 | Theming (YAML config, light/dark) |
