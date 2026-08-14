//! API Tests
//!
//! Split into parts for PMAT compliance (<2000 lines per file).
//!
//! Part organization:
//! - part_01: Unit tests, clean_chat_output, health/metrics endpoints (IMP-144-152)
//! - part_02: Generate endpoint tests, streaming tests
//! - part_03: Chat completion tests, OpenAI compatibility
//! - part_04: GPU inference tests (IMP-116+)
//! - part_05: Additional coverage tests
//! - part_06: Error response coverage (PMAT-803)
//! - part_07: Realize handlers coverage (Phase 37 - Scenario Blitz)
//! - part_08: OpenAI handlers coverage
//! - part_09: OpenAI handlers extended coverage
//! - part_10: Realize handlers extended coverage (ModelLineage, ReloadResponse, etc.)
//! - part_11: GPU handlers coverage (GpuBatchRequest, GpuBatchResponse, BatchConfig, etc.)
//! - part_12: OpenAI/Realize handlers - Request/Response type serialization
//! - part_13: OpenAI/Realize handlers - HTTP endpoint error paths and streaming
//! - part_14: Additional coverage tests
//! - part_15: T-COV-95 Directive 2: In-Process API Falsification (GPU/CUDA/quantized paths)
//! - part_16: T-COV-95 Popper Phase 2: Combinatorial API Sweep (stream/temp/max_tokens/invalid)

mod tests_01;
mod tests_02;
mod imp_134c;
mod chat_delta;
mod openai_models;
mod tests_06;
mod tests_07;
mod tests_08;
mod tests_09;
mod tests_10;
mod tests_11;
mod completion_request;
mod completions_invalid;
mod chat_completion;
mod tests_15;
mod tests_16;
mod gpu_warmup;
mod serde; // T-COV-95 Coverage Bridge B2+B3 (GPU handlers, Realize/OpenAI handlers, AppState)
mod tests_19; // T-COV-95 Deep Coverage Bridge (BatchConfig, ContinuousBatchResponse, streaming types, endpoints)
mod context_window_serde; // T-COV-95 Deep Coverage Bridge (ContextWindow, format_chat, clean_chat, HTTP handlers, serde)
mod build_trace; // T-COV-95 Extended Coverage (build_trace_data, streaming types, request/response serde)
mod predict_request; // T-COV-95 APR handlers coverage (predict, explain, audit, serde, error paths)
mod tests_23; // T-COV-95 gpu_handlers + realize_handlers coverage (BatchConfig, ContextWindow, format_chat)
mod tests_24; // T-COV-95 Protocol Falsification: Potemkin Village GPU Mocks
mod tests_25; // T-COV-95 Chaotic Citizens: GPU Batch Resilience Falsification
mod tests_26; // T-COV-95 Interleaved Chaos: GPU Batch Processor Saturation
mod tests_27; // T-COV-95 Generative Falsification: Proptest API Request Assault
mod tests_28; // Coverage: realize_handlers pure functions, ContextWindow, clean_chat, build_trace_data, serde
mod chat_template_contract; // PMAT-187: chat-template-v1.yaml contract enforcement (FALSIFY-CT-002)
mod embeddings_pmat803; // PMAT-803: model-backed embeddings (semantic-similarity falsifier, dim==hidden_size)
mod sse_stream_whitespace; // Dogfood 0.63.0: SSE deltas must reassemble with whitespace intact
mod native_routes_2376; // aprender#2376: native routes on a quantized server, KV-cache budget, sampling fields
mod router_flags; // --no-cors / --no-metrics must change HTTP behaviour, not just the banner
mod ollama_compat_http; // Dogfood 0.63.0 (#2396/#2402): /api/tags|show|version routed, stream:true is NDJSON, /realize/* stops fabricating
mod embed_and_envelope_2376; // aprender#2376(1 seventh route, 7, 8) + #2396(2): embeddings on a quantized server, one error envelope, / and /ready
mod explain_2375; // aprender#2375(2): /v1/explain must not fabricate SHAP values and a 0.95 prediction
mod openai_compat_2375; // Dogfood 0.63.0 (#2375): /v1/completions streams, finish_reason is measured, `n` is honoured or refused, /v1/predict stops lying
mod route_surface_2376; // aprender#2376(7,8): advertised surface == mounted surface; every error body is a JSON envelope
mod stream_and_metrics_2375; // aprender#2375(1 regression, 4, 7) + temperature:0 — streaming chat through the real router, /v1/metrics measures
mod batch_completions_tokenizer_2465; // aprender#2465(3): /v1/batch/completions tokenized from UTF-8 byte values, not tokens
