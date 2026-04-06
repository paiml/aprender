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
