# Banco Phase 2: Realizar Inference + Model Management

> Parent: [banco-spec.md](banco-spec.md) §5
> Tickets: PMAT-057..082
> Status: **Complete** — inference loop, tokenizer, embeddings, Ollama generate, conversation export/import
> Depends on: Phase 1 (PMAT-057, complete)

---

## Scope

Replace the Phase 1 echo handler with actual realizar inference. Add model loading, hot-swap, arena comparison, and inference parameter tuning. After this phase, `POST /api/v1/chat/completions` returns real generated tokens from a local GGUF/APR model.

## Prerequisites

- Phase 1 complete (PMAT-057)
- `realizar` crate (0.8.x, already optional dep)
- `aprender` crate (0.27.x, for tokenizer + APR format)
- A test model (TinyLlama GGUF) for integration tests

## What Changes

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| `chat_completions_handler` | Echo routing decision | Forward to realizar |
| `BancoStateInner` | No model | `ModelSlot` with hot-swap |
| `finish_reason` | `"dry_run"` | `"stop"` / `"length"` |
| `usage.completion_tokens` | Estimated from echo | Actual token count |
| Feature flag | `banco` | `banco` + `inference` |
| SSE streaming | Simulated chunks | Real token-by-token |
| Endpoints | 4 | 4 + model management + arena |

## New Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/v1/models/load` | Load model from path or pacha URI |
| POST | `/api/v1/models/unload` | Unload current model, free VRAM |
| GET | `/api/v1/models/status` | Model loaded, VRAM usage, quant format |
| POST | `/api/v1/chat/completions` | **Modified**: real inference when model loaded |
| POST | `/api/v1/arena` | Side-by-side comparison of two models |
| PUT | `/api/v1/chat/parameters` | Update default temperature/top_p/template |
| GET | `/api/v1/chat/parameters` | Read current inference parameters |

## Model Management

### Loading

```bash
# At startup via CLI
batuta serve --banco --model ./tinyllama.gguf --port 8090
batuta serve --banco --model pacha://tinyllama:1b --port 8090

# At runtime via API
curl -X POST http://localhost:8090/api/v1/models/load \
  -d '{"model": "./phi-3-mini.gguf"}'

# Pacha URI resolution
curl -X POST http://localhost:8090/api/v1/models/load \
  -d '{"model": "pacha://llama3:8b"}'
```

### Model Slot Architecture

```rust
pub struct ModelSlot {
    pub model: Arc<OwnedQuantizedModel>,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub format: ModelFormat,        // GGUF, APR, SafeTensors
    pub quant: String,              // "Q4_K_M", "Q6_K", "F16"
    pub loaded_at: Instant,
    pub vram_bytes: u64,
}

pub struct BancoStateInner {
    // Phase 1 (unchanged)
    pub backend_selector: BackendSelector,
    pub router: SpilloverRouter,
    pub circuit_breaker: CostCircuitBreaker,
    pub context_manager: ContextManager,
    pub template_engine: ChatTemplateEngine,
    pub privacy_tier: PrivacyTier,
    pub start_time: Instant,
    // Phase 2
    pub primary_model: RwLock<Option<ModelSlot>>,
    pub arena_model: RwLock<Option<ModelSlot>>,  // second slot for comparison
    pub inference_params: RwLock<InferenceParams>,
}

pub struct InferenceParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub template_format: TemplateFormat,  // auto-detect or explicit
}
```

### Hot-Swap

Model loading acquires a write lock on `primary_model`, drops the old model (freeing VRAM), loads the new one. In-flight requests on the old model continue via Arc reference counting — they hold their own Arc clone.

### No-Model Fallback

When `primary_model` is `None`, chat completions falls back to Phase 1 echo mode. This keeps the server useful for health checks and system info even without a model.

## Arena: Side-by-Side Comparison

```
POST /api/v1/arena
{
  "messages": [{"role": "User", "content": "Explain gravity"}],
  "max_tokens": 256
}

Response:
{
  "primary": { /* BancoChatResponse from primary_model */ },
  "arena":   { /* BancoChatResponse from arena_model */ },
  "metadata": {
    "primary_model": "phi-3-mini-Q4_K_M",
    "arena_model": "llama3-8b-Q6_K",
    "primary_latency_ms": 1420,
    "arena_latency_ms": 2180
  }
}
```

Load arena model:
```bash
curl -X POST http://localhost:8090/api/v1/models/load \
  -d '{"model": "./llama3.gguf", "slot": "arena"}'
```

## Inference Pipeline

```
POST /api/v1/chat/completions
  |
  +-- Validate (ContextManager)
  +-- Check budget (CostCircuitBreaker)
  +-- Route (SpilloverRouter)
  +-- Acquire read lock on primary_model
  +-- Apply template (ChatTemplateEngine)
  +-- Tokenize prompt
  +-- Inference loop:
  |     for pos in prompt_len..max_tokens:
  |       logits = model.forward_single_with_scratch(token, pos, &scratch)
  |       next = sample(logits, params)
  |       if next == eos: break
  |       if streaming: yield SSE chunk
  |       tokens.push(next)
  +-- Decode tokens → text
  +-- Record cost (CostCircuitBreaker)
  +-- Return BancoChatResponse
```

## Streaming

Phase 1 yields pre-built string chunks. Phase 2 yields real tokens:

```rust
let stream = async_stream::stream! {
    let model = state.primary_model.read()?.as_ref()?.clone();
    let mut scratch = InferenceScratchBuffer::new(&model.model);
    let mut pos = prompt_tokens.len();
    loop {
        let logits = model.model.forward_single_with_scratch(token, pos, &mut scratch)?;
        let next_token = sample(&logits, &params);
        if next_token == eos_token || pos >= max_tokens { break; }
        let text = model.tokenizer.decode(&[next_token]);
        yield Ok(Event::default().data(chunk_json(&text)));
        pos += 1;
    }
};
```

## Inference Parameter Tuning

```bash
# Read current params
curl http://localhost:8090/api/v1/chat/parameters

# Update (persists until server restart)
curl -X PUT http://localhost:8090/api/v1/chat/parameters \
  -d '{"temperature": 0.3, "top_k": 40, "template_format": "ChatML"}'
```

Per-request overrides via the existing `BancoChatRequest` fields take precedence over server defaults.

## `/api/v1/system` Updates

```json
{
  "privacy_tier": "Standard",
  "backends": ["Realizar", "Groq", "Together"],
  "gpu_available": true,
  "version": "0.7.2",
  "model_loaded": true,
  "model_id": "phi-3-mini-Q4_K_M",
  "model_format": "GGUF",
  "vram_used_mb": 2048,
  "arena_loaded": false
}
```

## Speculative Decoding

Realizar has `cuda/speculative.rs` — self-speculative decoding using the same model with early-exit layers as draft. Expose as an inference option:

```json
{
  "messages": [...],
  "speculative": true,
  "speculative_tokens": 4
}
```

When enabled, generates `speculative_tokens` draft tokens per step, then verifies in a single forward pass. 2-3x speedup on GPU with no quality loss (mathematically equivalent).

Only available when model is loaded on CUDA. CPU fallback: ignored (standard autoregressive).

## Prefix Caching

Realizar has `paged_kv/mod_compute_prefix.rs` — caches computed KV values for common prompt prefixes (system prompts). When multiple conversations share the same system prompt, subsequent requests skip recomputation.

```
First request:  [system prompt (512 tokens)] + [user message] → compute all KV
Second request: [system prompt (512 tokens)] + [different user message] → cache hit, only compute user part
```

Enabled by default. Cache size configurable:

```json
PUT /api/v1/chat/parameters
{"prefix_cache_size_mb": 256}
```

Reports in `/api/v1/system`:
```json
{"prefix_cache_hits": 142, "prefix_cache_misses": 8, "prefix_cache_size_mb": 128}
```

## Continuous Batching

Realizar has a continuous batching scheduler (lib.rs §8, based on vLLM/Orca). Phase 2 starts single-request, then enables batching for concurrent requests:

```
Concurrent requests → Scheduler groups into micro-batches
  → Forward pass processes multiple sequences simultaneously
  → Individual SSE streams demuxed from batch output
```

Config:
```json
PUT /api/v1/chat/parameters
{"max_batch_size": 8, "batch_timeout_ms": 50}
```

When `max_batch_size = 1` (default), behaves as single-request. Increase for throughput when serving multiple users.

## Structured Output

JSON schema constrained decoding — forces the model to produce valid JSON matching a schema:

```json
{
  "messages": [{"role": "User", "content": "Extract name and age from: John is 30."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  }
}
```

Implementation: grammar-constrained sampling that masks logits for tokens that would violate the schema at each generation step. Follows the OpenAI `response_format` API.

Also supports `"type": "json_object"` (any valid JSON) and `"type": "regex"` (match a regex pattern).

## Open Questions (Resolved)

1. **Tokenizer source**: ~~GGUF models include vocab in metadata.~~ **RESOLVED (PMAT-077)**: GGUF vocab extracted via `MappedGGUFModel.vocabulary()`, stored in `ModelSlot.vocab`. Greedy longest-match encoding in `inference::encode_prompt()`. Tokenize/detokenize endpoints use real vocab when model loaded.
2. **Sampling**: ~~Does realizar expose softmax → multinomial?~~ **RESOLVED (PMAT-077)**: Implemented in `inference::sample_token()` — temperature scaling + top-k + softmax over candidates. Greedy (argmax) when temperature=0.
3. **KV cache pool**: Per-request `OwnedQuantizedKVCache::new()` for Phase 2b (simple). Shared PagedAttention pool deferred to continuous batching work.
4. **Grammar engine**: Deferred. `ResponseFormat` types (JsonObject, JsonSchema, Regex) are defined but grammar-constrained sampling not yet implemented.

## Test Strategy

| Test | Type | What |
|------|------|------|
| Mock model returns fixed logits | Unit | Verify decode + response format |
| Load TinyLlama, generate 10 tokens | Integration | End-to-end inference |
| Stream 5 tokens, verify SSE format | Integration | OpenAI chunk compat |
| No model → echo fallback | Unit | Backward compat |
| Hot-swap model during idle | Integration | RwLock + Arc refcount |
| Arena with two models | Integration | Parallel inference + comparison |
| Parameter update persists | Unit | RwLock state mutation |
| VRAM reported in /system | Unit | ModelSlot.vram_bytes |
| Speculative decoding matches standard output | Integration | Equivalence check |
| Prefix cache hit on repeated system prompt | Integration | Cache stats increment |
| Two concurrent requests batched | Integration | Batch scheduler |
| JSON schema output is valid | Integration | Parse against schema |
| Regex constrained output matches | Unit | Grammar engine |
