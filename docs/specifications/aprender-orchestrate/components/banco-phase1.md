# Banco Phase 1: HTTP API Foundation

> Parent: [banco-spec.md](banco-spec.md) §5
> Ticket: PMAT-057
> Status: **Complete** (foundation for 52 endpoints built on this skeleton)

---

## Scope

API skeleton: health, model listing, chat completions (echo mode with SSE streaming), and system info. No actual inference, no UI, no training — just the HTTP foundation that future phases build on.

## Module Layout

```
src/serve/banco/
  +-- mod.rs              34 lines   Module declarations, test registrations
  +-- types.rs           169 lines   OpenAI-compatible request/response types
  +-- state.rs            94 lines   BancoState = Arc<BancoStateInner>
  +-- middleware.rs        61 lines   Privacy tier Tower middleware
  +-- handlers.rs        223 lines   Endpoint handler functions
  +-- router.rs           24 lines   axum Router wiring
  +-- server.rs           27 lines   TcpListener binding + banner
  +-- types_tests.rs     182 lines   BANCO_TYP_001..006
  +-- state_tests.rs     113 lines   BANCO_STA_001..006
  +-- middleware_tests.rs 142 lines   BANCO_MID_001..003
  +-- handlers_tests.rs  232 lines   BANCO_HDL_001..008
                        ──────────
                        1301 lines total (632 source, 669 test)
```

## Feature Flag

```toml
# Cargo.toml [dependencies]
axum = { version = "0.7", features = ["json"], optional = true }
tower = { version = "0.4", features = ["util"], optional = true }
async-stream = { version = "0.3", optional = true }
tokio-stream = { version = "0.1", optional = true }

# Cargo.toml [features]
banco = ["native", "axum", "tower", "async-stream", "tokio-stream"]
```

Default build is unaffected. Banco deps are fully optional.

## API Types

### Request: `BancoChatRequest`

```json
{
  "model": "llama3",
  "messages": [{"role": "User", "content": "Hello!"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false
}
```

- `model` optional — defaults to `"banco-echo"` in Phase 1
- `messages` reuses `ChatMessage` and `Role` from `crate::serve::templates`
- Defaults: max_tokens=256, temperature=0.7, top_p=1.0, stream=false

### Response: `BancoChatResponse`

```json
{
  "id": "banco-1700000000",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "banco-echo",
  "choices": [{
    "index": 0,
    "message": {"role": "Assistant", "content": "[banco dry-run] route=Local(Realizar) | ..."},
    "finish_reason": "dry_run"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

### SSE Streaming: `BancoChatChunk`

```
data: {"id":"banco-...","choices":[{"delta":{"role":"Assistant"}}]}
data: {"id":"banco-...","choices":[{"delta":{"content":"[banco"}}]}
data: {"id":"banco-...","choices":[{"delta":{"content":" dry-run]"}}]}
data: {"id":"banco-...","choices":[{"delta":{},"finish_reason":"dry_run"}]}
data: [DONE]
```

### Other Responses

| Endpoint | Type | Key Fields |
|----------|------|------------|
| `GET /health` | `HealthResponse` | status, circuit_breaker_state, uptime_secs |
| `GET /api/v1/models` | `ModelsResponse` | object="list", data: Vec\<ModelInfo\> |
| `GET /api/v1/system` | `SystemResponse` | privacy_tier, backends, gpu_available, version |
| Error | `ErrorResponse` | error.message, error.type, error.code |

## Middleware: Privacy Layer

Every response gets an `X-Privacy-Tier` header (`sovereign`, `private`, `standard`).

In Sovereign mode, requests with an `X-Banco-Backend` header hinting at an external backend are rejected with 403. Local backend hints (`realizar`, `ollama`, `llamacpp`, `llamafile`, `candle`, `vllm`, `tgi`, `localai`) are allowed.

## Handlers

| Handler | Route | Method | Behavior |
|---------|-------|--------|----------|
| `health_handler` | `/health` | GET | Circuit breaker state + uptime |
| `models_handler` | `/api/v1/models` | GET | BackendSelector.recommend() as model list |
| `system_handler` | `/api/v1/system` | GET | Privacy tier, backends, GPU, version |
| `chat_completions_handler` | `/api/v1/chat/completions` | POST | Validate → route → echo (Phase 1) |

Chat completions validation pipeline:
1. Empty messages → 400 `invalid_request`
2. Context window exceeded → 400 `context_length_exceeded`
3. Circuit breaker open → 429 `rate_limit`
4. Non-streaming → JSON echo with routing decision description
5. Streaming → SSE with simulated token chunks via `async_stream::stream!`

Phase 1 uses `finish_reason: "dry_run"` to signal no inference occurred.

## CLI Integration

```
batuta serve --banco [--host 127.0.0.1] [--port 8080]
```

The `--banco` flag on `Commands::Serve` dispatches to `batuta::serve::banco::start_server()` via a new tokio runtime. Gated by `#[cfg(feature = "banco")]` — without the feature, prints an error with rebuild instructions.

## Test Matrix

| File | Test IDs | Count | What |
|------|----------|-------|------|
| `types_tests.rs` | BANCO_TYP_001..006 | 12 | Serde roundtrip, defaults, error format |
| `state_tests.rs` | BANCO_STA_001..006 | 8 | State init, health, models, system info |
| `middleware_tests.rs` | BANCO_MID_001..003 | 5 | Privacy header, sovereign rejection |
| `handlers_tests.rs` | BANCO_HDL_001..008 | 7 | Router oneshot tests (no TCP) |
| **Total** | | **32** | All `#[cfg(feature = "banco")]` gated |

All handler tests use `tower::ServiceExt::oneshot()` — no TCP listener needed.

## Verification

```bash
cargo check --features banco
cargo test --features banco --lib banco      # 32/32 pass
cargo clippy --features banco -- -D warnings # zero warnings
```

## Lessons Learned During Implementation

1. **Binary crate vs lib crate**: `main_dispatch.rs` is in the binary crate, so serve module references must use `batuta::serve::banco::` not `crate::serve::banco::`. Other binary-side code (e.g., MCP) already follows this pattern.

2. **Role serialization is PascalCase**: `serde` derives on `Role` produce `"Assistant"` not `"assistant"`. Tests must match. This is a deviation from the OpenAI API (which uses lowercase). Phase 2 should add `#[serde(rename_all = "lowercase")]` to `Role` or use a banco-local role type.

3. **SpilloverRouter and CostCircuitBreaker are not Clone**: They use `AtomicU64`, `AtomicUsize`, and `RwLock` internally. This is why `BancoState = Arc<BancoStateInner>` is required — axum state must be Clone.

4. **axum middleware pattern**: Privacy enforcement uses `axum::middleware::from_fn` with a closure capturing the privacy tier, not a Tower Layer struct. Simpler and sufficient for Phase 1.

5. **Test isolation**: Handler tests use `create_banco_router()` + `oneshot()` directly. No TCP binding, no port conflicts, deterministic. This pattern should continue for Phase 2.

## Design Decisions

- **Echo mode**: Returns routing decision as content, not an error. Lets the full pipeline (validate → route → template → respond) run end-to-end.
- **`Arc<BancoStateInner>`**: Necessary due to atomics in router/circuit breaker.
- **Submodule of serve**: `src/serve/banco/` not top-level. Banco IS serving.
- **Feature-gated**: `banco` flag keeps axum/tower/async-stream optional.
- **Reuse over recreate**: ChatMessage, Role, BackendSelector, SpilloverRouter, CostCircuitBreaker, ContextManager, ChatTemplateEngine all reused from `crate::serve`.
