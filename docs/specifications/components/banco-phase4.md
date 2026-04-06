# Banco Phase 4: Browser UI + Advanced Chat Features

> Parent: [banco-spec.md](banco-spec.md) §5
> Ticket: —
> Status: **In Progress** — Browser UI, WebSocket, tool calling, registry shipped (PMAT-105..108)
> Depends on: Phase 3

---

## Scope

Serve a browser UI from the same binary and add advanced chat features: code execution sandbox, web search with citations, tool calling, file attachments (images/audio/documents), and multimodal inference. After this phase, `batuta serve --banco` is a complete local AI studio with a browser frontend.

## Stack Crates Used

| Crate | Version | Role |
|-------|---------|------|
| `presentar` | 0.3.x | WASM-first UI framework |
| `whisper-apr` | 0.2.x | Speech-to-text for audio input |
| `jugar-probar` | 1.0.x | Sandboxed code execution (browser feature) |
| `trueno-viz` | 0.2.x | Chart rendering for training metrics |

Feature flag: `banco-ui = ["banco-train", "viz", "speech"]`

---

## UI Architecture

### Serving Strategy

The WASM UI is compiled at build time and embedded as static assets in the banco binary. No separate frontend build step at runtime.

```
batuta serve --banco --port 8090
  |
  +-- /                       → SPA index.html (presentar WASM)
  +-- /assets/*               → JS/WASM/CSS bundles
  +-- /api/v1/*               → JSON API (Phases 1-3)
  +-- /api/v1/ws              → WebSocket for real-time updates
```

### UI Screens

| Screen | Routes | Phase 1-3 APIs Used |
|--------|--------|---------------------|
| **Chat** | `/chat` | `/api/v1/chat/completions` (SSE) |
| **Arena** | `/arena` | `/api/v1/arena` |
| **Models** | `/models` | `/api/v1/models/*` |
| **Data** | `/data` | `/api/v1/data/*` |
| **Training** | `/training` | `/api/v1/train/*` |
| **Experiments** | `/experiments` | `/api/v1/experiments/*` |
| **System** | `/system` | `/api/v1/system`, `/health` |

### Real-Time Updates

WebSocket at `/api/v1/ws` pushes:
- Training metrics (mirrors SSE `/api/v1/train/runs/{id}/metrics`)
- Model load/unload events
- Recipe completion notifications
- System health changes (circuit breaker open/close)

This replaces polling and enables the "monitor from your phone" use case.

---

## Advanced Chat Features

### Code Execution Sandbox

Users can request code execution in chat. The model generates code, Banco executes it in a sandbox, and returns the result.

```
POST /api/v1/chat/completions
{
  "messages": [{"role": "User", "content": "Calculate fibonacci(20) in Python"}],
  "tools": [{"type": "code_execution", "languages": ["python", "bash"]}]
}
```

Sandbox via `jugar-probar`:
- Process isolation (separate PID namespace where available)
- Timeout enforcement (default 30s)
- Memory limit (default 256MB)
- No network access
- Stdout/stderr captured and returned as tool result

```json
{
  "choices": [{
    "message": {
      "role": "Assistant",
      "content": "The 20th Fibonacci number is 6765.",
      "tool_calls": [{
        "type": "code_execution",
        "language": "python",
        "code": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\nprint(fib(20))",
        "result": {"stdout": "6765\n", "stderr": "", "exit_code": 0}
      }]
    }
  }]
}
```

### Web Search

Opt-in web search with source citations. Only available in Standard privacy tier (requires network).

```
POST /api/v1/chat/completions
{
  "messages": [{"role": "User", "content": "What happened in tech news today?"}],
  "tools": [{"type": "web_search", "max_results": 5}]
}
```

- Sovereign tier: web_search tool rejected (privacy gate)
- Private tier: search allowed through configured proxy only
- Standard tier: direct search via configurable provider

Response includes citations:

```json
{
  "choices": [{
    "message": {
      "content": "According to recent reports [1][2], ...",
      "citations": [
        {"index": 1, "title": "...", "url": "...", "snippet": "..."},
        {"index": 2, "title": "...", "url": "...", "snippet": "..."}
      ]
    }
  }]
}
```

### Tool Calling with Self-Healing

OpenAI-compatible function calling. When a tool call fails (malformed JSON, wrong params), Banco retries with the error message injected:

```
1. Model generates: {"name": "get_weather", "arguments": {"city": "NYC"}}
2. Tool returns error: "parameter 'location' required, got 'city'"
3. Banco injects error as system message, re-prompts
4. Model corrects: {"name": "get_weather", "arguments": {"location": "NYC"}}
```

Max self-heal retries: 3 (configurable).

### File Attachments

| Type | Processing | Model Requirement |
|------|-----------|-------------------|
| Images (PNG/JPG) | Base64 encode → vision model | Multimodal model (e.g., LLaVA) |
| Audio (WAV/MP3) | whisper-apr → text transcript | Any text model (transcript injected) |
| Documents (PDF/TXT) | Extract text → inject as context | Any text model |
| Code files | Syntax highlight → inject as context | Any text model |

```
POST /api/v1/chat/completions
Content-Type: multipart/form-data

file: @photo.jpg
json: {"messages": [{"role": "User", "content": "What's in this image?"}]}
```

Audio flow:
```
Upload WAV → whisper-apr transcribe → inject transcript as user message
→ model responds to transcript → return response + transcript
```

---

## Multimodal Support

### Vision

Requires a vision-capable model (LLaVA, Phi-3-Vision, etc.):

```json
{
  "messages": [{
    "role": "User",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

OpenAI vision API compatible. Image preprocessing (resize, normalize) handled by realizar's vision pipeline.

### Text-to-Speech (Future)

Placeholder for TTS output via a Rust TTS engine. Not in initial Phase 4 scope but the endpoint structure supports it:

```json
{
  "choices": [{
    "message": {
      "content": "Hello!",
      "audio": {"format": "wav", "data": "base64..."}
    }
  }]
}
```

---

## New Endpoints (Phase 4)

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Serve SPA (presentar WASM) |
| GET | `/assets/*` | Static assets |
| WS | `/api/v1/ws` | WebSocket for real-time events |
| POST | `/api/v1/chat/completions` | **Modified**: tools, files, multimodal |
| GET | `/api/v1/tools` | List available tools |
| PUT | `/api/v1/tools/{name}/config` | Configure tool (e.g., search provider) |

## Security Considerations

| Feature | Sovereign | Private | Standard |
|---------|-----------|---------|----------|
| Code execution | Allowed (local sandbox) | Allowed | Allowed |
| Web search | **Blocked** | Proxy only | Allowed |
| File upload | Allowed (stays local) | Allowed | Allowed |
| Tool calling | Local tools only | Local + approved | All |
| Image processing | Allowed (local model) | Allowed | Allowed |
| Audio transcription | Allowed (whisper-apr local) | Allowed | Allowed |

---

## Platform Support Matrix

| Platform | Chat | Training | Data Recipes | UI | Code Sandbox |
|----------|------|----------|-------------|-----|-------------|
| Linux + NVIDIA | Full | Full | Full | Full | Full |
| Linux + AMD | Chat only | Via ROCm (future) | Full | Full | Full |
| macOS (Apple Silicon) | Full | MLX backend (future) | Full | Full | Full |
| Windows + NVIDIA | Full | Full | Full | Full | Limited (no namespace isolation) |
| WASM (browser-only) | API client only | — | — | Full | — |

---

## Agent Runtime in Chat

Batuta already has an agent module (`src/agent/`) with a perceive-reason-act loop. Phase 4 wires this into chat for multi-step autonomous task execution.

### Endpoint

```json
POST /api/v1/chat/completions
{
  "messages": [{"role": "User", "content": "Research the top 3 Rust web frameworks and write a comparison table"}],
  "agent": true,
  "agent_config": {
    "max_steps": 10,
    "tools": ["web_search", "code_execution"],
    "memory": true
  }
}
```

### How It Differs from Tool Calling

| Feature | Tool Calling | Agent Mode |
|---------|-------------|------------|
| Steps | Single tool call + response | Multi-step loop (perceive → reason → act) |
| Planning | None | Model generates a plan, executes steps |
| Memory | Stateless | Maintains working memory across steps |
| Iteration | One-shot | Loops until task complete or max_steps |
| Output | Single response | Streaming steps + final answer |

### Agent SSE Stream

```
data: {"type": "plan", "content": "I'll search for Rust frameworks, then compare them."}
data: {"type": "tool_call", "tool": "web_search", "input": "best Rust web frameworks 2026"}
data: {"type": "tool_result", "tool": "web_search", "output": "..."}
data: {"type": "reasoning", "content": "Found Actix, Axum, Rocket. Let me get details on each."}
data: {"type": "tool_call", "tool": "web_search", "input": "Axum vs Actix performance benchmark"}
data: {"type": "tool_result", "tool": "web_search", "output": "..."}
data: {"type": "answer", "content": "| Framework | Performance | Ecosystem | ..."}
data: [DONE]
```

### Agent Memory

When `memory: true`, the agent persists working memory in `~/.banco/agent_memory/` via trueno-rag. Subsequent conversations can reference prior research:

```json
{"messages": [{"role": "User", "content": "Based on your earlier research, which framework would you pick for a REST API?"}],
 "agent": true, "agent_config": {"memory": true}}
```

### Privacy Constraints

| Tier | Available Tools |
|------|----------------|
| Sovereign | code_execution, local file access, RAG (uploaded docs only) |
| Private | Sovereign + approved MCP tools |
| Standard | All tools including web_search |

---

## Architecture Changes

```
src/serve/banco/
  +-- (all Phase 1-3 modules)
  +-- tools.rs            (new: tool registry, execution, self-heal)
  +-- sandbox.rs          (new: code execution sandbox via jugar-probar)
  +-- search.rs           (new: web search provider abstraction)
  +-- multimodal.rs       (new: image/audio preprocessing)
  +-- websocket.rs        (new: WS event broadcasting)
  +-- static_assets.rs    (new: embedded SPA serving)
  +-- agent_handler.rs    (new: agent loop, memory, streaming steps)
```

## Open Questions

1. **presentar maturity**: Is presentar ready for a full SPA, or do we need a minimal hand-written HTML/JS frontend for Phase 4.0?
2. **WASM bundle size**: How large is the presentar WASM bundle? Embedding in the binary adds to binary size.
3. **Code sandbox on macOS**: No PID namespaces. Use process isolation + resource limits only?
4. **Search provider**: Build a simple DuckDuckGo scraper, or require user to configure an API key?
5. **Vision model support in realizar**: Does realizar handle image tokens today, or is this a realizar dependency?
6. **Agent loop + streaming**: The existing agent runtime is synchronous. Need async adapter for SSE streaming of steps.
7. **Agent memory scope**: Per-conversation, per-user, or global? How to avoid memory pollution across unrelated tasks?

## Test Strategy

| Test | Type | What |
|------|------|------|
| Serve index.html at `/` | Integration | Static asset embedding |
| WebSocket connects + receives event | Integration | WS plumbing |
| Code execution returns stdout | Integration | Sandbox + capture |
| Code execution timeout enforced | Unit | 30s kill |
| Web search blocked in Sovereign | Unit | Privacy gate |
| Audio upload → transcript | Integration | whisper-apr pipeline |
| Image upload → vision response | Integration | Multimodal pipeline |
| Self-heal tool call (3 retries) | Unit | Retry logic |
| Tool call in Sovereign (local only) | Unit | Privacy filtering |
| Agent multi-step with 2 tool calls | Integration | perceive-reason-act loop |
| Agent memory persists across calls | Integration | trueno-rag memory store |
| Agent max_steps enforced | Unit | Loop termination |
| Agent SSE streams step types | Integration | Plan/tool/reasoning/answer events |
