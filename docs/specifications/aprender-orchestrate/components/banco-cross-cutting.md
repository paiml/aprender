# Banco Cross-Cutting Concerns

> Parent: [banco-spec.md](banco-spec.md) §5
> These concerns span all phases. Each section notes which phase introduces it.

---

## 1. Authentication & Access Control

**Introduced: Phase 2 (required before remote access in Phase 4)**

### Modes

| Mode | When | Auth |
|------|------|------|
| **Local** (default) | Bind to 127.0.0.1 | None required — localhost trusted |
| **LAN** | Bind to 0.0.0.0 | API key required |
| **Remote** | Exposed via tunnel/proxy | JWT + refresh tokens |

### API Key Auth (LAN)

```bash
# Generate key at startup
batuta serve --banco --host 0.0.0.0 --generate-key
# Output: BANCO_API_KEY=bk_a1b2c3d4...

# Client sends via header
curl -H "Authorization: Bearer bk_a1b2c3d4..." http://192.168.1.5:8090/api/v1/models
```

Keys stored in `~/.banco/keys.toml`. Scoped:

| Scope | Allows |
|-------|--------|
| `chat` | Chat completions, models list, health |
| `train` | All of `chat` + training, data, experiments |
| `admin` | All of `train` + model load/unload, system config |

### JWT Auth (Remote)

```
POST /api/v1/auth/login
{"username": "admin", "password": "..."}

Response:
{"access_token": "eyJ...", "refresh_token": "eyJ...", "expires_in": 3600}

POST /api/v1/auth/refresh
{"refresh_token": "eyJ..."}
```

User accounts stored in `~/.banco/users.toml` (bcrypt hashed). Single-user by default — multi-user is opt-in via config.

### Privacy Tier Interaction

| Tier | Local (127.0.0.1) | LAN (0.0.0.0) | Remote |
|------|-------------------|----------------|--------|
| Sovereign | No auth | API key + warning | **Blocked** (no remote in Sovereign) |
| Private | No auth | API key | JWT required |
| Standard | No auth | API key | JWT required |

Sovereign + remote = hard error at startup. Data never leaves the machine.

---

## 2. Conversation Persistence

**Introduced: Phase 2**

### Storage

```
~/.banco/
  +-- conversations/
  |     +-- conv-20260319-001.jsonl   (append-only message log)
  |     +-- conv-20260319-002.jsonl
  +-- banco.db                        (SQLite index: id, title, created, updated, model)
```

### Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/v1/conversations` | List conversations (paginated) |
| GET | `/api/v1/conversations/{id}` | Full message history |
| POST | `/api/v1/conversations` | Create new conversation |
| DELETE | `/api/v1/conversations/{id}` | Delete conversation |
| POST | `/api/v1/chat/completions` | **Modified**: accepts `conversation_id` to append |

### Auto-Titling

First user message → model generates a 5-word title (via loaded model, or heuristic fallback). Stored in index.

### Export

```bash
# Export all conversations as JSON
curl http://localhost:8090/api/v1/conversations/export > backup.json

# Import
curl -X POST http://localhost:8090/api/v1/conversations/import \
  -d @backup.json
```

---

## 3. OpenAI SDK Compatibility

**Introduced: Phase 2 (critical for ecosystem adoption)**

Goal: `pip install openai` + point at Banco = works.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/api/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

### Required Fixes from Phase 1

| Issue | Phase 1 | Fix |
|-------|---------|-----|
| Role serialization | PascalCase (`"User"`) | Add `#[serde(rename_all = "lowercase")]` or alias deserialize |
| `/v1/` prefix | `/api/v1/` | Mount duplicate routes at `/v1/*` |
| `model` field | Optional | Accept but ignore (single model server) |
| `api_key` header | Not checked | Accept and ignore in local mode |

### Compatibility Matrix

| OpenAI Endpoint | Banco Status |
|----------------|--------------|
| `POST /v1/chat/completions` | **Complete** — real inference with loaded model |
| `GET /v1/models` | **Complete** |
| `POST /v1/embeddings` | **Complete** — model embeddings when loaded |
| `POST /v1/audio/transcriptions` | Phase 4 (whisper-apr) |
| `POST /v1/images/generations` | Not planned |
| `GET /v1/files` | **Complete** — `/api/v1/data/files` |
| `POST /v1/batch` | **Complete** — `/api/v1/batch` |

---

## 4. Ollama API Compatibility Layer

**Introduced: Phase 2**

Many tools (Open WebUI, Continue.dev, Aider) speak Ollama protocol. Banco should support both.

| Ollama Route | Maps To | Status |
|-------------|---------|--------|
| `POST /api/generate` | `/api/v1/chat/completions` | **Complete** (PMAT-081) |
| `POST /api/chat` | `/api/v1/chat/completions` | **Complete** (Phase 2) |
| `GET /api/tags` | `/api/v1/models` | **Complete** (Phase 2) |
| `POST /api/show` | `/api/v1/models/status` | **Complete** (Phase 2) |
| `POST /api/pull` | `/api/v1/models/load` | Phase 3b (pacha integration) |
| `DELETE /api/delete` | `/api/v1/models/unload` | Phase 3b |

Implementation: thin adapter layer in `src/serve/banco/compat_ollama.rs` that translates request/response formats. Not a full Ollama reimplementation — just enough for tool compatibility.

---

## 5. Embeddings Endpoint

**Introduced: Phase 2**

```
POST /api/v1/embeddings
{
  "model": "local",
  "input": ["Hello world", "Another sentence"]
}

Response:
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.034, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.008, -0.021, ...]}
  ],
  "model": "local",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

Uses the model's embedding layer output. For BERT-style models (if loaded), uses the pooled output. For causal LLMs, uses mean-pooled hidden states from the last layer.

---

## 6. Tokenizer Endpoint

**Introduced: Phase 2**

```
POST /api/v1/tokenize
{"text": "Hello, world!"}

Response:
{"tokens": [15496, 11, 995, 0], "count": 4}

POST /api/v1/detokenize
{"tokens": [15496, 11, 995, 0]}

Response:
{"text": "Hello, world!"}
```

Also: context window usage in every chat response:

```json
{
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170,
    "context_window": 4096,
    "context_used_pct": 4.2
  }
}
```

---

## 7. Batch Inference

**Introduced: Phase 3**

Process a file of prompts without interactive chat.

```
POST /api/v1/batch
Content-Type: multipart/form-data

file: @prompts.jsonl
json: {"max_tokens": 256, "temperature": 0.7}
```

Input `prompts.jsonl`:
```jsonl
{"id": "req-1", "messages": [{"role": "user", "content": "Summarize: ..."}]}
{"id": "req-2", "messages": [{"role": "user", "content": "Translate: ..."}]}
```

Response: streaming JSONL or poll-based:
```
POST /api/v1/batch → {"batch_id": "batch-123", "status": "processing"}
GET /api/v1/batch/batch-123 → {"status": "complete", "results_url": "/api/v1/batch/batch-123/results"}
GET /api/v1/batch/batch-123/results → JSONL download
```

Useful for: dataset evaluation, bulk classification, generating training data.

---

## 8. Pacha Registry Integration

**Introduced: Phase 2**

Pacha is the Sovereign AI Stack's model registry (like Ollama's library, but with Ed25519 signatures).

| Action | CLI | API |
|--------|-----|-----|
| Pull model | `batuta pacha pull llama3:8b` | `POST /api/v1/models/pull {"model": "pacha://llama3:8b"}` |
| Push trained model | `batuta pacha push ./model` | `POST /api/v1/models/push {"run_id": "run-123", "name": "my-finetuned:v1"}` |
| List cached | `batuta pacha list` | `GET /api/v1/models?source=pacha` |
| Pull progress | — | SSE: `GET /api/v1/models/pull/{id}/progress` |

Pull progress SSE:
```
data: {"status": "downloading", "completed_bytes": 1048576, "total_bytes": 4294967296, "speed_mbps": 120}
data: {"status": "downloading", "completed_bytes": 2097152, ...}
data: {"status": "verifying", "signature": "ed25519:..."}
data: {"status": "complete", "model_id": "llama3-8b-Q4_K_M"}
data: [DONE]
```

---

## 9. Configuration Persistence

**Introduced: Phase 1 (basic), expanded each phase**

```toml
# ~/.banco/config.toml

[server]
host = "127.0.0.1"
port = 8090
privacy_tier = "standard"   # sovereign | private | standard

[inference]
temperature = 0.7
top_p = 1.0
top_k = 40
repeat_penalty = 1.1
template_format = "auto"    # auto | chatml | llama2 | mistral | ...

[model]
default = ""                # path or pacha:// URI to load at startup
auto_load = true            # load default model on startup

[budget]
daily_limit_usd = 10.0     # circuit breaker (for remote spillover)
max_request_usd = 1.0

[auth]
mode = "local"              # local | lan | remote
api_keys_file = "keys.toml"

[storage]
data_dir = "~/.banco"
max_upload_mb = 500
max_conversations = 10000
```

Config loaded at startup. Overridable via CLI flags. API endpoint for runtime updates:

```
GET  /api/v1/config          → current config (redacted secrets)
PUT  /api/v1/config          → update + persist to disk (admin scope)
```

---

## 10. Request Logging & Audit Trail

**Introduced: Phase 2 (critical for Sovereign compliance)**

Every API request logged to `~/.banco/audit.jsonl`:

```jsonl
{"ts": "2026-03-19T12:00:00Z", "method": "POST", "path": "/api/v1/chat/completions", "status": 200, "tokens_in": 42, "tokens_out": 128, "cost_usd": 0.0, "model": "phi-3-mini", "latency_ms": 1420, "privacy_tier": "sovereign", "user": "local"}
{"ts": "2026-03-19T12:00:05Z", "method": "GET", "path": "/health", "status": 200, "latency_ms": 1}
```

- Sovereign tier: log everything, never redact (your data, your audit)
- Log rotation: daily files, configurable retention
- Endpoint: `GET /api/v1/audit?since=2026-03-19&limit=100` (admin scope)

---

## 11. CORS Middleware

**Introduced: Phase 4 (required for browser UI)**

```rust
let cors = CorsLayer::new()
    .allow_origin(Any)              // Same-origin in production, permissive in dev
    .allow_methods([GET, POST, PUT, DELETE, OPTIONS])
    .allow_headers([CONTENT_TYPE, AUTHORIZATION])
    .expose_headers(["x-privacy-tier"]);
```

In Sovereign mode with LAN binding: restrict `allow_origin` to the server's own address.

---

## 12. System Prompt Presets

**Introduced: Phase 2**

```
GET  /api/v1/prompts                    → list saved system prompts
POST /api/v1/prompts                    → save new preset
GET  /api/v1/prompts/{id}              → get preset
PUT  /api/v1/prompts/{id}              → update
DELETE /api/v1/prompts/{id}            → delete
```

```json
{
  "id": "coding-assistant",
  "name": "Coding Assistant",
  "content": "You are an expert software engineer. Write clean, tested code.",
  "created": "2026-03-19T12:00:00Z"
}
```

Stored in `~/.banco/prompts.toml`. Used in chat via:

```json
{
  "messages": [{"role": "system", "content": "@preset:coding-assistant"}, ...]
}
```

The `@preset:` prefix is expanded server-side before template application.

---

## 13. No-Telemetry Guarantee

**All phases.**

Banco collects zero telemetry. No phone-home, no usage analytics, no crash reports. Stated in server startup banner, `GET /api/v1/system` response (`"telemetry": false`), and documentation. The audit log (§10) is local-only and user-controlled.

---

## 14. MCP Integration

**Introduced: Phase 4**

Banco both consumes and exposes MCP tools.

### As MCP Server (expose Banco tools to external agents)

```bash
# Banco exposes tools via MCP stdio transport
batuta serve --banco --mcp-server
```

Exposed tools:
- `chat` — send messages, get completions
- `models` — list, load, unload
- `train` — start/stop training runs
- `data` — upload files, run recipes

### As MCP Client (consume external tools in chat)

```toml
# ~/.banco/config.toml
[mcp.servers]
filesystem = { command = "npx", args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"] }
github = { command = "npx", args = ["-y", "@modelcontextprotocol/server-github"] }
```

External MCP tools appear in the tool registry (`GET /api/v1/tools`) and can be invoked via tool calling in chat (Phase 4). The self-healing retry loop (Phase 4) applies to MCP tool calls too.

---

## 15. Docker Image

**Introduced: Phase 2**

```dockerfile
FROM nvidia/cuda:12.4-runtime-ubuntu24.04
COPY target/release/batuta /usr/local/bin/
EXPOSE 8090
ENTRYPOINT ["batuta", "serve", "--banco", "--host", "0.0.0.0"]
```

```bash
# CPU only
docker run -p 8090:8090 batuta/banco

# With GPU
docker run --gpus all -p 8090:8090 -v ~/.banco:/root/.banco batuta/banco

# With model
docker run --gpus all -p 8090:8090 \
  -v ./models:/models \
  batuta/banco --model /models/llama3.gguf
```

Volume mounts:
- `~/.banco` — config, conversations, training runs, datasets
- `/models` — model files

---

## 16. On-Device Quantization

**Introduced: Phase 3 (as part of export)**

Convert models between quantization levels locally:

```
POST /api/v1/models/quantize
{
  "source": "./model-f16.gguf",
  "target_format": "gguf",
  "quantization": "Q4_K_M",
  "output": "./model-q4km.gguf"
}
```

Uses aprender's quantization pipeline (Q4_K, Q5_K, Q6_K, Q8_0, F16). No shell-out to llama.cpp.

Progress via SSE:
```
GET /api/v1/models/quantize/{id}/progress
data: {"status": "quantizing", "layers_done": 12, "layers_total": 32}
data: {"status": "complete", "output_size_mb": 4200, "compression_ratio": 3.8}
data: [DONE]
```

---

## Implementation Priority

| P0 (Phase 2) | P1 (Phase 2) | P2 (Phase 2-3) | P3 (Phase 3-4) |
|--------------|-------------|----------------|----------------|
| OpenAI SDK compat | Conversation persistence | Pacha registry | Batch inference |
| Config persistence | Tokenizer endpoint | System prompt presets | On-device quantization |
| No-telemetry guarantee | Request logging | Auth (API key) | Auth (JWT) |
| | Embeddings endpoint | Ollama compat layer | CORS |
| | Context window in usage | Docker image | MCP integration |
