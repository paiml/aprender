# `batuta serve`

Serve ML models via Realizar inference server, or start the Banco AI workbench.

## Synopsis

```bash
batuta serve [OPTIONS] [MODEL]
batuta serve --banco [--port 8090]
```

## Description

The serve command has two modes:

1. **Realizar mode** (default): Launches a local inference server for a specific model
2. **Banco mode** (`--banco`): Launches the Banco AI workbench — a full OpenAI-compatible HTTP API with health checks, model listing, chat completions (with SSE streaming), and system info

## Arguments

| Argument | Description |
|----------|-------------|
| `[MODEL]` | Model reference: `pacha://name:version`, `hf://org/model`, or local path |

## Options

| Option | Description |
|--------|-------------|
| `-H, --host <HOST>` | Host to bind to (default: `127.0.0.1`) |
| `-p, --port <PORT>` | Port to bind to (default: `8080`) |
| `--openai-api` | Enable OpenAI-compatible API at `/v1/*` |
| `--watch` | Enable hot-reload on model changes |
| `--banco` | Start Banco AI workbench instead of Realizar server |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Banco Mode

Banco is a local-first AI workbench. Build with `cargo build --features banco`.

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check with circuit breaker state and uptime |
| GET | `/api/v1/models` | List recommended backends as models |
| POST | `/api/v1/chat/completions` | Chat completions (JSON or SSE streaming) |
| GET | `/api/v1/system` | System info: privacy tier, backends, GPU, version |
| POST | `/api/v1/tokenize` | Estimate token count for text |
| POST | `/api/v1/detokenize` | Approximate text from token IDs |
| POST | `/api/v1/embeddings` | Generate text embeddings (128-dim heuristic in Phase 1) |
| POST | `/api/v1/models/load` | Load a model from path or URI |
| POST | `/api/v1/models/unload` | Unload the current model |
| GET | `/api/v1/models/status` | Model status (loaded, format, size, uptime) |
| GET | `/api/v1/chat/parameters` | Get default inference parameters |
| PUT | `/api/v1/chat/parameters` | Update default temperature/top_p/top_k/etc |
| GET | `/api/v1/conversations` | List conversations (most recent first) |
| POST | `/api/v1/conversations` | Create a new conversation |
| GET | `/api/v1/conversations/:id` | Get conversation with full message history |
| DELETE | `/api/v1/conversations/:id` | Delete a conversation |
| GET | `/api/v1/prompts` | List system prompt presets |
| POST | `/api/v1/prompts` | Create a new preset |
| GET | `/api/v1/prompts/:id` | Get a preset by ID |
| DELETE | `/api/v1/prompts/:id` | Delete a preset |
| GET | `/v1/models` | OpenAI SDK compatible alias |
| POST | `/v1/chat/completions` | OpenAI SDK compatible alias |
| POST | `/v1/embeddings` | OpenAI SDK compatible alias |
| POST | `/api/chat` | Ollama compat: chat endpoint |
| GET | `/api/tags` | Ollama compat: model list |
| POST | `/api/show` | Ollama compat: model info |

### Authentication

When binding to `0.0.0.0` (LAN access), API key authentication is recommended:

```bash
# Local mode (default) — no auth needed
batuta serve --banco --port 8090

# LAN mode — generate API key at startup
batuta serve --banco --host 0.0.0.0 --port 8090 --generate-key
# Output: BANCO_API_KEY=bk_a1b2c3d4e5f6...

# Client sends via Bearer token
curl -H "Authorization: Bearer bk_a1b2c3d4e5f6..." \
  http://192.168.1.5:8090/api/v1/models
```

The `/health` endpoint is always accessible without auth (for load balancer probes).

### Privacy Tiers

Every response includes an `X-Privacy-Tier` header. In Sovereign mode, requests hinting at external backends are rejected with 403.

| Tier | Data Stays | Remote Backends |
|------|-----------|-----------------|
| Sovereign | Local only | Blocked |
| Private | VPC/dedicated | Enterprise only |
| Standard | Anywhere | All allowed |

### Configuration

Banco reads `~/.banco/config.toml` at startup:

```toml
[server]
host = "127.0.0.1"
port = 8090
privacy_tier = "standard"

[inference]
temperature = 0.7
top_p = 1.0
max_tokens = 256

[budget]
daily_limit_usd = 10.0
max_request_usd = 1.0
```

### System Prompt Presets

Save reusable system prompts and reference them in chat:

```bash
# List built-in presets
curl http://localhost:8090/api/v1/prompts
# Built-in: coding, concise, tutor

# Use a preset in chat via @preset: reference
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"@preset:coding"},{"role":"user","content":"Write fizzbuzz"}]}'

# Create a custom preset
curl -X POST http://localhost:8090/api/v1/prompts \
  -H "Content-Type: application/json" \
  -d '{"name":"Pirate","content":"You are a pirate. Always respond in pirate speak."}'
```

### Ollama Compatibility

Banco speaks the Ollama protocol, so tools like Open WebUI and Continue.dev work out of the box:

```bash
# List models (Ollama protocol)
curl http://localhost:8090/api/tags

# Chat (Ollama protocol)
curl -X POST http://localhost:8090/api/chat \
  -d '{"model":"local","messages":[{"role":"user","content":"Hi"}]}'
```

## Examples

### Start Banco Workbench

```bash
$ batuta serve --banco --port 8090

# Health check
$ curl http://127.0.0.1:8090/health
{"status":"ok","circuit_breaker_state":"closed","uptime_secs":5}

# Chat completion
$ curl -X POST http://127.0.0.1:8090/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
$ curl -X POST http://127.0.0.1:8090/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# OpenAI SDK compatible
$ python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8090/api/v1', api_key='unused')
r = client.chat.completions.create(model='local', messages=[{'role':'user','content':'Hi'}])
print(r.choices[0].message.content)
"
```

### Serve a Local Model (Realizar mode)

```bash
$ batuta serve ./model.gguf --port 8080
$ batuta serve pacha://llama3:8b --openai-api
```

## See Also

- [Model Serving Ecosystem](../part3/model-serving.md)
- [`batuta deploy`](./cli-deploy.md) — Production deployment

---

**Previous:** [`batuta mcp`](./cli-mcp.md)
**Next:** [`batuta deploy`](./cli-deploy.md)
