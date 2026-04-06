# Banco: Local-First AI Workbench

Banco is a self-contained AI studio shipped as `batuta serve --banco`. One command, one binary, zero cloud dependencies.

## Quick Start

```bash
# Build with banco feature
cargo build --features banco

# Start the workbench
batuta serve --banco --port 8090

# Open browser UI
open http://localhost:8090/

# Chat via API (OpenAI-compatible)
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

The browser UI at `http://localhost:8090/` provides a chat interface with SSE streaming (tokens appear in real-time), model status, settings panel, file upload, and RAG toggle. No separate frontend build — the UI is compiled into the binary.

## Architecture

```
batuta serve --banco
  │
  ├── 86 Endpoints (80 routes)
  │   ├── Core:        /health /models /system
  │   ├── Chat:        /chat/completions (sync + SSE), /chat/parameters
  │   ├── Data:        /tokenize /detokenize /embeddings
  │   ├── Models:      /models/load|unload|status
  │   ├── Convos:      /conversations (CRUD + search + rename + export/import)
  │   ├── Presets:     /prompts (CRUD)
  │   ├── Files:       /data/upload|files (content-hash dedup)
  │   ├── Recipes:     /data/recipes|datasets (chunk/filter/format/dedup)
  │   ├── RAG:         /rag/index|status|search (BM25, auto-index)
  │   ├── Eval:        /eval/perplexity|runs
  │   ├── Training:    /train/start|runs|stop|metrics|export|presets
  │   ├── Merge:       /models/merge|strategies (TIES/DARE/SLERP)
  │   ├── Registry:    /models/pull|registry (pacha)
  │   ├── Audio:       /audio/transcriptions (whisper-apr)
  │   ├── MCP:         /mcp (Model Context Protocol, JSON-RPC 2.0)
  │   ├── Metrics:     /metrics (Prometheus exposition format)
  │   ├── Tools:       /tools (calculator, code, search + custom)
  │   ├── WebSocket:   /ws (real-time event push)
  │   ├── Experiments: /experiments (create + compare)
  │   ├── Batch:       /batch (multi-prompt)
  │   ├── Config:      /config (GET/PUT)
  │   ├── Audit:       /audit (query log)
  │   ├── OpenAI:      /v1/* (SDK compatible)
  │   └── Ollama:      /api/* (generate + chat + tags + show)
  │
  ├── Middleware (3 layers)
  │   ├── Audit logging               Every request → audit.jsonl
  │   ├── Authentication              API key for LAN mode
  │   └── Privacy + CORS              X-Privacy-Tier header
  │
  ├── Persistence (~/.banco/)
  │   ├── conversations/              JSONL per conversation
  │   ├── uploads/                    Content-hash dedup files
  │   ├── audit.jsonl                 Request audit trail
  │   └── config.toml                 Server config
  │
  └── State (Arc<BancoStateInner>) — 16 components
      ├── BackendSelector              Privacy-aware routing
      ├── SpilloverRouter              Heijunka load leveling
      ├── CostCircuitBreaker           Muda budget enforcement
      ├── ContextManager / ChatTemplateEngine
      ├── ConversationStore            Persistent + searchable
      ├── PromptStore / AuthStore
      ├── ModelSlot                    Arc<OwnedQuantizedModel>
      ├── FileStore                    Content-addressable
      ├── RecipeStore / RagIndex
      ├── EvalStore / TrainingStore / ExperimentStore
      ├── BatchStore / AuditLog
      └── InferenceParams (RwLock)
```

## Compatibility

Banco speaks four protocols from the same port:

| Protocol | Routes | Works With |
|----------|--------|------------|
| **Banco native** | `/api/v1/*` | curl, custom clients |
| **OpenAI** | `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/models/:id`, `/v1/embeddings`, `/v1/audio/transcriptions` | OpenAI Python SDK, LangChain, LlamaIndex |
| **MCP** | `/api/v1/mcp` (JSON-RPC 2.0) | Claude Desktop, Cursor, MCP clients |
| **Ollama** | `/api/generate`, `/api/chat`, `/api/tags`, `/api/show` | Open WebUI, Continue.dev, Aider |

### OpenAI SDK Example

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/api/v1", api_key="unused")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(r.choices[0].message.content)
```

### Text Completions (non-chat)

```bash
# POST /v1/completions — text completion (prompt, not messages)
curl -X POST http://localhost:8090/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 50}'
```

Supports both single string and array prompts. Uses the same model as chat completions.

## Model Loading

```bash
# Load GGUF at startup
batuta serve --banco --model ./tinyllama.gguf --port 8090

# Load APR (our native format) — same command
batuta serve --banco --model ./qwen2.5-coder.apr --port 8090

# Load at runtime via API
curl -X POST http://localhost:8090/api/v1/models/load \
  -d '{"model": "./phi-3-mini.gguf"}'

# Check status (shows architecture, vocab, layers when inference feature enabled)
curl http://localhost:8090/api/v1/models/status
# {"loaded":true,"model":{"model_id":"phi-3-mini","format":"gguf",
#   "architecture":"phi3","vocab_size":32064,"hidden_dim":3072,
#   "num_layers":32,"context_length":4096,"tensor_count":195}}

# Unload
curl -X POST http://localhost:8090/api/v1/models/unload
```

Build with `--features banco` for everything — inference, BPE tokenization, Arrow data, training/merge. One feature flag, zero confusion.

### Tokenizer Loading

Banco automatically searches for a proper BPE tokenizer alongside the model:

1. `{model-stem}.tokenizer.json` (sibling file)
2. `tokenizer.json` in the same directory

When found, the BPE tokenizer uses correct merge rules for tokenization. Without it, Banco falls back to greedy longest-match (approximate). For best results, keep the HuggingFace `tokenizer.json` next to your model file.

### apr-cli as Orchestration Entry Point

`apr-cli` (from aprender) is the recommended tool for model preparation:

```bash
# Convert HuggingFace model to APR format
apr import qwen2.5-coder-1.5b --output model.apr

# Profile model architecture
apr profile model.apr

# Benchmark inference
apr bench model.apr --prompt "Hello world"
```

Banco delegates model management to apr-cli's proven pipelines: `BpeTokenizer::from_huggingface()`, `OwnedQuantizedModel::from_apr()`, and `MappedGGUFModel::from_path()`.

### Checking Tokenizer Status

The model status and system info endpoints report whether proper BPE tokenization is active:

```bash
# Model status shows tokenizer mode
curl http://localhost:8090/api/v1/models/status
# {"loaded":true,"model":{...},"tokenizer":"bpe","uptime_secs":42}

# System info also includes tokenizer
curl http://localhost:8090/api/v1/system
# {...,"tokenizer":"bpe",...}
```

Values: `"bpe"` (proper merge rules from tokenizer.json) or `"greedy"` (approximate fallback).

### Supported Model Formats

| Format | Extension | Load | Inference | Metadata |
|--------|-----------|------|-----------|----------|
| **GGUF** | `.gguf` | Yes | Yes | Architecture, vocab, layers, context |
| **APR v2** | `.apr` | Yes | Yes (via `from_apr()`) | Architecture, hidden_dim, layers, vocab |
| SafeTensors | `.safetensors` | Metadata only | No | — |

## Conversations

Messages are persisted server-side. First user message auto-generates a title.

```bash
# Create
curl -X POST http://localhost:8090/api/v1/conversations -d '{}'
# {"id":"conv-1234-0","title":"New conversation"}

# Chat within conversation
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Explain gravity"}],
       "conversation_id":"conv-1234-0"}'

# List
curl http://localhost:8090/api/v1/conversations

# Get full history
curl http://localhost:8090/api/v1/conversations/conv-1234-0

# Search conversations by content
curl "http://localhost:8090/api/v1/conversations/search?q=rust"

# Export all conversations
curl http://localhost:8090/api/v1/conversations/export > backup.json

# Import conversations
curl -X POST http://localhost:8090/api/v1/conversations/import \
  -H "Content-Type: application/json" -d @backup.json
```

## System Prompt Presets

Three built-in presets, plus custom ones. Referenced via `@preset:name`:

```bash
# List presets
curl http://localhost:8090/api/v1/prompts
# Built-in: coding, concise, tutor

# Use in chat
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -d '{"messages":[
    {"role":"system","content":"@preset:coding"},
    {"role":"user","content":"Write fizzbuzz in Rust"}
  ]}'

# Create custom
curl -X POST http://localhost:8090/api/v1/prompts \
  -d '{"name":"Pirate","content":"You are a pirate. Arr!"}'
```

## Data Management

Upload files for use in data recipes, RAG, and training (Phase 3).

```bash
# Upload via JSON
curl -X POST http://localhost:8090/api/v1/data/upload/json \
  -H "Content-Type: application/json" \
  -d '{"name": "docs.txt", "content": "Your document text..."}'

# Upload via multipart
curl -X POST http://localhost:8090/api/v1/data/upload \
  -F "file=@training-data.csv"

# List files
curl http://localhost:8090/api/v1/data/files

# Delete
curl -X DELETE http://localhost:8090/api/v1/data/files/file-123-0
```

Supported formats: PDF, CSV, JSON, JSONL, DOCX, TXT. Files are content-hash deduplicated.

### File Info + Schema

```bash
# Get file details, preview, and schema (for structured files)
curl http://localhost:8090/api/v1/data/files/file-123-0/info
# {"name":"data.csv","content_type":"text/csv","schema":[
#   {"name":"text","data_type":"Utf8","nullable":true},
#   {"name":"label","data_type":"Int64","nullable":true}
# ],"preview_lines":["text,label","Hello,1"]}
```

With `--features ml`, schema detection uses alimentar's Arrow parser for accurate type inference.

## Data Recipes

Declarative pipelines that transform uploaded files into training datasets.

```bash
# Create a recipe
curl -X POST http://localhost:8090/api/v1/data/recipes \
  -H "Content-Type: application/json" \
  -d '{
    "name": "docs-to-training",
    "source_files": ["file-123-0"],
    "steps": [
      {"type": "extract_text", "config": {}},
      {"type": "chunk", "config": {"max_tokens": 512, "overlap": 64}},
      {"type": "filter", "config": {"min_length": 50}},
      {"type": "format", "config": {"template": "chatml"}}
    ]
  }'

# Run the recipe
curl -X POST http://localhost:8090/api/v1/data/recipes/recipe-123-0/run

# Preview dataset
curl http://localhost:8090/api/v1/data/datasets/ds-123-0/preview
```

Built-in steps: `extract_text`, `parse_csv` (column extraction with alimentar validation), `parse_jsonl` (field extraction), `chunk` (token-aware), `filter` (min/max length), `format` (chatml/alpaca/llama2), `deduplicate`.

## RAG (Retrieval-Augmented Generation)

Upload documents, index them, then chat with context retrieval:

```bash
# 1. Upload documents
curl -X POST http://localhost:8090/api/v1/data/upload/json \
  -H "Content-Type: application/json" \
  -d '{"name": "policy.txt", "content": "Refunds available within 30 days..."}'

# 2. Index all uploaded documents
curl -X POST http://localhost:8090/api/v1/rag/index

# 3. Chat with RAG enabled
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the refund policy?"}], "rag": true}'

# Check index status
curl http://localhost:8090/api/v1/rag/status

# Clear index
curl -X DELETE http://localhost:8090/api/v1/rag/index
```

RAG uses BM25 keyword search to find relevant chunks from indexed documents and prepends them as context before generation. Documents are **auto-indexed on upload** — no manual indexing needed.

With `--features rag`, Banco uses **trueno-rag's battle-tested BM25** implementation with proper stopword filtering and tokenization. Without the feature, a built-in BM25 implementation provides the same API with zero dependencies.

## Model Evaluation

Compute perplexity to measure model quality before and after fine-tuning:

```bash
# Run perplexity evaluation
curl -X POST http://localhost:8090/api/v1/eval/perplexity \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog.", "max_tokens": 512}'

# List eval runs
curl http://localhost:8090/api/v1/eval/runs
```

Returns `"status": "no_model"` without a loaded model; real perplexity with `--features inference`.

## Training

Start LoRA/QLoRA fine-tuning runs. With `--features ml`, training uses entrenar's LoRA/QLoRA/AdamW stack. Without `ml`, runs produce simulated metrics for API testing.

```bash
# Start with explicit config
curl -X POST http://localhost:8090/api/v1/train/start \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "ds-123", "method": "lora", "config": {"lora_r": 16, "epochs": 3}}'

# Start with a preset (quick-lora, standard-lora, deep-lora, qlora-low-vram, full-finetune)
curl -X POST http://localhost:8090/api/v1/train/start \
  -d '{"dataset_id": "ds-123", "preset": "quick-lora"}'

# List presets
curl http://localhost:8090/api/v1/train/presets

# List runs
curl http://localhost:8090/api/v1/train/runs

# Stream metrics (SSE)
curl http://localhost:8090/api/v1/train/runs/run-123/metrics

# Stop a run
curl -X POST http://localhost:8090/api/v1/train/runs/run-123/stop

# Export adapter or merged model
curl -X POST http://localhost:8090/api/v1/train/runs/run-123/export \
  -d '{"format": "safetensors", "merge": false}'
```

### Training Presets

| Preset | Method | R | Epochs | LR | Target |
|--------|--------|---|--------|-----|--------|
| `quick-lora` | LoRA | 8 | 1 | 2e-4 | q/v_proj |
| `standard-lora` | LoRA | 16 | 3 | 2e-4 | q/k/v/o_proj |
| `deep-lora` | LoRA | 32 | 5 | 1e-4 | all linear |
| `qlora-low-vram` | QLoRA | 16 | 3 | 2e-4 | q/k/v/o_proj |
| `full-finetune` | FFT | — | 3 | 5e-5 | all params |

### Export Formats

| Format | Extension | Use With |
|--------|-----------|----------|
| `safetensors` | .safetensors | HuggingFace, vLLM |
| `gguf` | .gguf | llama.cpp, Ollama |
| `apr` | .apr | realizar, Banco |

Training methods: `lora`, `qlora`, `full_finetune`. Config includes: `lora_r`, `lora_alpha`, `learning_rate`, `epochs`, `batch_size`, `max_seq_length`, `target_modules`, `optimizer` (adam/adamw/sgd), `scheduler` (constant/cosine/linear/step_decay), `warmup_steps`, `gradient_accumulation_steps`, `max_grad_norm`.

## Experiments

Group training runs into experiments for comparison:

```bash
# Create experiment
curl -X POST http://localhost:8090/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "LoRA vs QLoRA", "description": "Comparing fine-tune methods"}'

# Add runs to experiment
curl -X POST http://localhost:8090/api/v1/experiments/EXP-ID/runs \
  -H "Content-Type: application/json" -d '{"run_id": "run-123"}'

# Compare runs
curl http://localhost:8090/api/v1/experiments/EXP-ID/compare
```

Comparison shows final loss, total steps, method, and identifies the best run.

## Batch Inference

Process multiple prompts in a single request:

```bash
curl -X POST http://localhost:8090/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
    {"id": "q1", "messages": [{"role": "user", "content": "What is Rust?"}]},
    {"id": "q2", "messages": [{"role": "user", "content": "What is Python?"}]}
  ]}'
```

Uses real inference when a model is loaded; dry-run echo otherwise.

## Model Merge

Combine multiple fine-tuned models using TIES, DARE, SLERP, or weighted averaging. With `--features ml`, merging uses entrenar's tensor merge implementations.

```bash
# List available strategies
curl http://localhost:8090/api/v1/models/merge/strategies

# Weighted average (simple blend)
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{"models": ["model-a.gguf", "model-b.gguf"],
       "strategy": "weighted_average", "weights": [0.7, 0.3]}'

# TIES merge (noise reduction across fine-tunes)
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{"models": ["lora-1", "lora-2", "lora-3"],
       "strategy": "ties", "density": 0.2}'

# SLERP (smooth two-model interpolation)
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{"models": ["model-a", "model-b"],
       "strategy": "slerp", "interpolation_t": 0.5}'

# DARE (stochastic sparsity merge)
curl -X POST http://localhost:8090/api/v1/models/merge \
  -d '{"models": ["a", "b"], "strategy": "dare",
       "drop_prob": 0.5, "seed": 42}'
```

| Strategy | Models | Best For |
|----------|--------|----------|
| `weighted_average` | 2+ | Simple controlled blending |
| `ties` | 2+ | Multi-task merging, noise reduction |
| `dare` | 2+ | Stochastic sparsity-based merge |
| `slerp` | 2 only | Smooth rotation-invariant interpolation |

## Model Registry (pacha)

Pull and manage models from the pacha model registry.

```bash
# Pull a model by reference
curl -X POST http://localhost:8090/api/v1/models/pull \
  -d '{"model_ref": "llama3:8b-q4"}'
# {"model_ref":"llama3:8b-q4","status":"pulled","path":"/cache/llama3.gguf",
#  "size_bytes":4294967296,"cache_hit":false,"format":"gguf"}

# List cached models
curl http://localhost:8090/api/v1/models/registry

# Remove from cache
curl -X DELETE http://localhost:8090/api/v1/models/registry/llama3
```

Supported URI formats: `llama3:8b-q4`, `pacha://model:version`, `file://./model.gguf`.

## Privacy Tiers

Every response includes `X-Privacy-Tier`. Sovereign mode blocks all external backends.

| Tier | Data Location | Remote Allowed |
|------|--------------|----------------|
| **Sovereign** | Local only | No |
| **Private** | VPC/dedicated | Enterprise only |
| **Standard** | Anywhere | Yes |

## Speech-to-Text (whisper-apr)

Transcribe audio to text. With `--features speech`, uses whisper-apr for real transcription.

```bash
# Transcribe audio (base64-encoded)
curl -X POST http://localhost:8090/api/v1/audio/transcriptions \
  -d '{"audio_data": "BASE64_AUDIO_HERE", "format": "wav", "language": "en"}'
# {"text":"Hello world","language":"en","duration_secs":1.5,"segments":[...]}

# List supported formats
curl http://localhost:8090/api/v1/audio/formats
# {"formats":[{"extension":"wav"},{"extension":"mp3"}],"sample_rate":16000,"engine":"whisper-apr"}
```

Options: `language` (auto-detect if omitted), `translate` (to English), `format` (wav/mp3/flac/ogg).

## File Attachments in Chat

Attach documents or code files to chat requests — text is extracted and injected as context.

```bash
curl -X POST http://localhost:8090/api/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Summarize this code"}],
       "attachments": [
         {"name": "main.rs", "content": "fn main() { println!(\"hello\"); }"}
       ]}'
```

Supported: TXT, code files, CSV, JSON. Content is prepended as a system message.

## Tool Calling

OpenAI-compatible tool calling with built-in tools, custom registration, and self-healing retry.

```bash
# List available tools
curl http://localhost:8090/api/v1/tools

# Execute a tool directly
curl -X POST http://localhost:8090/api/v1/tools/execute \
  -d '{"id": "call-1", "name": "calculator", "arguments": {"expression": "(2+3)*4"}}'
# {"tool_call_id":"call-1","name":"calculator","content":"20"}

# Register a custom tool
curl -X POST http://localhost:8090/api/v1/tools \
  -d '{"name": "my_tool", "description": "Custom tool",
       "parameters": {"type": "object"}, "enabled": true}'

# Enable/disable a tool
curl -X PUT http://localhost:8090/api/v1/tools/web_search/config \
  -d '{"enabled": true}'
```

### Built-in Tools

| Tool | Description | Privacy |
|------|-------------|---------|
| `calculator` | Evaluate math expressions (+, -, *, /, parentheses) | All tiers |
| `code_execution` | Sandbox code execution (dry-run without jugar-probar) | All tiers |
| `web_search` | Web search (disabled by default, Standard tier only) | Standard |

Tools are privacy-tier aware: Sovereign mode blocks tools that require network access.

### Self-Healing Retry

When a tool call fails (bad arguments, unknown tool), Banco returns error context for re-prompting:

```bash
# Execute with retry context
curl -X POST http://localhost:8090/api/v1/tools/execute \
  -d '{"id": "c1", "name": "calculator", "arguments": {"expression": ""}}'
# Returns: {"should_retry":true,"error_context":"Tool call failed: Empty expression...","retries_remaining":2}
```

The chat handler can inject the error as a system message and re-prompt the model (max 3 retries).

### OpenAI Tool Calling Compatibility

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Calculate 2+2"}],
       "tools": [{"type": "function", "function": {"name": "calculator"}}],
       "tool_choice": "auto"}'
```

## MCP (Model Context Protocol)

Banco speaks MCP — connect Claude Desktop, Cursor, or any MCP client.

```bash
# Server info (for discovery)
curl http://localhost:8090/api/v1/mcp/info
# {"protocol":"mcp","version":"2024-11-05","server":"banco","transport":"http"}

# Initialize (JSON-RPC 2.0)
curl -X POST http://localhost:8090/api/v1/mcp \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# List tools
curl -X POST http://localhost:8090/api/v1/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Call a tool
curl -X POST http://localhost:8090/api/v1/mcp \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"6*7"}}}'
# {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"42"}]}}

# List resources
curl -X POST http://localhost:8090/api/v1/mcp \
  -d '{"jsonrpc":"2.0","id":4,"method":"resources/list","params":{}}'

# List prompts
curl -X POST http://localhost:8090/api/v1/mcp \
  -d '{"jsonrpc":"2.0","id":5,"method":"prompts/list","params":{}}'
```

Supported methods: `initialize`, `tools/list`, `tools/call`, `resources/list`, `prompts/list`, `ping`.

## Real-Time Events (WebSocket)

Connect to `/api/v1/ws` for push notifications:

```bash
# Using websocat (or any WebSocket client)
websocat ws://localhost:8090/api/v1/ws
```

Events are JSON with `type` and `data` fields:

```json
{"type":"connected","data":{"endpoints":67,"model_loaded":false}}
{"type":"file_uploaded","data":{"file_id":"file-123","name":"docs.txt"}}
{"type":"training_started","data":{"run_id":"run-456","method":"lora"}}
{"type":"training_metric","data":{"run_id":"run-456","step":10,"loss":1.2}}
{"type":"training_complete","data":{"run_id":"run-456"}}
{"type":"model_loaded","data":{"model_id":"phi-3","format":"gguf"}}
```

Event types: `model_loaded`, `model_unloaded`, `training_started`, `training_metric`, `training_complete`, `file_uploaded`, `rag_indexed`, `merge_complete`, `system_event`.

## Prometheus Metrics

```bash
curl http://localhost:8090/api/v1/metrics
# banco_requests_total 42
# banco_chat_requests_total 15
# banco_tokens_generated_total 3200
# banco_errors_total 0
# banco_uptime_seconds 3600
# banco_model_loaded 1
# banco_endpoints_total 86
```

Compatible with Prometheus/Grafana scraping. Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: banco
    static_configs:
      - targets: ['localhost:8090']
    metrics_path: /api/v1/metrics
```

## Health Probes (k8s)

```bash
# Liveness: always 200 if server is running
curl http://localhost:8090/health/live

# Readiness: 200 when model loaded, 503 otherwise
curl http://localhost:8090/health/ready

# Full health: JSON with circuit breaker state + uptime
curl http://localhost:8090/health
```

## Authentication

```bash
# Local (127.0.0.1) — no auth
batuta serve --banco

# LAN (0.0.0.0) — API key recommended
batuta serve --banco --host 0.0.0.0

# Client auth
curl -H "Authorization: Bearer bk_..." http://192.168.1.5:8090/api/v1/models
```

### Scoped API Keys

| Scope | Allows |
|-------|--------|
| `chat` | Chat, models (read), health, conversations, prompts, completions |
| `train` | All of chat + training, data, experiments, eval |
| `admin` | Everything including model load/unload, config, merge |

Chat keys cannot load/unload/merge models or access registry. Train keys cannot access model admin endpoints.

## Persistence

When started via `batuta serve --banco`, all data persists to `~/.banco/`:

```
~/.banco/
  ├── config.toml          Server config (privacy, inference, budget)
  ├── audit.jsonl           Request audit trail (every API call)
  ├── conversations/        JSONL message logs per conversation
  ├── experiments/          JSON per experiment (survives restarts)
  └── uploads/              Content-hash deduplicated files
```

Data survives server restarts — conversations, files, experiments, and audit are reloaded on startup. In-memory mode (`with_defaults()`) is used for testing only.

## Configuration

`~/.banco/config.toml`:

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

## Provable Contracts

5 YAML contracts in `contracts/banco/` formalize Banco's critical invariants:

| Contract | Property |
|----------|----------|
| privacy-enforcement | Sovereign blocks all remote backends |
| budget-conservation | Spending never exceeds daily budget |
| routing-determinism | Same state → same routing decision |
| context-enforcement | Truncated messages always fit window |
| template-correctness | User content preserved across formats |

Each has Kani harnesses and 20 falsification tests.

## Feature Flag

```toml
[dependencies]
batuta = { version = "0.7", features = ["banco"] }
```

Adds: axum, tower, async-stream, tokio-stream. Default build unaffected.

## Inference

When built with `--features banco,inference`, Banco performs real token generation using realizar's `forward_single_with_cache()` autoregressive loop:

```
POST /api/v1/chat/completions → tokenize prompt → prefill cache → decode tokens → response
```

- **Prefill**: Processes all prompt tokens through the model to populate KV cache
- **Decode**: Autoregressively generates tokens using greedy or top-k sampling
- **Streaming**: Each generated token is yielded as an SSE chunk in real time
- **Fallback**: Without inference feature or without a loaded model, returns echo/dry-run response

Sampling parameters (temperature, top_k, max_tokens) can be set per-request or via `PUT /api/v1/chat/parameters`.

## Testing

Banco has a 2-level test pyramid:

| Level | Tests | What |
|-------|-------|------|
| **L1** (unit) | 358 | `tower::ServiceExt::oneshot()` — in-process, no TCP |
| **L2** (integration) | 73 | Real TCP server + `reqwest`/probar `LlmClient` + WebSocket (74/74 routes, **100%**) |
| **L4** (browser) | 2 | Headless Chrome via probar CDP (page load, UI elements) |

Tests are in 5 files:
- `tests/banco_llm.rs` — 5 probar LlmClient chat completion tests
- `tests/banco_endpoints.rs` — 27 raw HTTP endpoint tests
- `tests/banco_workflows.rs` — 21 multi-step workflow tests
- `tests/banco_coverage.rs` — 17 coverage gap tests
- `tests/banco_browser.rs` — 2 L4 browser tests via probar CDP (requires Chrome)

The `banco` feature is **batteries-included** — includes realizar (inference), aprender (BPE tokenizer), alimentar (Arrow data), and entrenar (training/merge). Just `--features banco` for everything.

```bash
# Run all banco tests
cargo test --features banco

# Run just L2 integration tests
cargo test --features banco --test banco_llm --test banco_endpoints --test banco_workflows
```

## Phase Roadmap

| Phase | Status | What |
|-------|--------|------|
| **1** | **Complete** | HTTP API skeleton, 24 endpoints, 121 tests |
| **2a** | **Complete** | Model slot, load/unload/status, inference params, GGUF metadata |
| **2b** | **Complete** | Inference loop, greedy/top-k sampling, SSE streaming, Ollama generate |
| **3** | **Complete** | Files, recipes, RAG, training, merge, registry, experiments, batch |
| **4** | **Complete** | MCP, tools, audio, auth, metrics, probes — 82 endpoints |
| **5a** | **Complete** | APR loading, BPE tokenizer, 72 L2 + 2 L4 tests, honest labeling, zero-JS UI |
| 5b | Planned | presentar WASM UI (7 screens), TUI dashboard |

See [banco-spec.md](../../docs/specifications/components/banco-spec.md) for full specification.
