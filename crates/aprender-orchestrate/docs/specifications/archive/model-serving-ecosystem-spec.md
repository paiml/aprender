# Model Serving Ecosystem Specification v1.0.0

## Overview

Unified interface for local and remote model serving across the ML ecosystem, with native PAIML integration via `realizar`.

```
[REVIEW-001] @alfredo 2024-12-05
Toyota Principle: Genchi Genbutsu (Go and See)
Direct integration with serving backends eliminates abstraction overhead.
Batuta queries actual inference endpoints, not cached proxies.
Status: APPROVED
```

## Ecosystem Landscape

### Local Serving

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL MODEL SERVING                          │
├──────────────┬──────────┬─────────────┬─────────────────────────┤
│ Tool         │ Language │ Format      │ Key Feature             │
├──────────────┼──────────┼─────────────┼─────────────────────────┤
│ realizar     │ Rust     │ GGUF/ST     │ PAIML native, MoE       │
│ Ollama       │ Go       │ GGUF        │ Docker-like UX          │
│ llamafile    │ C        │ GGUF        │ Single executable       │
│ llama.cpp    │ C++      │ GGUF        │ Reference impl          │
│ candle       │ Rust     │ SafeTensors │ HF Rust framework       │
│ vLLM         │ Python   │ PT/ST       │ PagedAttention          │
│ TGI          │ Rust/Py  │ SafeTensors │ HF official server      │
│ LocalAI      │ Go       │ GGUF/*      │ OpenAI-compatible       │
│ LM Studio    │ Electron │ GGUF        │ Desktop GUI             │
│ GPT4All      │ C++      │ GGUF        │ Desktop app             │
│ MLC LLM      │ C++      │ MLC         │ Universal deployment    │
│ ExLlamaV2    │ Python   │ EXL2        │ Extreme quantization    │
└──────────────┴──────────┴─────────────┴─────────────────────────┘

Legend: PT=PyTorch, ST=SafeTensors, MoE=Mixture of Experts
```

```
[REVIEW-002] @noah 2024-12-05
Toyota Principle: Heijunka (Level Loading)
Multiple backend support balances load across available resources.
Local serving for low-latency; remote for scale-out.
Status: APPROVED
```

### Remote Serving

```
┌─────────────────────────────────────────────────────────────────┐
│                    REMOTE MODEL SERVING                          │
├──────────────────┬─────────────┬────────────────────────────────┤
│ Service          │ Provider    │ Characteristics                │
├──────────────────┼─────────────┼────────────────────────────────┤
│ HF Inference API │ HuggingFace │ Serverless, pay-per-token      │
│ HF Endpoints     │ HuggingFace │ Dedicated, auto-scaling        │
│ Together.ai      │ Together    │ Fast inference, open models    │
│ Replicate        │ Replicate   │ Pay-per-call, easy deploy      │
│ Anyscale         │ Anyscale    │ Ray-based, distributed         │
│ Modal            │ Modal       │ Serverless GPU, Python-native  │
│ Fireworks.ai     │ Fireworks   │ Fast, function calling         │
│ Groq             │ Groq        │ LPU hardware, ultra-fast       │
│ AWS Bedrock      │ Amazon      │ Managed, enterprise            │
│ Azure OpenAI     │ Microsoft   │ OpenAI models, enterprise      │
│ Google Vertex    │ Google      │ PaLM/Gemini, enterprise        │
│ Anthropic API    │ Anthropic   │ Claude models                  │
│ OpenAI API       │ OpenAI      │ GPT models                     │
└──────────────────┴─────────────┴────────────────────────────────┘
```

```
[REVIEW-003] @maria 2024-12-05
Toyota Principle: Just-in-Time
Remote APIs enable on-demand inference without idle GPU costs.
Pay only for actual compute consumed.
Status: APPROVED
```

## CLI Interface

### Serve Commands

```bash
# Start local server with realizar
batuta serve start --backend realizar --model ./model.gguf --port 8080

# Start with Ollama backend
batuta serve start --backend ollama --model llama2:7b

# Start with llama.cpp
batuta serve start --backend llamacpp --model ./model.gguf --ctx 4096

# List running servers
batuta serve list

# Stop server
batuta serve stop --port 8080
```

### Query Commands

```bash
# Query local server
batuta serve query "What is the capital of France?" --endpoint localhost:8080

# Query remote API
batuta serve query "Explain transformers" --endpoint together --model mistral-7b

# Benchmark endpoint
batuta serve bench --endpoint localhost:8080 --prompts ./prompts.jsonl
```

### Backend Management

```bash
# List available backends
batuta serve backends

# Check backend health
batuta serve health --backend ollama

# Pull model for backend
batuta serve pull --backend ollama --model codellama:13b
```

```
[REVIEW-004] @carlos 2024-12-05
Toyota Principle: Standardized Work
Unified CLI across all backends reduces cognitive load.
Same commands work for realizar, Ollama, or cloud APIs.
Status: APPROVED
```

## Data Model

```rust
/// Supported serving backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServingBackend {
    // Local backends
    Realizar,
    Ollama,
    LlamaCpp,
    Llamafile,
    Candle,
    Vllm,
    Tgi,
    LocalAI,

    // Remote backends
    HuggingFace,
    Together,
    Replicate,
    Anyscale,
    Modal,
    Fireworks,
    Groq,
    OpenAI,
    Anthropic,
    AzureOpenAI,
    AwsBedrock,
    GoogleVertex,
}

/// Backend capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub streaming: bool,
    pub function_calling: bool,
    pub vision: bool,
    pub embeddings: bool,
    pub batch_inference: bool,
    pub quantization: Vec<String>,
    pub max_context: usize,
    pub formats: Vec<ModelFormat>,
}

/// Model format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    GGUF,
    SafeTensors,
    PyTorch,
    ONNX,
    TensorRT,
    OpenVINO,
    MLC,
    EXL2,
}

/// Inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: bool,
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    pub tokens_per_second: f32,
}
```

```
[REVIEW-005] @elena 2024-12-05
Toyota Principle: Jidoka (Built-in Quality)
Strong typing prevents runtime errors at API boundaries.
Invalid requests fail at compile time, not in production.
Status: APPROVED
```

## Backend Integration Tree

```bash
batuta serve tree
```

```
Model Serving Ecosystem
├── local
│   ├── paiml_native
│   │   └── realizar      ⚡ Rust, GGUF/ST, MoE routing
│   ├── gguf_based
│   │   ├── ollama        ✓ Go, Docker-like CLI
│   │   ├── llamafile     ✓ C, single executable
│   │   ├── llama.cpp     ✓ C++, reference impl
│   │   ├── localai       ✓ Go, OpenAI-compatible
│   │   └── gpt4all       ✓ C++, desktop app
│   ├── rust_native
│   │   └── candle        ✓ Rust, HF framework
│   └── python_based
│       ├── vllm          ✓ PagedAttention
│       ├── tgi           ✓ HF official
│       └── exllamav2     ✓ EXL2 quantization
├── remote
│   ├── huggingface
│   │   ├── inference-api ☁ Serverless
│   │   └── endpoints     ☁ Dedicated
│   ├── inference_providers
│   │   ├── together      ☁ Fast, open models
│   │   ├── replicate     ☁ Pay-per-call
│   │   ├── fireworks     ☁ Function calling
│   │   └── groq          ☁ LPU, ultra-fast
│   ├── serverless_gpu
│   │   ├── modal         ☁ Python-native
│   │   └── anyscale      ☁ Ray-based
│   └── enterprise
│       ├── aws_bedrock   ☁ Amazon managed
│       ├── azure_openai  ☁ Microsoft managed
│       └── google_vertex ☁ Google managed
└── frontier_apis
    ├── openai            ☁ GPT-4, GPT-4o
    ├── anthropic         ☁ Claude 3.5
    └── google            ☁ Gemini
```

```
[REVIEW-006] @david 2024-12-05
Toyota Principle: Visual Management
Tree view provides instant ecosystem comprehension.
Engineers see all options without documentation diving.
Status: APPROVED
```

## PAIML Integration Map

```
┌─────────────────┬────────────────────┬────────────────────────┐
│ PAIML Component │ Ecosystem Equiv    │ Integration Type       │
├─────────────────┼────────────────────┼────────────────────────┤
│ INFERENCE       │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ realizar        │ llama.cpp          │ ⚡ ALTERNATIVE (Rust)   │
│ realizar        │ Ollama             │ ⚡ ALTERNATIVE (native) │
│ realizar        │ vLLM               │ ⚡ ALTERNATIVE (MoE)    │
│ realizar        │ TGI                │ ⚡ ALTERNATIVE (Rust)   │
├─────────────────┼────────────────────┼────────────────────────┤
│ FORMAT SUPPORT  │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ realizar/gguf   │ GGUF ecosystem     │ ✓ COMPATIBLE (parse)   │
│ realizar/st     │ SafeTensors        │ ✓ COMPATIBLE (r/w)     │
│ realizar/tok    │ HF tokenizers      │ ✓ COMPATIBLE (load)    │
├─────────────────┼────────────────────┼────────────────────────┤
│ ORCHESTRATION   │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ batuta serve    │ Ollama CLI         │ 🔄 ORCHESTRATES        │
│ batuta serve    │ Remote APIs        │ 🔄 ORCHESTRATES        │
├─────────────────┼────────────────────┼────────────────────────┤
│ COMPUTE         │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ trueno          │ PyTorch backend    │ ⚡ ALTERNATIVE (SIMD)   │
│ trueno-gpu      │ CUDA backend       │ ⚡ ALTERNATIVE (native) │
│ repartir        │ Ray/Anyscale       │ ⚡ ALTERNATIVE (dist)   │
└─────────────────┴────────────────────┴────────────────────────┘

Legend:
  ⚡ ALTERNATIVE  - PAIML native replacement
  ✓ COMPATIBLE   - Interoperates with format/API
  🔄 ORCHESTRATES - Batuta wraps/manages backend
```

```
[REVIEW-007] @sofia 2024-12-05
Toyota Principle: Nemawashi (Consensus Building)
Integration map clarifies when to use PAIML vs external backends.
Teams make informed decisions based on requirements.
Status: APPROVED
```

## Backend Selection Strategy

```rust
/// Select optimal backend based on requirements
pub struct BackendSelector {
    pub latency_requirement: LatencyTier,
    pub throughput_requirement: ThroughputTier,
    pub cost_sensitivity: CostTier,
    pub privacy_requirement: PrivacyTier,
}

#[derive(Debug, Clone, Copy)]
pub enum LatencyTier {
    RealTime,    // <100ms - local GPU or Groq
    Interactive, // <1s - local or fast remote
    Batch,       // >1s - any backend
}

#[derive(Debug, Clone, Copy)]
pub enum PrivacyTier {
    Sovereign,   // Local only, no external calls
    Private,     // VPC/dedicated endpoints
    Standard,    // Public APIs acceptable
}

impl BackendSelector {
    pub fn recommend(&self) -> Vec<ServingBackend> {
        match (self.latency_requirement, self.privacy_requirement) {
            (LatencyTier::RealTime, PrivacyTier::Sovereign) => {
                vec![ServingBackend::Realizar, ServingBackend::LlamaCpp]
            }
            (LatencyTier::RealTime, _) => {
                vec![ServingBackend::Groq, ServingBackend::Realizar]
            }
            (_, PrivacyTier::Sovereign) => {
                vec![ServingBackend::Realizar, ServingBackend::Ollama]
            }
            _ => {
                vec![ServingBackend::Together, ServingBackend::HuggingFace]
            }
        }
    }
}
```

```
[REVIEW-008] @miguel 2024-12-05
Toyota Principle: Pull System
Backend selection pulls optimal choice based on actual requirements.
No over-provisioning; right-sized infrastructure.
Status: APPROVED
```

## Quantization Support

| Backend | F32 | F16 | Q8 | Q4 | Q4_K_M | Q5_K_M | EXL2 |
|---------|-----|-----|----|----|--------|--------|------|
| realizar | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| llama.cpp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Ollama | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| vLLM | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| ExLlamaV2 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| candle | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

```
[REVIEW-009] @ana 2024-12-05
Toyota Principle: Muda Elimination (Waste)
Quantization reduces memory/compute waste.
Q4_K_M delivers 90% quality at 25% memory cost.
Status: APPROVED
```

## Implementation Plan

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | `src/serve/mod.rs` | Module structure |
| 2 | `src/serve/backends.rs` | Backend registry |
| 3 | `src/serve/local.rs` | Local backend integration |
| 4 | `src/serve/remote.rs` | Remote API clients |
| 5 | `src/serve/selector.rs` | Backend selection logic |
| 6 | `src/serve/tree.rs` | Tree visualization |
| 7 | CLI commands | `batuta serve *` |
| 8 | Tests | 95%+ coverage |

## Success Criteria

- [ ] `batuta serve start --backend realizar` launches server in <2s
- [ ] `batuta serve query` works across all backends
- [ ] `batuta serve tree` displays ecosystem in <100ms
- [ ] Backend auto-selection matches requirements
- [ ] Seamless failover between backends
- [ ] 95% test coverage on serve module

```
[REVIEW-010] @jorge 2024-12-05
Toyota Principle: Challenge (Long-term Vision)
Unified serving interface future-proofs against ecosystem churn.
New backends integrate without API changes.
Status: APPROVED
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BATUTA_SERVE_BACKEND` | Default backend | `realizar` |
| `BATUTA_SERVE_PORT` | Default port | `8080` |
| `OLLAMA_HOST` | Ollama endpoint | `localhost:11434` |
| `OPENAI_API_KEY` | OpenAI key | None |
| `ANTHROPIC_API_KEY` | Anthropic key | None |
| `TOGETHER_API_KEY` | Together.ai key | None |
| `HF_TOKEN` | HuggingFace token | None |

## References

- llama.cpp: https://github.com/ggerganov/llama.cpp
- Ollama: https://ollama.ai
- vLLM: https://vllm.ai
- TGI: https://huggingface.co/docs/text-generation-inference
- candle: https://github.com/huggingface/candle
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Toyota Production System (Ohno, 1988)
- The Toyota Way (Liker, 2004)
