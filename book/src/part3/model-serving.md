# Model Serving Ecosystem

The Model Serving Ecosystem provides a unified interface for local and remote model serving across the ML ecosystem. Built on Toyota Way principles, it ensures reliable, cost-effective, and privacy-aware model inference.

## Toyota Way Principles

| Principle | Implementation |
|-----------|---------------|
| **Standardized Work** | Chat templates ensure consistent model interaction |
| **Poka-Yoke** | Privacy gates prevent accidental data leakage |
| **Jidoka** | Stateful failover maintains context on errors |
| **Muda Elimination** | Cost circuit breakers prevent waste |
| **Heijunka** | Spillover routing enables load leveling |

## Components

### ChatTemplateEngine

Unified prompt templating supporting multiple formats:

```rust
use batuta::serve::{ChatTemplateEngine, ChatMessage, TemplateFormat};

// Auto-detect from model name
let engine = ChatTemplateEngine::from_model("llama-2-7b-chat");

let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("What is Rust?"),
];

let prompt = engine.apply(&messages);
```

**Supported Formats:**
- `Llama2` - Meta's Llama 2 format with `[INST]` tags
- `Mistral` - Mistral's format (similar to Llama2)
- `ChatML` - OpenAI-style `<|im_start|>` format
- `Alpaca` - Stanford Alpaca instruction format
- `Vicuna` - Vicuna conversation format
- `Raw` - Passthrough without formatting

### BackendSelector

Intelligent backend selection with privacy tiers:

```rust
use batuta::serve::{BackendSelector, PrivacyTier, ServingBackend};

let selector = BackendSelector::new()
    .with_privacy(PrivacyTier::Sovereign)  // Local only
    .with_latency(LatencyTier::Interactive);

let backends = selector.recommend();
// Returns: [Realizar, Ollama, LlamaCpp]
```

**Privacy Tiers:**

| Tier | Description | Allowed Backends |
|------|-------------|------------------|
| `Sovereign` | Local only, blocks ALL external API calls | Realizar, Ollama, LlamaCpp, Llamafile, Candle, Vllm, Tgi, LocalAI |
| `Private` | Dedicated/VPC endpoints only | Local + AzureOpenAI, AwsBedrock, GoogleVertex |
| `Standard` | Public APIs acceptable | All backends |

**Supported Backends:**

*Local (8):*
- Realizar, Ollama, LlamaCpp, Llamafile, Candle, Vllm, Tgi, LocalAI

*Remote (12):*
- HuggingFace, Together, Replicate, Anyscale, Modal, Fireworks, Groq
- OpenAI, Anthropic, AzureOpenAI, AwsBedrock, GoogleVertex

### CostCircuitBreaker

Daily budget limits to prevent runaway costs:

```rust
use batuta::serve::{CostCircuitBreaker, CircuitBreakerConfig};

let config = CircuitBreakerConfig {
    daily_budget_usd: 10.0,
    warning_threshold: 0.8,  // Warn at 80%
    max_request_cost_usd: 1.0,
    ..Default::default()
};

let breaker = CostCircuitBreaker::new(config);

// Before each request
match breaker.check(estimated_cost) {
    Ok(_) => { /* proceed */ },
    Err(CostError::DailyBudgetExceeded { .. }) => { /* block */ },
    Err(CostError::RequestTooExpensive { .. }) => { /* reject */ },
}

// After request completes
breaker.record(actual_cost);
```

**Token Pricing (per 1M tokens):**

| Model | Input | Output |
|-------|-------|--------|
| GPT-4 Turbo | $10.00 | $30.00 |
| GPT-4 | $30.00 | $60.00 |
| GPT-3.5 Turbo | $0.50 | $1.50 |
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Llama (local) | $0.00 | $0.00 |

### ContextManager

Automatic token counting and context truncation:

```rust
use batuta::serve::{ContextManager, TruncationStrategy};

let manager = ContextManager::for_model("gpt-4-turbo");

// Check if messages fit
if manager.fits(&messages) {
    // Proceed directly
} else {
    // Truncate using strategy
    let truncated = manager.truncate(&messages)?;
}
```

**Context Windows:**

| Model | Max Tokens | Output Reserve |
|-------|------------|----------------|
| GPT-4 Turbo | 128,000 | 4,096 |
| GPT-4 | 8,192 | 2,048 |
| Claude 3 | 200,000 | 4,096 |
| Llama 3 | 8,192 | 2,048 |
| Mixtral | 32,768 | 4,096 |

**Truncation Strategies:**
- `SlidingWindow` - Remove oldest messages first
- `MiddleOut` - Keep first and last, remove middle
- `Error` - Fail instead of truncating

### FailoverManager

Stateful failover for streaming with context preservation:

```rust
use batuta::serve::{FailoverManager, ServingBackend};

let mut manager = FailoverManager::with_defaults();

// Start tracking
manager.start_tracking("req-123", "Original prompt");

// Accumulate tokens during streaming
manager.append_tokens("req-123", "Generated ");
manager.append_tokens("req-123", "tokens here");

// On failure, prepare failover
if manager.should_failover("req-123") {
    let failover_request = manager.prepare_failover("req-123");
    // Contains continuation prompt with generated prefix
}

// On success
manager.complete("req-123");
```

### SpilloverRouter

Hybrid cloud spillover routing for load leveling:

```rust
use batuta::serve::{SpilloverRouter, RouterConfig};

let config = RouterConfig {
    spillover_threshold: 10,  // Queue depth before spillover
    max_queue_depth: 50,
    local_backend: ServingBackend::Realizar,
    spillover_backends: vec![
        ServingBackend::Groq,
        ServingBackend::Together,
    ],
    ..Default::default()
};

let router = SpilloverRouter::new(config);

match router.route() {
    RoutingDecision::Local(backend) => { /* use local */ },
    RoutingDecision::Spillover(backend) => { /* use remote */ },
    RoutingDecision::Reject(reason) => { /* queue full */ },
}
```

## Integration Example

Complete example combining all components:

```rust
use batuta::serve::{
    ChatTemplateEngine, ChatMessage,
    BackendSelector, PrivacyTier,
    CostCircuitBreaker, CircuitBreakerConfig,
    ContextManager,
    SpilloverRouter, RouterConfig,
};

// 1. Select backend based on privacy requirements
let selector = BackendSelector::new()
    .with_privacy(PrivacyTier::Private);
let backend = selector.recommend().first().copied()
    .expect("No backend available");

// 2. Check cost budget
let breaker = CostCircuitBreaker::with_defaults();
let estimated_cost = 0.01;
breaker.check(estimated_cost)?;

// 3. Prepare messages with context management
let messages = vec![
    ChatMessage::system("You are helpful."),
    ChatMessage::user("Explain quantum computing."),
];

let manager = ContextManager::for_model("llama-2-70b");
let messages = manager.truncate(&messages)?;

// 4. Apply chat template
let engine = ChatTemplateEngine::from_model("llama-2-70b");
let prompt = engine.apply(&messages);

// 5. Route request
let router = SpilloverRouter::with_defaults();
let decision = router.route();

// 6. Execute and record cost
// ... inference call ...
breaker.record(actual_cost);
```

## Configuration

Default configurations are provided for common use cases:

```rust
// Sovereign mode - local only
let config = RouterConfig::sovereign();

// Enterprise mode - private endpoints
let selector = BackendSelector::new()
    .with_privacy(PrivacyTier::Private);

// Cost-conscious mode
let config = CircuitBreakerConfig {
    daily_budget_usd: 5.0,
    max_request_cost_usd: 0.50,
    ..Default::default()
};
```

## Model Security (Spec §8)

The serving ecosystem integrates with Pacha's security features for model integrity and confidentiality.

### Model Signing (§8.2)

Ed25519 digital signatures ensure model integrity:

```rust
use pacha::signing::{generate_keypair, sign_model, verify_model};

// Generate signing keypair (once)
let (signing_key, verifying_key) = generate_keypair();

// Sign model before distribution
let model_data = std::fs::read("model.gguf")?;
let signature = sign_model(&model_data, &signing_key)?;
signature.save("model.gguf.sig")?;

// Verify before loading
let sig = ModelSignature::load("model.gguf.sig")?;
verify_model(&model_data, &sig)?;
```

**CLI Usage:**
```bash
# Generate signing key
batuta pacha keygen --identity alice@example.com

# Sign a model
batuta pacha sign model.gguf --identity alice@example.com

# Verify signature
batuta pacha verify model.gguf
```

### Encryption at Rest (§8.3)

ChaCha20-Poly1305 encryption for secure model distribution:

```rust
use pacha::crypto::{encrypt_model, decrypt_model, is_encrypted};

// Encrypt for distribution
let encrypted = encrypt_model(&model_data, "secure-password")?;
std::fs::write("model.gguf.enc", &encrypted)?;

// Decrypt at load time
let encrypted = std::fs::read("model.gguf.enc")?;
if is_encrypted(&encrypted) {
    let password = std::env::var("MODEL_KEY")?;
    let decrypted = decrypt_model(&encrypted, &password)?;
}
```

**CLI Usage:**
```bash
# Encrypt model
batuta pacha encrypt model.gguf --password-env MODEL_KEY

# Decrypt at runtime
MODEL_KEY=secret batuta pacha decrypt model.gguf.enc
```

**Encrypted File Format:**
- Magic: `PACHAENC` (8 bytes)
- Version: 1 byte
- Salt: 32 bytes (key derivation)
- Nonce: 12 bytes
- Ciphertext: variable
- Auth tag: 16 bytes

### Content-Addressed Storage (§8.1)

All models in Pacha are content-addressed with BLAKE3:

```rust
// Verify before loading
let expected = "blake3:a1b2c3...";
let actual = blake3::hash(&model_data);
assert_eq!(expected, format!("blake3:{}", actual.to_hex()));
```

## Banco: Local-First AI Workbench

Banco wraps the entire serving ecosystem into a single HTTP API accessible via `batuta serve --banco`. It reuses all the components above (BackendSelector, SpilloverRouter, CostCircuitBreaker, ContextManager, ChatTemplateEngine) behind axum endpoints.

```bash
# Start Banco
batuta serve --banco --port 8090

# Health
curl http://localhost:8090/health

# Chat (OpenAI-compatible)
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

**Key features:**
- OpenAI SDK drop-in: `openai.base_url = "http://localhost:8090/api/v1"`
- Privacy middleware: `X-Privacy-Tier` header on every response
- Sovereign mode: blocks all external backends at the middleware layer
- SSE streaming: real-time token delivery for chat completions
- Config persistence: `~/.banco/config.toml` survives restarts
- Zero telemetry: `"telemetry": false` in every system response

**Feature flag:** `banco` (adds axum, tower, async-stream, tokio-stream)

See [`batuta serve` CLI reference](../part6/cli-serve.md) for full usage.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `native` | Core serve module (templates, backends, router, circuit breaker) |
| `banco` | Banco HTTP API (axum, tower, SSE streaming) |
| `inference` | Realizar inference engine integration |
