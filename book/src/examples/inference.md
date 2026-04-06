# Getting Started with Inference

This guide shows you how to get Realizar up and running and perform your first inference.

## Quick Start (5 Minutes)

### 1. Build Realizar

```bash
# Clone the repository
git clone https://github.com/paiml/realizar.git
cd realizar

# Build in release mode
cargo build --release
```

This creates the binary at `target/release/realizar`.

### 2. Start the Server

```bash
# Start in demo mode
./target/release/realizar serve --demo --port 8080
```

You should see:
```
🚀 Realizar server starting...
📍 Host: 127.0.0.1
🔌 Port: 8080
🎯 Mode: Demo
✅ Server ready at http://127.0.0.1:8080
```

### 3. Test the API

In a new terminal:

```bash
# Health check
curl http://127.0.0.1:8080/health

# Generate text
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'
```

**Congratulations!** 🎉 You're now running Realizar.

---

## Complete Walkthrough

### Step 1: Installation

#### Prerequisites

- **Rust 1.75+**: Install from [rustup.rs](https://rustup.rs)
- **Git**: For cloning the repository

```bash
# Verify Rust installation
rustc --version
cargo --version
```

#### Build Realizar

```bash
# Clone
git clone https://github.com/paiml/realizar.git
cd realizar

# Run tests (optional but recommended)
cargo test

# Build release binary (optimized)
cargo build --release
```

The release build takes ~2-3 minutes and creates an optimized binary with:
- ✅ SIMD optimizations
- ✅ Link-time optimization (LTO)
- ✅ Minimal size

### Step 2: Start the Server

The `serve` command starts the HTTP inference server:

```bash
./target/release/realizar serve --demo --port 8080
```

**Options:**
- `--demo` - Use demo model (required in Phase 1)
- `--port` - Port number (default: 8080)
- `--host` - Host address (default: 127.0.0.1)

**What happens:**
1. Server initializes the demo model
2. Binds to the specified host/port
3. Starts listening for HTTP requests

### Step 3: Health Check

Verify the server is running:

```bash
$ curl http://127.0.0.1:8080/health
{"status":"ok"}
```

This confirms:
- ✅ Server is responsive
- ✅ HTTP stack is working
- ✅ Ready to handle requests

### Step 4: Tokenization

Convert text to tokens:

```bash
$ curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

{"tokens":[7,4,11,11,14,0,22,14,17,11,3]}
```

**What this shows:**
- Text "Hello world" → 11 token IDs
- Each token represents a piece of text
- Model operates on these numeric IDs, not text

### Step 5: Text Generation

Generate text from a prompt:

```bash
$ curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "strategy": "greedy"
  }'

{
  "text": "Hello world this is a test",
  "num_generated": 5
}
```

**Parameters explained:**
- `prompt`: Starting text
- `max_tokens`: Maximum tokens to generate
- `strategy`: Sampling method (`greedy`, `top_k`, `top_p`)

---

## Generation Strategies

Realizar supports three sampling strategies:

### 1. Greedy Sampling (Deterministic)

Always picks the most likely token:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 5,
    "strategy": "greedy"
  }'

# Output: "The capital of France is Paris"
# Same input → always same output
```

**Use case:** Factual generation, reproducible results

### 2. Top-k Sampling (Controlled Randomness)

Samples from the k most likely tokens:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 20,
    "strategy": "top_k",
    "top_k": 40,
    "temperature": 0.8
  }'

# Output: "Once upon a time there lived a young princess..."
# Varies with each run
```

**Use case:** Creative writing with some control

### 3. Top-p/Nucleus Sampling (Dynamic Vocabulary)

Samples from tokens whose cumulative probability ≥ top_p:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In the future",
    "max_tokens": 30,
    "strategy": "top_p",
    "top_p": 0.95,
    "temperature": 1.0
  }'

# Output: "In the future humans will explore distant galaxies..."
# Natural, diverse outputs
```

**Use case:** Natural language generation, storytelling

---

## Temperature Control

Temperature controls randomness (0.0 = deterministic, 2.0 = very random):

```bash
# Low temperature (focused, predictable)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The answer is",
    "max_tokens": 10,
    "strategy": "top_k",
    "temperature": 0.3
  }'
# → "The answer is 42" (very focused)

# High temperature (creative, diverse)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The answer is",
    "max_tokens": 10,
    "strategy": "top_k",
    "temperature": 1.5
  }'
# → "The answer is found through exploration" (more creative)
```

---

## Integration Examples

### Shell Script

```bash
#!/bin/bash
# generate.sh - Simple generation script

PROMPT="$1"
MAX_TOKENS="${2:-20}"

curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$PROMPT\",
    \"max_tokens\": $MAX_TOKENS,
    \"strategy\": \"greedy\"
  }" | jq -r '.text'
```

Usage:
```bash
chmod +x generate.sh
./generate.sh "Hello world" 10
# → Hello world this is a test
```

### Python Client

```python
#!/usr/bin/env python3
# client.py - Python client for Realizar

import requests
import json

class RealizarClient:
    def __init__(self, base_url="http://127.0.0.1:8080"):
        self.base_url = base_url

    def health(self):
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def tokenize(self, text):
        """Tokenize text"""
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text}
        )
        return response.json()["tokens"]

    def generate(self, prompt, max_tokens=50, strategy="greedy",
                 temperature=1.0, top_k=50, top_p=0.9):
        """Generate text from prompt"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "strategy": strategy,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
        )
        return response.json()

# Example usage
if __name__ == "__main__":
    client = RealizarClient()

    # Health check
    print("Health:", client.health())

    # Tokenize
    tokens = client.tokenize("Hello world")
    print(f"Tokens: {tokens}")

    # Generate
    result = client.generate(
        prompt="Once upon a time",
        max_tokens=30,
        strategy="top_k",
        temperature=0.8
    )
    print(f"Generated: {result['text']}")
    print(f"Tokens generated: {result['num_generated']}")
```

### Node.js Client

```javascript
// client.js - Node.js client for Realizar

const fetch = require('node-fetch');

class RealizarClient {
  constructor(baseUrl = 'http://127.0.0.1:8080') {
    this.baseUrl = baseUrl;
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async tokenize(text) {
    const response = await fetch(`${this.baseUrl}/tokenize`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    return data.tokens;
  }

  async generate({prompt, maxTokens = 50, strategy = 'greedy',
                  temperature = 1.0, topK = 50, topP = 0.9}) {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt,
        max_tokens: maxTokens,
        strategy,
        temperature,
        top_k: topK,
        top_p: topP
      })
    });
    return response.json();
  }
}

// Example usage
(async () => {
  const client = new RealizarClient();

  // Health check
  console.log('Health:', await client.health());

  // Tokenize
  const tokens = await client.tokenize('Hello world');
  console.log('Tokens:', tokens);

  // Generate
  const result = await client.generate({
    prompt: 'Once upon a time',
    maxTokens: 30,
    strategy: 'top_k',
    temperature: 0.8
  });
  console.log('Generated:', result.text);
  console.log('Tokens generated:', result.num_generated);
})();
```

---

## Performance Tips

### 1. Use Release Builds

```bash
# Debug (slow, for development)
cargo run -- serve --demo

# Release (fast, for testing/production)
cargo build --release
./target/release/realizar serve --demo
```

**Performance difference:** ~10-20x faster

### 2. Adjust Generation Parameters

```bash
# Fast (fewer tokens)
{"max_tokens": 10}  # ~1-2ms

# Slower (more tokens)
{"max_tokens": 100}  # ~10-15ms
```

### 3. Choose Appropriate Strategy

- **Fastest**: `"greedy"` (deterministic, no sampling overhead)
- **Medium**: `"top_k"` (limited sampling space)
- **Slowest**: `"top_p"` (dynamic vocabulary selection)

---

## Troubleshooting

### Server Won't Start

**Problem**: `Address already in use`

**Solution**: Change port or kill existing process
```bash
# Use different port
./target/release/realizar serve --demo --port 8081

# Or kill existing process
lsof -i :8080
kill <PID>
```

### Slow Generation

**Problem**: Generation takes >100ms

**Checklist**:
- ✅ Using release build? (`--release`)
- ✅ CPU governor set to performance? (Linux)
- ✅ Reasonable `max_tokens` value? (< 100)

### Empty Responses

**Problem**: Server returns empty text

**Check**:
- Is prompt empty? (minimum 1 character required)
- Is `max_tokens` > 0?
- Check server logs for errors

---

## Next Steps

Now that you have Realizar running:

1. **Learn the API** - [API Endpoints](../api/endpoints.md)
2. **Understand Sampling** - [Sampling Strategies](../generation/sampling.md)
3. **Tune Parameters** - [Generation Parameters](../generation/parameters.md)
4. **Build Integrations** - Use the client examples above

## See Also

- [Serve Command Reference](../cli/serve.md) - Complete CLI documentation
- [REST API Reference](../api/endpoints.md) - All endpoints
- [Generation Strategies](../generation/sampling.md) - Strategy comparison
- [Tokenization Overview](../tokenization/overview.md) - How tokenization works
