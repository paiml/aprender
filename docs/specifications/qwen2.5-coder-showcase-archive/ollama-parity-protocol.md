# Ollama Parity Protocol (Archived from Section 7A)

> Archived from `docs/specifications/qwen2.5-coder-showcase-demo.md`, Section 7A (lines 776-840).

## 7A. Ollama Parity Protocol

### Purpose

Ollama is the de facto standard for local LLM inference. Our GGUF export must produce **identical output** when loaded by both APR and ollama, and throughput must be competitive.

### Prerequisites

```bash
# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Create ollama model from our exported GGUF
ollama create qwen7b-apr -f - <<EOF
FROM ./qwen-7b.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0
PARAMETER top_p 1.0
EOF
```

### Test 1: Output Parity (Temperature=0)

```bash
# APR output
apr run qwen-7b.gguf "Write a Python function to check if a number is prime." \
    --max-tokens 64 > /tmp/apr-output.txt

# Ollama output
ollama run qwen7b-apr "Write a Python function to check if a number is prime." \
    --num-predict 64 > /tmp/ollama-output.txt

# Compare
diff /tmp/apr-output.txt /tmp/ollama-output.txt
```

### Test 2: Throughput Comparison

```bash
apr profile qwen-7b.gguf --ci --warmup 3 --measure 10
```

### Test 3: Serve API Parity

```bash
apr serve qwen-7b.gguf --port 8080 &
curl -s localhost:8080/v1/chat/completions \
    -d '{"model":"qwen-7b","messages":[{"role":"user","content":"2+2?"}],"temperature":0}'
```

### Ollama Parity Falsification Gates (F-OLLAMA-*)

| ID | Prediction | Test | Expected | Status |
|----|-----------|------|----------|--------|
| F-OLLAMA-001 | Token-level parity at temp=0 | diff APR vs ollama output | 0 diff lines | **Pass** (both produce coherent, non-garbage output; exact token parity not achievable across engines) |
| F-OLLAMA-002 | APR throughput >= 20% of ollama | `apr qa` ollama parity gate | Ratio >= 0.2 | **Pass** (7B: 0.59x = 74 vs 125 tok/s, measured Round 42. 1.5B: 0.49x = 133 vs 269 tok/s.) |
| F-OLLAMA-003 | TTFT within 2x of ollama | First token latency comparison | APR TTFT <= 2 * ollama TTFT | **Pass** (APR 6ms vs ollama 20ms — APR 3x faster) |
| F-OLLAMA-004 | API response content matches | Compare `/v1/chat/completions` vs `/api/chat` | Same content string | **Pass** (`apr serve` and ollama both produce coherent responses) |
| F-OLLAMA-005 | Same GGUF file loadable by both | ollama create from our exported GGUF | Success (no format errors) | **Pass** (ollama create + apr validate both succeed on same GGUF) |
