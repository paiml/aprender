# Complete Test Matrix Results

**Date:** 2026-01-08
**Hardware:** NVIDIA RTX 4090 (24GB VRAM), AMD CPU, 125GB RAM
**Software:** apr-cli v0.2.2, realizar v0.5.0, Ollama 0.5.x

---

## Test Matrix Summary

### GGUF Format Results

| Tier | Size | Backend | APR tok/s | Ollama tok/s | Speedup | Status |
|------|------|---------|-----------|--------------|---------|--------|
| tiny | 0.5B | CPU | 2.0 ± 0.0 | 432.9 | -99.5% | FAIL |
| tiny | 0.5B | GPU | 131.7 ± 25.7 | 411.5 | -68.0% | FAIL |
| small | 1.5B | CPU | 2.0 ± 0.1 | 309.9 | -99.3% | FAIL |
| small | 1.5B | GPU | 132.3 ± 24.6 | 315.3 | -58.1% | FAIL |
| large | 32B | GPU | 148.1 ± 20.3 | N/A* | N/A | N/A |

*Ollama does not have qwen2.5-coder:32b installed

### APR Format Results

| Tier | Size | Backend | Status | Notes |
|------|------|---------|--------|-------|
| large | 32B | N/A | N/A | APR format is for traditional ML, not LLMs |

### LLaMA Format Results

LLaMA format is synonymous with GGUF format (llama.cpp uses GGUF).

---

## Key Findings

### 1. GPU Acceleration Works
- GPU provides ~66x speedup over CPU (132 vs 2 tok/s)
- APR GPU achieves 130-150 tok/s across model sizes

### 2. Ollama is Significantly Faster
- Ollama achieves 300-430 tok/s on the same models
- APR is 58-68% slower than Ollama on GPU

### 3. CPU Performance is Poor
- APR CPU achieves only ~2 tok/s
- This is 99% slower than Ollama

### 4. Statistical Quality
- CV ranges from 15-20% (target: <5%)
- High variance indicates inconsistent performance

---

## PMAT Falsification Status

| Point | Requirement | Status | Notes |
|-------|-------------|--------|-------|
| 41 | +25% vs llama.cpp | FAIL | No llama.cpp baseline available |
| 42 | ≥60 tok/s | PASS* | GPU achieves 130-150 tok/s |
| 49 | CV <5% | FAIL | CV is 15-20% |
| 50 | ≥30 runs | FAIL | Tests used 10 runs |

*Point 42 passes on throughput but context matters - Ollama is still 2-3x faster.

---

## Detailed Results

### Tiny (0.5B) CPU
```
APR: 2.0 ± 0.0 tok/s, TTFT: 499.6ms (10 runs)
Ollama (qwen2.5-coder:0.5b): 432.9 tok/s, TTFT: 147.0ms
Speedup vs Ollama: -99.5% FAIL
```

### Tiny (0.5B) GPU
```
APR: 131.7 ± 25.7 tok/s, TTFT: 7.9ms (10 runs)
Ollama (qwen2.5-coder:0.5b): 411.5 tok/s, TTFT: 6.0ms
Speedup vs Ollama: -68.0% FAIL
```

### Small (1.5B) CPU
```
APR: 2.0 ± 0.1 tok/s, TTFT: 495.0ms (10 runs)
Ollama (qwen2.5-coder:1.5b): 309.9 tok/s, TTFT: 109.0ms
Speedup vs Ollama: -99.3% FAIL
```

### Small (1.5B) GPU
```
APR: 132.3 ± 24.6 tok/s, TTFT: 7.8ms (10 runs)
Ollama (qwen2.5-coder:1.5b): 315.3 tok/s, TTFT: 5.0ms
Speedup vs Ollama: -58.1% FAIL
```

### Large (32B) GPU
```
APR: 148.1 ± 20.3 tok/s, TTFT: 6.8ms (5 runs)
Ollama baseline: N/A (32B model not installed)
```

---

## Reproduce Commands

```bash
# Build with CUDA
cargo build --release -p apr-cli --features cuda

# Test tiny CPU
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier tiny --runs 10 --model-dir models

# Test tiny GPU
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier tiny --runs 10 --gpu --model-dir models

# Test small CPU
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier small --runs 10 --model-dir models

# Test small GPU
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier small --runs 10 --gpu --model-dir models

# Test large GPU
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier large --runs 5 --gpu --model-dir models
```

---

## Conclusions

1. **GPU acceleration is essential** - CPU performance is unusable for production
2. **APR is 2-3x slower than Ollama** - Performance gap exists
3. **High variance needs investigation** - CV of 15-20% is unacceptable
4. **The +25% speedup target is not achievable** with current implementation

## Recommendations

1. Profile GPU kernels to identify bottlenecks
2. Implement speculative decoding
3. Optimize KV cache management
4. Consider tensor core utilization
