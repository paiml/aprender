# Benchmarks

This directory contains benchmark results and comparison tools for Realizar.

## Running Benchmarks

Realizar includes three comprehensive benchmark suites:

### 1. Inference Benchmarks

Measures end-to-end inference latency:

```bash
cargo bench --bench inference
```

Benchmarks include:
- Model forward pass (varying sequence lengths: 1, 5, 10, 20)
- Generation with different sampling strategies (greedy, top-k, top-p)
- Varying generation lengths (1, 5, 10, 20 tokens)
- Batch generation (batch sizes: 1, 2, 4, 8, 16)

### 2. Tensor Operation Benchmarks

Measures low-level tensor operations:

```bash
cargo bench --bench tensor_ops
```

Benchmarks include:
- Matrix multiplication (various sizes)
- Vector operations (dot product, addition, etc.)
- Activation functions (ReLU, GELU, softmax)

### 3. Cache Benchmarks

Measures model cache performance:

```bash
cargo bench --bench cache
```

Benchmarks include:
- Cache hit latency (~40ns)
- Cache miss + load latency (~14µs)
- Concurrent access (4 threads: ~94µs)
- Eviction performance

## Benchmark Results

Benchmark results are saved in `target/criterion/` directory after running `cargo bench`.

### Example Results (Test Model: 32h, 2L)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Forward pass (1 token) | 17.5 µs | Single token inference |
| Forward pass (10 tokens) | 78.6 µs | Batch of 10 tokens |
| Generation (5 tokens) | 1.54 ms | Greedy sampling |
| Generation (10 tokens) | 3.12 ms | Greedy sampling |
| Cache hit | 40 ns | ⚡ Cached model access |
| Cache miss | 14.2 µs | Cold model load |

## Comparing with llama.cpp

Realizar provides a comparison tool to benchmark against llama.cpp (the reference C++ implementation).

### Step 1: Export Realizar Benchmarks

First, run Realizar benchmarks and export results to JSON:

```bash
# Run all benchmarks
cargo bench --bench inference
cargo bench --bench cache

# Export results (manual process - see format below)
# Results are in target/criterion/ directory
```

Create a JSON file (`realizar_results.json`) with this format:

```json
{
  "model": "model_name",
  "config": {
    "vocab_size": 100,
    "hidden_dim": 32,
    "num_heads": 1,
    "num_layers": 2
  },
  "benchmarks": {
    "model_forward": {
      "seq_len_1": {"mean": 17.5, "std": 0.5, "unit": "µs"}
    },
    "generation": {
      "greedy_5_tokens": {"mean": 1.54, "std": 0.03, "unit": "ms"}
    }
  }
}
```

**Example files provided:**
- `realizar_results_example.json` - Sample Realizar results
- `llamacpp_results_example.json` - Sample llama.cpp results

### Step 2: Run llama.cpp Benchmarks (Optional)

If you have llama.cpp installed, run equivalent benchmarks:

```bash
# Install llama.cpp (if not already installed)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run benchmarks with similar model configuration
./main --model path/to/model.gguf --n-predict 10 --threads 1 --prompt "test"

# Export timing results to JSON format (llamacpp_results.json)
# Format should match the Realizar JSON structure above
```

### Step 3: Generate Comparison Report

Use the comparison script to generate a markdown report:

```bash
# Compare Realizar vs llama.cpp
python3 scripts/compare_benchmarks.py \\
  --realizar benchmarks/realizar_results.json \\
  --llamacpp benchmarks/llamacpp_results.json \\
  --output benchmarks/comparison_report.md

# Or just show Realizar results
python3 scripts/compare_benchmarks.py \\
  --realizar benchmarks/realizar_results.json
```

### Step 4: Review Results

The comparison report includes:
- Configuration comparison
- Performance metrics side-by-side
- Speedup calculations (Realizar vs llama.cpp)
- Winner determination (with ±5% equivalence threshold)

**Sample comparison:**

```
| Benchmark | Realizar | llama.cpp | Speedup | Winner |
|-----------|----------|-----------|---------|--------|
| model_forward/seq_len_1 | 17.50 µs | 19.20 µs | 1.10x | ✅ Realizar |
| generation/greedy_5_tokens | 1.540 ms | 1.680 ms | 1.09x | ✅ Realizar |
```

## Fair Comparison Guidelines

To ensure fair comparisons between Realizar and llama.cpp:

### 1. **Model Configuration**
- Use identical model architectures (hidden_dim, num_heads, num_layers)
- Same quantization format (Q4_0, Q8_0, etc.)
- Equivalent vocab size

### 2. **System Configuration**
- Same hardware (CPU/GPU)
- Same thread count
- Same SIMD features enabled (AVX2, NEON, etc.)
- Disable power management/frequency scaling

### 3. **Benchmark Methodology**
- Warm-up runs before measurement
- Multiple iterations for statistical significance
- Report mean ± std dev
- Use same input prompts/tokens

### 4. **Workload Characteristics**
- Compare equivalent operations:
  - Forward pass: Same sequence lengths
  - Generation: Same token counts, same sampling strategy
  - Batch: Same batch sizes

### 5. **Environment**
- Run benchmarks with minimal background processes
- Use release builds with optimizations enabled
- Document system specs (CPU, RAM, OS)

## Performance Targets

Realizar aims to match or exceed llama.cpp performance while maintaining:
- 🦀 100% Pure Rust (zero unsafe in public API)
- ⚡ SIMD acceleration via Trueno
- 🎯 EXTREME TDD quality (95+ TDG score)
- 📦 Multiple format support (GGUF, SafeTensors)
- 🌐 Production-ready HTTP API

**Current Performance (Test Model):**
- Forward pass: ~17µs per token
- Generation: ~300µs per token (greedy)
- Cache hits: ~40ns
- Cache misses: ~14µs

## Benchmark Philosophy

Realizar's benchmark philosophy:
1. **Transparency**: All benchmarks are open-source and reproducible
2. **Fairness**: Use equivalent configurations and workloads
3. **Honesty**: Report both strengths and weaknesses
4. **Context**: Include system specs and methodology
5. **Reproducibility**: Provide scripts and instructions

## Contributing Benchmarks

To contribute new benchmarks:
1. Follow EXTREME TDD (write tests first)
2. Use criterion for statistical rigor
3. Document methodology clearly
4. Include multiple workload sizes
5. Report mean, std dev, and sample size

## Resources

- [Criterion Documentation](https://bheisler.github.io/criterion.rs/book/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Realizar Documentation](../README.md)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Built with EXTREME TDD** 🦀⚡
