<!-- PCU: getting-started-first-inference | contract: contracts/apr-page-getting-started-first-inference-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# First Inference

Run a language model in 3 commands:

```bash
# 1. Install
cargo install aprender

# 2. Download a model from HuggingFace
apr pull qwen2.5-coder-1.5b

# 3. Run inference
apr run qwen2.5-coder-1.5b "Explain quicksort in Rust"
```

## From a Local File

```bash
# GGUF format (most common)
apr run model.gguf "What is 2+2?"

# SafeTensors format
apr run model.safetensors --prompt "Hello, world"

# APR format (native)
apr run model.apr "Translate to Spanish: Good morning"
```

## Options

```bash
# Control generation
apr run model.gguf "prompt" --max-tokens 100 --temperature 0.7 --top-p 0.9

# JSON output
apr run model.gguf "prompt" --json

# Verbose (show timing, tokens/sec)
apr run model.gguf "prompt" --verbose

# GPU acceleration
apr run model.gguf "prompt" --gpu
```

## Interactive Chat

```bash
apr chat model.gguf
# > What is Rust?
# Rust is a systems programming language...
# > How does ownership work?
# ...
```
