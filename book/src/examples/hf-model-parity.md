# Hugging Face Top 50 Model Parity Report

This report verifies the compatibility of `apr` with the top 50 text-generation models on Hugging Face, mapped to our available hardware.

## Hardware Profiles
- **lambda-labs**: RTX 4090 (24GB VRAM) + 128GB System RAM
- **gx10**: NVIDIA GB10 + 128GB System RAM
- **mini**: Apple Silicon M4 with 16GB Unified RAM

## Verification Matrix

| Model | Family | Est. Size | Supported Hardware | `apr` Compatibility |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | Qwen3 | 0.6B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-8B | Qwen3 | 8B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| facebook/opt-125m | Meta OPT | 125M | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| openai-community/gpt2 | OpenAI GPT-2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-7B-Instruct | Qwen2 / Qwen2.5-Coder | 7B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen2.5-1.5B-Instruct | Qwen2 / Qwen2.5-Coder | 1.5B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-4B | Qwen3 | 4B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-Embedding-0.6B | Qwen3 | 0.6B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 | Qwen2 / Qwen2.5-Coder | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| meta-llama/Llama-3.1-8B-Instruct | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ⚠️ Gated (Auth Required) |
| meta-llama/Llama-3.2-1B-Instruct | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ⚠️ Gated (Auth Required) |
| deepseek-ai/DeepSeek-R1 | DeepSeek | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| openai/gpt-oss-20b | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-3B-Instruct | Qwen2 / Qwen2.5-Coder | 3B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| antirez/deepseek-v4-gguf | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ❌ Unsupported |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | Qwen3 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen3-1.7B | Qwen3 | 1.7B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-4B-Instruct-2507 | Qwen3 | 4B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| hmellor/tiny-random-LlamaForCausalLM | LLaMA 3 / LLaMA 3.2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen3-32B | Qwen3 | 32B | lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen2.5-0.5B-Instruct | Qwen2 / Qwen2.5-Coder | 0.5B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| google/gemma-3-270m | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ⚠️ Gated (Auth Required) |
| dphn/dolphin-2.9.1-yi-1.5-34b | LLaMA 3 / LLaMA 3.2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| openai/gpt-oss-120b | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| google/gemma-3-1b-it | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ⚠️ Gated (Auth Required) |
| Qwen/Qwen3-14B | Qwen3 | 14B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen2.5-Coder-14B-Instruct | Qwen2 / Qwen2.5-Coder | 14B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen2.5-7B-Instruct-AWQ | Qwen2 / Qwen2.5-Coder | 7B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen2-1.5B-Instruct | Qwen2 / Qwen2.5-Coder | 1.5B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| distilbert/distilgpt2 | OpenAI GPT-2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| EleutherAI/pythia-160m | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-32B-Instruct | Qwen2 / Qwen2.5-Coder | 32B | lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-30B-A3B | Qwen3 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ❌ Unsupported |
| Qwen/Qwen3-Embedding-4B | Qwen3 | 4B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| zai-org/GLM-4.7-Flash | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| deepseek-ai/DeepSeek-R1-0528 | DeepSeek | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| HuggingFaceTB/SmolLM2-135M-Instruct | LLaMA 3 / LLaMA 3.2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-0.5B | Qwen2 / Qwen2.5-Coder | 0.5B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-Reranker-0.6B | Qwen3 | 0.6B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| Qwen/Qwen3-Coder-Next-FP8 | Qwen3 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| meta-llama/Llama-3.2-3B-Instruct | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ⚠️ Gated (Auth Required) |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | LLaMA 3 / LLaMA 3.2 | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| nvidia/Gemma-4-26B-A4B-NVFP4 | Google Gemma | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| zai-org/GLM-5-FP8 | Unknown | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-14B-Instruct | Qwen2 / Qwen2.5-Coder | 14B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| deepseek-ai/DeepSeek-V4-Flash | DeepSeek | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| deepseek-ai/DeepSeek-V3.2 | DeepSeek | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
| Qwen/Qwen2.5-Coder-7B-Instruct | Qwen2 / Qwen2.5-Coder | 7B | mini (16GB), lambda-labs (128GB RAM), gx10 (128GB RAM) | ✅ Supported |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | NVIDIA Llama-3.1-Nemotron | Unknown | lambda-labs (24GB), gx10 (128GB), mini (16GB) [Est.] | ✅ Supported |
