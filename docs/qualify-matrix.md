# APR Qualify Matrix

Cross-subcommand smoke test results across all local and cached models.
Each cell shows whether `apr <subcommand>` completes without crashing on the model file.

**Tool:** `apr qualify` (11-gate cross-subcommand smoke test)
**Last Updated:** 2026-02-22 14:50 UTC

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | PASS — subcommand completed successfully |
| ❌ | FAIL — subcommand returned an error |
| ⏭️ | SKIP — gate skipped (e.g. missing feature flag) |
| 💥 | PANIC — subcommand panicked (crash bug) |
| ⏰ | TIMEOUT — exceeded 180s deadline |

## Results

<!-- QUALIFY_MATRIX_START -->
| Model | Format | Size | Inspect | Validate | Val.Quality | Tensors | Lint | Debug | Tree | Hex | Flow | Explain | Check | Score | Duration |
|-------|--------|------|------|------|------|------|------|------|------|------|------|------|------|-------|----------|
| TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf | GGUF | 637M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 25.9s |
| model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 9.6s |
| qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 29.4s |
| qwen2.5-coder-1.5b-instruct-q4k.apr | APR | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.5s |
| qwen2.5-coder-1.5b.apr | APR | 6.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 56.0s |
| qwen2.5-coder-7b-instruct-q4k.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 128.7s |
| qwen2.5-coder-7b-instruct-exported.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 111.9s |
| qwen2.5-coder-7b-instruct.apr | APR | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 107.9s |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Deep... | GGUF | 4.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 113.8s |
| sentence-transformers/all-MiniLM-L6-v2:model.sa... | SafeTensors | 86M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | .3s |
| bartowski/Falcon3-3B-Instruct-GGUF:Falcon3-3B-I... | GGUF | 1.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 45.5s |
| bartowski/Falcon3-1B-Instruct-GGUF:Falcon3-1B-I... | GGUF | 1008M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.2s |
| bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:De... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.6s |
| unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf | GGUF | 2.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 58.1s |
| openai-community/gpt2-medium:model.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 5.3s |
| HuggingFaceTB/SmolLM2-1.7B:model.safetensors | SafeTensors | 3.1G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 36.2s |
| Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:qwen2.5-c... | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 10.7s |
| openai-community/gpt2:model.safetensors | SafeTensors | 522M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.8s |
| bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Qwen2.... | GGUF | 1.7G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 45.3s |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0:model.safete... | SafeTensors | 2.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 20.6s |
| stabilityai/stablelm-2-zephyr-1:6b_stablelm-2-z... | GGUF | 937M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 24.6s |
| TheBloke/Mistral-7B-v0.1-GGUF:mistral-7b-v0.1.Q... | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 97.8s |
| bartowski/Phi-3.5-mini-instruct-GGUF:Phi-3.5-mi... | GGUF | 2.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 💥 | 10/11 | 57.0s |
| bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral... | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 114.9s |
| bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K... | GGUF | 1.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 39.4s |
| bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-... | GGUF | 1.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 49.0s |
| bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-... | GGUF | 770M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.2s |
| HuggingFaceTB/SmolLM2-360M:model.safetensors | SafeTensors | 690M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 7.1s |
| HuggingFaceTB/SmolLM2-135M:model.safetensors | SafeTensors | 256M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 2.4s |
| Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instru... | GGUF | 1.9G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 49.8s |
| unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.1s |
| unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf | GGUF | 378M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.5s |
| Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf | GGUF | 609M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.4s |
| Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-in... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.2s |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-in... | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 11.0s |
| openai/whisper-small:model.safetensors | SafeTensors | 922M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 3.3s |
| openai/whisper-base:model.safetensors | SafeTensors | 276M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.0s |
| Qwen/Qwen2-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.2s |
| Qwen/Qwen2.5-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.4s |
| bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-... | GGUF | 4.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 126.9s |
| bartowski/Yi-1.5-6B-Chat-GGUF:Yi-1.5-6B-Chat-Q4... | GGUF | 3.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 95.4s |
| bartowski/Falcon3-7B-Instruct-GGUF:Falcon3-7B-I... | GGUF | 4.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 110.5s |
| Qwen/Qwen3-0.6B:model.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 10.3s |
| Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf | GGUF | 4.6G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 122.0s |
| bigscience/bloom-560m:model.safetensors | SafeTensors | 256M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 2.2s |
| microsoft/phi-1:5_model.safetensors | SafeTensors | 2.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 17.6s |
| deepseek-ai/deepseek-coder-1.3b-instruct:model.... | SafeTensors | 2.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.9s |
| facebook/galactica-125m:model.safetensors | SafeTensors | 238M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 1.3s |
| EleutherAI/gpt-neo-125m:model.safetensors | SafeTensors | 501M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.7s |
| EleutherAI/pythia-410m-deduped:model.safetensors | SafeTensors | 869M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 4.6s |
<!-- QUALIFY_MATRIX_END -->

## Running

```bash
# Run all local + cached models
bash scripts/qualify-matrix.sh

# Run only cached models (from apr pull)
bash scripts/qualify-matrix.sh --cached-only

# Run a single model
bash scripts/qualify-matrix.sh ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf
```

## Per-Model JSON

Raw JSON results are saved in [`docs/qualify-results/`](qualify-results/).
Each file contains the full gate-by-gate breakdown from `apr qualify --json`.
