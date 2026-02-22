# APR Qualify Matrix

Cross-subcommand smoke test results across all local and cached models.
Each cell shows whether `apr <subcommand>` completes without crashing on the model file.

**Tool:** `apr qualify` (11-gate cross-subcommand smoke test)
**Last Updated:** 2026-02-22 13:43 UTC

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
| qwen2.5-coder-1.5b.apr | APR | 6.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 56.7s |
| qwen2.5-coder-7b-instruct-q4k.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 128.7s |
| qwen2.5-coder-7b-instruct-exported.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 111.9s |
| qwen2.5-coder-7b-instruct.apr | APR | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 107.9s |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0:model.safete... | SafeTensors | 2.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 20.6s |
| stabilityai/stablelm-2-zephyr-1:6b_stablelm-2-z... | GGUF | 937M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 24.6s |
| TheBloke/Mistral-7B-v0.1-GGUF:mistral-7b-v0.1.Q... | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 99.8s |
| bartowski/Phi-3.5-mini-instruct-GGUF:Phi-3.5-mi... | GGUF | 2.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 💥 | 10/11 | 59.3s |
| bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral... | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 114.9s |
| bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K... | GGUF | 1.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 41.0s |
| bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-... | GGUF | 1.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 49.0s |
| bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-... | GGUF | 770M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.2s |
| HuggingFaceTB/SmolLM2-360M:model.safetensors | SafeTensors | 690M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 7.1s |
| HuggingFaceTB/SmolLM2-135M:model.safetensors | SafeTensors | 256M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 2.4s |
| Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instru... | GGUF | 1.9G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 51.2s |
| unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.1s |
| unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf | GGUF | 378M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.5s |
| Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf | GGUF | 609M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.4s |
| Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-in... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.2s |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-in... | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 11.0s |
| openai/whisper-small:model.safetensors | SafeTensors | 922M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 3.6s |
| openai/whisper-base:model.safetensors | SafeTensors | 276M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.1s |
| Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf | GGUF | 4.6G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 121.2s |
| bigcode/tiny:starcoder_py_model.safetensors | SafeTensors | 626M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 2.3s |
| facebook/galactica-125m:model.safetensors | SafeTensors | 238M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 1.5s |
| EleutherAI/gpt-neo-125m:model.safetensors | SafeTensors | 501M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.9s |
| bigscience/bloom-560m:model.safetensors | SafeTensors | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 6.9s |
| Qwen/Qwen3-0.6B:model.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 10.9s |
| microsoft/phi-1:5_model.safetensors | SafeTensors | 2.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 18.1s |
| EleutherAI/pythia-410m-deduped:model.safetensors | SafeTensors | 869M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 4.7s |
| deepseek-ai/deepseek-coder-1.3b-instruct:model.... | SafeTensors | 2.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.9s |
| Qwen/Qwen2-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.2s |
| Qwen/Qwen2.5-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.4s |
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
