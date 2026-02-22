# APR Qualify Matrix

Cross-subcommand smoke test results across all local and cached models.
Each cell shows whether `apr <subcommand>` completes without crashing on the model file.

**Tool:** `apr qualify` (11-gate cross-subcommand smoke test)
**Last Updated:** 2026-02-22 13:13 UTC

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
| qwen2.5-coder-1.5b.apr | APR | 6.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 58.7s |
| qwen2.5-coder-7b-instruct-q4k.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 128.7s |
| qwen2.5-coder-7b-instruct-exported.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 111.9s |
| qwen2.5-coder-7b-instruct.apr | APR | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 107.9s |
| Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instru... | GGUF | 1.9G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 51.0s |
| unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.1s |
| unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf | GGUF | 378M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.5s |
| Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf | GGUF | 609M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.4s |
| Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-in... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.2s |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-in... | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 11.0s |
| openai/whisper-small:model.safetensors | SafeTensors | 922M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 3.3s |
| openai/whisper-base:model.safetensors | SafeTensors | 276M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.0s |
| Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf | GGUF | 4.6G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 120.7s |
| bigcode/tiny:starcoder_py_model.safetensors | SafeTensors | 626M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 2.3s |
| facebook/galactica-125m:model.safetensors | SafeTensors | 238M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 1.3s |
| EleutherAI/gpt-neo-125m:model.safetensors | SafeTensors | 501M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.8s |
| bigscience/bloom-560m:model.safetensors | SafeTensors | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 7.0s |
| Qwen/Qwen3-0.6B:model.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 11.1s |
| microsoft/phi-1:5_model.safetensors | SafeTensors | 2.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 18.7s |
| EleutherAI/pythia-410m-deduped:model.safetensors | SafeTensors | 869M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 4.7s |
| deepseek-ai/deepseek-coder-1.3b-instruct:model.... | SafeTensors | 2.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.9s |
| Qwen/Qwen2-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.2s |
| Qwen/Qwen2.5-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.4s |
| Qwen/Qwen2.5-Coder-1.5B-Instruct:model.safetensors | SafeTensors | 2.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 35.1s |
| Qwen/Qwen2.5-Coder-0.5B-Instruct:model.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.3s |
| bigcode/starcoder2-3b:model.safetensors | SafeTensors | 11.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 57.4s |
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
