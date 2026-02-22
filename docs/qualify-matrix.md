# APR Qualify Matrix

Cross-subcommand smoke test results across all local and cached models.
Each cell shows whether `apr <subcommand>` completes without crashing on the model file.

**Tool:** `apr qualify` (11-gate cross-subcommand smoke test)
**Last Updated:** 2026-02-22 16:21 UTC

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
| qwen2.5-coder-1.5b.apr | APR | 6.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 50.9s |
| qwen2.5-coder-7b-instruct-q4k.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 128.7s |
| qwen2.5-coder-7b-instruct-exported.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 111.9s |
| qwen2.5-coder-7b-instruct.apr | APR | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 107.9s |
| bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral... | GGUF | 7.1G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 154.7s |
| unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf | GGUF | 378M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.5s |
| unsloth/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf | GGUF | 4.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 136.0s |
| bartowski/Qwen2.5-3B-Instruct-GGUF:Qwen2.5-3B-I... | GGUF | 1.7G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 45.8s |
| bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF:Dee... | GGUF | 4.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 119.7s |
| bartowski/Falcon3-1B-Instruct-GGUF:Falcon3-1B-I... | GGUF | 1008M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.2s |
| Qwen/Qwen2-1.5B-Instruct:model.safetensors | SafeTensors | 2.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 29.9s |
| bartowski/Yi-1.5-6B-Chat-GGUF:Yi-1.5-6B-Chat-Q4... | GGUF | 3.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 95.4s |
| bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-... | GGUF | 4.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 126.9s |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Deep... | GGUF | 4.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 113.8s |
| bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Qwen... | GGUF | 940M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.2s |
| bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Qwen2.... | GGUF | 1.7G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 45.3s |
| bartowski/Falcon3-3B-Instruct-GGUF:Falcon3-3B-I... | GGUF | 1.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 45.5s |
| bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:De... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.6s |
| bartowski/Falcon3-7B-Instruct-GGUF:Falcon3-7B-I... | GGUF | 4.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 110.5s |
| BAAI/bge-small-en-v1.5:model.safetensors | SafeTensors | 127M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | .4s |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-in... | GGUF | 644M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 21.2s |
| bartowski/codegemma-2b-GGUF:codegemma-2b-Q4_K_M... | GGUF | 1.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 38.1s |
| bartowski/Phi-3.5-mini-instruct-GGUF:Phi-3.5-mi... | GGUF | 3.7G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 💥 | 10/11 | 77.1s |
| bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-... | GGUF | 3.1G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 86.7s |
| bartowski/gemma-2-9b-it-GGUF:gemma-2-9b-it-Q4_K... | GGUF | 5.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 127.8s |
| unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf | GGUF | 1.7G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 52.6s |
| bartowski/internlm2:5-1_8b-chat-GGUF_internlm2_... | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.7s |
| bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-... | GGUF | 770M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 18.9s |
| unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q8_0.gguf | GGUF | 3.9G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 102.1s |
| 1f324656416dd0c0.gguf | GGUF | 5.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 148.8s |
| 9a368706285251ee.gguf | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 109.2s |
| 6a520edef23b374e.gguf | GGUF | 1.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 49.0s |
| 59dce3937a1047d8.gguf | GGUF | 2.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 31.1s |
| 5c84a14af979daf5.safetensors | SafeTensors | 11.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 57.4s |
| b55269866992c38b.safetensors | SafeTensors | 626M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 2.3s |
| 12d2cd8a877ef2cd.safetensors | SafeTensors | 256M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 2.2s |
| ab38e015b12ecacd.safetensors | SafeTensors | 2.5G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.9s |
| d85d9e4e87e3f5e7.safetensors | SafeTensors | 501M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.9s |
| dfbede4c565dbc61.safetensors | SafeTensors | 869M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 4.8s |
| c0a9a5ca190a98ea.safetensors | SafeTensors | 238M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | 1.6s |
| 2d8c454d9196a7a4.safetensors | SafeTensors | 256M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 2.4s |
| dbdc8d8af7bb5874.safetensors | SafeTensors | 3.1G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 36.2s |
| bfee789b8139c227.safetensors | SafeTensors | 690M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 7.1s |
| 0b805cd343d00fa3.safetensors | SafeTensors | 2.6G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 21.7s |
| 58b398c3ec4c9f82.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 4.8s |
| 3bca1b7f6c327dae.safetensors | SafeTensors | 522M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 1.8s |
| 38e63c2a12def40a.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.3s |
| eb23c3e8527110b8.safetensors | SafeTensors | 690M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 5.7s |
| 4895d53b19436198.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.2s |
| d4c4d9763127153c.gguf | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 11.0s |
| 59dce3937a1047d8.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.4s |
| 2619da49b802f7bc.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 27.2s |
| a2ccd7e2387fc42e.safetensors | SafeTensors | 2.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 32.1s |
| c8490f8cd005ac4e.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 30.1s |
| e910cab26ae116eb.gguf | GGUF | 468M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 10.7s |
| d71534cb948e32eb.safetensors | SafeTensors | 942M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 8.3s |
| b7a969a05a81cc52.safetensors | SafeTensors | 2.8G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 35.1s |
| e0abfc1f71fa8f14.gguf | GGUF | 4.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 127.2s |
| 9be9143c23d8d0cd.gguf | GGUF | 609M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 23.4s |
| dc5e5f1509eee291.safetensors | SafeTensors | 1.4G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 11.3s |
| 26d2f46c2813fe27.gguf | GGUF | 4.6G | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 9/11 | 141.0s |
| 8087e9bf97c265f8.safetensors | SafeTensors | 86M | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8/11 | .3s |
| cb7ee7cd50590460.gguf | GGUF | 937M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 24.6s |
| f04c88340c00d6e8.gguf | GGUF | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 115.3s |
| 684041cd46bfafc7.safetensors | SafeTensors | 2.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 20.6s |
| 0693e999fc960643.safetensors | SafeTensors | 4.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 24.5s |
| f6de13c77f843535.gguf | GGUF | 1.2G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 41.8s |
| 927fe84ea33e6527.gguf | GGUF | 1.0G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11/11 | 26.1s |
| 88cf0561b5415ce3.gguf | GGUF | 2.3G | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 10/11 | 59.0s |
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
