# [X] load evidence: llama.cpp d1d3c3396 loads Qwen3.5-0.8B on intel CPU

The orchestrator measured this at 2026-09-15 ~16:52Z, independently of the pin-bump worker's receipt.
The worker's version of this line existed only in a `scripts/llama_pin.toml` comment. This file puts it on disk.

- tree: aprender pin-bump branch 03e301475; llama.cpp ~/src/llama.cpp-d1d3c3396 (build b2744-d1d3c3396, GGML_NATIVE=ON, CPU)
- host: intel
- model: ~/models/Qwen3.5-0.8B-Q4_K_M.gguf, 532517120 bytes, sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517

## Command and result

    timeout 300 ./build/bin/llama-cli -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -p "hi" -n 1 -st --no-warmup -t 8 -v < /dev/null

    rc = 0
    llama_model_loader: - kv   0:  general.architecture str = qwen35
    llama_model_loader: - kv  14:  qwen35.block_count    u32 = 24
    print_info: arch                  = qwen35
    (no "unknown model architecture" line in 2313 log lines)

The old pin 39173bcac fails on the same model: `error loading model architecture: unknown model architecture: 'qwen35'`, rc=1 (#3303).

## Two traps at this commit

- **The loader lines only print under `-v`.** The new llama-cli is a chat TUI. Without `-v`, the same run shows only `Prompt: 76.1 t/s`. That proves a forward pass ran, but not which architecture loaded, so an architecture grep on a default-verbosity log finds nothing.
- **`-no-cnv` was removed.** At this commit it fails with `error: invalid argument: -no-cnv`, before any model is read. The single-turn replacement is `-st` / `--single-turn`. A caller still passing `-no-cnv` looks like it failed to load when it never tried. The orchestrator's first re-run hit exactly this and returned rc=1.

For logits, use llama-perplexity or a raw-logit producer, never llama-cli. It is interactive, and it hangs on stdin unless every invocation closes stdin under `timeout`.
