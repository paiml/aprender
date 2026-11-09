# Qwen3.5-4B batched witnesses (#5312): notes

- `witness-qwen35-4b-batched-run{1,2,3,4}.json`: four `apr test llm parity --ladder 1,4,8,16` runs. Same binary
  (`apr 0.71.3 (8f601caff2)`, `--features cuda`, sha256 `26073455f173…`), same model (sha256 `00fe7986ff5f…`), RTX 4090,
  `CUDA_BATCH_WINDOW_MS=200`. `apr test llm shape-check` = ADMISSIBLE for all four.
- **run2, band c=8:** slots `i=6` and `i=7` have bit-identical `decode_tok_s` (27.574169291500333). The per-slot rate is
  (completion_tokens − 1) ÷ (last − first client-side token arrival), at nanosecond resolution, and each slot comes from
  its own HTTP stream. The run's logs carry no per-slot timestamps, so the cause is **unproven**. The file is kept
  unchanged, as measured.
- **run4** re-runs run2 to check for the duplicate. Every slot rate is distinct in every band (1/1, 4/4, 8/8, 16/16).
- No admissibility result or S7 figure depends on run2's slots 6 or 7 individually (the band value is the median).
