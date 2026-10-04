# Qwen3.5-9B temp-0 refs at 850 / 4k / 32k (#4261, 0.69.4 epic #4249)

0.69.3 G1 showed 9B as SKIPPED (cop scope cut: 4B only). This directory is the 9B reference set owed
before 0.69.4 acceptance.

## Inputs (all pinned)
- Model: `Qwen3.5-9B-Q4_K_M.gguf`, sha256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` (the sha in #4261).
- Prompts: the 0.69.3 G1 prompts (f5-g1), sha256 in `refs.json` per prompt (`p850` 85bfde6e…, `p4k` ac2b35c4…, `p32k` 722b514d…).
- apr rc: `apr 0.69.3 (7ff50ec2a)` = v0.69.3-rc.1, sha256 `9a04c8cc…632fe` (the same binary as the 4B G1 receipt).
- apr base: `apr 0.69.1 (eed4a959a)`, sha256 `301336320a942572…`.
- Oracle: llama.cpp `llama-server` build 10987 (d1d3c3396), `--jinja`, the GGUF's own chat template with
  `enable_thinking: false`, prompt ids fed to `/completion` as ids, `temperature 0, top_k 1`.
- Every apr cell: `apr run <gguf> -i <p>.txt --chat --temperature 0 -n 64 --json --backend cuda`, one gpu-q hold per cell, RTX 4090 (lambda).
  Every cell's envelope is `backend {requested: gpu, ran: gpu, fell_back: false}`. `nvidia-smi --query-compute-apps` was empty inside every hold.

## Result (`python3 build_refs.py <harness out>` prints this; rc=0)
| prompt | prompt tokens | apr rc ids | rc vs base 0.69.1 | rc vs llama.cpp | rc prefill trace |
|---|---|---|---|---|---|
| p850 | 1006 | 23 (stop) | IDENTICAL (23 ids) | agree on first 13 ids | 1006 tok / 492 ms / 2045 tok/s |
| p4k | 4133 | 64 (length) | IDENTICAL (64 ids) | agree on first 2 ids | 4133 tok / 2228 ms / 1855 tok/s |
| p32k | 31848 | 16 (stop) | not run (see below) | IDENTICAL (16 ids) | 31848 tok / 20074 ms / 1587 tok/s |

- **Template equivalence:** llama.cpp's official template gives the same prompt-token count as apr on all three prompts
  (1006 / 4133 / 31848). The first oracle pass used llama-server's default `enable_thinking: true` and was 2 tokens short
  (`<think>\n` vs apr's `<think>\n\n</think>\n\n`). It is not used here.
- **The two divergences are near-ties, measured, not assumed** (`margins.json`, llama.cpp top-5 logprobs at the divergence, the shared prefix fed as ids):
  - p850 pos 13: `;` −1.132, ` or` −1.202 (apr's pick), `,` −1.214 (llama's pick in the full run). The top 3 are within 0.08 nats.
    llama.cpp itself picked a different one of those three between its full run and this probe.
  - p4k pos 2: ` \`` −0.738 (llama), ` is` −0.855 (apr's pick). That is 0.117 nats, and apr's token is llama's #2.
  - Both continuations are equivalent rewordings of the same answer (see `refs.json` `apr_rc_text`). p32k agrees id-for-id.
- **base 0.69.1 at p32k was stopped by me after 527 s** (exit 143). 0.69.1 prefills one token at a time, so 31848 tokens would
  have held the GPU for ~40 min, and aprender-98's 0.69.3 rc.2 acceptance cell was queued behind it. rc == base holds
  at 850 and 4k. At 32k the ref is the llama.cpp oracle, which agrees id-for-id.

## Files
- `refs.json`: per prompt, the apr rc ids and text, the llama.cpp ids, the prompt sha256, and the first-divergence index.
- `margins.json`: the near-tie measurement above.
- `build_refs.py`: builds `refs.json` and the table from a harness `out/` directory. A cell that did not run is written as missing, never as a match.
- `harness/`: `run.sh` (all cells; its llama stage ran with the thinking-on oracle and was re-run), `oracle-only.sh` + `oracle.py`
  (the thinking-off oracle used here), `margin-only.sh` + `margin.py`. The paths are the lambda scratch dir `/mnt/nvme-raid0/tmp/g1-9b`.
