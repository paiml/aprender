# L0-1 RED-first record — lambda (2026-09-06T13:5xZ), BEFORE any kernel edit
- host: noah-Lambda-Vector, GPU 0 NVIDIA GeForce RTX 4090 (`nvidia-smi -L`)
- binary: `/tmp/apr-0652-cuda/bin/apr` = `apr 0.65.2 (v0.65.2+no-git)`, sha256 prefix `c642576eecb62daa` (the 0.65.2 post-publish cuda install, `evidence/dogfood/0.65.2/lambda-cuda-install.txt`)
- command: `apr parity <model> --prompt "<78-token English paragraph>" --json` (files beside this record; stderr tails kept)
- models: `/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, `/home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf`
- result (78 positions each, `metrics[].cosine_similarity`, derive: `python3 -c` over the JSON):
  | model | positions < 0.98 | min cosine (position) | max |Δlogit| at that position | argmax mismatches | apr parity verdict |
  |---|---|---|---|---|---|
  | qwen2.5-coder-1.5b-instruct-q4_k_m | **1** | **0.9508 (position 0 = the prompt's first token, token_id 785 — NOT the BOS 151643 the load-time gate measures; corrected by the root-cause quorum)** | 11.97 | 2 | passed 78/78 (its own per-position thresholds) |
  | qwen2.5-coder-7b-instruct-q4_k_m | 0 | 0.9986 (position 0) | 0.78 | 2 | passed 78/78 |
- reading: under the L0-1 horizon rule (min over ≥ 64 positions ≥ 0.98 [U]) the 1.5B is RED and the 7B is GREEN on lambda/sm_89 — the pattern the driver names, at position 0 — the prompt's first token (785), a DIFFERENT token from the BOS the load-time gate measures, so a gate PASS and a parity RED are consistent. The driver's 0.9418 / 5.38 are not this host's numbers; gx10 (GB10) is the other required host and is reached only through fleet-verify (G-11b).
- threshold 0.98 is [U] (driver); `apr parity` itself passed every position under its own bands — a threshold nobody measured decides RED here, which is exactly item (5) of the card.
