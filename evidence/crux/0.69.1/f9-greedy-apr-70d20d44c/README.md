# F9 greedy rows — apr 0.69.1 (70d20d44c) vs llama.cpp b10987, lambda RTX 4090, 2026-09-23

Producer: `scripts/crux_inference_dogfood.sh 0.69.1 --engines apr,llama.cpp --verbs run --greedy
--greedy-max-tokens 1024` on PMAT-3952-crux-greedy2, apr pinned to a snapshot of `70d20d44c`
(sha256 408c726b7f78991a…, which has `apr run --thinking`). Models: Qwen3.5-0.8B-IQ4_XS, 0.8B-Q4_K_M, 2B-Q4_K_M,
4B-Q4_K_M (control). Prompt: the positive control `golden-2plus2`.

- `greedy-manifest.jsonl`: the 24 kind=greedy rows (6 per model: apr ON/OFF, llama.cpp@apr ON/OFF,
  llama.cpp@official ON/OFF). Each `tokens` path is the raw object in `greedy/`.
- The dogfood's own receipt (lambda-gpu.json) is deliberately NOT committed (quorum lane 2, 2026-09-23: it is
  self-invalidating). Its gen cells are RED only because the run used the v1 prompt set, which F6 refuses, and its
  `greedy[]` came from the release judge's pre-F9 `report_greedy`, which keys on (model, host, prompt) and
  collapses the six rows per model. **Judge `greedy-manifest.jsonl` with aprender-36's judge (fix/3957-f9-f10).**

## What the rows say (read off greedy-manifest.jsonl; the judge's reading is aprender-36's)

| model | thinking | apr (own prompt) | llama.cpp on apr's ids | llama.cpp on the OFFICIAL template | apr vs llama.cpp first divergence |
|---|---|---|---|---|---|
| 0.8B-IQ4_XS | off | 7 tok "2+2 = 4" | 7 tok "2+2 = 4" | 9 tok "2 + 2 = **4**" | none |
| 0.8B-IQ4_XS | on | 1024 tok, UNCLOSED | 1024 tok, UNCLOSED | **332 tok, closed**, "2 + 2 = 4" | token 41 |
| 0.8B-Q4_K_M | off | 7 tok "2+2 = 4" | 7 tok | 10 tok | none |
| 0.8B-Q4_K_M | on | 14 tok, EMPTY block | 14 tok, EMPTY block | **241 tok, closed**, "2+2 equals 4." | none |
| 2B-Q4_K_M | off | 8 tok "2 + 2 = 4" | 8 tok | 9 tok | none |
| 2B-Q4_K_M | on | 447 tok, closed | 1024 tok, UNCLOSED | **544 tok, closed**, "2 + 2 = 4" | token 185 |
| 4B-Q4_K_M (control) | off | 9 tok | 9 tok | 13 tok | none |
| 4B-Q4_K_M (control) | on | 140 tok, closed | 181 tok, closed | 175 tok, closed | token 21 |

- On the OFFICIAL template every model closes its think block and answers correctly. The IQ4_XS "never closes"
  (#3907) and Q4_K_M "empty block" (#3948) are properties of apr's ON prompt (#3990), not of the models.
- apr's rendering differs from the official template in BOTH modes (`prompt_ids_equal: false` on every parity row).
- OFF: apr and llama.cpp generate identical ids from identical prompts on all four models. ON, on identical input
  ids, they diverge at tokens 41 / 185 / 21 (IQ4_XS / 2B / 4B); on the 2B that flips the outcome (apr closes at
  447, llama.cpp is still reasoning at 1024).
