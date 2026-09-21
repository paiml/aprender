# #3571 unit (2) — `apr serve` answers Qwen3.5 from its resident session, every size, streaming and not

Measured by aprender-c7, 2026-09-21, `apr 0.69.0 (51c5bb2ce)` (the unit-(2) head with the /health fix), `gpu-q --prio 1`,
`apr serve run Qwen3.5-<size>-Q4_K_M.gguf --gpu-layers all`, then one `/v1/chat/completions` request non-streaming and one
streaming: greedy, `max_tokens` 64, user *"What is the capital of Peru? Answer in one word."* (`serve_e2e.sh`).

| host | size | up (/health 200) | load s | `Backend: GPU` | fallback | gpu-layers | chat template | non-stream | stream |
|---|---|---|---|---|---|---|---|---|---|
| lambda (RTX 4090) | 0.8B | yes | 3 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima.⏎</think>⏎⏎Lima.` | 200 `Lima.⏎</think>⏎⏎Lima.` |
| lambda (RTX 4090) | 2B | yes | 6 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda (RTX 4090) | 4B | yes | 6 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda (RTX 4090) | 9B | yes | 13 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda (RTX 4090) | 27B | yes | 60 | 1 | 0 | requested=all resolved=64 total=64 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 (GB10) | 0.8B–27B | pending — queued `gpu-q --prio 1`; follows as an evidence-only commit, not claimed by this receipt | | | | | | | |

Every lambda size: the server reports healthy, holds all its layers on the GPU (`resolved = total`: 24/24/32/32/64), prints
the template line, and answers HTTP 200 streaming and non-streaming, 2B–27B "Lima". The 0.8B reply carries a stray
`</think>` — the `Qwen3NoThink` scaffold (`<think>\n</think>\n`) differs from the model's own (`<think>\n\n</think>\n\n`),
#3755 (aprender-fd), which replaces it; this route takes whatever the shared selector renders.
