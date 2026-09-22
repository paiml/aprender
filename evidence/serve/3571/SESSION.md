# #3571 unit (2) — `apr serve` answers Qwen3.5 from its resident session: every size, both GPU hosts, GPU and `--no-gpu`, streaming and not

Measured by aprender-c7, 2026-09-21, `apr 0.69.0 (51c5bb2ce)` (the unit-(2) head with the /health fix), every run `gpu-q --prio 1`:
`apr serve run Qwen3.5-<size>-Q4_K_M.gguf --gpu-layers all` (GPU rows) or `--no-gpu` (CPU rows), wait for `/health` 200, then one
`/v1/chat/completions` request non-streaming and one streaming — greedy, `max_tokens` 64, user *"What is the capital of Peru? Answer in one word."*
(`serve_e2e.sh`). "up" is `/health` answering 200.

| route | size | up | load s | `Backend: GPU` | fallback | gpu-layers | chat template | non-stream | stream |
|---|---|---|---|---|---|---|---|---|---|
| lambda GPU | 0.8B | yes | 3 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima.⏎</think>⏎⏎Lima.` | 200 `Lima.⏎</think>⏎⏎Lima.` |
| lambda GPU | 2B | yes | 6 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda GPU | 4B | yes | 6 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda GPU | 9B | yes | 13 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda GPU | 27B | yes | 60 | 1 | 0 | requested=all resolved=64 total=64 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 GPU | 0.8B | yes | 3 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima.⏎</think>⏎⏎Lima.` | 200 `Lima.⏎</think>⏎⏎Lima.` |
| gx10 GPU | 2B | yes | 6 | 1 | 0 | requested=all resolved=24 total=24 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 GPU | 4B | yes | 8 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 GPU | 9B | yes | 30 | 1 | 0 | requested=all resolved=32 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 GPU | 27B | yes | 47 | 1 | 0 | requested=all resolved=64 total=64 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda --no-gpu | 0.8B | yes | 2 | 0 | 0 | requested=none resolved=0 total=24 | Qwen3NoThink (thinking off) | 200 `Lima.⏎</think>⏎⏎Lima.` | 200 `Lima.⏎</think>⏎⏎Lima.` |
| lambda --no-gpu | 2B | yes | 5 | 0 | 0 | requested=none resolved=0 total=24 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda --no-gpu | 4B | yes | 9 | 0 | 0 | requested=none resolved=0 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda --no-gpu | 9B | yes | 8 | 0 | 0 | requested=none resolved=0 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| lambda --no-gpu | 27B | yes | 40 | 0 | 0 | requested=none resolved=0 total=64 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 --no-gpu | 0.8B | yes | 1 | 0 | 0 | requested=none resolved=0 total=24 | Qwen3NoThink (thinking off) | 200 `Lima.⏎</think>⏎⏎Lima.` | 200 `Lima.⏎</think>⏎⏎Lima.` |
| gx10 --no-gpu | 2B | yes | 3 | 0 | 0 | requested=none resolved=0 total=24 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 --no-gpu | 4B | yes | 2 | 0 | 0 | requested=none resolved=0 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 --no-gpu | 9B | yes | 11 | 0 | 0 | requested=none resolved=0 total=32 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |
| gx10 --no-gpu | 27B | yes | 19 | 0 | 0 | requested=none resolved=0 total=64 | Qwen3NoThink (thinking off) | 200 `Lima` | 200 `Lima` |

Every row: `/health` 200, HTTP 200 streaming and non-streaming, 2B–27B "Lima". GPU rows hold every layer on the GPU (`resolved = total`) with no
fallback; `--no-gpu` rows resolve 0 layers and never print `Backend: GPU` — the route reported is the route taken. The 0.8B reply's stray
`</think>` is the `Qwen3NoThink` scaffold (`<think>\n</think>\n`, not the model's own `<think>\n\n</think>\n\n`) that #3755 (aprender-fd) replaces;
this route renders whatever the shared selector picks.

## Round reconciliation — the one real round judged an older head (aprender-c7, 2026-09-22)

The single quorum round this row has (2 PASS, 1 FAIL) is **not a live FAIL against this receipt**; it judged a head that predates the
table above. Which head each side judged:

| side | head | what it saw |
|---|---|---|
| the round | `b3a1336c7` (code range `015a6b38d..51c5bb2ce`) | `gemini-3.1-pro-high` PASS, `gemini-3.1-pro-low` PASS, third lane FAIL on **evidence gaps only** — no finding against the code |
| this receipt | `c9691a7d6` (the branch head; `b3a1336c7` was the evidence-only commit that answered the FAIL) | the 20/20 table above |

Seat 3's two findings, and their answers:

1. *"the receipt omits the `gx10` host — done_when requires serve loads every Qwen3.5 Q4_K_M size on CUDA on both hosts"* — **closed.**
   The table has 10 gx10 rows (GPU and `--no-gpu`, 0.8B through 27B), each `/health` 200 with the reply recorded.
2. *"no evidence for `--no-gpu` runs (Acceptance 2)"* — **closed.** The table has 10 `--no-gpu` rows across both hosts, each showing
   `requested=none resolved=0` and no `Backend: GPU` line — the route reported is the route taken.

Both findings were raised against the receipt, not the diff, and both were answered by the evidence-only commit `b3a1336c7`. The artifact
for that round is **not committed** (it sits in the author's worktree, `docs/audits/quorum-PMAT-3571-unit2.json`, `agreed: false`): it is
recorded here so a later reader does not re-litigate a FAIL that its own follow-up commit already closed, and so nobody mistakes 2 PASS
on an older head for seats on this one. **This row is 0/3 at `c9691a7d6`.**

A separate #3595 round on 2026-09-21 returned NO-VERDICT ×3 (gemini and gpt-oss both 429, every lane's fallback chain exhausted). That is
an empty model pool, not a review, and is recorded nowhere as a verdict.
