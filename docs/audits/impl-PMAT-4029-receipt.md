# PMAT-4029 — implementation receipt

- **Ticket:** PMAT-4029 (GH #4029), kind:code. Branch `fix/4029-crux-vllm-model-len`. Reviewed head `b98e2eeac`.
- **Base:** `origin/chore/0.69.1-merge-back` (#4046). `main` has no `scripts/crux_vllm/`, so the PR is retargeted to `main` after #4046 lands.
- **Verdict:** DONE up to the quorum receipt. Not armed. Acceptance 2 (the GPU rerun) is **NOT RUN**.

## What changed
`scripts/crux_vllm/engine.py`: vLLM `max_model_len` = context + the largest `max_tokens` of the group that actually
loads. That group is taken after the `item_error` refusals and the mixed-interface refusal. The value goes to both
`vllm.LLM(max_model_len=)` and `vllm serve --max-model-len`, and every row records it. Before this change it was the
context alone, and `vllm serve` returned HTTP 400 on 15 thinking-ON rows (context 4096, ON budget 4096).

## Verification (re-run by the orchestrator, not taken from lanes)
| Check | Result |
|---|---|
| `python3 scripts/crux_vllm/test_engine.py` | exit 0, 52/52, log sha256 `8aa228bc…54d0` |
| Case table on the original fix commit, with `model_len()` reverted | 39/39 green. It was **vacuous** for the fix |
| Mutants across rounds, each compiled, RED on the final case table | call sites (inproc, serve), context-only, 2×context, +4096, off-by-one, min, clamps at 8192 and 16384, pow2 / round-even / round-16, sizing over live / slots / refused, row never set, row records context, thinking gate, single-item skip, empty default 4096, no `max()` default, sizing after preflight, fakes not restored |
| `check_roadmap_{fragment_required,sorted,ids_unique,completion_is_cited,diff_additive}.sh` | rc 0 |

## Quorum (`docs/audits/quorum-PMAT-4029.json`)
- **Rounds 1–3 (agy, three gemini ids):** no verdict counted. Every lane exited 3 because other sessions' worktrees moved refs in the shared `.git`. Each round's substantive finding was still fixed.
- **Rounds 4–7 (Claude Code `claude-sonnet-5` ×3, degraded: same-family, per the operator fall-back rule):** FAIL findings in rounds 4, 5 and 6 were each measured and fixed.
- **Round 7:** **3/3 PASS** on `b98e2eeac`.

## Not run
- A real thinking-ON `vllm serve` rerun at the certified ON budget on GPU. No GPU on lambda or gx10 until the cop pings "0.69.1 published".
- `scripts/guard_tree.sh --no-cargo` locally: a partial run was stuck in `check_crux_inference_judge.sh` (a file this diff does not touch). CI runs the full tree.
