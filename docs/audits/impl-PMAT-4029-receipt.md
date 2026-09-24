# PMAT-4029 — implementation receipt

- **Ticket:** PMAT-4029 (GH #4029), kind:code. Branch `fix/4029-crux-vllm-model-len`. Reviewed head `b98e2eeac`.
- **Base:** `origin/chore/0.69.1-merge-back` (#4046). `main` has no `scripts/crux_vllm/`, so the PR is retargeted to `main` after #4046 lands.
- **Verdict:** DONE up to the quorum receipt, including acceptance 2 (the GPU rerun, below). Not armed.

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

## GPU rerun (acceptance 2) — run by aprender-83, verified here

Run on lambda (RTX 4090) under `/tmp/apr-gpu.lock`, on a clean card, 2026-09-24:
- **harness:** this branch @ `c460c63a7`, `engine.py` sha256 `06f12d5e2c79d237…`;
- **apr:** the 0.69.1 release binary `apr 0.69.1 (d8a6df53a)`;
- **shard:** Qwen3.5-4B-Q4_K_M (`00fe7986ff5f…`), thinking ON, `ctl-2plus2` + `ctl-code-add`, with max_tokens 4096 and context 4096.

| vLLM ON row | pre-fix driver (X2 smoke manifest) | this branch |
|---|---|---|
| run / chat, both prompts (4 rows) | answered, no `max_model_len` field | answered, `max_model_len` 8192 |
| `serve run` ctl-2plus2 | **HTTP 400** | answered `<answer>4</answer>`, 8192 |
| `serve stream` ctl-2plus2 | **HTTP 400** | answered `<answer>4</answer>`, 8192 |
| `code` ctl-code-add | **HTTP 400** | answered `def add(a, b): return a + b`, 8192 |

**Mechanism proof.** All three `vllm serve` server logs read `'max_model_len': 8192`, which is 4096 + 4096.

**Verdict.** The judge over the rerun returned rc 0, with 19/19 cells GREEN. vLLM answered in each quorum.

**Checked by the orchestrator.** I re-read the rerun manifest (7/7 vLLM rows, rc 0, 8192), grepped the server logs, compared the engine sha with this branch, and re-read the pre-fix manifest (the same 3 rows refused with HTTP 400).

**Where the evidence lives.** The artifacts sit outside the tree, in lambda `/mnt/nvme-raid0/tmp/a83-4029/`: `RERUN-RECEIPT.md`, `out/lambda-gpu.json` and `work/`. The pre-fix manifest is `evidence/crux/0.69.1/d8a6df53a/lambda-gpu.manifest.jsonl` in aprender-83's X2 worktree. The freeze-v2 work dir, from the 15-row sweep on 4f81bdea7, was cleaned from /tmp, so the X2 smoke manifest (the same pre-fix driver) is the baseline.

## Not run
- `scripts/guard_tree.sh --no-cargo` locally: a partial run was stuck in `check_crux_inference_judge.sh` (a file this diff does not touch). CI runs the full tree.
