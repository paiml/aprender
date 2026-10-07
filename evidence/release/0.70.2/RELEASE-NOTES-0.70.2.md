# aprender 0.70.2 — release notes

## How this release was gated

0.70.2 ships under a recorded operator emergency scope, `crux-smoke`, as 0.70.1 and 0.69.1 did.
The operator's words: "0.70.2 ships on CRUX smoke on lambda and gx10 GPU. Everything bigger is nightly."
The record is the `crux-smoke` entry for release "0.70.2" under `emergency_scopes` in
`contracts/model-capability-ladder-v1.yaml`. It is data only: the judge
(`scripts/lib/crux_smoke_scope.py`) matches a recorded scope by its whole release string, so no code changed
to admit it.

**The gate.** The CRUX smoke runs on the two GPUs the quote names (an RTX 4090 and a GB10), with nothing else on
either GPU. It covers every certified model (Qwen3.5-2B Q4_K_M, Qwen3.5-4B Q4_K_M and Qwen3.5-4B UD-Q4_K_XL),
every admitted thinking mode (off, on), on the control prompts, and the verbs run, chat, serve and code. The
smoke is run with the binary built at the release commit. A commit cannot hold results produced from itself,
so those receipts are published with the GitHub release for v0.70.2.

**The prompt certification is carried from 0.70.1, not re-run.**
`evidence/crux/0.70.2/prompt-certification.json` and `prompt-certification-inventory.json` are byte copies of
the `evidence/crux/0.70.1/` files. Carrying is allowed because nothing the certification depends on changed:
- the models are the same: the three certified files have the same sha256 as for 0.70.1;
- the prompts are the same: between v0.70.1 and this release's base, `evidence/crux/0.70.1`, the 0.69.1 prompt
  certification, `scripts/crux_inference_dogfood.sh`, `scripts/lib/crux_smoke_scope.py` and
  `scripts/crux_sweep_shards.sh` have no diff;
- the oracle engines are the same: the oracle record is still the 0.69.1 manifests, pinned by sha256 inside
  the certification, which 0.70.1 carried the same way;
- the smoke does not re-run the oracle engines.

**Printed as evidence, not the gate for this release.** The full model matrix and the release-readiness grade
still run at publish, and their verdicts are printed. They stay red where they were red; see Known failures.

**Unchanged.** Clean tree, tag on HEAD, release branch, dogfood GO for this commit, and the clean-room green on
the tagged commit.

## Changed since 0.70.1
- `apr serve`: tools in a chat request now reach the chat template, and Qwen3.5 XML tool calls are parsed
  (#4650).
- `apr serve`: KV caches no longer commit every byte when they are built (#4769).
- The PP-26 witness marker under `evidence/perf041/` is re-measured on the base of this release,
  so the release-phase check `scripts/check_perf041_marker.sh` judges fresh evidence (#4888).
- Release, CI and guard fixes on main since 0.70.1, among them: the coverage gate reads the nightly's
  sha-keyed receipt and is checked before the tag (#4734); one list in code of what merge, tag and publish
  require (#4688); no public GitHub release before its assets, clean-room and preflight (#4690).

## Known failures (shipped, moved to 0.71.0)
The operator refused all six for 0.70.2 under the scope above. None has a per-model contract edit, and none is
marked known red or green. Each stays red in the printed evidence and in its ticket, and each is on the 0.71.0
milestone. Each was re-measured on the RTX 4090 on 2026-10-06, against 0.70.1 built from main, and still fails the same way.

| model | GPU | check | failure | ticket |
|---|---|---|---|---|
| qwen35-0.8b-q4km | RTX 4090, GB10 | golden_output thinking_on | think block unclosed within 2048 tokens | https://github.com/paiml/aprender/issues/4666 |
| Qwen2.5-0.5B-Instruct-f16.gguf | RTX 4090 | cuda serve /api/chat | gibberish output | https://github.com/paiml/aprender/issues/4661 |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | RTX 4090, GB10 | cuda serve /api/chat | gibberish output | https://github.com/paiml/aprender/issues/4662 |
| Qwen3.5-0.8B-IQ4_XS.gguf | RTX 4090 | golden_output thinking_on | no measured thinking budget (named refusal) | https://github.com/paiml/aprender/issues/4663 |
| Qwen3.5-0.8B-UD-IQ2_XXS.gguf | RTX 4090, GB10 | golden_output | GPU forward fell back to CPU (run and chat rc=14); `apr qa` also fails tensor_contract | https://github.com/paiml/aprender/issues/4664 |
| Qwen3.5-35B-A3B-UD-IQ4_XS.gguf | RTX 4090 | capability_match | no CUDA forward for qwen35moe (named refusal) | https://github.com/paiml/aprender/issues/4665 |

## Moved to 0.71
- The six known failures above (#4661, #4662, #4663, #4664, #4665, #4666).
- One unattended autopilot pass. `scripts/release/autopilot.sh` (its readiness step and `cut_tag`) does not
  read a recorded emergency scope; only the publish preflight does. No fix before this release.

## Not measured for 0.70.2
- Capability cells (`model_ladder.sh --cells`), as for 0.70.1: a gap, not a pass
  (https://github.com/paiml/aprender/issues/4667).
- The full CRUX sweep (beyond the smoke above) and CPU lanes: nightly, per the scope.
