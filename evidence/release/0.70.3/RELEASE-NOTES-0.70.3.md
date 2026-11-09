# aprender 0.70.3 — release notes

0.70.3 is a patch release that carries one change: streaming tool calls (Refs #4918).
Its published-path diff against `v0.70.2` (`crates`, `src`, `Cargo.toml`, `Cargo.lock`) is the #4918
fix's files plus the version bump, and nothing else.

## How this release was gated

0.70.3 ships under a recorded operator emergency scope, `crux-smoke`, as 0.70.2, 0.70.1 and 0.69.1 did.
The operator's words: "0.70.3 ships on CRUX smoke on lambda and gx10 GPU. Everything bigger is nightly."
The record is the first `emergency_scopes` entry, `crux-smoke` for release "0.70.3", in
`contracts/model-capability-ladder-v1.yaml`. The standing policy there still starts at 0.71.0.

**The gate.** The CRUX smoke runs on the RTX 4090 and the GB10, with nothing else on either GPU, over every
certified model, both thinking modes, the control prompts, and the verbs run, chat, serve and code, with the
binary built at the release commit. Those receipts are published with the GitHub release for v0.70.3.

**The prompt certification is carried from 0.70.2, not re-run.**
`evidence/crux/0.70.3/prompt-certification.json` and `prompt-certification-inventory.json` are byte copies of
the `evidence/crux/0.70.2/` files. Between v0.70.2 and this release's base, `evidence/crux/0.70.1`, the 0.69.1
prompt certification, `scripts/crux_inference_dogfood.sh`, `scripts/lib/crux_smoke_scope.py` and
`scripts/crux_sweep_shards.sh` have no diff, and the certified model files are the ones pinned by sha256
inside the certification.

**Unchanged.** Clean tree, tag on HEAD, release branch, dogfood GO for this commit, and the clean-room green on
the tagged commit.

## Changed since 0.70.2

- `apr serve`: with `tools` in a request, a streamed chat (`stream: true` on `/v1/chat/completions`, and
  `/v1/chat/completions/stream`) now sends the model's tool call as `delta.tool_calls` and ends with
  `finish_reason: "tool_calls"`, as the non-streaming path already did. Both paths parse the call with the
  same parser. With no tools, or `tool_choice: "none"`, the stream is unchanged (Refs #4918).

## Known failures (shipped, moved to 0.71.0)
The operator refused all six for 0.70.2 and they stand for 0.70.3 under the scope above. None has a per-model contract edit, and none is
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

## Not measured for 0.70.3
- Capability cells (`model_ladder.sh --cells`), as for 0.70.2: a gap, not a pass
  (https://github.com/paiml/aprender/issues/4667).
- The full CRUX sweep (beyond the smoke above) and CPU lanes: nightly, per the scope.
