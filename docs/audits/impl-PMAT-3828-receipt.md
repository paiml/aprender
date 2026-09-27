# #3828 — the release gate's verb dimension computed over ONE verb

**Branch** `release/0.69.1-batch-2` · **Author** aprender-cop (claude-opus-5) · 2026-09-22

## The defect

`contracts/model-capability-ladder-v1.yaml:75` has declared `verbs: [run, chat, serve, code]`
all along. The producer measured two things:

```
$ grep -oE "apr_locked (run|chat|serve|code|qa)" scripts/model_ladder.sh | sort | uniq -c
      1 apr_locked qa
      1 apr_locked run
```

So three of the four verb columns in `release-readiness-v1` (#3715) were computed over
nothing. This is **declared-vs-measured** — the same class as PMAT-124's declared-vs-measured
model id — located in the release gate itself.

**A live defect walked through it.** @alfredodeza found (#3715, 2026-09-22) that `/api/chat`
cannot reach the Qwen3.5 hybrid session. Confirmed on this branch **after** #3571 unit (2)
landed: `api/ollama_handlers.rs` has **zero** qwen35 references and
`api/cuda_batch_scheduler.rs` has two unconditional `generate_gpu_resident_streaming` calls.
#3571 stated the cause in its own words — *"every gate we own is single-stream `apr run`; no
rung has ever asked `apr serve` to load a model"* — and that was still literally true.

## What changed

**Producer** (`scripts/model_ladder.sh`), per rung per backend:

| verb | form |
|---|---|
| `run` | unchanged |
| `chat` | stdin-driven one turn, `apr chat --json --max-tokens 16`, `/exit` closes it |
| `code` | `apr code -p ... --model ... --output-format json` |
| `serve` | `ladder_serve_probe()` — starts `apr serve run --port N`, bounded 90 s `/health` wait, probes each generation route **non-streaming and streaming**, shuts down |

The **route set is DERIVED** from the string literals registered in
`crates/aprender-serve/src/api/router.rs`, never hand-listed. A hand-listed set is exactly how
`/api/chat` went unprobed, and a constant in the producer would drift from the router the
moment anyone adds an endpoint. The derivation currently yields `/api/chat`,
`/v1/chat/completions`, `/v1/completions` — the three routes @alfredodeza named, **found rather
than typed**. A server that never answers `/health` is a FAIL naming the wait, never a skip.

**Judge** (`scripts/check_model_ladder.sh`, inside `why_of`) refuses **by name**: a receipt with
no `verbs` object; any of `run`/`chat`/`code` missing or not run; `serve` missing; `serve` not
probed (quoting the recorded reason); an empty route set; any route not returning 200; and a
`serve` probe that recorded **no `/api/chat`** — because the ollama-compat layer has its own
translation and has diverged from the OpenAI-compat one twice independently (#3825's
`tool_calls` gap, and this defect), so its coverage is never inherited from a representative
route.

## Case table

**51 cases, 0 bad.** Baseline **48 cases, 1 bad** — measured by stashing this change and
re-running in the same directory, because a control run from a different directory is how I
once talked myself into reverting a correct change.

| case | asserts |
|---|---|
| `red-verb-missing` (new) | a verb absent from a receipt is refused by name |
| `red-serve-no-api-chat` (new) | `serve` green on `/v1` with `/api/chat` never probed is REFUSED — **the case that would have caught @alfredodeza's defect** |
| `red-serve-not-probed` (new) | an unprobed `serve` is a FAIL naming the reason, not a skip |
| `red-escaped-special` (rebuilt) | see below |

`red-escaped-special` **could not fire**: a v1-schema fixture with no `inventory` and an
`opt-q4km` rung at `required: false`, cpu-only, so #3712's rules refused it three ways before
the escaped-special check was ever reached. Another check that never fired on the thing it
exists for. Rebuilt on the v2 shape with `escaped_special` as its only defect; now passes with
`must_match` `templated twice`.

## Mutants — run, not asserted

Deleting the verb+route block from `why_of` turns **all three** new cases GREEN, i.e. each is
KILLED by the mutation. The judge was restored byte-identical afterwards (`diff -q` clean). A
rule that cannot be killed proves nothing.

## READING THIS DIFF — the 91 fixture receipts are mechanical

The diff is ~465 KB, and **almost all of it is data, not logic.** 91 fixture receipts under
`scripts/lib/model_ladder_cases/*/receipts/*.json` gained a `verbs` object. It is the **same
object** in every one, inserted by a script:

```
distinct added lines across ALL 91 receipts: 62
```

So the reviewable surface is `scripts/model_ladder.sh`, `scripts/check_model_ladder.sh`, and
the three new case dirs' small `expected_rc`/`must_match` files. The 91 receipts are worth
checking for *uniformity* (62 distinct lines for 91 files) and for the fact that the self-test
moved 48/1-bad → 51/0-bad, not for reading individually.

## Verified

`bash scripts/check_model_ladder.sh --self-test` rc=0, 51 cases 0 bad · `bash -n` on both
scripts · `bashrs lint` 0 errors · `pv validate` on the ladder contract 0 errors 0 warnings ·
`cargo fmt --all -- --check` rc=0.

## NOT claimed

- The `/api/chat` routing **fix** is not here (aprender-d8 holds it, with the two contracts
  @alfredodeza asked for). This row makes the gate *able to see* the defect; it does not repair it.
- CRUX slice 4 (#3797) is still unbuilt.
- The anti-shrink floor still refuses #3817's deliberate `qwen3moe` removal — two correct rules
  in tension, needing a declared-removal mechanism rather than a relaxation.
- **The four verbs have not yet been measured on either host.**
  `evidence/dogfood/models/0.69.1/{lambda,gx10}.json` do not exist. Nothing here should be read
  as those cells being green; this row only makes their absence refusable.

Refs #3828, #3715, #3712, #3571, #3825, #3791.
