# The `contracts` dogfood gate — diagnosed, half fixed, half is c7's receipts

**Blocker 1 of the three CPU-bound rehearsal blockers.** Worker: aprender-d8.
Branch: `fix/census-readme-after-3715` off `origin/release/0.69.1-batch-2`.

## Verdict

The dogfood row read:

```
[FAIL] contracts   armed meet: Fail (11 armed; not armed: reverse-coverage)
```

That note names the wrong thing. **`reverse-coverage` is not the cause** — it is
`Unknown(Skip)` and explicitly *not armed*, so it cannot move the meet. The meet
is Fail because exactly one armed gate is Fail: **`shapes`**.

And `make contracts` was failing for a **second, unrelated reason** before it ever
reached that verdict — one I caused.

## 1. The cause I caused, and fixed here

`make contracts` regenerates `contracts/census.json` and then
`git diff --exit-code`s it against the tracked copy. Folding my
`qwen35-hybrid-serve-dispatch-v1.yaml` (#3715) added one `registry: true`
contract without regenerating it:

```
- "registry": 519      →  + "registry": 520
- "unanchored": 1822   →  + "unanchored": 1823
  id_set_sha256 changed
FAIL: the tracked census differs from a fresh one
```

Regenerating exposed a second consequence: `readme_sync` requires README's two
`CONTRACT_COUNT` blocks to equal `census.json .n_files`, so 1829 → **1830**.

This branch is the two-file remedy: regenerated census + `make readme-sync`.
After it, `make contracts` gets past both steps and
`lint_passes_on_real_contracts`'s report shows every gate `Pass` except the two
below.

**Adding a registry contract is a three-part change** — the YAML, the census, and
the README count. Nothing told me that at the time; the census diff is the only
thing that does, and only when `make contracts` runs. Worth knowing before the
next contract lands.

## 2. The cause that is not mine: `shapes` / `ladder-green`

One violation, on a tree with 1830 contracts and 2357 focus nodes:

```
ont:model/d98cdcbd… (rung qwen3-8b-q4km) violates shape `ladder-green` (maxCount):
  has 1 value(s) of ont:model/missingGreenHost, maxCount is 0: lambda
```

`missingGreenHost` is computed in `ontology/receipts.rs::resolve` as
`expected_hosts − hosts_with_a_green_witness`, joined by sha256 over **every**
`*.json` under `evidence/dogfood/models/**` — all versions, not just the current
one. Only `0.68.1/` and `0.68.2/` exist in the tree.

The 0.68.2 receipts show exactly why lambda is missing:

| host | `qa_rc` | `golden_output` |
|---|---|---|
| **lambda** | 5 | **false** — `"golden_output: Empty output"` |
| gx10 | 0 | true — `"3 golden test cases passed"` |

Both rows carry `required: false`. **This is the 0.69.0 hold, surfaced.** The cop
flipped `qwen3-8b-q4km` to `required: true`, which armed `ladder-green` for that
rung, and the historical lambda failure it had been masking is now a violation.
The shape is doing precisely its job.

### Consequence for the release

**`contracts` and `declared:check_model_ladder` are ONE cause, not two.** Both
clear when a `0.69.1` receipt exists showing that rung green on lambda. Because
`resolve` unions green witnesses across all versions, a green 0.69.1 lambda
receipt is sufficient — the stale 0.68.2 failure row does not have to be removed
or edited.

The cop reports c7 has already measured the rung 8/8 green on both hosts after
#3724. That measurement is not yet an `evidence/dogfood/models/0.69.1/lambda.json`
receipt, and the receipt is what the shape reads.

## 3. `reverse-coverage` is a red herring, and the note is why

```
GateResult { name: "reverse-coverage", passed: false, skipped: true,
             verdict: Unknown(Skip),
             detail: Skipped { reason: "no --binding or --crate-dir provided" } }
```

It is skipped by design when `pv lint` is invoked without `--binding`/`--crate-dir`,
lands as `Unknown(Skip)` in the ONT-6 lattice, and is excluded from the armed set
(`11 armed; not armed: reverse-coverage`). It is named in the dogfood note only
because that note is the first line matching `error|fail|warning:` — the same
note-picker that showed a benign nightly-rustfmt warning on a PASSING `fmt` row.
A reader of the row would chase `reverse-coverage` and find nothing.

## 4. A laundered exit code, noted not fixed

`Makefile:588` is

```make
@. scripts/pv_bin.sh && "$$PV" lint contracts/ 2>&1 | tail -5
```

A pipeline, so `pv lint`'s exit status is discarded and only `tail`'s survives.
The `armed meet: Fail` line is therefore **printed but not enforced** by the
`contracts` target; the target fails later, on the census/README/engine-test
steps. The repo already has a gate for exactly this shape —
`contracts-exit-integrity`, which checks the recipe for `|| true` and bare
for-loops — and it passes here, so it does not catch a `| tail`. Flagged rather
than changed: altering what the release's contract gate enforces is not a
release-night edit, and it is the cop's file.

## 5. Checks

`pv census` regenerated and committed · `make readme-sync` rc=0, both
`CONTRACT_COUNT` blocks at 1830 · `cargo test -p aprender-contracts --lib
lint::tests::lint_passes_on_real_contracts` — every gate `Pass` except `shapes`
(the receipt cause) and `reverse-coverage` (`Unknown(Skip)`, not armed).

## 6. Not claimed

* Not claimed that `make contracts` is green. It is not, and cannot be until the
  0.69.1 lambda receipt lands. This branch removes the two causes that are not
  about receipts.
* The `| tail -5` exit laundering is reported, not fixed.
