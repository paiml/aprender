---
phase: 01-differentiable-minilm-conformance
plan: 04
subsystem: fixtures
tags: [fixtures, contracts, setfit, minilm, tolerances, reproducibility, cb-510, supply-chain]
requires:
  - contract:setfit-encoder-conformance-v1
provides:
  - fixture:setfit-slice-apr
  - fixture:setfit-tokenizer-cases
  - fixture:setfit-forward-per-layer
  - fixture:setfit-pooling-normalize
  - fixture:setfit-loss-pair
  - fixture:setfit-gradients
  - fixture:setfit-optimizer-step
  - fixture:setfit-batch-invariance
  - fixture:setfit-full-model-reference
  - fixture:setfit-activation-reference
  - fixture:setfit-manifest-sha256
  - data:analytically-zero-gradients
  - data:zero-grad-floor
  - tooling:setfit-fixture-generator
  - tolerance-table:setfit-encoder-conformance-v1
affects:
  - crates/aprender-core/tests/fixtures/setfit/
  - contracts/setfit-encoder-conformance-v1.yaml
  - crates/aprender-core/Cargo.toml
  - scripts/setfit_fixtures/
tech-stack:
  added:
    - "uv project: setfit 1.1.3, sentence-transformers 5.7.0, torch 2.13.0, scikit-learn 1.9.0 (dev-only, never CI)"
    - "transformers >=4.41.0,<5 (constraint, not a reference pin)"
  patterns:
    - "corpus of record shared by slicer and generator so fixture texts and slice vocab cannot drift"
    - "per-family tolerance floor = 8*sqrt(W)*EPS_F32 so no measured-zero can disable a comparison"
    - "fail-closed upstream digest verification at BOTH entry points (slicer and generator)"
    - "structure-indented / numeric-inline JSON writer for reviewable large fixtures"
key-files:
  created:
    - scripts/setfit_fixtures/pyproject.toml
    - scripts/setfit_fixtures/uv.lock
    - scripts/setfit_fixtures/.python-version
    - scripts/setfit_fixtures/README.md
    - scripts/setfit_fixtures/corpus.py
    - scripts/setfit_fixtures/jsonfmt.py
    - scripts/setfit_fixtures/slice_model.py
    - scripts/setfit_fixtures/generate_fixtures.py
    - scripts/setfit_fixtures/fetch_full_weights.py
    - crates/aprender-core/tests/fixtures/setfit/slice_model.apr
    - crates/aprender-core/tests/fixtures/setfit/slice_config.json
    - crates/aprender-core/tests/fixtures/setfit/vocab_remap.json
    - crates/aprender-core/tests/fixtures/setfit/upstream_manifest.json
    - crates/aprender-core/tests/fixtures/setfit/tokenizer.json
    - crates/aprender-core/tests/fixtures/setfit/tokenizer_cases.json
    - crates/aprender-core/tests/fixtures/setfit/activation_reference.json
    - crates/aprender-core/tests/fixtures/setfit/forward_per_layer.json
    - crates/aprender-core/tests/fixtures/setfit/pooling_normalize.json
    - crates/aprender-core/tests/fixtures/setfit/loss_pair.json
    - crates/aprender-core/tests/fixtures/setfit/gradients.json
    - crates/aprender-core/tests/fixtures/setfit/optimizer_step.json
    - crates/aprender-core/tests/fixtures/setfit/batch_invariance.json
    - crates/aprender-core/tests/fixtures/setfit/full_model_reference.json
    - crates/aprender-core/tests/fixtures/setfit/tolerances_measured.json
    - crates/aprender-core/tests/fixtures/setfit/manifest.sha256
  modified:
    - contracts/setfit-encoder-conformance-v1.yaml
    - crates/aprender-core/Cargo.toml
decisions:
  - "`apr import --arch bert` is the SafeTensors->APR path; `apr convert` was tried first per plan and is APR->APR only (its own help says 'Path to .apr model file')"
  - ".python-version pins 3.13.7 not 3.13 — a bare 3.13 resolves to a stale x86_64 CPython on this arm64 box and torch 2.13.0 has no macOS-x86_64 wheel"
  - "transformers capped <5: setfit 1.1.3 declares an unbounded transformers>=4.41.0 and v5 removed default_logdir, breaking `import setfit`"
  - "slice vocabulary is 97 canonical ids (not the ~256 estimated) because the truncation case is built by repetition, adding tokens without widening the embedding table"
  - "added 4 proof obligations so pooling/loss/post-step/full-model fixtures each have a gate; without them three frozen fixtures had nothing requiring Rust to match them"
  - "measured GELU exact-vs-tanh gap is 4.734993e-04, NOT the >1e-3 the plan asserted; recorded the measurement and set the assertion from it"
  - "zero_grad_floor derived from the observed gradient distribution (6.1291398e-05), one order below the smallest genuinely non-zero max|grad|"
metrics:
  duration: ~2h
  completed: 2026-08-08
  tasks: 3
  commits: 3
  fixtures_committed: 16
---

# Phase 1 Plan 04: Frozen Reference Plane and Tolerance Freeze Summary

A hash-locked Python reference environment, a 437 KB real-weight MiniLM slice that preserves
the model's true 32-dim head boundaries, the complete ENC-01..06 fixture corpus generated in
one deterministic pass, and a measured tolerance table frozen into the phase contract in its
own commit before any Rust comparison exists.

## What Was Built

**`scripts/setfit_fixtures/`** — a `uv` project pinning the four D-12 reference packages with a
committed hashed lockfile, plus five modules:

| File | Role |
|------|------|
| `corpus.py` | The CORPUS OF RECORD. Every fixture text, defined once, with stable case ids |
| `slice_model.py` | Pinned download, fail-closed digest check, index-slice, APR conversion |
| `generate_fixtures.py` | The whole ENC-01..06 corpus in one pass + SHA-256 manifest |
| `fetch_full_weights.py` | D-10 path: 86.7 MB `full_model.apr` + source/APR digest manifest |
| `jsonfmt.py` | Structure-indented / numeric-inline JSON writer (D-11) |

`corpus.py` exists as its own module rather than living inside the generator because
`slice_model.py` needs the texts *before* the generator runs — the slice vocabulary is the
closure of the ids those texts produce. Duplicating the list in both scripts would reintroduce
exactly the drift B6 exists to prevent.

**The slice (D-09).** 2 layers, hidden 64, **2 heads x 32**, intermediate 256, 97-id remapped
vocabulary, 64 positions — 437 KB of real index-sliced weights, no re-initialisation anywhere.
The head-boundary property is *proven*, not asserted in a comment: the source is 12x32, and
`assert_head_boundaries` checks that `[0:64)` is a whole number of complete original heads and
that the kept heads exactly tile the slice with no partial head. Slicing to 4x16 would have cut
each real 32-dim head in half and manufactured a synthetic attention structure out of real
numbers.

**16 committed fixtures**, 4.6 MB raw / ~2.0 MB compressed. Canonical HF ids and slice rows are
always separate, named fields, so tokenizer identity is never rewritten to fit the slice.

## What Makes These Fixtures Non-Tautological

**The B6 join is real and independently checked.** Every model-driven fixture carries `case_id`
plus a verbatim `texts` array, so 01-08 can call `SetFitMiniLm::tokenize(case.texts)` — the only
public batch producer after the D-08 seal — instead of hand-assembling a `SentenceBatch` and
silently bypassing the tokenizer boundary. The join is asserted twice: inside the generator, and
again by a standalone checker reading only the *committed* bytes. An assertion that lives only
in the producer proves the producer is self-consistent, not that the committed files are right.

**The ENC-04 exemption set is evidence.** Both `encoder.layer.{0,1}.attention.self.key.bias`
land at max|grad| of 9.5e-11 and 6.9e-11 against a `zero_grad_floor` of 6.13e-5 read off the
observed distribution — six orders of magnitude of separation, matching the softmax
shift-invariance proof exactly. The generator hard-fails if `key.bias` is absent from the list,
so a degenerate slice or a wrong reference gradient stops the run rather than quietly shrinking
the expectation.

**The tolerance floor is load-bearing, and provably so.** `full_model_reference` measured an
f32/f64 delta of **exactly 0.0**. Without the mandatory floor its tolerance would have been
`10 * 0.0 = 0.0` — a comparison that can never fail, i.e. silently disabled (T-1-07). This is the
precise scenario the floor was specified to prevent, and it occurred on the first run.

**Assumption A1 is now a verified fact.** Read from the locked sentence-transformers 5.7.0
source in this environment: `Pooling._forward_padded` clamps at `min=1e-9`, `Normalize.forward`
calls `F.normalize(p=2, dim=1)` (torch default eps `1e-12`). Then `full_model_reference`
cross-checks our manual masked-mean/L2 pipeline against a real `SentenceTransformer` forward —
they agree to **0.0**. A wrong constant fails here, not in wave 6.

**Generation is deterministic.** Two independent full runs produced a byte-identical
`manifest.sha256`, so a future manifest diff means the inputs changed, not that the generator is
noisy.

## Tasks and Commits

| Task | Commit | Result |
|------|--------|--------|
| 1 — uv env, slice, APR, CB-510 fix | `3b4fdc5a4` | slice 437 KB; all 5 fixtures packaged |
| 2 — full corpus + manifest | `900b400af` | 15 files under manifest; shasum -c 0 |
| 3 — tolerance freeze (single file) | `f9681e8cc` | pv validate 0/0; touches 1 file |

The tolerance commit is last and **no Rust comparison code exists anywhere in this plan**, so
D-14's ordering requirement is satisfied structurally: the table precedes every comparison
01-05/01-08 will write. This closes the STATE blocker *"freeze numerical tolerances from pinned
reference fixtures before examining Rust discrepancies."*

## The Frozen Tolerance Table

`tolerance = max(10 * measured_delta, floor)`, `floor(W) = 8*sqrt(W)*EPS_F32`.

| family | W | measured delta | frozen tolerance |
|--------|---|----------------|------------------|
| tokenizer | - | n/a | EXACT (no tolerance) |
| activation | 1 | 4.47239421e-07 | 4.47239421e-06 |
| forward_per_layer | 256 | 7.05113256e-07 | 1.52587891e-05 |
| pooling_normalize | 64 | 7.39103287e-08 | 7.62939453e-06 |
| loss_pair | 64 | 1.51211275e-07 | 7.62939453e-06 |
| batch_invariance | 64 | 5.21540642e-08 | 7.62939453e-06 |
| gradients | 1024 | 3.63171590e-07 | 3.05175781e-05 |
| optimizer_step | 1024 | 3.63171590e-07 | 3.05175781e-05 |
| full_model_reference | 1536 | **0.00000000e+00** | 3.73762473e-05 |

`zero_grad_floor` = 6.1291398e-05, carried as the `tolerance` of
`OBLIG-ENC-04-GRADIENT-AND-STEP-GATE` because that is the threshold its (d)/(e) clauses test.

## Verification

Every exit code below was captured with `cmd > file 2>&1; rc=$?` — never read through a pipe
(CLAUDE.md rule 1). That mattered twice here: see deviation 1.

| Check | Result |
|-------|--------|
| `uv lock --check` | 0 |
| all four reference pins import at the pinned versions on arm64 | torch 2.13.0 / st 5.7.0 / setfit 1.1.3 / sklearn 1.9.0 |
| `shasum -a 256 -c manifest.sha256` | 0, 15 files |
| two independent generator runs -> identical manifest | byte-identical (deterministic) |
| standalone B6 join + acceptance checker (85 assertions) | PASS |
| `git check-ignore -v` on every committed fixture | 1 (not ignored) for all 16 |
| `cargo package -p aprender-core --list --allow-dirty` | all 16 fixtures present |
| `scripts/check_include_files.sh` | 0 |
| `scripts/check_package_includes.sh` | 0 |
| `pv validate contracts/setfit-encoder-conformance-v1.yaml` | 0 errors, 0 warnings |
| tolerance commit file count | exactly 1 |
| no Rust file references the tolerance values as literals | 0 matches |
| `fetch_full_weights.py` (D-10) | 86.7 MB APR + config/tokenizer/modules + full_manifest.json |
| loss decreases across the recorded AdamW step | 0.141698 -> 0.138319 |

**Two guards were proven by mutation rather than by inspection** (CLAUDE.md rules 4 and 7):

1. *The CB-510 exclude fix.* Reverting `"/tokenizer.json"` to the bare `"tokenizer.json"` and
   re-running `cargo package --list` in this worktree packages only **4 of 5** fixtures —
   `tests/fixtures/setfit/tokenizer.json`, which the ENC-02 parity gate loads, is silently
   stripped while `tokenizer_cases.json` survives. Root-anchored, all 5 package. The fix is
   load-bearing, not decorative.
2. *The contract tolerances are genuinely parsed.* Mutating one to a string makes `pv` fail with
   `proof_obligations[3].tolerance: invalid type: string "not-a-number", expected f64`. They
   deserialize into the schema; they are not ignored text.

## Codegen Drift Check (Task 3 requirement)

`pv codegen contracts/ -o <tmp>` versus the committed
`crates/aprender-core/src/generated_contracts.rs` reports 61,263 changed lines, which looks like
catastrophic drift and is **entirely formatting**. Proven, not assumed:

- both sides define exactly **3415** macros with **identical name sets**
- whitespace-normalized digests are **IDENTICAL**: `0474ab4a26e1c764a9f4abce9585ea27a4206627c6d895c3935ad05c24134d04`

So this plan's contract edit produces **no semantic codegen change**, and the regenerated file
was correctly excluded from the single-file tolerance commit. `pv codegen` emits unformatted
Rust while the committed file is rustfmt'd — that is the whole difference.

**Action for 01-08 Task 1:** its planned "regenerate `generated_contracts.rs` if drift is
detected" step must NOT act on a naive `diff`, which will always report drift and would commit a
61k-line pure-reformatting churn that buries any real future change. Compare normalized
(`tr -d ' \n\t' | shasum -a 256`) or rustfmt the temp file first. Logged as deferred item D7.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `uv add` failed: wrong-architecture interpreter, not a bad pin**

- **Found during:** Task 1
- **Issue:** `uv add torch==2.13.0` failed with *"only has wheels for ... macosx_14_0_arm64"*
  while reporting the environment as `macosx_26_0_x86_64`. This reads exactly like the plan's A2
  fallback ("if the torch wheel does not resolve for 3.13.7, pin Python to 3.12") — but A2 would
  not have helped, because torch 2.13.0 has **no macOS-x86_64 wheel at any Python version**.
  The machine is an Apple **M4 Pro (arm64)**; `uv` had selected a stale *x86_64* managed CPython
  3.13.0 from `~/.local/share/uv/python/`. Reproduced in a clean scratch directory to confirm it
  was deterministic and not project state (CLAUDE.md rule 6: vary the input before naming a cause).
- **Fix:** pinned `.python-version` to `3.13.7`, which uniquely selects the arm64 interpreter
  (the stale managed x86_64 build is 3.13.0, so the patch version disambiguates). **No package
  version was changed** — all four D-12 pins are exactly as specified.
- **Files modified:** `scripts/setfit_fixtures/.python-version`
- **Commit:** `3b4fdc5a4`
- **Measurement note:** the background harness reported this failing command as *"exit code 0"*,
  because that was the trailing `echo`'s status rather than `uv`'s. The real `rc=2` was only
  visible in the captured log. This is CLAUDE.md rule 1 occurring live, and is why every status
  in this plan was captured directly.

**2. [Rule 3 - Blocking] `import setfit` was broken in the freshly locked environment**

- **Found during:** Task 1
- **Issue:** setfit 1.1.3 declares an **unbounded** `transformers>=4.41.0`, so uv resolved
  transformers 5.14.1, where `default_logdir` was removed from `transformers.training_args`.
  `import setfit` died with `ImportError`. `uv lock --check` passed the whole time — a lockfile
  can be perfectly consistent and still describe an unusable environment.
- **Fix:** added the missing cap `transformers>=4.41.0,<5` (resolved 4.57.6 — the same 4.57.x
  series 01-RESEARCH.md verified BERT dropout placement against). This also required tightening
  `requires-python` to `>=3.13,<3.14` (transformers 4.x has no 3.14 wheels, so an open-ended
  `>=3.13` produced an unsolvable resolution split) and dropping an over-tight
  `huggingface-hub>=1.27.0` that uv had auto-added (transformers 4.x caps it `<1.0`).
- **Files modified:** `scripts/setfit_fixtures/pyproject.toml`
- **Commit:** `3b4fdc5a4`

**3. [Rule 3 - Blocking] `apr convert` cannot read SafeTensors**

- **Found during:** Task 1
- **Issue:** the plan's primary path, `apr convert <slice.safetensors> -o <out.apr>`, fails:
  `error: Validation failed: At least one of --quantize or --compress must be specified`. Its own
  help states the input is a *"Path to .apr model file"* — it is an APR→APR quantize/compress
  optimizer, not an importer. (`CLAUDE.md` documents a `apr convert model.safetensors` example
  that cannot work; logged as D8.)
- **Fix:** used `apr import <file> -o <out> --arch bert`, which maps the sliced BERT tensor names
  natively — 37 tensors validated against `tensor-layout-v1.yaml`, HF dotted names preserved
  verbatim (D-18). The plan's Python-dump + dev-only-Rust-writer fallback was **not** needed. The
  importer needs a `config.json` beside the weights, so the slice's own HF-shaped config is
  written to `build/` first rather than letting it infer dims from tensor shapes.
- **Files modified:** `scripts/setfit_fixtures/slice_model.py`
- **Commit:** `3b4fdc5a4`

### Deliberate Departures

**4. The measured GELU exact-vs-tanh gap is 4.73e-4, not the asserted >1e-3**

The plan required `activation_reference.json` to record a tanh-vs-exact max delta **> 1e-3**, and
the 01-01 contract stated the forms *"differ by roughly 1e-3 near |x| ~ 2"*. Measured over a dense
[-6, 6] grid with the pinned torch: the true maximum is **4.734993e-04, at x ≈ -2.699**. The
planned threshold is simply false against the pinned reference.

Rather than fudge the fixture to satisfy the assertion, the measurement is recorded and the
assertion is set from it. **The conclusion is unaffected and is now stronger, because it is
enforced numerically:** the generator hard-fails unless the frozen activation tolerance is more
than 10x below the measured gap. It is 106x below (4.47e-6 vs 4.73e-4), so a tanh implementation
still fails the gate. The contract's prose was corrected to the measured number and location.

**5. Four proof obligations added to the contract**

The plan's acceptance criteria require tolerance values on *pooled/normalized embedding*, *loss
forward*, *post-optimizer-step parameters* and *full-model reference* obligations. Those
obligations **did not exist** — 01-01 authored 9, covering none of these four. Transcribing
tolerances onto absent obligations is impossible, and the alternative (leaving them out) means
`pooling_normalize.json`, `loss_pair.json`, `optimizer_step.json` and `full_model_reference.json`
would be frozen, manifest-covered, tolerance-bearing fixtures with **nothing in the contract
requiring Rust to match them** — the gate-as-theater failure this contract exists to prevent.

So four obligations were added (Rule 2, missing critical functionality), each carrying its frozen
tolerance: `OBLIG-ENC-03-POOLED-EMBEDDING-PARITY`, `OBLIG-ENC-06-LOSS-FORWARD-PARITY`,
`OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY`, `OBLIG-ENC-01-FULL-MODEL-REFERENCE-PARITY`. Post-step
parity is deliberately split from the gradient gate so the two thresholds (parity epsilon vs
`zero_grad_floor`) are not conflated in one `tolerance` slot. The contract is 13 obligations, and
the commit still touches exactly one file.

**6. Slice vocabulary is 97 ids, not the estimated ~256**

The truncation case is built by repeating one sentence 40 times, so it exceeds the 256-token
bound while adding almost no new vocabulary. Fewer retained ids means a smaller embedding table
and a smaller committed APR at no loss of coverage — every corpus token plus all five special
tokens is present, and `[PAD]`=0 maps to slice row 0 so `pad_token_id` stays 0.

**7. Two files not in the plan's `files_modified`**

`scripts/setfit_fixtures/jsonfmt.py` (the D-11 readable-JSON writer; `json.dump(indent=2)` puts
each of ~110k floats on its own line, which is indented but unreviewable) and
`scripts/setfit_fixtures/corpus.py` (the shared corpus of record, see above). Also
`scripts/setfit_fixtures/.gitignore` and `build/.gitignore` so `__pycache__` and the intermediate
slice safetensors are not left untracked or accidentally committed.

## Fixture Size Note

`gradients.json` (1.8 MB) and `optimizer_step.json` (1.6 MB) dominate the 4.6 MB corpus because
the plan requires *all* slice parameters and *all* post-step values. Floats are written with
`%.9g`, the exact round-trip width for binary32, so this is lossless and roughly half the size of
naive float64 repr. Compressed footprint is ~2.0 MB. Not a problem today, but worth knowing before
more parameter-complete fixtures are added to a published crate.

## Known Stubs

None. Every fixture is generated from the real pinned model; no placeholder, mock, or
hardcoded-empty value exists in the corpus or the scripts.

## Threat Flags

None new. The plan's register is fully discharged: T-1-06 (pinned immutable revision + fail-closed
per-file digests re-verified at **both** entry points), T-1-07 (SHA-256 manifest + versioned
tolerances + per-family floors that provably prevented a measured-zero tolerance), T-1-20
(`analytically_zero` generated from measured gradients with a versioned floor), T-1-27 (`texts` +
`case_id` on every model-driven fixture), T-1-SC (exact pins, hashed lockfile, never in CI),
T-1-25 (exclude root-anchored and packaging asserted by mutation).

One observation rather than a flag: `fetch_full_weights.py` writes ~87 MB into
`$APRENDER_MINILM_DIR` (default `~/.cache/aprender/...`) outside the repo, by design (D-10), and
verifies every byte against the recorded digests before converting.

## For the Orchestrator

- `STATE.md` and `ROADMAP.md` were **not** touched (worktree mode).
- `REQUIREMENTS.md` was **not** touched. This plan's frontmatter lists ENC-01..ENC-06, but it
  ships the *reference plane* those requirements are gated against — no Rust encoder, tokenizer,
  gradient path or loss exists yet. Marking any ENC requirement complete here would be labelling
  by intent rather than evidence (CLAUDE.md rule 2). Plans 01-05..01-09 close them.
- New deferred items appended as **D7-D9** (D7 is the most actionable — it is a direct trap for
  01-08 Task 1). Known items D1/D2/D5 were used as documented and not re-logged.
- The tolerance table is frozen. If a later wave finds a Rust discrepancy, the correct response is
  to fix Rust or to raise a contract edit that `pv diff` flags — not to widen a number here.

## Self-Check: PASSED

All 26 created files verified present on disk; the four largest verified **tracked by git** rather
than merely existing. All 3 commits verified as commit objects and ancestors of HEAD. Working tree
clean apart from this SUMMARY and `deferred-items.md`.

(The first self-check run reported all three commits MISSING — the check itself was faulty,
grepping reformatted `git log --oneline` output instead of asking git. Corrected to
`git cat-file -t` + `git merge-base --is-ancestor`. Recording it because it is the same
measure-the-wrong-thing class as deviation 1.)
