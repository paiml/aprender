---
phase: 02-deterministic-pair-and-data-protocol
plan: 04
subsystem: contrastive-data
tags: [fixtures, setfit-reference, sha256-manifest, environment-attestation, tdd, d15, d22, d23, data-04, data-05]
requires:
  - "02-02 (the crate, tests/ tree, sha2+serde as normal deps, contrastive-pair-protocol-v1.yaml's closed forms and OBLIG-CPP-DEVIATION-DECLARED wording)"
  - "Phase 1 D-12 hash-locked uv venv (setfit==1.1.3) and D-13 fixture-manifest pattern"
provides:
  - "crates/aprender-contrastive-data/tests/setfit_reference/ — 6 setfit_measured_*.json + 6 aprender_contracted_*.json + manifest.sha256"
  - "tests/common/mod.rs — MeasuredFixture / ContractedFixture / DeviationClause (deny_unknown_fields), fixture_dir(), manifest_drift(), fixture_files(), manifest_names(), load_measured(), load_contracted()"
  - "tests/reference_fixtures.rs — the canonical, working-directory-independent integrity gate (FALSIFY-CPP-016 / -017)"
  - "generate_fixtures.py --pairs [--rebaseline] — a second, independent generator mode needing no torch model and no network"
affects:
  - "02-07 (tests/pair_counts.rs does `mod common;` and asserts against these rows instead of hand-typed constants)"
  - "02-08 (the K=N adversarial capacity gate reads aprender_contracted_singletons_32.json; the negatives-only row is now data)"
  - "any future re-baseline — it is a reviewable manifest diff produced by the generator's temp-dir compare, never a quiet overwrite"
tech-stack:
  added: []
  patterns:
    - "two fixture FAMILIES separated by filename prefix, so a doc-derived number cannot masquerade as a measured one"
    - "three-way agreement or FATAL: measurement == closed form read out of the reference source == the literal stated by plan/contract"
    - "generate into a temp dir, diff against the committed tree, replace only under an explicit --rebaseline"
    - "environment attestation is a triple (package version + lockfile digest + resolver version), because a version string alone does not identify an environment"
    - "manifest paths relative to the manifest FILE's directory, so the verifier's resolution rule is well-defined and cwd-independent"
    - "vacuity guards: pin the expected population before asserting a relation over it"
key-files:
  created:
    - crates/aprender-contrastive-data/tests/setfit_reference/manifest.sha256
    - crates/aprender-contrastive-data/tests/setfit_reference/setfit_measured_{8_4_8,4_1,8_8_8,64_64_64,8_4_8_maxpairs100,singletons_32}.json
    - crates/aprender-contrastive-data/tests/setfit_reference/aprender_contracted_{8_4_8,4_1,8_8_8,64_64_64,8_4_8_maxpairs100,singletons_32}.json
    - crates/aprender-contrastive-data/tests/common/mod.rs
    - crates/aprender-contrastive-data/tests/reference_fixtures.rs
  modified:
    - scripts/setfit_fixtures/generate_fixtures.py
decisions:
  - "Fixtures are keyed by fixture_id, not by the raw layout vector: 8_4_8 and 8_4_8_maxpairs100 share [8,4,8], so a layout-keyed map would silently drop one and shrink the evidence base"
  - "self_pair_count under a max_pairs cap is RNG-dependent and says so in rng_dependent_fields, rather than being recorded as if it were layout-derived"
  - "The --pairs mode is a separate entry point writing to a separate output tree, so a pair-fixture run structurally cannot re-baseline a Phase 1 fixture"
  - "Dry run exits 1 on drift and 0 when identical, so the generator doubles as a drift check"
metrics:
  duration: ~50m
  tasks: 2
  files: 15
  completed: 2026-08-09
requirements: [DATA-04, DATA-05]
---

# Phase 2 Plan 04: SetFit Pair-Count Reference Fixtures Summary

Two fixture families now sit side by side under integrity protection: what the pinned
`setfit==1.1.3` sampler **does**, and what Aprender's contract **says**. They disagree on
purpose, and the disagreement is the artifact — plans 02-07 and 02-08 measure a Rust
sampler against these rows rather than against SetFit's documentation, which the pinned
implementation contradicts.

Three commits on `gsd/phase-2-contract-gate` (no branches, no PRs, per 02-01's policy):

| # | Task | Commit | What |
|---|------|--------|------|
| 1 | Generator + both fixture families | `e7c8efc27` | 14 files, +1033 |
| 2 RED | 7 failing tests + stub models | `c39b41e5c` | 2 files, 0 passed / 7 failed |
| 2 GREEN | Verifier + loaders | `0c4526d5f` | 2 files, 7 passed |

## The fixtures are derived independently of the Rust implementation

This is the property the whole plan exists to protect, so it is worth stating exactly how
it was obtained rather than asserting it.

**No Aprender Rust code was executed to produce any number in these files.** It could not
have been: `pairs.rs` is still an empty stub, and plan 02-07 builds the sampler these
fixtures will judge. Each family has its own derivation path:

- **`setfit_measured_*`** — produced by executing `setfit.sampler.ContrastiveDataset` in
  the Phase 1 hash-locked venv, and then cross-checked against a closed form derived by
  **reading `setfit/sampler.py`**: enumeration is `np.triu_indices(n, 0)`, so
  `stored_pos = Σ_k [C(n_k,2) + n_k]` (the `+n_k` being the included diagonal),
  `stored_neg = Σ_{j<k} n_j·n_k`, each capped at `max_pairs // 2`, and
  `total = 2·max(stored_pos, stored_neg)` under oversampling.
- **`aprender_contracted_*`** — computed from the closed forms in
  `contracts/contrastive-pair-protocol-v1.yaml` (`positive_capacity`,
  `negative_capacity`, `default_epoch_budget`, `budget_resolution`,
  `pair_stream_degenerate_policy`).

**Every number must agree three ways or the generator aborts:** the measurement, the
closed form read out of the reference source, and — where plan 02-04 or the contract
states one — a hardcoded literal. Measurement alone would let a broken venv silently
become the new baseline. The literal alone would let a wrong closed form through. The two
`sys.exit` paths print all three numbers and say explicitly *"do NOT re-baseline; fix the
environment"* and *"surface the discrepancy; do NOT edit the fixture to match code."*

Every contracted literal the plan named was reproduced without adjustment. Nothing had to
be reconciled, so nothing had to be surfaced.

**The counts are RNG-independent by construction.** `shuffle_combinations` permutes with a
hardcoded `np.random.RandomState(seed=42)`, so the trainer seed never reaches pair
identity — and every recorded number is a *cardinality of the triangle*, not a function of
the order it is walked in. One exception was found and is recorded rather than hidden:
under `max_pairs=100`, **which** 50 positives survive does depend on the permutation, so
`self_pair_count` there (13) is listed in that fixture's `rng_dependent_fields`. Every
other fixture's list is empty and the test asserts that correspondence both ways.

## The two families, in numbers

| Layout | Measured (pinned setfit 1.1.3) | Contracted (Aprender) | Relationship |
|---|---|---|---|
| `[8,4,8]` | stored 82 pos (62 + **20 self-pairs**) / 128 neg, epoch **256** | pos 62 / neg 128, budget **256** | totals agree, composition does not |
| `[4,1]` | stored 11 pos (incl. **5 self-pairs**) / 4 neg, epoch **22** | pos **6** / neg 4, budget **12** | **DIVERGE** — the Pitfall 2 case |
| `[8,8,8]` | epoch **384** | budget **384** | agree (D-14 worked value) |
| `[64,64,64]` | epoch **24,576** | budget **24,576** | agree (D-14 worked value) |
| `[8,4,8]` + `max_pairs=100` | stored exactly **50 pos + 50 neg** | explicit budget 100 → 50/50 | cap is per LIST, not a total |
| `[1]×32` (K = N) | stored 32 pos (**all self-pairs**) / 496 neg, epoch **992** | pos **0** / neg **496**, budget **992**, `degenerate_case: negatives_only` | totals coincide, **every pair differs** |

The last row is the one worth pausing on. The two families agree on 992 and share not a
single pair: the reference emits 496 self-pair "positives", Aprender emits 992 negatives.
The fixture says so in its `divergence_note` — *"Agreement on a total is not agreement."*
That row is also what plan 02-08's K≈N capacity gate reads instead of a hand-typed
constant; a three-class fixture set could never expose an O(K²) sampler.

Every fixture carries the reproducibility **triple**, not just a version:
`setfit_version: 1.1.3`, `uv_lock_sha256: c2b36114…5c75e`,
`uv_version: uv 0.9.5 (d5f39331a 2025-10-21)`. The same `setfit 1.1.3` resolves
differently under a different lockfile or resolver, so a version string alone does not
identify the environment that produced a number.

## The temp-dir-then-diff path, exercised

Three runs, in order:

1. **Dry run, nothing committed** (`rc=1`): staged 12 fixtures + manifest into
   `/tmp/apr-pair-fixtures-7jpjzob3`, self-verified with `shasum -c` there, then printed
   the diff — `+ ADDED` for all 13 names — and refused to write:
   *"DRY RUN — nothing was written. Re-run with --rebaseline…"*
2. **`--rebaseline`** (`rc=0`): copied the staged tree into place and re-ran `shasum -c`
   **in the committed directory**, not just the staging one.
3. **Dry run again** (`rc=0`): `IDENTICAL — 12 fixtures + manifest already committed`.

Run 3 is the useful one: the generator is idempotent, so a future diff means something
actually changed. The dry run's exit status (1 on drift, 0 on identical) makes it usable
as a drift check as well as a re-baseline tool.

## Phase 1 fixtures: byte-identical, proven by regeneration

The plan's outcome condition is that the Phase 1 tree survives untouched. Two levels of
evidence, because the structural argument alone is the kind of claim that is usually true
and occasionally not:

- **Structural:** `--pairs` writes only into
  `crates/aprender-contrastive-data/tests/setfit_reference/`. It cannot reach the Phase 1
  tree even by accident.
- **Measured:** the **full** Phase 1 pipeline was re-run
  (`uv run python generate_fixtures.py`, rc=0, all 16 fixtures + manifest rewritten,
  upstream digests re-verified at `1110a243fdf4`), and afterwards
  `git status --porcelain crates/aprender-core/tests/fixtures/` was **empty**.

The one shared helper that changed is `write_manifest()`, which gained a
`directory: Path = FIXTURE_DIR` parameter so both trees can use one writer. The Phase 1
call site is unchanged and its output is byte-identical; only a stdout line now prints
`setfit/manifest.sha256 covers 16 files` instead of `manifest.sha256 covers 16 files`.

**Control for the emptiness check** (the rtk hook inverts porcelain-emptiness assertions,
so an empty result needs a witness): the same command run against
`scripts/setfit_fixtures/` in the same invocation printed
` M scripts/setfit_fixtures/generate_fixtures.py`. The check reports drift when drift
exists; its silence on the Phase 1 tree is therefore meaningful.

## Drift detection, observed rather than assumed

`"total": 384` in `setfit_measured_8_8_8.json` was changed to `385` — one byte — and the
suite was re-run. It failed `rc=101` in **two independent ways**:

```
fixture integrity failed:
  setfit_measured_8_8_8.json: digest drift — manifest 353b269ffcbb…3449452, on disk ecc8be4f7a61…c3ea4
```

and, separately, `every_measured_fixture_deserializes_and_is_internally_consistent`
panicked because `total != len_pos + len_neg` no longer held. The digest check names the
file and both hashes; the consistency check would have caught a *coordinated* edit that
also updated the manifest. Reverted; porcelain clean; suite green again.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] Two of the seven tests were green against empty stub loaders**

- **Found during:** Task 2, the RED run.
- **Issue:** the first RED run reported **2 passed, 5 failed**. The orphan check compared
  an empty on-disk set against an empty manifest set, and the loader test compared the key
  sets of two empty maps and then iterated nothing. Both are green whenever the data is
  absent — precisely the failure mode the manifest exists to prevent, reproduced inside
  the manifest's own test.
- **Fix:** each now pins the expected population *before* asserting a relation over it
  (`fixture_files().len() == 12`, `measured.len() == 6`), with the observed vacuity
  recorded in a comment at both sites so the guards are not removed as noise later.
- **Verified:** the corrected RED run is **0 passed, 7 failed**.
- **Commit:** `c39b41e5c`

**2. [Rule 2 — Missing critical functionality] Loaders key by `fixture_id`, not by layout**

- **Found during:** Task 2, designing `load_measured` / `load_contracted`.
- **Issue:** the plan says the loaders "return every fixture keyed by layout". Two of the
  six cases share the class layout `[8,4,8]` — the plain one and the `max_pairs=100`
  one — so a layout-keyed map silently overwrites one with the other. The loader would
  return five rows where six exist, and nothing would be red.
- **Fix:** keyed by `fixture_id` (the layout id plus a case qualifier: `8_4_8`,
  `8_4_8_maxpairs100`), with the reason in the `load_family` doc comment, and a
  `two fixtures claim fixture_id` assertion so a genuine key collision fails loudly.
  Every test still resolves a fixture by naming its layout case; only the key type changed.
- **Commit:** `0c4526d5f`

**3. [Rule 2 — Missing critical functionality] `rng_dependent_fields`**

- **Found during:** Task 1, measuring the `max_pairs=100` case.
- **Issue:** the plan asks each measured fixture to record `self_pair_count`, and treats
  every recorded count as RNG-independent. Under a cap that is false — 13 of the surviving
  50 positives are self-pairs, and *which* 50 survive is decided by the reference's
  hardcoded `RandomState(42)` permutation. Recording 13 with no qualifier would present a
  permutation artifact as a layout fact.
- **Fix:** every measured fixture carries `rng_dependent_fields`; it is empty for the five
  uncapped layouts and `["self_pair_count"]` for the capped one. The generator's closed
  form returns `None` for that field rather than a wrong prediction, and
  `assert_measured_counts` asserts the correspondence in both directions — uncapped
  fixtures must have an empty list *and* `self_pair_count == n_examples`.

### Scope note

No `README.md` was added to the fixture directory. The manifest covers **every** file in
that tree except itself, so a hand-edited README would either sit outside the integrity
guarantee or force a re-baseline on every documentation edit. The `cd`-plus-`shasum`
convenience and the resolution rule are documented where they are used instead: in the
generator's section banner and in `tests/common/mod.rs`'s module doc.

## Verification

| Gate | rc | Result |
|---|---|---|
| `shasum -a 256 -c manifest.sha256` (run **from** the fixture dir) | 0 | 12 files verified |
| measured fixture count / contracted fixture count | — | 6 / 6 |
| `cargo test -p aprender-contrastive-data --test reference_fixtures` (repo root) | 0 | **7 passed** |
| `cargo test -p aprender-contrastive-data --test reference_fixtures` (crate dir) | 0 | **7 passed** — identical, which is the cwd-independence proof |
| one-byte fixture corruption | 101 | 2 tests fail, digest check names the file and both hashes; reverted |
| `cargo test -p aprender-contrastive-data` | 0 | 79 passed (67 lib + 7 integration + 5 doc) |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `cargo fmt -p aprender-contrastive-data --check` | 0 | clean |
| `make contrastive-data-boundary` | 0 | `contrastive-data-boundary: PASSED` — the tests/ tree is outside the src/ scan, as intended |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| cross-crate baseline (`--lib -- --skip gpu::`) | 0 | **14,102 passed**, 27 ignored, 0 failed |
| full Phase 1 regeneration | 0 | 16 fixtures rewritten; `git status --porcelain crates/aprender-core/tests/fixtures/` **empty** |
| struct definitions outside `tests/common/mod.rs` | — | **none** (all three at `tests/common/mod.rs:48/60/91`) |
| `.snap.new` files after the aprender-train run | — | restored; no deletion staged |

The cross-crate baseline is **unchanged at 14,102**, and that is the correct outcome: this
plan added integration tests, which `--lib` does not select. The crate-level count moved
72 → 79 (+7).

## Measurement notes (CLAUDE.md Verification Discipline)

- **Statuses captured directly, never through a pipe.** Several Bash invocations here
  reported a non-zero *outer* status purely because a trailing `grep` matched nothing; the
  `rc` values quoted above are the ones captured with `cmd > log 2>&1; rc=$?`.
- **`rtk proxy` for every porcelain and grep result.** The hook prints a literal `ok` on a
  clean porcelain path, which inverts emptiness assertions, and abridges other output.
- **The porcelain emptiness claim carries a control** (see the Phase 1 section above)
  rather than being read from a silent command.
- **The pinned reference was executed, not cited.** `setfit.__version__` is checked
  against `1.1.3` inside the generator before any number is recorded.

## Notes for the next plans

- **02-07** — add `mod common;` to your new `tests/pair_counts.rs` and read the rows from
  `common::load_contracted()`. The `[4,1]` and `[1]×32` rows are the ones that separate a
  correct sampler from a plausible one; asserting only the balanced layouts would pass
  against a sampler that includes self-pairs.
- **02-08** — `aprender_contracted_singletons_32.json` is your K≈N reference row
  (`positive_capacity: 0`, `negative_capacity: 496`, `resolved_neg_count: 992`).
- **Re-baselining** is `uv run python generate_fixtures.py --pairs --rebaseline` from
  `scripts/setfit_fixtures/`, and it must be a reviewed diff. If the generator FATALs on a
  measurement mismatch, the environment is wrong — do not reach for `--rebaseline`.
- **Adding a fixture field** means adding it to `MeasuredFixture` / `ContractedFixture`
  too: `deny_unknown_fields` turns an unmirrored field into a red test rather than a
  silently ignored one.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/tests/setfit_reference/manifest.sha256` | FOUND |
| 6 × `setfit_measured_*.json` | FOUND |
| 6 × `aprender_contracted_*.json` | FOUND |
| `crates/aprender-contrastive-data/tests/common/mod.rs` | FOUND (3 structs, 4 × `deny_unknown_fields`) |
| `crates/aprender-contrastive-data/tests/reference_fixtures.rs` | FOUND (`mod common;` at line 12) |
| `scripts/setfit_fixtures/generate_fixtures.py` | FOUND, contains `setfit_reference` |
| commit `e7c8efc27` | FOUND |
| commit `c39b41e5c` | FOUND |
| commit `0c4526d5f` | FOUND |
