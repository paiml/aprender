# PMAT-4201 — PVL-001 EV-7b receipt: the comparator

Branch `PMAT-4201-pvl-7b-comparator` @ `85068313e`, stacked on EV-6b (`feat/4199-pvl-6b-leanchecker` @ `4cb4dffd2`). Author: claude-opus-5-5 (aprender-98). Cop: aprender-cf.

## What ships
- `crates/aprender-contracts-staging/lean/scripts/Comparator.lean`: it elaborates each `Challenge/*.lean` against the built tree and prints NDJSON `{name, challenge_type_hash, solution_type_hash, axioms}`. The hash is pure-Lean SHA-256 over `canon`: binder names are dropped and binder info is kept. `--self-test` checks the FIPS 180-4 vectors.
- `discharge::comparator` judges the rows. Each of these rejects with rc 1:
  - MISMATCH
  - MISSING-ROOT
  - SORRY
  - DUPLICATE
  - MALFORMED: no challenge hash, or a solution with no axioms list
  - a comparator exit that is not 0

  Each of these declines with rc 2 and is never a pass:
  - no Challenge file
  - no script
  - lake could not be spawned
  - zero rows
- `pv discharge check --comparator` (conflicts with `--no-lake`) runs after elaboration and before leanchecker. `Report.challenges` carries the `Closure {closed,total}` for EV-8a.
- CI: `ci/explicit-test-commands.d/480-aprender-contracts-cli-pvl-comparator.cmd`. There is no ci.yml edit (cop ruling).

## Evidence (re-run by the author on lambda under the host rule: nice/ionice, CARGO_BUILD_JOBS=8, --test-threads 8)
| Check | Result |
|---|---|
| `aprender-contracts --lib discharge::comparator` | 15 passed |
| `aprender-contracts-cli --lib commands::discharge` | 14 passed |
| `aprender-contracts-cli --test pvl_comparator` | 10 passed |
| clippy `-D warnings`, both crates, `--lib --tests` | rc 0 |
| `cargo fmt --all -- --check` · `check_tree_reader_tests.sh` | rc 0 · rc 0 |
| Mutants in `comparator.rs`: mismatch, sorry, duplicate, missing-root, zero-row decline, namespace filter, sorry-substring, no-axioms | 8/8 killed. Two first attempts were compile errors and were re-run as compiling mutants |
| Mutants in the CLI `compare`: no-challenge, no-script, rc, parse-reject, gate-cmp, gate-open, closure | 7/7 killed |
| Real Lean run (v4.29.0-rc4), fixture tree | weakened hypothesis DIFFERS · renamed binders MATCH · `:= sorry` gives `sorryAx` · renamed solution gives MISSING-ROOT · explicit→implicit binder DIFFERS · FIPS self-test 3/3 |
| Probe A-7b.1 (reads ci.yml's `--run` of the .d dir plus that script's `--list`) | GREEN 0 → .d entry removed 1 → restored 0 → ci.yml points elsewhere 1 → restored 0 |

## Quorum
- Round 1 (`4cb4dffd2..71f5077e2`, diff sha256 `5cdcef7d…3950c2`):
  - sonnet-5 PASS with 3 minors
  - haiku-4-5 PASS
  - agy gemini-3.1-pro-high returned 429 (RESOURCE_EXHAUSTED, cascade 5189a39a…) and gave no verdict
  - Minor #1 (`axioms: null` was treated as closed) is fixed in 85068313e, fail-closed with a killed mutant. Minor #3 (the roadmap listed proved-is-derived) is fixed. Minor #2 (no timeout on the comparator's lake call) is not fixed: `elaborate()` has the same shape, and a hang cannot produce a wrong verdict.
- Round 2, final diff (`4cb4dffd2..85068313e`, sha256 `96ee1243…37ffa68`): **3/3 PASS, `degraded: same-family`** (agy 429). The author id (claude-opus-5-5) did not review.
  - sonnet-1 (claude-sonnet-5): PASS. Minors:
    - `canon` interpolates Names without escaping, a theoretical collision not reachable through EV-7a's generator.
    - Universe-param names are not normalized. That can only cause a false reject, the safe direction.
  - sonnet-2 (claude-sonnet-5, in the agy seat): PASS. Minors:
    - `--self-test` is not run by any gate yet.
    - `rowsOf` drops `isInternal` names silently, which would understate the total.
  - haiku (claude-haiku-4-5): PASS, no findings.
- Minors carried to EV-8a / EV-9, not fixed here:
  - Run `Comparator.lean --self-test` in the lambda accept step.
  - Cross-check the row count against the Challenge roots EV-7a wrote, so a dropped row can't hide.

## Scope decisions (cop rulings, 2026-09-24)
- Ruling (A) plus option (1): `proved-is-derived` is born armed in EV-8a (#4202). EV-8a tracks `discharge-summary.json`, adds the gate to `armed_gates` and sets `underived_proved_claims=95` shrink-only, all in ONE diff stacked on EV-11. The baseline must reach 0, or the claims are downgraded (PVL-001 A-7b.2).
- EV-6b's probe had the same literal-ci.yml defect. aprender-dd fixed it on #4199 @ ff44809b4.

## Not measured
- The full comparator run over EV-7a's 39 Challenge files. It needs EV-7a's tree and a Lean run, and on lambda that waits on the host rule and a cop OK.
- EV-7a is pushed (feat/4200-pvl-7a-challenge @ 2be44b4b6) with the `_root_.PvlChallenge.<fqn>` format. The constant name the comparator keys on is unchanged. No rebase is needed to compile.
