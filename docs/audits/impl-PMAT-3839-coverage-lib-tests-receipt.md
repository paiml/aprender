# PMAT-3839 — the 0.69.1 coverage row

**Worker:** aprender worker two. **Cop:** aprender-3e.
**Branch:** `cov/0691-lib-tests` off `f3fba7c1f`. Three commits, one per crate.

## Verdict

**+577 covered lines** against a gap of ~1,705. Two of the three crates moved;
the third moved nothing and that is stated rather than implied.

But the number is not the finding. **The dominant cause of the 0.69.1 coverage
deficit is that the gate refuses to run its own tests**, and the gap report that
directs coverage work cannot see this.

## 1. The measurement excludes 1,207 tests by name substring

`Makefile:628` passes nineteen `--skip` patterns to the instrumented run.
libtest `--skip` is a **substring match on the full test path**, so it matches
module names, not just test names. Measured with `--list` against the gate's
exact skip list:

| crate | in the binary | survive the skips | removed |
|---|---|---|---|
| aprender-serve | 16044 | 15267 | **777** |
| aprender-train | 7646 | 7472 | 174 |
| aprender-compute | 3518 | 3388 | 130 |
| aprender-core | 14284 | 14158 | 126 |
| | | | **1207** |

Two mechanisms, both accidental:

* `--skip falsification` eats the **module** `prune::falsification_checklist`
  entire — all six of its tests. That is why `get_checklist` reads **0.0%
  coverage with 153 uncovered lines and a grade of A**. It is thoroughly tested
  code the measurement will not execute.
* `--skip gpu_` matches inside `wgpu_` (35 tests), and matches ~1,100 tests
  whose names merely *mention* GPU reporting — `compute_class_is_cpu_without_a_gpu_feature`,
  `force_cpu_preloads_no_accelerator_for_any_format` (#3794's, a pure CPU
  predicate test). None of these needs a GPU.

**17,119 uncovered lines — 15% of all uncovered lines in the measured tree —
sit in modules whose path matches a skip pattern.** That is ten times the gap to
the floor. Six files in the top 18 by uncovered lines are of this kind, and they
carry 271 tests between them:

```
829 uncov  0%  aprender-test-lib/src/presentar/falsification.rs      15 tests
796 uncov  0%  aprender-orchestrate/src/falsification/performance_waste.rs   48 tests
884 uncov  0%  aprender-test-lib/src/gpu_pixels/mod.rs               82 tests
791 uncov  0%  aprender-cgp/src/profilers/cuda.rs
771 uncov  0%  aprender-simulate/src/falsification/mod.rs            44 tests
691 uncov  0%  aprender-test-cli/src/load_testing.rs                 41 tests
```

**The consequence is worse than the number.** `pmat --coverage-gaps` and issue
#2307 rank functions by uncovered lines, and code whose tests are skipped is
indistinguishable from code with no tests. The gap report has therefore been
directing coverage work at well-tested modules. Anyone working that list
top-down does invisible no-op work — and would have, on the first target
assigned to me.

Not fixed here. The fix is skip patterns that match intent rather than
substrings (anchored names, `--skip-exact`, or renaming the offenders), and the
operator's ruling is minimal tests now with the measurement work in 0.70.0.

## 2. What was written

Targets were chosen from an lcov ranking with the skip-mirage filter applied,
excluding GPU code, AVX-512 code, and `aprender-qa-cli` (whose handlers call
`std::process::exit` 94 times — `dispatch` cannot be unit tested without
terminating the runner).

| crate | file | before | after | delta |
|---|---|---|---|---|
| aprender-contracts-cli | `commands/kaizen.rs` | 0/722 (0%) | 282/912 | **+282** |
| aprender-present-yaml | `src/executor.rs` | 0/850 (0%) | 295/1023 | **+295** |
| aprender-train | `finetune/training_plan.rs` | 527/843 | 527/843 | **+0** |

Both 0% files had **zero tests**, not skipped tests — verified before writing.

15 mutants across the three, **14 RED**. The one survivor is recorded as an
equivalent mutant, not a missed kill: removing `repo_grade`'s `sites == 0` guard
makes the quality term `0/0`, every `>=` against NaN is false, and the ladder
falls through to `"F"` regardless.

### The zero-delta result, stated plainly

The training_plan tests add **no covered lines**. Verified at line level: every
line they reach was already covered by the module's existing 141 tests, and the
hit counts merely rise (`execute_plan`'s entry 5 → 8, `from_str`'s body 59 →
84). The 196 uncovered lines #2307 attributes to `execute_plan` are its training
body, which needs model weights. The commit buys guards — a plan the auditor
BLOCKED can no longer reach the apply phase — and it buys no coverage. Kept
because the guard is worth having, reported because "7 new tests" would
otherwise imply movement.

## 3. A regression I introduced and caught before committing

Adding `proof_obligations` tripped the **sigma** gate:

```
formal_prose rose 1464 -> 1469: a `formal:` entry carrying no symbol Σ declares
was added. The baseline in contracts/lint-baseline.json is shrink-only
```

Five of my six new `formal:` entries were prose. (The sixth escaped because it
contained `len()`, and `len` is a declared Σ function.) Fixed by rewriting all
six in Σ-declared glyphs — `∀ P: fromStr(toJson(P)) ≡ P ∧ fromStr(toYaml(P)) ≡ P`
— **not** by touching the baseline, which is the move the shrink-only ratchet
exists to prevent. `formal_prose` is back to 1464.

## 4. Rule 7

| crate | contract | churn |
|---|---|---|
| aprender-contracts-cli | `pv-cli-surface-v1.yaml` extended | none — existing file |
| aprender-train | `classification-finetune-v1.yaml` extended | none — existing file |
| aprender-present-yaml | `present-yaml-expression-executor-v1.yaml` **new** | census 1830→1831 + README ×2 |

Nothing in `contracts/` covered aprender-present-yaml. A new contract file is a
**three-part change** — YAML, `contracts/census.json`, and README's two
`CONTRACT_COUNT` blocks — and all three are in the same commit so the branch is
self-consistent. `make readme-sync` rc=0; the `readme_contract` drift gate passes.

## 5. Checks

| check | result |
|---|---|
| `cargo test -p aprender-contracts-cli --lib` | 108 passed, 0 failed |
| `cargo test -p aprender-present-yaml --lib` | 217 passed, 0 failed |
| `cargo test -p aprender-train --lib` | 7639 passed, 0 failed, 14 ignored |
| `cargo test -p aprender-core --test readme_contract` | 15 passed |
| clippy `-D warnings`, all three crates | rc=0 |
| `cargo fmt --all -- --check` | rc=0 |
| `pv validate`, all three contracts | 0 errors, 0 warnings |
| `aprender-contracts --lib` | 1700 passed, **1 failed — `shapes`, pre-existing** |

The single failure is `lint_passes_on_real_contracts` on the `ladder-green`
shape (`rung qwen3-8b-q4km … missingGreenHost: lambda`) — the 0.69.1 lambda
receipt, not this branch. `sigma` is green again after §3.

## 6. Not claimed

* **The workspace coverage percentage from my full llvm-cov run is NOT quoted
  here and should not be used.** It disagrees with the nightly's 87.82% in the
  direction that would unblock the release, and it has four uncontrolled
  differences: it ran on the release branch rather than `main`, it skipped
  `lint_passes_on_real_contracts`, an earlier attempt SIGSEGV'd in the trueno
  test binary, and I checked out a different commit in that worktree 23 seconds
  before the report was written. Any one of those is enough to disqualify a
  release-deciding number. A clean re-run on a fixed checkout is owed.
* The per-file deltas above ARE trustworthy: each is a like-for-like per-crate
  llvm-cov run with the new tests present and then `git stash`ed away.
* The 17,119-line figure counts uncovered lines in modules whose path matches a
  skip pattern. It is not a claim that all 17,119 would be covered if the
  patterns were fixed — only the subset with tests would, and I verified test
  presence for six files, not for all of them.
