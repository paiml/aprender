# Implementation receipt: GH-4117, every caller in the publish path runs the 0.70 release gate

**Branch.** `fix/4117-release-scope-callers` is stacked on `origin/feat/4033-release-process-0.70` @ 7684eb1d9 (#4045). `--scope release`, `--nightly` and `ladder.release_gate` exist only there (0 `--nightly` hits on chore/0.69.1-merge-back), so the cop ruled the stack.

**Commits.**
- 215da7f79: nightly relative paths, a separate commit for #4040's owner (aprender-6c) to review.
- 3519d2697: the callers.

## Ticket intent (#4117) → the change

- **"The models step and R7 run `--scope release --nightly <root> --crux <smoke> --cut-commit`"**
  - `scripts/release/models_t1.sh` (autopilot's `models` step) does it from `release_gate.from` on.
  - `check_publish_preflight.sh` `rule_r7_release` does it at T-4 over the models step's outputs. autopilot's `run_preflight` hands those over as `PUBLISH_PREFLIGHT_NIGHTLY_ROOT`, `_CRUX_DIR` and `_CUT_COMMIT`.
  - R7 can't read the smoke from the tree: the smoke of the release commit can't be committed into that commit.
  - With those variables unset, R7 is RED by name and never falls back to the full ladder.
- **"The smoke on the release binary is produced by a step, not by hand"**
  - `models_t1.sh` builds and proves the binary per host (unchanged), then runs `crux_sweep_shards.sh` on it under `choom -n 1000` with the newest committed certification.
  - gx10's receipt comes back the same way its ladder receipt did.
- **"The candidate watch runs the same gate as a REAL row"**
  - `candidate_watch.sh` adds the row `models:release-scope`, real by definition like `g-ont:complete`.
  - The smoke is measured once per candidate sha (`models_t1.sh` into `<state>/models-<sha>`), and every later run re-judges it against a freshly gathered nightly. Admission is time-bound (≤ 24 h), so a cached PASS never outlives its night.
- **Which gate applies** is ONE rule for all four callers: `crux_smoke_scope.py applies <contract> <version>`, which exits 0 for the release gate, 1 for the full ladder (before `from`), and 2 when undecidable (refuse).

## Widened by cop ruling (2026-09-24)

- **A 4th caller:** `prepare_bump.sh --ship` refused any bump without green full-ladder receipts, which kept Phase 2 alive.
  - From `from` on, that requirement is **replaced**, not dropped: `--ship` now requires an admissible nightly for the candidate on every required host, via the judge's own `nightly_admission.py`.
- **Nightly relative paths (215da7f79):** the gate judges two hosts on one host.
  - `certify_nightly.sh` recorded host-absolute receipt paths, and admission read them as-is. gx10's night was therefore admissible on lambda only at the identical absolute path.
  - Verdicts now record paths relative to themselves, and `nightly_admission.resolve()` resolves them against the loaded verdict's directory.
  - A `..` escape is refused, and absolute paths from nights already on disk are still accepted.
- **Shared gather:** `scripts/release/gather_nightly.sh` gathers both hosts' nights. It copies each night's verdict and the receipts it names, never the night's checkout, and is used by `models_t1` and `prepare_bump`.

## Measured

Each "mutant … killed" below means the named row went RED on a copy with that one rule removed.

| table | result |
|---|---|
| `check_publish_preflight.sh --selftest` | 60/60 rows + 3 mutants killed (falls-back-to-ladder, unwired-ok, undecidable-is-ladder) = 63/63 |
| `check_release_models_t1.sh` | 38/38 rows. New: release-green, release-no-nightly, release-red-smoke, release-stale-binary. Mutants killed: models-scope-dropped, models-no-gather, models-never-smoke, models-smoke-no-choom |
| `check_publish_reads_watch.sh` | PASS. New row fresh-hands-r7-the-models-outputs (the exact handoff incl. the case's release commit), mutant r7-handoff-dropped killed |
| `check_release_shift_left.sh` | PASS. New rows watch-release-gate-red-andons, -green, -smoke-once-per-sha-then-rejudged. Mutants release-gate-skipped and release-smoke-uncached killed |
| `check_release_bump_pr_body.sh` | 14/15. New row release-no-nightly-refuses, mutants drop-admission and never-release killed. The body rows now run on a release-gate fixture (cop ruling (a)). The one red row is `ladder-ignored` (see below) |
| `nightly_admission_cases.py --mutants` | 19 rows, 14/14 mutants (new: moved-root-admitted-by-relative-paths, relative-path-escaping-refused, relative-ladder-escaping-refused) |
| `check_certify_nightly.sh` | PASS. New row verdict-relative, mutant absolute-paths killed |
| `check_model_ladder.sh --self-test` | 155 cases, 0 bad |
| bashrs `--no-ignore --level error` on the 10 changed/new scripts | 0 gating findings (base: 0). The 5 I introduced were fixed with explicit guards, plus reasoned per-line disables for validated caller paths |
| `cargo fmt --all -- --check` | rc 0 (no Rust changed) |

## Found on the base (feat/4033 @ 7684eb1d9), measured on a clean worktree of it

- **`check_release_models_t1.sh`** exited rc 2 ("the table judged nothing"). Its `r7-no-fail-lines` mutant anchor occurred twice once `rule_r7_scope` copied the line.
  - Fixed here: the mutant now anchors on the rule_r7-only pair of lines.
- **`check_release_bump_pr_body.sh`** was RED on 5/12 rows. The fixture's full-ladder receipts drifted behind the judge: no 40-hex `apr_sha`, no CRUX receipts or certification, no cells.
  - The body rows now run on a release-gate fixture (every 0.70+ bump takes that path).
  - The pre-`from` fixture repair is **#4128**, kept out of this row by cop ruling. Until it lands, `ladder-ignored` stays red, and its `drop-ignored` mutant counts as "killed" only because that row already fails, so that kill proves nothing.

## Caught by their own rows while building

- **ssh stub:** it forwarded `APR_NIGHTLY_ROOT` to "gx10" (real ssh forwards no such variable), so gx10 read lambda's root. `release-green` failed and exposed it.
- **Vacuous row:** `release-no-nightly` passed only while gathering was broken. Its `FX_NO_NIGHTLY_HOST` was exported after the fixture was built, so it's now set when the fixture is built.
- **Mutants that proved nothing:**
  - `release-gate-skipped` first survived. Forcing the `if` false let the `elif` still emit a FAIL row with the same id, so the mutant is now "the rule answers not-applies".
  - `release-gate-bookkeeping` tested nothing: an unclassified watch gate is already real, so I dropped it rather than keep a vacuous mutant.

## Limits

- **No live run yet: #4132.** The real smoke and gather have not been run end to end on lambda + gx10 in this branch. The case tables stub `crux_sweep_shards.sh`, `ssh` and the judge's release scope. The real judge's release scope is tested in `check_model_ladder.sh --self-test`, and the real admission in `nightly_admission_cases.py`. The live dress rehearsal through all four callers is **#4132**. Its done_when: a SCOPED verdict through all four callers on real hardware before the 0.70 freeze, with both negative controls RED.
- **Legacy nights:** nights written before 215da7f79 record absolute paths. Measured on another host, they're refused as "receipts are gone". The nightly timer (paiml/infra#959) isn't applied yet, so none exist in production.
