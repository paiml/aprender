---
phase: 04-apr-artifact-and-production-parity
plan: 22
subsystem: gate surface / shell + Makefile lint
tags: [bashrs, F-07, gate-honesty, false-positives, tier3, gap-closure, wave-12]
wave: 12

# Dependency graph
requires:
  - "bashrs 6.66.3 installed OUTSIDE this plan (verified, not installed by it — threat T-04-SC)"
  - "04-20's four new eval::setfit tests, landed at 4b78c6f46"
  - "04-21's setfit-serve-tests floor edit (Makefile :1920 / :2177), landed at 7ddb0ad98"
provides:
  - "`bashrs-lint-makefile` that reads its own status and was PROVEN to fail on an induced real defect"
  - "`bashrs-scoped-lint` — a baseline-non-increase gate over the Makefile + the three apr-pinning guards, wired into tier3, with a non-vacuity guard and two control-backed discriminators"
  - "the measured 12-cell exit-code semantics of `bashrs lint` AND `bashrs make lint`"
  - "two bashrs FALSE POSITIVES refuted by independent controls, discriminated rather than suppressed"
  - "D-04-22-A — the full inventory, the fix-shape triage, a named owner, and an 8-item NOT-RUN list"
  - "the `setfit-cli-eval-tests` row and floor corrected against a POST-04-20 measurement"
affects: [Makefile, tier3, phase-05-planning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "The exit code is not the verdict: when rc=2 means BOTH 'has an error' and 'file not found', a gate must parse the report and treat an unparseable report as FAILURE"
    - "Discriminate, don't suppress: a false positive is discounted only by a control that is re-run every invocation AND demonstrably fires on the real defect"
    - "A baseline entry without a named control is rejected by the gate, so the baseline cannot decay into a silent suppression list"
    - "Run the case table before believing the design — the first design was refuted by its own must-not-match row"

key-files:
  created:
    - .planning/phases/04-apr-artifact-and-production-parity/04-22-SUMMARY.md
  modified:
    - Makefile
    - .planning/phases/04-apr-artifact-and-production-parity/deferred-items.md

key-decisions:
  - "The first baseline design was REFUTED by its own case table and replaced: one new help line containing the word 'local' turned the gate red on correct prose, so both false positives are now discriminated by controls and every baseline is 0"
  - "SC2168 is NOT blanket-suppressed — an induced `@local probe=1` recipe proved it catches a real runtime defect, so only the `##`-help-comment position is discounted"
  - "SC1078 is discounted only per-file and only when that file's own `bash -n` exits 0 — the same control that fires rc=2 on a genuinely unterminated quote"
  - "`lint-scripts` deliberately left UNWIRED: 51 of 59 scripts are non-zero, and a permanently red blocking gate gets disabled along with the real gates beside it"
  - "The eval floor is 18 against a measured 20, matching the ~90% convention every other row in the gate table uses"

requirements-completed: []

# Metrics
duration: 71min
completed: 2026-08-16
---

# Phase 04 Plan 22: F-07 — the bashrs gate that could not fail Summary

**`bashrs-lint-makefile` ended in `|| echo`, so it exited 0 whatever bashrs said and had never been able to report the 34 findings it really produces; it now reads its own status, was proven RED on an induced real defect, and is joined by a scoped tier3 gate whose two known false positives are discriminated by controls rather than suppressed — a design its own case table forced, after the first version went red on the word "local" in a help string.**

## Base

Ran **SEQUENTIALLY on the main working tree**, branch `gsd/phase-2-contract-gate`, base
**`9ae41fcaa02fdc8cf2283bd1b5f23e0f1dd39102`**. Confirmed before acting; all three plan anchors
matched (`Makefile` 2432 lines, `:1109` the bashrs target, `:1918` the eval row, `:2327` `dev-setup`).
No `git reset`, `checkout <sha>`, `update-ref`, `stash` or `clean` was run.

## Performance

- **Duration:** 71 min
- **Tasks:** 3 of 3
- **Files modified:** 2

## Task Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 — measure every leg, refute two findings by control | `e8fe1e9f3` | `deferred-items.md` (+232, −0) |
| 2 — make the gate able to fail; wire one scoped gate | `23a5a3574` | `Makefile` (+373, −5) |
| 3 — triage the backlog, name an owner, list NOT-RUN | `eedb63102` | `deferred-items.md` (+181, −0) |

---

## The tool, proven before any number it printed was trusted

| probe | result |
|-------|--------|
| `whence -a bashrs` (zsh) | **exactly ONE path**: `/Users/guy/.cargo/bin/bashrs` |
| `command -v bashrs` | rc=0, same path |
| `bashrs --version` | rc=0, **`bashrs 6.66.3`** |

`type -aP bashrs` — the form CLAUDE.md rule 8 suggests — is unavailable here: the shell is zsh,
whose `type` has no `-P` (`bad option: -P`, rc=1). `whence -a` was used instead and the substitution
recorded rather than made silently. bashrs was **verified, not installed** by this plan (T-04-SC).

## The exit-code control table — TWELVE measured cells, and the help text is wrong

Established by control on throwaway inputs, one per severity state, for both subcommands. They are
different code paths and were measured separately. Every status captured with `cmd > log 2>&1; rc=$?`.

**`bashrs lint` (shell scripts):**

| input | findings | default rc | `--strict` rc |
|-------|----------|-----------|--------------|
| info-only | 0E 0W 3I | **0** | **0** |
| warning-present | 0E 1W 2I | **1** | **1** |
| error-present (a REAL unterminated quote) | 1E 0W 0I | **2** | **2** |

**`bashrs make lint` (Makefiles):**

| input | findings | default rc | `--strict` rc |
|-------|----------|-----------|--------------|
| no findings | 0E 0W 0I | **0** | **0** |
| warning-only | 0E 6W 0I | **1** | **1** |
| error-present (a REAL `local` in a recipe) | 1E 0W 0I | **2** | **2** |
| the real 2432-line `Makefile` | 1E 33W 0I | **2** | **2** |

Three conclusions, all load-bearing:

1. The mapping is **severity-tiered and identical across both subcommands**: 0 = nothing above info,
   1 = warnings, 2 = at least one error. It is not pass/fail.
2. **`--strict` is a NO-OP in all seven states.** Its help text says "fail on warnings", but warnings
   already fail without it. The gate does not use it, and the recipe records why.
3. **The info-only cell is not constructible for `make lint`** — no info-severity Makefile rule was
   observed to fire, on the real Makefile or any probe. Substituted with no-findings and warning-only
   rows and recorded as a substitution in the NOT-RUN list, not silently omitted.

**The measurement that decided the design — rc=2 is AMBIGUOUS:**

| control | rc |
|---------|----|
| `bashrs make lint Makefile` (1 error) | **2** |
| `bashrs make lint /nonexistent/Makefile` | **2** ← same code |
| `bashrs make lint` with bashrs off `PATH` | **127** |

A gate keyed on rc alone cannot tell "this file has an error" from "the gate was pointed at nothing".
Both gates therefore parse the report and treat an unparseable report as FAILURE. A second parsing
hazard was measured too: a fully clean file prints `✓ No issues found` with **no `Summary:` line**, so
a parser greping only for `Summary:` reads nothing and, if it defaults to 0, cannot tell clean from
unparsed. Both shapes are handled.

## The shell-semantic control for every error-severity finding on the four gated files

| file | errors | control | verdict |
|------|--------|---------|---------|
| `Makefile` | 1 (SC2168) | `make -n dev-setup` rc=**0**, expanded recipe has ZERO shell `local` (grep count 0) | **FALSE POSITIVE** |
| `scripts/check_apr_bin_pinned.sh` | 1 (SC1078) | `bash -n` rc=**0** | **FALSE POSITIVE** |
| `scripts/apr_bin.sh` | 0 | `bash -n` rc=**0** | n/a |
| `scripts/check_sourced_libs_option_neutral.sh` | 0 | `bash -n` rc=**0** | n/a |

**FP-2** flags columns 22-28 of `dev-setup: ## Set up local dev environment with sibling repo
overrides` — extracted as literally `local d`, the English words in a help comment.

**FP-1** flags the `'"'"'` single-quote-escaping idiom inside the `ABS_APR` regex, whose own lines
70-71 say it "has now been gotten wrong four times in this repo; if you change it, re-run the table
rather than reading it".

**The controls are not vacuous, and that was proven rather than asserted:** the same `bash -n` returns
rc=**2** on a probe with a genuinely unterminated quote (`unexpected EOF while looking for matching '"'`).
A control that never fires would prove nothing.

## The per-leg inventory

**Leg A — `bashrs make lint Makefile`: rc=2**, `1 error(s), 33 warning(s), 0 info(s)` — MAKE012 18,
MAKE010 11, MAKE003 2, MAKE018 1, MAKE001 1, SC2168 1. Identical to the planner's pre-04-21 tally:
**04-21's two-line edit produced a delta of ZERO**, recorded because "expected but absent" is a
measurement.

**Leg B — `bashrs lint` over each of `scripts/*.sh`, ONE FILE AT A TIME** (59 invocations, so no
file's failure masks another's): **59 scripts, 51 non-zero.** With the measured semantics that
decomposes usefully:

| rc | meaning | files |
|----|---------|-------|
| 0 | nothing above info | **8** |
| 1 | warnings, no errors | **29** |
| 2 | ≥1 error-severity | **22** |

Corpus totals, ANSI stripped before counting: **106 error / 754 warning / 1412 info**. All 59 per-file
exit codes are tabulated in D-04-22-A §3.

**Leg C — `bashrs score`** on the three guards: rc=0 each; B- 7.1, C- 5.6, B 7.9. Advisory.

**Leg D — `bashrs gate`: RUN, VACUOUS, and NOT recorded as a pass.** `bashrs gate --strict .`, the
form CLAUDE.md documents, **does not exist in 6.66.3** (rc=2, `unexpected argument '--strict'`). The
valid form passes at every tier having enabled nothing:

```
$ bashrs gate --tier 1              → rc=0
Executing Tier 1 Quality Gates...
Gates enabled:
----------------------------------------
----------------------------------------
✅ Tier 1 Gates Passed!
```

Tiers 2 and 3 are identical in shape. **A check reporting success having checked nothing — the exact
F-07 defect class, inside the tool this plan was sent to adopt.** Nothing is gated on it.

## The induced-defect run that proved `bashrs-lint-makefile` can fail

Appended to a scratch copy of the Makefile:

```make
04-22-induced-defect-probe:
	@local probe=1; echo "$$probe"
```

```
PROBE_A_rc=2
✗ 2591:22-28 [error] SC2168: 'local' is only valid in functions
✗ 2699:3-9   [error] SC2168: 'local' is only valid in functions
Summary: 2 error(s), 36 warning(s), 0 info(s)
FAIL: 2 error-severity finding(s), baseline is 1 (bashrs rc=2).
```

**And the induced defect is REAL, not merely flagged** — running the target:

```
/bin/bash: line 0: local: can only be used in a function
```

This is why SC2168 is not blanket-suppressed. (Note the probe target itself returned rc=0 despite
that error, because `local` fails and the following `echo` succeeds — the `|| echo` defect class in
miniature, met by accident while proving the point.) Reverted; the target returned to rc=0.

## The case table — RUN, and it REFUTED the first design

CLAUDE.md rule 7 requires a must-match / must-not-match table for the guard actually wired. Running
it killed the first design.

**The refutation.** v1 baselined the Makefile at 1 error-severity finding, carrying SC2168 as a
justified entry. Appending one ordinary help line —
`04-22-probe-help: ## Run the local smoke suite against a local endpoint` — raised a **second**
SC2168, took the count to 2 and turned the gate **RED on correct prose**. That is precisely the
liability the plan forbade building: a blocking tier3 gate that trips on the word "local" in help
text is a gate that gets disabled, and it takes the real gates beside it down with it.

**The fix: discriminate, don't baseline.** Two positional/controlled discriminators replaced the
baselines, and every baseline is now **0**:

- **SC2168** is discounted only when the flagged start column falls after a `##` on a line that does
  not begin with a TAB (recipe lines always do; target/help lines never do).
- **SC1078** is discounted only for a file whose own `bash -n` exits 0 — re-run every invocation, so
  it cannot go stale. Deliberately narrowed to SC1078 rather than all parse-class rules.

**The table as finally run, against the shipped design:**

| # | row | expectation | observed |
|---|-----|-------------|----------|
| 0 | the four files as they stand | PASS | **rc=0**, `4 file(s) examined, none above its baseline` |
| 1 | a REAL `local` in a recipe body | FAIL | **rc=2**, `Makefile(1>0)`, 1 counted / 1 discriminated |
| 2 | a NEW help string containing "local" | PASS | **rc=0**, 0 counted / **2** discriminated |
| 3 | a REAL unterminated quote in a guard | FAIL | **rc=2**, `check_sourced_libs_option_neutral.sh(1>0)`; control `bash -n` rc=2 |
| 4 | the `ABS_APR` idiom copied into a SECOND guard | PASS | **rc=0**, 0 counted / 1 discriminated |
| 5 | a baseline entry with no justification | FAIL | **rc=2**, `FAIL: baseline entries carrying no justification: scripts/ci.sh` |
| 6 | the file list matches nothing | FAIL | **rc=2**, `FAIL: this gate examined NOTHING and was about to report success.` |
| 7 | a malformed entry containing `;` | FAIL | **rc=2**, `FAIL: baseline entries outside the safe charset: scripts/bad;rm%lint%0%probe` |

Rows 1+2 together are the whole point: the **same rule**, two instances, correctly separated by
position. Rows 3+4 are the same for SC1078, separated by `bash -n`.

**Row 4 is reported only because the mutation was verified applied.** Its first attempt silently
did not modify the file (`grep -c` returned 0) and was **discarded, not reported** — a passing gate
against an unmutated file proves nothing. On the retry the exact idiom line was copied, bashrs
**did** raise `120:83-84 [error] SC1078` (rc=2), and the gate still passed — a false positive that
genuinely fired and was correctly not counted.

Row 7 also required a fix: with unquoted entries a `;` aborted the recipe with a raw bash parse dump
*before* the charset guard could run. Entries are now single-quoted so the guard is reachable and
emits a real diagnostic. Fail-closed either way, but a parse dump is not a diagnostic.

## The zero-match probe that proved the gate non-vacuous

```
FAIL: this gate examined NOTHING and was about to report success.
BASHRS_SCOPED_BASELINE is empty, or every entry was skipped. A gate whose
file list stops matching must go RED, not green (contract-audit-phase4 above
carries the same guard for the same reason).
```

rc=2. Reverted; `4 file(s) examined` restored.

## The tool-absent path

Both targets are wired into blocking tier3, so their behaviour when bashrs is missing is part of
their contract. Measured on a `PATH` proven to lack bashrs (`command -v bashrs` → ABSENT):

```
absent_lint_makefile_rc=2   FAIL: bashrs is not installed, so this check DID NOT RUN.
absent_scoped_rc=2          FAIL: bashrs is not installed, so this gate DID NOT RUN.
                            A check that did not run is never reported as passing (F-07).
                            Install with: cargo install bashrs
```

The repo's ambient idiom is the swallowing `|| echo "... not found"` form, still present at
`pmat-score`; the in-file precedent copied instead is `lint-scripts`, which exits 1.

**The first attempt at this probe returned rc=127 and was discarded rather than reported:** stripping
`PATH` also removed `rtk`, so `env` failed before `make` ever ran. The result said nothing about the
target. Re-run with a `PATH` that keeps everything except the cargo bin directory.

## No source was edited to satisfy a linter

- `rtk proxy git diff --name-only -- scripts/` → **0 bytes**. No script modified.
- `rtk proxy git diff --name-only -- .github/` → **0 bytes**.
- The `dev-setup:` help line is **byte-unchanged**; every `dev-setup` line in the diff is a `+` in the
  new comment block.
- **SC2168 is STILL PRESENT in the lint log, which is the correct outcome** — final tally rc=2,
  `1 error(s)`, and the gate reports `0 counted, 1 discriminated`.
- The **only** deleted line containing `##` is `bashrs-lint-makefile: ## Lint Makefile with bashrs`,
  my own target's help text, replaced with a more descriptive one.

**The five deleted lines, in full** (the whole `−5` of the Makefile diff):

```
-bashrs-lint-makefile: ## Lint Makefile with bashrs
-	@echo "🔍 Linting Makefile with bashrs..."
-	@bashrs make lint Makefile || echo "⚠️  Makefile linting found issues"
-#   setfit-cli-eval-tests      cli   eval::setfit            15 / 0        13
-	@$(call assert_tests_ran,target/setfit-cli-eval-tests.log,13,setfit-cli-eval-tests)
```

`grep -c '|| echo "⚠️  Makefile linting found issues"' Makefile` → **0**. The swallowing form is
gone, not supplemented.

## The `setfit-cli-eval-tests` bookkeeping, closed against a POST-04-20 measurement

```
rtk proxy cargo test -p apr-cli --features setfit --lib eval::setfit   → rc=0
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 6764 filtered out
```

| | old | new |
|---|-----|-----|
| gate-table row | `eval::setfit  15 / 0  13` | **`20 / 0  18`** |
| `assert_tests_ran` floor | 13 | **18** |

18 satisfies `13 < 18 < 20` and matches the ~90% convention every other row uses (and exactly the
`20 / 18` pair `setfit-parity` already carries). **Floor proven non-vacuous:** with the filter pointed
at a zero-match name, libtest exits 0 reporting `0 passed` while the gate correctly fails rc=2 —
*"FAIL: setfit-cli-eval-tests reported 0 test(s) passed, expected at least 18. A name filter that
matches nothing exits 0 (REVIEW CR-02)"*. Reverted.

That probe's first attempt also failed to apply and was **discarded, not reported**, then re-run with
the mutation verified (`grep -c` = 1) before the RED was believed. `:1920` and `:2177`, which 04-21
owns, were not re-touched.

## Verification

Every status captured with `cmd > log 2>&1; rc=$?`, never through a pipe.

| # | command | rc | result |
|---|---------|----|--------|
| 1 | `make bashrs-lint-makefile` | **0** | 0 counted / 1 discriminated; induced-defect run recorded NON-ZERO above |
| 2 | `make bashrs-scoped-lint` | **0** | `4 file(s) examined, none above its baseline`; zero-match probe recorded FAILING |
| 3 | `rtk proxy make -n tier3` | 0 | both targets present at recipe lines 543 and 586; `lint-scripts` count **0** |
| 4 | `make contract-audit-phase4` | **0** | `1 contract(s) audited, every equation is bound` |
| 5 | `make setfit-serve-tests` | **0** | 11 passed / 0 failed — 04-21's floor of 10 intact |
| 5b | `make setfit-cli-eval-tests` | **0** | 20 passed / 0 failed against the new floor 18 |
| 6 | `bashrs make lint Makefile` | 2 | 1 error (the discriminated FP) + 35 warnings |

**Leg 3 was re-run under `rtk proxy` after the first attempt proved worthless:** the rtk hook
truncated `make -n tier3` to 51 lines ending in `... (695 lines truncated)`, so the grep found zero
matches for targets that were in fact present. Reporting "not wired" from that log would have been a
measurement artifact. Raw output: 745 lines, both targets found.

**Leg 6 delta vs the pre-edit baseline:** 33 → 35 warnings, entirely **+2 MAKE012** from the two
`$(MAKE)` lines this plan adds to tier3. MAKE003 stayed at 2 — an interim +1 on a baseline data line
was resolved by the single-quoting fix. The error count is unchanged at 1.

## Deviations from Plan

### [Rule 1 — bug in my own gate] The first gate design was refuted by its own case table

- **Found during:** Task 2, running the must-not-match rows.
- **Issue:** The plan specified a baseline-non-increase gate carrying both false positives as
  justified baseline entries, and listed "a Makefile help string containing the word local" as a
  must-not-match row that PASSES. **It did not pass.** One new help line took the Makefile count
  1 → 2 and turned the gate red on correct prose.
- **Fix:** Both false positives are DISCRIMINATED by controls re-run every invocation, and all
  baselines are 0. The baseline mechanism and its justification requirement are kept — Row 5 proves
  an unjustified entry is still rejected.
- **Why not just accept it:** the plan's own reasoning forbids wiring a gate that trips on prose. A
  gate satisfying the letter of "baseline" while failing the row the plan named would have been the
  liability, one step removed.

### [Rule 1 — bug] Three defects in my own gate, each found by running it

1. Unquoted `|` separators made the shell read every entry as a pipeline → switched to `%`.
2. A literal `##` inside a make variable started a **comment** and truncated the counter mid-string,
   producing an identical unexpected-EOF error → built as `sprintf("%c%c",35,35)`.
3. `rc=$$?` was capturing an assignment's status, not bashrs's — **the exact CLAUDE.md rule 1 defect
   this plan exists to fix**, committed inside the fix for it, and caught only because the printed
   `rc=0` contradicted the measured semantics (a file with an error must report 2). Fixed by
   capturing immediately after bashrs.

Also an off-by-one: `[error] ` is 8 characters, not 9, so the rule id parsed as `C2168` and the
discriminator silently never fired. Caught by debugging the awk against real output rather than
trusting the green run that preceded it.

### [Rule 2 — missing critical functionality] A charset guard that was unreachable

Row 7 showed the malformed-entry guard could never run for `;`, `|` or `&`, because make pastes the
list into shell source and the parse fails first. Entries are now single-quoted so the guard is
reachable. Fail-closed before and after; the change buys a diagnostic instead of a parse dump.

### [Scope, recorded not fixed] A third false positive, in my own change

bashrs reads a baseline **data** line as a command and raises MAKE003 "Unquoted variable in command"
on the `$(...)` inside it. Warning-severity, so it moves no baseline, but it is why the Makefile
warning count moved during development. Recorded inline rather than left as an unexplained +1.

### [Environment] Sequential on the main tree, as briefed

`git rev-parse --git-dir` is a directory; this is the main checkout. The prior dispatch of this plan
correctly refused to execute against a 413-commit-stale worktree. HEAD was verified `9ae41fcaa`
before any edit and no history-rewriting command was run. All commits used explicit pathspecs.

No Rule 3 or Rule 4 deviations. No dependency added, no `Cargo.toml` touched (T-04-SC not triggered).

## Threat Model Disposition

| Threat ID | Disposition | Outcome |
|-----------|-------------|---------|
| T-04-87 (Repudiation: `bashrs-lint-makefile` cannot fail) | mitigate | **Mitigated.** `|| echo` deleted; status read from the report; proven RED on an induced defect that also fails at runtime. |
| T-04-88 (Repudiation: reporting an unavailable check as passing) | mitigate | **Mitigated.** Tool-absent path exits non-zero naming the reason and the install command; an 8-item NOT-RUN list includes `bashrs gate`'s vacuous pass, which is explicitly not counted as green. |
| T-04-89 (Tampering: hardcoded `/tmp` in `scripts/`) | transfer | **Transferred, as planned.** 50 security findings (SEC013 24, SEC014 24, SEC020 4, SEC006 2) triaged as Group A with representative files and a named owner. NOT fixed here. |
| T-04-90 (DoS: wiring a permanently-red gate) | mitigate | **Mitigated.** `lint-scripts` left unwired with the measured 51-of-59 rationale inline; only a scoped gate that is green today was wired. |
| T-04-91 (Tampering: mechanical SC2086 fix over 180 sites) | accept, deferred | Deferred with an explicit must-match/must-not-match precondition on the fixer. |
| T-04-SC (dependency installs) | mitigate | **Not triggered.** bashrs was verified (one path, version recorded), not installed. |

## What remains OPEN — stated plainly

- **This plan closes NEITHER OPS-01 NOR OPS-02.** It does not touch either.
- **SAFE-01 and SAFE-02 both stay UNCHECKED.** This plan improves the honesty of the gate surface
  they are measured on; it delivers neither requirement.
- **SAFE-02's "in CI" clause remains OPEN.** The 16 legs at `.github/workflows/ci.yml:378-397` have
  never executed. Nothing pushed, no PR, no `.github/` file touched.
- **F-10 remains OPEN. SC1, SC2, SC3 and the "over a produced artifact" halves of SC4/SC5 remain
  OPEN** — Phase 5 work per the ROADMAP's blocking note.
- **The per-crate cargo-mutants gate (04-11 must-have 4) remains OPEN**, ≥10 h projected per
  D-04-11-B. Untouched.
- **IN-06 / D-04-14-B** (the `unwrap()` ban being lint-inert in apr-cli) is untouched.
- **The 51-of-59 script backlog is MEASURED and TRIAGED but NOT FIXED.** 106 error / 754 warning /
  1412 info findings remain in the tree; no script was modified.
- **`bashrs gate` is vacuous at every tier** and CLAUDE.md's documented `bashrs gate --strict .`
  invocation does not exist in 6.66.3. Recorded; the CLAUDE.md line should be corrected by whoever
  next edits that section.
- **Upstream bug reports for the two false positives are RECOMMENDED and NOT FILED** — outside
  autonomous scope, recorded as a decision so nobody assumes it was handled.
- **A full `make tier3` was not run.** Both new targets were verified standalone and their presence
  in the recipe confirmed by dry run; tier3 carries standing reds unrelated to this plan
  (D-04-04-B, D-04-08-A).

## Known Stubs

None. Every gate added runs, and each of its failure modes was induced and observed rather than
asserted.

## Threat Flags

None. No network endpoint, auth path, or schema change. The new filesystem interactions are lint
logs written under `target/`.

## Self-Check: PASSED

Files:

- `FOUND: Makefile` — contains `bashrs-scoped-lint`, `bashrs_count_errors`, floor 18; `dev-setup:`
  byte-unchanged; the `|| echo` form absent (grep count 0)
- `FOUND: .planning/phases/04-apr-artifact-and-production-parity/deferred-items.md` — `## D-04-22-A`
  present, "NOT RUN" ×8, additions only across the whole plan (0 deletions vs `9ae41fcaa`)
- `FOUND: .planning/phases/04-apr-artifact-and-production-parity/04-22-SUMMARY.md`

Commits:

- `FOUND: e8fe1e9f3` docs(04-22) — the inventory
- `FOUND: 23a5a3574` fix(04-22) — the gates
- `FOUND: eedb63102` docs(04-22) — the triage

`git diff --name-only` across the plan lists exactly `Makefile` and the phase's `deferred-items.md`.
No file under `scripts/` and none under `.github/` was modified.

Per the orchestrator's instruction, **STATE.md and ROADMAP.md were NOT modified** — the orchestrator
owns those writes.
