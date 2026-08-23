# Dogfood runner divergence triage — #2640

**Required artifact per #2640 A.2.** Produced BEFORE any delete, because the 86 diverging
lines are evidence, not noise: given #2361 (user-scope shadowed the repo skill, so hardening
it edited a file that never ran), some hardening had almost certainly landed only in the copy
that does not run. Deleting either copy first would destroy the record of which.

This is CLUS-000's rule applied to the runner: **prove the copies agree, or catalog the
divergence, before consolidating.**

## Subjects

| copy | path | lines |
|---|---|---:|
| `user-scope` | `~/.claude/skills/dogfood/dogfood.sh` | 1172 |
| `repo` | `scripts/dogfood.sh` | 1165 |

`diff` = 9 hunks, 62 changed lines (86 including context). Gate names: **61 shared, 0 unique
to either.** The copies run the same protocol.

## Headline result

**Every hunk is `hardening-port`. Zero are `drift-discard`.**

This is not random drift. Each copy independently gained a *distinct, documented, measured*
hardening — and both hardenings solve **the same defect class**: a release gate running a
stale binary.

| copy | hardening | its own measured evidence |
|---|---|---|
| user-scope | `PMAT_BIN` — when the crate under test **is** pmat, run the just-built artifact, not PATH | installed `3.32.0` commit `8134bb373` vs built `3.32.0` commit `7a7409e03`. **Both print 3.32.0**; only the commit differs. The installed copy predated CB-200 becoming a ratchet, so the gate ran the OLD zero-tolerance check and returned Fail against a tree the shipped code passes. |
| repo | `PV` **pinned** via `scripts/pv_bin.sh`, never PATH-resolved; REPORT (not silent fallback) where absent | PATH `pv` 0.49.0 vs in-tree 0.63.0 **disagreed on the gate that decides the release** — strict-test-binding: 253 refs / 51 missing stale, 371 / 27 at HEAD. |

Neither author knew about the other. **Deleting either copy silently loses a real hardening**,
which is precisely the outcome A.2 exists to prevent.

Provenance (`git log -S`): the PV pinning entered the repo copy at `c0e63c9ce` (#2613).
`PMAT_BIN` returns **no commits** in `scripts/dogfood.sh` history — it never landed in the
repo copy at all, confirming it is user-scope-only work.

## Triage table

| # | hunk | copy | class | rationale | disposition |
|---|---|---|---|---|---|
| 1 | `@@ -278,31 +278,6` | user-scope | `hardening-port` | The `PMAT_BIN` selection block + its 20-line measured rationale. Guards the self-referential case: the tool validating its own release. | **PORT.** Keep verbatim, comment included — the comment *is* the evidence. |
| 2 | `@@ -333,14 +308,30` | repo | `hardening-port` | `PV=""` + `. scripts/pv_bin.sh` + its 12-line rationale, and drops `pv` from the PATH-probe loop. | **PORT.** |
| 3 | `@@ -353,11 +344,11` | repo | `hardening-port` | Positive-control guard becomes `[ -n "$PV" ] && [ -x "$PV" ]` instead of `command -v pv`. Mechanism of #2. | **PORT.** |
| 4 | `@@ -371,7 +362,7` | repo | `hardening-port` | Per-contract validate uses `"$PV"`. Mechanism of #2. | **PORT.** |
| 5 | `@@ -386,7 +377,7` | repo | `hardening-port` | `pv lint` uses `"$PV"`. Mechanism of #2. | **PORT.** |
| 6 | `@@ -414,7 +405,7` | repo | `hardening-port` | `pv verify-bindings` uses `"$PV"`. Mechanism of #2. | **PORT.** |
| 7 | `@@ -434,6 +425,8` | repo | `hardening-port` | The `else` arm: REPORT "pv is not pinned in this repo" rather than validating with an unknown binary. **This is the load-bearing half** — without it #2 degrades to a silent skip. | **PORT.** |
| 8 | `@@ -610,7 +603,7` | user-scope | `hardening-port` | `gate pmat-verify "$PMAT_BIN"`. Mechanism of #1. | **PORT.** |
| 9 | `@@ -635,12 +628,12` | user-scope | `hardening-port` | `pmat query` / `pmat comply check` use `"$PMAT_BIN"`. Mechanism of #1. | **PORT.** |

**`unknown`: 0.** Stated explicitly per A.2, which warns that zero is suspicious at this size.
It is defensible here because the divergence is two coherent changes plus their call sites, not
scattered edits: hunks 3–7 are mechanically entailed by hunk 2, and 8–9 by hunk 1. Every hunk
was read; none was classified by pattern.

**`drift-discard`: 0. `env-specific`: 0.**

## Consequence for the merge

The unified runner must carry **both**, and they compose without interacting — one selects the
`pmat` binary, the other the `pv` binary, and neither reads the other's variable.

## The class is at FIVE instances, not two

Recorded because it changes what the fix must be. The same defect — *a gate that decides a
release resolving its verifier through `PATH`* — already has **five independent ad-hoc fixes**,
none of which knew about the others:

| # | instance | symptom |
|---|---|---|
| 1 | `PMAT_BIN` (user-scope runner) | gate ran a pmat that predated CB-200 becoming a ratchet |
| 2 | `scripts/pv_bin.sh` (repo runner) | PATH pv 0.49.0 vs in-tree 0.63.0, disagreeing on the release-deciding gate |
| 3 | `scripts/apr_bin.sh` | four `apr` binaries coexisted; a bare `apr` resolved to a 26-day-old copy |
| 4 | aprender#2384 | MCP spawned a bare `apr`, ran 0.60.0 while reporting 0.63.0 |
| 5 | APR-BENCH-RFC-001 | unpinned `llama.cpp` comparator — same class, different domain |

Five rediscoveries is the evidence that a rule merely **stated** is documentation. It ships as
a **gate**, or the sixth tool rediscovers it in a sixth copy.

The two hardenings generalise to one rule the merged runner should state once, and ENFORCE:

> A gate that decides a release must not resolve its verifier through `PATH`. Where the repo
> pins the tool, use the pin; where it does not, REPORT rather than fall back — a skipped gate
> that says so beats a green one measured with an unknown binary.

`PMAT_BIN` is the same rule for `pmat`, with the extra self-referential case. Stating it once
is what stops a third copy re-discovering it for a third tool.

## What this triage does NOT cover

The three prose documents (`SKILL.md` ×2, `pre-release/SKILL.md`) are not diffed here — A.5
handles them. Prose duplication is harder to detect than executable duplication precisely
because nothing runs it and no diff surfaces it.
