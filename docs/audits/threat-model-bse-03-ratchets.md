# Threat model — the three ratchet classes (BSE-03, PMAT-1068)

Phase 0 of BSE-03 (`docs/specifications/build-system-enhancement.md:172`, paiml/infra,
§4 wave 4, report O7). **No code in this phase.** This document is written to be
grilled before `scripts/*` is touched, and every claim about a current script carries
a `file:line` measured on this branch (`bse-03-threat-model`, from `bse-m2-pr-a`
@ `ce311458a`).

The subject is three ratchets that fail on **merge commits** — the spec's metric is
"merge-tree failures from these three guards 3/day → 0". The common cause proposed
here: all three compare a **measurement of the working tree** against a **literal on
disk**, so the merge of two individually-legal branches carries a file whose contents
neither branch wrote.

---

## Universe

Three classes. For each: where the literal lives, who writes it, how it goes STALE.

### 1. README count literals

| | |
|---|---|
| Where | `README.md:44` (`1812** provable contracts`), `README.md:225` (`1812 provable YAML contracts`), `README.md:265` (`1812 contracts across inference, training, …`) |
| Measured against | `find "$REPO_ROOT/contracts" -name "*.yaml" \| wc -l` (`scripts/check_readme_claims.sh:90-92`) |
| Claim extractor | `claimed_contract_counts()` — every number preceding `contract(s)` through ≤2 qualifier words, deduplicated (`scripts/check_readme_claims.sh:168-172`) |
| Who writes it | **A human, by hand.** `--regen` *prints* the numbers "for manual README edit" (`scripts/check_readme_claims.sh:8`); it does not write the file. There is no `readme-sync` target in this repo's `Makefile` (`grep -rn readme Makefile` → no match). The only `readme-sync` binary, `apr-qa-readme-sync` (`crates/aprender-qa-certify/src/main.rs:3,38,78-101`), rewrites the region between `<!-- CERTIFICATION_TABLE_START -->` / `..._END -->` (`crates/aprender-qa-certify/src/lib.rs:451,453`) — **markers that `README.md` does not contain** (`grep -n '<!--' README.md` → no match). It cannot regenerate any of the three contract-count literals. |
| Runs in | `.github/workflows/ci.yml:1079` (`--self-test` then the live check) |
| How it goes STALE | The measurement moves whenever *any* PR adds or deletes a `contracts/*.yaml`; the literal moves only when someone edits three separate prose sites. **Measured now: literals 1812, filesystem 1814 — already two behind.** |
| Current polarity | `compare_count()` (`:25-35`): `claimed > measured` → FAIL (`:27-29`); `claimed < measured` → PASS with a "README lags" line, unless `README_EXACT=1` (`:24,30-33,257`). So today's 1812-vs-1814 drift is GREEN by design (G-11 / PMAT-1062, `:20-23`). |
| Second, independent literal universe | The same file also carries `**N** workspace crates` (`:150-155`) and `**K** CLI commands` (`:174-179`); both have the same authored-literal shape and are in scope for the same normaliser. |

### 2. SATD baseline rows

Named in the spec as one of the three classes; **the guard is not identified in the
tree** — see [Located SATD guard](#located-satd-guard). The nearest thing this tree
actually carries is the CB-200/TDG count:

| | |
|---|---|
| Where | `scripts/cb200_baseline.txt` (one integer, `609`) **and** `.pmat-gates.toml:96-98` (`[tdg] baseline = 609`) — the same number in two files |
| Meaning | count of functions below `[tdg] min_grade = "B"` (`.pmat-gates.toml:101`) that `pmat comply` reports as CB-200 Warn at-or-under, Fail above (`scripts/check_complexity_ratchet.sh:368-374`) |
| Who writes it | A human, in both places, by hand |
| Enforced by | `cb200_pair_check()` — the two must be **exactly equal** (`scripts/check_complexity_ratchet.sh:388-406`), called at `:522`; and `baseline_ratchet_check … count` at `:523` |
| Classified | `count` in `scripts/check_baseline_ratchets.sh:91` ("mirrors `.pmat-gates.toml [tdg]` baseline (PMAT-937)") |
| How it goes STALE | `count` is a **ceiling only** — `_br_cmp_count` reds a rise (`scripts/lib_baseline_ratchet.sh:288-291`) and treats a fall as `BR_REMOVED` (`:292-294`). So an *improvement* leaves `609` overstating the tree indefinitely, in two files, and the number nobody re-derives becomes a budget for the next regression. |
| SATD relevance | TDG is "a weighted sum of complexity, coverage, SATD, mutation metrics" (`contracts/tdg-scoring-v1.yaml:15`), so SATD moves this number without ever being named by it. |

### 3. Complexity rows

| | |
|---|---|
| Where | `scripts/complexity_baseline.txt`, rows `<path>::<function> <cyclomatic> <cognitive>` (`scripts/check_complexity_ratchet.sh:44,66`) |
| Measured by | `cx_measure()` over `cx_universe()` = `git ls-files '*.rs'` ∪ `find` (tracked ∪ working tree), fed to `pmat` in chunks of 400 (`:83,97-112`); thresholds 30/25, copied from `.git/hooks/pre-commit` and explicitly "not a dial" (`:53-57,71-72`) |
| Who writes it | `bash scripts/check_complexity_ratchet.sh --update`, i.e. a human running a regenerator that snapshots *the tree in front of them* (`:489-501`) |
| Runs in | `.github/workflows/ci.yml:991` |
| Verdict | `NEW → RED`, `GROWN → RED`, `STALE → RED (delete it)` (`:34-36`), emitted by `cx_verdict()` (`:169-199`, STALE at `:190`); plus `baseline_ratchet_check … keyed2` against `origin/main` (`:521`) |
| How it goes STALE | **By being right.** `STALE` fires the moment a recorded function is fixed (`:36`), and that is deliberate — "the half that makes this a ratchet rather than an allowlist … without it a row survives its own repair" (`:38-40`). The row must be deleted **in the same commit** as the fix. Nothing derives it. |

### The shape all three share

`check_complexity_ratchet.sh:511-518` already states it for its own class:

> *NEW and STALE are the only two properties a working tree can answer, and a commit
> that appends a row AND lands the matching offender satisfies both at once.*

The merge-commit failure is the mirror of that sentence. Branch A fixes function `f`
and deletes its row. Branch B, in parallel, deletes a different row. Both are legal
against `origin/main`. Their **merge** carries a `complexity_baseline.txt` that is the
union-minus of two edits, measured against a tree that is the union of two fixes —
a state neither author ever ran the guard on. Same for `cb200_baseline.txt` (two
branches lowering `609` produce a merged literal that mirrors neither) and for the
README (two branches adding contracts leave a literal that lags by their sum, which
`--exact` reds at `:31,331`).

---

## Normaliser

**Measure `origin/main@SHA` and the merge commit in the same job. The file on disk is
never a comparand.**

### What "measure" is, per class

| Class | `measure(rev)` | Cost |
|---|---|---|
| README counts | `git ls-tree -r --name-only <rev> -- contracts/ \| grep -c '\.yaml$'` — the same universe as `measured_contract_count()` (`check_readme_claims.sh:91`), read from the object store rather than from `find` on the checkout | O(tree), no build |
| CB-200 / SATD | `pmat comply` (or the located SATD guard) run once per rev | expensive; see the open question below |
| Complexity | `cx_measure()` (`check_complexity_ratchet.sh:114-158`) over a worktree of `<rev>` | ~9 s per rev on intel (`:461-463`), so ~18 s for the pair |

The verdict is then a **diff of two measurements**, not a comparison of a measurement
to a stored number:

* regression = `measure(merge)` worse than `measure(base)` → RED
* improvement = better → GREEN, with **nothing to delete**
* unchanged → GREEN

### Why a file on disk can never be the comparand

Three independent reasons, each already observed in this tree:

1. **The author can rewrite it.** `lib_baseline_ratchet.sh:22-29` — "a baseline line
   and its matching violation, added in the SAME commit, satisfy both [NEW and STALE]
   at once … a sweep appending one entry cloned from each file's own last real entry
   found 12 of 12 green." A ratchet is "a property of the DIFF against a ref the
   author cannot rewrite" (`:27-29`).
2. **Nobody writes it on a merge.** A merge commit is produced by git, not by
   `--update`. The file it carries is the textual merge of two edits; it is a
   *claim about a tree that never existed* until the merge created it.
3. **It records a threshold crossing, not a value.** `complexity_baseline.txt` holds
   only functions **over** 30/25 (`check_complexity_ratchet.sh:44-49`), so the file
   cannot distinguish "fixed to 24" from "deleted" from "renamed" — all three read as
   STALE. Only a measurement of the comparand can.

### Reuse the existing resolver — do not write a second one

`baseline_ratchet_resolve()` (`lib_baseline_ratchet.sh:360-400`) already produces the
SHA and the loud-missing semantics this normaliser needs, and it is the **only**
resolver that may be used:

| Verdict | Condition | Consequence in `baseline_ratchet_check` |
|---|---|---|
| `MERGEBASE` | `merge-base(HEAD, origin/main)` exists **and** carries the path (`:367-370`) | preferred comparand |
| `TIP` | tip of `origin/main` carries it, no usable merge-base (`:371-374`) — "CI checks out shallow, so the CI path IS the tip path" (`:53-56`) | stricter comparand |
| `BOOTSTRAP` | ref is literally `origin/main`, neither carries the path, and it is in the working tree (`:375-397`) | `REPORT`, rc 0, "NOT ARMED on this commit" (`:438-446`) |
| `ABSENT` | resolvable but no such path (`:399`) | **RED** — "a missing comparand is not 'no growth'" (`:447-454`) |
| `UNRESOLVABLE` | ref does not resolve (`:362-365`) | **RED**, and it prints the `git fetch` line CI must run first (`:431-437`) |

The default ref is `origin/main` (`:151`); `BASELINE_RATCHET_BASE_REF` overrides it and
**every verdict row then says the ref is not protected** (`:490-494`) — the override
also keeps `ABSENT` loud rather than falling into `BOOTSTRAP` (`:384-387`).

BSE-03's addition is **not** a new resolver. It is:

* a wrapper that turns the resolved `<MODE>\t<sha>` into the pair `(comparand SHA,
  merge SHA)` and prints both **before** any measurement (D2, `APR-QUALITY-001` §7);
* per class, a `measure(rev)` function;
* the diff-of-measurements verdict above.

The three call sites that consume it stay where they are:
`check_complexity_ratchet.sh:521` (`keyed2`), `:522` (`cb200_pair_check`), `:523`
(`count`), and the classification table in `check_baseline_ratchets.sh:78-105`, whose
universe is `find` ∪ `git ls-files` with an unclassified file a **hard failure**
(`:110-131`). A new baseline file must still land in that table.

---

## Polarity table

`$BASE` = `origin/main@SHA` as printed by the resolver; `$MERGE` = the commit under
test. `RED` = exit 1, `GREEN` = exit 0. Rows marked **NEW** are the behaviour BSE-03
must produce; rows marked *today* are what the tree does now, at the cited line.

| # | Case | Command | Expected exit | Status |
|---|---|---|---|---|
| P1 | Complexity regression (a function crosses 30/25, or a recorded one rises) | `bash scripts/check_complexity_ratchet.sh` on `$MERGE` with `$BASE` resolved | **1** | today: RED via `NEW`/`GROWN` (`check_complexity_ratchet.sh:34-35,169-186`) |
| P2 | Complexity improvement (a recorded function drops under both thresholds) | same | **0** | today: **RED** via `STALE` (`:36,190`) unless the row is deleted in the same commit. This is the row BSE-03 changes: with `measure($BASE)` as the comparand there is no row to be stale. |
| P3 | CB-200 / SATD count regression | `bash scripts/check_complexity_ratchet.sh` (the `count` leg, `:523`) | **1** | today: RED (`lib_baseline_ratchet.sh:288-291`) |
| P4 | CB-200 / SATD count improvement | same | **0** | today: GREEN, but only because the literal is a stale ceiling (`:292-294`); after P4 is derived, the number must come from `measure($BASE)`, and `cb200_pair_check`'s exact-equality (`:388-406`) must not turn the improvement RED |
| P5 | Missing comparand SHA (ref unresolvable) | any of the three, `BASELINE_RATCHET_BASE_REF=refs/heads/nope` | **1 at preflight**, before any measurement | today: RED but only at the ratchet call, after the ~9 s scan (`check_complexity_ratchet.sh:470-486` runs first). BSE-03 moves it to preflight (`APR-QUALITY-001` §7 D2). |
| P6 | Comparand resolves but carries no baseline | as above with `$BASE` predating the file | **1** | today: RED (`ABSENT`, `lib_baseline_ratchet.sh:399,447-454`) |
| P7 | README literal **absent** (the count is derived, no literal in the file) | `bash scripts/check_readme_claims.sh --claim contract_count` | **0** | today: **1** — `"README makes no contract-count claim"` (`check_readme_claims.sh:200-203`). BSE-03 must invert this **and** land the generator that makes the count derived; inverting the guard alone deletes the check. |
| P8 | README literal hand-edited to a number the filesystem does not carry, **overstating** | `bash scripts/check_readme_claims.sh --claim contract_count` | **1** | today: RED (`:27-29`, self-test rows `:332-333`) |
| P9 | README literal hand-edited **understating** (lag) | same | **0** normally, **1** under `README_EXACT=1` | today: as stated (`:30-33`, rows `:329-331`) |
| P10 | README carries two different contract counts | same | **1** | today: RED (`:209-212`, row `:334`) |
| P11 | Merge commit, both parents individually green, no regression in the merged tree | all three, on `$MERGE` | **0** | today: unproven — this is the 3/day failure the ticket exists to remove |
| P12 | Merge commit whose merged tree regresses although neither parent did | all three | **1** | must not be lost while fixing P11 |

P11/P12 are the pair that says the fix is a fix and not a hole.

---

## Located SATD guard

**I could not locate it, and I will not name a guess as a finding.**

What I read, and what it rules out:

* **Not in the shell guards.** `grep -rn -i satd scripts/*.sh` returns four sites, none
  of them a baseline: `scripts/dogfood.sh:889` (a comment listing `pmat verify`'s
  stages), `scripts/check_pr_review_receipt.sh:756` (a jq field list including
  `satd_introduced`), `scripts/check_no_claim_literals.sh:1135` (a comment saying SATD
  is *pmat's* gate, not this guard's), and `scripts/verify_pmat_116.sh:30-52`, a
  one-shot PMAT-116 scan asserting **zero** residual `SATD.*PMAT-116` markers — an
  absolute assertion with no baseline and no comparand.
* **Not in the config.** `grep -n -i satd .pmat-gates.toml` → no match. There is no
  `[satd]` section and no SATD number for a PR to edit.
* **Not in CI as a named step.** `grep -rn -i satd .github/workflows/*.yml` returns one
  hit, `book-contracts.yml:204`, a *documentation exclusion* for pages that discuss
  TODO/SATD policy.
* **Not a lower bound anywhere in the shell tree.** `grep -rn -i 'lower bound\|MIN_SATD\|satd_floor'` over `scripts/` and `.github/` returns only perf-statistics
  usages (`scripts/perf_gate.sh:638,711`, `scripts/lib/perf_receipt.py:153,1420`,
  `scripts/perf-receipt-fields.yaml:852,1187,1214,1234`) and one unrelated comment
  (`.github/workflows/ci.yml:1659`). The one numeric floor that does exist is
  `MIN_RS_FILES=5000` (`scripts/check_complexity_ratchet.sh:79`), a **vacuity** floor
  on the file count, not on SATD.
* **Not in `.pmat/jidoka.jsonl`.** The path is untracked in both repos
  (`git log --all -- .pmat/jidoka.jsonl` → empty; the file exists only as
  `/home/noah/src/aprender/.pmat/jidoka.jsonl` and `/home/noah/src/infra/.pmat/jidoka.jsonl`).
  `grep -in 'satd\|lower bound' ` over **both** returns nothing. The aprender log's
  18 tickets (PMAT-742, 929-934, 950-961) contain no SATD entry, and its single
  2026-09-04 entry is PMAT-957 (CUDA driver tests under 48-thread parallelism).

* **A fourth place to look, observed while committing this file.** The local
  `.git/hooks/pre-commit` (the pmat hook, the same one whose 30/25 thresholds
  `check_complexity_ratchet.sh:53-57,71-72` copies) runs a stage it prints as
  `SATD check... ✅ (4 SATD comments)`. So a SATD *count* is produced on every
  local commit, by the hook, over staged files only — the exact asymmetry
  `check_complexity_ratchet.sh:10-16` describes for complexity: the hook
  enforces and CI does not. Whether that count is compared against a stored
  bound, and where that bound lives, is not readable from the tracked tree
  (`.git/hooks/` is not tracked). This is a lead, not the finding.

**The one real lead, and why it is a lead and not a finding.** `git log --all --grep=satd -i` surfaces
`cc9a96b75 test(satd): tighten SATD_PRODUCTION_BASELINE 54 -> 37, the count this tree measures (F-CHECKLIST-005 ratchet after the PMAT-938 sweep)`,
sitting above a run of `PMAT-938: clear N strict-SATD markers …` commits and
`64ba4e784 docs(roadmap): PMAT-750..767 — one ticket per strict SATD marker the pmat-verify row counts`.
So a symbol named `SATD_PRODUCTION_BASELINE` exists, it is a **Rust test** constant
(the `test(satd):` type), it is owned by `F-CHECKLIST-005`, and it was **tightened
downward after an improvement sweep** — which is exactly the shape that reds an
improvement if the constant is an equality or a floor rather than a ceiling.

I did not read that file, so I do not know (a) its path and line, (b) whether the
comparison is `==`, `<=` or `>=`, or (c) whether it is the guard that fired on
2026-09-04. Naming it as the located guard on a commit subject alone would be the
defect this document is about.

**Missing evidence, named so the next phase can close it in one step:**

1. `git show cc9a96b75 --stat` and the diff — gives the file:line of
   `SATD_PRODUCTION_BASELINE` and the comparison operator, i.e. its polarity.
2. The CI log or check-run annotation for the 2026-09-04 failure the report §3.4
   cites, or the run id — nothing in the tree records which guard produced it.
3. `.git/hooks/pre-commit` — the SATD stage above; is its count bounded, and by what?
4. Report §3.4 itself (paiml/infra, report O7 source), which is where the phrase
   "SATD lower bound" originates; it is not a string in either repository.

Until (1) and (2) exist, the SATD class in this model is **`cb200_baseline.txt` +
`.pmat-gates.toml [tdg] baseline` by TDG's SATD component**
(`contracts/tdg-scoring-v1.yaml:15`), carried as class 2 above and marked as the
spec's `[U]` candidate — not as the located guard.

---

## Attack surface

Every way a PR could turn a regression GREEN, and the control that closes it.

| # | Attack | Why it works today | Control |
|---|---|---|---|
| A1 | **Edit the baseline file in the same PR as the regression** (append a row / raise `609` / add a `<path>::<fn>` line) | `NEW` and `STALE` are both satisfiable from the working tree (`lib_baseline_ratchet.sh:16-25`); "12 of 12 green" | Already closed for files: `baseline_ratchet_check` diffs against a ref the author cannot rewrite (`:415-508`). **BSE-03 closes it structurally**: with `measure($BASE)` as the comparand there is no file to edit. |
| A2 | **Edit the comparand resolver** (`baseline_ratchet_resolve`, or add a second one in the new code) | a guard that resolves its own comparand can be told to resolve `HEAD` | One resolver only (`lib_baseline_ratchet.sh:360-400`); its case table is in `check_baseline_ratchets.sh:213-303`, built on a scratch repo and covering `UNRESOLVABLE` / `ABSENT` / `MERGEBASE` / `BOOTSTRAP` / `TIP` including the "overridden ref never bootstraps" row. Any second resolver introduced by BSE-03 is a review-stop. |
| A3 | **Pin a stale SHA** — set `BASELINE_RATCHET_BASE_REF` to an old commit whose measurement is worse | the variable is an ordinary env override (`:151`) | Every verdict row already prints `OVERRIDDEN via BASELINE_RATCHET_BASE_REF — NOT a protected ref` (`:490-494`). BSE-03 must additionally **print the resolved SHA and its committer date** at preflight and RED if the workflow (not a human) did not supply it, so a pin is visible in the log rather than only in the row. |
| A4 | **Delete the baseline to disarm the gate** | `ABSENT` would otherwise read as "no growth" | Closed: missing-from-tree is RED (`:419-423`), and comparand-carries-nothing is RED with the words "a missing comparand is not 'no growth'" (`:447-454`). |
| A5 | **Bootstrap laundering** — introduce a "new" baseline so the `BOOTSTRAP` free pass applies | `BOOTSTRAP` returns rc 0 (`:438-446`) | Already narrowed to three simultaneous conditions and unreachable for any existing baseline (`:375-397`); an override can never reach it (`:384-387`). BSE-03 must not widen it. |
| A6 | **Run under a different tool version** — a `pmat` that scores fewer functions makes any tree look improved | `check_complexity_ratchet.sh:465` prints the binary and version, but **nothing asserts them**; `cb200` depends on `pmat comply`'s grading | The comparand and the merge must be measured **by the same binary in the same job** — that is the whole point of the normaliser. BSE-03 records it in the receipt via **BSE-10a's `tool_version` header** and REDs when the two measurements were not produced by one version. Without this, A6 defeats the entire model: two measurements from two `pmat`s are not a diff. |
| A7 | **Hand-edit the README block** to a number that flatters the tree | the literal is authored (`check_readme_claims.sh:8`) and nothing regenerates it (no `readme-sync` target; `apr-qa-readme-sync` writes a table between markers `README.md` does not have) | Overstatement is already RED (`:27-29`). BSE-03 makes the count **derived** (P7) so the literal is generated, and the generator's output must be diffed in CI — "does the file match what the sources produce", never "does it match a lock". |
| A8 | **Widen the exclusion set** — add a path to `.pmat-gates.toml [tdg] exclude` (`:102-108`) or `[exclude] paths` (`:31-42`) so offenders leave the universe | the universe is config, and the config is in the PR | The universe must be measured at **both** revs: an exclusion added by the PR shrinks `measure($MERGE)`'s universe but not `measure($BASE)`'s, so a universe-size delta is itself a finding and must be printed and REDed, not silently absorbed into "improvement". |
| A9 | **Shrink the scan** — break `cx_universe()` so few files are scanned and everything reads as fixed | a broken scan and a clean tree are indistinguishable | `MIN_RS_FILES=5000` vacuity floor (`check_complexity_ratchet.sh:79,481-486`). The same floor is needed for the new `measure($BASE)` path, or A9 reopens at the comparand end. |
| A10 | **Skip the stage** — `pmat verify --skip satd` | BSE-16 legitimately uses `--skip satd` because SATD is owned by *this* ticket | The forbidden-flag list applies: `--skip` may appear only in the BSE-16 `gate:` recipe with SATD owned here. If BSE-03 does not land a SATD ratchet, `--skip satd` is an unguarded surface, not a delegation. |

---

## Contract

For `contracts/` in the next phase. Stated so a `pv` kernel can be written against it.

> **`APR-RATCHET-D2-001` — a ratchet verdict is a function of `(comparand SHA, merge SHA)` and of nothing on disk.**
>
> Let `M` be the class measurement and `V` the verdict. For every class
> `c ∈ {readme_counts, satd, complexity}`:
>
> 1. **Determinism / purity.** `V_c = f(M_c(base_sha), M_c(merge_sha))`. There exists no
>    input to `V_c` that is a path in the working tree. Falsifier: mutate any tracked
>    file that is not `contracts/**`, `**/*.rs` or `README.md` — the verdict is
>    unchanged.
> 2. **Polarity.** `M_c(merge) > M_c(base)` (worse) ⇒ `V_c = RED`;
>    `M_c(merge) ≤ M_c(base)` ⇒ `V_c = GREEN`. There is **no lower bound**: no value of
>    `M_c(merge)` below `M_c(base)` produces RED. (This is what `STALE → RED`,
>    `check_complexity_ratchet.sh:36`, violates today, and it is the registered
>    mutation below.)
> 3. **Fail-closed on the comparand.** If `base_sha` cannot be resolved, or the
>    measurement at either rev does not complete, `V_c = RED` and the message says
>    UNMEASURED — never "no growth". Verdict must be reached **before** the expensive
>    measurement (P5).
> 4. **One resolver.** `base_sha` is produced by `baseline_ratchet_resolve`
>    (`scripts/lib_baseline_ratchet.sh:360`) and by no other code path. Falsifier: a
>    second function in the repo returning a comparand SHA is a contract violation.
> 5. **One instrument.** `M_c(base)` and `M_c(merge)` are produced by the same tool
>    version in the same job, and that version is recorded (BSE-10a `tool_version`).
>    Falsifier: two `tool_version` values in one receipt ⇒ RED.
> 6. **Universe parity.** The universe of `M_c` is measured at both revs; a change in
>    its size is reported and is never silently an improvement (A8).

Kind: this is cross-cutting, so `metadata.kind: pattern` rather than a kernel contract
with equations — the numeric content is the polarity relation, not a formula.

---

## Acceptance

`bash scripts/tests/ratchet_semantics_test.sh` (new; sibling of the existing
`scripts/tests/{guard_tree,gate_touched_crates,witness_diff,check_roadmap_sorted}_test.sh`).

**The scratch pair.** Not this repository — a throwaway git repo, the pattern
`check_baseline_ratchets.sh:213-232` already uses (`git init`, pinned
`user.email`/`user.name`, `-c commit.gpgsign=false`), extended to two worktrees:

* `BASE` — one commit on `main` holding: `contracts/a.yaml`, `contracts/b.yaml`;
  `README.md` claiming `2 contracts`; `src/lib.rs` with two functions of known
  complexity (the measured fixture at `check_complexity_ratchet.sh:305-364`:
  `branchy` 35/34 and `cognitive_only` 8/28 are over, `tidy` 1/0 and `nested` 7/21 are
  under — measured, not guessed); a `complexity_baseline.txt` matching; a `609`-style
  count file.
* `MERGE` — a second worktree of a branch off `BASE`, mutated per row.
* `refs/remotes/origin/main` is set to `BASE` so the real (non-overridden) resolver
  path is exercised and `BOOTSTRAP` stays unreachable (`lib_baseline_ratchet.sh:384-387`).

**Row setup, one per polarity row above.**

| Row | Setup on `MERGE` | Want |
|---|---|---|
| P1 | raise `nested` above 25 cognitive | 1 |
| P2 | reduce `branchy` under both thresholds, **leave the baseline file untouched** | 0 |
| P3 | count file 609 → 610 | 1 |
| P4 | count file 609 → 600 with the tree improved to match | 0 |
| P5 | `BASELINE_RATCHET_BASE_REF=refs/heads/no-such-xyzzy` | 1, and the failure precedes the scan (assert no `pmat` invocation in the trace) |
| P6 | `BASE` = the commit before the baseline was added | 1 |
| P7 | delete the count literal from `README.md` entirely | 0 |
| P8 | `README.md` claims `9 contracts` over 2 files | 1 |
| P9 | `README.md` claims `1 contract` over 2 files; then again with `README_EXACT=1` | 0, then 1 |
| P10 | `README.md` claims `2 contracts` in one place and `3 contracts` in another | 1 |
| P11 | branch X fixes `branchy`; branch Y adds `contracts/c.yaml`; merge both | 0 |
| P12 | branch X and branch Y each add a function at 24 cognitive that inline into one over-threshold function only when merged | 1 |

Every row asserts an **exit code and a named verdict line**, never a substring the
banner already prints. A row that cannot be built (the scratch repo failed) is a
FAIL, not a skip — the form at `check_baseline_ratchets.sh:263-268`.

---

## Mutation

**Registered mutation: reintroduce a lower bound.**

Restore `STALE → RED` — i.e. put back the `END { for (k in base) if (!(k in cur)) print "STALE" }`
block of `cx_verdict` (`check_complexity_ratchet.sh:190`), or make `cb200_pair_check`'s
equality (`:388-406`) apply to the derived number, or add any `M_c(merge) < M_c(base) ⇒ RED`
branch.

**Expected effect:** the `improvement → GREEN` rows (**P2**, and P4/P7 for the other two
classes) go **RED**. Rows P1/P3/P8/P10 stay RED and P11 stays GREEN, so the mutation is
discriminating rather than merely breaking the suite.

If P2 stays GREEN under this mutation, the test is not measuring the property and the
acceptance is vacuous.

**Second mutation, for A6:** measure the comparand with a different `pmat` than the
merge. The `tool_version` assertion must go RED; if it does not, every row above is a
diff of two incomparable numbers.

---

## Open questions for the grill

1. **P7 vs `check_readme_claims.sh:200-203`.** The spec's polarity is "README literal
   absent → GREEN (count derived, `readme-sync` regenerates the block)". There is **no
   `readme-sync` target** in this repo's `Makefile`, and `apr-qa-readme-sync` writes a
   certification table between markers `README.md` does not carry. So BSE-03 must
   *create* the generator, not reuse one. Until it exists, inverting `:200-203` removes
   a check and adds nothing. Which is it: build the generator in this ticket, or keep
   P7 RED?
2. **Cost of `measure($BASE)` for SATD/CB-200.** Complexity is ~9 s/rev
   (`check_complexity_ratchet.sh:461-463`), so the pair is affordable per PR. A full
   `pmat comply` for CB-200 at two revs is not obviously so, and BSE-16 explicitly
   `--skip`s SATD because it is RED at phase 0. Is the CB-200 leg in scope for the D2
   normaliser, or does it stay a mirrored literal with the pair-check?
3. **The located SATD guard** — blocked on the three pieces of evidence named above.
