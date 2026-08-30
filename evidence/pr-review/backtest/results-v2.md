# PRREV-011 — the backtest, re-run against the three repair lanes

**Spec:** `PR-REVIEW-SKILL-002 v2` §9 step 7 — the acceptance test for the whole design.
**Supersedes:** `results.md` (PRREV-007, 2026-08-30), which scored **1 of 3** and corrected
three of its own findings into F1–F5.
**Tree under test:** `feat/prrev-011-rebacktest` = `origin/feat/prrev-006-wiring`
∪ `prrev-008-guardfix` (F1/F2/F3/F5) ∪ `prrev-009-dupfix` (F4) ∪ `prrev-010-specfix`.
`bats tests/pr-review.bats` on the merged tree: **112 tests, 0 failures.**
**Method:** genchi genbutsu. Every verdict below was produced by running a guard
predicate, the duplication scan, the docs server or the guard itself against the real
merged PR or a signed probe over its real SHAs — never by reasoning about the spec's
account of them.

`scripts/perf-matrix.yaml` is untouched.

---

## The number

**2 of 3 against §9's three named acceptance cases** — up from 1 of 3 — and the one
still failing is the one §9's prose singles out hardest.

| §9 named case | subject | verdict | why |
|---|---|---|---|
| ungrounded CUDA stream claim | **#2771** (merged) | **CAUGHT, and no longer hollow** | trigger fires 17/27 paths; `queries: []` now REJECTS over #2771's real SHAs; the authority is one query away and was fetched this run |
| PERF-055 duplication | **#2742** (merged) | **CAUGHT — both halves of F4** | 17 prior-art files on two concurrent unmerged siblings, 4 of them non-Rust |
| never-ran-Ollama benchmark | `da069a25f` (the real publication) | **NOT CAUGHT** | B4 excludes `book/src/examples/` — 34.7% of the published book, and the directory the claim was published in. **F6** |

**Two new findings, both measured, both reproducible: F6 and F7.** F6 is why case 3 still
fails. F7 is why #2781 returns `duplication_hits: []`, and it also **falsifies PRREV-007's
own account of #2781** — that PR's prior art was never reachable at the diff boundary §2
mandates, and the earlier backtest believed otherwise because it measured in a worktree
sitting on a descendant of main.

**Per §9 step 7's own terms — *"If the skill would not have caught … the never-ran-Ollama
benchmark, the design is wrong and changes before it is enabled"* — the design is still
wrong and must change before PRREV-006 enables it.** F6 is a one-line scope fix whose
precision has already been measured here. F7 needs a decision, not a patch.

---

## Divergence from the brief, taken deliberately (the spec wins)

**#2781 is not the PERF-055 duplication PR.** The brief asks whether the F4 repair now
catches "the sibling-branch case" on #2781. Measured at the actual artifact, #2781 is
+368/−3 across 13 files — it *fixes* a join key. §11's row cites *"~7,200 lines across 46
files"*; that is **#2742** (`a184073ef` → merge `9d45b927d`, 46 files, **7,244
insertions**), also merged, also APR-PERF-GATE-001. PRREV-007 already recorded the
divergence in prose ("the PERF-055 duplication did not happen") and then went on testing
#2781 anyway.

Backtesting a named defect means backtesting the PR that carries it. So the spine here is
**four** merged PRs — #2771, #2781, #2763, #2742 — which satisfies §9's "≥3 merged PRs"
with #2781 kept in full because F7 is only visible on it.

Epic membership checked from the commit messages, not assumed: #2771 names
`github_issue 2706` + PERF-009/PERF-050, #2781 names 2706 + PERF-055, #2763 names
`APR-PERF-GATE-001` outright, and #2742's merge commit names `APR-PERF-GATE-001` and
PERF-004 … PERF-019 (PERF-019 carries `github_issue: 2706` in `roadmap.yaml`). All four
have **`reviews=0, comments=0`**; "defects those reviews missed" continues to mean defects
the author's own verification section missed.

`pmat`'s MCP server was `ConnectionRefused` for this entire run — §3.0's row 3 on day one,
as designed. The `pmat` CLI (3.34.0) is present; the semantic half of §3.A was not
exercised, and where that matters it is said so rather than papered over.

---

## What was run, and what it returned

### #2771 — §3.B: **caught, and F2 removed the hollowness**

**The trigger discriminates, recomputed by the guard, not read by eye**
(`check_pr_review_receipt.sh --match-path` / `--match-message`, one invocation per path
and per commit-message line):

| PR | paths firing `CUDA_PATH_RE` | commit-message lines firing `CUDA_MSG_RE` |
|---|---|---|
| **#2771** | **17 / 27** | **8 / 993** |
| #2781 | 0 / 13 | 0 / 53 |
| #2763 | 0 / 12 | 0 / 309 |

Two must-match, two must-not-match, on real PRs.

**The claim merged with no source.** `git log a596b063f..05d2c0a63`:

> THE MECHANISM. `CudaStream::new` creates this crate's streams with
> `CU_STREAM_NON_BLOCKING`, which is explicitly excluded from legacy default-stream
> ordering. `GpuBuffer::copy_from_host` and `copy_to_host` are `cuMemcpyHtoD` and
> `cuMemcpyDtoH` — LEGACY-stream transfers.

Restated in the PR body. No citation in either.

**The authority exists and was fetched this run.** One `search_cuda_docs` call returned the
CUDA C++ Programming Guide **§2.5.6.1 Legacy Default Stream**, verbatim:

> The key difference between the blocking and non-blocking streams is how they synchronize
> with the **default stream**. CUDA provides a legacy default stream (also known as the
> NULL stream or the stream with stream ID 0) which is used when no stream is specified in
> kernel launches or in blocking `cudaMemcpy()` calls. This default stream, which was
> shared amongst all host threads, is a blocking stream.

The claim is **true and ungrounded** — which is exactly the defect §3.B exists for, and
exactly what the merge did not have to demonstrate.

**The compulsion is now real — signed discrimination pair over #2771's REAL SHAs.**
`PR_REVIEW_REPO=/home/noah/src/aprender`, `base_sha = git merge-base origin/main HEAD =
a596b063f`, `head_sha = 05d2c0a63`, receipts signed with the TEST-ONLY key and verified
against the committed test pubkey:

| probe | receipt | guard |
|---|---|---|
| **E-2771-A** | `cuda.status: consulted`, `queries: []` | **REJECT [B1], exit 1** — *"a consultation that asked nothing is DEGRADED, not clean, exactly as `mutation.attempted=0` is"* |
| **E-2771-B** | identical, plus one `cited` query carrying the §2.5.6.1 excerpt and its matching `excerpt_sha256` | **ACCEPT, exit 0** |

All **4/4** positive controls fired first in both runs, so the guard was live and
discriminating. The third escape — `cuda: not-triggered` — is closed by the trigger
recomputation above (guard line 481). The review can no longer record silence.

That is PRREV-007's F2 discharged against the real PR, not against a fixture.

**§3.A on #2771: 12 hits, 0 true positives.** `record_error`, `ternary` — ambient names in
unrelated crates. See "duplication precision" below.

### #2763 — B4 is silent, and the silence is **correct**

The brief asks whether B4 now catches `2.93× Ollama` and `36.9x over FasterTransformer`
from #2763's diff. Measured against the artifact, the question is malformed: those strings
are **the claim guard's own case table**, added by the PR that hardens it.

Staged breakdown of every added line of `5be3aab55..4e0546ead`, using the guard's own
predicates:

| stage | count |
|---|---|
| added lines where `match_comparative` fires | **35** |
| …of which `match_shipped_surface` accepts | **0** |
| B4 rejections | **0** |

The 35 sit in `scripts/check_no_claim_literals.sh` (25 — its self-test rows),
`scripts/claim_literal_baseline.txt` (3), `docs/specifications/APR-PERF-GATE-001-v2.2.md`
(3), `scripts/lib_baseline_ratchet.sh` (2), and one each in two further guards. **None of
#2763's 12 changed files is on B4's scanned surface**, so B4 cannot fire here by
construction.

That 35 is itself the evidence **F5 landed**: `36.9x over FasterTransformer` — the spelling
APR-PERF-GATE-001 §0.1 uses, and which PRREV-007 measured as a `nomatch` for the old
pattern — now matches. The regex was repaired; the scope then, correctly, declines to
block it. Blocking any of the 35 would red the guard that bans the claim, and §3.C.1's only
exit is a comparator log, which does not exist for a number nobody measured.

**#2763 yields no catch, and should not.** It is a true negative, not a miss.

### The never-ran-Ollama benchmark, tested against the actual event — **F6, NOT CAUGHT**

PRREV-007 tested this case with a synthetic head. This run went to the real commit.

```
$ git log --all -S'2.93' -- 'book/**'
da069a25f  docs: Showcase complete - 851.8 tok/s (2.93x Ollama)  (Refs SHOWCASE-BRICK-001)
```

`da069a25f` adds to `book/src/examples/showcase-benchmark.md`:

```
- **GGUF GPU**: 851.8 tok/s = **2.93x Ollama** (291 tok/s baseline)
**CORRECTNESS-012 fixed! Both GGUF and APR formats exceed 2X Ollama on GPU.**
```

**B4 over `da069a25f^..da069a25f`: 0 fires.** The publication that §3.C.1, §11 and §9 all
name as the scar is **ACCEPTED**.

**Cause, isolated to one line.** `published_claim` gates on `match_shipped_surface`, whose
exclusion list opens:

```sh
tests/*|*/tests/*|test/*|*/test/*)             return 1 ;;
benches/*|*/benches/*|examples/*|*/examples/*) return 1 ;;
```

`*/examples/*` was written for a Rust project layout. The one directory it removes from the
book is **`book/src/examples/` — 153 of the book's 441 published `.md` pages, 34.7%** — and
that is where the claim was published. `match_comparative` fires on the line; the surface
predicate throws it away.

*Published* is checked, not assumed: **all 153 appear in `book/src/SUMMARY.md`** on
`origin/main`, so every one is a rendered mdBook chapter, and
`- [Case Study: Showcase Benchmark](./examples/showcase-benchmark.md)` was in `SUMMARY.md`
at `da069a25f` itself. `book/src/examples/` is the only excluded subtree in the book — no
`tests/`, `benches/` or `fixtures/` directory exists there — so the 34.7% is entirely this
one exclusion.

**Case table, run against the guard's own `--match-shipped-surface`** (14 rows,
`evidence/pr-review/backtest/shipped-surface-f6-cases.tsv`): **12 ok, 2 FAIL, both
must-match, both `book/src/examples/`.** The 12 passes prove the runner is live — an
earlier attempt at this table reported 8 spurious passes because `bash` was not found
inside the loop and a stale `$rc` was read; only the must-match rows exposed it.

**Counterfactual pair, same diff, one line changed** (`book/*) ;;` inserted ahead of the
benches/examples exclusion):

| predicate region | B4 fires on `da069a25f` |
|---|---|
| as merged | **0** |
| with `book/` un-excluded | **2** — `851.8 tok/s = **2.93x Ollama**` and `exceed 2X Ollama on GPU` |

**Precision of the widened scope, measured on the same protocol PRREV-008 used for
`docs/`:** 0 hits over all 153 current `book/src/examples/` pages, and 0 hits over every
added `book/**` line in the last 300 commits of `origin/main`. Stated honestly, that is
**no measured false positives and no measured true positives in the window** — the same
caveat PRREV-008 recorded for the scope it kept. The one true positive is outside the
window, at `da069a25f`, and it is the one the spec is about.

**Fix:** `book/**` is prose, not a Rust project layout. Exempt it from the
`tests|benches|examples|fixtures` exclusions — the counterfactual above inserted
`book/*) ;;` *after* the `tests` line and *before* the `benches|examples` line, which is
enough to catch `da069a25f` but is not the whole exemption; the shipped fix should sit
ahead of all four so a future `book/src/tests/` page is covered too (none exists today).

Then **re-mutate in the widened scope — the old proof does not transfer**, which is F1's
own lesson applied to its successor. `mutate-guard.sh` already carries
`book-removed-from-b4-scope`; it needs a sibling that puts `book/src/examples/` back out
of scope, and §6.3 needs a row-16 variant published under `book/src/examples/` rather than
`book/src/tools/`. Without that row the fixture table would go green on a guard that still
cannot see a third of the book — the exact shape of the guard-universe defect that has now
been found six times in this repository and seven with this one.

### #2781 — `duplication_hits: []`, and the reason is **F7**, not F4

**PRREV-007 read #2781's base wrong, and so did its conclusion.** It used GitHub's
`baseRefOid` (`9d45b927d` = #2742's merge, i.e. main's tip when #2781 merged). §2 mandates
`git merge-base origin/main HEAD`:

```
merge-base(origin/main, 808f1a9b2) = c00ba00cb   (#2772, merged 2026-08-29 09:51)
git merge-base --is-ancestor 9d45b927d 808f1a9b2 -> NO
git grep -q -w run_bands 808f1a9b2  -> rc=1   (absent)
git grep -q -w run_bands 9d45b927d  -> rc=0   (present)
git grep -q -w run_bands origin/main -> rc=0  (present)
```

#2781's branch was cut **before** #2742 merged. The prior art PRREV-007 called *"reachable
because #2742 had merged 17 hours earlier"* was reachable only inside the backtest's own
worktree, which sat on a descendant of `main`. At #2781's own `HEAD` it does not exist —
and §3.A's B6 rule (`index_commit` an ancestor of `HEAD`) means no admissible index can
hold it either. **This is the "READ `origin/main`, not the checkout" scar recurring inside
the instrument built to catch it.**

**The scan at the spec's own boundary** (`c00ba00cb..808f1a9b2`, `--horizon all`):

```
23 needles, 0 hits, 774/774 sibling branches, 20 s
duplication_horizon: ["HEAD", "refs/remotes/origin/* unmerged into origin/main"]
```

**F7 — the horizon has a third region, and it is neither swept nor recorded.** Prior art
that landed on `origin/main` **after the merge-base** is in neither `HEAD` nor the
unmerged-sibling set. Measured per PR, taking main-at-merge-time as the endpoint:

| PR | commits on main not in HEAD | files | of which the PERF prior art |
|---|---|---|---|
| #2771 | 0 | 0 | — |
| #2763 | 0 | 0 | — |
| **#2781** | **1** | **46** | **11** (`test_llm_band.rs`, `perf_gate/*`, `perf_receipt*`) |

#2781's blind region **is exactly #2742**. This is the most ordinary shape there is — your
branch is a day behind and someone merged the thing you are about to write — and it is the
one region the receipt does not even mention. An unstated horizon region is precisely the
defect F4 was raised to fix, one region over: rows 23/24 forbid an unsearched *language*
surface from sitting under a `PASS`; nothing forbids an unsearched *ref* region.

**Fix, in order of cost:** (a) at minimum, record `merge-base..origin/main` in
`duplication_coverage` as `none` so it cannot sit under a `PASS` — the rule rows 23/24
already state; (b) sweep it, which is one `git grep` over one ref and cheaper than the
774-branch sweep already being paid for.

### #2742 — the actual PERF-055 duplication: **CAUGHT, both halves of F4**

Scan at #2742's boundary (`a596b063f..a184073ef`, `--horizon all`): **246 needles, 68 hits,
768/768 sibling branches, 218 s.** 51 hits are on `HEAD`; **17 are `where: branch`**:

| sibling ref | tip / date | prior-art files found |
|---|---|---|
| `origin/feat/v7-receipt` | `3bb5eb4f6`, 2026-08-29 02:37 | `crates/apr-cli/src/commands/test_llm_band.rs`, `aprender-test-lib/src/llm/band.rs`, `perf_gate/{bootstrap,protocol,samples,window}.rs`, `docs/perf-024-measurement-protocol.md`, **`scripts/check_comparator_one_client.sh`**, **`scripts/check_perf_receipt_fields_have_producers.sh`**, **`scripts/lib/perf_receipt.py`**, **`scripts/perf-receipt-fields.yaml`** |
| `origin/feat/n1-band-cli` | `aecf51ea8`, 2026-08-28 12:42 | `band.rs`, `perf_gate/{bootstrap,protocol,samples,window}.rs`, `docs/perf-024-measurement-protocol.md` |

Neither branch is an ancestor **or** a descendant of #2742's head; both tips predate its
merge (2026-08-29 15:34). They are genuine **concurrent siblings** — the configuration F4(b)
named "invisible by construction".

**The four bolded files are shell, Python and YAML.** F4(a) — pmat's Rust-only semantic
reach — and F4(b) — the sibling horizon — both fire on the same real PR, which is the
strongest single result in this backtest.

Honest caveat: the sweep enumerates **today's** refs, not the refs as of 2026-08-29. Both
branches existed and were unmerged then, so the hit is not anachronistic — but the horizon
is not time-travelled and nothing here claims it is.

---

## Duplication precision, measured

| subject | needles | hits | true positives |
|---|---|---|---|
| #2771 | 73 | 12 (all `HEAD`) | **0** — `record_error`, `ternary` in unrelated crates |
| #2781 | 23 | 0 | 0 |
| #2763 | 38 | 1 (`HEAD`) | **0** — `mutation_registry` in an unrelated contract YAML |
| #2742 | 246 | 68 (51 `HEAD` / 17 `branch`) | **17** — all on the branch side |

**0 true positives in 64 `HEAD`-side hits; 17 of 17 on the branch side.** The lexical
HEAD sweep is noise on these four PRs; the sibling sweep is where the value is.
`duplication_hits` is advisory, so §7's ≥90% admission rule does not bite — but a field
that returns 12 ambient names on a GPU PR will be skimmed, and the script's own
`symbols_searched` / `hits_total` fields exist so this ratio is judged rather than trusted.
Raising `NEEDLE_MIN_LEN` or dropping single-word ambient names is a candidate; it is not
proposed here because nothing was measured about what it would cost in recall.

---

## Status of PRREV-007's five findings

| # | discharged? | evidence in this run |
|---|---|---|
| **F1** — B4 never reads the diff | **yes** | `published_claim` runs over the diff; rows 16/17 green; B4 fires 2× on `da069a25f` once the scope reaches it |
| **F2** — `cuda: consulted, queries: []` accepted | **yes** | E-2771-A REJECT / E-2771-B ACCEPT over #2771's real SHAs, 4/4 controls |
| **F3** — `pmat: not-triggered` accepted | **yes** | guard line 468 rejects it unconditionally; row 19 green |
| **F4** — duplication blind to 48.8% and to siblings | **yes** | 17 sibling hits on #2742, 4 of them non-Rust |
| **F5** — B4's regex weaker than `RATIO_RE` | **yes** | `36.9x over FasterTransformer` now matches; 35 firings on #2763 |
| **F6** — B4 excludes `book/src/examples/` | **NEW, open** | 0→2 counterfactual on `da069a25f`; 2/14 must-match failures |
| **F7** — `merge-base..origin/main` in no horizon | **NEW, open** | #2781's blind region is exactly #2742: 1 commit, 46 files, 11 of them the prior art |

---

## What did NOT go wrong

- The merged tree's fixture suite is **112 tests, 0 failures**, and `mutate-guard.sh`'s
  baseline proved GREEN in a mutant tree before a single mutant ran.
- The §3.B trigger discriminates on real PRs: 2/2 must-match, 2/2 must-not-match, both on
  paths and on commit messages.
- The guard was live in every probe: 4/4 positive controls fired before each verdict, and
  E-2771-B proves it is not a guard that reads red because it refuses everything.
- B4's scope declines #2763's 35 ratio lines for the right reason and would have blocked
  the guard that bans them had it not.
- The duplication scan reports `symbols_searched`, `hits_total`, `horizon_branches_total`
  and `horizon_branches_scanned`, which is why the 0-true-positive ratio above is
  computable at all rather than hidden behind an adjective.

---

## Does this meet §9 step 7's bar, and is it safe to enable?

**The §9 table row — "it catches ≥1 real defect those reviews missed" across ≥3 merged
PRs — is MET.** Two, on four merged PRs, each proved by a running mechanism:

1. **#2771** — a device-behaviour claim merged with no source, where the authority exists;
   the review can no longer record silence, and the discrimination pair over the real SHAs
   proves both directions.
2. **#2742** — 17 prior-art files across two concurrent unmerged siblings, 4 of them
   outside any semantic index, on the very PR §11 cites as the duplication scar.

**The §9 prose bar is NOT met: 2 of 3.** §9 step 7 names three cases and says the design is
wrong if the skill would not have caught them. The never-ran-Ollama benchmark — tested
this time against `da069a25f`, the commit that actually published it, rather than against a
fixture — is **still accepted**, because B4's surface predicate discards 34.7% of the
published book.

**It is NOT safe to enable.** Two things must change first:

- **F6 (blocking-class, one line):** exempt `book/**` from the Rust-layout exclusions, add
  a `mutate-guard.sh` mutant that puts `book/src/examples/` back out of scope, and add a
  fixture row publishing under `book/src/examples/`. Precision is already measured: 2 true
  positives on the scar commit, 0 false positives across 153 current pages and 300 commits.
- **F7 (design, not a patch):** decide whether `merge-base..origin/main` is swept or merely
  recorded. Either is acceptable; **silence is not**, because the receipt currently reads
  `duplication_horizon: ["HEAD", "…unmerged into origin/main"]` under a `PASS` while a
  46-file region sits outside both.

Enabling with F6 open would ship the gate whose failure the spec names in §3.C.1, §9, §11
and §12 — a receipt reading `crux: consulted, comparative_claims: []` over a diff that
publishes `2.93× Ollama` to the book, signed, and green.

---

## Reproduction

```bash
git worktree add /tmp/wt-prrev-011 feat/prrev-011-rebacktest && cd /tmp/wt-prrev-011
bats tests/pr-review.bats                                   # 112/0

R=/path/to/aprender   # a clone with origin/main and the 774 remote heads

# S3.B triggers, one guard invocation per path / per commit-message line
for f in $(git -C $R diff --name-only a596b063f 05d2c0a63); do
  scripts/check_pr_review_receipt.sh --match-path "$f"; done      # 17 of 27

# duplication, at the boundary S2 mandates (NOT baseRefOid)
scripts/pr_review_duplication_scan.sh \
  --base $(git -C $R merge-base origin/main a184073ef) --head a184073ef \
  --repo $R --horizon all --json /tmp/2742.json                   # 17 branch hits

# F6: B4 over the real publication, as merged and with book/ un-excluded
#   see evidence/pr-review/backtest/guard-transcripts-v2.txt
```

Every probe is signed with `tests/fixtures/pr-review/keys/pr-review-test-TEST-ONLY.key`,
so none can pass on the signature branch. The two `--match-*` case tables and the four
duplication-scan JSON blocks are committed beside this file.
