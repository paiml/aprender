# PRREV-012 — the backtest, re-run against F6 and F7 on the merged tree

**Spec:** `PR-REVIEW-SKILL-002 v2` §9 step 7 — the acceptance test for the whole design.
**Supersedes:** `results-v2.md` (PRREV-011), which scored **2 of 3** and raised F6, F7, F8.
**Tree under test:** `feat/prrev-012-final` = the merged tree of `prrev-008-guardfix`
(F1/F2/F3/F5) ∪ `prrev-009-dupfix` (F4) ∪ `prrev-010-specfix` ∪ `prrev-011-rebacktest`
(F8 fixed), **plus F6 and F7 landed here**.

`scripts/perf-matrix.yaml` is untouched. `.github/workflows/ci.yml` is inherited from
PRREV-006 unchanged — **this branch adds no line to it**, and `check_pr_review_wiring.sh`
is re-run below to say so mechanically rather than in prose.

**Method:** genchi genbutsu. Every verdict below was produced by RUNNING a guard, a
predicate, the scan or the docs server against the real merged commit — never by reasoning
about the spec's account of it. Where a run contradicted something this report first
asserted, the assertion is corrected in place and the correction is named.

---

## The number

**3 of 3 against §9's three named acceptance cases.** The prose bar and the table row are
both met.

| §9 named case | subject | verdict | mechanism that produced it |
|---|---|---|---|
| ungrounded CUDA stream claim | **#2771** (merged) | **CAUGHT** | trigger fires 17/27 paths and 8/1309 commit-message lines; 0 citation lines in the corpus; signed pair over the real SHAs: `queries: []` → **REJECT [B1]**, one cited query → **ACCEPT** |
| PERF-055 duplication | **#2742** (merged) | **CAUGHT, now in three regions** | 246 needles → 70 hits: **46 HEAD + 17 sibling-branch + 7 `merge-base..origin/main`**, the last of them new here |
| never-ran-Ollama benchmark | **`da069a25f`** (the real publication) | **CAUGHT — this is what changed** | B4's diff scan **0 → 2** on the real commit; end-to-end, one byte-identical receipt run against both guards: **PRE-F6 ACCEPT / POST-F6 REJECT [B4]**, and the honest arm still ACCEPTs |

**`guard_mutation_score` = 100% (185/185) on the merged tree**, re-derived after F6 and F7
widened the guard, because the old proof does not transfer.

**`bats tests/pr-review.bats`: 121 tests, 0 failures** (112 before; +9 for F6's two fixture
rows and F7's seven probes and mechanism tests).

**One new finding, F9, measured and NOT fixed here** — the same commit published the same
ratio to `README.md`, and B4 still does not see it. It is recorded with its counterfactual
so the next ticket starts from a number. See below.

---

## Case 3 — `da069a25f`, the case that had failed twice

### The boundary, stated rather than substituted

The brief says to use `git merge-base` for every base. For this subject it returns
**nothing**:

```
$ git merge-base origin/main da069a25f
                                        (empty — exit 1)
$ git merge-base --is-ancestor da069a25f origin/main ; echo $?
1
```

`da069a25f` (2026-01-18) predates the APR-MONO history rewrite and **shares no ancestry
with today's `origin/main`**. There is no merge base to use. The honest boundary is the
commit's own parent, `099c32287`, and that is what is used — recorded here rather than
quietly replaced with a base that would have looked plausible.

(The other three subjects are ordinary: all of `05d2c0a63`, `a184073ef` and `808f1a9b2`
were **squash**-merged, so none is an ancestor of `origin/main` and
`git merge-base origin/main <head>` is non-degenerate for each — `a596b063f`, `a596b063f`,
`c00ba00cb`. Those are the same three bases PRREV-011 recorded, recomputed here rather
than copied.)

### F6, isolated and measured

`match_shipped_surface` opened with a Rust project layout:

```sh
tests/*|*/tests/*|test/*|*/test/*)             return 1 ;;
benches/*|*/benches/*|examples/*|*/examples/*) return 1 ;;
```

`*/examples/*` is a cargo target-directory rule. The one directory it removed from the
book is `book/src/examples/`:

| | measured on `origin/main` |
|---|---|
| `book/src/**/*.md` | 441 |
| `book/src/examples/*.md` | **153 (34.7%)** |
| …of those, listed in `book/src/SUMMARY.md` | **153** — every one a rendered mdBook chapter |
| other excluded-name subtrees under `book/` (`tests`\|`benches`\|`fixtures`) | **none** |

so the 34.7% is entirely that one exclusion, and the fix is not a fix to an instance.

**The counterfactual, on the real commit, with both guards**
(`evidence/pr-review/backtest/f6-counterfactual-v3.txt`):

| guard | sha256 | B4 fires on `da069a25f^..da069a25f` |
|---|---|---|
| PRE-F6 (`e875f5912`, as handed over) | `fecd9eca…` | **0** |
| POST-F6 (this branch) | `ec1c41f7…` | **2** |

```
FIRE  book/src/examples/showcase-benchmark.md: - **GGUF GPU**: 851.8 tok/s = **2.93x Ollama** (291 tok/s baseline)
FIRE  book/src/examples/showcase-benchmark.md: **CORRECTNESS-012 fixed! Both GGUF and APR formats exceed 2X Ollama on GPU.**
```

`match_target` ran on every candidate line in the same pass, so the two survivors are
survivors of the suppressor, not lines it never saw.

### The end-to-end run, which PRREV-011 could not do

PRREV-011 measured this case at the predicate. This run puts a **signed receipt** through
the whole guard over the real commit, in a scratch clone whose `origin/main` ref is moved
to `da069a25f`'s own parent. The commit, its parent and all content are the real ones; the
only thing moved is a ref pointer, and it is moved because there is no merge base to use.

**One receipt, two guards.** The receipts are byte-identical between the two runs
(`sha256 cf75b7b7…` for the silent arm), so the guard is the only variable:

| probe | receipt says | PRE-F6 guard | POST-F6 guard |
|---|---|---|---|
| **E-SHOWCASE-A** | `crux: consulted`, surface named, `comparative_claims: []`, `verdict: PASS` | **ACCEPT (exit 0)** | **REJECT [B4] (exit 1)** |
| **E-SHOWCASE-B** | identical diff, the ratio RECORDED with command/version/env/artifact/log | **ACCEPT** | **ACCEPT** |

```
REJECT [B4] the diff publishes a comparative claim on a user-facing surface --
  book/src/examples/showcase-benchmark.md: **CORRECTNESS-012 fixed! Both GGUF and APR
  formats exceed 2X Ollama on -- while consultations.crux.comparative_claims is empty;
  a competitor ratio the review never recorded is unverified and blocks (S3.C.1)
```

All **4/4** positive controls fired before every one of those four verdicts, so the guard
was live in each. The PRE-F6 column is the finding: the same signed receipt over the same
commit was **green**, both ways round — the pre-F6 guard could not tell the silent receipt
from the honest one, because it never saw the page.

### Precision of the widened scope

Same protocol PRREV-008 used to measure `docs/` **out** of this scope
(`f6-precision-v3.txt`):

| measurement | result |
|---|---|
| every line of all 153 current `book/src/examples/` pages (35,388 lines) | **0 fires** |
| every added `book/**` line over the last 300 commits of `origin/main` (855 lines) | **0 fires** |
| `da069a25f` | **2 fires**, both the published claim |

Stated as what it is: **no measured false positives, and no measured true positives inside
the window** — this repository has not published a competitor ratio to the book in 300
commits. The only true positives are outside it, at the commit the spec is about. Same
caveat PRREV-008 recorded for the scope it kept; not quietly upgraded here.

### The case table, and the row that had to flip

PRREV-011's 17-row F6 table, re-run against both guards
(`shipped-surface-f6-result-v3.txt`): **PRE-F6 got 2 rows wrong, POST-F6 gets 1, and
exactly 3 rows changed.**

The third change is an **expectation** change, not a defect: `book/examples/x.md` was
written as a control demonstrating that "the exclusion is on the NAME anywhere in the path,
not on `book/src/` depth". Once `book/` is exempt at any depth, a page there is a page. The
shipped table carries it as `MATCH` with the flip and its reason written into the row.

The 14 unchanged rows are the ones that matter for over-reach: every Rust exclusion
(`tests/`, `benches/`, `examples/`, `_tests.rs`), every `docs/` row, the fixture-generator
row and `README.md` all read exactly as before. The exemption is scoped to `book/`, and the
shipped table pins that with a single-variable control:

```
MATCH     book/src/examples/showcase-benchmark.md
NO-MATCH  crates/aprender-core/examples/demo.rs
```

Same directory name, opposite verdict.

### A runner that could not fail, caught by the must-match rows — again

PRREV-011 recorded that an earlier run of this table "reported 8 spurious passes because
`bash` was not found inside the loop and a stale `$rc` was read". **The first attempt at
this re-run failed the same way for a different cause**: the loop variable was named
`path`, which in `zsh` is tied to `$PATH`, so every guard invocation inside the loop lost
its PATH and all 17 rows read `nomatch` — 8 of them "correctly". Only the must-match rows
exposed it. The shipped runner now proves it can produce **both** answers before scoring a
single row, and that control is the first thing in the output file.

---

## Case 1 — #2771, re-measured at the merge base

`base = git merge-base origin/main 05d2c0a63 = a596b063f` (recomputed, not copied).

| | measured |
|---|---|
| S3.B **path** trigger | **17 of 27** changed paths |
| S3.B **message** trigger | **8 of 1309** commit-message lines |
| citation lines in that corpus (`docs.nvidia`\|`programming guide`\|`cuda c++`\|`nvidia.com`) | **0** |

(PRREV-011 wrote "993-line commit corpus"; measured here with `git log --format='%B'` the
corpus is 1309 lines. The corrected number is used. The count that matters — 0 citations —
is unchanged.)

**The authority exists and was fetched this run.** One `search_cuda_docs` call returned the
CUDA C++ Programming Guide §2.5.8 *Implicit Synchronization*, verbatim:

> Two operations from different streams cannot run concurrently if any CUDA operation on
> the NULL stream is submitted in-between them, unless the streams are non-blocking streams
> (created with the `cudaStreamNonBlocking` flag).

`excerpt_sha256 = f225c5c6663796073beac11b63a21bc3ec8b36a975b252e155d23060f90015e8`.

**Signed discrimination pair over the real SHAs**, 4/4 controls fired in both:

| probe | receipt | verdict |
|---|---|---|
| **E-2771-A** | `cuda: consulted`, `queries: []` | **REJECT [B1]** — *"a consultation that asked nothing is DEGRADED, not clean, exactly as `mutation.attempted=0` is"* |
| **E-2771-B** | identical, plus one `cited` query carrying the §2.5.8 excerpt and its matching digest | **ACCEPT** |

Caught, both directions, on the real PR.

---

## Case 2 — #2742, and F7's third region

`base = git merge-base origin/main a184073ef = a596b063f`; 46 files, **7,244 insertions**.
Scan with `--horizon all` (`dupscan-2742-v3.json`):

```
246 needles, 70 hits, 769/769 sibling branches, merge_base_to_main_files=106, 370 s
HEAD 46   branch 17   main 7
duplication_horizon = [ "head=a184073ef…",
                        "siblings=refs/remotes/origin/* unmerged into origin/main",
                        "merge_base_to_main=a596b063f…..refs/remotes/origin/main" ]
```

**The 17 sibling-branch hits are byte-identical to PRREV-011's**, including the four
bolded non-Rust files (`check_comparator_one_client.sh`,
`check_perf_receipt_fields_have_producers.sh`, `scripts/lib/perf_receipt.py`,
`scripts/perf-receipt-fields.yaml`). F4(a) and F4(b) are intact under the change.

**The 7 new `where: main` hits** are the region F7 added. Counted rather than eyeballed: **all seven are non-Rust** — three `.py`, three `.json`, one `.txt` — so every one of them is outside `pmat`'s semantic index. F4(a) and F7 compound rather than overlap:

```
peak_in_flight      evidence/perf-055/0-band-run-stdout.txt:9
test_llm_band.rs    evidence/perf-055/findings.json:11, :33
comparator_status   evidence/perf-055/receipt.r1.json:77
test_llm_band.rs    scripts/lib/bench_receipt.py:52
run_band            scripts/perf041_client.py:158, :199
```

**HEAD went 51 → 46, and the cause is named rather than waved at.** Exactly one needle,
`agg_tok_s`, crossed the ambient-drop threshold (`> 8` hits) because the third region added
to its count, and a needle over the threshold is dropped whole; `dropped_ambient` went
41 → 42 in step. One needle appeared, `run_band` — the very symbol PRREV-011 identified as
#2781's prior art. All four same-name redefinitions PRREV-011 named
(`resolve_base_ref`, `itl_gaps_ms`, `exceeds_budget`,
`a_dominating_request_is_annotated_suspect`) are still returned. Recall given up is a
number here, not an adjective.

### F7's own predict-then-verify, on #2781

PRREV-011 predicted that sweeping `merge-base..origin/main` "returns `test_llm_band.rs`".
Same 23 needles, same script, the horizon the only variable:

| run | hits | the hit |
|---|---|---|
| PRREV-011 (`dupscan-2781.json`) | **0** | — |
| this branch (`dupscan-2781-v3.json`) | **1** | `crates/apr-cli/src/commands/test_llm_band.rs:51`, needle `receipt.r1.json` |

**Measured cost:** the whole scan, all three regions, **22 s** — against 20 s for the
sibling sweep alone in PRREV-011, so the third region costs ~1–2 s on a 75-file region.
That is the 1 s the brief's ruling cites, re-measured.

Honest about what the hit is: a lexical **filename** match landing inside that file's
`//!` doc comment. It is a pointer at the module producing the receipt format #2781 is
fixing the join key of — not a semantic proof of duplication. Recall is unknown and is not
claimed.

### What F7 changed in the artifact

The horizon used to be built from the **method**, so a region that was not swept was simply
**absent**:

```
before:  ["HEAD", "refs/remotes/origin/* unmerged into origin/main"]
after:   ["head=<sha>",
          "siblings=refs/remotes/origin/* unmerged into origin/main",
          "merge_base_to_main=<base>..refs/remotes/origin/main"]
```

Now the horizon says **which regions exist** and `duplication_coverage` says **which were
searched** — two questions, two fields, neither inferable from the other. Per §3.0, an
unsearched region is recorded as unsearched: `merge_base_to_main` is a required coverage
key, so the existing rule "`none` may not sit under a `PASS`" reaches it with **no new
branch to mutate**. Only the horizon-names-three-regions rule is new, and it carries its
own drop and flip mutants (derived, not listed) plus five probes including the pre-F7
spelling quoted verbatim.

---

## F9 — NEW, measured, and deliberately NOT fixed here

Checking *why* the other five files of `da069a25f` do not fire — because "they have no
comparative line" was an assertion, not a measurement — found that one of them does:

```
README.md: The `apr` CLI achieves **2.93x Ollama** performance on Qwen2.5-Coder-1.5B …
```

`match_comparative` fires on it. `match_target` does not suppress it. It is dropped by
`match_shipped_surface`, because a root-level `.md` is not on B4's inclusion list — which
the shipped case table already records as a **KNOWN GAP** ("widening here would put the two
definitions out of step silently"). That row was written before anyone knew `README.md`
carried this claim.

**Not widened in this PR**, for the reason F6 itself follows: a scope change ships with its
precision measurement and with the sibling definition in `check_no_claim_literals.sh` moved
in the same commit, or the two go out of step. The measurement is done, so the next ticket
starts from a number rather than an argument:

| | measured |
|---|---|
| added root-level `*.md` lines over the last 300 commits of `origin/main` | 1,457 |
| of those, B4 would fire on | **0** |
| on `da069a25f` | **1 real claim**, the one above |

The `.svg` hero image on the same commit also carries the ratio and is likewise out of
scope. Recorded, not widened.

---

## The guard's own falsifier, re-derived in the widened scope

F1's lesson applied to its successors: **extending a guard's scope requires re-mutating in
the new scope; the old proof does not transfer.**

`scripts/mutate-guard.sh` derives `drop`/`flip` sites by rescanning the guard for
`reject B<n>`, so the two mutants for F7's new rule appeared without anyone listing them.
Three text mutants were written by hand for F6, because a `case` arm is not a uniform
`reject` site and cannot be derived:

| mutant | what it restores | killed by |
|---|---|---|
| `book-removed-from-b4-scope` | the whole book out of B4's scope | the four `book/**` must-match rows, rows 16/17/25/26 |
| `book-examples-back-out-of-scope` | **exactly the pre-F6 behaviour** — `*/examples/*` re-excluded under `book/` | the two `book/src/examples/` must-match rows and **row 25**, which is the reason row 25 exists |
| `docs-prose-back-in-b4-scope` | `docs/**.md` back in scope | the three `docs/` must-not-match rows |

`182 → 185` mutants (2 derived for the new reject site, 1 hand-written for F6).

**`book-examples-back-out-of-scope` is checked to be a BEHAVIOURAL mutant, not a parse
error** — a mutant that breaks the shell scores a kill it never earned, which is failure
mode 1 at the top of `mutate-guard.sh`. Applied by hand: `bash -n` passes, the line count
is unchanged, and on five probe paths its `match_shipped_surface` agrees with the PRE-F6
guard **exactly**:

| path | mutant | pre-F6 guard |
|---|---|---|
| `book/src/examples/showcase-benchmark.md` | 1 | 1 |
| `book/src/tools/apr-cli.md` | 0 | 0 |
| `book/examples/x.md` | 1 | 1 |
| `crates/aprender-core/examples/demo.rs` | 1 | 1 |
| `docs/benchmarking-gate-spec.md` | 1 | 1 |

So the mutant is the pre-F6 world, and killing it is the statement that the fixtures can
tell the two worlds apart.

```
baseline GREEN: 121 tests, 0 failures      (in a mutant tree, before any mutant ran)
attempted 185   killed 185   survived 0   invalid 0
guard_mutation_score = 100% (185/185)
```

Committed verbatim as `evidence/pr-review/backtest/guard-mutation-run3.tsv`, beside
PRREV-011's `run1` (184/183/1, the FAIL) and `run2` (182/182/0).

**Each new mutant's killer is named, and one of them is a caution.** The workdir is deleted
on success, so both were re-applied by hand afterwards and the trees proved changed before
`bats` ran:

| mutant | killed by |
|---|---|
| `book-examples-back-out-of-scope` | the shipped surface case table · `B4 does not block a claim it has no honest remedy for` · **row 25** |
| `reject-55-drop` (F7's horizon rule) | `probe duplication_horizon naming only two of its three regions` · `probe the pre-F7 horizon spelling, verbatim` |
| `reject-55-flip` | 32 tests, the first being `S6.1 all four positive controls fire` |

**`reject-55-flip`'s kill is not evidence the branch is live, and PRREV-011 already
explained why.** `A || reject … && return 1` parses left-to-right as
`(A || reject …) && return 1`, so when `A` *succeeds* — as it does for every well-formed
receipt — the `return 1` fires anyway and the mutant breaks the **success** path rather
than the rejection path. Only the `drop` mutant asks §6.4's question, and `reject-55-drop`
is killed by the two probes written for it. Recorded so the 32 is not read as coverage.

`guard_mutation_score = 100% (185/185)` — §8 fixes it at one with no ratchet, and §7 makes a
sub-100% score on a guard-touching PR a blocking class.

---

## Status of every finding in this epic

| # | state | evidence in this run |
|---|---|---|
| **F1** — B4 never reads the diff | discharged | B4 fires 2× on `da069a25f`, end-to-end REJECT |
| **F2** — `cuda: consulted, queries: []` accepted | discharged | E-2771-A REJECT / E-2771-B ACCEPT, 4/4 controls |
| **F3** — `pmat: not-triggered` accepted | discharged | row 19 green |
| **F4** — duplication blind to 48.8% and to siblings | discharged | 17 sibling hits on #2742, 4 non-Rust, byte-identical to PRREV-011 |
| **F5** — B4's regex weaker than `RATIO_RE` | discharged | unchanged from PRREV-011 |
| **F6** — B4 excludes `book/src/examples/` | **FIXED here** | 0→2 on the real commit; PRE-F6 ACCEPT / POST-F6 REJECT on one byte-identical receipt; 0 FPs over 153 pages and 300 commits; rows 25/26; two mutants |
| **F7** — `merge-base..origin/main` in no horizon | **FIXED here — swept, and recorded** | #2781 0→1 hit, and it is `test_llm_band.rs`; #2742 gains 7 main-region hits; 22 s total; horizon names three regions; 5 new probes |
| **F8** — the merge left a dead validation branch | fixed in PRREV-011, re-verified here | the 185-mutant set re-derives its sites by rescanning; no survivor at that site |
| **F9** — `README.md` publishes the same ratio | **NEW, open, measured** | 1 true positive on `da069a25f`, 0 would-be false positives over 1,457 added root-`.md` lines in 300 commits |

---

## What did NOT go wrong

- `bats tests/pr-review.bats` is **121/0** on the merged tree, and `mutate-guard.sh` proved
  the unmutated guard GREEN **in a mutant tree** before a single mutant ran.
- The 17 sibling-branch hits on #2742 are unchanged by F7, so the third region was added
  without disturbing the second.
- `scripts/check_pr_review_wiring.sh` is **PASS** (R1–R4, including all four
  `github.event_name` arms) — this branch adds no line to `.github/workflows/ci.yml`.
- `pv validate contracts/pr-review-skill-v2.yaml` — **0 errors, 0 warnings**, with the
  surface set, the region set, two new falsification tests (F-PRREV-010, F-PRREV-011) and
  three new checks added.
- Every probe is signed with the committed TEST-ONLY key, so none can pass on the signature
  branch, and 4/4 positive controls fired before every verdict reported here.

---

## Does this meet §9 step 7's bar, and is it safe to enable?

**The §9 table row is MET, and so is the §9 prose bar: 3 of 3.**

1. **#2771** — a device-behaviour claim merged with no source; 0 of 1309 commit-message
   lines cite the authority that exists and was fetched this run; the guard now rejects
   every spelling of silence and accepts the honest one.
2. **#2742** — 17 prior-art files on two concurrent unmerged siblings, 4 outside any
   semantic index, plus 7 more in the region that landed on `main` after the fork; and the
   four same-name redefinitions the PR's own 1,100-line merge message never mentions.
3. **`da069a25f`** — the never-ran-Ollama benchmark, tested against the commit that
   actually published it, **end-to-end and signed**: accepted by the guard as handed over,
   rejected [B4] by the guard on this branch, with the honest arm still accepted.

**`guard_mutation_score` = 100% (185/185) on the merged tree**, re-derived after the widening
rather than inherited.

**Safe to enable: YES**, on the terms below. The two findings that blocked PRREV-011 are closed with
mechanisms, both measured on the real PRs rather than on fixtures, and the guard's own
falsifier is re-verified in the scope the fixes widened.

Two things a reader should carry forward rather than assume away:

- **F9 is open.** `README.md` published the same ratio on the same commit and B4 does not
  see it. Its counterfactual is measured (0 would-be false positives over 1,457 lines) and
  it is a ticket, not a blocker: the case §9 names is the book publication, and that is
  caught.
- **A signed receipt is still not an honest one.** §4.3 says so and `attestation_level`
  reads `L1-self`. Nothing here may be cited as evidence that a review was diligent — only
  that the artifact it produces can no longer record silence over a published competitor
  ratio, an unasked docs question, or an unsearched region of the duplication horizon.

---

## Reproduction

```bash
git worktree add /tmp/wt-prrev-012 feat/prrev-012-final && cd /tmp/wt-prrev-012
bats tests/pr-review.bats                       # 121/0
scripts/mutate-guard.sh --jobs 24               # 185/185
scripts/check_pr_review_wiring.sh               # PASS
pv validate contracts/pr-review-skill-v2.yaml   # 0 errors

R=/path/to/aprender   # a clone with origin/main and the ~775 remote heads

# F6, on the real publication. NOTE: `git merge-base origin/main da069a25f` is EMPTY.
git -C $R diff --unified=0 da069a25f^ da069a25f   # the boundary that exists
#   see evidence/pr-review/backtest/f6-counterfactual-v3.txt for the 0 -> 2 pair

# F7, on the PR that motivated it
scripts/pr_review_duplication_scan.sh --repo $R \
  --base $(git -C $R merge-base origin/main 808f1a9b2) --head 808f1a9b2 \
  --horizon all --json /tmp/2781.json            # 1 hit: test_llm_band.rs, 22 s

# Case 2, all three regions
scripts/pr_review_duplication_scan.sh --repo $R \
  --base $(git -C $R merge-base origin/main a184073ef) --head a184073ef \
  --horizon all --json /tmp/2742.json            # 46 HEAD + 17 branch + 7 main
```
