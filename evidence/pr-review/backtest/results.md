# PRREV-007 — the backtest

**Spec:** `PR-REVIEW-SKILL-002 v2` §9 step 7, the acceptance test for the whole design.
**Subjects:** four PRs from APR-PERF-GATE-001 (#2706). Three merged (#2781, #2771, #2763);
one open (#2776 — see "Divergence 1").
**Under test:** `.claude/skills/pr-review/SKILL.md` (`839745accbfa171a…`) and
`scripts/check_pr_review_receipt.sh` (`dabbe9d16646d581…`) at `feat/prrev-005-skill`.
**Method:** genchi genbutsu. Every verdict below was produced by running the trigger
matcher, the consultation, or the guard against the real PR or a signed probe — not by
reasoning about the spec's account of them.

---

## The number

**It would have caught something real on 1 of the 3 merged PRs** — #2771, where §3.B
demands a citation for a device-behaviour claim that merged ungrounded, and the
authority exists. On #2781 it reproduces mechanically a catch the author had already
made by hand. On #2763 it catches nothing, and reviewing #2763 is what exposed that
the skill's own blocking regex is weaker than the guard #2763 hardened.

**Against the spec's three named acceptance cases: 1 caught, 1 caught-but-hollow, 1 not
caught.** The third — the never-ran-Ollama benchmark, §3.C.1's whole reason for
existing — is not caught, and that is proved below with a signed discrimination pair,
not argued.

Per §9 step 7's own terms — *"If the skill would not have caught … the never-ran-Ollama
benchmark, the design is wrong and changes before it is enabled"* — **the design is
wrong and must change before PRREV-006 enables it.** Four defects, all reproducible,
are listed as F1–F4 with the fix each requires.

| spec §9 named case | PR | verdict | evidence |
|---|---|---|---|
| ungrounded CUDA stream claim | #2776 (open), restated in merged #2771 | **caught, but nothing forces the ask** | trigger fires 5/7 and 17/27 paths + message; docs return Programming Guide §2.5.8 verbatim; `queries: []` is still ACCEPTED (F2) |
| PERF-055 duplication | #2781 | **caught on the Rust half; blind on 48.8%; optional at the gate** | `run_bands` returned as hit #1; 3,533/7,244 lines outside the index; `pmat: not-triggered` ACCEPTED (F3, F4) |
| never-ran-Ollama benchmark | #2763 | **NOT caught** | E1/E1-control: identical diff, ACCEPT vs REJECT turns only on whether the reviewer mentioned it (F1) |
| three structural CUDA defects | #2771 | **0 of 3** | three real docs queries, no authority for any |

---

## What was run, and what it returned

### #2776 / #2767 — would §3.B have forced the citation?

**Trigger: yes, measured.** Asked of the guard, never by eye
(`check_pr_review_receipt.sh --match-path` / `--match-message`):

| PR | paths firing `CUDA_PATH_RE` | body fires `CUDA_MSG_RE` |
|---|---|---|
| #2776 | **5 / 7** (`stream.rs`, `transfer.rs`, `memory/tests.rs`, `cuda/executor/mod.rs`, `cuda-nightly.yml`) | yes |
| #2771 | **17 / 27** | yes |
| #2781 | 0 / 13 | no |
| #2763 | 0 / 12 | no |

Two must-match and two must-not-match on real PRs. The trigger discriminates. The
message column is the PR body; re-run against the **commit messages**, which is the
corpus the guard actually recomputes from (`git log --format=%B "$base..$head"`, line
270), the four verdicts are identical. Worth one line of precision: SKILL.md Â§3.B lists
"a PR body / commit message", the guard reads only `git log`. It did not matter on any
of these four â #2776's and #2771's paths fire regardless â but a device claim that
lives only in a PR body is outside the guard's recomputation.

**The authority exists, measured.** #2776's central claim is asserted with no source:

> `CudaStream::new` created every stream in this crate with `CU_STREAM_NON_BLOCKING`,
> which CUDA explicitly **excludes** from legacy default-stream ordering

One `mcp__nvidia-cuda-docs__search_cuda_docs` call returned the CUDA C++ Programming
Guide **§2.5.8 Implicit Synchronization**, verbatim:

> Two operations from different streams cannot run concurrently if any CUDA operation on
> the NULL stream is submitted in-between them, unless the streams are non-blocking
> streams (created with the `cudaStreamNonBlocking` flag).

So §3.B works as designed on this case: the trigger fires, `not-triggered` is
guard-rejected, and the excerpt that grounds the claim is one query away.

**But the same claim is restated, still ungrounded, in merged #2771**, whose §3.B
trigger also fires — so this is a live grounding defect on a *merged* PR, not only on
an open one. It is the clearest "real thing the review missed" in the set. The review
that missed it was the author's own: **all four PRs have `reviews=0, comments=0`.**

**And the compulsion is hollow — see F2.** The guard forces a *status*, not an *ask*.

### #2781 / PERF-055 — would §3.A `duplication_hits` have caught it?

**On the Rust half, yes — first query, first hit.** Index built cold in this worktree
(**85 s**, 87,592 functions in 10,317 files; SKILL.md quotes 45.4 s — measured under
load average 21):

```
$ pmat query "render a benchmark band run into a provenance receipt" --limit 8 --exclude-tests
crates/apr-cli/src/commands/test_llm_band.rs:349-444 │ run_bands │ TDG: A- │ O(n^2)
```

That is exactly the prior art #2781's body names. The prior art was reachable because
#2742 had already merged: #2781's `baseRefOid` **is** #2742's merge commit
`9d45b927d`. Honest reading: the author found this by hand; the skill reproduces the
catch mechanically in two minutes. It is a real reduction in effort, not a defect the
review missed.

**Three limits, each measured** — F3 and F4 below.

### #2771 — would §3.B have surfaced any of the three structural defects?

**No. 0 of 3.** All three are repo-internal and no external corpus can see them:

| defect | why §3.B cannot reach it | query run |
|---|---|---|
| PTX rescale was a compile-time 1.0 (`sub.f32 %f27, %f8, %f8`) | a Rust PTX builder aliasing its own `VirtualReg`, not device semantics | "PTX ISA virtual register … sub.f32 with identical operands" → `cvt` semantics and operand rules; **no authority** |
| `dispatch_mul_mat` passed `Q4K` unconditionally | GGUF quantisation dispatch; NVIDIA docs do not model GGUF | — |
| graph path never applied the QKV bias | Qwen2.5 weights vs this repo's two forward paths | — |

Worth recording: defect 2 was found with **`apr tensors`**, and §3 has **no consultation
that inspects the artifact under test**, though this repo's own standing rule is
"`apr qa` first". Adding one is out of PRREV-007's scope; the gap is noted so it is not
rediscovered.

### #2763 — §3.C.1's comparator rule

#2763/PERF-049 fixed `check_no_claim_literals.sh`, whose *own registered mutation left
it green* because `RATIO_RE` matched ASCII `x` only while the book publishes `×`. The
pr-review guard's `COMPARATIVE_RE` handles both — the lesson was carried. Two things it
did not carry are F1.

---

## Findings

### F1 — B4 never reads the diff. The 2.93× Ollama case is ACCEPTED. (blocking-class)

`match_comparative` has exactly one call site in the guard (line 378), inside a loop
over **findings the reviewer wrote**. B4's two `jq` inputs are the receipt and the
SARIF. The guard's only `git diff` (line 266) feeds the §3.B path trigger. Nothing
scans the diff, the PR body, the docs or the benchmark output for a ratio — the four
surfaces SKILL.md §3.C.1 says are in scope.

Proved with a signed discrimination pair against one fixture repo and one diff. Head
`7e1105bc3…` adds `book/src/tools/apr-cli.md` containing
`apr sustains 2.93× Ollama on 1.5B Q4_K decode.` — a string the guard's own matcher
calls a comparative claim.

| probe | receipt | guard |
|---|---|---|
| **E1** | `verdict: PASS`, `comparative_claims: []`, no finding mentions the ratio | **ACCEPT, exit 0** |
| **E1-control** | *same diff, same empty `comparative_claims`*, ratio written into a finding | **REJECT [B4], exit 1** |

All four positive controls fired first in both runs, so the guard was live and
discriminating. The only variable is whether the reviewer chose to mention it.

B4 therefore cannot distinguish *"there was no comparative claim"* from *"the reviewer
did not look."* §3.C.1 claims to make the book's ratio "unwriteable rather than merely
discouraged"; as built it is discouraged. This is the §11 row *"Competitor claim with no
source → §3.C.1, blocking"* not holding, and it is the same shape as the scar it cites:
a gate that does not scan the surface where the decision is made.

**Fix:** B4 must run `match_comparative` over `git diff "$base" "$head"` and over the PR
body, exactly as it already recomputes the §3.B trigger, and reject when a ratio appears
there with `comparative_claims: []`. Then re-mutate in the widened scope — the old proof
does not transfer.

### F2 — `cuda: consulted` with `queries: []` is ACCEPTED. (vacuous consultation)

**E2:** row-14's receipt with `consultations.cuda.queries` emptied, re-signed, over a
diff touching `src/cuda/kernel.cu` → **ACCEPT, exit 0**.

The guard enforces the analogous rule for mutation — `attempted: 0` with
`status: consulted` is rejected (line 255) — and §8 sets `vacuous_consultations = 0` as
one of its four zeros. CUDA has no such branch. So the ungrounded stream claim of #2776
and #2771 could ship under a receipt reading `cuda: consulted` that asked nothing:
*"the docs said nothing"* and *"I did not ask"* become the same artifact again, one
level up from where §3.B stops it.

Per-consultation audit of the guard, by grep:

| consultation | trigger recomputed from the diff | vacuity checked |
|---|---|---|
| pmat | no | no |
| cuda | **yes** | no |
| crux | no | no |
| mutation | no | **yes** |

**No consultation has both.** Only cuda's `not-triggered` is falsifiable, and only
mutation's emptiness is.

**Fix:** `cuda.status: consulted` with zero `queries[]` is `DEGRADED`, never `PASS` —
and a `no-authority-found` entry is a query, so the honest path stays open. Same for
`crux.surfaces[]` when its trigger fires.

### F3 — §3.A is optional at the gate, though §3.A calls it unconditional

**E3:** row-14's receipt with `pmat` replaced by `{"status":"not-triggered"}`, re-signed,
over a **code** diff → **ACCEPT, exit 0**.

Spec §3.A: *"Trigger: unconditional"*, *"every PR"*. §3.A also calls `duplication_hits`
*"the highest-EV field in the receipt"* and cites PERF-055 as its evidence. The guard
lets a reviewer skip it by writing three words. Fixture row 7 blesses
`pmat: not-triggered` on a docs-only PR — and its own `trigger_reason` reads *"pmat is
unconditional; not-triggered is never correct for it"*, a fixture that states the rule
it exempts.

**Fix:** `pmat: not-triggered` is a rejection on any diff with a code file, or on any
diff at all if §3.A is read strictly. Row 7 must then carry `pmat: consulted`.

### F4 — `duplication_hits` is blind to 48.8% of the very diff it was designed for

Two measured limits on the mechanism §3.A prescribes:

**(a) pmat's semantic index is Rust-only.** 10,247 tracked `.rs` files; pmat indexed
10,317 files. A semantic query aimed squarely at `perf_gate.sh`'s job — *"validate a
benchmark receipt json against the performance gate schema and print a verdict"* —
returned **10 results, all `.rs`**, and not `perf_gate.sh`. `pmat query --literal` finds
shell only by falling back to a raw file scan, which it labels *"Raw file matches (8
non-indexed)"*. The repo's 214 `.sh` and 71 `.py` files are outside semantic reach.

The prior art #2781 avoided duplicating is **#2742: 46 files, 7,244 insertions** — the
spec's "~7,200 lines across 46 files", confirmed:

| ext | insertions | in pmat's semantic index |
|---|---|---|
| rs | 3,711 | yes |
| sh | 1,873 | **no** |
| py | 828 | **no** |
| yaml | 531 | **no** |
| md / yml / txt / csv / toml / lock | 301 | **no** |

**3,533 of 7,244 insertions — 48.8% — are invisible to `duplication_hits`.** The guards,
gates, receipt libraries and perf harness this epic keeps re-implementing are precisely
the shell and Python half.

This backtest is itself an instance: the pr-review guard's `COMPARATIVE_RE` is a second
implementation of `check_no_claim_literals.sh`'s `RATIO_RE`, in shell, and neither
`duplication_hits` nor any reviewer flagged it.

**(b) Prior art on an unmerged sibling branch is invisible by construction.** §3.A stamps
`index_commit` and B6 requires it to be an ancestor of `HEAD` — correct for staleness,
and it also guarantees the index can only contain this branch's history. Measured live:
`scripts/check_pr_review_wiring.sh` exists on the unmerged sibling
`feat/prrev-006-wiring` and not in this worktree; a semantic query for what it does
returned unrelated Rust functions from `setfit/artifact.rs` and
`apr_transformer/helpers.rs`. An agent about to write it a second time gets
`duplication_hits: []` — a green light to duplicate.

#2781 got lucky on timing: #2742 merged 2026-08-29T15:34, #2781 merged 2026-08-30T09:26.
Had the order reversed, §3.A would have returned nothing. #2781's body names a second
such lineage still open at merge time: `feat/v7-receipt` (`3bb5eb4f6`, 45 files, 9,181
insertions).

**Fix:** `duplication_hits` must also sweep the non-Rust surface (`pmat query --literal`
/ `--regex` over `scripts/**`, or a plain `git grep`), and must state its branch horizon
in the receipt — e.g. `duplication_horizon: ["HEAD"]` vs `["HEAD", "origin/feat/*"]` —
so that "nothing found" is distinguishable from "did not look off this branch". An
unstated horizon is the same defect as an unstated `no-authority-found`.

### F5 (advisory) — B4's regex and its case table were built from the pattern, not the corpus

`evidence/pr-review/backtest/comparative-claim-backtest-cases.tsv` runs 16 subjects drawn
from this repo's real claim corpus through both regexes. Zero false positives on either.
Three must-match misses for B4:

| subject | prrev B4 | main's hardened `RATIO_RE` |
|---|---|---|
| `36.9x over FasterTransformer` — the spelling **APR-PERF-GATE-001 §0.1 uses**, named in #2763 | **nomatch** | MATCH |
| `2x speedup versus Ollama` | **nomatch** | MATCH |
| `3.2x faster than HuggingFace transformers` | **nomatch** | nomatch |

Cause is structural, not a typo. B4's `COMPARATIVE_RE` allows a **zero-word gap** between
the ratio and the competitor; #2763 measured a **five-word** bound (identical hit sets at
5 and 6, zero false positives across 6,900 files at widths 0–6) precisely because
`36.9x over FasterTransformer` "passed on one intervening word". B4's competitor list also
drops `FasterTransformer, SGLang, TGI, LMDeploy, TurboMind, Orca, static batching` while
adding `sklearn, unsloth, candle, burn, ggml, mlx, tinygrad`. Neither list is a superset.

`tests/fixtures/pr-review/comparative-claim-cases.tsv` passes 13/13 because its subjects
were written from `COMPARATIVE_RE`'s own vocabulary — a guard universe built from the
wrong side.

**Fix:** one regex, sourced from `check_no_claim_literals.sh`, with the union of the two
competitor lists and the measured five-word gap. Two independently drifting patterns for
one blocking rule is the duplication F4 exists to detect.

---

## Divergences from the brief (the spec wins)

1. **#2776 is OPEN, and #2767 is an issue, not a PR.** The brief calls the candidates
   "all merged, all real". `gh pr view 2767` → *"Could not resolve to a PullRequest"*;
   #2767 is an open issue, #2776 an open PR. §9 step 7 requires **≥3 merged PRs**, so the
   backtest's spine is #2781, #2771, #2763, and #2776 is used only as corroboration. The
   ungrounded stream claim is not thereby lost: #2771 **merged** restating it.
2. **"The PERF-055 duplication" did not happen.** #2781's author found the prior art and
   wrote one commit. The spec's §11 row reads as though the duplication shipped. What is
   testable is whether §3.A *would* have found it — measured above, yes for the Rust half.
3. **The spec's own §4 wording.** SKILL.md §4 already records that §4 says "DSSE-wrapped"
   while §4.1/§4.3 specify a bare Statement plus a detached signature, and that the guard
   implements the latter. Reconfirmed here; not re-litigated.
4. **No PR had a review.** All four: `reviews=0, comments=0`. "Defects those reviews
   missed" means defects the author's own verification section missed. §5's
   author/reviewer separation — the control A5 calls "the first configuration that beats
   single-agent" — has never been exercised on this epic.

## What did NOT go wrong

Recorded so the fixes above are not read as a verdict on the whole design:

- The §3.B path and message triggers discriminate correctly on four real PRs (2/2
  must-match, 2/2 must-not-match), including the deliberately over-broad `*cuda*`.
- The guard's four positive controls fired first on every run in this backtest. It is not
  a guard that reads green because it refuses everything, and E1-control proves it is not
  one that reads red because it refuses everything either.
- B6 and the merge-base recomputation behaved exactly as specified throughout.
- Neither comparative regex produced a false positive on 16 real subjects.
- The guard resolves its vendored schemas from the repository root, and pointed at a
  checkout without `schemas/` it **halted with POSITIVE CONTROL MISFIRED** rather than
  validating. A control that fired for the wrong reason refused to be evidence. That is
  the behaviour every gate in this repo is supposed to have and most do not.

## Reproduction

```bash
git worktree add /tmp/wt-prrev-007 feat/prrev-007-backtest && cd /tmp/wt-prrev-007
pmat query "x" >/dev/null                      # ~85 s cold; S3.A precondition
pmat query "render a benchmark band run into a provenance receipt" --limit 8 --exclude-tests
tests/fixtures/pr-review/make-fixture-repo.sh /tmp/fixrepo
# then E1 / E1-control / E2 / E3 per evidence/pr-review/backtest/guard-transcripts.txt
```

Every probe is signed with `tests/fixtures/pr-review/keys/pr-review-test-TEST-ONLY.key`,
so none can pass on the signature branch. Fixture SHAs are deterministic and asserted by
`make-fixture-repo.sh` against `expected-shas.txt`; the claim branch head is
`7e1105bc3172e1739ed92ee537cff3137711b111`.

`scripts/perf-matrix.yaml` is untouched.
