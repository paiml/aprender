# APR-PERF-GATE-001 — status review

**Epic:** paiml/aprender#2706, opened 2026-08-27. **Reviewed:** 2026-09-01, at `origin/main` `b7bfcafa1`.
**Spec:** [`APR-PERF-GATE-001-v2.2.md`](./APR-PERF-GATE-001-v2.2.md) (1265 lines) · [`APR-PERF-GATE-001-RESTART.md`](./APR-PERF-GATE-001-RESTART.md)

This is an effectiveness review, not a progress report. It asks one question: **of the work
done in five days, which part produced the thing the epic exists to produce?**

## §1 The epic's own definition of done

From the spec's title: *"a performance number is evidence only if something can prove how it
was measured."* From the RESTART prompt's own priority 1:

> 8/8 perf-matrix cells are `status: UNMEASURED` -> the gate arms nothing. Measure one, honestly.

So the deliverable is **measured cells**, and the gate is the means. Progress is to be reported
"as N/25 from the roadmap, not as a feeling."

## §2 What is measured

| | |
|---|---|
| perf-matrix cells with a §4.4-conformant measurement | **2 of 8** (lambda/W1 #2831, gx10/W1 #2833) |
| perf-matrix cells whose `status:` still reads `UNMEASURED` | **7 of 8** |
| roadmap tasks completed | **16 of 40** |
| roadmap size vs the RESTART prompt's stated 25 | **40** (+60%) |

Both measured cells landed on the **final day** of the five, within hours of each other. Note
the second row: `scripts/perf-matrix.yaml` was deliberately **not** ratcheted for either cell, so
by the epic's own instrument the score is 1/8 measured, not 2/8. That is defensible — Arm A
REPORTS rather than gates while N is small — but it means **the artifact that decides whether the
gate arms has not moved at all in five days.**

## §3 Where the effort actually went

33 PRs merged 2026-08-27 → 2026-09-01, classified by what each PR's own title says it did.

| category | PRs |
|---|---|
| **Measurement delivered** (#2781, #2831, #2833) | **3** |
| **Repairing the instrument** — guards that could not fail, ratchets, gate wiring | **12** |
| **CI / infrastructure firefighting** | **10** |
| Product defects found *through* the epic (#2707, #2771, #2776) | 3 |
| Spec / scope | 2 |
| Unrelated | 3 |

### §3.1 A quantitative effort claim was attempted, and is WITHDRAWN

The first draft said "two thirds of the epic's effort went into the instrument." Review objected
that counting PRs treats a one-line fix and a 1,500-line integration as equal. Lines changed were
then measured, and appeared to support the claim more strongly (repair 78%, measurement 6%).

**A second review attacked that number, and it does not survive.** Two defects, both found by
re-measuring rather than by arguing:

1. **One PR was 58% of the "repair" total, and it was misclassified.** #2705 is 32,432 lines, of
   which **19,813 are `evidence/parity-http/*.json`** — measurement receipts, filed under *repair*.
2. **Re-measured over code only** (`scripts/`, `crates/`, `src/`, `.github/`), the split becomes
   repair 34,013 lines vs measurement **188**. That looks damning and is *also* meaningless:
   **a measurement PR's product is an artifact, not code.** Counting lines of code rewards the
   category that writes code, by construction.

Both proxies are category-confused, in opposite directions. The number that would settle it is
developer-hours, and nobody recorded any. **The quantitative claim is withdrawn.** What survives
is the count above — 12 PRs repaired the instrument, 3 produced a measurement — and that is a
count of PRs, not of effort, and is reported as such.

The first review predicted the effort measurement would invert the conclusion; it did not. The
second review predicted the figure was an artifact; **it was**, for a reason neither review named.

## §4 What was effective

**4.1 The receipt rule found real defects, and they were not cosmetic.** The epic's method —
refuse a number without a producer — surfaced three product defects that had shipped:
`PERF-021`'s boolean accelerator request (#2707, shipped twice under two spellings), three
structural defects behind batched-decode garbage (#2771), and unordered CUDA streams giving
*eleven distinct answers to one greedy question* (#2776). None of these is a benchmarking
artifact; all three are correctness bugs in shipped inference, found by demanding provenance.

The stream finding is `cited`, not `asserted`: the CUDA Programming Guide states that with
non-blocking streams *"we cannot assume any ordering of execution of the kernels and should
perform explicit synchronization"*, and that stream priority *"acts as a hint to the runtime to
influence the scheduling, but does not guarantee a specific order of execution."* Eleven distinct
answers to one greedy prompt is the documented consequence of an unordered reduction, not a
surprise — which is why it should have been caught by design review rather than by a benchmark.

**4.2 "Every change ships its gate, with the named mutation observed RED" works.** Where it was
followed, the gate held. The mutation-verified guards (`check_pass_grep_anchored.sh`,
`check_no_claim_literals.sh`, `mutate-guard.sh`) each caught escapes after they shipped —
which is what a ratchet is for.

**4.3 The `UNMEASURED` / `NOT_APPLICABLE` vocabulary is a genuine contribution.** Distinguishing
*temporary, counted against the denominator, needs owner and expiry* from *permanent, needs
`decided_by`* is what stops a matrix from quietly shrinking its own denominator. `PERF-056`'s
expiry-anchor work (dating an expiry from an event rather than an invented calendar date) is
the same idea applied to time.

**4.4 The aperture ratchet solved a real and subtle problem.** `set-aperture` distinguishes
"this branch wrote a new violation" from "widening the guard revealed a pre-existing one" —
a distinction the working tree cannot make and the comparand can. Without it a guard could not
be widened at all.

## §5 What was not effective

**5.1 The repairs were rework, not construction — and that is the narrower, defensible claim.**
Cross-vendor review argued this section commits a category error: building a falsifiable gate is
capital expenditure paid once, and comparing it to the operating cost of the first two
measurements is "project-management myopia." **That objection is half right, and it changes the
claim.**

It is right that CapEx-vs-OpEx is the correct frame, and the first draft did not use it. It is
wrong that the 12 PRs were CapEx. They were not building the gate for the first time; they were
repairing gates that had already shipped. #2705's own title is *"repair nine gates that could not
fail"* — nine gates that existed, were relied upon, and could not fail. That is rework, and
rework does not amortise.

So the claim is not "too much was spent on infrastructure." It is: **the epic's own rule — every
change ships its gate, with the named mutation observed RED — was written because gates kept
shipping unfalsified, and gates kept shipping unfalsified after it was written.**

**This remains contested and is recorded as such.** A second review argued the rework/construction
distinction is a semantic move to preserve a prior conclusion: if a gate "could not fail" it never
functioned as a gate, so repairing it is part of building it, and therefore amortises. That is a
coherent reading. The distinction this review relies on is narrower — #2705 repaired gates that
**pre-dated the epic** (the 2026-08-13 batch), so they were not this epic's construction — but the
disagreement is real, it is not settled by the evidence available, and §3.1 has already withdrawn
the quantitative claim that would have settled it.

**5.2 One defect class consumed four sequential PRs, and it was a class already named in this
repository.** `#2776 → #2799 → #2804 → #2829` is a four-deep chain on one shape: a
`producer | grep -q` returning 141 under SIGPIPE, or a guard whose payload contains the string it
reports as missing. #2804 alone swept **103 sites**.

Cross-vendor review called the "should have enumerated the class first" charge hindsight bias,
arguing such a class is emergent and only visible after the top layers are chipped away. A second
review then charged that the first draft's dated evidence **equivocated** — between "gates that
could not fail" (a symptom, named 2026-08-13) and "SIGPIPE returns 141 though grep MATCHED" (the
mechanism). **That charge is correct and the claim is corrected.** The 08-13 commit concerns
`pipefail` and pass-grep hygiene, not the 141 inversion.

The mechanism was first named on `main` on **2026-08-29** (#2742, and the corresponding lesson
file the same day). #2776 is 08-31, #2799 is 08-31, #2804 is 08-31, #2829 is 09-01. The window is
**two days, not eighteen** — and the first draft's "eighteen days" was wrong.

The weaker claim still holds: the mechanism was named, written down, and then still fixed
site-by-site across three more PRs before #2804 ran the 103-site census. But two days is a
narrow window, and the hindsight-bias objection is *partly* sustained: nobody had the class in
view when #2776 was written.

**5.3 CI throughput is the binding constraint, and the epic made it worse.** 30% of PRs were
fleet firefighting. Worse, the enforcement machinery the epic built runs a full mutation sweep
on **every** PR — contradicting §3.D's own trigger table, which says docs/non-code is *"not
triggered."* Measured 2026-09-01: three receipt jobs started within nine minutes, the runner
host's load went 48 → 133, and **two jobs were cancelled at exactly their 150-minute cap having
proved nothing.** A gate that cannot finish is not a gate.

**5.4 Scope grew 60% while completion sat at 40%.** The roadmap went 25 → 40 tasks. Growth is
legitimate when measurement reveals work, but the RESTART prompt asked for progress "as N/25";
against a moving denominator that number cannot be read.

**5.6 The instrument is invisible to the quality gates that govern everything it gates.** The
epic's apparatus is bash: **145 `.sh` files under `scripts/`, 85 of them `check_*.sh` guards.**
`pmat analyze complexity` over that tree reports:

```
✅ Successfully analyzed 40 file(s)
   271 of 311 file(s) were not analyzed
   no complexity analyzer for: ... .sh (161) ...
```

**Not one shell script is analyzed.** `.pmat-gates.toml` enforces complexity, coverage ≥95% and
TDG on Rust; none of it reaches the guards that decide whether Rust may merge.

**The first draft concluded from this that "the measurement apparatus is the least-measured code
in the repository." Review called that hyperbole, and it was** — it conflates *a static-analysis
tooling gap* with *an absence of quality measurement*, and it contradicts §4.2 of this same
document, which credits the mutation-verified guards for catching real escapes. The guards do
carry quality signals: `bashrs` lint, per-guard `--self-test` case tables, and a derived mutation
set that must kill 100%. Several are stronger than what the Rust code gets.

The accurate, narrower claim: **the guards are governed by no complexity, coverage or TDG
threshold, and no automated gate would have flagged a 105-cognitive-complexity guard or an
untested branch.** That is a real gap — §5.2's dead validation branch was found by the mutation
set, not by any threshold — but it is a gap in *static* coverage, not an absence of measurement.

## §6 A structural finding this review surfaced

`scripts/perf-matrix.yaml` declares `mini` with `compute_class: metal`. **`apr` has no Metal
inference path**: `aprender-serve` pins `aprender-gpu` to `features = ["cuda"]`, nothing enables
`aprender-gpu/metal`, its `Backend` trait has no compute method, the 13 Metal shader constants
are never dispatched, and there is no Q4_K kernel in any form. Measuring that cell today would
have produced a **CPU run recorded as `metal`**, and under I-9 that record is permanent.

The matrix asserted a capability the binary does not have, and no gate caught it because the
matrix is an input to the gate, not an output. Filed as **#2841**. The wgpu route is **#2825**,
titled "Unblocking `mini`'s W1 cell", which records that `--features wgpu` *had never compiled
on any host*.

## §7 Recommendations

1. **When a guard defect matches a class already named in the repository, the first PR runs the
   census.** The original wording — "enumerate a defect class before the first fix" — was
   correctly called an inactionable platitude in review ("know the unknown before you discover
   it"). It is actionable once narrowed to *recurrence*: §5.2's class was named on 2026-08-13 and
   still fixed site-by-site on 2026-08-31. The trigger is a documented class, not omniscience.
2. **Gate the mutation sweep on its own inputs** (§5.3). The sweep's answer is a function of
   exactly seven paths; if none changed, the score cannot have changed.
3. **Decide `mini` before it expires.** Review's strongest omission: §6 finds a missing
   architecture and the first draft recommended nothing about it. Either implement an inference
   path for Apple silicon or formally re-declare the cell — see **#2841**, which carries the
   options and the prerequisites. **Do not enable `aprender-gpu/metal` to make the label true**:
   it would flip `is_available()` without changing what computes tokens, and I-9 makes the record
   permanent.
4. **Ratchet the two measured cells, or record in the matrix why not.** The epic's own instrument
   cannot presently see its only two results.
5. **Bring the guards under a quality gate** (§5.6). 145 shell files decide whether Rust merges
   and no complexity, coverage or TDG threshold applies to any of them.
6. **Report scope growth as a separate line from completion.** The first draft said "freeze the
   denominator"; review objected that measurement legitimately discovers work (§6 is exactly
   that) and a frozen denominator would lie. Report `completed / total` **and** `total added since
   start` — 16/40 with +15 discovered says something 16/40 alone does not.

## §7.1 How this document was reviewed

Four independent passes, and each changed it:

| reviewer | what it changed |
|---|---|
| **`agy /teamwork`** (Gemini, cross-vendor) | Called the PR-count methodology an artifact and demanded an effort measure; called §5.1 a CapEx/OpEx category error; called §7.1 a platitude; noted §6 found an architectural hole and recommended nothing about it (now §7.3) |
| **`pmat`** | `analyze complexity` over `scripts/` — 40 files analyzed, **161 `.sh` skipped, no bash analyzer**. Source of §5.6 |
| **NVIDIA CUDA docs MCP** | Grounded §4.1's stream claim as `cited`: non-blocking streams have no guaranteed ordering, and priority "does not guarantee a specific order of execution" |
| **Adversarial pass** (Gemini 3.1 Pro, against the *revision*) | Broke §3's effort number (found the #2705 misclassification), proved §5.2's dates equivocated symptom for mechanism, and called §5.6 hyperbole. All three sustained |

**Three of the document's own claims were withdrawn or corrected by review, not by the author.**
The effort figure (§3.1), the eighteen-day window (§5.2, actually two days), and the
"least-measured code" claim (§5.6). A status review that survived four passes unchanged would be
the more suspicious artifact.

## §8 What this review did not establish

- Whether the 12 instrument-repair PRs were avoidable, or the irreducible cost of building a
  falsifiable gate. §5.1 asserts the former is partly true; it is not proven.
- Whether the three product defects (§4.1) would have been found by other means. They are the
  epic's strongest justification and the counterfactual is untested.
- Any throughput or latency conclusion. **Two cells on two CUDA hosts is not a matrix**, and
  no cross-silicon claim is available from them.
- Cost. No token or wall-clock accounting exists for the epic, and §3.1 explains why the two
  proxies tried are not substitutes.
- **Whether the instrument-repair burden was avoidable.** This is the question the review set out
  to answer and it is the one it could not. §3.1 withdraws the number; §5.1 records the
  disagreement. Any future attempt needs developer-hours recorded at the time, not reconstructed.
