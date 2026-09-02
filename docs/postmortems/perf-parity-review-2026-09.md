# Post-mortem — the performance-parity spec review, 2026-09

**Extracted from `docs/specifications/performance-parity-llama.cpp.md` on 2026-09-02**, per an
external audit finding (B-12): a governing specification that must be re-read on every pull
request should be normative, and ~290 of its 657 lines described what earlier drafts of it got
wrong. The archaeology is worth keeping and does not belong in the rule book.

---

## §11.1 How this document was reviewed, and what review changed

Reviewed before leaving DRAFT by an agent quorum and by `agy /teamwork` (cross-vendor).
**Four of its rules were wrong and were changed by review, not by the author:**

| # | what review found | what changed |
|---|---|---|
| 1 | **I-4 as first written outlawed its own fix.** Gating `dec_ratio ≥ 1.0` on every band demands apr dominate two metrics that trade against each other; the continuous-batching PR that fixes §9 #1 would necessarily lower per-user decode and be rejected | §2.2 and §7 made **asymmetric** — decode vs comparator at c=1, aggregate vs comparator at c>1, decode as non-regression against apr's own prior release |
| 2 | **The model-size scaling claim was doing policy work on two confounded points** | Demoted to `[U]`, load-bearing on nothing. *(That round moved the policy onto §2.1's figure; a later round withdrew §2.1 entirely, and the policy now rests on the conservative default — see §2.3. This row records what round 2 decided, not the current rule.)* |
| 3 | **Single-cell gating invites silent rot on the other backends** — a PR doubling CUDA and breaking Metal merges green | §7.2 added: parity gated narrowly, **non-regression gated broadly** |
| 4 | **Refusing attribution outright was over-correction.** The over-count applies to *summing* overlapping spans, not to averaging or utilization | §10 rewritten: summing refused *with the measured ×c proof*; averages and utilization adopted |

Review also supplied the argument for **§7.1**, the kernel microbenchmark gate — the one
mechanism that makes this document cheaper to run than its predecessor. Its verdict on the
first draft was **do not leave DRAFT**; §7.1, §7.2 and the I-4 correction are the response.

The quorum's 37 verified objections against the rejected v3.0 plan supplied §10's measurement
and the comparator-vocabulary bound.

## §11.2 The review loop was stopped on measured reviewer precision, not on agreement

Seven adversarial passes by a cross-vendor reviewer (`agy`, Gemini) under §5's separation rule.
Verdicts: **BLOCK · BLOCK · BLOCK · BLOCK · FINDINGS · BLOCK · BLOCK**, ~23 distinct findings,
**every one of which was accepted and fixed** — including six that were structurally fatal
(a gate that rejected its own fix, a spec that invalidated the evidence it relied on, a status
vocabulary its own tables violated).

**Pass 7 re-raised four findings already fixed in passes 1–2** — `incomplete-withdrawal`,
`missing-archives`, `contradictory-parity-definition`, `unimplemented-matrix-update`. Each was
checked against the exact bytes the reviewer read (`md5sum` identical), and each is demonstrably
present in fixed form. Precision on that pass is **1 of 5 = 20%**.

**The rule that stopped the loop is not in this document.** It is
`PR-REVIEW-SKILL-002-v2.md`'s §7 admission rule — *a class may block only while its measured
precision on the rolling sample is ≥ 90%* — and its §8 `effective_fp_rate` metric. Both govern
the review process; **this specification governs inference performance and defines no such
rules**, so citing "§7" and "§8" here without naming the other document was a dangling
cross-reference into a section of *this* file that says nothing of the kind. Named properly, the
claim is checkable: `grep -c effective_fp_rate docs/specifications/PR-REVIEW-SKILL-002-v2.md`
returns 7, and the same grep against this file returns 0.

So the loop was stopped by a rule that exists, in the document that owns it, and 20% is well
under its bar.

**What this does not license.** It is one reviewer on one document, and 20% is a sample of five.
It is not a precision measurement of the arm, and §8 still owes 30 samples before any threshold
is set from it.


---

## Round 4 — `agy /teamwork`, and the three forks the team did not resolve

A three-role adversarial team (systems architect, performance engineer, QA/release) was pointed
at the revision after rounds 2 and 3. It found three defects — all applied to the spec — and
split three ways on questions that are judgement, not fact. The splits are recorded here rather
than averaged away in the spec, because a spec that hides its live disagreements is how a
premise survives that nobody actually believes.

### What it found (applied)

**The two-term decomposition was stated wrong.** §10 claimed `agg_ratio ÷ dec_ratio` at c=1
*is* the prefill-plus-overhead share, isolated. Expand it and it is
`(apr_agg/apr_dec) ÷ (llama_agg/llama_dec)` — the ratio of the two servers' **own** overhead
fractions. If both spend the same share outside decode it returns 1.0 while saying nothing about
how many milliseconds that share is. The absolute is the per-lane term `agg ÷ dec`, which needs
no ratio at all. Both are now recorded; a decomposition that only ever appears as a quotient
cannot tell you which lane moved. *This defect was introduced by the round-2 fix for B-13.*

**PP-24 defined an unpassable band.** If `apr serve` caps admission at 11 because the KV budget
on a 24 GB card cannot hold sixteen sequences, then c=16 was `admission_capped` forever, the
§12 expiry rule turned that into `FAIL`, and the cell blocked every release with no legal move
available. The band ladder is now *derived* from what both lanes admit, and a deliberate,
server-reported ceiling yields `NOT_APPLICABLE` with `decided_by` rather than a permanent
`UNMEASURED`. **This is the third rule in this document to have outlawed its own remedy**, and
the first to have done so by making a state unreachable rather than by rejecting a change.

**A pinned argv and a per-band concurrency are not compatible.** §5.2 pinned the comparator's
full argv; PP-8 requires its concurrency to equal the band's `c`; §6.1b records that the
comparator is started once, outside the band loop. Pinning harder guarantees the violation. The
pin is now a per-band **template** and the comparator is relaunched once per band.

### Where the team split, and which side the spec takes

| fork | the two sides | the spec's position |
|---|---|---|
| **§2.1: withdraw in place, or delete?** | *Delete* — people skim; a banner will not stop "apr is 10× slower" being quoted from a build with batching compiled out. *Retain* — the asymmetric gate is derived from understanding why that number happened, and the corpse on display is what stops the repeat | **Retain**, and the dissent is real: if the figure is ever cited outside its banner, that is evidence for deleting it and the decision should be revisited, not defended |
| **DESIGNED-NOT-ARMED: theater, or contract?** | *Theater* — an 800-line normative document that says its own gate is not implementable gates nothing. *Contract* — the harness does not exist yet and the producer needs a specification to build against, or it will build the wrong one | **Contract**, conditionally: §7 has an owner and an expiry like any other unmeasured item, and if order 1 of §12 slips, "not armed" stops being a phase and becomes the objection |
| **Roofline or parity as the epic's target?** | *Roofline* — 10.6% of memory bandwidth is objective, single-host, needs no second server, and fixing it yields parity for free. *Parity* — roofline is a theoretical ceiling llama.cpp may itself be far below on this chip; binding the epic to it risks a research project | **Both, with different jobs**: roofline is a bug-finding instrument (PP-23 changes *status*, never a verdict), parity is the product claim. §9 #1 is worked first because it is the only live sized finding, not because roofline replaces parity |
| **§12: serialize behind the comparator lane, or parallelize?** | *Serialize* — optimizing without a trusted instrument is what produced the §2.1 withdrawal. *Parallelize* — a seven-step tooling chain pushes user-visible speed months out | **Neither, as posed**: the chain gates what may be **claimed**, not what may be **investigated**. Profiling, kernels and single-host fixes proceed today; ratios wait for order 1 |
