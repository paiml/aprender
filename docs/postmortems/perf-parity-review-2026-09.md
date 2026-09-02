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

