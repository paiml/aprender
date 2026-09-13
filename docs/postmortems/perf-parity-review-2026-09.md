# Post-mortem — the performance-parity spec review, 2026-09

**Extracted from `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md` on 2026-09-02**, per an
external audit finding (B-12 — that audit was never committed to any repository, so every `B-nn` reference in this file is `[U]`: unretrievable under §0.3 of the master): a governing specification that must be re-read on every pull
request should be normative, and ~290 of its 657 lines described what earlier drafts of it got
wrong. The archaeology is worth keeping and does not belong in the rule book.

---

## §11.1 How this document was reviewed, and what review changed

Reviewed before leaving DRAFT by an agent quorum and by `agy /teamwork` (cross-vendor).
**Four of its rules were wrong and were changed by review, not by the author:**

| # | what review found | what changed |
|---|---|---|
| 1 | **I-4 (now `PP-7` — see Appendix A of `docs/specifications/PP-LLAMA-001-MASTER.md`) as first written outlawed its own fix.** Gating `dec_ratio ≥ 1.0` on every band demands apr dominate two metrics that trade against each other; the continuous-batching PR that fixes §9 #1 would necessarily lower per-user decode and be rejected | §2.2 and §7 made **asymmetric** — decode vs comparator at c=1, aggregate vs comparator at c>1, decode as non-regression against apr's own prior release |
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
cannot tell you which lane moved. *This defect was introduced by the round-2 fix for B-13 (`[U]`: the `B-1…B-14` audit was never committed).*

**PP-24 defined an unpassable band.** If `apr serve` caps admission at 11 because the KV budget
on a 24 GB card cannot hold sixteen sequences, then c=16 was `admission_capped` forever, the
§12 expiry rule turned that into `FAIL`, and the cell blocked every release with no legal move
available. The band ladder is now *derived* from what both lanes admit, and a deliberate,
server-reported ceiling yields `NOT_APPLICABLE` — spelled `NA` in the master's §7.4 status vocabulary, while `NOT_APPLICABLE` survives only as the legacy `comparator_status` wire token — with `decided_by` rather than a permanent
`UNMEASURED`. **This is the third rule in this document to have outlawed its own remedy**, and
the first to have done so by making a state unreachable rather than by rejecting a change.

**A pinned argv and a per-band concurrency are not compatible.** §5.2 pinned the comparator's
full argv; PP-8 requires its concurrency to equal the band's `c`; §6.1b records that the
comparator is started once, outside the band loop. Pinning harder guarantees the violation. The
pin is now a per-band **template** and the comparator is relaunched once per band.

### Where the team split, and which side the spec takes

| fork | the two sides | the spec's position |
|---|---|---|
| **§2.1: withdraw in place, or delete?** | *Delete* — people skim, and a banner will not stop the withdrawn ratio being quoted out of the build it came from. *Retain* — the asymmetric gate is derived from understanding why that number happened, and the corpse on display is what stops the repeat | **Retain**, and the dissent is real: if the figure is ever cited outside its banner, that is evidence for deleting it and the decision should be revisited, not defended. The master resolved it in the dissent's favour: §2.1 of `docs/specifications/PP-LLAMA-001-MASTER.md` names the withdrawn series only to forbid it |
| **DESIGNED-NOT-ARMED: theater, or contract?** | *Theater* — an 800-line normative document that says its own gate is not implementable gates nothing. *Contract* — the harness does not exist yet and the producer needs a specification to build against, or it will build the wrong one | **Contract**, conditionally: §7 has an owner and an expiry like any other unmeasured item, and if order 1 of §12 slips, "not armed" stops being a phase and becomes the objection |
| **Roofline or parity as the epic's target?** | *Roofline* — 10.6% of memory bandwidth is objective, single-host, needs no second server, and fixing it yields parity for free. *Parity* — roofline is a theoretical ceiling llama.cpp may itself be far below on this chip; binding the epic to it risks a research project | **Both, with different jobs**: roofline is a bug-finding instrument (PP-23 changes *status*, never a verdict), parity is the product claim. §9 #1 is worked first because it is the only live sized finding, not because roofline replaces parity |
| **§12: serialize behind the comparator lane, or parallelize?** | *Serialize* — optimizing without a trusted instrument is what produced the §2.1 withdrawal. *Parallelize* — a seven-step tooling chain pushes user-visible speed months out | **Neither, as posed**: the chain gates what may be **claimed**, not what may be **investigated**. Profiling, kernels and single-host fixes proceed today; ratios wait for order 1 |

---

## Rounds 5–8 (2026-09-02)

Four more rounds landed on the reviewed draft after this file was extracted, so their narrative
lived only inside the draft and would have been lost when it was archived. It is recorded here,
one paragraph per commit, and every line range below points into
`docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md` — the draft as archived by the
`PP-LLAMA-001` pull request, byte-identical to the file those commits wrote.

**Round 5 — `494fdaa02`, "a roofline rule that fired on correct batching, and a ten-merge path
to zero speed"** (`+171 −32` across the draft and `evidence/parity/LEDGER.md`). An adversarial
lens was pointed at the tree rather than at the document and told to find the ways the spec fails
*even if implemented exactly as written*. Three of its worst findings had been introduced by
rounds 2–4. **PP-23 declared a correctly-batching server a harness bug**: bandwidth ÷ bytes-per-token
is a *per-sequence decode* ceiling, but PP-23 said "above roofline is schema-fatal" without naming
a metric, and decode was not captured, so aggregate was the only rate it could reach. Against the
epic's own committed receipts the gx10 aggregates at c=8 and c=16 sit **1.45× and 2.81× over** a
vendor ceiling of roughly 58 (tok/s), i.e. schema-fatal on the two bands that carry the parity
claim — receipt `evidence/perf-gate-001-w1-gx10/receipt.r1.json`, table at draft `:470-486`, rule
at `:404`. It is now stated on decode only, which leaves it with no applicable input today, and
that is the honest position rather than one it can apply wrongly. The round also deleted a bare
`25%` literal (a rule that blocks a cell from `MEASURED` is a threshold whatever it is called) and
fixed **§7.2's broad arm, which was monotone in the wrong direction**: `scaling_efficiency(c)`
carries `agg(1)` in its denominator, so halving single-stream throughput *raises* it; the one
speed metric gated across all cells rewarded regressing the very gap the document twice refuses
to concede (`evidence/perf-gate-001-w1-lambda/receipt.r1.json`; draft `:825-837`). The master's
answer is PP-31, which ratchets `agg(c)`, `dec(c)` and `prefill(1)` per band and never ratchets
`scaling_efficiency`.

**Round 6 — `8cb4ca945`, "the premise the whole plan rested on was false, and the kernel lever is
back"** (`+130 −34`). Four lenses — engine, measuring apparatus, git history, adversary — were
pointed at the tree; six findings refuted the document. **The comparator-ratio producer exists and
runs today.** The draft's §6.2a had said the gate has no producer, and the entire §12 chain was
ordered from that one sentence. `scripts/parity_host_receipt.sh` drives both servers with the same
client and varies `--concurrency` per band on both lanes, and `scripts/lib/perf_receipt.py`
already emits both ratio series; run against
`evidence/parity-http/bands/` it reproduces the withdrawn table to four decimals — aggregate
`0.5341 / 0.2308 / 0.1685 / 0.0967` and decode `0.5873 / 0.9231 / 1.3525 / 1.5540` (draft
`:625-628`). What is missing is the JOIN, not the lane: the comparator lane is bolted to the
legacy `apr test llm bench`, whose receipts fail integrity on `timeouts`,
`tokenization.method` and `drain_ms`, while the conformant `--band` producer passes merge and has
no comparator. The round also **re-opened the kernel lever**: decode runs under CUDA graph
capture, and the dispatch reads `use_cublas = m >= 4 && … && !self.is_capturing`, so under capture
cuBLAS is never used and batched GEMV is — with the deficiency named in the tree by the team
itself at `cublas_prefill/mod.rs` (draft `:773`, `:856`).

**`02362ef8d`, "four OPEN measured defects the spec omitted, one of them a P0 about its own
premise"** (`+27 −8`). A history lens read the tracker instead of the document. **Single-stream
decode was sized and the spec said unmeasured**: #2694, open, measured 2026-08-24 with one client
driving both servers, streaming, at c=1 — decode `0.650×` and inter-token latency p50 9.68
against 6.29 ms, receipt `evidence/parity-http/findings.json` (draft `:853`). **Prefill is the
bigger gap and was not mentioned at all**: #2693, same run, `0.275×` with TTFT p50 35.66 against
9.81 ms — `evidence/parity-http/findings.json`, draft `:854`. **Its cause is measured too**:
#2697's `nsys` profile puts `cuLaunchKernel` at 0.7% of CUDA API time while synchronous copies and
device allocations account for 93.5% (draft `:855`) — the draft had advanced that as a hypothesis
after the profile had already settled it. And **the P0 that bears on §2.1**: #2753, batched CUDA
decode emits a constant token for every `m > 1`, never emits a stop token, and always runs to the
cap, so any aggregate above c=1 is throughput of garbage tokens (draft `:858`, `:864-867`). The
master carries all four as §9 rows #4, #3, #3 lever and #2, and makes the last of them a rule
rather than a paragraph: PP-26.

**Round 7 — `33a25cc1b`, "the structural cure for the trap, and a cheap statistic that reproduces
nothing"** (`+68 −4`). Four independent execution strategies were drafted against the tree and
scored by four judges on separate criteria; no strategy dominated, and the disagreement was the
useful output. Its lasting contribution is **P-4a, the structural cure, in two rules** (draft
`:230-244`): a ratchet is *seeded at a number already achieved*, because every rejection the
document had suffered came from a floor derived from measurements taken on the architecture the
fix removes; and *every gate ships a must-not-fire case beside its must-fire case, in the same
commit*. PP-23 is the worked example — its must-not-fire is the gx10 c=8 **aggregate** of 84.417
against a ceiling near 58 (`evidence/perf-gate-001-w1-gx10/receipt.r1.json`), which is correct
batching; had the companion been required, PP-23 could not have shipped in the form that declared
the epic's own receipts a harness bug. Both rules survive into the master as P-6 (first-PASS
arming) and P-7 (must-not-fire), and PP-29 is what makes P-7 checkable.

**Round 8 — `3f18bfda6`, "§9 #1 stops being a roofline ratio and becomes a named code path"**
(`+68 −4`). The synthesis of four strategies, planning against the tree, found the mechanism
behind the epic's headline anomaly: **gx10 prefills at exactly one decode step per prompt token.**
`crates/aprender-serve/src/gguf/cuda/generate_2.rs:284-288` sets `is_blackwell = cc >= 120` and
defaults such a device to the serial per-token prefill loop, so GB10 takes it. Splitting the c=1
samples by mode and fitting `wall_s = fixed_s + ms_per_token × generated_tokens` on the slow mode
reproduces it on all three replicates — intercepts 16.751 / 16.769 / 16.705 s over a 513-token
prompt, i.e. 32.65 / 32.69 / 32.56 ms per prompt token against decode steps of 32.11 / 31.85 /
32.25 ms, from `evidence/perf-gate-001-w1-gx10/samples.c1.r1.jsonl.gz` and its two siblings (draft
`:851`). The same round found **`prefill_multi_prompt` has no Blackwell guard** (draft `:857`), so
every `m >= 2` batch on that chip takes the path the tree documents as corrupting KV — which is
why "route `m = 1` through the batched path" is not the fix. Two things the master corrects about
this round: the fit **excludes 2 of the 30 samples per replicate** (those two completed 128 tokens
in 4.21–4.24 s and are unexplained, `evidence/perf-gate-001-w1-gx10/samples.c1.r1.jsonl.gz`), and
the 16.75 s cost was observed on the **blocking** transport only — the streaming c=1 gx10 run
already in the tree shows TTFT of 34.27 ms with no such cost
(`evidence/parity-http/findings.json`, `FINAL_quiet_box_parity`).
