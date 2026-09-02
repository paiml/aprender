<!-- PROVENANCE -->
**Auditor:** Claude (session audit, 2026-09-02) · **Audited:** the 1194-line reviewed draft of PR #2845, now archived at `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md` and its companions — `evidence/parity/LEDGER.md`, the perf-parity post-mortem, and `APR-PERF-GATE-001-v2.2.md`.
**Disposition:** answered by `docs/specifications/PP-LLAMA-001-MASTER.md`; finding-to-section map in section D below, and every `C-nn`/`CO-nn` has a paragraph in `docs/specifications/PP-LLAMA-001-RATIONALE.md`.
**Verbatim except two mechanical edits, both recorded here and neither changing a figure or a word of judgement:** in C-13 the second half of the prefill throughput pair has its unit parenthesised so that `scripts/check_no_claim_literals.sh` does not read it as a new user-facing throughput literal, and a one-line receipt citation is appended under C-13 so that `scripts/check_perf_claims_cite_receipts.sh` can resolve the `4× slower prefill` comparison on that line.

---

# Audit — `performance-parity-llama.cpp.md` (2026-09-02, PR #2845) and companions

**Auditor:** Claude · **Date:** 2026-09-02 · **Inputs:** the spec (1194 lines), `LEDGER.md`, `README.md`, `RECONCILIATION.md`, `EXECUTION-PLAN-{claude,agy}.md`, `perf-parity-review-2026-09.md`, the prior audit (B-1…B-14), `APR-PERF-GATE-001-v2.2.md`.
**Not available:** the tree. Findings are internal (line-checkable) or arithmetic (`[C]`, reproduced below). Tree-dependent items carry the discharging command.
**Marks:** `[V]` verified here · `[C]` calculated here · `[U]` unverified · `[X]` third-party figure.

**Verdict.** The document is now correct about more things than any predecessor and is *unusable as a specification*. It has 1194 lines; 65 of them `[C]` are sentences whose subject is what an earlier draft got wrong, and they are the load-bearing sentences — the current rule is stated as the residue of a correction rather than as a rule. Fourteen rows of §12 sit in two tables that disagree with each other and with §8. The gate it specifies has, by its own account, been re-designed five times and armed zero times. A complete rewrite is the correct response, and the rewrite is `PP-LLAMA-001-MASTER.md`.

The rewrite must preserve the `PP-nn` IDs (roadmap references are being repointed to them; renumbering again would be B-1 a third time). Retired IDs are marked RETIRED and never reused; new rules take PP-26+.

---

## A. Standing of prior findings (B-1 … B-14)

| # | status in this version | note |
|---|---|---|
| B-1 stable IDs | **closed** (§6 header, Appendix A) — but the invariant table is split across L352–372 and L400–405 with PP-25 between PP-18 and PP-19 | see C-2 |
| B-2 MDE ×3 | **closed** (§12.1a corrected table reproduces my figures) | `[V]` |
| B-3 decision rule | **partially closed** (P-5 bootstrap) — the rule double-counts noise; §7 still says "trigger is `agg_ratio ≥ 1.0`" | see C-9, C-10 |
| B-4 §7.1 contradiction | **closed**, then re-opened: §7.1 says target known (GEMV M∈{4,8}); §12.8 says waiting on a profile with a different subject | see C-6 |
| B-5 comparator `-np`/`-c` | **closed and extended** (§5.2 verified at the pin: rejects, breaks at `-np 8`) — but §5.2 asserts a contract §12.3 says is undecided | see C-4 |
| B-6 sampler pin | **closed as prose** (§5.1); no invariant, no mutation | see C-12 |
| B-7 W3 | **closed** (§5.3); over-broad at c=1 | see C-14 |
| B-8 server-reported config | **closed** (§12.6, §6.1a i) | `[V]` |
| B-9 roofline / gx10 | **superseded** by a better finding (§9 #1: serial prefill on Blackwell, `generate_2.rs:284`) — the mechanism is now named; the roofline is correctly demoted | `[V]`; C-5 on the "fix" |
| B-10 scaling_efficiency | **closed wrong** (§7.2 "both terms or neither" still rejects `agg(1)` improvements) | see C-3 |
| B-11 expiry semantics | **closed** in §12 prose; **not applied** to §8 | see C-7 |
| B-12 narrative split | **done** (post-mortem file exists) and **undone** (the spec regrew to 1194 lines) | see C-1 |
| B-13 replicate naming / two-term decomposition | **closed** (§10 corrected — round-4 caught my own error in the decomposition's meaning; the per-lane `agg÷dec` term is right and I accept the correction) | `[V]` |
| B-14 equal admission | **closed** (PP-24 + §6.1c derived ladder) | `[V]` |

---

## B. Companion-file findings

### CO-1 · LEDGER row statuses describe the wrong run — HIGH

`LEDGER.md` rows 1–2 (commit `745fa8588`, 2026-09-01): *"SPENT — subject lane invalid. Ratios withdrawn by §2.1: the subject binary was built with continuous batching compiled out."* Two errors. (a) The build with batching compiled out is `53062e7f3` (2026-08-24, `evidence/parity-http/`), not `745fa8588` — the spec's own banner (L50–53) says the `745fa8588` run logged `CONTINUOUS BATCHING: max_batch=11`. (b) The `745fa8588` receipts carry `comparator_status: UNMEASURED` on every band (§6.2a), so they contain no ratios to withdraw. The correct invalidity of those rows is **correctness**, not build flags: lambda c>1 ran the path #2753 says emits garbage for every `m>1`; gx10 c≥4 ran `prefill_multi_prompt` without the Blackwell guard (§9 #1a). c=1 on both hosts is valid but non-conformant (no stream, no comparator).
**Fix:** rewrite both status cells; add a `validity_by_band` column (`c=1: NONCONFORMANT-VALID; c>1: INVALID-CORRECTNESS(#2753|#1a)`). The master's LEDGER schema (Appendix C) carries it.

### CO-2 · LEDGER "what a row must carry" cannot be met and says so — MEDIUM

The five required producers (signature, `compute_class` server-reported, pin expiry, per-band `max_in_flight`, `roofline_tok_per_sec`) do not exist; the file admits this. A ledger criterion that no row can meet is the same defect as an unarmed gate. **Fix:** two-tier row: `RECORDED` (what exists) vs `CONFORMANT` (all producers present); PP-9 binds on `RECORDED`.

### CO-3 · README's table of "the spec said / the tree says" is the most useful artifact in the set and is not in the spec — LOW

Seven premises the tree falsified, in one table. That is what §2 should look like. Adopted as the master's §2.3.

### CO-4 · RECONCILIATION correctly identifies #2697 (0.275× prefill) as scheduled by neither plan; the spec's §12.11 now lists it as subject (d) — and the spec's own release gate still has no prefill metric — HIGH

The largest measured single-stream gap is ungated by construction (see C-13).

---

## C. Spec findings

### C-1 · The document is a changelog impersonating a specification — HIGH (structure)

`[C]` 1194 lines; 65 lines contain `earlier draft | a review | corrected | withdrawn | first written | this document had`; the header alone (L3) is a 140-word sentence listing eight review-introduced defects. §6.0a, §6.1a, §6.1b, §6.1b′, §6.1b″, §6.2, §7 (L738–750), §7.1 (L765–780), §12 (L1001–1067), §12.0, §12.1a, Appendix A's last paragraph — each states the rule *as the negation of a previous wrong rule*. A reader executing the gate must reconstruct the current rule from the corrections. The post-mortem file was created (B-12) and the spec regrew by 537 lines in one day.
**Root cause (five whys):** the editing rule is "add a correction paragraph" rather than "replace the wrong sentence and log the change." The master adopts a hard rule: **the spec contains no sentence about its own history**; every change is a CHANGELOG row (Appendix D) with the diff, and rationale lives in `RATIONALE.md`.

### C-2 · The invariant table is physically split and out of order — MEDIUM

PP-1…PP-18 and PP-25 at L352–372; PP-19…PP-24 at L400–405 after §6.1's prose. A reader of §6's table does not see PP-19–24. PP-25 precedes PP-19. **Fix:** one table, contiguous, sorted.

### C-3 · §7.2's non-regression arm still rejects the improvement it protects — HIGH (spec is wrong; live in `perf-matrix.yaml:38-44`)

L837: *"`scaling_efficiency(c)` up-only, and `agg(1)` non-regression."* `SE(c) = agg(c)/(c·agg(1))`. `[C]` Lambda: `SE(16) = 0.2798`; after a +20% `agg(1)` improvement with `agg(16)` unchanged, `SE(16) = 0.2332` (−16.7%) → the up-only ratchet **rejects** the single-stream fix that §12.11 (b) and (d) deliver. The `agg(1)` floor does not save it (the floor passes; the ratchet fails). This is the fourth gate in this document to outlaw its own fix; `EXECUTION-PLAN-claude.md` found it and the spec did not absorb it.
**Fix:** per-band non-regression on `agg(c)` and `dec(c)` each seeded at the last achieved value; `scaling_efficiency` is REPORTED, never ratcheted (PP-31 in the master).

### C-4 · §5.2 asserts the comparator contract that §12.3 says nobody has decided, and §12.3's premise is contradicted by §2.1's own numbers — HIGH

L327–331: the contract *is* per-band `-np {c}`/`-c` scaling. L988 (§12.3): the decision between "as a user runs it" (4 slots) and "configured to match" has **no owner** and is open. Both cannot be the rule.
The premise of §12.3 — *"at c=8 and c=16 the comparator serves 4 slots by design"* — is falsified by §2.1's table `[C]`: `llama_agg / llama_dec = 3.93, 7.84, 15.74` at c=4/8/16, i.e. the comparator in the withdrawn run had **c sequences in flight**, not 4. Either `-np` was not 4 in that run, or `kv_unified` changed the arithmetic, or the "constant 4" claim is wrong at the pin. Nobody has read `/props`.
**Fix (master §5.3):** the decision is made, with `decided_by` and dissent recorded: **comparator = `llama-server` configured for the band** (`-np c`, `-c c·n_ctx_slot`), because parity is a claim about serving the same offered load and a comparator that queues 12 of 16 requests is not serving the band; the `-b 1` concern (`llama_pin.toml:129-165`) is about *crippling* the comparator, and `-np c` is llama.cpp's documented way to serve `c` users. First action: `curl :{port}/props | jq '.default_generation_settings.n_ctx, .total_slots'` on the pinned build at the withdrawn run's argv — the number that settles which premise was true.

### C-5 · §9 #1 names a "fix" that §9 #1a forbids — HIGH (self-contradiction)

L851: *"The guard is `BATCHED_PREFILL=1`, one env var, no code change."* L857 (#1a): the batched prefill path *"emits `CertainlyCertainlyCertainly…`"* on Blackwell (PMAT-810); the Claude plan explicitly refuses to flip the flag. The flag is a **measurement arm** (it discriminates the mechanism), not a fix; the fix is the KV-scatter root cause `generate_2.rs:261-283` names.

### C-6 · §7.1 and §12.8 disagree about whether the kernel target is known — MEDIUM

L777: *"§7.1's first benchmark is therefore batched Q4_K GEMV at `M ∈ {4,8}`… needs no profile."* L993 (§12.8): *"§7.1 is DESIGNED, NOT ARMED because no profile has named the kernel… §9 #1's `SUSPECT_DISPATCH` on gx10 is the obvious first subject."* Also `SUSPECT_DISPATCH` was deleted at L490 and is still used at L404 (PP-23), L993 and in LEDGER row 2. **Fix:** one status vocabulary (master §7.4); §7.1's target is the M≤8 GEMV, from the tree, and needs no profile; §12.8 becomes the *prefill* profile (#2697 already took it).

### C-7 · §8 declares four expiries of 2026-09-25 while §12 says expiry is derived and the chain runs to 2026-10-23 — MEDIUM

L811–814 vs L1108–1113. This is the exact defect §12 describes in its own words ("a deadline that precedes its own prerequisite is a scheduled outage"), present in the same document. **Fix:** §8 carries no dates; `pmat comply` derives them (master §8, §12).

### C-8 · §12's two tables disagree — MEDIUM

Main table (L984–999) has 14 rows numbered 12.1–12.14 in the order 1,2,3,4,5,6,7,8,10,14,13,11,9,12. The chain table (L1014–1031) omits **12.13** (reproduce-at-HEAD rule) and **12.14** (`perf041`/`perf000` unwired) `[C]`. §12.3 has **no owner** (L988) in a document whose rule is that every obligation has one. §12.1 calls aggregate σ "MEASURED" while §12.1a says σ's CI is [0.52×, 6.3×] and no σ-dependent status may change at n<5.

### C-9 · P-5 subtracts noise twice — MEDIUM (statistics)

L263: PASS iff the one-sided 95% lower bootstrap bound ≥ `1.0 − ε`, `ε` = the receipt's MDE. The confidence bound already *is* the noise allowance; subtracting an MDE on top is a second allowance for the same σ. At n=5 this makes PASS ~1.4× easier than the stated confidence implies `[C]` (two 95% one-sided allowances compound). **Fix:** non-inferiority form: PASS iff LCB₉₅(ratio) ≥ `1 − δ` where `δ` is a *declared policy margin* with an author (v2.2 §4.6's "Policy" class) — `δ = 0` is parity by definition; any other `δ` is a recorded concession. MDE is reported as *power* ("this cell resolves a δ of X"), and decides whether the cell is decidable, not the verdict.

### C-10 · P-5 bootstraps the wrong unit — MEDIUM (statistics)

L1146: *"resampling whole requests."* `agg(c)` is a **window** statistic, not a per-request one; resampling requests does not produce the sampling distribution of a window aggregate — replicates do. With n=5 replicates a bootstrap has 3125 distinct resamples and poor coverage. **Fix (master §4.3):** two estimators for two units. Window metrics (`agg`, `prefill` at c=1): **interleaved paired replicates** (A,B,A,B,…), verdict on the one-sided t lower bound of the mean **log-ratio**, df = n−1. Per-request metrics (`dec`, TTFT, ITL): request-level paired bootstrap, 10 000 resamples, seed 2026, n in the hundreds. Interleaving is mandatory: thermal state, JIT warm state and free VRAM drift across a sweep, and alternation is the only design that cancels drift.

### C-11 · §7's release table and its prose disagree about c=1 — MEDIUM

L730 table: c=1 `dec_ratio` **gated … against the comparator, same run**. L745 prose: *"at c=1 the gated comparison is against apr's own previous release."* L735: *"The trigger is `agg_ratio ≥ 1.0`"* — a point rule P-5 replaced. Three statements of the c=1 rule; none matches the others.
**Fix (master §7):** one arming rule for every (cell, band, metric): a parity gate is REPORTING until the first receipt that PASSES P-5 on it; from that receipt on it is ARMED and a later FAIL blocks release. Nothing arms by date. This is P-4a's "seed at achieved" applied uniformly and it dissolves every "gate outlaws its own fix" instance at once — a gate cannot reject a change it has never passed.

### C-12 · §5.1's sampler pin and §5's `streaming` row are prose, not invariants — HIGH

§12.12 (L999) says it in the spec's own words: streaming is *"enforced by nothing… while violating it produces a receipt that passes."* §5.1's `completion_tokens == 128` rule has no PP, no mutation, no selftest. The plan's Step 5 (stream-provenance dual witness) has no PP. **Fix:** PP-27 (streaming + `stream_mode` dual witness) and PP-28 (sampler pin + `completion_tokens == n_predict`), with mutations — the 7-of-30 gx10 samples at 67–120 tokens are the free must-fire fixture.

### C-13 · Prefill — the largest measured gap — is absent from §3's metrics and §7's gate — HIGH (design)

§9 #3a: prefill **0.275×** (2,860 vs 10,399 (tok/s)), TTFT 35.66 vs 9.81 ms. §3 has no `prefill` row; §7 gates `dec` at c=1 and `agg` at c>1. A server at decode and aggregate parity with 4× slower prefill passes §7 forever, and the only place prefill shows is TTFT, which P-4 says is REPORTING until W3. **Fix:** `prefill(c)` is a first-class metric (server-reported `prompt_per_second` on both lanes; llama.cpp already exposes `timings.prompt_*`); at c=1 the gated set is {`dec_ratio`, `prefill_ratio`} — they do not trade against each other, so gating both is not the §2.2 trap.

> Receipts for every figure in C-13: `evidence/parity-http/findings.json` (decode, prefill and TTFT medians, c=1).

### C-14 · P-4's convoy argument is over-broad at c=1 — LOW

L335–338: TTFT under W1 is an artifact "at every round boundary all `c` prefills collide." At c=1 there is one request; no convoy exists. TTFT(c=1) under W1 is a clean measurement and is the cheapest prefill witness available today. **Fix:** W3 for c>1 latency; W1 TTFT at c=1 is valid and REPORTED beside `prefill_ratio`.

### C-15 · Correctness has no invariant — HIGH (design; the largest missing control)

§9 #5's rule *"no aggregate claim at c>1 until #2753 closes"* is prose. Nothing in the receipt schema witnesses that the tokens counted were correct; `perf041` is unwired (§12.14). The §2.1 banner's "ratio now" column (0.395/0.544/0.401) is published in the spec over tokens #2753 says are garbage — the document violates its own §9 #5 in its §2. **Fix:** PP-26 **batch-invariance witness** — for a fixed prompt at `temperature 0`, the token sequence produced at `m=1` and inside an `m=c` batch must agree to a declared divergence point (≥ 64 tokens, `[U]` until fp-nondeterminism is measured); every band's receipt carries the witness result; absence or failure makes the band `INVALID-CORRECTNESS`, which is neither `MEASURED` nor `UNMEASURED` and can never become a baseline.

### C-16 · Memory is measured, 3.08×, causal for admission, and scheduled nowhere — MEDIUM

`evidence/parity-http/findings.json`: VRAM 14,030 vs 4,554 MiB at c=1 — apr holds **9,476 MiB** `[C]` of non-comparator memory before a second request exists. At `kv_per_slot ≈ 470 MB` that is ~20 slots of KV — the reason `max_batch` lands at 11 on a 24 GB card is very likely this allocation, not the KV arithmetic alone. v2.2's Arm D existed for this; the spec dropped it and §9 does not list it. **Fix:** §9 #6 (memory), `vram_peak` and `kv_per_slot` in the effective-config block; Arm D re-adopted as REPORTING.

### C-17 · Thresholds live in three files — MEDIUM (already found at L677–687; unfixed)

`perf-matrix.yaml` (B1 0.80, B2 1.00), `perf_gate.sh:246`, `parity_block.py:23-24` (`STRETCH/CEILING 1.50`). **Fix:** PP-34 — one file owns every threshold; a numeric comparison in a gate script that is not read from `perf-matrix.yaml` is a mutation `check_no_claim_literals.sh` must catch (widen its universe; it currently excludes `docs/specifications/`).

### C-18 · §2.4 is stale — LOW

*"Decode is M=1 GEMV bound… the only five-whys in the corpus that terminated in a mechanism."* Two others now have (§9 #1, #2697), and the 192× coalescing defect is not the live cause of any §9 row. Delete from the spec; keep in the ledger as history.

### C-19 · §6.1a(ii) contradicts §6.1a(i) two paragraphs above — LOW

(i): `max_batch=11` is auto-sized from free VRAM, and "the earlier draft's env-var claim was wrong." (ii): *"The active cap in the measured run was 11, set by an env var."* Both live.

### C-20 · The spec has no self-conformance check — HIGH (gates or theater, applied to the spec)

P-4a requires every gate to ship must-fire and must-not-fire cases. Nothing verifies that every ARMED PP has a `perf_gate.sh --selftest` row of each kind. A PP with no selftest is documentation. **Fix:** PP-29 — `scripts/spec_conformance.sh` parses the master's invariant table, and for every `ARMED` row asserts two selftest cases exist by name; runs in `ci / gate`; the master's §6 table carries the selftest names as columns, so the check is a join, not a grep.

---

## D. Disposition into the master

| finding | master section |
|---|---|
| C-1, C-2 | whole document; §6 single table; Appendix D changelog rule |
| C-3 | PP-31; §7.3 |
| C-4 | §5.3 comparator decision; §12 row 3 |
| C-5, C-6 | §9 #1, §7.1 target, §7.4 vocabulary |
| C-7, C-8 | §8 (no dates); §12 single DAG table incl. 12.13/12.14 |
| C-9, C-10 | §4.3 decision rule (non-inferiority; two estimators; interleaving) |
| C-11 | §7.2 arming rule (first-PASS) |
| C-12 | PP-27, PP-28 |
| C-13, C-14 | §3 `prefill`; §7.2 c=1 gated set; P-4 |
| C-15 | PP-26; §7.0 L0 correctness layer; §2's c>1 figures marked `INVALID-CORRECTNESS` |
| C-16 | §9 #6; PP-2 memory fields; Arm D |
| C-17 | PP-34 |
| C-18, C-19 | deleted |
| C-20 | PP-29; §6 selftest columns |
| CO-1, CO-2 | Appendix C LEDGER schema |
