# PERF-062 — the band ladder re-run, and what it actually says

Re-run of the PERF-059 band ladder (#2789) on an RTX 4090 (sm_89), 1.5B Q4_K_M,
asking the question APR-PERF-GATE-001 needs answered before any cell is measured:
**is there now ONE route configuration that is correct across bands {1,4,8,16}?**

**Short answer: no — but the reason has changed completely, and two of the three
remaining failures are defects in the ladder, not in the CUDA path.**

---

## 0. The premise of the task is false: #2776 has NOT merged

The work item said "ordered streams should be on main." They are not.

| Ticket | PR | State at time of run |
|---|---|---|
| #2770 cuBLAS Q6_K dequant | #2783 | **MERGED** `c6a6dfce3`, 2026-08-30T08:28Z |
| #2767 ordered streams | #2776 | **OPEN**, `mergeStateStatus: BLOCKED`, branch `feat/v14-streamdefault` |

`origin/main` at run time was `a866988e4`. So a ladder run against plain main would
have measured the *racy* stream default and told us nothing new.

### What I ran instead, and why it is equivalent

I ran **main `a866988e4` with `APR_STREAM_LEGACY=1`**, which is content-verified
equivalent to #2776's *default*. This is not an assumption; it is the whole diff:

```
$ git diff --stat origin/main...origin/feat/v14-streamdefault
 .github/workflows/cuda-nightly.yml                  |  35 +++   CI only
 contracts/continuous-batching-v1.yaml               |  22 +-    contract only
 crates/aprender-gpu/src/driver/memory/tests.rs      |  11 +-    tests only
 crates/aprender-gpu/src/driver/memory/transfer.rs   |  69 ++-   see below
 crates/aprender-gpu/src/driver/stream.rs            |  89 ++-   the polarity flip
 crates/aprender-serve/src/cuda/executor/mod.rs      |  14 +-    COMMENT-ONLY (verified)
 crates/aprender-serve/tests/falsify_..._2767.rs     | 281 +++   new test
```

- **`stream.rs`** is a *polarity flip of one knob*. main: `APR_STREAM_LEGACY=1`
  selects `CU_STREAM_DEFAULT`, default is `CU_STREAM_NON_BLOCKING`. #2776: default
  is `CU_STREAM_DEFAULT`, `APR_STREAM_NONBLOCKING=1` selects the racy one.
  Setting `APR_STREAM_LEGACY=1` on main reaches the identical `cuStreamCreate` flag.
- **`transfer.rs`** adds exactly one new runtime path, `APR_ORD9_DRAIN_SKIP`, which
  **defaults off**. Non-comment changed lines, in full:
  `if ord9_drain_disabled() { return Ok(()); }` + the `OnceLock` accessor.
- **`executor/mod.rs`** has **zero** non-comment changed lines (mechanically checked).

So the only runtime difference between "#2776 default" and what I ran is a knob that
is off in both. **I did not build the branch** — the box is at 100% disk (see §6) —
and I state that plainly rather than implying I ran the PR.

---

## 1. Binary provenance, proven by content not by `--version`

```
path    /tmp/.../scratchpad/y5-pin/apr.a866988e4   (copied and pinned; the source
        target dir was being concurrently rewritten by another agent's cargo test)
sha256  40eb0a8c8df48b35...
--version                       apr 0.64.0 (a866988e4)   == origin/main
strings -a | grep -c cublasGemmEx           3   <- real CUDA build
strings -a | grep -c APR_STREAM_LEGACY      1   <- main's knob PRESENT
strings -a | grep -c APR_STREAM_NONBLOCKING 0   <- #2776's knob ABSENT
strings -a | grep -c APR_ORD9_DRAIN_SKIP    0   <- #2776's knob ABSENT
strings -a | grep -c FP8_DECODE             1
strings -a | grep -c CUBLAS_GEMM_THRESHOLD  1
```

The two absent strings are positive content-proof that this binary is main
*without* #2776, which is what makes the equivalence argument above checkable
rather than rhetorical.

**Every run below additionally confirmed, per run, from the server log:**
- `gpu-layers: requested=all resolved=28 total=28 (backend=cuda)` — resolved, not requested (I-2)
- `CB-006-OUT` line count = **0** — `APR_LAYER_TRACE` genuinely off (#2764)
- harness self-test (`--prove-can-fail`) passed **both polarities**: the verdict
  correctly PASSes identical input and FAILs on diverged / empty / zero / truncated.

**Protocol statement.** This is **not** the spec's §4.4 measurement protocol. §4.4
is the throughput protocol (closed-loop client, `2×c` warmup, `max(30, 8×c)` samples,
60 s floor, bootstrap CIs). This is the PERF-059 *correctness* ladder plus a
self-consistency extension — a precondition check that must pass **before** §4.4
runs at all. No tok/s is reported here and no W1 receipt is recorded (I-9).

---

## 2. The ladder at HEAD — strict PERF-059 verdict

`scripts/perf059_band_ladder.sh` verdict: every slot in band *c* must be
**string-identical** to a single c=1 reference. Model 1.5B Q4_K_M, 40 max tokens,
one prompt, bands 1/2/4/6/8/16 (m≥6 included because FP8 fires at m≥5).

| Arm | `STREAM_LEGACY` | `CUBLAS_THR` | `FP8_DECODE` | c=1 | c=2 | c=4 | c=6 | c=8 | c=16 |
|---|---|---|---|---|---|---|---|---|---|
| A0 (main default, racy) | 0 | def | def | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| A1 (**= #2776 default**) | 1 | def | def | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| A2 (#2789 row 4) | 1 | 32 | def | FAIL | PASS | PASS | FAIL | FAIL | FAIL |
| A3 (#2789 row 5) | 1 | 32 | 0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| A4 (#2789 row 6) | 1 | def | 0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

**Every arm fails band 1.** A band-1 failure means "one c=1 request disagreed with
another c=1 request." That is internally contradictory with A2 passing bands 2 and 4,
and it is the thread that unravels the instrument.

---

## 3. Why the strict verdict is not sound — three defects in the ladder

### 3.1 The c=1 reference is the *un-warmed first request*, and it is the outlier

The ladder takes its reference from the very first request after server start.
Sequential c=1 replicates, one server, greedy, 15 per arm (10 before the bands,
5 after, to separate warm-up from drift). md5 prefixes, **every replicate**:

| Arm | pre (10, in order) | post (5) | distinct | mode |
|---|---|---|---|---|
| A0 | `94b918ea` `fc42d43e`×9 | `fc42d43e`×5 | 2 | 14/15 |
| A1 | `94b918ea` `fc42d43e`×9 | `fc42d43e`×5 | 2 | 14/15 |
| A2 | `94b918ea` `fc42d43e`×9 | `fc42d43e`×5 | 2 | 14/15 |
| A3 | `a7ad49d4` `08912486`×9 | `08912486`×5 | 2 | 14/15 |
| A4 | `94b918ea` `fc42d43e`×9 | `fc42d43e`×5 | 2 | 14/15 |

**In 5 of 5 server starts the first request differs from every one of the following
14, and the following 14 are unanimous.** This is a warm-up transient, not
nondeterminism: the post-band replicates are identical to the pre-band steady state,
so there is no drift.

Spec **§4.4.2 already legislates this away** — "Warmup requests `2 × c`, discarded,
not written to the receipt" plus a 5 s quiesce. The ladder script does no warmup, so
it systematically anchors its reference on the one sample §4.4 would have thrown out.
**This alone accounts for every band-1 FAIL in the table above.**

### 3.2 The band label is not the batch size — bands 2 and 4 never formed

From `[PMAT-044] Batch m=` in the server logs, batches actually formed:

| Arm | batches formed (count × m) |
|---|---|
| A0 | 15×m=1, 1×m=2, 1×m=4, 1×m=6, 1×m=8, 1×m=16 |
| A1 | 18×m=1, 1×m=3, 1×m=6, 1×m=8, 1×m=16 |
| A2 | 18×m=1, 1×m=3, 1×m=6, 1×m=8, 1×m=16 |
| A3 | 18×m=1, 1×m=3, 1×m=6, 1×m=8, 1×m=15 |
| A4 | 18×m=1, 1×m=3, 1×m=6, 1×m=8, 1×m=16 |

Under every ordered arm, the two c=2 requests **ran as two separate m=1 batches** and
c=4 formed **m=3**. So **A2's "PASS at c=2" is vacuous — it never exercised the
batched path**, and c=4 is a partial batch whose composition varies run to run. A
ladder row that reports a band it did not form is a green that proves nothing.

### 3.3 The verdict cannot distinguish "corrupted" from "a different valid answer"

Exact string equality collapses two very different outcomes into one FAIL. They are
not the same failure and only one of them is a CUDA defect — see §4.

---

## 4. The self-consistency ladder — what ordered streams actually fixed

Re-measured with the reference taken as the **mode of 15 warmed c=1 replicates**, and
each band scored on (a) distinct completions within the band and (b) whether the
band's mode equals the c=1 mode. Every replicate hash retained.

### A0 — main's current default (racy, `CU_STREAM_NON_BLOCKING`)

| band | distinct/total | self-consistent | vs c=1 mode |
|---|---|---|---|
| 2 | 1/2 | yes | DIFFERENT |
| 4 | 2/4 | **no** | DIFFERENT |
| 6 | **3/6** | **no** | DIFFERENT |
| 8 | **4/8** | **no** | DIFFERENT |
| 16 | **5/16** | **no** | DIFFERENT |

c=16 replicates: `4818b621 ed4dd642 4818b621 2fbf99f4 ed4dd642 09745a44 2fbf99f4
ec351630 09745a44 09745a44 ec351630 09745a44 09745a44 09745a44 09745a44 09745a44`.
The text is genuine garbage, e.g. `` 1:=\).\0).\).AI).00=1).".().).Ắ).IZ(isis) ``.

### A1 — the #2776 default (`CU_STREAM_DEFAULT`)

| band | distinct/total | self-consistent | vs c=1 mode |
|---|---|---|---|
| 2 | 1/2 | yes | SAME |
| 4 | 2/4 | no | DIFFERENT |
| 6 | **1/6** | **yes** | DIFFERENT |
| 8 | **1/8** | **yes** | DIFFERENT |
| 16 | **1/16** | **yes** | DIFFERENT |

A2, A3 and A4 have the identical shape (A3's m≥6 answer is `94b918ea`, A1/A2/A4's is
`1bcbabf3`); full per-replicate hashes in the run log.

**This is the headline. Ordered streams do what #2767 claims:** at c=16 the racy
default returns **5 distinct answers out of 16**, mostly garbage; the ordered default
returns **1 out of 16**, and it is coherent, correct Python. #2789's `GARBAGE` cells
are gone. What remains is not corruption — it is a *different valid completion*.

---

## 5. Is "different from c=1" a defect? Not on the evidence — it is a near-tie

Per the #2359 scar (four neighbouring prompts inverted a "GPU correctness defect"
into "the gate sampled a near-tie"), one prompt is an anecdote. Six prompts, A1
config, c=1 mode of 5 replicates vs the mode of a c=8 band:

| prompt | c=1 mode (n=5) | band-8 mode | verdict | b8 distinct |
|---|---|---|---|---|
| Python sum of a list | `fc42d43e` (4/5) | `1bcbabf3` (7/8) | **DIFFERENT** | 2 |
| Explain a hash table | `c68cd8e1` (5/5) | `c68cd8e1` (8/8) | SAME | 1 |
| Capital of France | `4804f8d3` (5/5) | `4804f8d3` (8/8) | SAME | 1 |
| Rust reverse a string | `9585e9e0` (5/5) | `37a97e76` (7/8) | **DIFFERENT** | 2 |
| Three primes > 100 | `2e80aea2` (5/5) | `2e80aea2` (8/8) | SAME | 1 |
| HTTP 404 | `e2870f22` (5/5) | `e2870f22` (8/8) | SAME | 1 |

**4 of 6 prompts agree exactly between c=1 and c=8**, 8/8 self-consistent.

The two that differ are the two open-ended code-generation prompts, and both
alternatives are coherent and correct:

```
c=1 : ```python\ndef sum_list(numbers):\n    # Initialize the sum to 0\n    total_sum = 0 ...
c=8 : ```python\ndef sum_list(numbers):\n    return sum(numbers)\n```
```

That is the signature of an **early argmax near-tie** flipped by floating-point
reduction order between an m=1 and an m=8 GEMM — not of a corrupted batched path.
The closed-form factual prompts, which have no near-tie, are bit-identical across
bands. **This is evidence, not proof**; a logprob-margin probe at the first divergent
position would settle it, and has not been run.

### Incidental: `FP8_DECODE` changes the **c=1** answer, but only with `THR=32`

A3 (`THR=32, FP8=0`) is the only arm whose c=1 mode differs (`08912486` vs
`fc42d43e`); A4 (`THR=def, FP8=0`) does not move. Since `fp8_will_fire =
gpu_profile.fp8_decode && m >= 5` (`batched_ffn.rs:111`, `batched_qkv.rs:119`,
`cublas_prefill/attention.rs:1021` at `a866988e4`), and a **prefill** of this ~20-token
prompt has m≈20 ≥ 5, FP8 fires at c=1 through prefill. `CUBLAS_GEMM_THRESHOLD=32`
simultaneously pushes that same m≈20 prefill **off** the cuBLAS route (default
threshold is 4). The interaction is consistent with that reading but was not
isolated — flagged, not concluded.

---

## 6. Host conditions

`load average 13.94` at start (over-subscription is provisioned; not a diagnosis).
**Disk `/` was at 100%, 3.6 GB → 561 MB free during the run**, driven by two
concurrent agents' cargo target dirs (45 GB and 39 GB). This is why the #2776 branch
was not built. Reported, not resolved.

---

## 7. The plain answer

> **Is there now ONE configuration correct across bands 1/4/8/16?**

**Under the ladder as written: no — no arm passes.** But the ladder as written cannot
answer the question, because two of its three failure modes are its own:

1. **band 1 fails in 5/5 arms** because the reference is the un-warmed first request,
   which §4.4.2 mandates discarding (§3.1). Fix the harness and this disappears.
2. **bands 2 and 4 never form m=2/m=4** (§3.2), so those cells report on batches that
   did not happen.
3. **bands 6/8/16 genuinely differ from c=1** — but under ordered streams they are
   *1-distinct-out-of-16, coherent*, and 4 of 6 prompts show no difference at all.

**What is now settled:** ordered streams (#2776, reached here as
`APR_STREAM_LEGACY=1`) removes the corruption. #2789's `GARBAGE` cells do not
reproduce at HEAD under any ordered arm. The ladder collapsed *most* of the way toward
one configuration, and the surviving disagreement is plausibly a near-tie rather than
a defect — the opposite end of the severity scale from where #2789 left it.

**What is not settled, and blocks Arm A:**

- **#2776 must land.** Everything above holds only with `APR_STREAM_LEGACY=1`;
  main's *default* is still A0, which is unusable at every band.
- **The ladder needs §4.4.2 warmup and a formed-batch assertion** before its verdict
  can gate anything. Right now it cannot tell a cold first request from a defect, and
  it reports on bands it did not form.
- **The near-tie hypothesis needs a logprob-margin probe** to become a finding. If it
  holds, the ladder's verdict should be "self-consistent within the band **and**
  agreeing with c=1 wherever the argmax margin exceeds the m=1↔m=8 numeric delta" —
  not raw string equality, which no correct batched implementation can satisfy on an
  open-ended prompt.

**No W1 receipt is recorded** (I-9: a band may not be re-run to green, and no cell may
be measured until the ladder is clean). The ladder is not yet clean, and two of the
three reasons are in the harness.

---

## 8. Reproduce

```bash
. scripts/apr_bin.sh || exit 1        # or pin an explicit --target-dir build
APR_STREAM_LEGACY=1 scripts/perf059_band_ladder.sh \
    --model /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --bands 1,2,4,6,8,16 --port 18802 --max-tokens 40
```

Note `scripts/perf059_band_ladder.sh` is **not on `origin/main`** — it exists only on
`origin/feat/y1-7bgarbage` (`6fcb879b2`). A ladder that gates a release should not
live on one unmerged branch.
