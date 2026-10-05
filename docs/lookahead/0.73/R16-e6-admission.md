# Row 16: E6 admission through `rex admit`

Read at origin/main 11f844a772: `contracts/rex-cell-admission-v1.yaml` and
`crates/aprender-review-experiment/src/admission.rs`. No build.

## What `rex admit` checks
- One JSONL row per PRM cell, six cells (`admission.rs:27-58`). A missing or doubled
  row rejects the file (FALSIFY-RCA-001).
- A row is `Admitted { parity }`, `Refused { removed_by }` or `NotRun { reason }`.
  Admitted needs an oracle, a threshold basis, a 64-hex parity receipt sha and a finite
  `cosine >= threshold` (`admission.rs:161-171`, FALSIFY-RCA-002).
- The oracle constant is `llama.cpp@d1d3c3396` (`admission.rs:16`).
- No Admitted cell raises S-7, and `rex admission-check` exits 10 (FALSIFY-RCA-003).

## Finding A16-1: the same cell ids name different hardware
| id | 0.73 R1 (E1/E2 receipts) | PRM (`admission.rs` CELLS) | same cell? |
|---|---|---|---|
| C0 | lambda, CUDA sm_89 | none | none in PRM |
| C1 | intel, wgpu, AMD GPU 0 | intel, wgpu | yes, if PRM's wgpu is GPU 0 [U] |
| C2 | intel, wgpu, AMD GPU 1 | intel, cpu | **no** |
| C3 | mini, wgpu/Metal | lambda-labs, cpu | **no** |
| C4 | gx10, aarch64 CPU | gx10, cuda | **no** |
| C5 | gx10, CUDA sm_121 | none (C5a mini cpu, C5b mini metal) | **no** |
Row 16's own text ("C1 intel-wgpu and C5b mini-metal") uses PRM's ids, while every
E1/E2 receipt uses R1's. An E1 PASS receipt for R1 C4 (gx10 CPU) read as PRM C4 would
admit gx10 CUDA. Fix: the E6 step joins receipts to PRM rows on (host, backend), never
on the id, and FALSIFY-E6-ID-001 plants an R1-C4 receipt for PRM-C4 and must reject it.
Renaming either table is the other fix, and it is a PRM-001 owner call.

## Finding A16-2: PRM C5b says `metal`, and there is no native Metal backend
The E4 census (`E4-census.md`, G1; `aprender-compute/src/registry/mod.rs:568-573`)
reads that `--backend metal` is always unavailable and a forced run is refused. Apple
GPUs run only as wgpu on the Metal transport, which is R1 C3. So PRM C5b can only be
NotRun or Refused, unless its `metal` means wgpu on Metal. This is RQ-10 again, from
the PRM side. Under RQ-10's default, C5b's row is joined to R1 C3's receipts, and the
CELLS entry should say `wgpu` (host mini) so the join on (host, backend) holds.

## Finding A16-3, evidence for RQ-3: PRM admits on parity alone
RQ-3 asks whether PRM-001 agrees that an E6 cell needs E1 PASS + E2 PASS receipts.
Read from the code: PRM admits on one parity receipt with a cosine at or above a
cited threshold. No speed (E2) receipt enters `Status::Admitted`. Two more
differences:
- PRM's oracle is llama.cpp@d1d3c3396. E1's reference is apr's own CPU forward at
  `fp32_act` (RQ-5). A cosine against one is not a cosine against the other.
- PRM keeps one threshold per row with its basis. E1 uses 0.995 (BPM thresholds).
So the two rules differ. E6 can still be stricter than PRM without PRM changing:
E6-admissible(c) = PRM Admitted(c) AND an E1 PASS receipt for the same (host,
backend) AND an E2 PASS receipt for it. That reading needs no PRM edit; whether the
Admitted row's `receipt_sha256` must be the E1 receipt (oracle apr fp32_act) or a
separate llama.cpp cosine receipt is the question left for PRM-001's owner.

## Draft for the row-16 ticket
1. An `e6-admit` join (bash, per C301) that reads the PRM admission file and the
   E1/E2 receipts, joins on (host, backend), and writes E6-admissible cells.
2. FALSIFY-E6-ID-001 (A16-1), FALSIFY-E6-002: an Admitted PRM row with no E2 PASS
   receipt is not E6-admissible; FALSIFY-E6-003: an E1 receipt whose cell is HYBRID
   (RQ-4) is reported HYBRID, never GPU.
3. Ranked as today: after the rows that produce C1 and C3 receipts.

## Open
- RQ-10 (Metal meaning) decides A16-2. A (quorum, 2026-10-05, degraded: same-family): (a) wgpu on Metal, 2-0; and C5b should say `wgpu`, 2-0, which is a PRM-001 owner call to make.
- RQ-3 stays conditional: A16-3 is the evidence to hand PRM-001's owner.
