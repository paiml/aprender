# P1 — backend-parity cell receipt: schema and planted fixtures (draft, la-73, 2026-10-03)

P1 is the first bundle in `falsifier-landing-map.md`. It holds 19 falsifiers and needs no GPU, no model and no aarch64 host.
This is a spec only. No code is written and no ticket is minted (C277). Citations are at origin/main 316dee2cd4 unless a line says otherwise.

## 1. Reuse first: one validator per artifact family
`contracts/parity-receipt-v2.yaml` [V] already governs **logit** parity records (`apr-parity-receipt/v2`, `evidence/parity/**`).
Throughput parity is a different family (`scripts/check_parity_receipt.sh` over `bench_receipt.py --parity`), and v2
refuses to fold the two. P1 follows that rule:

- **Leg A and leg B are not new shapes.** Each leg is an existing `apr-parity-receipt/v2` record, referenced by path and sha256:
  - **Leg A** compares the apr backend with the apr CPU. It is a v2 `SelfComparedReceipt`: `comparator.kind = self`, a `reason` is required, and no `comparatorSha` (PRC-INV-002/003).
  - **Leg B** compares the apr CPU with llama.cpp. It is a v2 `OracleComparedReceipt`: `comparator.kind = llama_cpp`, and `comparatorSha` is required (PRC-INV-004). BPM-004 adds only the VALUE pin.
- **E2 is a throughput-family record**, referenced the same way and validated by its own checker.
- **The new family is the CELL receipt** (`apr-bpm-cell-receipt/v1`) under `evidence/bpm/**`. It binds the three records and carries the fields that only exist per cell. These are the cell, the adapter fields, `cpu_ref_path`, the per-position cosines, `op_placement`, `kernel_path` and the captured stderr lines.
- **Thresholds are resolved, never typed into a shape** (PRC-INV-005). The values 0.995, 0.98 and 0.5 go in `evidence/parity/thresholds.yaml` under a `bpm:` key. The checker reads them and fails with `thresholdSourceMissing` when the key is absent.
- **Reach is pinned** (PRC-INV-006). `evidence/bpm/EXPECTED_RECEIPTS` holds the count from an independent predicate. A mismatch is `Unknown{WrongCorpus}` with exit 2.

Where it lands: a new extractor `crates/aprender-contracts/src/ontology/extract/bpm_cell_receipt.rs`, next to
`parity_receipt.rs` [V]. Its case table is run by `cargo test -p aprender-contracts --lib`, which is already in the
pre-push checklist and in CI.

## 2. Verdicts
`Pass | Fail(rule) | Refused(rule) | NotRun(reason) | Unknown(WrongCorpus)`.
- **Fail:** the receipt is admissible, and it measured a defect.
- **Refused:** the receipt cannot be judged, because it is malformed, inconsistent or self-compared. It never counts toward E6.
- **NotRun:** the cell did not run what it claims to have run.
- No path from a missing field leads to Pass.

## 3. Cell receipt fields (beyond the referenced records)
| Field | Type | Read by |
|---|---|---|
| `cell` | C0..C5 | all |
| `leg_a`, `leg_b` | {record path, record sha256, `cpu_ref_path`, `prompt_set_sha256`, `n`, `model_sha256`, `cos[p][t]` (prompt × position, position 0 = prefill-final, then ≥ 8 decode)} | 001, 002, 008-011, 013 |
| `e2` | {record path, sha256, `n_gen_apr`, `n_gen_llamacpp`, decode/prefill {r, ci_lo, ci_hi, n}} | 005, 015 |
| `llamacpp_sha` | string | 004 |
| `adapter` | {name, `device_type` ∈ wgpu DeviceType, banner line} | 003, 006, 014 |
| `kernel_path` | {backend: map tensor → path, reference: map tensor → path} | 012 |
| `op_placement` | map op → {device, host}; keys = DECODE_OPS | 016, WGF-005, WGF-009 |
| `gemv_backend` | map qtype → {wgpu, cpu} | 016 |
| `hybrid` | bool | WGF-005 |
| `stderr_lines` | captured lines of the serving process; `stderr_captured: true` | 003, 017, R4-003 |
| `used_gpu`, `no_gpu_flag` | bool | R4-003 |

## 3a. Alignment with the OBS stack (found 2026-10-03; unmerged, #4487 / #4574)
`apr-obs-row-identity-v1` and `apr-kernel-path-v1` are not on main. They sit on the unmerged OBS-00 commit 96d2fe4aa1,
which `origin/a01/4574-obs15-kernel-path` (b1244f6fb7) carries with a Rust checker in
`crates/aprender-qa-report/src/obs_kernel_path.rs` (`check_kernel_path`, `kernel_diff`, `admit_kreg_entry`).
- **Identity.** The cell receipt carries the OBS IDENTITY block unchanged:
  - the fields are schema, ts, host, apr_version, apr_tag, crate_tarball_sha256, binary_sha256, build_identity, model_id, model_sha256, backend and request_id, plus `gpu_proof`;
  - P1 calls the OBS-01 shared lint and does not copy the list;
  - `model_sha256` there is the same field BPM-011 compares across the three runs.
- **kernel_path.** The `kernel_path` field of §3 uses the `apr-kernel-path-v1` shape, one per run (backend, reference). Each entry is `{op, kernel_id, qtype, layout, arch, shape_class, precision}`.
  - R3's `kernel_path(k)` string, e.g. `q4k-q8k/neon-sdot`, becomes `kernel_id`, with `arch = aarch64`.
  - BPM-012 compares `(op, shape_class) -> kernel_id` between the backend and reference paths, using `kernel_diff` semantics. An empty diff on a CPU cell is REFUSED.
- **Conflict to resolve.** `apr-kernel-path-v1` `trace_cut` says "CPU rows name no GPU kernel and are outside the rule", so a CPU row may carry `kernel_path = null`.
  - BPM-012 needs a non-null kernel_path on every CPU cell, C4 above all, or the self-comparison guard has nothing to read.
  - P1 adds this as its own REQUIRE for cell receipts. It does not change the OBS rule, which governs perf-ledger rows.
  - Flag it to the OBS-15 owner when the stack merges; do not edit their branch.
- **KREG link.** OBS `kreg_admission` admits a kernel only with a `parity = pass` receipt for the same (arch, backend, kernel_id). A BPM cell receipt with E1 PASS is that receipt. It names its kernel_ids, so one artifact serves E1 and E5.
- **Order.** P1 depends on the OBS stack merging. If it has not merged when P1 is minted, P1 lands the cell checker with the identity check as `NotRun(identity lint absent)`, never as a pass.

## 4. Planted fixtures (`tests/fixtures/bpm/`)
Every fixture is `base.json` plus one edit. `base.json` is a C0 CUDA cell with:
- both legs ≥ 0.999 at every position, and E2 r = 0.7 with ci_lo = 0.6;
- equal n_gen, the ruled pin, `device_type` DiscreteGpu and the CUDA banner;
- a "F2 guard: GPU matches" line, different kernel paths and a total `op_placement` on device.

**Control:** `base.json` must be `Pass`. A checker that refuses everything fails this row.

| Fixture | Edit to base | Expected | Falsifier |
|---|---|---|---|
| f001 | leg_a: 15 prompts 0.999, 1 prompt 0.990 (median 0.999) | Fail(E1 min) | BPM-001 |
| f002 | leg_a = leg_b = 0.99 everywhere | Fail(composed 0.960 < 0.98) | BPM-002 |
| f003 | adapter.name = llvmpipe, leg_a 1.0 | Fail(FALLBACK) | BPM-003 |
| f004 | llamacpp_sha = 39173bcac | Fail(pin) | BPM-004 |
| f005 | e2.decode r 0.55, ci_lo 0.45 | Fail(E2 ci_lo) | BPM-005 |
| f006 | cell C1, Q8_0, real adapter, no wgpu forward line | NotRun | BPM-006 |
| f008a | leg_a.cpu_ref_path fp32_act, leg_b q8k_act | Refused | BPM-008 |
| f008b | cpu_ref_path absent on one leg | Refused | BPM-008 |
| f008c | both legs q8k_act; pin `bpm.cpu_ref_path` = fp32_act (provisional, RQ-5) | Refused | BPM-008 |
| f008d | both legs fp32_act, but the reference `kernel_path` has blk.0.ffn_gate Q4_K at precision q8k (the scratch forward) | Refused | BPM-008 |
| f009 | one cosine = NaN (as a JSON string "NaN"; JSON has no NaN literal, so the extractor must parse it and not drop it) | Refused | BPM-009 |
| f010 | leg_b.prompt_set_sha256 differs; and a second fixture with n 16 vs 15 | Refused | BPM-010 |
| f011 | model_sha256 differs between the three records | Refused | BPM-011 |
| f012 | cell C4, kernel_path backend = reference for blk.0.ffn_down (q6k-f32/scalar both), leg_a 1.0 | Refused | BPM-012 |
| f013 | prefill-final 1.0, decode position 4 = 0.90; and a second fixture with only 7 decode positions | Fail / Refused | BPM-013 |
| f014 | adapter "FooSoft Renderer", device_type Cpu | Fail(FALLBACK) | BPM-014 |
| f015 | n_gen_apr 37, n_gen_llamacpp 128 | Refused | BPM-015 |
| f016 | cell C1, Q8_0, DiscreteGpu, banner, gemv_backend Q8_0 = cpu | label HYBRID, not scored as wgpu | BPM-016 |
| f017a/b/c | stderr has SKIP_PARITY_GATE=1 / has "nothing was judged" / lacks "GPU matches" | Refused as E3 GPU | BPM-017 |
| fW05 | op_placement.attention = host, hybrid false | Refused | WGF-005 |
| fW09 | op_placement without the attention key, hybrid false | Refused | WGF-009 |
| fR3 | no_gpu_flag true, used_gpu false, cell C0 E3 GPU | Refused | R4-003 |
| fS | stderr_captured false, no lines | Refused (absence of the fallback line counts only with the banner present) | R4 L25 #4 |

## 5. Mutations (each must turn its fixture GREEN)
| Mutation of the checker | Fixture that must flip |
|---|---|
| gate on the median | f001 |
| raise the per-leg floor check to `>` instead of `>=` | f002b: both legs exactly 0.995 at the boundary must PASS (composed cos 0.98005 ≥ 0.98) |
| fold with f32::min | f009 |
| name denylist only | f014 |
| drop the n_gen equality | f015 |
| accept on used_gpu alone | f017a |
| accept any equal `cpu_ref_path` pair (ignore the pin) | f008c |
| trust the `cpu_ref_path` label (skip the trace act-path check) | f008d |
| hybrid from keys present | fW09 |
| refuse everything | base (control) |

Note on f002: legs of 0.99 already fail the per-leg floor, so f002 cannot isolate the composed bound. The composed bound is
implied by both legs ≥ 0.995, so it is a proof obligation (BPM-002, the triangle inequality), not a separate runtime gate.
Its test is the arithmetic 2·acos(0.995) = 0.200083 ≤ acos(0.98) = 0.200335 (checked 2026-10-03, so cos(2·acos(0.995)) = 0.98005), plus f002b at the boundary. f002 stays as a per-leg Fail fixture.

## 6. Open
- RQ-5 sets the allowed `cpu_ref_path` value. It is not a blocker (C293.3). P1 lands on the provisional S-4 default `fp32_act`, held as data in `bpm.cpu_ref_path` with `bpm.ref_mixed_qtypes` = [Q4_0, Q8_0]. A different ruling flips that line and f008c's expected verdict.
- RQ-4 decides whether a HYBRID cell counts for E6. Provisional default: E1 may pass hybrid, and E2/E6 name it. f016 asserts only the label, so it holds under either ruling.
- The DECODE_OPS list is defined in wgpu-forward-v1 (WGF-009). P1 imports that list and does not restate it.
