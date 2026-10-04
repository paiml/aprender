# P1 — backend-parity cell receipt: schema and planted fixtures (draft, la-73, 2026-10-03)

P1 is the first bundle in `falsifier-landing-map.md`. It holds 22 falsifiers and needs no GPU, no model and no aarch64 host.
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
| `leg_a`, `leg_b` | {record path, record sha256, `cpu_ref_path`, `cpu_run_sha256` (the apr CPU run the leg reads), `prompt_set_sha256`, `n`, `model_sha256`, `cos[p][t]` (prompt × position, position 0 = prefill-final, then ≥ 8 decode)} | 001, 002, 008-011, 013; NEON-Q4K-009 reads leg_a's `cpu_run_sha256` |
| `e2` | {record path, sha256, `n_gen_apr`, `n_gen_llamacpp`, decode/prefill {r, ci_lo, ci_hi, n}} | 005, 015 |
| `llamacpp_sha` | string | 004 |
| `adapter` | {name, `device_type` ∈ wgpu DeviceType, banner line} | 003, 006, 014 |
| `kernel_path` | {backend, reference}: one `apr-kernel-path-v1` path per run, whose entries also carry `tensor` (the GGUF name); a (tensor, op) has one entry per kernel it reached | 012 |
| `op_placement` | map op → {device, host}; keys = DECODE_OPS | 016, WGF-005, WGF-009 |
| `gemv_backend` | map qtype → {wgpu, cpu} | 016 |
| `hybrid` | bool | WGF-005 |
| `stderr_lines` | captured lines of the serving process; `stderr_captured: true` | 003, 017, R4-003 |
| `used_gpu`, `no_gpu_flag` | bool | R4-003 |
| `info_rows` | list of {`name`, `route`, `runs` {role → {`run_sha256`, `model_sha256`, `prompt_set_sha256`, `kernel_path`}}, `cos` {pair → {record path, record sha256, min, median, max, n}}}. The checker derives each row's status (measured, or not_measured with a reason) and labels, and reads none of them from the producer (RQ-5: never in the verdict) | 018, NEON-Q4K-009 |
| `routes` | {`run`, `serve`} → {`binary_sha256`, `model_sha256`, `host`, `prompt_set_sha256`, `n_gen`, `forward_trace_line` (the prefill forward with its attention path and rows per chunk, and the decode forward), `kernel_path`, `tokens` [prompt → generated ids]}. Greedy and non-batched; `serve` goes through `POST /generate` (`token_ids`, `aprender-serve/src/api/types.rs:110`). The checker binds each route to leg A's backend run (its `kernel_path` and forward line), never to a field the producer asserts (R1 §13) | 019 |

## 3a. Alignment with the OBS stack (found 2026-10-03; unmerged, #4487 / #4574)
`apr-obs-row-identity-v1` and `apr-kernel-path-v1` are not on main. They sit on the unmerged OBS-00 commit 96d2fe4aa1,
which `origin/a01/4574-obs15-kernel-path` (b1244f6fb7) carries with a Rust checker in
`crates/aprender-qa-report/src/obs_kernel_path.rs` (`check_kernel_path`, `kernel_diff`, `admit_kreg_entry`).
- **Identity.** The cell receipt carries the OBS IDENTITY block unchanged:
  - the fields are schema, ts, host, apr_version, apr_tag, crate_tarball_sha256, binary_sha256, build_identity, model_id, model_sha256, backend and request_id, plus `gpu_proof`;
  - P1 calls the OBS-01 shared lint and does not copy the list;
  - `model_sha256` there is the same field BPM-011 compares across the three runs.
- **kernel_path.** The `kernel_path` field of §3 uses the `apr-kernel-path-v1` shape, one per run (backend, reference). Each entry is `{op, kernel_id, qtype, layout, arch, shape_class, precision}`, plus `tensor`, the GGUF tensor name (added 2026-10-03; OBS entries do not carry it).
  - R3's `kernel_path(k)` string, e.g. `q4k-q8k/neon-sdot`, becomes `kernel_id`, with `arch = aarch64`. The C0 x86 reference names its arms the same way, e.g. `q4k-f32/avx2` (fused_k.rs:201).
  - BPM-012 compares, per `(tensor, op)`, the sets of `kernel_id`s the backend and reference runs reached, over the quantized matmul tensors (a quantized qtype on a GEMV or matmul op; the embedding gather is not one). One shared `kernel_id` on one tensor REFUSES a CPU cell, whatever the others do.
  - A set, because the route can switch per call. On the default route a crushed activation block sends that one Q4_K call to f32 (ffn_block.rs:790-792), so one tensor runs q4k-q8k and q4k-f32 in one run (f012e).
  - BPM-012 does not reuse `kernel_diff`, because each of its three choices lets a self-comparison pass (R1 contract L25, 2026-10-03):
    - it compares whole paths, so only an empty diff refuses. A same-host reference differs from a default-route backend on Q4_K (q4k-q8k against q4k-f32) while every Q6_K tensor meets itself (f012);
    - it keys by the slot `(op, shape_class)` and keeps one entry per slot (`slots()`, obs_kernel_path.rs:118 on #4574 @b1244f6fb7, last wins). A Q4_K_M file mixes qtypes in one slot: in Qwen3.5-0.8B-Q4_K_M, ffn_down (3584, 1024) is Q6_K in 12 layers and Q4_K in 12 (f012d);
    - it compares whole entries, and `arch` differs on every C0-against-C4 entry, so one scalar kernel on both hosts reads as different (f012c).
- **Conflict to resolve.** `apr-kernel-path-v1` `trace_cut` says "CPU rows name no GPU kernel and are outside the rule", so a CPU row may carry `kernel_path = null`.
  - BPM-012 needs a non-null kernel_path on every CPU cell, C4 above all, or the self-comparison guard has nothing to read.
  - P1 adds this as its own REQUIRE for cell receipts. It does not change the OBS rule, which governs perf-ledger rows.
  - Flag it to the OBS-15 owner when the stack merges; do not edit their branch. The slot collision above went to the cop as a ROUTE line on 2026-10-03, because it also hides a night-over-night kernel change from `kernel_diff`.
- **KREG link.** OBS `kreg_admission` admits a kernel only with a `parity = pass` receipt for the same (arch, backend, kernel_id). A BPM cell receipt with E1 PASS is that receipt. It names its kernel_ids, so one artifact serves E1 and E5.
- **Order.** P1 depends on the OBS stack merging. If it has not merged when P1 is minted, P1 lands the cell checker with the identity check as `NotRun(identity lint absent)`, never as a pass.

## 4. Planted fixtures (`tests/fixtures/bpm/`)
Every fixture is `base.json` plus one edit. `base.json` is a C0 CUDA cell with:
- both legs ≥ 0.999 at every position, and E2 r = 0.7 with ci_lo = 0.6;
- equal n_gen, the ruled pin, `device_type` DiscreteGpu and the CUDA banner;
- a "F2 guard: GPU matches" line, different kernel paths and a total `op_placement` on device;
- `routes` run and serve, each a copy of leg A's backend run (identity, forward line, `kernel_path`), with equal tokens on all 16 prompts. Every fixture keeps that copy unless its edit names a route, so no earlier verdict changes.

**Control:** `base.json` must be `Pass`. A checker that refuses everything fails this row.

**C4 control:** f012b (`base_c4.json`) must be `Pass`. f012, f012c, f012d and f012e are `base_c4.json` plus one edit, and f012b fixes which GPU-only fields a CPU cell omits. A checker that refuses every CPU cell fails this row. Its routes are planted bound, as in `base.json`; a real C4 route at 316dee2cd4 runs the default route, not leg A's `fp32_act` run, so it is Refused while RQ-6 holds (R1 §13e).

**Info rows:** fN9a..fN9d are `base_c4.json` plus one `c4_default_route_info` row (neon-q4k-q6k-v1): d4 has q4k-q8k/neon-sdot on Q4_K and q6k-f32/neon on Q6_K, d0 has q4k-q8k/avx2, f0 is leg_a's `cpu_run_sha256`, and r1, r2, r3 have min 0.993, 0.994, 0.9995. Then one edit. The receipt verdict stays Pass in all four (BPM-018), and fN9b is the measured control.

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
| f008c | both legs q8k_act; pin `bpm.cpu_ref_path` = fp32_act (RQ-5, ruled) | Refused | BPM-008 |
| f008d | both legs fp32_act, but the reference `kernel_path` has blk.0.ffn_gate Q4_K at precision q8k (the scratch forward) | Refused | BPM-008 |
| f009 | one cosine = NaN (as a JSON string "NaN"; JSON has no NaN literal, so the extractor must parse it and not drop it) | Refused | BPM-009 |
| f010 | leg_b.prompt_set_sha256 differs; and a second fixture with n 16 vs 15 | Refused | BPM-010 |
| f011 | model_sha256 differs between the three records | Refused | BPM-011 |
| f012 | `kernel_path` replaced by a same-host pair of two entries per run: blk.0.attn_q Q4_K is q4k-q8k/neon-sdot in the backend and q4k-f32/neon in the reference; blk.0.ffn_down Q6_K is q6k-f32/neon in both; leg_a 1.0 | Refused | BPM-012 |
| f012b | `base_c4.json`, the C4 control: cell C4, backend cpu, no adapter, banner, `gpu_proof` or F2 guard line, `op_placement` on the host; the reference run is on the C0 host; every Q4_K and Q6_K entry is `/neon` on aarch64 in the backend and `/avx2` on x86_64 in the reference | Pass | BPM-012 (control) |
| f012c | blk.0.ffn_down is q6k-f32/scalar in both runs (aarch64 and x86_64); every other entry as f012b | Refused | BPM-012 |
| f012d | `kernel_path` replaced by a same-host pair of three entries per run, blk.4, blk.5 and blk.6.ffn_down at one shape_class (the layer pattern of a 24- or 28-layer Q4_K_M file): Q4_K q4k-q8k/neon-sdot against q4k-f32/neon, Q6_K q6k-f32/neon in both, Q4_K as blk.4 | Refused | BPM-012 |
| f012e | `kernel_path` replaced by a same-host pair for blk.0.attn_q Q4_K: the backend entries are q4k-f32/neon (one crushed call), then q4k-q8k/neon-sdot; the reference entry is q4k-f32/neon; and a second fixture with the two backend entries in the other order | Refused | BPM-012 |
| f013 | prefill-final 1.0, decode position 4 = 0.90; and a second fixture with only 7 decode positions | Fail / Refused | BPM-013 |
| f014 | adapter "FooSoft Renderer", device_type Cpu | Fail(FALLBACK) | BPM-014 |
| f015 | n_gen_apr 37, n_gen_llamacpp 128 | Refused | BPM-015 |
| f016 | cell C1, Q8_0, DiscreteGpu, banner, gemv_backend Q8_0 = cpu | label HYBRID, not scored as wgpu | BPM-016 |
| f017a/b/c | stderr has SKIP_PARITY_GATE=1 / has "nothing was judged" / lacks "GPU matches" | Refused as E3 GPU | BPM-017 |
| f018a | f001 plus one info row at 0.999 everywhere | Fail(E1 min) | BPM-018 |
| f018b | one info row (the q8k_act reference, RQ-5) at min 0.985; and a second fixture whose info row holds one cosine "NaN" | Pass; the second row reads not_measured(non_finite) | BPM-018 (control) |
| fN9a | every Q4_K entry of d4 is q4k-f32/neon (an fp32_act run); the row still says route default | Pass; row not_measured(route_unproven) | NEON-Q4K-009 |
| fN9b | every Q4_K and Q6_K entry of d4 is /scalar (the 316dee2cd4 kernels), arch aarch64 | Pass; row measured, label(d4) scalar | NEON-Q4K-009 (control) |
| fN9c | f0 is a C0 fp32_act run of another prompt set, not leg_a's `cpu_run_sha256` | Pass; row not_measured(unbound) | NEON-Q4K-009 |
| fN9d | r1, r2, r3 min 0.95, 0.999, 0.9999 (a1 = 0.3176 rad > a2 + a3 + 1e-3 = 0.0599) | Pass; row not_measured(inconsistent) | NEON-Q4K-009 |
| fW05 | op_placement.attention = host, hybrid false | Refused | WGF-005 |
| fW09 | op_placement without the attention key, hybrid false | Refused | WGF-009 |
| fR3 | no_gpu_flag true, used_gpu false, cell C0 E3 GPU | Refused | R4-003 |
| fS | stderr_captured false, no lines | Refused (absence of the fallback line counts only with the banner present) | R4 L25 #4 |
| f019a | `routes.run.kernel_path` reaches a `kernel_id` on blk.0.ffn_up Q4_K that leg A's backend run never reached (a batched-prefill GEMM); and a second fixture with the same edit on `routes.serve` only | Refused(route binding) | BPM-019 |
| f019b | `routes.serve` tokens differ from `routes.run` at decode step 5 of prompt 11 of 16 | Fail(route tokens) | BPM-019 |
| f019c | `routes.serve` absent; and a second fixture whose `routes.serve` `binary_sha256` is not the cell's | NotRun(route); the second Refused(route identity) | BPM-019 |
| f019d | the `routes.run` forward line names the session's batched prefill while leg A's backend run printed per-token; `kernel_path` unchanged | Refused(route binding) | BPM-019 |

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
| compare whole paths (refuse only an empty `kernel_diff`) | f012 |
| compare whole entries (`arch` included) | f012c |
| key by the OBS slot `(op, shape_class)`, first or last entry | f012d |
| keep one `kernel_id` per (tensor, op), first or last entry | f012e (both orders) |
| take the best of a leg and its info rows | f018a |
| fold the info rows into the leg min | f018b (control) |
| refuse a receipt on a non-finite info row, as BPM-009 does for a leg | f018b, second fixture (control) |
| trust the row's route field (skip the route proof) | fN9a |
| take the label from `arch` | fN9b |
| skip the run binding | fN9c |
| drop the triangle check | fN9d |
| skip the route `kernel_path` check | f019a |
| check the run route only | f019a, second fixture |
| bind on `kernel_path` alone (skip the forward line) | f019d |
| compare the routes' tokens on the first prompt only | f019b |
| accept a cell with one route | f019c |
| skip the route identity check (binary, model, host, prompt set) | f019c, second fixture |
| let bound, equal routes pass without the leg checks | f001 |
| refuse every CPU cell | f012b (control) |
| refuse everything | base (control) |

Note on f002: legs of 0.99 already fail the per-leg floor, so f002 cannot isolate the composed bound. The composed bound is
implied by both legs ≥ 0.995, so it is a proof obligation (BPM-002, the triangle inequality), not a separate runtime gate.
Its test is the arithmetic 2·acos(0.995) = 0.200083 ≤ acos(0.98) = 0.200335 (checked 2026-10-03, so cos(2·acos(0.995)) = 0.98005), plus f002b at the boundary. f002 stays as a per-leg Fail fixture.

## 6. Open
- RQ-5 is ruled (cop, 2026-09-27 20:12Z): the E1 CPU reference is `fp32_act`, held as data in `bpm.cpu_ref_path` with `bpm.ref_mixed_qtypes` = [Q4_0, Q8_0]. A `q8k_act` run is reported as an info row only, so P1 refuses it as an E1 receipt (f008c) and it never counts toward a gate. Dense parity and parity-moe use the same reference. Item (e) gives info rows their place, `info_rows` (§3), and two rules: a row never changes the verdict (BPM-018, f018a, f018b), and the C4 default-route row is measured only on a proven route, one bound run triple and consistent angles (NEON-Q4K-009, fN9a..fN9d). Whether the gx10 default route should gate is RQ-6 (handoff); the provisional S-4 default is info only.
- RQ-4 decides whether a HYBRID cell counts for E6. Provisional default: E1 may pass hybrid, and E2/E6 name it. f016 asserts only the label, so it holds under either ruling.
- The DECODE_OPS list is defined in wgpu-forward-v1 (WGF-009). P1 imports that list and does not restate it.
- Found 2026-10-04 (item p): the §3 receipt records `used_gpu` but neither the exit code nor `backend.ran` / `backend.fell_back`. A forced wgpu run that falls back to the CPU exits 14 and, with `--json`, prints `ran` = `cpu` and `fell_back` = true (R5 F-R5-2). Open: whether the receipt records those too. Until then R1 F6 refuses such a cell by its trace. No fixture or gate count changes.
- RQ-9 (who builds leg A for each E1 route) does not block P1: f019a..f019d are planted receipts. The real `routes` fields need P2 and the route runs of R1 §13.
- If RQ-6 flips, the C4 routes are compared without the crushed-block f32 switch entries on either side (`aprender-serve/src/gguf/inference/forward/ffn_block.rs:790-792`, f012e), because that switch depends on the activations. Under the default the C4 routes are Refused.
