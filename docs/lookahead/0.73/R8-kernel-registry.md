# Row 8: kernel-registry coverage for the E1 backends

PROVISIONAL. `contracts/kernel-registry-v1.yaml` is not on main (4098007133). Read on
the unmerged `kreg/4539-parity-receipts` @02b0f7f7fc (#4539, another lane's branch, read
only). Re-read when #4539 lands; every cite here is at 02b0f7f7fc. `J:` =
`crates/aprender-serve/kernel-registry.json`, `KR:` =
`crates/aprender-serve/src/kernel_registry.rs`, `A:` =
`crates/aprender-serve/src/gpu/adapters/wgpu_adapter.rs`. No build.

## What #4539 gives
- 41 rows in J: 21 `cpu matvec`, 18 `cuda gemv`, 2 `wgpu gemv` (`wgpu.gemv.q4_k`,
  `wgpu.gemv.f32`). The key is (backend, arch, isa_features, op, qtype, layout); layout
  is closed to `row_major`.
- `Backend` (KR:32) already has `Metal`, with no rows, so every Metal combination is
  refused (the E5 input from this lane).
- Admit call sites: CPU `matmul_fused.rs:396`, CUDA `cuda/types.rs:275`, and on wgpu only
  the raw-Q4_K upload (`wgpu_takes_raw_q4k`, A:304-315).
- Parity receipts: 16, all `cpu` (`evidence/kreg/parity/cpu.*.json`). 25 of 41
  tolerances are `unmeasured`, both wgpu rows among them.

## Finding R8-1: on wgpu, a refusal becomes a host dequant the registry cannot see
`wgpu_takes_raw_q4k` (A:304) asks the registry; on a refusal the tensor is dequantized
on the host to F32 (`dequant_model_weights_except`, A:76) and dispatched to
`encode_matmul`, whose row is `wgpu.gemv.f32`. Nothing calls `admit` for that F32
dispatch. So a Q5_K or Q6_K weight on wgpu is never refused by the registry (Q8_0 and Q4_0 fail the dequant itself, E4 census): it runs as F32 at
several times the memory (R19: 5.8x for Q5_K) and the registry records F32. E4 counts
this as a fallback; the registry, as keyed, can't.
Fix: the wgpu selector calls `admit` for every weight with its SOURCE qtype; a host
dequant is a row of its own (op `dequant_f32`, backend `wgpu`, the source qtype) so it is
declared, counted, and can be refused per run (E4 kind F).

## Finding R8-2: the op vocabulary is GEMV only
The scope (contract `scope:`) is the CPU, CUDA and wgpu GEMV selectors. The E1 routes
also dispatch RMSNorm, RoPE, attention, SiLU/SwiGLU, argmax and, after row 6, the nine
Gated DeltaNet and 256-wide attention kernels. "Every kernel key the E1 backends
dispatch" needs either those ops as rows (qtype `F32`) or a stated cut. Proposal: v1
keeps the cut at weight-consuming ops (where LAYOUT-001/002 lives) and says so in the
contract; v2 adds a row per WGSL pipeline. The coverage falsifier names which.

## Finding R8-3: the coverage test is per selector, and wgpu has two
"Every dispatchable kernel has a row" is proved by the selector-coverage test (contract
header). On wgpu the dispatch is chosen in two places, A: (raw or dequant) and
`WgslForwardPass` (W:, which pipeline). The test must enumerate both, or a new WGSL GEMV
(P4, row 19 Q5_K) can land without a row.

## Finding R8-4: wgpu and Metal tolerances need host receipts
All receipts are CPU, measured on this box. `wgpu.gemv.*` tolerances can only be measured
on C1 (intel), C5b (mini) and C4 (gx10, CUDA rows): M runs. Until then the rows say
`unmeasured`, which is honest, and E5 must read `unmeasured` as not passed.

## Draft for the row-8 ticket (after #4539 lands)
1. wgpu: `admit(Wgpu, source_qtype)` for every weight; `wgpu.dequant_f32.<qtype>` rows.
2. Contract `scope:` names the op cut (v1 weight ops).
3. Selector coverage enumerates A: and W: dispatch sites.
4. Metal rows land with the Metal backend (none until then; refused is correct).
5. Parity receipts for wgpu and CUDA rows on their hosts (M).

## Falsifiers
| id | claim | how |
|---|---|---|
| F8-1 | no wgpu weight bypasses admit | a registry with no `wgpu.dequant_f32.q5_k` row refuses a Q5_K model on wgpu (loud, exit 14 when forced) instead of running F32 |
| F8-2 | a new WGSL GEMV needs a row | add a pipeline in W: without a J row; selector coverage fails |
| F8-3 | Metal is refused until registered | `admit(Metal, any)` is Err for every qtype |
| F8-4 | `unmeasured` is not a pass | E5 checker on a row with `tolerance: unmeasured` reports NOT_PASSED |
| F8-5 | the op cut is declared | contract `scope:` lists the ops; a dispatched op outside it is named in the census, not silently skipped |

## Open
- Whether #4539 lands as read; re-cite then.
- Whether E5 wants v1's op cut or every WGSL pipeline (operator or cop call; default v1).
