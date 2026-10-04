# OBS-18 prerequisite: `gpu_proof` for the Metal and wgpu backends (#4575, epic #3999)

Status: DRAFT, la-73, 2026-09-28. Branch-only (APR-LOOKAHEAD-001 v1.2). No GPU was run to write this; every
code fact below was read from origin/main.

OBS-18's done-when says: "`gpu_proof` for Metal and wgpu defined (a device-kernel trace line) before any row
is admitted". The CUDA rule already exists (`apr-obs-row-identity-v1` `gpu_claim_is_proven`: backend ∈
{cuda, wgpu, metal} ⇒ gpu_proof ≠ null; a flag, env var or feature list is never proof). This file defines
what the non-null value must be for the two new backends.

## Facts from the code (origin/main)

| # | Fact | Where |
|---|---|---|
| F1 | apr has no native Metal backend. "Metal" means wgpu on its Metal backend: `gpu_backends()` = `wgpu::Backends::PRIMARY` (Vulkan, Metal, DX12, WebGPU; never GL/GLES). | `crates/aprender-compute/src/backends/gpu/device/mod.rs:58` |
| F2 | The backend wgpu actually picked is only knowable from `adapter.get_info()` (`backend`, `name`, `vendor`, `device`, `driver`, `driver_info`). | same file, `list_adapters_async` (`device/mod.rs:215`) |
| F3 | **DEFECT.** The user-visible backend line is a HARDCODED string: `eprintln!("Backend: wgpu (Vulkan)")`. It is printed after `GpuDevice::new()` succeeds, from `AdapterInfo`, and before any kernel runs. On mini, wgpu runs on Metal and the line still says Vulkan. | `crates/aprender-serve/src/infer/gguf_gpu_generate.rs:178` and `:653`; `crates/aprender-serve/src/infer/batch_wgpu.rs:144` |

F3 is the #2644 class ("device: GPU" printed by intent): the label says what was hoped for, not what ran.
It must not be accepted as `gpu_proof` on any host.

## Definition

A wgpu or metal row's `gpu_proof` is ONE line, emitted by the process that produced the timing, AFTER the
first compute dispatch of the timed run has been submitted and its result read back:

```
gpu_proof wgpu backend=<AdapterInfo.backend> adapter=<name> vendor=0x<vendor> device=0x<device> driver=<driver> dispatch=<first kernel name> readback_ok=1
```

Admission rules (checked by the ledger reader, not by the writer's say-so):

| Row `backend` | Required in `gpu_proof` | Else |
|---|---|---|
| `metal` | `backend=Metal`, host is darwin | `backend_unproven`, excluded from GPU series |
| `wgpu` (intel, AMD) | `backend=Vulkan`, `vendor=0x1002` (AMD) | `backend_unproven` |
| `wgpu`, any host | `readback_ok=1` and a non-empty `dispatch=` | `backend_unproven`. Device init alone is not a kernel |
| any GPU backend | the line comes from `AdapterInfo`, never from a literal string | REJECT (a literal-string line is the F3 hole) |
| `cpu` | `gpu_proof = null` | REJECT a non-null proof (existing rule) |

Why after a readback: `GpuDevice::new()` succeeding proves an adapter opened, not that the timed kernels ran
on it. A CPU fallback after a failed dispatch would otherwise inherit the proof.

## Falsifiers (for OBS-18's contract rows, not yet written)

- FALSIFY-OBS18-GP-001: a metal row whose proof says `backend=Vulkan` → backend_unproven. Mutant: accept any
  non-null proof, which must turn RED.
- FALSIFY-OBS18-GP-002: a wgpu row whose proof lacks `readback_ok=1` → backend_unproven. Mutant: admit on
  device init, which must turn RED.
- FALSIFY-OBS18-GP-003: the literal `Backend: wgpu (Vulkan)` line offered as proof → REJECT. Mutant: grep for
  "wgpu" in stderr, which must turn RED.
- FALSIFY-OBS18-GP-004: intel wgpu proof with `vendor` ≠ 0x1002 (e.g. llvmpipe, a software rasterizer) →
  backend_unproven. Negative control: llvmpipe is a Vulkan adapter, so backend alone does not prove a GPU.

## Proposed work (NOT minted; PROPOSE-TICKET via the inbox)

1. Replace the three hardcoded `Backend: wgpu (Vulkan)` prints with a line built from `AdapterInfo` (the F3
   fix; CPU-only unit test: format the line from a synthetic `AdapterInfo` with backend=Metal).
2. Emit the `gpu_proof wgpu …` line after the first readback in the timed path.
3. Put rules GP-001..004 into `apr-perf-ledger-v1` when OBS-05's reader lands (its FALSIFY-OBS-PERF-004 is
   still "NOT YET WRITTEN").

Depends on OBS-05, OBS-13 and OBS-15 (per #4575); none of this changes the CUDA rule.
