# PMAT-4089 receipt: `apr serve` and `apr run` take the same default backend

**Ticket:** #4089. Assigned by the cop (aprender-cf). **Base:** `origin/chore/0.69.1-merge-back`
@ bfafca251, per the cop's ruling (A): it is the only lineage carrying the serve qwen35 route and
serve's `used_gpu`, the surfaces the ticket measured (9345331b9). main 49fe19c28 refuses the
Qwen3.5 GGUF at serve load (#3571), and its dense `/v1/completions` carries no `used_gpu`.
**Branch:** `fix/4089-serve-run-backend-default-mb`. It is NOT pushed into #4046's head, and it folds
after #4046.

## The two resolution paths, made one
- **Before:**
  - run was GPU unless `--no-gpu` (plus a device check on the APR path only);
  - serve was CPU unless `--gpu`/`--gpu-layers` (`ServerConfig::resolve_layers`: `gpu_layers: None` → 0).
- `accel::default_wants_accelerator(no_gpu)` is the ONE rule: `!no_gpu && CudaExecutor::is_available()`
  on cuda builds, false otherwise.
- **run:** `accel::run_no_gpu(no_gpu, accel_forced)` sets `RunOptions.no_gpu`.
  - Flagless: the rule decides.
  - Forced (`--gpu` / `--backend cuda|wgpu|gpu`): `no_gpu` passes through untouched, so the device check
    never lowers it (I-17). The forward is attempted, then reconciled or refused as before.
- **serve:** `ServerConfig::effective_gpu_layers()` returns the explicit request, else `Auto` when the rule
  says accelerator.
  - `wants_accelerator()` and `resolve_layers()` read it.
  - Explicit requests are untouched (I-17).
  - A defaulted start is soft: serve's CUDA path already falls back to CPU, visibly, when init fails.
- **Log:** both serve routes print `requested=` through `requested_gpu_layers_label()`, which reads
  `auto(default)` for a flagless start instead of `none`.
- **OffloadReport** `gpu_layers_requested` / `explicit_args` still record only what the user typed.

## Must-REDs
| Row | Mutant | Result |
|---|---|---|
| `accel::tests::a_forced_accelerator_is_never_lowered_by_the_device_check` | `accel_forced` guard removed from `run_no_gpu` | RED: "forced --gpu must reach the forward (I-17)", run with `CUDA_VISIBLE_DEVICES=` |
| `tests/falsify_serve_run_default_backend_4089.rs` on Qwen3.5-4B-Q4_K_M | the base binary (no fix) | RED: `apr run used_gpu=true fell_back=Some(false)  apr serve used_gpu=false` |

Unit rows `serve::types::default_backend_parity_4089` (3 tests) plus the I-17 row: 4/4 pass at f9c942c39.

## Device measurement (lambda RTX 4090, all through gpu-q, `nvidia-smi --query-compute-apps` empty before and after every leg)
- Binary: `apr-mbbase` = `apr 0.69.1 (bfafca251)`, sha256 4eafe3688063c32c. Qwen3.5-4B-Q4_K_M, no backend flag. Serve: `requested=none resolved=0 total=32 (backend=cpu)`, `/v1/completions used_gpu=false`. Run: `used_gpu=true fell_back=false`. **RED.**
- Binary: `apr-mbfix` = `apr 0.69.1 (f9c942c39)`, sha256 50b0435a18fcca12. Qwen3.5-4B-Q4_K_M, no backend flag. Serve: `requested=auto(default) resolved=32 total=32 (backend=cuda)`, `/v1/completions used_gpu=true`. Run: `used_gpu=true fell_back=false`. **GREEN** (with `APR_EXPECT_GPU=1`).
- Qwen3-1.7B-Q4_K_M (dense), both binaries: resolution changes from base `requested=none resolved=0` to fix `requested=auto(default) resolved=28 (backend=cuda)`, then "CUDA optimized model ready". But the dense serve route reports NO `used_gpu` on `/v1/completions` or `/v1/chat/completions`, so the falsifier stays RED ("no provenance to assert"), by design: there is no third state. Filed as **#4146**. The ticket had called the dense case "undetermined", and it still is, at the provenance level.

## Side effects a reviewer should weigh
- On a cuda build with NO device, a flagless `apr run` of a GGUF now runs on CPU. It used to error "CUDA init failed": the rule's device check now applies on every run path, not only APR. A forced `--gpu` keeps the old behaviour.
- `apr serve` on a cuda build with a device now uses the GPU by default. That is the ticket's intent, and it changes serve's default VRAM footprint. `--gpu-layers 0` or `--no-gpu` restores CPU.

## Lint
- `cargo fmt --all -- --check`: rc 0.
- `cargo clippy -p apr-cli --lib --tests -- -D warnings`: rc 101, but no finding lands in any file this branch touches. The same failures appear at base bfafca251, in pre-existing test targets such as `falsification_crux_*`.
- The `--features cuda` clippy aborts earlier in `aprender-train` (59 trivial-cast findings, untouched here).
