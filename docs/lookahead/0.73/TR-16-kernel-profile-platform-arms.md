# TR-16: kernel-profile arms for wgpu on intel (AMD) and Metal on mini (#4571, epic #3999)

Status: DRAFT, la-73, 2026-10-04. Branch-only (APR-LOOKAHEAD-001 v1.2). No GPU, host or profiler was run to write
this. Code facts are read at origin/main 316dee2cd4. TRACE-001 v1.2 is read at origin/la-71/4487-obs-00 5af60751c3,
because it is not on main. wgpu facts are read from the versions Cargo.lock pins: wgpu 27.0.1, wgpu-types 27.0.1 and
wgpu-hal 27.0.4. [V] means verified there. [A] means assumed, to be checked on the host. A wgpu cite starts with the
crate's source dir, such as `wgpu-hal-27.0.4/src/`.

## 1. The row

TR-16 (TRACE-001 §4, train 0.73, EV 7, K̂ 90; deps TR-11, TR-15 and APR-OBS OBS-18):

- **Work:** "Extend `crux-F-NN-kernel-profile` with wgpu (intel AMD) and Metal (mini) arms against each platform's
  native profiler, pinned or `Refused{reason}`; registry `platforms` updated; macOS surfaces report
  `NotRun{UnsupportedPlatform}` where syscall tracing does not exist".
- **Done when:** "every 0.73 backend has an arm or a recorded refusal; 0 silent platform skips".
- **Source rows:**
  - §2.4: "`cgp profile` (cuda, simd, wgpu) | new `crux-F-NN-kernel-profile` (F-16 stays export-only) | nsys, ncu
    (pinned) | E-S3".
  - §3: `crux-F-NN-kernel-profile-v1`. Its falsifier is "planted cgp build missing one kernel (J < 1) passing". Its
    positive specimen is "gx10 W1 decode with J = 1".
- **E-S3:**
  - J = |K_cgp ∩ K_nat| / |K_cgp ∪ K_nat| = 1.
  - For every kernel k: |ln(τ_cgp,k / τ_nat,k)| ≤ q0.95(|ln(τ_nat1,k / τ_nat2,k)|).
- **Milestone (TRACE-001 line 252):** TR-16 runs "After OBS-18 has admissible rows on the new hosts".

## 2. Dependencies at 316dee2cd4: none has landed

| Dep | State | Evidence |
|---|---|---|
| TR-11: the kernel-profile contract (define, 0.71) | absent | No `*kernel-profile*` contract exists on main or on any origin branch. Main has 318 `contracts/crux-*.yaml` files, and `Refused` occurs in none of them |
| TR-15: first records (0.72) | absent | Needs TR-11 |
| TR-08: the registry `contracts/trace-consumers-v1.yaml` (0.72), which carries `platforms` | absent | Not on main and not on any of the 1291 origin refs |
| OBS-18: `gpu_proof` for wgpu and metal | draft only | `docs/lookahead/0.73/OBS-18-gpu-proof-metal-wgpu.md` |

TR-16 therefore has nothing to extend yet. This file is the design, so the row can start the day TR-11 lands
(C293.3: 0.73 does not wait on rulings).

## 3. Facts from the code

| # | Fact | Where |
|---|---|---|
| K1 | **`cgp profile wgpu` is a print-only stub.** It prints the shader path, the dispatch dims and "Method: TIMESTAMP_QUERY for GPU-side timing (~1ns resolution)", then returns `Ok`. It opens no adapter, dispatches nothing and creates no query set. The `--json` flag never reaches it [V] | `crates/aprender-cgp/src/profilers/wgpu_profiler.rs:82-127`, `crates/aprender-cgp/src/cli.rs:446-452` |
| K2 | **The backend it prints is set at compile time** by `cfg(target_os)`: linux gives "Vulkan" and macos gives "Metal". It does not come from `AdapterInfo`. This is the OBS-18 F3 class (a label by intent) [V] | `wgpu_profiler.rs:8-25` |
| K3 | **`cgp profile metal` does nothing.** On macOS it prints one line and returns `Ok(())`. Elsewhere it bails with "use --backend wgpu", but no `cgp profile` target has a `--backend` flag (`compare` takes `--backends`) [V] | `cli.rs:453-463` |
| K4 | **The tests prove nothing about timing.** FALSIFY-CGP-079 asserts only `profile_wgpu("test.wgsl", …).is_ok()`, which passes with no GPU and no file. `test_detect_backend` asserts the `cfg` constant "Vulkan" [V] | `wgpu_profiler.rs:133-137`, `:141-145`, `:164-168` |
| K5 | **No crate uses timestamp queries.** None requests `Features::TIMESTAMP_QUERY` or creates a `QuerySet`. All 13 `timestamp_writes` in aprender-compute are `None`. The only `TIMESTAMP_QUERY` token in the tree is cgp's `println!` [V] | `git grep` at 316dee2cd4 |
| K6 | **27 compute passes, 12 files, none dark.** aprender-compute has 27 `begin_compute_pass` sites in 12 files; aprender-serve has none. None of the 12 files is in the #4700 dark baseline. Each pass issues one dispatch: read for `wgsl_forward.rs` and `cached_matmul.rs`, and counted (passes = dispatch tokens) for the other 10 files. Two passes choose among the gemv, tiled and matmul pipelines inside the pass [V] | `wgsl_forward.rs:2040`, `cached_matmul.rs:308` |
| K7 | **14 passes have no label and one has a wrong label.** The 8 WGSL forward passes use `&Default::default()` (`wgsl_forward.rs:1853`, `:1882`, `:1969`, `:1998`, `:2040`, `:2091`, `:2118`, `:2144`). The 6 backward passes use `ComputePassDescriptor::default()` (`device/backward.rs:142`, …). The other 13 are labelled, but `cached_matmul.rs:308` is labelled "matmul" even when it runs gemv or tiled. The pipelines carry kernel labels (`wgsl_forward.rs:326-482`: matmul, rmsnorm, silu_mul, rope, residual, tiled_matmul, causal_attention, gemv, q4k_gemv, batch_rope) [V] | under `crates/aprender-compute/src/backends/gpu/device/` |
| K8 | **Release builds turn debug labels off.** The instance is `InstanceDescriptor { backends: gpu_backends(), ..Default::default() }`. wgpu's default flags are `from_build_config()`, which in a release build is `VALIDATION_INDIRECT_CALL` only, so `DEBUG` is off. apr never calls `with_env()`, so `WGPU_DEBUG=1` is ignored [V] | `crates/aprender-compute/src/backends/gpu/device/mod.rs:93`; `wgpu-types-27.0.1/src/instance.rs:173-203` |
| K9 | **Vulkan needs `DEBUG` for labels; Metal does not.** On Vulkan, `VK_EXT_debug_utils` is loaded only under `DEBUG`, and a pass label becomes `cmd_begin_debug_utils_label` only when that extension is loaded. On Metal, `encoder.set_label(label)` runs whenever the pass has a label, and pass timestamps come from counter sample buffers at encoder start and end when the device supports them [V] | `wgpu-hal-27.0.4/src/vulkan/instance.rs:320-322`, `wgpu-hal-27.0.4/src/vulkan/command.rs:983-986`, `:1280-1281`; `wgpu-hal-27.0.4/src/metal/command.rs` `begin_compute_pass` |
| K10 | **Pass timestamps are portable.** `TIMESTAMP_QUERY` is a WebGPU-level feature and is enough for pass-boundary timestamps. Timestamps inside a pass need the native-only `TIMESTAMP_QUERY_INSIDE_PASSES`. `QUERY_SET_MAX_QUERIES = 4096` [V] | `wgpu-types-27.0.1/src/features.rs:675`, `:1389`; `wgpu-types-27.0.1/src/lib.rs:124` |
| K11 | **cgp's CUDA numbers come from nsys and ncu.** `profile_kernel` runs ncu, and `profile_binary` runs nsys and parses its stats [V] | `crates/aprender-cgp/src/profilers/cuda.rs:468` → `:63`; `:693` → `:700`; `parse_nsys_stats` `:725` |
| K12 | **The verdict lattice already has NotRun.** `Verdict { Fail, Unknown(Reason), Pass }`, and `Reason::NotRun` is the lowest Unknown. `meet` is `min` [V] | `crates/aprender-contracts/src/ontology/verdict.rs:26`, `:85` |

## 4. Arm design (recommendation)

**Independence rule.** The cgp side and the native side of an arm must measure with different mechanisms. For CUDA
(K11), cgp parses nsys's and ncu's own output. E-S3 there tests cgp's parsing and aggregation: the J falsifier still
bites, but the τ test compares nsys with nsys. That is TR-11's to decide, not a TR-16 change. For wgpu, cgp measures
in-process with pass timestamps, and the native profiler is a separate capture. That gives a real second witness,
and it is the method cgp's own module doc names: "Cross-platform GPU profiling via wgpu timestamp queries".

### cgp side (the same code on Vulkan and Metal)

1. **K_cgp** is the set of pass labels, and each pass label is the label of the pipeline that pass dispatches.
   - Because a pass issues one dispatch (K6), its label names one kernel.
   - The two branch sites choose the pipeline before they begin the pass.
   - Labels are derived from the pipeline choice in code. They are never a literal: K7's "matmul" on a gemv run is
     the failure this rule prevents.
2. **τ_cgp,k** = Σ (end − begin) × `queue.get_timestamp_period()`, summed over the passes labelled k.
   - The timestamps come from `ComputePassTimestampWrites`.
   - This needs only `TIMESTAMP_QUERY` (K10).
3. **The timestamp mode is opt-in.** One switch turns it on; the env name below is a proposal: `APR_WGPU_KPROF=<path>`.
   - It requests `TIMESTAMP_QUERY` at device creation.
   - It adds `InstanceFlags::DEBUG`, but not `VALIDATION`, so Vulkan sees the labels (K8, K9).
   - The default path changes only by gaining the labels.
   - An adapter without `TIMESTAMP_QUERY` writes `Refused{reason: "adapter lacks TIMESTAMP_QUERY"}` and exits
     non-zero. It never falls back to wall-clock time (R-1).
4. **Query slots.** Each pass takes 2 slots, and a set holds at most 4096 (K10), so the mode resolves once per
   submit. The pass count of a decode token is to be measured on the host [A].
5. **The receipt carries the adapter.**
   - It records `AdapterInfo` (backend, vendor, device, driver, device_type).
   - It records the OBS-18 `gpu_proof` line from the same process.
   - Admission follows OBS-18: intel requires backend Vulkan and vendor 0x1002; mini requires backend Metal.
6. **The cgp verbs.**
   - `cgp profile wgpu` replaces K1: it runs the workload with the mode on, reads the table, and honours `--json`.
   - `cgp profile metal` stays as a verb. It becomes the same arm with an asserted backend of Metal. Deleting the
     verb would be S-4.

### Native side

| Cell (R1) | Host and backend | Native profiler (candidate) [A] | K_nat from | Pin, or refusal |
|---|---|---|---|---|
| C1, C2 | intel, AMD GPU 0 and 1, Vulkan | An RGP capture from the Vulkan driver: RADV `MESA_VK_TRACE=rgp`, or AMDVLK with the Radeon Developer Panel | debug-utils markers, which are the pass labels | Driver version, RGP version and the sha256 of each binary used. With no headless per-dispatch export at the pinned version: `Refused{reason}` |
| C3 | mini, Metal | Instruments Metal System Trace: `xcrun xctrace record`, then `xcrun xctrace export` | compute encoder labels, which are the pass labels | `xctrace version`, the Xcode build and the sha256 of the xctrace binary. If it cannot be reprovisioned from a pinned source: S-7 `comparator: unrunnable-hermetically`, which is report-only and never cited |

The parsers for both exports are Rust (R-3).

### Coverage: every 0.73 backend has an arm or a refusal

| Cell | Backend | Kernel-profile arm | Owner |
|---|---|---|---|
| C0 lambda | CUDA sm_89 | nsys, ncu | TR-11, TR-15 |
| C5 gx10 | CUDA sm_121 | nsys, ncu (the §3 specimen) | TR-11, TR-15 |
| C1, C2 intel | wgpu on Vulkan, AMD | the wgpu arm above | TR-16 |
| C3 mini | wgpu on Metal | the wgpu arm above | TR-16 |
| C4 gx10 | aarch64 CPU | the simd arm. cgp already launches `perf` (`analysis/bench.rs:197`) | TR-11's simd arm; TR-16 checks that a row exists |

**Syscall tracing on mini.** The syscall-trace surfaces report `NotRun{UnsupportedPlatform}` on mini: renacer is
ptrace-based (TRACE-001 line 19, §2.2 line 112). The value is `Verdict::Unknown(Reason::NotRun)` with the detail
`UnsupportedPlatform` (K12). It sits below Pass by construction, it is counted, and it is never a pass.

**Registry `platforms` (proposed).** TR-08 owns the file; these are the entries TR-16 needs:
- `cgp profile wgpu`: `[linux, macos]`.
- `cgp profile metal`: `[macos]`.
- The renacer syscall-trace surfaces: `[linux]`.

## 5. Falsifiers (planted; for TR-11's contract once it exists)

| ID | Planted case | Must be |
|---|---|---|
| F1 | A cgp build that drops one labelled pass from its table (J < 1) | RED |
| F2 | The backend printed from `cfg(target_os)` disagrees with `AdapterInfo.backend`. The stub at main prints "Vulkan" with no adapter at all (K2) | RED |
| F3 | `TIMESTAMP_QUERY` is absent, or the period is 0, or some end < begin, or a τ row comes from wall-clock time | `Refused` or RED; never a τ row (R-1) |
| F4 | `AdapterInfo.device_type` is Cpu (lavapipe, llvmpipe, SwiftShader) | RED (R1 BPM-014) |
| F5 | The kernel-profile surface on mini reports `NotRun{UnsupportedPlatform}` | RED. macOS has a GPU arm, and that escape is only for syscall tracing |
| F6 | A (surface, cell) pair from §4's coverage table has no row | RED. The done-when count comes from the table, not from the rows present |
| F7 | A native arm has none of these: (version, digest), `Refused{reason}`, or an S-7 marking | RED |
| F8 | A pass runs while profiling is on with no label, or with a label that is not its pipeline's | RED |

**Positive specimens.**
- intel C1: W1 decode on wgpu with J = 1, or the recorded refusal.
- mini C3: W1 decode on wgpu-on-Metal with J = 1, or the refusal or S-7 marking.

## 6. Work list for the row

1. **aprender-compute labels.** Label the 14 unlabelled passes and fix `cached_matmul.rs:308`, so every pass label
   comes from its pipeline (K7). Metal sees the labels at once; Vulkan sees them only under `DEBUG` (K9).
2. **aprender-compute timestamp mode.** Add the opt-in mode from §4, which refuses when the adapter lacks the
   feature.
3. **cgp.**
   - Replace the stub (K1).
   - Take the backend from `AdapterInfo` (K2).
   - Turn `profile metal` into the wgpu arm with a Metal assertion (K3).
   - Replace K4's tests with tests that fail on today's stub.
4. **Native parsers.** Write the Rust parsers for the native exports, pinned or refused as in §4.
5. **Contract and registry.** Add the contract arms and the registry entries once TR-11 and TR-08 exist.
6. **First records.** Take the first records on intel and mini once OBS-18 rows are admissible.

## 7. Questions only a host can answer

- **H1 (intel).** Which Vulkan driver runs, at which version, and is there a headless per-dispatch RGP export at that
  version? If there is none: `Refused{reason}`.
- **H2 (mini).** Which Xcode and xctrace versions are installed, and which export table holds per-encoder GPU
  intervals with their labels?
- **H3 (each cell).** Does the adapter expose `TIMESTAMP_QUERY`, and what is the timestamp period?
- **H4 (τ bound).** Do the pass-boundary timestamps move τ_cgp outside the native noise on a correct build? The first
  record decides. Widening the E-S3 bound needs a ruling; it is not this row's to change.

## 8. Not done here

No GPU, host or profiler was run. No contract was written, because TR-11 has not landed. No ticket was minted and no
milestone was moved.
