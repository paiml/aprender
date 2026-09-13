## aprender 0.65.2 — the parity-instrument release, fully published

0.65.0 was cut at `587ad0797` and its crates.io cascade stopped at 48 of 74 crates: five sibling dev-dependencies carried a version through the workspace alias, so `aprender-core` (needs `aprender-data`) could not be uploaded before `aprender-test-lib` (needs `aprender-core`). 0.65.1 (cut at `752f55346`) fixed that and reached 64 of 74 crates before `aprender-test-lib` failed its publish verification (an `include_str!` of a file outside the crate). 0.65.2 supersedes both for every crate (74/74 on crates.io, drained 2026-09-04 23:20Z); `v0.65.0` and `v0.65.1` stay where they were cut.

### Fixed since 0.65.0
- **Packaging (PMAT-958, #2866).** `aprender-test-lib` embeds `scripts/perf-matrix.yaml` through `build.rs` and a vendored copy that may never drift; the CB-510 guard now reports any `include_str!`/`include_bytes!` in host-compiled code whose target escapes the crate.
- **Publish cycle (PMAT-955, #2864).** The five dev-dependencies are path-only; `check_publish_preflight.sh` rule **R6** refuses any sibling dev-dependency that carries a version (fixture on both polarities, mutation RED).
- **Drain log (PMAT-954, #2864).** `cascade-drain.sh` keeps the `DEFER` lines that say why a crate did not publish.
- **wgpu init deadlock (PMAT-952, #2862).** The `shared_instance` initializer no longer takes `DEVICE_INIT_LOCK`, which `GpuDevice::new` already holds while calling it (ABBA on a process's first initialization; the 0.65.0 release clean-room parked 49 threads on it).
- **stdio MCP transport (PMAT-953, #2863; PMAT-950, #2860).** A server that answers and exits before reading the request keeps its response and its exit status instead of surfacing `write stdin: Broken pipe`.

### What 0.65.x is
PP-LLAMA-001 v3.1: the parity spec and the instrument it describes — `perf_gate.sh` with merge and release phases, paired/interleaved receipts with bootstrap statistics, the PP-26 batch-invariance witness, the PP-9 spend ledger with its scanner (PMAT-930..935), the F-9 publish preflight, and UNMEASURED cells with owners and expiries instead of numbers. No parity ratio is published in this release; every c>1 band is INVALID-CORRECTNESS until the witness passes on the release hosts (see `docs/specifications/0.66-performance-parity-report.md` for the 0.66 plan).

### Known
- `CHANGELOG.md` has no 0.65.1 or 0.65.2 entry: the claim-literal baseline is line-keyed and refuses the coordinate shift a changelog insert causes (PMAT-956); the entry lands with that guard fix.
