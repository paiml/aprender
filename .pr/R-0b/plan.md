# R-0b — P1 plan (2026-09-07, A): backend resolution reads the registry

Issue #3002 · PMAT-1073 · DAG row R-0b (blockers R-0 = #3004 unmerged, G-11 ✓, L0-1a = #3026 unmerged) · expiry 2026-09-26.
Branch agent/R-0b from origin/main e4b423790. **P2 starts when #3004 (R-0a) is on main** — the registry
(`aprender_compute::registry::{BackendRegistry, BackendEntry, Status, Reason, Selection, MockBackendFactory}`) is its API.

## P0 (discover.json written; the sites, from origin/main)
20 `cfg!(… feature = "cuda"|"wgpu")` reads in `crates/apr-cli/src`: serve/mod.rs ×6 (incl. :236 the `--gpu` acceptance
and :656 the refusal test), bench.rs ×5, serve/handler_gpu_completion.rs ×2, finetune.rs ×2, lib.rs ×1, dispatch.rs ×1 (:173),
accel.rs ×1 (:28 `build_has_accelerator`), serve/tests_offload_report_pp14.rs ×2 (tests). `parity_admission::admit` has
no caller: the `selected:` line is not printed by run/chat/serve today (measured 2026-09-07: `apr chat --gpu <1.5B>` prints
no admission line on either binary).

## Phases and acceptance (every A_i is a command in accept.sh)
1. `apr_cli::backend::resolve(request) -> Result<Selection, CliError>` over `BackendRegistry::discover()`: Ready ∧
   serviceable (the manifest row for `--model`, C14) → `Selection`; a forced request (`--gpu`, `--backend X`, `--device`)
   that is not Ready → the refusal variant from error.rs with the registry's `Reason::text()`; never `selected: cpu` for a
   forced GPU. accel.rs:28 / dispatch.rs:173 / serve/mod.rs:236 consult `resolve`, not `cfg!`.
   A1: `git grep -c 'cfg!(any(feature = "cuda"' crates/apr-cli/src` (outside `backend.rs`/tests) prints 0.
2. `selected: <kind> [device] (parity: PASS cosine=… positions=… | reason: …)` printed on run/chat/serve at load, from
   `parity_admission::admit` + the Selection (REG-15 line joins the registry line).
   A2: `cargo test -p apr-cli --test backend_refusal_case_table` — rows generated from {build features} × {host facts via
   MockBackendFactory} × BACKEND_VALUES; refuse ⇔ request ∉ Ready; every refusal names an error.rs code; the forced-GPU
   row never downgrades.
3. `GET /v1/effective-config` reports `backend == the resolved Selection` and `discovered_at` (REG-12); C11 cross-check on a
   CPU-only host (`scripts/check_backend_registry.sh --static` in `ci / gate`).
   A3: `bash scripts/check_backend_registry.sh --static` exit 0; the route test asserts backend == selection.
4. Contract `contracts/apr-backend-registry-v1.yaml` invariants ii, iv, v discharged by the case table (kind: pattern, `pv validate`).
5. Mutations: (a) restore `cfg!` at accel.rs:28 → A1 RED; (b) print from the request instead of the Selection → the C11
   cross-check RED on a CPU host; (c) let a forced `--gpu` fall to cpu → the case-table row RED. Both CI run ids in the body.
6. P3: review-only (3-lane review of the diff was the design's ask; the design itself was judged with R-0).

K̂ [U] (no receipt of this class yet).
