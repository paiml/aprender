---
phase: 6
slug: native-time-series-forecasting-stack
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-05
updated: 2026-09-05
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Seeded by plan-phase from
> `06-RESEARCH.md` § Validation Architecture (measured on this branch, 2026-09-05); the per-task map
> is completed by the planner's task IDs and finalised by `/gsd-validate-phase 6`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust libtest via `cargo test` 1.93.0; CI runs `cargo nextest run --profile ci` 0.9.102 (`.config/nextest.toml`: retries 2, fail-fast, slow-timeout 60 s × 20) |
| **Config file** | `.config/nextest.toml` (existing); root `Cargo.toml` gains `[profile.dev.package.aprender-forecast] opt-level = 3` iff 06-01 Task 2's measured debug e2e fit wall projects the seven-fixture ladder past 60 s (decided in wave 1 so wave 2 reads a stable root manifest; debug-profile Prophet fits measured 35–39× slower than release — 9.15 s vs 0.26 s for one Peyton round) |
| **Quick run command** | `cargo test -p aprender-forecast --lib prophet::parity::peyton` (one fixture, one ladder) |
| **Full suite command** | `cargo nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute` (CI's exact leg) **plus** `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` (the two CI-wired drift gates, currently 5 failures on this branch) |
| **Estimated runtime** | quick ≈ 1–6 s release / up to 40 s debug; full workspace lib leg tens of minutes (80k tests); the two drift gates < 10 s |

---

## Sampling Rate

- **After every task commit:** `cargo clippy -p <crate> --all-targets -- -D warnings && cargo fmt --all -- --check && cargo test -p <crate> --lib`
- **After every plan wave:** CI's nextest lib leg + `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` + `make contract-validate`
- **Before `/gsd-verify-work`:** all of the above green, plus `just chronos-gate` (embedded build, `0 ignored`), `just forecast-bench`, `just chronos-coldstart` on the aarch64 host with results recorded as evidence; the FALSIFY-MONO-011 allowlist decision recorded
- **Max feedback latency:** 60 s for the per-task quick loop (release-profile parity tests); the full leg is per wave, not per task

---

## Per-Task Verification Map

Task IDs are assigned by the planner; the rows below are the requirement → test mapping the plans must realise (from `06-RESEARCH.md` § Validation Architecture). `/gsd-validate-phase 6` rewrites this table against the executed plans.

**Regenerated 2026-09-05 against the plan frontmatter as committed** (the wave column is read from each plan's `wave:` key, the task IDs from each plan's `<name>Task N:` elements — not reconstructed from an earlier draft). Three shifts the `--reviews` replan introduced and this table now carries: the np.rs port moved from 06-01 into 06-04 (so there is no `06-01-T3`); 06-05 moved from wave 2 to wave 3 on a declared `06-04` dependency, pushing 06-06/07/08/09 to waves 4/5/6/7; and 06-05 gained a blocking `checkpoint:decision` as its Task 2, renumbering its port to T3 and its contract + gated ladder to T4. Checkpoint tasks (06-02-T1, 06-05-T2, 06-08-T3) carry no `<automated>` verify by construction and so have no row here — they are governance gates, listed under Wave 0 Requirements.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-T1 | 06-01 | 1 | D-01 / D-06 / SC1 tracer (one Prophet round trip over in-process streamable-HTTP; stdio via run_stdio) | T-06-01 | unknown fields refused at the one door | tracer (`--lib` e2e) | `cargo test -p aprender-mcp-forecast --lib e2e` | ❌ W1 | ⬜ pending |
| 06-01-T2 | 06-01 | 1 | D-04 Peyton rung 2 on the 17 byte-verified fixtures + the MEASURED `[profile.dev.package.aprender-forecast]` decision (RESEARCH Pitfall 9) | T-06-10 | fixtures byte-identical to the spike originals; the profile table exists only on a measured wall | unit (parity module) + shell (17-file `cmp` loop; timed e2e) | `cargo test -p aprender-forecast --lib prophet::parity`; the `cmp` loop; timed `cargo test -p aprender-mcp-forecast --lib e2e` | ❌ W1 | ⬜ pending |
| 06-02-T2 | 06-02 | 1 | FALSIFY-MONO-011 allowlist per the human decision (SC5) | T-06-08 | ratchet still fails on growth (two-sided control) | integration (CI-wired) | `cargo test -p aprender-core --test monorepo_invariants` | ✅ exists — red today | ⬜ pending |
| 06-02-T3 | 06-02 | 1 | readme_contract CRATE-001/002 (three README fixes) | — | N/A | integration (CI-wired) | `cargo test -p aprender-core --test readme_contract --no-fail-fast` (counts stay red until 06-09) | ✅ exists — red today | ⬜ pending |
| 06-03-T2 | 06-03 | 2 | D-04 / SC2 (Prophet parity ladder on 7 fixtures; bars read from `prophet-parity-v1.yaml`) | T-06-10 | tolerances live in one pv-validated contract | unit (`--lib`; profile decided in 06-01-T2 — warm ladder wall recorded here, no manifest edit) | `cargo test -p aprender-forecast --lib prophet::parity` | ❌ W2 | ⬜ pending |
| 06-04-T1 | 06-04 | 2 | D-02 / D-10 / D-11 (np.rs ported verbatim; the `neuralprophet` arm replaces 06-01's refusing stub; dispatch, residual band, the three NP refusals; tape empty after the fit) | T-06-12 | fits never interleave on one thread | unit | `cargo test -p aprender-forecast --lib forecast::` | ❌ W2 | ⬜ pending |
| 06-04-T3 | 06-04 | 2 | D-10 / SC3 (NP data prep vs `np_oracle_peyton.json`; lag-free MAE ≤ 0.47; `n_lags = 30` beats naive; Huber connected) | T-06-12 | tape cleared per step | unit | `cargo test -p aprender-forecast --lib np::parity` | ❌ W2 | ⬜ pending |
| 06-05-T4 | 06-05 | 3 | D-13 / D-14 / SC4 (Bolt ladder 1e-6 f32; probes; 365 rollout; f16 within 2 % of std) + D-18 counted skip both ways | T-06-13 | never a vacuous weight test | unit, `cfg_attr`-ignored without weights | `CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f32 cargo test -p aprender-forecast --lib -- bolt::parity chronos::parity` (and the same unarmed) | ❌ W3 + `just fetch-chronos-tiny` (06-05-T1) | ⬜ pending |
| 06-06-T2 | 06-06 | 4 | D-11 / SC1 (every refusal is `pmcp::Error::validation` naming the fix; prophet + NP happy paths) | T-06-01 input validation | malformed input never reaches a fit | unit `#[cfg(test)] mod e2e` | `cargo test -p aprender-mcp-forecast --lib e2e::` | ❌ W4 | ⬜ pending |
| 06-06-T3 | 06-06 | 4 | D-12 / SC5 (8 concurrent and 16 concurrent bit-identical) + SC1 stdio round trip. The wall-clock ratio bar is NOT this test's — it belongs to `just forecast-pool-ratio` in 06-08 | T-06-02 resource exhaustion | bounded pool, per-request budget | unit — **equality only; no timing assertion anywhere in `mod pool_equality`** (REVIEW-06-04 removed it: with 4 Tokio workers and 8 heterogeneous blocking fits the ratio moves with CPU throttling, and a flaky bar inside the correctness suite buys nothing). The test instead prints exactly one machine-parsable `POOL SPEEDUP:` line carrying ratio, both walls, `arch`, `profile`, `workers`, `cpus`, which 06-08's recipe greps + `tests/e2e_stdio.rs` (dark in CI until 06-08's decision) | `cargo test -p aprender-mcp-forecast --lib pool_equality -- --nocapture` (verify asserts 2 passed and exactly one well-formed `POOL SPEEDUP:` line — never a ratio); `cargo test -p aprender-mcp-forecast --test e2e_stdio` | ❌ W4 | ⬜ pending |
| 06-07-T2 | 06-07 | 5 | D-03 / D-11 / D-13 / SC4 (Chronos refusals; `allow_long_horizon` → `warning` + `forwards: 46`; 1e-6 / 2 % parity through the server; shared-shape invariant) | T-06-01 | as above | unit, `cfg_attr`-ignored without weights + ungated invariants | `CHRONOS_MODEL_DIR=… cargo test -p aprender-mcp-chronos --lib` (and unarmed) | ❌ W4 | ⬜ pending |
| 06-07-T3 | 06-07 | 5 | D-13 / D-18 (embedded build resolves from include_bytes!; arming and embedding independent) | T-06-06 | weights only from the pinned fetch | unit (embed env) | `CHRONOS_EMBED_DIR=… CHRONOS_MODEL_DIR=… cargo test -p aprender-mcp-chronos --lib` | ❌ W5 | ⬜ pending |
| 06-08-T2 | 06-08 | 6 | SC1 (< 2 s), SC4 (< 100 ms, < 30 MB, < 150 ms), SC5 (< ½ wall), D-18 clause 2 (local `just chronos-gate`) | T-06-14 | evidence carries host/commit/log | host-gated just recipes (aarch64 release) → `06-EVIDENCE.md` | `just chronos-gate && just chronos-embed-build && just chronos-coldstart 5 && just chronos-bench && just forecast-bench && just forecast-pool-ratio` | ❌ W5 | ⬜ pending |
| 06-03-T1, 06-04-T2, 06-05-T4, 06-06-T1, 06-09-T1 | 06-03/04/05/06/09 | 2–7 | D-15 (the FOUR contracts validate — `prophet-parity-v1` (06-03-T1), `neuralprophet-parity-v1` (06-04-T2), `chronos-bolt-parity-v1` (06-05-T4), `forecast-tool-boundary-v1` (06-06-T1) — counts non-zero; all four wired into `$(CONTRACTS)` by 06-09-T1; zero BIND-) | T-06-17 | a contract validated by nothing is decoration | shell gate | `cargo run -p aprender-contracts-cli --bin pv -- validate contracts/<f>.yaml` + `pv status`; `make contract-validate`; `make contract-audit-phase6` | ❌ W2–W7 | ⬜ pending |
| 06-09-T3 | 06-09 | 7 | SC5 lint/fmt/check + nextest lib leg on the three crates | — | N/A | shell | `cargo clippy -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --all-targets -- -D warnings && cargo fmt --all -- --check && cargo check --workspace --exclude aprender-profile && cargo nextest run --profile ci -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --lib` | ✅ commands exist | ⬜ pending |
| 06-09-T2 | 06-09 | 7 | F7 drift gates (README counts re-derived, CLAUDE.md D-07 row paths, allowlist) | T-06-18 | counts pasted from the derivation commands | integration (CI-wired) | `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` | ✅ exists — **5 failures today** | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/aprender-forecast/tests/fixtures/` — DECIDED (06-01): COPY the 17 distinct committed oracle files (13 JSON + 4 CSV) from spikes 001, 002, 003, 005, 006, 007, byte-verified with `cmp` (RESEARCH Open Question 2). Lambda wrapper crates DEFERRED (Open Question 3). SmoothL1Loss filed as a core ticket in 06-09 (Open Question 4). Timing bars host-gated via `just` recipes in 06-08 (Open Question 5).
- [ ] `crates/aprender-forecast/build.rs` + `crates/aprender-mcp-chronos/build.rs` — `cargo::rustc-check-cfg=cfg(chronos_weights)` + `rustc-cfg` when `CHRONOS_MODEL_DIR/model.safetensors` exists (RESEARCH F6); server `build.rs` also stages `CHRONOS_EMBED_DIR` (D-13)
- [ ] `justfile` recipes: `fetch-chronos-tiny` (pinned revision `a0e552de83495b5c28c14c71c374f3e33280b340`, sha256-checked), `chronos-gate`, `forecast-bench`, `chronos-coldstart`
- [ ] `models/chronos-bolt-tiny/{f32,f16}/` populated locally (gitignored — `*.safetensors` is ignored globally); shas recorded in the crate README
- [ ] Root `Cargo.toml`: three `members` lines; `[profile.dev.package.aprender-forecast] opt-level = 3` decided in 06-01 Task 2 from the measured debug e2e fit wall (Pitfall 9) — no wave-2 plan edits the root manifest
- [ ] FOUR contracts, all on the `setfit-apr-v1.yaml` shape (NOT `neon-blis-v1.yaml`, which `pv status` shows as hollow) + Makefile `$(CONTRACTS)` lines: `contracts/prophet-parity-v1.yaml` (06-03-T1), `contracts/neuralprophet-parity-v1.yaml` (06-04-T2), `contracts/chronos-bolt-parity-v1.yaml` (06-05-T4), `contracts/forecast-tool-boundary-v1.yaml` (06-06-T1). The NP contract is the fourth: the phase ships one parity contract per port plus the shared tool boundary, and `make contract-audit-phase6` (06-09-T1) reports 4 audited
- [ ] Human checkpoint (06-02-T1, blocking): FALSIFY-MONO-011 `[[bin]]` allowlist (`crates/aprender-core/tests/monorepo_invariants.rs:235-395`) — an exemption class for thin MCP deployment units, baseline 27 → 33, decided before any executor edits it
- [ ] Human checkpoint (06-05-T2, blocking): the D-13 memory clause. D-13 says "Only transposed `[in, out]` weights are kept in memory"; D-14 says single rows go through `dot8`, which needs contiguous `[out, in]` rows — the two cannot both hold. The planner's recommendation is `amend-memory-clause` (keep both layouts, state the residency cost, preserve the measured 9.54e-7 aarch64 parity), but a locked CONTEXT decision is the human's to amend, so the branch is decided before 06-05-T3 ports `bolt.rs`. Recorded as `D-ITEM-06-08` in `deferred-items.md` by 06-09-T3 and grep-gated there
- [ ] Human checkpoint (06-08-T3, blocking): the `.github/workflows/ci.yml` embedded-weights leg and the `e2e_stdio` target — applied, if approved, by 06-09-T3
- [ ] README fixes the gates demand: `README.md` crate/contract counts re-derived; `crates/aprender-mcp-setfit/README.md` monorepo link; READMEs for `aprender-mcp-setfit-lambda` and `aprender-contrastive-data`; three new crate READMEs with the `paiml/aprender` link
- [ ] Framework install: none — cargo, nextest, just, uv present

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Peyton round trip < 2 s; 2 048-pt Chronos forward < 100 ms | SC1, SC4 | Timing is host- and profile-dependent; the X64 CI box and debug builds cannot assert it | `just forecast-bench`, `just chronos-bench` on the aarch64 host, release profile; paste the tables into the SUMMARY |
| Release binary < 30 MB with embedded tiny-f16 | SC4 | Needs the embedded build, which has no CI leg without a workflow edit (human check-in) | `CHRONOS_EMBED_DIR=models/chronos-bolt-tiny/f16 cargo build --release -p aprender-mcp-chronos && ls -l target/release/aprender-mcp-chronos` |
| Cold start to first forecast over stdio < 150 ms | SC4 | Spawn timing, aarch64 release only | `just chronos-coldstart 3` (spike-007 harness) |
| Demo page performs initialize → tools/list → tools/call and charts the result | D-06 (demo page copied verbatim) | Browser interaction | `cargo run -p aprender-mcp-forecast -- --http 8787`, open `http://127.0.0.1:8787/`, load Peyton, press Forecast; repeat for `aprender-mcp-chronos --http 8788` |
| Speed-up from the router pool (3.9× measured in spike 010) | SC5 | Wall-clock ratio needs cores and release profile; the `pool_equality` unit test asserts equality ONLY and never the ratio (REVIEW-06-04) | `just forecast-pool-ratio` on the aarch64 host (06-08-T2) — the recipe runs `pool_equality` release with `--nocapture` and greps the single `POOL SPEEDUP:` line the test prints (ratio, both walls, `arch`, `profile`, `workers`, `cpus`); that recipe, not the unit test, is where the SC5 wall-clock bar is asserted, and its output goes into `06-EVIDENCE.md` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60 s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
