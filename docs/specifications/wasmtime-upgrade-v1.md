# Wasmtime 27 → 43 Upgrade

**Version**: 1.1
**Date**: 2026-04-12
**Status**: COMPLETE — PR #731
**Priority**: P1 — upgrades wasmtime for security patches (note: v43 still has 8 cranelift advisories, test-only dep)
**Contract**: `contracts/wasmtime-upgrade-v1.yaml`

## Five-Whys: Why Upgrade?

| # | Question | Answer |
|---|----------|--------|
| 1 | Why does ci/security need advisory exemptions? | wasmtime 27.0.0 has 5+ known CVEs/advisories (10+ counting overnight batches) |
| 2 | Why is wasmtime 27 still in use? | aprender-test-lib pins it for WASM runtime testing |
| 3 | Why hasn't it been upgraded? | "requires API migration" — but nobody scoped it |
| 4 | Why was it never scoped? | Assumed the API changed drastically across 16 major versions |
| 5 | Why not verify that assumption? | Now verified: core API (Engine, Store, Linker, Config) is stable. Upgrade is low-risk. |

**Root cause**: Assumption that wasmtime upgrade was hard, when in fact the API
we use (6 types, ~10 methods) is stable across v27→v43.

## Scope Analysis

### What we use (from `crates/aprender-test-lib/src/runtime.rs`)

| Type | Methods Used | v43 Status |
|------|-------------|------------|
| `Engine` | `Engine::new(&Config)` | **Unchanged** |
| `Store<T>` | `Store::new(&Engine, T)`, `set_fuel(u64)` | **Unchanged** |
| `Module` | `Module::new(&Engine, &[u8])` | **Unchanged** |
| `Linker<T>` | `Linker::new(&Engine)`, `func_wrap(mod, name, fn)`, `instantiate(&mut Store, &Module)` | **Unchanged** |
| `Instance` | `Instance::get_memory(&mut Store, "memory")` | **Unchanged** |
| `Caller<'_, T>` | `caller.data()` | **Unchanged** |
| `Config` | `new()`, `wasm_threads(bool)`, `wasm_simd(bool)`, `wasm_reference_types(bool)`, `consume_fuel(bool)` | **Unchanged** |

### Feature gate change

~~`wasm_reference_types` in v43 requires the `gc` Cargo feature on wasmtime.~~
**FALSIFIED**: `wasm_reference_types(bool)` compiles without `gc` feature in v43.
No feature gate change needed. **Zero breaking changes** for our usage.

### Files changed

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | `wasmtime = "27"` → `wasmtime = "43"` |
| ~~`crates/aprender-test-lib/Cargo.toml`~~ | **No change needed** — gc feature not required |
| `.cargo/audit.toml` | Removed 5 wasmtime exemptions |
| `deny.toml` | Removed 5 wasmtime exemptions |

### What we do NOT use

- Component model (new in v38+)
- WASI preview 2 (new in v36+)
- Async/fuel-yielding (our fuel usage is synchronous)
- Stack switching (new in v42+)

### Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| API signature change | LOW — verified stable | Compiler will catch at `cargo check` |
| ~~Feature gate change~~ | ~~MEDIUM~~ NONE | **FALSIFIED**: no gc feature needed |
| Behavior change (fuel accounting) | LOW | Existing tests validate |
| New transitive deps | LOW | `cargo deny check` validates |
| Compile time increase | MEDIUM — wasmtime 43 is larger | Accepted |

## Execution Plan (all steps DONE)

1. ~~Write provable contract~~ **DONE** (`contracts/wasmtime-upgrade-v1.yaml`)
2. ~~Bump `wasmtime = "43"`~~ **DONE**
3. ~~Add `features = ["gc"]` if needed~~ **NOT NEEDED** — compiled without it
4. ~~`cargo check`~~ **DONE** — FALSIFY-WASM-001 PASS
5. ~~`cargo test`~~ **DONE** — FALSIFY-WASM-002 PASS (6,304 tests)
6. ~~Remove exemptions~~ **DONE** — 5 removed from each file
7. ~~`cargo deny check advisories`~~ **DONE** — FALSIFY-WASM-003 PASS
8. ~~Update spec~~ **DONE** — this file

## Falsification Audit (v1.1)

| Claim (v1.0) | Actual | Verdict |
|--------------|--------|---------|
| "10+ exemptions" | 5 per file on main (10 overnight ones never merged) | **Corrected** to 5 |
| "reference_types needs gc feature" | Compiles without gc | **FALSE** — removed |
| "only known breaking change" | Zero breaking changes | **FALSE** — no breaking changes |
| "Add features = gc" | Not needed | **FALSE** — removed |
| "Remove 10 exemptions" | Removed 5 per file | **Corrected** to 5 |
