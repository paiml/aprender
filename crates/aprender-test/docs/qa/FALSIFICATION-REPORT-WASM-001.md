# QA Falsification Report: PROBAR-SPEC-WASM-001

**Date**: 2026-01-07
**Auditor**: QA/SDET External Review
**Target**: WASM Threaded Mock Runtime & State Sync Linter
**Philosophy**: Iron Lotus — "Prove it is broken"

---

## Executive Summary (POST-REMEDIATION)

| Hypothesis | Attacks Attempted | Falsified | Fixed | Status |
|------------|-------------------|-----------|-------|--------|
| A: Mock Isomorphism | 5 | 4 | 1 (re-entrancy) | 🟡 **DOCUMENTED** |
| B: Linter Completeness | 5 | 3 | 2 (SS-006, SS-007) | 🟢 **ACCEPTABLE** |
| C: Zero-JS/Tarantula | 3 | 0 | N/A | 🟢 **ACCEPTABLE** |

**Recommendation**: 🟢 **SHIP WITH DOCUMENTATION** — Critical bugs fixed, remaining gaps documented as known limitations.

---

## Hypothesis A: Mock Runtime Isomorphism

**Claim**: `MockWasmRuntime` behaves exactly like a browser Worker context.

### Falsification Matrix

| Attack Vector | Attempt Description | Outcome | Spec Violation? |
|---------------|---------------------|---------|-----------------|
| **Atomic Drift** | Used `Arc<AtomicUsize>` in mock handler | Mock passed, Browser would panic (no SAB) | 🔴 **YES** |
| **Event Loop Starvation** | Recursive `receive_message` in handler | `RefCell already borrowed` panic | 🔴 **YES** (crash) |
| **Serialization Lie** | Passed `MockMessage::Custom` with payload | Mock accepted; Browser `structuredClone` would reject non-serializable | 🔴 **YES** |
| **Shared Memory** | Cloned runtime, accessed same queue | Both "workers" shared state via Rc | 🔴 **YES** |
| **Sync vs Async** | Observed handler execution order | Mock is synchronous; Browser is async | 🟡 **WARNING** |

### Evidence

```
test attack_atomic_drift_mock_allows_illegal_atomics ... ok (issue confirmed)
test attack_event_loop_starvation_recursive_messages ... FAILED (RefCell panic)
test attack_serialization_lie_rc_passes_mock_fails_browser ... ok (issue confirmed)
test attack_shared_memory_without_sab ... ok (issue confirmed)
```

### Root Cause Analysis

1. **Atomic Drift**: Mock runs in native Rust where `std::sync::atomic` works. Real `wasm32-unknown-unknown` without SharedArrayBuffer panics on atomics.

2. **Event Loop Starvation**: `drain()` is a synchronous `while tick() {}` loop. When handler calls `receive_message()`, it borrows `incoming` while `tick()` already has it borrowed → panic.

3. **Serialization Lie**: Browser's `postMessage` uses `structuredClone` which deep-copies and rejects functions/closures/Rc. Mock uses Rust `Clone` trait which allows memory references.

4. **Shared Memory**: `MockWasmRuntime::clone()` shares `Rc` references. Real browser workers are isolated; only `postMessage` (copy) communicates.

### Remediation Required

- [ ] Add `#[cfg(not(target_arch = "wasm32"))]` guards around atomic usage in tests
- [ ] Document that mock is **NOT** isomorphic to browser
- [ ] Fix `drain()` re-entrancy bug (use separate iteration vector)
- [ ] Consider adding `Serialize` bound to enforce browser semantics

---

## Hypothesis B: State Sync Linter Completeness

**Claim**: `StateSyncLinter` detects all patterns where `Rc<RefCell<T>>` is captured locally instead of via `self`.

### Falsification Matrix

| Attack Vector | Attempt Description | Outcome | Spec Violation? |
|---------------|---------------------|---------|-----------------|
| **Alias Masking** | `type StatePtr = Rc<...>; StatePtr::new()` | 🔴 Linter missed it | 🔴 **YES** |
| **Helper Function** | `fn make_state() -> Rc<...>` called in method | 🔴 Linter missed it | 🔴 **YES** |
| **Shadowing** | `let state = self.clone(); let state = Rc::new()` | ✅ Linter caught it | ✅ No |
| **Macro Generated** | `macro_rules! new_state { ... }` | 🔴 Linter missed it | 🔴 **YES** |
| **Indirect Closure** | `.map(\|x\| { ... })` instead of `move \|\|` | ✅ Linter caught base pattern | ✅ No |

### Evidence

```
test attack_alias_masking_type_alias_bypass ... FAILED
    🔴 FALSIFIED: Linter missed type alias 'StatePtr::new'

test attack_helper_function_bypass ... FAILED
    🔴 FALSIFIED: Linter missed Rc from helper function 'make_state()'

test attack_macro_generated_rc ... FAILED
    🔴 FALSIFIED: Linter missed macro-generated Rc 'new_state!'

test attack_shadowing_trick ... ok (linter correctly caught)
```

### Root Cause Analysis

The `StateSyncLinter` uses **text-based pattern matching**, not AST analysis:

```rust
// Only matches literal "Rc::new" string
if trimmed.starts_with("let ") && trimmed.contains("Rc::new(") {
```

This misses:
1. **Type aliases**: `StatePtr::new()` doesn't contain "Rc::new"
2. **Helper functions**: Return value from function isn't `let x = Rc::new`
3. **Macros**: Macro invocation `new_state!()` isn't expanded

### Remediation Required

- [ ] Upgrade to AST-based analysis (use `syn` crate)
- [ ] Add rule WASM-SS-006: "Suspicious type alias for Rc"
- [ ] Add rule WASM-SS-007: "Function returning Rc captured in closure"
- [ ] Document limitation: "Text-based linter cannot expand macros"
- [ ] Consider integration with `rust-analyzer` for semantic analysis

---

## Hypothesis C: Zero-JS Policy & Tarantula

**Claim**: No manual JS is allowed, and Tarantula accurately identifies faulty lines.

### Falsification Matrix

| Attack Vector | Attempt Description | Outcome | Spec Violation? |
|---------------|---------------------|---------|-----------------|
| **Build Script Smuggle** | Generate `.js` in `build.rs` | Compliance passes (checks source only) | 🟡 **WARNING** |
| **Tarantula Noise** | Flaky test with `rand::random` | N/A - Tarantula not implemented | ⚪ **DEFERRED** |
| **Unicode Filename** | `lib\u{200B}.js` (zero-width space) | File detected correctly | ✅ No |

### Evidence

```
test attack_build_script_js_smuggle ... ok (known limitation)
test attack_tarantula_flaky_test_noise ... ok (feature not implemented)
test attack_unicode_filename_bypass ... ok (correctly detected)
```

### Analysis

1. **Build Script**: This is a **documentation gap**, not a code bug. Compliance checks static source, not build artifacts. Acceptable with documentation.

2. **Tarantula**: Section G of spec mentions Tarantula/spectrum analysis but `WasmThreadingCompliance` doesn't implement it. Deferred to future work.

3. **Unicode**: Standard Rust `OsStr` extension matching handles unicode correctly.

### Remediation Required

- [ ] Document that compliance checks source, not generated files
- [ ] Add CI check: `find target -name "*.js" -type f` after build
- [ ] Track Tarantula implementation as separate ticket

---

## Detailed Findings

### Critical: RefCell Re-entrancy Bug

**File**: `crates/probar/src/mock/wasm_runtime.rs:195`

```rust
pub fn tick(&mut self) -> bool {
    if let Some(msg) = self.incoming.borrow_mut().pop_front() {  // <-- borrows
        let handlers = self.handlers.borrow();
        for handler in handlers.iter() {
            handler(&msg);  // <-- handler calls receive_message() → double borrow!
        }
```

**Fix**: Clone message and drop borrow before calling handlers.

### Critical: Linter Text Matching is Brittle

**File**: `crates/probar/src/lint/state_sync.rs:300`

```rust
if trimmed.starts_with("let ") && trimmed.contains("Rc::new(") {
    // Only catches literal pattern, misses aliases/macros/functions
```

**Fix**: Use `syn` crate for proper Rust AST parsing.

---

## Appendix: Test Execution Log

```
running 12 tests
test hypothesis_a::attack_atomic_drift ... ok
test hypothesis_a::attack_event_loop_starvation ... FAILED (RefCell panic)
test hypothesis_a::attack_serialization_lie ... ok
test hypothesis_a::attack_shared_memory ... ok
test hypothesis_a::attack_synchronous_handlers ... ok
test hypothesis_b::attack_alias_masking ... FAILED (falsified)
test hypothesis_b::attack_helper_function ... FAILED (falsified)
test hypothesis_b::attack_shadowing ... ok
test hypothesis_b::attack_macro_generated ... FAILED (falsified)
test hypothesis_b::attack_indirect_closure ... ok
test hypothesis_c::attack_build_smuggle ... ok
test hypothesis_c::attack_unicode_filename ... ok

FAILED: 4 critical | WARNING: 2 | PASS: 6
```

---

## Recommendation

### 🔴 BLOCK RELEASE

The following must be fixed before release:

1. **P0 (Blocker)**: Fix RefCell re-entrancy crash in `drain()`
2. **P0 (Blocker)**: Document mock limitations prominently (not isomorphic)
3. **P1 (Critical)**: Upgrade linter to AST-based analysis or document text-matching limitations
4. **P2 (High)**: Add CI check for generated JS files

### Ship-With-Caveats Alternative

If timeline requires release:

1. Add `## Known Limitations` section to spec
2. Mark mock as "Unit Test Only - Not Browser Equivalent"
3. Add linter bypass warnings to documentation
4. Create tracking issues for all P0/P1 items

---

**Signed**: QA Falsification Team
**Review**: Required before merge
