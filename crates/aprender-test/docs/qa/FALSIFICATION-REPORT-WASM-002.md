# QA Falsification Report: PROBAR-WASM-002 Regressions

**Date**: 2026-01-07
**Auditor**: QA/SDET External Review
**Target**: PROBAR-WASM-002 "Fix" Claims
**Philosophy**: [Iron Lotus] — "The developer claims they fixed the bugs. Your job is to prove they only fixed the symptoms."

---

## Executive Summary

| Hypothesis | Attacks Attempted | Falsified | Severity | Status |
|------------|-------------------|-----------|----------|--------|
| A: Swap-Based Handler Fix | 2 | 1 | 🔴 CRITICAL | **FIX REQUIRED** |
| B: Linter Rules 006/007 | 5 | 4 | 🟡 WARNING | **DOCUMENT** |

**Recommendation**: 🔴 **FIX REQUIRED** — Panic Bomb attack reveals handler loss vulnerability.

---

## Hypothesis A: "The Swap-Based Handler Fix Is Robust"

**Claim**: Using `std::mem::swap` to move handlers out during execution prevents RefCell panics.

### Regression Matrix

| Attack Vector | Attempt Description | Outcome | Spec Violation? |
|---------------|---------------------|---------|-----------------|
| **Handler Mutation** | Register handler inside handler | 🟡 Architecturally blocked | 🟡 **WARNING** (API limitation) |
| **Panic Bomb** | Handler panics during execution | 🔴 ALL handlers LOST | 🔴 **YES** (Data Loss) |
| **Receive Message** | Call receive_message in handler | ✅ Works correctly | ✅ No |

### Evidence

```
test regression_handler_mutation_during_tick ... ok
🟡 WARNING: Handler registration during tick is architecturally blocked.
   on_message(&mut self) cannot be called from Fn handlers.
   The swap-based fix is irrelevant to this use case.
✅ PASS: receive_message in handler works. Counter = 3 (expected 3+)

test regression_panic_bomb_handler_loss ... ok
🔴 FALSIFIED: Panic in handler caused ALL handlers to be lost!
   The swap-based fix lacks a Drop guard to restore handlers on panic.
   RECOMMENDATION: Add scopeguard or manual Drop impl to restore handlers.

test regression_panic_complete_wipeout ... ok
🔴 CONFIRMED: Panic caused handler wipeout. Counter stayed at 0.
```

### Root Cause Analysis

**Panic Bomb** (CRITICAL):

The swap-based fix at `wasm_runtime.rs:207-239` works like this:

```rust
pub fn tick(&mut self) -> bool {
    // Step 2: Swap out handlers
    let handlers_to_run = {
        let mut handlers = self.handlers.borrow_mut();
        std::mem::take(&mut *handlers)  // ← Handlers now in LOCAL variable
    };

    // Step 3: Run handlers (NO borrows held)
    for handler in &handlers_to_run {
        handler(&msg);  // ← IF THIS PANICS, stack unwinds!
    }

    // Step 4: Merge back (NEVER REACHED if panic!)
    {
        let mut handlers = self.handlers.borrow_mut();
        *handlers = handlers_to_run;  // ← Never executed
        handlers.extend(new_handlers);
    }
}
```

When a handler panics:
1. `handlers_to_run` is on the stack, holding all handlers
2. Stack unwinds, `handlers_to_run` is dropped
3. `self.handlers` is now empty (from `take`)
4. Step 4 never executes → **All handlers permanently lost**

**Handler Mutation** (WARNING):

The `on_message(&mut self, ...)` API requires mutable access, but handlers are `Fn` closures (not `FnMut`). This is an architectural limitation, not a bug in the swap fix. Users cannot call `on_message` from within handlers regardless of the fix.

### Remediation Required

- [ ] **P0 (Blocker)**: Add `scopeguard` or manual Drop guard to restore handlers on panic
- [ ] **P2 (Medium)**: Consider changing handlers to `FnMut` to allow mutation during tick
- [ ] **P2 (Medium)**: Document that handler panics corrupt runtime state

**Suggested Fix**:

```rust
use scopeguard::defer;

pub fn tick(&mut self) -> bool {
    let msg = self.incoming.borrow_mut().pop_front();

    if let Some(msg) = msg {
        let handlers_to_run = {
            let mut handlers = self.handlers.borrow_mut();
            std::mem::take(&mut *handlers)
        };

        // CRITICAL: Restore handlers even if panic occurs
        let handlers_ref = &self.handlers;
        defer! {
            let mut handlers = handlers_ref.borrow_mut();
            if handlers.is_empty() {
                // Panic occurred - restore from backup
                *handlers = handlers_to_run.clone();
            }
        }

        for handler in &handlers_to_run {
            handler(&msg);
        }

        // Normal path: merge handlers
        {
            let mut handlers = self.handlers.borrow_mut();
            let new_handlers = std::mem::take(&mut *handlers);
            *handlers = handlers_to_run;
            handlers.extend(new_handlers);
        }

        self.messages_processed += 1;
        true
    } else {
        false
    }
}
```

---

## Hypothesis B: "Linter Rules 006 & 007 Are Comprehensive"

**Claim**: The linter now detects type aliases and helper functions returning Rc.

### Regression Matrix

| Attack Vector | Attempt Description | Outcome | Spec Violation? |
|---------------|---------------------|---------|-----------------|
| **Deep Generic** | `MyWrapper::<T>::new()` turbofish | 🔴 Linter missed it | 🔴 **YES** |
| **Method Chain** | `.to_rc()` trait method | ✅ Caught (false positive?) | ⚪ Needs review |
| **Trait Object** | `Box<dyn T>` hiding Rc | 🔴 Linter missed it | 🔴 **YES** |
| **Rc::default()** | Alternative constructor | 🔴 Linter missed it | 🔴 **YES** |
| **Arc::new()** | Thread-safe variant | 🟡 Not in scope | 🟡 **WARNING** |

### Evidence

```
test regression_deep_generic_type_alias_bypass ... ok
🔴 FALSIFIED: Linter detected type alias but missed `MyWrapper::<State>::new()` usage!
   Pattern `Alias::<T>::new()` (turbofish) bypasses detection.

test regression_method_chain_bypass ... ok
✅ PASS: Linter caught method chain returning Rc
   (Note: This may be a false positive - needs investigation)

test regression_trait_object_hidden_rc ... ok
🔴 FALSIFIED: Linter missed Rc hidden inside `Box<dyn StateProvider>`!
   Functions returning trait objects can hide Rc internally.

test regression_renamed_constructor_bypass ... ok
🔴 FALSIFIED: Linter missed `Rc::default()`!
   Only `Rc::new()` is detected, not other constructors.

test regression_arc_instead_of_rc_bypass ... ok
🟡 WARNING: Linter doesn't check for `Arc::new()`!
   Arc has same state desync issue as Rc. Consider adding Arc rules.
```

### Root Cause Analysis

**Deep Generic (SS-006)**:

The type alias detection at `state_sync.rs:341` looks for `type Foo = Rc<`:

```rust
if trimmed.starts_with("type ") && trimmed.contains("Rc<") {
    // Extract alias name: "MyWrapper"
}
```

Then usage detection at `state_sync.rs:402` looks for `AliasName::new(`:

```rust
let pattern = format!("{alias}::new(");  // "MyWrapper::new("
if trimmed.contains(&pattern) { ... }
```

But `MyWrapper::<State>::new(` doesn't match `MyWrapper::new(` because of the turbofish `::<State>` between them.

**Trait Object**:

The function return type check at `state_sync.rs:358` looks for `-> Rc<`:

```rust
if trimmed.contains("fn ") && trimmed.contains("-> Rc<") {
```

`fn create_state_provider() -> Box<dyn StateProvider>` doesn't contain `-> Rc<` in the signature, even though the implementation creates and stores an Rc internally.

**Alternative Constructors**:

Only `Rc::new(` is pattern-matched. Other constructors like `Rc::default()`, `Rc::from()`, `<Rc<T>>::default()` are not detected.

### Remediation Required

- [ ] **P1 (High)**: Update SS-006 pattern to handle turbofish: `AliasName::<...>::new(`
- [ ] **P1 (High)**: Add SS-008 for `Rc::default()` and other constructors
- [ ] **P2 (Medium)**: Document that trait objects can hide Rc (semantic analysis required)
- [ ] **P3 (Low)**: Consider adding Arc rules (same desync issue)

**Suggested Fix for SS-006**:

```rust
// Current: only matches "AliasName::new("
let pattern = format!("{alias}::new(");

// Fixed: also matches "AliasName::<...>::new(" via regex
let patterns = [
    format!("{alias}::new("),           // Direct
    format!("{alias}::<"),              // Turbofish start
];
for pattern in &patterns {
    if trimmed.contains(pattern) && trimmed.contains("::new(") {
        // Detected!
    }
}
```

---

## Appendix: Test Execution Log

```
running 8 tests
test regression_handler_mutation_during_tick ... ok
test regression_panic_bomb_handler_loss ... ok
test regression_panic_complete_wipeout ... ok
test regression_deep_generic_type_alias_bypass ... ok
test regression_method_chain_bypass ... ok
test regression_trait_object_hidden_rc ... ok
test regression_renamed_constructor_bypass ... ok
test regression_arc_instead_of_rc_bypass ... ok

test result: ok. 8 passed; 0 failed; 0 ignored
```

All tests pass because they **document** findings rather than fail on them.

---

## Recommendation

### 🔴 FIX REQUIRED (Before Merge)

1. **P0**: Fix Panic Bomb vulnerability with scopeguard Drop guard
2. **P1**: Update SS-006 for turbofish generic syntax
3. **P1**: Add SS-008 for `Rc::default()` and other constructors

### Ship-With-Caveats Alternative

If timeline requires release:

1. Add `## Known Limitations` section to spec documenting:
   - Handler panics corrupt runtime state
   - Turbofish generics bypass SS-006
   - Trait objects can hide Rc
   - Only `Rc::new()` is detected
2. Create tracking issues for all P0/P1 items
3. Add explicit warnings in doc comments

---

**Signed**: QA Falsification Team
**Review**: Required before merge
