# QA Falsification Report: PROBAR-WASM-003 Final Boss

**Date:** 2026-01-07
**Target:** PROBAR-SPEC-WASM-001 Complete Implementation (Phase 3 "Final Sprint")
**Philosophy:** [Iron Lotus] — "If it compiles, break it at runtime. If it runs, exhaust its resources."

## Executive Summary

**Recommendation: HOTFIX REQUIRED**

3 critical vulnerabilities found, 4 semantic warnings documented.

---

## Falsification Matrix

| Attack Vector | Hypothesis | Outcome | Severity |
|---------------|------------|---------|----------|
| **Cfg-Gated Block** | A: Linter | ✅ PASS - Linter caught 2 Rc patterns across cfg branches | N/A |
| **Const Expression** | A: Linter | ✅ PASS - Linter caught Rc in nested block expression | N/A |
| **Raw Pointer Laundering** | A: Linter | 🔴 FALSIFIED - Linter missed Rc::new inside into_raw() | **HIGH** |
| **Macro-Expanded Rc** | A: Linter | ✅ PASS - Linter caught macro-style fully-qualified Rc::new | N/A |
| **Deep Recursion** | B: Mock | ✅ PASS - Serializer handled 500-level nesting | N/A |
| **Shared Reference Identity** | B: Mock | 🟡 DOCUMENTED - Clone semantics differ from browser | LOW |
| **Large Payload** | B: Mock | 🟡 WARNING - Mock accepted 10MB payload | LOW |
| **Zero-Coverage NaN** | C: Tarantula | ✅ PASS - Lines with no coverage excluded from report | N/A |
| **Coincidental Correctness** | C: Tarantula | 🟡 WARNING - High suspiciousness (0.5) for heavily-passed line | MEDIUM |
| **Empty LCOV** | C: Tarantula | ✅ PASS - Empty LCOV handled gracefully | N/A |
| **MIME-Type Smuggling** | D: Zero-JS | 🔴 FALSIFIED - Scanner missed JS in .txt file | **HIGH** |
| **Hidden Directory** | D: Zero-JS | 🔴 FALSIFIED - Scanner missed target/.hidden/malware.js | **HIGH** |
| **Symlink Escape** | D: Zero-JS | ✅ PASS - Scanner followed symlink and found external JS | N/A |
| **Unicode RTL Override** | D: Zero-JS | 🟡 DOCUMENTED - Visual spoofing out of scope | INFO |

---

## Critical Findings (🔴 FALSIFIED)

### 1. Raw Pointer Laundering Bypasses Linter

**Attack Code:**
```rust
let ptr = Rc::into_raw(Rc::new(RefCell::new(0)));
let state = unsafe { Rc::from_raw(ptr) };

// state is Rc, but linter only caught the inner Rc::new()
let callback = move || { state.borrow_mut(); };
```

**Impact:** The final `state` variable holds a disconnected `Rc`, but the linter only flagged the nested `Rc::new()`, not the overall pattern. Developers may see the warning but not understand that `from_raw` launders the Rc.

**Recommendation:** Add detection for `Rc::from_raw` and `Rc::into_raw` patterns. Flag any local variable of type `Rc<_>` captured in closures, regardless of creation method.

---

### 2. Zero-JS Scanner: MIME-Type Smuggling

**Attack:** Create `target/payload.txt` containing valid JavaScript code.

**Impact:** The scanner only checks `.js` extensions. A malicious build.rs could write JavaScript to a `.txt`, `.json`, or extensionless file and bypass detection.

**Recommendation:**
1. Check file content for JS signatures (shebang, `function`, `const`, etc.)
2. OR restrict to known safe extensions only (whitelist approach)

---

### 3. Zero-JS Scanner: Hidden Directory Blind Spot

**Attack:** Write to `target/.hidden/malware.js`

**Finding:** Scanner skips dot-directories (`.hidden/`), allowing hidden JS files.

**Impact:** Build scripts can hide contraband in `.cache/`, `.build/`, or other hidden directories within target/.

**Recommendation:** Remove the `!name.starts_with('.')` filter for target/ scanning. Hidden directories in target/ are suspicious by definition.

---

## Warnings (🟡)

### 1. Tarantula: Coincidental Correctness Problem

**Scenario:** Line executed 1000 times in passing tests, 1 time in failing test.
**Suspiciousness Score:** 0.5 (HIGH)

This is a known limitation of the Tarantula formula. A line executed heavily by passing tests is probably NOT the bug, but Tarantula weights by test count, not execution count.

**Recommendation:** Consider implementing Ochiai or DStar formulas as alternatives:
- **Ochiai:** `failed(s) / sqrt(total_failed * (failed(s) + passed(s)))`
- **DStar:** `failed(s)^2 / (passed(s) + (total_failed - failed(s)))`

### 2. Mock: Clone vs Identity Semantics

**Impact:** Browser `structuredClone` preserves object identity for shared references within a message. bincode serializes each occurrence independently.

**Mitigation:** Document this as a known semantic difference. Code relying on shared reference identity should use explicit clone() in the message.

### 3. Mock: No Size Limits

**Impact:** Mock accepts arbitrarily large payloads. Browser postMessage has implementation-dependent size limits (varies by browser).

**Mitigation:** Consider adding configurable size limits to mock for stricter testing.

### 4. Unicode Visual Spoofing

**Documented:** The scanner correctly handles Unicode filenames by their actual extension, not visual appearance. RTL override attacks are a visual spoofing concern, not a file system bypass.

---

## Test Results Summary

```
running 14 tests
test attack_cfg_gated_rc_creation ... ok
test attack_const_expr_rc_creation ... ok
test attack_raw_pointer_laundering ... ok (🔴 FALSIFIED detected)
test attack_macro_expanded_rc ... ok
test attack_deep_recursion_stack_overflow ... ok
test attack_shared_reference_identity_loss ... ok
test attack_large_payload ... ok
test attack_tarantula_zero_coverage_nan ... ok
test attack_tarantula_coincidental_correctness ... ok (🟡 WARNING detected)
test attack_tarantula_empty_lcov ... ok
test attack_mime_type_smuggling ... ok (🔴 FALSIFIED detected)
test attack_hidden_directory_traversal ... ok (🔴 FALSIFIED detected)
test attack_symlink_escape ... ok
test attack_unicode_filename_bypass ... ok

test result: ok. 14 passed; 0 failed
```

---

## Compliance Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| AST Linter (P1) | 🟡 PARTIAL | Works for most patterns; raw pointer laundering escapes |
| Mock Serialization (P2) | ✅ PASS | bincode round-trip works; semantic differences documented |
| Tarantula (P2) | 🟡 PARTIAL | Formula works; coincidental correctness is known limitation |
| Zero-JS Scanner (P3) | 🔴 FAIL | Hidden directories and MIME-type smuggling bypass |

---

## Hotfix Priority

1. **P0 (Block Ship):** Hidden directory bypass in Zero-JS scanner
2. **P0 (Block Ship):** MIME-type smuggling in Zero-JS scanner
3. **P1 (Next Sprint):** Raw pointer laundering detection in linter
4. **P2 (Backlog):** Alternative fault localization formulas (Ochiai/DStar)

---

## Sign-Off

**QA Verdict:** HOTFIX REQUIRED before PROBAR-WASM-003 can ship.

The Zero-JS scanner vulnerabilities are critical - a malicious crate could easily bypass the "no JS in target/" check. The linter raw pointer bypass is concerning but lower priority since it requires unsafe code.

---

*Generated by Iron Lotus Falsification Protocol*
*"Your job is not to verify that it works. Your job is to prove that it is broken."*
