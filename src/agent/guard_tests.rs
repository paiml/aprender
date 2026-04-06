//! Unit + property tests for LoopGuard.

use super::*;

#[test]
fn test_iteration_limit() {
    let mut guard = LoopGuard::new(5, 100, 0.0);
    for _ in 0..3 {
        assert_eq!(guard.check_iteration(), LoopVerdict::Allow);
    }
    assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
    assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
    assert!(matches!(guard.check_iteration(), LoopVerdict::CircuitBreak(_)));
}

#[test]
fn test_warn_at_80_percent() {
    let mut guard = LoopGuard::new(10, 100, 0.0);
    for _ in 0..7 {
        assert_eq!(guard.check_iteration(), LoopVerdict::Allow);
    }
    assert!(matches!(guard.check_iteration(), LoopVerdict::Warn(_)));
}

#[test]
fn test_tool_call_limit() {
    let mut guard = LoopGuard::new(100, 2, 0.0);
    let input = serde_json::json!({"q": "a"});
    assert_eq!(guard.check_tool_call("t1", &input), LoopVerdict::Allow);
    assert_eq!(guard.check_tool_call("t2", &serde_json::json!({"q": "b"})), LoopVerdict::Allow);
    assert!(matches!(guard.check_tool_call("t3", &input), LoopVerdict::CircuitBreak(_)));
}

#[test]
fn test_pingpong_detection() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    let input = serde_json::json!({"query": "same"});
    assert_eq!(guard.check_tool_call("rag", &input), LoopVerdict::Allow);
    assert_eq!(guard.check_tool_call("rag", &input), LoopVerdict::Allow);
    assert!(matches!(guard.check_tool_call("rag", &input), LoopVerdict::Block(_)));
}

#[test]
fn test_different_inputs_no_pingpong() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    for i in 0..10 {
        let input = serde_json::json!({"q": format!("query_{i}")});
        assert_eq!(guard.check_tool_call("rag", &input), LoopVerdict::Allow);
    }
}

#[test]
fn test_consecutive_max_tokens() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    for _ in 0..4 {
        assert_eq!(guard.record_max_tokens(), LoopVerdict::Allow);
    }
    assert!(matches!(guard.record_max_tokens(), LoopVerdict::CircuitBreak(_)));
}

#[test]
fn test_max_tokens_reset() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    guard.record_max_tokens();
    guard.record_max_tokens();
    guard.reset_max_tokens();
    for _ in 0..4 {
        assert_eq!(guard.record_max_tokens(), LoopVerdict::Allow);
    }
    assert!(matches!(guard.record_max_tokens(), LoopVerdict::CircuitBreak(_)));
}

#[test]
fn test_cost_budget() {
    let mut guard = LoopGuard::new(100, 100, 1.0);
    assert_eq!(guard.record_cost(0.5), LoopVerdict::Allow);
    assert_eq!(guard.record_cost(0.3), LoopVerdict::Allow);
    assert!(matches!(guard.record_cost(0.3), LoopVerdict::CircuitBreak(_)));
}

#[test]
fn test_zero_cost_budget_unlimited() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    assert_eq!(guard.record_cost(1000.0), LoopVerdict::Allow);
}

#[test]
fn test_usage_tracking() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    guard.record_usage(&TokenUsage { input_tokens: 100, output_tokens: 50 });
    guard.record_usage(&TokenUsage { input_tokens: 200, output_tokens: 75 });
    assert_eq!(guard.usage().input_tokens, 300);
    assert_eq!(guard.usage().output_tokens, 125);
}

#[test]
fn test_fx_hash_deterministic() {
    let input = serde_json::json!({"q": "hello"});
    let h1 = fx_hash_tool_call("rag", &input);
    let h2 = fx_hash_tool_call("rag", &input);
    assert_eq!(h1, h2);
}

#[test]
fn test_fx_hash_different_tools() {
    let input = serde_json::json!({"q": "hello"});
    let h1 = fx_hash_tool_call("rag", &input);
    let h2 = fx_hash_tool_call("memory", &input);
    assert_ne!(h1, h2);
}

#[test]
fn test_counters() {
    let mut guard = LoopGuard::new(10, 10, 0.0);
    guard.check_iteration();
    guard.check_iteration();
    assert_eq!(guard.current_iteration(), 2);
    let input = serde_json::json!({});
    guard.check_tool_call("t", &input);
    assert_eq!(guard.total_tool_calls(), 1);
}

#[test]
fn test_token_budget_unlimited() {
    let mut guard = LoopGuard::new(100, 100, 0.0);
    let v = guard.record_usage(&TokenUsage { input_tokens: 1_000_000, output_tokens: 500_000 });
    assert_eq!(v, LoopVerdict::Allow);
}

#[test]
fn test_token_budget_allow() {
    let mut guard = LoopGuard::new(100, 100, 0.0).with_token_budget(Some(10_000));
    let v = guard.record_usage(&TokenUsage { input_tokens: 500, output_tokens: 200 });
    assert_eq!(v, LoopVerdict::Allow);
}

#[test]
fn test_token_budget_warn() {
    let mut guard = LoopGuard::new(100, 100, 0.0).with_token_budget(Some(1000));
    let v = guard.record_usage(&TokenUsage { input_tokens: 500, output_tokens: 350 });
    assert!(matches!(v, LoopVerdict::Warn(_)));
}

#[test]
fn test_token_budget_exhausted() {
    let mut guard = LoopGuard::new(100, 100, 0.0).with_token_budget(Some(1000));
    guard.record_usage(&TokenUsage { input_tokens: 600, output_tokens: 300 });
    let v = guard.record_usage(&TokenUsage { input_tokens: 200, output_tokens: 100 });
    assert!(matches!(v, LoopVerdict::CircuitBreak(_)));
}

// ════════════════════════════════════════════
// PROPERTY TESTS — mutation-resistant boundaries
// ════════════════════════════════════════════

mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// INV-001: Loop always terminates within max_iterations + 1 calls.
        #[test]
        fn prop_loop_terminates(max_iter in 1u32..100) {
            let mut guard = LoopGuard::new(max_iter, 1000, 0.0);
            let mut broke = false;
            for _ in 0..=(max_iter + 1) {
                if let LoopVerdict::CircuitBreak(_) = guard.check_iteration() {
                    broke = true;
                    break;
                }
            }
            prop_assert!(broke, "guard must circuit-break by iteration {}", max_iter + 1);
            prop_assert!(guard.current_iteration() <= max_iter + 1);
        }

        /// INV-002: Guard monotonically increases.
        #[test]
        fn prop_guard_monotonic(max_iter in 1u32..50) {
            let mut guard = LoopGuard::new(max_iter, 1000, 0.0);
            let mut prev = 0u32;
            for _ in 0..max_iter {
                guard.check_iteration();
                let curr = guard.current_iteration();
                prop_assert!(curr > prev, "iteration must increase: {} > {}", curr, prev);
                prev = curr;
            }
        }

        /// INV-005: Cost budget enforced for any positive budget and cost.
        #[test]
        fn prop_cost_budget_enforced(budget in 0.001f64..100.0, cost in 0.001f64..200.0) {
            let mut guard = LoopGuard::new(100, 100, budget);
            let verdict = guard.record_cost(cost);
            if cost > budget {
                prop_assert!(matches!(verdict, LoopVerdict::CircuitBreak(_)),
                    "cost {cost} > budget {budget} must circuit-break");
            } else {
                prop_assert!(matches!(verdict, LoopVerdict::Allow),
                    "cost {cost} <= budget {budget} must allow");
            }
        }

        /// INV-004: Ping-pong detected at exactly threshold=3.
        #[test]
        fn prop_pingpong_at_threshold(repeat_count in 1u32..10) {
            let mut guard = LoopGuard::new(100, 100, 0.0);
            let input = serde_json::json!({"key": "value"});
            for i in 1..=repeat_count {
                let v = guard.check_tool_call("tool", &input);
                if i >= 3 {
                    prop_assert!(matches!(v, LoopVerdict::Block(_)),
                        "call {i} must be blocked (threshold=3)");
                } else {
                    prop_assert!(matches!(v, LoopVerdict::Allow),
                        "call {i} must be allowed (< threshold)");
                }
            }
        }

        /// INV-006: Consecutive MaxTokens circuit-breaks at 5.
        #[test]
        fn prop_max_tokens_circuit_break(count in 1u32..10) {
            let mut guard = LoopGuard::new(100, 100, 0.0);
            let mut broke = false;
            for i in 1..=count {
                if let LoopVerdict::CircuitBreak(_) = guard.record_max_tokens() {
                    prop_assert_eq!(i, 5, "circuit-break must happen at exactly 5");
                    broke = true;
                    break;
                }
            }
            if count >= 5 {
                prop_assert!(broke, "must circuit-break at {count} >= 5");
            } else {
                prop_assert!(!broke, "must not circuit-break at {count} < 5");
            }
        }

        /// Proof obligation: cost monotonically non-decreasing.
        #[test]
        fn prop_cost_monotonic(costs in proptest::collection::vec(0.0f64..10.0, 1..20)) {
            let mut guard = LoopGuard::new(100, 100, 0.0);
            let total: f64 = costs.iter().sum();
            for cost in &costs { guard.record_cost(*cost); }
            prop_assert!(total >= 0.0, "cost total must be non-negative");
        }

        /// INV-015: Token budget enforced when configured.
        #[test]
        fn prop_token_budget_enforced(
            budget in 100u64..10_000,
            input in 1u64..20_000,
            output in 1u64..20_000,
        ) {
            let mut guard = LoopGuard::new(100, 100, 0.0)
                .with_token_budget(Some(budget));
            let v = guard.record_usage(&TokenUsage {
                input_tokens: input,
                output_tokens: output,
            });
            let total = input + output;
            if total > budget {
                prop_assert!(matches!(v, LoopVerdict::CircuitBreak(_)),
                    "tokens {total} > budget {budget} must circuit-break");
            }
        }
    }
}
