//! Agent Loop Guard Demo
//!
//! Demonstrates the LoopGuard (Jidoka pattern):
//! - Iteration limit enforcement
//! - Ping-pong detection (FxHash duplicate tool call tracking)
//! - Cost budget enforcement (Muda: prevent runaway spend)
//! - Token budget enforcement
//! - MaxTokens truncation tracking
//!
//! Run with: `cargo run --example agent_guard --features agents`

#[cfg(feature = "agents")]
fn main() {
    use batuta::agent::guard::{LoopGuard, LoopVerdict};
    use batuta::agent::result::TokenUsage;

    println!("Agent Loop Guard Demo (Jidoka)");
    println!("==============================");
    println!();

    // --- Iteration limit ---
    println!("--- Iteration Limit (INV-001) ---");
    let mut guard = LoopGuard::new(5, 100, 0.0);
    for i in 1..=6 {
        let verdict = guard.check_iteration();
        let label = match &verdict {
            LoopVerdict::Allow => "Allow",
            LoopVerdict::Warn(_) => "Warn",
            LoopVerdict::Block(_) => "Block",
            LoopVerdict::CircuitBreak(_) => "CircuitBreak",
        };
        println!("  Iteration {i}: {label}");
        if let LoopVerdict::CircuitBreak(msg) = &verdict {
            println!("    -> {msg}");
        }
    }
    println!();

    // --- Ping-pong detection (INV-004) ---
    println!("--- Ping-Pong Detection (INV-004) ---");
    let mut guard = LoopGuard::new(100, 100, 0.0);
    let input = serde_json::json!({"query": "same question"});
    for i in 1..=4 {
        let verdict = guard.check_tool_call("search", &input);
        let label = match &verdict {
            LoopVerdict::Allow => "Allow",
            LoopVerdict::Block(msg) => {
                println!("  Call {i}: Block -> {msg}");
                "Block"
            }
            _ => "Other",
        };
        if label != "Block" {
            println!("  Call {i}: {label}");
        }
    }
    // Different input resets detection
    let different = serde_json::json!({"query": "new question"});
    let v = guard.check_tool_call("search", &different);
    println!(
        "  Different input: {}",
        if matches!(v, LoopVerdict::Allow) { "Allow" } else { "Not Allow" }
    );
    println!();

    // --- Cost budget (INV-005, Muda) ---
    println!("--- Cost Budget Enforcement (INV-005) ---");
    let mut guard = LoopGuard::new(100, 100, 0.10); // $0.10 budget
    for cents in [3, 3, 3, 5] {
        let cost = f64::from(cents) / 100.0;
        let verdict = guard.record_cost(cost);
        let label = match &verdict {
            LoopVerdict::Allow => "Allow".to_string(),
            LoopVerdict::CircuitBreak(msg) => {
                format!("CircuitBreak -> {msg}")
            }
            _ => format!("{verdict:?}"),
        };
        println!("  +${cost:.2}: {label}");
    }
    println!();

    // --- Token budget (INV-016) ---
    println!("--- Token Budget Enforcement (INV-016) ---");
    let mut guard = LoopGuard::new(100, 100, 0.0).with_token_budget(Some(1000));
    let usages = [
        TokenUsage { input_tokens: 200, output_tokens: 100 },
        TokenUsage { input_tokens: 300, output_tokens: 200 },
        TokenUsage { input_tokens: 100, output_tokens: 200 },
    ];
    let mut cumulative = 0u64;
    for usage in &usages {
        cumulative += usage.input_tokens + usage.output_tokens;
        let verdict = guard.record_usage(usage);
        let label = match &verdict {
            LoopVerdict::Allow => "Allow".to_string(),
            LoopVerdict::Warn(msg) => format!("Warn -> {msg}"),
            LoopVerdict::CircuitBreak(msg) => {
                format!("CircuitBreak -> {msg}")
            }
            _ => format!("{verdict:?}"),
        };
        println!(
            "  +{}in/{}out (total={cumulative}): {label}",
            usage.input_tokens, usage.output_tokens,
        );
    }
    println!();

    // --- MaxTokens truncation (INV-006) ---
    println!("--- MaxTokens Circuit Break (INV-006) ---");
    let mut guard = LoopGuard::new(100, 100, 0.0);
    for i in 1..=6 {
        let verdict = guard.record_max_tokens();
        let label = match &verdict {
            LoopVerdict::Allow => "Allow",
            LoopVerdict::CircuitBreak(_) => "CircuitBreak",
            _ => "Other",
        };
        println!("  Consecutive MaxTokens {i}: {label}");
    }
    println!("  Reset (ToolUse received)...");
    guard.reset_max_tokens();
    let after_reset = guard.record_max_tokens();
    println!(
        "  After reset, MaxTokens 1: {}",
        if matches!(after_reset, LoopVerdict::Allow) { "Allow" } else { "Not Allow" }
    );
    println!();

    println!("All guard demos completed.");
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "Enable `agents` feature: \
         cargo run --example agent_guard --features agents"
    );
    std::process::exit(1);
}
