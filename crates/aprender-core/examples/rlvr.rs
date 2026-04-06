//! Reinforcement Learning on Verifiable Rewards (RLVR) (GH-450)
//!
//! Demonstrates RLVR: policy gradient training with binary-verifiable
//! reward functions (math correctness, code pattern matching).
//!
//! Run: cargo run --example rlvr

use aprender::online::rlvr::{
    CodeReward, MathReward, RlvrConfig, RlvrLoss, RlvrMetrics, VerifiableReward,
};

fn main() {
    println!("=== RLVR: Reinforcement Learning on Verifiable Rewards (GH-450) ===\n");

    // ── 1. Math Reward Verification ──
    println!("── 1. Math Reward ──");
    let math = MathReward;
    let cases = [
        (
            "What is 2+2? expected: 4",
            "Let me think... the answer is 4.",
        ),
        ("Solve x=3+5. expected: 8", "x = 3 + 5 = 8"),
        ("What is 10/3? expected: 3", "\\boxed{3}"),
        ("What is 2+2? expected: 4", "I think it's 5."),
    ];
    for (prompt, response) in &cases {
        let r = math.verify(prompt, response);
        println!(
            "  {} -> score={:.1}, correct={}, {}",
            &response[..response.len().min(30)],
            r.score,
            r.correct,
            r.explanation.as_deref().unwrap_or("")
        );
    }

    // ── 2. Code Reward Verification ──
    println!("\n── 2. Code Reward ──");
    let code = CodeReward;
    let code_cases = [
        (
            "Write fibonacci. must contain: fn, fibonacci, return",
            "fn fibonacci(n: u64) -> u64 { if n <= 1 { return n; } fibonacci(n-1) + fibonacci(n-2) }",
        ),
        (
            "Print hello. must contain: println, hello",
            "fn main() { println!(\"hello\"); }",
        ),
        (
            "Sort array. must contain: sort, vec, fn",
            "let x = 42;", // missing requirements
        ),
    ];
    for (prompt, response) in &code_cases {
        let r = code.verify(prompt, response);
        println!(
            "  score={:.2}, correct={} | {}",
            r.score,
            r.correct,
            r.explanation.as_deref().unwrap_or("")
        );
    }

    // ── 3. Policy Gradient Loss ──
    println!("\n── 3. Policy Gradient ──");
    let cfg = RlvrConfig {
        kl_coeff: 0.1,
        reward_scale: 1.0,
        ..Default::default()
    };
    let loss = RlvrLoss::new(cfg);

    let log_probs = vec![-1.0, -2.0, -1.5, -3.0];
    let rewards = vec![1.0, 0.0, 1.0, 0.0];
    let pg = loss.compute_policy_gradient(&log_probs, &rewards);
    println!("  Log probs: {:?}", log_probs);
    println!("  Rewards:   {:?}", rewards);
    println!("  Policy gradient loss: {:.4}", pg);

    // ── 4. KL Penalty ──
    println!("\n── 4. KL Divergence Penalty ──");
    let policy_lp = vec![-1.0, -1.5, -2.0];
    let ref_lp = vec![-1.2, -1.8, -2.5];
    let kl = loss.compute_kl_penalty(&policy_lp, &ref_lp);
    println!("  Policy logprobs: {:?}", policy_lp);
    println!("  Ref logprobs:    {:?}", ref_lp);
    println!("  KL penalty:      {:.4}", kl);

    // ── 5. Total Loss ──
    println!("\n── 5. Total Loss ──");
    let total = loss.compute_total_loss(&log_probs, &rewards, &log_probs);
    println!("  Total = pg + kl_coeff * kl = {:.4}", total);

    // ── 6. Batch Metrics ──
    println!("\n── 6. Batch Metrics ──");
    let metrics = RlvrMetrics::from_batch(&loss, &log_probs, &rewards, &log_probs);
    println!("  Samples:    {}", metrics.num_samples);
    println!("  Avg reward: {:.2}", metrics.avg_reward);
    println!("  Accuracy:   {:.1}%", metrics.accuracy * 100.0);
    println!("  Avg KL:     {:.4}", metrics.avg_kl);
    println!("  Avg loss:   {:.4}", metrics.avg_loss);

    println!("\n=== RLVR verified ===");
}
