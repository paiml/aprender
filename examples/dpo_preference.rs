//! Direct Preference Optimization (DPO) (GH-449)
//!
//! Demonstrates DPO loss computation, gradient calculation, implicit rewards,
//! and metrics tracking for preference-based LLM alignment.
//!
//! Run: cargo run --example dpo_preference

use aprender::online::dpo::{DpoConfig, DpoLoss, DpoMetrics, PreferencePair};

fn main() {
    println!("=== Direct Preference Optimization (GH-449) ===\n");

    // ── 1. Basic DPO Loss ──
    println!("── 1. DPO Loss Computation ──");
    let loss = DpoLoss::new(DpoConfig::default());

    let good_pair = PreferencePair {
        chosen_logprob: -1.0,
        rejected_logprob: -5.0,
        ref_chosen_logprob: -2.0,
        ref_rejected_logprob: -3.0,
    };
    let bad_pair = PreferencePair {
        chosen_logprob: -5.0,
        rejected_logprob: -1.0,
        ref_chosen_logprob: -3.0,
        ref_rejected_logprob: -2.0,
    };

    println!(
        "  Policy prefers chosen:   loss = {:.4}",
        loss.compute(&good_pair)
    );
    println!(
        "  Policy prefers rejected: loss = {:.4}",
        loss.compute(&bad_pair)
    );

    // ── 2. Gradients ──
    println!("\n── 2. DPO Gradients ──");
    let (gc, gr) = loss.gradient(&good_pair);
    println!("  grad_chosen  = {:.6} (negative = encourage)", gc);
    println!("  grad_rejected = {:.6} (positive = discourage)", gr);
    println!("  sum = {:.10} (zero-sum property)", gc + gr);

    // ── 3. Implicit Rewards ──
    println!("\n── 3. Implicit Rewards ──");
    let r_chosen = loss.implicit_reward(-1.0, -2.0);
    let r_rejected = loss.implicit_reward(-5.0, -3.0);
    println!("  r(chosen)  = {:.4}", r_chosen);
    println!("  r(rejected) = {:.4}", r_rejected);
    println!("  margin = {:.4}", r_chosen - r_rejected);

    // ── 4. Training Metrics ──
    println!("\n── 4. Batch Metrics ──");
    let pairs = vec![
        PreferencePair {
            chosen_logprob: -1.0,
            rejected_logprob: -4.0,
            ref_chosen_logprob: -2.0,
            ref_rejected_logprob: -3.0,
        },
        PreferencePair {
            chosen_logprob: -1.5,
            rejected_logprob: -3.5,
            ref_chosen_logprob: -2.0,
            ref_rejected_logprob: -2.5,
        },
        PreferencePair {
            chosen_logprob: -2.0,
            rejected_logprob: -2.5,
            ref_chosen_logprob: -2.0,
            ref_rejected_logprob: -2.0,
        },
    ];
    let metrics = DpoMetrics::from_batch(&loss, &pairs);
    println!("  Pairs:          {}", metrics.num_pairs);
    println!("  Avg loss:       {:.4}", metrics.avg_loss);
    println!("  Accuracy:       {:.1}%", metrics.accuracy * 100.0);
    println!("  Chosen reward:  {:.4}", metrics.avg_chosen_reward);
    println!("  Rejected reward: {:.4}", metrics.avg_rejected_reward);
    println!("  Reward margin:  {:.4}", metrics.reward_margin);

    // ── 5. Label Smoothing ──
    println!("\n── 5. Label Smoothing ──");
    let smooth_loss = DpoLoss::new(DpoConfig {
        label_smoothing: 0.1,
        ..Default::default()
    });
    let l_no = loss.compute(&good_pair);
    let l_yes = smooth_loss.compute(&good_pair);
    println!("  Without smoothing: {:.4}", l_no);
    println!("  With smoothing:    {:.4}", l_yes);

    println!("\n=== DPO pipeline verified ===");
}
