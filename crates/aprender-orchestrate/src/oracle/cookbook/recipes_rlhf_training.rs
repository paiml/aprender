//! RLHF training infrastructure recipes: PPO, Stability, Evaluation
//!
//! Extracted from `register_rlhf_recipes` for TDG compliance (Refs #22).

use super::Recipe;

pub fn register_rlhf_training_recipes(cookbook: &mut super::Cookbook) {
    // PPO for RLHF
    cookbook.add(
        Recipe::new("rlhf-ppo", "PPO for RLHF")
            .with_problem("Train language models with PPO using a reward model")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "ppo", "reinforcement-learning", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::ppo::*;

// Load models
let policy = Model::load("sft-model.apr")?;
let reference = Model::load("sft-model.apr")?;  // KL anchor
let reward_model = RewardModel::load("reward-model.apr")?;

// Configure PPO
let config = PpoConfig {
    // Learning
    learning_rate: 1e-6,
    batch_size: 64,
    mini_batch_size: 8,
    gradient_accumulation: 8,

    // PPO hyperparameters
    ppo_epochs: 4,
    clip_range: 0.2,
    clip_range_value: 0.2,
    gamma: 1.0,
    lam: 0.95,                    // GAE lambda

    // KL control
    kl_penalty: KlPenalty::Adaptive {
        target: 6.0,
        horizon: 10000,
    },

    // Generation
    max_new_tokens: 128,
    temperature: 0.7,
    top_p: 0.9,

    ..Default::default()
};

// Create PPO trainer
let trainer = PpoTrainer::new(policy, reference, reward_model, config);

// Training loop with prompts
let prompts = PromptDataset::load("prompts.txt")?;
for epoch in 0..10 {
    let stats = trainer.step(&prompts)?;
    println!("Epoch {}: reward={:.3}, kl={:.3}, loss={:.4}",
        epoch, stats.mean_reward, stats.kl_div, stats.loss);
}

trainer.policy().save("rlhf-model.apr")?;
"#,
            )
            .with_related(vec!["rlhf-reward-model", "rlhf-sft", "rlhf-stability"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_clip_range_valid() {
    let clip_range = 0.2_f64;
    assert!(clip_range > 0.0 && clip_range <= 1.0);
}

    #[test]
    fn test_ppo_epochs_positive() {
    let ppo_epochs = 4;
    assert!(ppo_epochs > 0);
}

    #[test]
    fn test_temperature_positive() {
    let temperature = 0.7_f64;
    assert!(temperature > 0.0);
}
}",
            ),
    );

    // RLHF Stability & Best Practices
    cookbook.add(
        Recipe::new("rlhf-stability", "RLHF Stability & Best Practices")
            .with_problem("Ensure stable RLHF training and avoid common pitfalls")
            .with_components(vec!["entrenar", "aprender", "trueno-viz"])
            .with_tags(vec!["rlhf", "stability", "best-practices", "debugging", "monitoring"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::ppo::*;

// === Stability Techniques ===

// 1. Reward normalization (running statistics)
let config = PpoConfig {
    reward_normalization: true,
    reward_clip: 10.0,           // Clip extreme rewards
    ..Default::default()
};

// 2. Advantage normalization
let config = PpoConfig {
    advantage_normalization: true,
    ..config
};

// 3. Value function clipping
let config = PpoConfig {
    clip_range_value: 0.2,       // Clip value function updates
    ..config
};

// 4. Adaptive KL penalty (InstructGPT style)
let config = PpoConfig {
    kl_penalty: KlPenalty::Adaptive {
        target: 6.0,             // Target KL divergence
        horizon: 10000,          // Adaptation horizon
    },
    ..config
};

// 5. Gradient clipping
let config = PpoConfig {
    max_grad_norm: 1.0,
    ..config
};

// === Monitoring ===
let callback = |stats: &PpoStats| {
    // Check for reward hacking
    if stats.mean_reward > 10.0 {
        println!("Warning: Possible reward hacking!");
    }

    // Check KL divergence
    if stats.kl_div > 15.0 {
        println!("Warning: High KL divergence - policy drifting!");
    }

    // Check for mode collapse
    if stats.response_entropy < 0.5 {
        println!("Warning: Low entropy - possible mode collapse!");
    }
};

// === Evaluation ===
// Always evaluate on held-out prompts
let eval_results = trainer.evaluate(&eval_prompts)?;
println!("Win rate vs SFT: {:.2}%", eval_results.win_rate * 100.0);
println!("Mean length: {:.1}", eval_results.mean_length);
println!("Diversity: {:.3}", eval_results.diversity);
"#,
            )
            .with_related(vec!["rlhf-ppo", "rlhf-evaluation"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_reward_clip_positive() {
    let reward_clip = 10.0_f64;
    assert!(reward_clip > 0.0);
}

    #[test]
    fn test_grad_norm_positive() {
    let max_grad_norm = 1.0_f64;
    assert!(max_grad_norm > 0.0);
}

    #[test]
    fn test_kl_target_positive() {
    let kl_target = 6.0_f64;
    assert!(kl_target > 0.0);
}
}",
            ),
    );

    // RLHF Evaluation
    cookbook.add(
        Recipe::new("rlhf-evaluation", "RLHF Model Evaluation")
            .with_problem("Comprehensively evaluate aligned models")
            .with_components(vec!["entrenar", "aprender", "trueno-viz"])
            .with_tags(vec!["rlhf", "evaluation", "benchmarks", "metrics", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::eval::*;

// Load models to compare
let sft_model = Model::load("sft-model.apr")?;
let rlhf_model = Model::load("rlhf-model.apr")?;

// === Pairwise Evaluation ===
let evaluator = PairwiseEvaluator::new(reward_model);
let results = evaluator.compare(
    &sft_model,
    &rlhf_model,
    &eval_prompts,
)?;
println!("RLHF win rate: {:.2}%", results.model_b_wins * 100.0);
println!("Tie rate: {:.2}%", results.ties * 100.0);

// === Safety Evaluation ===
let safety_eval = SafetyEvaluator::new()
    .add_detector(ToxicityDetector::new())
    .add_detector(BiasDetector::new())
    .add_detector(HarmfulContentDetector::new());

let safety_results = safety_eval.evaluate(&rlhf_model, &safety_prompts)?;
println!("Toxicity rate: {:.3}%", safety_results.toxicity_rate * 100.0);
println!("Refusal rate: {:.2}%", safety_results.refusal_rate * 100.0);

// === Helpfulness Benchmarks ===
let benchmarks = vec![
    ("MT-Bench", MtBench::new()),
    ("AlpacaEval", AlpacaEval::new()),
    ("HumanEval", HumanEval::new()),
];

for (name, bench) in benchmarks {
    let score = bench.evaluate(&rlhf_model)?;
    println!("{}: {:.2}", name, score);
}

// === Diversity Metrics ===
let diversity = DiversityMetrics::compute(&rlhf_model, &prompts)?;
println!("Distinct-1: {:.3}", diversity.distinct_1);
println!("Distinct-2: {:.3}", diversity.distinct_2);
println!("Self-BLEU: {:.3}", diversity.self_bleu);
"#,
            )
            .with_related(vec!["rlhf-stability", "rlhf-ppo", "rlhf-dpo"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_win_rate_in_range() {
    let win_rate = 0.65_f64;
    assert!(win_rate >= 0.0 && win_rate <= 1.0);
}

    #[test]
    fn test_distinct_n_in_range() {
    let distinct_2 = 0.82_f64;
    assert!(distinct_2 >= 0.0 && distinct_2 <= 1.0);
}

    #[test]
    fn test_toxicity_rate_in_range() {
    let toxicity = 0.03_f64;
    assert!(toxicity >= 0.0 && toxicity <= 1.0);
}
}",
            ),
    );
}
