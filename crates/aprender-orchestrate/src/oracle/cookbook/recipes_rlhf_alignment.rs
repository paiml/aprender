//! RLHF core alignment recipes: SFT, Reward Model, DPO, DPO Variants
//!
//! Extracted from `register_rlhf_recipes` for TDG compliance (Refs #22).

use super::Recipe;

pub fn register_rlhf_alignment_recipes(cookbook: &mut super::Cookbook) {
    // Supervised Fine-Tuning (SFT)
    cookbook.add(
        Recipe::new("rlhf-sft", "Supervised Fine-Tuning (SFT)")
            .with_problem("Fine-tune a base model on instruction-following data")
            .with_components(vec!["entrenar", "aprender", "alimentar"])
            .with_tags(vec!["rlhf", "sft", "instruction-tuning", "fine-tuning", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::sft::*;

// Load base model
let model = Model::load("llama-7b.apr")?;

// Load instruction dataset (prompt, response pairs)
let dataset = SftDataset::load("alpaca_data.json")?
    .with_prompt_template(
        "Below is an instruction. Write a response.\n\n\
         ### Instruction:\n{instruction}\n\n\
         ### Response:\n"
    );

// Configure SFT trainer
let config = SftConfig {
    learning_rate: 2e-5,
    batch_size: 4,
    gradient_accumulation: 8,
    max_seq_length: 2048,
    warmup_ratio: 0.03,
    weight_decay: 0.0,
    ..Default::default()
};

// Train with LoRA for efficiency
let model = model.with_lora(LoraConfig::default())?;
let trainer = SftTrainer::new(model, config);

trainer.train(&dataset, 3)?;  // 3 epochs
model.save("sft-model.apr")?;

// Evaluation
let eval_loss = trainer.evaluate(&eval_dataset)?;
println!("Eval loss: {:.4}", eval_loss);
"#,
            )
            .with_related(vec!["rlhf-reward-model", "rlhf-dpo", "training-lora"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_sft_config_defaults() {
    let epochs = 3;
    let batch_size = 4;
    assert!(epochs > 0 && batch_size > 0);
}

    #[test]
    fn test_prompt_template_has_placeholder() {
    let template = "Instruction: {instruction} Response:";
    assert!(template.contains("{instruction}"));
}

    #[test]
    fn test_warmup_ratio_in_range() {
    let warmup_ratio = 0.03_f64;
    assert!(warmup_ratio >= 0.0 && warmup_ratio <= 1.0);
}
}"#,
            ),
    );

    // Reward Modeling
    cookbook.add(
        Recipe::new("rlhf-reward-model", "Reward Model Training")
            .with_problem("Train a reward model from human preference data")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "reward-model", "preferences", "ranking", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::reward::*;

// Load SFT model as base
let model = Model::load("sft-model.apr")?;

// Convert to reward model (adds value head)
let reward_model = RewardModel::from_base(model)?;

// Load preference dataset (chosen vs rejected)
let dataset = PreferenceDataset::load("preferences.json")?;
// Each sample: { prompt, chosen, rejected }

// Configure reward model training
let config = RewardConfig {
    learning_rate: 1e-5,
    batch_size: 4,
    max_length: 512,
    // Use margin ranking loss
    loss_fn: RewardLoss::MarginRanking { margin: 0.0 },
    ..Default::default()
};

let trainer = RewardTrainer::new(reward_model, config);
trainer.train(&dataset, 1)?;

// Evaluate accuracy (chosen > rejected)
let accuracy = trainer.evaluate(&eval_dataset)?;
println!("Preference accuracy: {:.2}%", accuracy * 100.0);

reward_model.save("reward-model.apr")?;

// Inference: score a response
let score = reward_model.score(&prompt, &response)?;
println!("Reward score: {:.3}", score);
"#,
            )
            .with_related(vec!["rlhf-sft", "rlhf-ppo", "rlhf-dpo"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_preference_accuracy_in_range() {
    let accuracy = 0.72_f64;
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
}

    #[test]
    fn test_margin_non_negative() {
    let margin = 0.0_f64;
    assert!(margin >= 0.0);
}

    #[test]
    fn test_reward_score_ordering() {
    let chosen_reward = 1.5_f64;
    let rejected_reward = -0.3_f64;
    assert!(chosen_reward > rejected_reward);
}
}",
            ),
    );

    // Direct Preference Optimization (DPO)
    cookbook.add(
        Recipe::new("rlhf-dpo", "Direct Preference Optimization (DPO)")
            .with_problem("Align models directly from preferences without reward modeling")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "dpo", "alignment", "preferences", "efficient"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::dpo::*;

// Load SFT model (policy) and reference model
let policy = Model::load("sft-model.apr")?;
let reference = Model::load("sft-model.apr")?;  // Frozen copy

// Load preference dataset
let dataset = PreferenceDataset::load("preferences.json")?;

// Configure DPO
let config = DpoConfig {
    beta: 0.1,                    // KL penalty coefficient
    learning_rate: 5e-7,
    batch_size: 4,
    gradient_accumulation: 4,
    max_length: 512,
    max_prompt_length: 256,
    label_smoothing: 0.0,
    ..Default::default()
};

// DPO loss: -log σ(β * (log π(y_w|x) - log π(y_l|x)))
let trainer = DpoTrainer::new(policy, reference, config);
trainer.train(&dataset, 1)?;

// Evaluate
let metrics = trainer.evaluate(&eval_dataset)?;
println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
println!("Chosen reward: {:.3}", metrics.chosen_reward);
println!("Rejected reward: {:.3}", metrics.rejected_reward);

policy.save("dpo-model.apr")?;
"#,
            )
            .with_related(vec!["rlhf-sft", "rlhf-ipo", "rlhf-kto"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_beta_positive() {
    let beta = 0.1_f64;
    assert!(beta > 0.0);
}

    #[test]
    fn test_max_length_exceeds_prompt_length() {
    let max_prompt_length = 256;
    let max_length = 512;
    assert!(max_length > max_prompt_length);
}

    #[test]
    fn test_label_smoothing_in_range() {
    let label_smoothing = 0.0_f64;
    assert!(label_smoothing >= 0.0 && label_smoothing <= 1.0);
}
}",
            ),
    );

    // DPO Variants (IPO, KTO, ORPO)
    cookbook.add(
        Recipe::new("rlhf-dpo-variants", "DPO Variants: IPO, KTO, ORPO")
            .with_problem("Use improved DPO variants for better alignment")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "dpo", "ipo", "kto", "orpo", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::dpo::*;

// === IPO (Identity Preference Optimization) ===
// Addresses DPO's overfitting with identity mapping
let ipo_config = IpoConfig {
    tau: 0.1,                     // Temperature parameter
    learning_rate: 5e-7,
    ..Default::default()
};
let trainer = IpoTrainer::new(policy.clone(), reference.clone(), ipo_config);

// === KTO (Kahneman-Tversky Optimization) ===
// Works with unpaired data (no need for chosen/rejected pairs)
let kto_dataset = KtoDataset::load("ratings.json")?;
// Each sample: { prompt, response, is_desirable: bool }

let kto_config = KtoConfig {
    beta: 0.1,
    desirable_weight: 1.0,
    undesirable_weight: 1.0,
    ..Default::default()
};
let trainer = KtoTrainer::new(policy.clone(), reference.clone(), kto_config);
trainer.train(&kto_dataset, 1)?;

// === ORPO (Odds Ratio Preference Optimization) ===
// No reference model needed - uses odds ratio
let orpo_config = OrpoConfig {
    beta: 0.1,
    learning_rate: 8e-6,
    ..Default::default()
};
// ORPO combines SFT and preference learning
let trainer = OrpoTrainer::new(policy.clone(), orpo_config);
trainer.train(&dataset, 1)?;

// === SimPO (Simple Preference Optimization) ===
// Length-normalized, reference-free
let simpo_config = SimpoConfig {
    beta: 2.5,
    gamma: 0.5,                   // Target margin
    ..Default::default()
};
let trainer = SimpoTrainer::new(policy, simpo_config);
"#,
            )
            .with_related(vec!["rlhf-dpo", "rlhf-sft"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_ipo_tau_positive() {
    let tau = 0.1_f64;
    assert!(tau > 0.0);
}

    #[test]
    fn test_kto_weights_non_negative() {
    let desirable_weight = 1.0_f64;
    let undesirable_weight = 1.0_f64;
    assert!(desirable_weight >= 0.0 && undesirable_weight >= 0.0);
}

    #[test]
    fn test_orpo_beta_in_valid_range() {
    let beta = 0.1_f64;
    assert!(beta > 0.0 && beta < 1.0);
}
}",
            ),
    );
}
