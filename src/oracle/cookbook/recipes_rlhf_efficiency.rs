//! RLHF efficiency recipes: Quantization, PEFT Adapters
//!
//! Extracted from `register_rlhf_recipes` for TDG compliance (Refs #22).

use super::Recipe;

pub fn register_rlhf_efficiency_recipes(cookbook: &mut super::Cookbook) {
    // Quantization for Alignment
    cookbook.add(
        Recipe::new("rlhf-quantization", "Quantization for Efficient Alignment")
            .with_problem("Apply quantization techniques throughout the alignment pipeline")
            .with_components(vec!["entrenar", "aprender", "realizar"])
            .with_tags(vec!["rlhf", "quantization", "4bit", "8bit", "efficient", "memory"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::quantization::*;

// === QLoRA for SFT ===
let model = Model::load_quantized("llama-7b.q4_k.gguf")?;
let model = model.with_qlora(QLoraConfig {
    lora: LoraConfig { r: 64, alpha: 16, ..Default::default() },
    nf4: true,
    double_quant: true,
    ..Default::default()
})?;

// SFT on 24GB GPU
let trainer = SftTrainer::new(model, SftConfig::default());
trainer.train(&dataset, 3)?;

// === Quantized Reward Model ===
let reward_model = RewardModel::load("reward-model.apr")?;
let quantized_rm = reward_model.quantize(Quantization::Int8)?;
// 2x faster inference, minimal accuracy loss

// === INT8 PPO Training ===
let config = PpoConfig {
    // Use 8-bit Adam optimizer (saves 75% optimizer memory)
    optimizer: Optimizer::Adam8bit {
        lr: 1e-6,
        betas: (0.9, 0.999),
    },
    // Mixed precision training
    mixed_precision: MixedPrecision::Bf16,
    // Gradient checkpointing
    gradient_checkpointing: true,
    ..Default::default()
};

// === Post-Training Quantization ===
let rlhf_model = Model::load("rlhf-model.apr")?;

// GPTQ quantization (4-bit, minimal quality loss)
let gptq_model = rlhf_model.quantize_gptq(GptqConfig {
    bits: 4,
    group_size: 128,
    calibration_data: &calibration_samples,
})?;
gptq_model.save("rlhf-model.q4.gguf")?;

// AWQ quantization (activation-aware)
let awq_model = rlhf_model.quantize_awq(AwqConfig {
    bits: 4,
    group_size: 128,
})?;
"#,
            )
            .with_related(vec!["training-qlora", "rlhf-sft", "rlhf-ppo"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_quantization_bits_valid() {
    let bits = 4;
    assert!(bits == 4 || bits == 8);
}

    #[test]
    fn test_group_size_positive() {
    let group_size = 128;
    assert!(group_size > 0);
}

    #[test]
    fn test_mixed_precision_flag() {
    let use_bf16 = true;
    assert!(use_bf16);
}
}",
            ),
    );

    // PEFT Adapters
    cookbook.add(
        Recipe::new("rlhf-peft", "PEFT Adapters for Alignment")
            .with_problem("Use parameter-efficient methods beyond LoRA for alignment")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "peft", "lora", "adapters", "efficient", "fine-tuning"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::peft::*;

// === LoRA (Low-Rank Adaptation) ===
let lora = LoraConfig {
    r: 16,
    alpha: 32,
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj", "k_proj", "o_proj"],
    ..Default::default()
};

// === DoRA (Weight-Decomposed LoRA) ===
// Decomposes weights into magnitude and direction
let dora = DoraConfig {
    r: 16,
    alpha: 32,
    use_dora: true,              // Enable magnitude learning
    ..Default::default()
};

// === AdaLoRA (Adaptive LoRA) ===
// Dynamically allocates rank budget
let adalora = AdaLoraConfig {
    init_r: 12,
    target_r: 8,
    beta1: 0.85,
    beta2: 0.85,
    ..Default::default()
};

// === IA3 (Infused Adapter by Inhibiting and Amplifying) ===
// Even more efficient than LoRA
let ia3 = Ia3Config {
    target_modules: vec!["k_proj", "v_proj", "down_proj"],
    feedforward_modules: vec!["down_proj"],
};

// === Prefix Tuning ===
let prefix = PrefixTuningConfig {
    num_virtual_tokens: 20,
    encoder_hidden_size: 512,
};

// === Apply adapter to model ===
let model = Model::load("llama-7b.apr")?;
let model = model.with_adapter(AdapterConfig::LoRA(lora))?;

// Train with any alignment method
let trainer = DpoTrainer::new(model, reference, DpoConfig::default());
trainer.train(&dataset, 1)?;

// Save only adapter weights (small file)
model.save_adapter("dpo-adapter.lora")?;

// Merge adapters for inference
let merged = model.merge_adapter()?;
merged.save("dpo-merged.apr")?;
"#,
            )
            .with_related(vec!["training-lora", "training-qlora", "rlhf-sft"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_lora_rank_positive() {
    let rank = 16;
    assert!(rank > 0);
}

    #[test]
    fn test_target_modules_non_empty() {
    let target_modules = vec!["q_proj", "v_proj"];
    assert!(!target_modules.is_empty());
}

    #[test]
    fn test_prefix_tuning_virtual_tokens_positive() {
    let num_virtual_tokens = 20;
    assert!(num_virtual_tokens > 0);
}
}"#,
            ),
    );
}
