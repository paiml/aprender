//! Tests for model paradigm classification.

use super::*;

#[test]
fn test_fine_tune_method_lora() {
    let lora = FineTuneMethod::lora(64);
    match lora {
        FineTuneMethod::LoRA { rank, alpha } => {
            assert_eq!(rank, 64);
            assert!((alpha - 64.0).abs() < f32::EPSILON);
        }
        _ => panic!("Expected LoRA"),
    }
}

#[test]
fn test_fine_tune_method_qlora() {
    let qlora = FineTuneMethod::qlora(32);
    match qlora {
        FineTuneMethod::QLoRA { rank, bits } => {
            assert_eq!(rank, 32);
            assert_eq!(bits, 4);
        }
        _ => panic!("Expected QLoRA"),
    }
}

#[test]
fn test_fine_tune_method_memory_reduction() {
    let full = FineTuneMethod::Full;
    let lora = FineTuneMethod::lora(64);
    let qlora = FineTuneMethod::qlora(32);

    assert!((full.memory_reduction_factor() - 1.0).abs() < 0.01);
    assert!(lora.memory_reduction_factor() > 1.0);
    assert!(qlora.memory_reduction_factor() > lora.memory_reduction_factor());
}

#[test]
fn test_fine_tune_method_trainable_params() {
    let full = FineTuneMethod::Full;
    let lora = FineTuneMethod::lora(8);
    let ia3 = FineTuneMethod::IA3;

    assert!((full.trainable_params_percent() - 100.0).abs() < 0.01);
    assert!(lora.trainable_params_percent() < 1.0);
    assert!(ia3.trainable_params_percent() < lora.trainable_params_percent());
}

#[test]
fn test_fine_tune_method_display() {
    assert_eq!(format!("{}", FineTuneMethod::lora(64)), "LoRA(r=64, α=64)");
    assert_eq!(format!("{}", FineTuneMethod::qlora(32)), "QLoRA(r=32, 4-bit)");
    assert_eq!(format!("{}", FineTuneMethod::IA3), "IA³");
}

#[test]
fn test_model_paradigm_lora() {
    let paradigm = ModelParadigm::lora(64, 16.0);
    match paradigm {
        ModelParadigm::FineTuning(FineTuneMethod::LoRA { rank, alpha }) => {
            assert_eq!(rank, 64);
            assert!((alpha - 16.0).abs() < f32::EPSILON);
        }
        _ => panic!("Expected LoRA fine-tuning"),
    }
}

#[test]
fn test_model_paradigm_memory_multiplier() {
    let traditional = ModelParadigm::TraditionalMl;
    let deep = ModelParadigm::DeepLearning;
    let lora = ModelParadigm::lora(64, 64.0);
    let distill = ModelParadigm::Distillation;

    assert!(traditional.typical_memory_multiplier() < deep.typical_memory_multiplier());
    assert!(lora.typical_memory_multiplier() < deep.typical_memory_multiplier());
    assert!(distill.typical_memory_multiplier() > deep.typical_memory_multiplier());
}

#[test]
fn test_model_paradigm_training_speedup() {
    let deep = ModelParadigm::DeepLearning;
    let lora = ModelParadigm::lora(64, 64.0);
    let traditional = ModelParadigm::TraditionalMl;

    assert!((deep.typical_training_speedup() - 1.0).abs() < 0.01);
    assert!(lora.typical_training_speedup() > deep.typical_training_speedup());
    assert!(traditional.typical_training_speedup() > lora.typical_training_speedup());
}

#[test]
fn test_model_paradigm_quality_retention() {
    let deep = ModelParadigm::DeepLearning;
    let lora = ModelParadigm::lora(64, 64.0);
    let distill = ModelParadigm::Distillation;
    let ensemble = ModelParadigm::Ensemble;

    assert!((deep.typical_quality_retention() - 1.0).abs() < 0.01);
    assert!(lora.typical_quality_retention() > 0.9);
    assert!(distill.typical_quality_retention() < 1.0);
    assert!(ensemble.typical_quality_retention() > 1.0);
}

#[test]
fn test_model_paradigm_requires_pretrained() {
    assert!(!ModelParadigm::TraditionalMl.requires_pretrained());
    assert!(!ModelParadigm::DeepLearning.requires_pretrained());
    assert!(ModelParadigm::lora(64, 64.0).requires_pretrained());
    assert!(ModelParadigm::Distillation.requires_pretrained());
}

#[test]
fn test_model_paradigm_is_parameter_efficient() {
    assert!(!ModelParadigm::DeepLearning.is_parameter_efficient());
    assert!(!ModelParadigm::FineTuning(FineTuneMethod::Full).is_parameter_efficient());
    assert!(ModelParadigm::lora(64, 64.0).is_parameter_efficient());
    assert!(ModelParadigm::qlora(32, 4).is_parameter_efficient());
}

#[test]
fn test_model_paradigm_batch_size_multiplier() {
    let deep = ModelParadigm::DeepLearning;
    let qlora = ModelParadigm::qlora(32, 4);
    let distill = ModelParadigm::Distillation;

    assert!((deep.batch_size_multiplier() - 1.0).abs() < 0.01);
    assert!(qlora.batch_size_multiplier() > deep.batch_size_multiplier());
    assert!(distill.batch_size_multiplier() < deep.batch_size_multiplier());
}

#[test]
fn test_model_paradigm_default() {
    let default = ModelParadigm::default();
    assert!(matches!(default, ModelParadigm::DeepLearning));
}

#[test]
fn test_model_paradigm_display() {
    assert_eq!(format!("{}", ModelParadigm::TraditionalMl), "Traditional ML");
    assert_eq!(format!("{}", ModelParadigm::DeepLearning), "Deep Learning");
    assert_eq!(format!("{}", ModelParadigm::MoE), "Mixture of Experts");
    assert!(format!("{}", ModelParadigm::lora(64, 64.0)).contains("LoRA"));
}

#[test]
fn test_model_paradigm_serialization() {
    let paradigm = ModelParadigm::lora(32, 16.0);
    let json = serde_json::to_string(&paradigm).expect("JSON serialization should succeed");
    let parsed: ModelParadigm =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert!(parsed.is_parameter_efficient());
}

#[test]
fn test_fine_tune_method_display_all_variants() {
    assert_eq!(format!("{}", FineTuneMethod::Adapter), "Adapter");
    assert_eq!(format!("{}", FineTuneMethod::Prefix), "Prefix");
    assert_eq!(format!("{}", FineTuneMethod::Full), "Full");
}

#[test]
fn test_fine_tune_method_memory_reduction_all_variants() {
    // Test all non-parameterized variants
    assert!((FineTuneMethod::Adapter.memory_reduction_factor() - 10.0).abs() < 0.01);
    assert!((FineTuneMethod::Prefix.memory_reduction_factor() - 20.0).abs() < 0.01);
    assert!((FineTuneMethod::IA3.memory_reduction_factor() - 50.0).abs() < 0.01);
    assert!((FineTuneMethod::Full.memory_reduction_factor() - 1.0).abs() < 0.01);
}

#[test]
fn test_fine_tune_method_trainable_params_all_variants() {
    assert!((FineTuneMethod::Adapter.trainable_params_percent() - 5.0).abs() < 0.01);
    assert!((FineTuneMethod::Prefix.trainable_params_percent() - 1.0).abs() < 0.01);
    assert!((FineTuneMethod::IA3.trainable_params_percent() - 0.01).abs() < 0.001);
    assert!((FineTuneMethod::Full.trainable_params_percent() - 100.0).abs() < 0.01);
}

#[test]
fn test_fine_tune_method_qlora_memory_with_different_bits() {
    let qlora_4bit = FineTuneMethod::QLoRA { rank: 32, bits: 4 };
    let qlora_8bit = FineTuneMethod::QLoRA { rank: 32, bits: 8 };
    // 4-bit has higher compression than 8-bit
    assert!(qlora_4bit.memory_reduction_factor() > qlora_8bit.memory_reduction_factor());
}

#[test]
fn test_fine_tune_method_lora_rank_zero() {
    // Edge case: rank 0 uses max(1.0) to avoid division by zero
    let lora = FineTuneMethod::LoRA { rank: 0, alpha: 0.0 };
    let reduction = lora.memory_reduction_factor();
    assert!((reduction - 100.0).abs() < 0.01); // 100 / max(0, 1) = 100
}

#[test]
fn test_fine_tune_method_high_rank_trainable_params() {
    // High rank caps at 0.2% (min(2.0) * 0.1)
    let lora_high = FineTuneMethod::LoRA { rank: 256, alpha: 256.0 };
    let lora_low = FineTuneMethod::LoRA { rank: 8, alpha: 8.0 };
    assert!((lora_high.trainable_params_percent() - 0.2).abs() < 0.01);
    assert!((lora_low.trainable_params_percent() - 0.1).abs() < 0.01);
}

#[test]
fn test_model_paradigm_typical_memory_multiplier_all_finetuning() {
    // Test all FineTuneMethod variants via ModelParadigm
    let full = ModelParadigm::FineTuning(FineTuneMethod::Full);
    let adapter = ModelParadigm::FineTuning(FineTuneMethod::Adapter);
    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    let ia3 = ModelParadigm::FineTuning(FineTuneMethod::IA3);

    assert!((full.typical_memory_multiplier() - 4.0).abs() < 0.01);
    assert!((adapter.typical_memory_multiplier() - 1.5).abs() < 0.01);
    assert!((prefix.typical_memory_multiplier() - 1.3).abs() < 0.01);
    assert!((ia3.typical_memory_multiplier() - 1.1).abs() < 0.01);
}

#[test]
fn test_model_paradigm_typical_training_speedup_all_finetuning() {
    let full = ModelParadigm::FineTuning(FineTuneMethod::Full);
    let qlora = ModelParadigm::FineTuning(FineTuneMethod::qlora(32));
    let adapter = ModelParadigm::FineTuning(FineTuneMethod::Adapter);
    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    let ia3 = ModelParadigm::FineTuning(FineTuneMethod::IA3);

    assert!((full.typical_training_speedup() - 2.0).abs() < 0.01);
    assert!((qlora.typical_training_speedup() - 6.0).abs() < 0.01);
    assert!((adapter.typical_training_speedup() - 4.0).abs() < 0.01);
    assert!((prefix.typical_training_speedup() - 5.0).abs() < 0.01);
    assert!((ia3.typical_training_speedup() - 8.0).abs() < 0.01);
}

#[test]
fn test_model_paradigm_lora_training_speedup_varies_with_rank() {
    let lora_low = ModelParadigm::lora(8, 8.0);
    let lora_high = ModelParadigm::lora(128, 128.0);
    // Higher rank = slower speedup
    assert!(lora_low.typical_training_speedup() > lora_high.typical_training_speedup());
}

#[test]
fn test_model_paradigm_typical_quality_retention_all_finetuning() {
    let full = ModelParadigm::FineTuning(FineTuneMethod::Full);
    let qlora = ModelParadigm::FineTuning(FineTuneMethod::qlora(32));
    let adapter = ModelParadigm::FineTuning(FineTuneMethod::Adapter);
    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    let ia3 = ModelParadigm::FineTuning(FineTuneMethod::IA3);

    assert!((full.typical_quality_retention() - 1.0).abs() < 0.01);
    assert!((qlora.typical_quality_retention() - 0.93).abs() < 0.01);
    assert!((adapter.typical_quality_retention() - 0.92).abs() < 0.01);
    assert!((prefix.typical_quality_retention() - 0.88).abs() < 0.01);
    assert!((ia3.typical_quality_retention() - 0.90).abs() < 0.01);
}

#[test]
fn test_model_paradigm_lora_quality_varies_with_rank() {
    let lora_low = ModelParadigm::lora(8, 8.0);
    let lora_high = ModelParadigm::lora(64, 64.0);
    // Higher rank = better quality (up to cap)
    assert!(lora_high.typical_quality_retention() > lora_low.typical_quality_retention());
}

#[test]
fn test_model_paradigm_batch_size_multiplier_all_finetuning() {
    let full = ModelParadigm::FineTuning(FineTuneMethod::Full);
    let adapter = ModelParadigm::FineTuning(FineTuneMethod::Adapter);
    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    let ia3 = ModelParadigm::FineTuning(FineTuneMethod::IA3);

    assert!((full.batch_size_multiplier() - 1.0).abs() < 0.01);
    assert!((adapter.batch_size_multiplier() - 1.5).abs() < 0.01);
    assert!((prefix.batch_size_multiplier() - 1.8).abs() < 0.01);
    assert!((ia3.batch_size_multiplier() - 3.0).abs() < 0.01);
}

#[test]
fn test_model_paradigm_moe_characteristics() {
    let moe = ModelParadigm::MoE;
    assert!((moe.typical_memory_multiplier() - 2.0).abs() < 0.01);
    assert!((moe.typical_training_speedup() - 0.8).abs() < 0.01);
    assert!(moe.typical_quality_retention() > 1.0); // 1.05
    assert!((moe.batch_size_multiplier() - 1.2).abs() < 0.01);
    assert!(!moe.requires_pretrained());
    assert!(!moe.is_parameter_efficient());
}

#[test]
fn test_model_paradigm_ensemble_characteristics() {
    let ensemble = ModelParadigm::Ensemble;
    assert!((ensemble.typical_memory_multiplier() - 3.0).abs() < 0.01);
    assert!((ensemble.typical_training_speedup() - 0.5).abs() < 0.01);
    assert!(ensemble.typical_quality_retention() > 1.0); // 1.02
    assert!((ensemble.batch_size_multiplier() - 0.3).abs() < 0.01);
    assert!(!ensemble.requires_pretrained());
    assert!(!ensemble.is_parameter_efficient());
}

#[test]
fn test_model_paradigm_distillation_characteristics() {
    let distill = ModelParadigm::Distillation;
    assert!((distill.typical_memory_multiplier() - 5.0).abs() < 0.01);
    assert!((distill.typical_training_speedup() - 1.5).abs() < 0.01);
    assert!((distill.typical_quality_retention() - 0.85).abs() < 0.01);
    assert!((distill.batch_size_multiplier() - 0.5).abs() < 0.01);
    assert!(distill.requires_pretrained());
    assert!(!distill.is_parameter_efficient());
}

#[test]
fn test_model_paradigm_traditional_ml_characteristics() {
    let trad = ModelParadigm::TraditionalMl;
    assert!((trad.typical_memory_multiplier() - 1.5).abs() < 0.01);
    assert!((trad.typical_training_speedup() - 10.0).abs() < 0.01);
    assert!((trad.typical_quality_retention() - 0.7).abs() < 0.01);
    assert!((trad.batch_size_multiplier() - 10.0).abs() < 0.01);
    assert!(!trad.requires_pretrained());
    assert!(!trad.is_parameter_efficient());
}

#[test]
fn test_model_paradigm_display_all_variants() {
    assert_eq!(format!("{}", ModelParadigm::Distillation), "Knowledge Distillation");
    assert_eq!(format!("{}", ModelParadigm::Ensemble), "Ensemble");

    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    assert!(format!("{prefix}").contains("Prefix"));
}

#[test]
fn test_model_paradigm_serialization_all_variants() {
    let variants = vec![
        ModelParadigm::TraditionalMl,
        ModelParadigm::DeepLearning,
        ModelParadigm::Distillation,
        ModelParadigm::MoE,
        ModelParadigm::Ensemble,
        ModelParadigm::FineTuning(FineTuneMethod::Adapter),
        ModelParadigm::FineTuning(FineTuneMethod::Prefix),
        ModelParadigm::FineTuning(FineTuneMethod::IA3),
        ModelParadigm::FineTuning(FineTuneMethod::Full),
    ];

    for paradigm in variants {
        let json = serde_json::to_string(&paradigm).expect("JSON serialization should succeed");
        let parsed: ModelParadigm =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(paradigm, parsed);
    }
}

#[test]
fn test_fine_tune_method_serde_roundtrip() {
    let methods = vec![
        FineTuneMethod::lora(64),
        FineTuneMethod::qlora(32),
        FineTuneMethod::Adapter,
        FineTuneMethod::Prefix,
        FineTuneMethod::IA3,
        FineTuneMethod::Full,
    ];

    for method in methods {
        let json = serde_json::to_string(&method).expect("JSON serialization should succeed");
        let parsed: FineTuneMethod =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(method, parsed);
    }
}

#[test]
fn test_model_paradigm_is_parameter_efficient_adapter_prefix_ia3() {
    let adapter = ModelParadigm::FineTuning(FineTuneMethod::Adapter);
    let prefix = ModelParadigm::FineTuning(FineTuneMethod::Prefix);
    let ia3 = ModelParadigm::FineTuning(FineTuneMethod::IA3);

    assert!(adapter.is_parameter_efficient());
    assert!(prefix.is_parameter_efficient());
    assert!(ia3.is_parameter_efficient());
}

#[test]
fn test_fine_tune_method_qlora_trainable_params() {
    let qlora = FineTuneMethod::qlora(16);
    // Same formula as LoRA
    let expected = 0.1 * (16.0 / 8.0_f64).min(2.0);
    assert!((qlora.trainable_params_percent() - expected).abs() < 0.01);
}
