//! Tests for fine-tuning module

use super::*;
use crate::lora::LoRAConfig;

// =========================================================================
// FineTuneMethod Tests
// =========================================================================

#[test]
fn test_fine_tune_method_default() {
    let method = FineTuneMethod::default();
    assert!(matches!(method, FineTuneMethod::LoRA(_)));
}

#[test]
fn test_fine_tune_method_qlora() {
    let method = FineTuneMethod::QLoRA { lora_config: LoRAConfig::default(), bits: 4 };
    if let FineTuneMethod::QLoRA { bits, .. } = method {
        assert_eq!(bits, 4);
    } else {
        panic!("Expected QLoRA");
    }
}

// =========================================================================
// FineTuneConfig Tests
// =========================================================================

#[test]
fn test_fine_tune_config_default() {
    let config = FineTuneConfig::default();
    assert!(config.model_id.is_empty());
    assert_eq!(config.epochs, 3);
    assert_eq!(config.batch_size, 8);
}

#[test]
fn test_fine_tune_config_builder() {
    let config = FineTuneConfig::new("microsoft/codebert-base")
        .learning_rate(1e-4)
        .epochs(5)
        .batch_size(16)
        .output_dir("/tmp/output");

    assert_eq!(config.model_id, "microsoft/codebert-base");
    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.epochs, 5);
    assert_eq!(config.batch_size, 16);
}

#[test]
fn test_fine_tune_config_with_lora() {
    let lora = LoRAConfig::new(16, 16.0).target_attention_projections();
    let config = FineTuneConfig::new("model").with_lora(lora.clone());

    if let FineTuneMethod::LoRA(c) = &config.method {
        assert_eq!(c.rank, 16);
    } else {
        panic!("Expected LoRA method");
    }
}

#[test]
fn test_fine_tune_config_with_qlora() {
    let lora = LoRAConfig::new(8, 8.0);
    let config = FineTuneConfig::new("model").with_qlora(lora, 4);

    if let FineTuneMethod::QLoRA { bits, .. } = &config.method {
        assert_eq!(*bits, 4);
    } else {
        panic!("Expected QLoRA method");
    }
}

#[test]
fn test_fine_tune_config_full() {
    let config = FineTuneConfig::new("model").full_fine_tune();
    assert!(matches!(config.method, FineTuneMethod::Full));
}

// =========================================================================
// Validation Tests
// =========================================================================

#[test]
fn test_validate_empty_model_id() {
    let config = FineTuneConfig::default();
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_valid_config() {
    let config = FineTuneConfig::new("valid/model");
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_invalid_learning_rate() {
    let config = FineTuneConfig::new("model").learning_rate(0.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_invalid_qlora_bits() {
    let config = FineTuneConfig::new("model").with_qlora(LoRAConfig::default(), 3);
    assert!(config.validate().is_err());
}

// =========================================================================
// Memory Estimation Tests
// =========================================================================

#[test]
fn test_estimate_trainable_params_full() {
    let config = FineTuneConfig::new("model").full_fine_tune();
    assert_eq!(config.estimate_trainable_params(1_000_000), 1_000_000);
}

#[test]
fn test_estimate_trainable_params_lora() {
    let lora = LoRAConfig::new(8, 8.0).target_attention_projections();
    let config = FineTuneConfig::new("model").with_lora(lora);
    let trainable = config.estimate_trainable_params(7_000_000_000);
    // LoRA should have far fewer trainable params
    assert!(trainable < 100_000_000);
}

#[test]
fn test_estimate_memory_full_vs_lora() {
    let full_config = FineTuneConfig::new("model").full_fine_tune();
    let lora_config = FineTuneConfig::new("model").with_lora(LoRAConfig::default());

    let params = 7_000_000_000u64;
    let full_mem = full_config.estimate_memory(params);
    let lora_mem = lora_config.estimate_memory(params);

    // LoRA should use significantly less memory
    assert!(lora_mem.total() < full_mem.total());
}

#[test]
fn test_estimate_memory_qlora_vs_lora() {
    let lora_config = FineTuneConfig::new("model").with_lora(LoRAConfig::default());
    let qlora_config = FineTuneConfig::new("model").with_qlora(LoRAConfig::default(), 4);

    let params = 7_000_000_000u64;
    let lora_mem = lora_config.estimate_memory(params);
    let qlora_mem = qlora_config.estimate_memory(params);

    // QLoRA should use less memory than LoRA (quantized base)
    assert!(qlora_mem.model < lora_mem.model);
}

// =========================================================================
// MemoryRequirement Tests
// =========================================================================

#[test]
fn test_memory_requirement_total() {
    let mem = MemoryRequirement { model: 1000, optimizer: 500, gradients: 250, activations: 100 };
    assert_eq!(mem.total(), 1850);
}

#[test]
fn test_memory_requirement_fits_in() {
    let mem = MemoryRequirement { model: 1000, optimizer: 500, gradients: 250, activations: 100 };
    assert!(mem.fits_in(2000));
    assert!(!mem.fits_in(1000));
}

#[test]
fn test_memory_requirement_savings() {
    let mem = MemoryRequirement { model: 500, optimizer: 100, gradients: 50, activations: 50 };
    // Full memory for 1000 params = 1000*4 + 1000*8 + 1000*4 = 16000
    let savings = mem.savings_vs_full(1000);
    assert!(savings > 0.0);
    assert!(savings < 1.0);
}

#[test]
fn test_memory_format_human() {
    let mem = MemoryRequirement {
        model: 14_000_000_000,
        optimizer: 2_000_000_000,
        gradients: 1_000_000_000,
        activations: 500_000_000,
    };
    let formatted = mem.format_human();
    assert!(formatted.contains("14.0GB"));
    assert!(formatted.contains("Total:"));
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_gradient_checkpointing_builder() {
    let config = FineTuneConfig::new("model").gradient_checkpointing(false);
    assert!(!config.gradient_checkpointing);

    let config2 = FineTuneConfig::new("model").gradient_checkpointing(true);
    assert!(config2.gradient_checkpointing);
}

#[test]
fn test_mixed_precision_builder() {
    let config = FineTuneConfig::new("model").mixed_precision(Some(MixedPrecision::Fp16));
    assert_eq!(config.mixed_precision, Some(MixedPrecision::Fp16));

    let config2 = FineTuneConfig::new("model").mixed_precision(None);
    assert!(config2.mixed_precision.is_none());
}

#[test]
fn test_mixed_precision_variants() {
    assert_ne!(MixedPrecision::Fp16, MixedPrecision::Bf16);
    assert_eq!(MixedPrecision::Fp16, MixedPrecision::Fp16);
}

#[test]
fn test_estimate_trainable_params_qlora() {
    let lora = LoRAConfig::new(8, 8.0);
    let config = FineTuneConfig::new("model").with_qlora(lora, 4);
    let params = config.estimate_trainable_params(1_000_000_000);
    // QLoRA should return much fewer params
    assert!(params < 1_000_000_000);
    assert!(params > 0);
}

#[test]
fn test_prefix_tuning_method() {
    let method = FineTuneMethod::PrefixTuning { prefix_length: 10 };
    if let FineTuneMethod::PrefixTuning { prefix_length } = method {
        assert_eq!(prefix_length, 10);
    } else {
        panic!("Expected PrefixTuning");
    }
}

#[test]
fn test_estimate_trainable_params_prefix() {
    // Test prefix tuning branch if it exists
    let config = FineTuneConfig {
        method: FineTuneMethod::PrefixTuning { prefix_length: 20 },
        ..FineTuneConfig::new("model")
    };
    let params = config.estimate_trainable_params(1_000_000);
    // Should return some estimate
    assert!(params > 0);
}

#[test]
fn test_fine_tune_config_clone() {
    let config = FineTuneConfig::new("model").epochs(10).batch_size(32);
    let cloned = config.clone();
    assert_eq!(config.epochs, cloned.epochs);
    assert_eq!(config.batch_size, cloned.batch_size);
}

#[test]
fn test_fine_tune_method_clone() {
    let method = FineTuneMethod::QLoRA { lora_config: LoRAConfig::default(), bits: 4 };
    let cloned = method.clone();
    if let FineTuneMethod::QLoRA { bits, .. } = cloned {
        assert_eq!(bits, 4);
    } else {
        panic!("Expected QLoRA after clone");
    }
}
