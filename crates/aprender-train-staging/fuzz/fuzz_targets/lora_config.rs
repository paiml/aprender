#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

/// Fuzz target for LoRA configuration validation
///
/// Tests that LoRA configuration calculations are robust to arbitrary inputs.
/// Validates parameter reduction calculations and memory estimation.

#[derive(Arbitrary, Debug)]
struct LoRAFuzzConfig {
    hidden_size: u16,
    num_layers: u8,
    rank: u16,
    alpha: u16,  // Will be converted to f32
    quantize_4bit: bool,
}

fuzz_target!(|config: LoRAFuzzConfig| {
    let hidden_size = config.hidden_size as usize;
    let num_layers = config.num_layers as usize;
    let rank = config.rank as usize;
    let alpha = (config.alpha as f32) / 10.0; // Convert to reasonable f32 range

    // Skip invalid configurations
    if hidden_size == 0 || num_layers == 0 || rank == 0 {
        return;
    }

    // Invariant 1: LoRA rank must be <= hidden_size (valid constraint)
    if rank > hidden_size {
        return; // Skip invalid configuration
    }

    // Invariant 2: Base model parameters calculation should never panic
    let base_params_per_layer = hidden_size
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(4)); // 4 attention projections

    let base_params = base_params_per_layer
        .and_then(|per_layer| per_layer.checked_mul(num_layers));

    // Invariant 3: LoRA adapter parameters calculation should never panic
    // Per projection: A [rank, hidden] + B [hidden, rank] = 2 * rank * hidden
    let lora_params_per_proj = rank
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(2));

    // 4 projections per layer
    let lora_params_per_layer = lora_params_per_proj
        .and_then(|per_proj| per_proj.checked_mul(4));

    let lora_params = lora_params_per_layer
        .and_then(|per_layer| per_layer.checked_mul(num_layers));

    // Invariant 4: Parameter reduction calculation should be valid
    if let (Some(base), Some(lora)) = (base_params, lora_params) {
        if base > 0 && base > lora {
            // Calculate reduction percentage (only when base > lora to avoid overflow)
            let reduction = ((base - lora) as f32 / base as f32) * 100.0;

            // Sanity check: reduction should be positive when rank < hidden_size
            // Edge case: when rank == hidden_size, LoRA doesn't reduce params (equivalent to full fine-tuning)
            if rank < hidden_size {
                assert!(reduction > 0.0, "LoRA should reduce parameters when rank < hidden_size");
                assert!(reduction < 100.0, "LoRA should not reduce to zero");
            }
        }
    }

    // Invariant 5: Memory calculations for different quantization levels
    if let Some(base) = base_params {
        // FP32: 4 bytes per parameter
        let _fp32_mb = (base as f64 * 4.0) / 1_000_000.0;

        // FP16: 2 bytes per parameter
        let _fp16_mb = (base as f64 * 2.0) / 1_000_000.0;

        // 4-bit: 0.5 bytes per parameter
        let _4bit_mb = (base as f64 * 0.5) / 1_000_000.0;
    }

    // Invariant 6: QLoRA memory calculation (4-bit base + FP32 adapters)
    if config.quantize_4bit {
        if let (Some(base), Some(lora)) = (base_params, lora_params) {
            // Base model in 4-bit
            let base_mem_4bit = (base as f64 * 0.5) / 1_000_000.0;

            // Adapters in FP32
            let adapter_mem_fp32 = (lora as f64 * 4.0) / 1_000_000.0;

            let _total_qlora_mem = base_mem_4bit + adapter_mem_fp32;

            // Sanity check: QLoRA memory should be less than FP32 LoRA
            let lora_fp32_mem = ((base + lora) as f64 * 4.0) / 1_000_000.0;
            assert!(
                base_mem_4bit + adapter_mem_fp32 < lora_fp32_mem,
                "QLoRA should use less memory than FP32 LoRA"
            );
        }
    }

    // Invariant 7: Scaling factor (alpha / rank) should be computable
    if rank > 0 {
        let _scaling_factor = alpha / (rank as f32);
        // Scaling factor can be any value, just should not panic
    }

    // Invariant 8: Test various rank values relative to hidden_size
    for test_rank in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        if test_rank <= hidden_size {
            let test_lora_params = test_rank
                .checked_mul(hidden_size)
                .and_then(|v| v.checked_mul(2))
                .and_then(|per_proj| per_proj.checked_mul(4))
                .and_then(|per_layer| per_layer.checked_mul(num_layers));

            if let (Some(base), Some(lora)) = (base_params, test_lora_params) {
                if base > 0 && base > lora {
                    let _reduction = ((base - lora) as f32 / base as f32) * 100.0;
                }
            }
        }
    }
});
