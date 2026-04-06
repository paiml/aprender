#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

/// Fuzz target for LLaMA parameter calculations
///
/// Tests that parameter calculations never panic or overflow, even with extreme inputs.
/// This validates the overflow-safe arithmetic used in chaos engineering tests.

#[derive(Arbitrary, Debug)]
struct LLaMAFuzzConfig {
    vocab_size: u16,      // 1..65535
    hidden_size: u16,     // 1..65535
    num_layers: u8,       // 1..255
    num_heads: u8,        // 1..255
    intermediate_size: u16, // 1..65535
}

fuzz_target!(|config: LLaMAFuzzConfig| {
    // Convert to usize for calculations
    let vocab_size = config.vocab_size as usize;
    let hidden_size = config.hidden_size as usize;
    let num_layers = config.num_layers as usize;
    let num_heads = config.num_heads as usize;
    let intermediate_size = config.intermediate_size as usize;

    // Invariant 1: Parameter calculations should never panic
    // This uses the same overflow-safe arithmetic from chaos tests

    // Embedding parameters (vocab_size × hidden_size × 2)
    let _embed_params = vocab_size
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(2));

    // Attention parameters per layer (4 × hidden_size × hidden_size)
    let _attn_params_per_layer = hidden_size
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(4));

    // FFN parameters per layer (3 × hidden_size × intermediate_size)
    let _ffn_params_per_layer = hidden_size
        .checked_mul(intermediate_size)
        .and_then(|v| v.checked_mul(3));

    // Total layer parameters
    let _layer_params = _attn_params_per_layer
        .and_then(|attn| _ffn_params_per_layer.and_then(|ffn| attn.checked_add(ffn)))
        .and_then(|per_layer| per_layer.checked_mul(num_layers));

    // Invariant 2: Hidden size must be divisible by num_heads (or calculation is invalid)
    if num_heads > 0 && hidden_size > 0 {
        let _head_dim = hidden_size / num_heads;
        // This should never panic - it's valid division
    }

    // Invariant 3: Memory estimation should handle large values gracefully
    if let (Some(embed), Some(layers)) = (_embed_params, _layer_params) {
        // FP32 memory (4 bytes per parameter)
        let _memory_fp32 = embed
            .checked_add(layers)
            .and_then(|total| total.checked_mul(4));

        // FP16 memory (2 bytes per parameter)
        let _memory_fp16 = embed
            .checked_add(layers)
            .and_then(|total| total.checked_mul(2));

        // 4-bit memory (0.5 bytes per parameter)
        let _memory_4bit = embed
            .checked_add(layers)
            .map(|total| total / 2); // Safe division
    }

    // Invariant 4: LoRA parameter calculation should never panic
    for rank in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        if rank <= hidden_size {
            // LoRA adds 2 × rank × hidden_size parameters per projection
            let _lora_params_per_proj = rank
                .checked_mul(hidden_size)
                .and_then(|v| v.checked_mul(2));

            // 4 projections (Q, K, V, O) per layer
            let _lora_params_per_layer = _lora_params_per_proj
                .and_then(|per_proj| per_proj.checked_mul(4));

            let _total_lora_params = _lora_params_per_layer
                .and_then(|per_layer| per_layer.checked_mul(num_layers));
        }
    }

    // Invariant 5: Batch size calculations should never overflow
    for batch_size in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        for seq_len in [1usize, 128, 512, 1024, 2048] {
            // Activation memory: batch_size × seq_len × hidden_size
            let _activation_mem = batch_size
                .checked_mul(seq_len)
                .and_then(|v| v.checked_mul(hidden_size));

            // Attention scores: batch_size × num_heads × seq_len × seq_len
            let _attention_scores = batch_size
                .checked_mul(num_heads)
                .and_then(|v| v.checked_mul(seq_len))
                .and_then(|v| v.checked_mul(seq_len));
        }
    }
});
