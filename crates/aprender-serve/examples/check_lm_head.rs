//! Check LM head and embedding patterns
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== LM Head Check ===\n");

    let hidden_dim = model.config().hidden_dim;

    println!("LM head weight:");
    println!("  qtype: {}", model.lm_head_weight().qtype);
    println!("  in_dim: {}", model.lm_head_weight().in_dim);
    println!("  out_dim: {}", model.lm_head_weight().out_dim);

    // For Q8_0, let's verify that dequantized weights make sense
    // Q8_0 format: blocks of 32 values, each block has:
    // - 2 bytes f16 scale
    // - 32 bytes int8 quantized values
    // Total: 34 bytes per block

    let num_blocks_per_row = model.lm_head_weight().in_dim.div_ceil(32);
    let bytes_per_row = num_blocks_per_row * 34;

    println!("  Q8_0 blocks per row: {}", num_blocks_per_row);
    println!("  Bytes per row: {}", bytes_per_row);
    println!(
        "  Expected total bytes: {}",
        bytes_per_row * model.lm_head_weight().out_dim
    );
    println!("  Actual data len: {}", model.lm_head_weight().data.len());

    // Verify by dequantizing first row (token 0)
    let mut dequant_row0 = vec![0.0f32; model.lm_head_weight().in_dim];
    for block_idx in 0..num_blocks_per_row {
        let block_offset = block_idx * 34;
        let scale_bytes = [
            model.lm_head_weight().data[block_offset],
            model.lm_head_weight().data[block_offset + 1],
        ];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        for i in 0..32 {
            let idx = block_idx * 32 + i;
            if idx >= model.lm_head_weight().in_dim {
                break;
            }
            let quant = model.lm_head_weight().data[block_offset + 2 + i] as i8;
            dequant_row0[idx] = scale * quant as f32;
        }
    }

    // Compare dequantized LM head row 0 with token embedding 0
    let emb_0 = &model.token_embedding()[0..hidden_dim];

    println!("\n=== Comparing LM head row 0 with token 0 embedding ===");
    println!("Dequant LM head[0] first 8: {:?}", &dequant_row0[..8]);
    println!("Token embedding[0] first 8: {:?}", &emb_0[..8]);

    let diff: f32 = dequant_row0
        .iter()
        .zip(emb_0.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / hidden_dim as f32;

    println!("Mean absolute difference: {:.6}", diff);

    if diff < 0.01 {
        println!("✓ LM head appears to be tied embeddings (same as token_embedding)");
    } else {
        println!("✗ LM head is different from token_embedding");
    }

    // Compute what logit[0] would be for a test hidden state
    println!("\n=== Manual logit computation test ===");

    // Use a simple test hidden: all ones normalized
    let test_hidden: Vec<f32> = vec![1.0f32; hidden_dim];
    let test_norm: f32 = test_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    let test_normalized: Vec<f32> = test_hidden.iter().map(|x| x / test_norm).collect();

    // Dot product with token 0 embedding (which equals LM head row 0 for tied)
    let manual_logit_0: f32 = test_normalized
        .iter()
        .zip(emb_0.iter())
        .map(|(h, e)| h * e)
        .sum();

    println!("Test: uniform hidden vector (normalized)");
    println!("  Manual logit[0] (dot with emb): {:.4}", manual_logit_0);

    // Check logit[19] ("4") as well
    let emb_19 = &model.token_embedding()[19 * hidden_dim..20 * hidden_dim];
    let manual_logit_19: f32 = test_normalized
        .iter()
        .zip(emb_19.iter())
        .map(|(h, e)| h * e)
        .sum();

    println!("  Manual logit[19] (dot with emb): {:.4}", manual_logit_19);

    // The key question: what hidden state makes token 0 rank highest?
    // Token 0 has the highest dot product when hidden aligns with emb_0

    println!("\n=== What makes token 0 rank high? ===");

    // Compute the direction that maximizes token 0's logit: it's emb_0 itself
    // Any hidden state with positive components along emb_0's direction will favor token 0

    // Let's see if emb_0 has a specific pattern
    let emb_0_sum: f32 = emb_0.iter().sum();
    let emb_0_abs_sum: f32 = emb_0.iter().map(|x| x.abs()).sum();
    let emb_0_pos_count = emb_0.iter().filter(|&&x| x > 0.0).count();
    let emb_0_neg_count = emb_0.iter().filter(|&&x| x < 0.0).count();

    println!("Token 0 embedding:");
    println!("  sum: {:.4}", emb_0_sum);
    println!("  abs_sum: {:.4}", emb_0_abs_sum);
    println!("  positive dims: {}", emb_0_pos_count);
    println!("  negative dims: {}", emb_0_neg_count);

    // Compare with digit tokens
    println!("\nDigit token embeddings:");
    for d in ['0', '1', '2', '3', '4'] {
        let tok_id = vocab
            .iter()
            .enumerate()
            .find(|(_, s)| s.as_str() == d.to_string())
            .map(|(i, _)| i);
        if let Some(tok_id) = tok_id {
            let emb = &model.token_embedding()[tok_id * hidden_dim..(tok_id + 1) * hidden_dim];
            let sum: f32 = emb.iter().sum();
            let abs_sum: f32 = emb.iter().map(|x| x.abs()).sum();
            let pos = emb.iter().filter(|&&x| x > 0.0).count();
            println!(
                "  '{}' ({}): sum={:.4}, abs_sum={:.4}, pos={}",
                d, tok_id, sum, abs_sum, pos
            );
        }
    }

    Ok(())
}
