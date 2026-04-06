//! Check raw tensor values from GGUF
use realizar::gguf::GGUFModel;
use std::fs;

fn stats(x: &[f32]) -> (f32, f32, f32, f32) {
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let std: f32 = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
    (min, max, mean, std)
}

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    // Check token_embd.weight
    println!("Checking token_embd.weight...");
    if let Some(t) = model.tensors.iter().find(|t| t.name == "token_embd.weight") {
        println!("  dims: {:?}, qtype: {}", t.dims, t.qtype);
        let values = model
            .get_tensor_f32("token_embd.weight", &data)
            .expect("test");
        let (min, max, mean, std) = stats(&values);
        println!("  total elements: {}", values.len());
        println!(
            "  stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );

        // First token embedding (token 0)
        let hidden_dim = 2048;
        let tok0 = &values[0..hidden_dim];
        let (min, max, mean, std) = stats(tok0);
        println!(
            "  token 0 (UNK): min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );

        // BOS token embedding (token 1)
        let tok1 = &values[hidden_dim..2 * hidden_dim];
        let (min, max, mean, std) = stats(tok1);
        println!(
            "  token 1 (BOS): min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );

        // A common word token - let's try a few
        let tok100 = &values[100 * hidden_dim..101 * hidden_dim];
        let (min, max, mean, std) = stats(tok100);
        println!(
            "  token 100: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );
    }

    // Check output.weight (lm_head)
    println!("\nChecking output.weight...");
    if let Some(t) = model.tensors.iter().find(|t| t.name == "output.weight") {
        println!("  dims: {:?}, qtype: {}", t.dims, t.qtype);
        let values = model.get_tensor_f32("output.weight", &data).expect("test");
        let (min, max, mean, std) = stats(&values);
        println!("  total elements: {}", values.len());
        println!(
            "  stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );
    }

    // Check blk.0.attn_q.weight
    println!("\nChecking blk.0.attn_q.weight...");
    if let Some(t) = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
    {
        println!("  dims: {:?}, qtype: {}", t.dims, t.qtype);
        let values = model
            .get_tensor_f32("blk.0.attn_q.weight", &data)
            .expect("test");
        let (min, max, mean, std) = stats(&values);
        println!("  total elements: {}", values.len());
        println!(
            "  stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );
    }

    // Check blk.0.ffn_gate.weight (SwiGLU)
    println!("\nChecking blk.0.ffn_gate.weight...");
    if let Some(t) = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight")
    {
        println!("  dims: {:?}, qtype: {}", t.dims, t.qtype);
        let values = model
            .get_tensor_f32("blk.0.ffn_gate.weight", &data)
            .expect("test");
        let (min, max, mean, std) = stats(&values);
        println!("  total elements: {}", values.len());
        println!(
            "  stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            min, max, mean, std
        );
    }
}
