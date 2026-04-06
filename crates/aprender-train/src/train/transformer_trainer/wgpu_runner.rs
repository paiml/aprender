//! WGPU Training Runner — end-to-end Qwen3-4B QLoRA training on AMD GPUs
//!
//! Connects: tokenizer → data loading → 36-layer forward/backward → AdamW
//!
//! # Contract: wgpu-transformer-trainer-v1.yaml (C-WGPU-TRAIN-001)
//!
//! # Usage
//!
//! ```bash
//! cargo run --features gpu --release --example wgpu_train -- \
//!   --model /home/noah/src/models/qwen3-4b \
//!   --data /home/noah/src/bashrs/training/conversations_v4.jsonl \
//!   --epochs 1 --lr 5e-4 --lora-rank 16 --seq-len 128
//! ```

#[cfg(feature = "gpu")]
use super::wgpu_trainer::{WgpuModelState, WgpuTransformerTrainer};

/// Training configuration
#[cfg(feature = "gpu")]
pub struct WgpuTrainConfig {
    pub model_dir: std::path::PathBuf,
    pub data_path: std::path::PathBuf,
    pub epochs: usize,
    pub lr: f32,
    pub lora_rank: u32,
    pub lora_alpha: f32,
    pub seq_len: usize,
    pub batch_size: usize,
    pub log_every: usize,
    pub save_every: usize,
    pub output_dir: std::path::PathBuf,
    /// Gradient accumulation steps (effective batch_size = accumulation_steps)
    pub accumulation_steps: usize,
}

/// Load and tokenize one training example
#[cfg(feature = "gpu")]
fn tokenize_example(
    tokenizer: &crate::tokenizer::HfTokenizer,
    text: &str,
    max_len: usize,
) -> (Vec<u32>, Vec<u32>) {
    let tokens = tokenizer.encode(text);
    let len = tokens.len().min(max_len);
    let input_ids = tokens[..len].to_vec();
    // Target = shifted input (next token prediction)
    let target_ids: Vec<u32> = if len > 1 { tokens[1..len].to_vec() } else { vec![0] };
    (input_ids, target_ids)
}

/// Run WGPU training
///
/// # Contract (C-WGPU-TRAIN-001)
#[cfg(feature = "gpu")]
pub fn run_wgpu_training(config: &WgpuTrainConfig) -> Result<(), String> {
    use crate::tokenizer::HfTokenizer;
    use crate::transformer::TransformerConfig;

    eprintln!("=== WGPU Training: Qwen3-4B QLoRA on AMD GPU ===\n");

    // 1. Load tokenizer
    let tokenizer_path = config.model_dir.join("tokenizer.json");
    let tokenizer =
        HfTokenizer::from_file(&tokenizer_path).map_err(|e| format!("Tokenizer: {e}"))?;
    eprintln!("Tokenizer loaded: {}", tokenizer_path.display());

    // 2. Load model
    let mut model =
        WgpuModelState::load_qwen3_4b(&config.model_dir, config.lora_rank, config.lora_alpha)?;
    eprintln!("Model: {} trainable params\n", model.trainable_params());

    // 3. Load data
    let data_str = std::fs::read_to_string(&config.data_path).map_err(|e| format!("Data: {e}"))?;
    let examples: Vec<String> = data_str
        .lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| v["text"].as_str().map(std::string::ToString::to_string))
        })
        .collect();
    eprintln!("Data: {} examples from {}\n", examples.len(), config.data_path.display());

    // 4. Create trainer
    let mut tc = TransformerConfig::llama2_7b();
    tc.hidden_size = model.hidden_size;
    tc.intermediate_size = model.intermediate_size;
    tc.num_hidden_layers = model.num_layers;
    tc.num_attention_heads = model.num_heads;
    tc.num_kv_heads = model.num_kv_heads;
    tc.vocab_size = model.vocab_size;

    // Scale lr by 1/accumulation_steps for gradient accumulation equivalence
    let effective_lr = config.lr / config.accumulation_steps.max(1) as f32;
    let mut trainer = WgpuTransformerTrainer::new(&tc, effective_lr)?;
    eprintln!(
        "Effective lr: {effective_lr} (lr={} / accum={})\n",
        config.lr, config.accumulation_steps
    );

    // 5. Training loop
    let mut total_loss = 0.0f32;
    let mut step = 0usize;

    let mut best_loss = f32::INFINITY;
    for epoch in 0..config.epochs {
        // Shuffle data each epoch (deterministic seed for reproducibility)
        let mut indices: Vec<usize> = (0..examples.len()).collect();
        if epoch > 0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            epoch.hash(&mut hasher);
            let seed = hasher.finish();
            // Fisher-Yates shuffle with deterministic seed
            for i in (1..indices.len()).rev() {
                let j =
                    ((seed.wrapping_mul(i as u64 + 1).wrapping_add(7)) % (i as u64 + 1)) as usize;
                indices.swap(i, j);
            }
        }
        eprintln!("--- Epoch {}/{} ({} examples) ---", epoch + 1, config.epochs, examples.len());

        for (idx, &ei) in indices.iter().enumerate() {
            let text = &examples[ei];
            let (input_ids, target_ids) = tokenize_example(&tokenizer, text, config.seq_len);
            if input_ids.len() < 2 {
                continue;
            }

            // Create embedding (simplified: use token IDs as indices into lm_head)
            let seq_len = target_ids.len() as u32;
            let h = model.hidden_size;
            let mut hidden = vec![0.0f32; seq_len as usize * h];
            for (si, &tid) in input_ids[..target_ids.len()].iter().enumerate() {
                let tid = (tid as usize).min(model.vocab_size - 1);
                for hi in 0..h {
                    hidden[si * h + hi] = model.lm_head[tid * h + hi];
                }
            }

            // Training step
            let (loss, gnorm) = trainer.full_train_step(&hidden, &target_ids, &mut model)?;

            total_loss += loss;
            step += 1;

            if step.is_multiple_of(config.log_every) {
                let avg_loss = total_loss / step as f32;
                eprintln!(
                    "  step={step} loss={loss:.3} avg_loss={avg_loss:.3} gnorm={gnorm:.2e} [{}/{}]",
                    idx + 1,
                    examples.len()
                );
            }

            if config.save_every > 0 && step.is_multiple_of(config.save_every) {
                model.save_checkpoint(
                    &config.output_dir,
                    step as u32,
                    loss,
                    config.lora_rank,
                    config.lora_alpha,
                )?;
            }
            if loss < best_loss {
                best_loss = loss;
            }
        }
    }

    let final_avg = total_loss / step.max(1) as f32;

    // Save final checkpoint
    model.save_checkpoint(
        &config.output_dir,
        step as u32,
        final_avg,
        config.lora_rank,
        config.lora_alpha,
    )?;

    eprintln!("\n=== Training complete: {step} steps, avg_loss={final_avg:.3} ===");
    Ok(())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// Smoke test: tokenize + 1 training step with real data
    #[test]
    fn test_wgpu_training_smoke() {
        let model_dir = std::path::Path::new("/home/noah/src/models/qwen3-4b");
        let data_path =
            std::path::Path::new("/home/noah/src/bashrs/training/conversations_v4.jsonl");

        if !model_dir.exists() || !data_path.exists() {
            eprintln!("Skipping: model or data not found");
            return;
        }

        // Load tokenizer
        let tokenizer = crate::tokenizer::HfTokenizer::from_file(model_dir.join("tokenizer.json"))
            .expect("tokenizer");

        // Tokenize first example
        let data = std::fs::read_to_string(data_path).expect("read data");
        let first_line = data.lines().next().expect("first line");
        let text: serde_json::Value = serde_json::from_str(first_line).expect("parse json");
        let text = text["text"].as_str().expect("text field");

        let (input_ids, target_ids) = tokenize_example(&tokenizer, text, 32);
        eprintln!(
            "Tokenized: {} tokens, first 5: {:?}",
            input_ids.len(),
            &input_ids[..5.min(input_ids.len())]
        );

        assert!(input_ids.len() >= 2, "Need at least 2 tokens");
        assert_eq!(target_ids.len(), input_ids.len() - 1);

        // Load model and run 1 step
        let mut model = WgpuModelState::load_qwen3_4b(model_dir, 16, 32.0).expect("model");

        let mut config = crate::transformer::TransformerConfig::llama2_7b();
        config.hidden_size = 2560;
        config.intermediate_size = 9728;
        config.num_hidden_layers = 36;
        config.vocab_size = 151936;

        let mut trainer = WgpuTransformerTrainer::new(&config, 5e-4).expect("trainer");

        // Embedding lookup
        let seq_len = target_ids.len();
        let h = 2560;
        let mut hidden = vec![0.0f32; seq_len * h];
        for (si, &tid) in input_ids[..seq_len].iter().enumerate() {
            let tid = (tid as usize).min(151935);
            for hi in 0..h {
                hidden[si * h + hi] = model.lm_head[tid * h + hi];
            }
        }

        let start = std::time::Instant::now();
        let (loss, gnorm) =
            trainer.full_train_step(&hidden, &target_ids, &mut model).expect("train step");
        let elapsed = start.elapsed();

        eprintln!(
            "WGPU smoke: loss={loss:.3}, gnorm={gnorm:.2e}, time={:.1}s",
            elapsed.as_secs_f64()
        );
        assert!(loss.is_finite(), "Loss must be finite");
        assert!(loss > 0.0, "Loss must be positive");
    }
}
