//! WGPU Training Example — Qwen3-4B QLoRA on AMD/Intel/Apple GPUs
//!
//! ```bash
//! cargo run --features gpu --release --example wgpu_train -- \
//!   --model /home/noah/src/models/qwen3-4b \
//!   --data /home/noah/src/bashrs/training/conversations_v4.jsonl \
//!   --steps 10 --lr 5e-4 --seq-len 64
//! ```

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("Error: --features gpu required");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    {
        if let Err(e) = run() {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), String> {
    use entrenar::train::transformer_trainer::wgpu_runner::{run_wgpu_training, WgpuTrainConfig};

    let args: Vec<String> = std::env::args().collect();

    let model_dir =
        get_arg(&args, "--model").unwrap_or_else(|| "/home/noah/src/models/qwen3-4b".to_string());
    let data_path = get_arg(&args, "--data")
        .unwrap_or_else(|| "/home/noah/src/bashrs/training/conversations_v4.jsonl".to_string());
    let steps: usize = get_arg(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(10);
    let lr: f32 = get_arg(&args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(1e-4);
    let seq_len: usize = get_arg(&args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(128);

    let config = WgpuTrainConfig {
        model_dir: model_dir.into(),
        data_path: data_path.into(),
        epochs: 3,
        lr,
        lora_rank: 32,
        lora_alpha: 64.0,
        seq_len,
        batch_size: 1,
        log_every: 1,
        save_every: steps,
        output_dir: get_arg(&args, "--output")
            .unwrap_or_else(|| "/home/noah/training-output".to_string())
            .into(),
        accumulation_steps: 4,
    };

    eprintln!("WGPU Training: {steps} steps, lr={lr}, seq_len={seq_len}");
    run_wgpu_training(&config)
}

#[cfg(feature = "gpu")]
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
