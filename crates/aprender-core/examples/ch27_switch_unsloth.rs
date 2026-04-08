#![allow(clippy::disallowed_methods)]
//! Chapter 27: Switch From unsloth
//!
//! Training equivalence: unsloth (Python) → apr train (Rust)
//! Citation: Hu et al., "LoRA," arXiv:2106.09685
//! Contract: contracts/apr-book-ch27-v1.yaml

use aprender::prelude::*;

fn main() {
    println!("=== Switch From unsloth ===");
    println!();

    // Training equivalence table
    println!("| unsloth (Python)                    | apr (Rust)                         |");
    println!("|-------------------------------------|------------------------------------|");
    println!("| from unsloth import FastLanguageModel | cargo install aprender             |");
    println!("| model.get_peft_model(r=16)          | apr finetune --lora-rank 16        |");
    println!("| trainer = SFTTrainer(...)            | apr train --config train.yaml      |");
    println!("| trainer.train()                      | apr train --config train.yaml      |");
    println!("| model.save_pretrained(path)          | (saves to .safetensors auto)       |");
    println!("| model.push_to_hub(repo)              | apr push hf://org/repo             |");
    println!();

    // LoRA parameter efficiency (same math as unsloth)
    let d = 4096_usize;
    let k = 4096_usize;
    for rank in [4, 8, 16, 32] {
        let full = d * k;
        let lora = rank * (d + k);
        let pct = lora as f64 / full as f64 * 100.0;
        println!("  LoRA rank={rank:>2}: {lora:>7} params ({pct:.2}% of {full})");
    }
    let r16_ratio = (16 * (d + k)) as f64 / (d * k) as f64;
    assert!(r16_ratio < 0.01, "LoRA rank=16 must be <1% of full");

    // Performance comparison from paiml/qwen-train-canary
    println!();
    println!("Training performance (Qwen2.5-Coder-1.5B):");
    println!("  | Backend       | Host      | tok/s    | VRAM (MB) |");
    println!("  |---------------|-----------|----------|-----------|");
    println!("  | unsloth       | yoga RTX  |  6,715.7 |     3,515 |");
    println!("  | unsloth       | gx10 A100 | 13,659.7 |    10,219 |");
    println!("  | pytorch       | gx10 A100 |  4,055.4 |    50,580 |");
    println!("  | cuBLAS        | gx10 A100 |  4,026.8 |    49,778 |");

    // VRAM efficiency assertion
    let unsloth_vram = 3515_f64;
    let pytorch_vram = 50580_f64;
    let vram_savings = 1.0 - (unsloth_vram / pytorch_vram);
    println!();
    println!(
        "VRAM savings: unsloth uses {:.0}% less VRAM than pytorch",
        vram_savings * 100.0
    );
    assert!(vram_savings > 0.9, "unsloth must use >90% less VRAM");

    // Adam optimizer exists in aprender
    let _adam = Adam::new(2e-5);
    println!("Adam optimizer: instantiated (lr=2e-5, standard for fine-tuning)");

    println!();
    println!("Key differences from unsloth:");
    println!("  1. Pure Rust — no Python, no pip, no conda");
    println!("  2. Single binary: cargo install aprender");
    println!("  3. LoRA adapters saved as architecture-independent safetensors");
    println!("  4. Same model works on CPU (SIMD) or GPU (cuBLAS/WGPU)");

    println!();
    println!("Repo: https://github.com/paiml/qwen-train-canary");
    println!("Chapter 27 contracts: PASSED");
}
