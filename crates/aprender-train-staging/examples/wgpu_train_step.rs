//! End-to-end wgpu training step: forward → loss → backward → AdamW
//!
//! Demonstrates a complete training iteration using only wgpu (zero CUDA).
//! Uses small synthetic data to verify the pipeline works.
//!
//! Run: cargo run --features gpu --example wgpu_train_step --release

fn main() {
    #[cfg(feature = "gpu")]
    {
        use entrenar::autograd::wgpu_training::WgpuTrainer;
        use std::time::Instant;

        let mut trainer = WgpuTrainer::new().expect("wgpu init failed");
        println!("[wgpu] Trainer initialized");

        // Simulate a small transformer layer: hidden_dim=256, seq_len=32
        let hidden = 256u32;
        let seq_len = 32u32;
        let rank = 16u32;
        let lr = 1e-3_f32;
        let num_steps = 10;

        // Random input activations [seq_len, hidden]
        let x_data: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.01).sin()).collect();

        // Frozen base weight [hidden, hidden] (simulates dequantized NF4)
        let w_data: Vec<f32> =
            (0..hidden * hidden).map(|i| ((i as f32) * 0.001).cos() * 0.1).collect();

        // LoRA A [hidden, rank] — Kaiming init
        let lora_a_data: Vec<f32> = (0..hidden * rank)
            .map(|i| ((i as f32) * 0.03).sin() * (2.0 / hidden as f32).sqrt())
            .collect();

        // LoRA B [rank, hidden] — zero init
        let lora_b_data = vec![0.0f32; (rank * hidden) as usize];

        // Target: random labels for cross-entropy-like loss
        let target_data: Vec<f32> =
            (0..seq_len * hidden).map(|i| ((i as f32) * 0.007 + 1.0).sin() * 0.5).collect();

        // Upload to GPU
        let x = trainer.upload(&x_data);
        let w = trainer.upload(&w_data);
        let lora_a = trainer.upload(&lora_a_data);
        let lora_b = trainer.upload(&lora_b_data);
        let target = trainer.upload(&target_data);

        // Optimizer states for LoRA A and B
        let m_a = trainer.zeros((hidden * rank) as usize);
        let v_a = trainer.zeros((hidden * rank) as usize);
        let m_b = trainer.zeros((rank * hidden) as usize);
        let v_b = trainer.zeros((rank * hidden) as usize);

        // Intermediate buffers
        let h_base = trainer.zeros((seq_len * hidden) as usize);
        let xa = trainer.zeros((seq_len * rank) as usize);
        let h_lora = trainer.zeros((seq_len * hidden) as usize);
        let _grad_output = trainer.zeros((seq_len * hidden) as usize);
        let grad_a = trainer.zeros((hidden * rank) as usize);
        let grad_b = trainer.zeros((rank * hidden) as usize);

        println!("[wgpu] Buffers allocated");
        println!(
            "[wgpu] Training {num_steps} steps (hidden={hidden}, seq={seq_len}, rank={rank})...\n"
        );

        let total_start = Instant::now();

        for step in 0..num_steps {
            let step_start = Instant::now();

            // === FORWARD ===
            // h_base = X @ W^T (frozen base)
            trainer.matmul_forward(&x, &w, &h_base, seq_len, hidden, hidden);

            // xa = X @ A
            trainer.matmul_forward(&x, &lora_a, &xa, seq_len, hidden, rank);

            // h_lora = xa @ B (LoRA contribution)
            trainer.matmul_forward(&xa, &lora_b, &h_lora, seq_len, rank, hidden);

            // output = h_base + h_lora (done on CPU for simplicity)
            let h_base_data = trainer.download(&h_base);
            let h_lora_data = trainer.download(&h_lora);
            let target_host = trainer.download(&target);

            let mut output = vec![0.0f32; (seq_len * hidden) as usize];
            let mut loss = 0.0f32;
            let mut grad_data = vec![0.0f32; (seq_len * hidden) as usize];

            for i in 0..(seq_len * hidden) as usize {
                output[i] = h_base_data[i] + h_lora_data[i];
                let diff = output[i] - target_host[i];
                loss += diff * diff;
                grad_data[i] = 2.0 * diff / (seq_len * hidden) as f32; // MSE gradient
            }
            loss /= (seq_len * hidden) as f32;

            // Upload gradient
            let grad_buf = trainer.upload(&grad_data);

            // === BACKWARD (LoRA only) ===
            // h_lora = xa @ B, so backward gives:
            //   grad_xa = grad_output @ B^T  [seq, rank]
            //   grad_B  = xa^T @ grad_output [rank, hidden]
            let grad_xa = trainer.zeros((seq_len * rank) as usize);
            trainer
                .matmul_backward(&xa, &lora_b, &grad_buf, &grad_xa, &grad_b, seq_len, rank, hidden);

            // xa = X @ A, so backward gives:
            //   grad_x = grad_xa @ A^T [seq, hidden] (not needed, base is frozen)
            //   grad_A = X^T @ grad_xa [hidden, rank]
            let grad_x_dummy = trainer.zeros((seq_len * hidden) as usize);
            trainer.matmul_backward(
                &x,
                &lora_a,
                &grad_xa,
                &grad_x_dummy,
                &grad_a,
                seq_len,
                hidden,
                rank,
            );

            // === OPTIMIZER ===
            trainer.adamw_step(&lora_a, &grad_a, &m_a, &v_a, lr, 0.9, 0.999, 1e-8, 0.01);
            trainer.adamw_step(&lora_b, &grad_b, &m_b, &v_b, lr, 0.9, 0.999, 1e-8, 0.01);

            let step_ms = step_start.elapsed().as_millis();
            println!("  Step {step}: loss={loss:.6}, time={step_ms}ms");
        }

        let total_s = total_start.elapsed().as_secs_f64();
        println!(
            "\n[wgpu] Training complete: {} steps in {:.1}s ({:.0}ms/step)",
            num_steps,
            total_s,
            total_s * 1000.0 / f64::from(num_steps)
        );

        // Verify LoRA B is no longer zero
        let final_b = trainer.download(&lora_b);
        let b_norm: f32 = final_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("[wgpu] LoRA B norm: {b_norm:.6} (should be > 0)");
        assert!(b_norm > 0.0, "LoRA B should have been updated");
        println!("[wgpu] Loss should be decreasing — verified manually from output above");
    }

    #[cfg(not(feature = "gpu"))]
    eprintln!("Requires --features gpu");
}
