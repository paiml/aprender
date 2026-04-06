//! Continual Pre-Training (CPT) Pipeline (GH-448)
//!
//! Demonstrates the CPT pipeline components: LR scheduling, data mixing,
//! replay buffer, and progress tracking.
//!
//! Run: cargo run --example continual_pretraining

use aprender::online::cpt::{CptConfig, CptProgress, CptSchedule, DataMixer, ReplayBuffer};

fn main() {
    println!("=== Continual Pre-Training Pipeline (GH-448) ===\n");

    let config = CptConfig {
        learning_rate: 1e-4,
        warmup_steps: 10,
        total_steps: 50,
        seq_length: 128,
        domain_mix_ratio: 0.7,
        replay_buffer_size: 20,
        ..Default::default()
    };
    config.validate().expect("Config should be valid");

    // ── 1. Learning Rate Schedule ──
    println!("── 1. Learning Rate Schedule ──");
    let schedule = CptSchedule::new(&config);
    for step in [0, 5, 10, 25, 40, 50] {
        println!("  Step {:3}: lr = {:.8}", step, schedule.lr_at_step(step));
    }

    // ── 2. Data Mixing ──
    println!(
        "\n── 2. Data Mixing (ratio={:.1}) ──",
        config.domain_mix_ratio
    );
    let mut mixer = DataMixer::new(config.domain_mix_ratio, config.seed);

    let domain_data = vec![
        vec![100, 101, 102],
        vec![103, 104, 105],
        vec![106, 107, 108],
    ];
    let general_data = vec![vec![200, 201, 202], vec![203, 204, 205]];

    let batch = mixer.mix_batches(&domain_data, &general_data, 4);
    println!("  Mixed batch ({} samples):", batch.len());
    for (i, seq) in batch.iter().enumerate() {
        let source = if seq[0] >= 100 && seq[0] < 200 {
            "domain"
        } else {
            "general"
        };
        println!("    [{i}] {:?} ({})", seq, source);
    }

    // ── 3. Replay Buffer ──
    println!("\n── 3. Experience Replay Buffer ──");
    let mut replay = ReplayBuffer::new(config.replay_buffer_size);

    // Fill with "original training data"
    for i in 0..25 {
        replay.add(vec![i, i + 1, i + 2]);
    }
    println!("  Buffer: {}/{} capacity", replay.len(), replay.capacity());

    let replayed = replay.sample(3, 42);
    println!("  Sampled {} replay examples:", replayed.len());
    for (i, seq) in replayed.iter().enumerate() {
        println!("    [{i}] {:?}", seq);
    }

    // ── 4. Training Loop Simulation ──
    println!("\n── 4. Training Progress ──");
    let mut progress = CptProgress::new(config.total_steps);
    let mut domain_mixer = DataMixer::new(config.domain_mix_ratio, config.seed);

    for step in 0..config.total_steps {
        let lr = schedule.lr_at_step(step);
        let is_domain = domain_mixer.next_is_domain();
        // Simulate loss (decreasing over time)
        let loss = 5.0 * (1.0 - step as f64 / config.total_steps as f64) + 0.5;
        progress.update(lr, loss, is_domain);

        if step % 10 == 0 || step == config.total_steps - 1 {
            println!(
                "  Step {:3}/{}: lr={:.2e}, avg_loss={:.4}, domain={}/{}, progress={:.0}%",
                progress.step,
                progress.total_steps,
                progress.current_lr,
                progress.avg_loss,
                progress.domain_samples,
                progress.general_samples,
                progress.fraction() * 100.0
            );
        }
    }

    assert!(progress.is_done());
    println!("\n=== CPT pipeline complete ===");
}
