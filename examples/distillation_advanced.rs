//! Advanced Distillation Strategies (GH-451)
//!
//! Demonstrates hidden-state matching, quantization-aware distillation,
//! and online distillation with EMA smoothing.
//!
//! Run: cargo run --example distillation_advanced

use aprender::online::distillation_advanced::{
    HiddenProjection, HiddenStateConfig, HiddenStateDistiller, OnlineDistillConfig,
    OnlineDistiller, QuantAwareConfig, QuantAwareDistiller,
};

fn main() {
    println!("=== Advanced Distillation Strategies (GH-451) ===\n");

    // ── 1. Hidden-State Matching ──
    println!("── 1. Hidden-State Matching (FitNets) ──");
    let config = HiddenStateConfig {
        teacher_dim: 8,
        student_dim: 4,
        layer_map: vec![(0, 0), (2, 1)],
        projection_lr: 0.01,
        ..Default::default()
    };
    let mut distiller = HiddenStateDistiller::new(config);

    let teacher = vec![
        vec![1.0, -0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.6],
        vec![0.0; 8], // unused layer
        vec![0.5, 0.5, -0.5, -0.5, 1.0, -1.0, 0.0, 0.0],
    ];
    let student = vec![vec![0.3, -0.1, 0.5, 0.2], vec![-0.2, 0.4, 0.1, -0.3]];

    let loss_before = distiller.hidden_loss(&teacher, &student);
    for _ in 0..200 {
        distiller.update_projections(&teacher, &student);
    }
    let loss_after = distiller.hidden_loss(&teacher, &student);
    println!("  Layer map: {:?}", distiller.layer_map());
    println!("  Loss before: {:.6}", loss_before);
    println!(
        "  Loss after:  {:.6} ({:.1}% reduction)",
        loss_after,
        (1.0 - loss_after / loss_before) * 100.0
    );

    // ── 2. Quantization-Aware Distillation ──
    println!("\n── 2. Quantization-Aware Distillation ──");
    let weights = vec![0.3, -0.7, 0.1, 0.9, -0.5, 0.2, -0.8, 0.4];

    for bits in [4, 8] {
        let d = QuantAwareDistiller::new(QuantAwareConfig {
            bits,
            symmetric: true,
            ..Default::default()
        });
        let error = d.quantization_error(&weights);
        let quantized = d.fake_quantize(&weights);
        println!("  {}-bit symmetric:", bits);
        println!("    Error:     {:.8}", error);
        println!("    Original:  {:?}", &weights[..4]);
        println!("    Quantized: {:?}", &quantized[..4]);
    }

    // Error diffusion
    let d = QuantAwareDistiller::new(QuantAwareConfig {
        bits: 4,
        symmetric: true,
        error_diffusion: 1.0,
        ..Default::default()
    });
    let diffused = d.fake_quantize_diffused(&weights);
    println!("  4-bit with error diffusion:");
    println!("    Diffused:  {:?}", &diffused[..4]);

    // Polynomial activation approximation
    let d_poly = QuantAwareDistiller::new(QuantAwareConfig {
        poly_degree: 3,
        ..Default::default()
    });
    let x: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 / (1.0 + (-xi).exp())).collect();
    let coeffs = d_poly.polynomial_activation_approx(&x, &y).unwrap();
    println!(
        "  Sigmoid poly approx (degree 3): {:?}",
        coeffs
            .iter()
            .map(|c| format!("{:.4}", c))
            .collect::<Vec<_>>()
    );

    // ── 3. Online Distillation ──
    println!("\n── 3. Online Distillation with EMA ──");
    let config = OnlineDistillConfig {
        ema_decay: Some(0.9),
        temperature: 3.0,
        alpha: 0.7,
        ..Default::default()
    };
    let mut online = OnlineDistiller::new(config);

    let student_logits = vec![1.0, 2.0, 0.5];
    let labels = vec![0.0, 1.0, 0.0];

    // Simulate 5 steps with varying teacher outputs
    for step in 0..5 {
        let teacher_logits = vec![1.5 + step as f64 * 0.1, 2.5 - step as f64 * 0.1, 0.3];
        let loss = online
            .step(&student_logits, &teacher_logits, &labels)
            .unwrap();
        let ema = online.ema_logits().unwrap();
        println!("  Step {}: loss={:.4}, EMA[0]={:.4}", step, loss, ema[0]);
    }
    println!("  Total updates: {}", online.update_count());

    println!("\n=== All distillation strategies verified ===");
}
