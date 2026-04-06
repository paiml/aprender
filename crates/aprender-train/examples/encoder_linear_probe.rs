//! Example: CodeBERT Encoder + Linear Probe pipeline
//!
//! Demonstrates the SSC v11 Stage 1 classifier:
//! 1. Create CodeBERT-sized encoder (mock weights)
//! 2. Extract [CLS] embeddings
//! 3. Train linear probe on cached embeddings
//! 4. Evaluate with MCC, bootstrap CI, baselines, generalization
//! 5. Check ship gate C-CLF-001
//!
//! Usage: cargo run -p entrenar --example encoder_linear_probe

use entrenar::finetune::{
    bootstrap_mcc_ci, check_ship_gate, compare_baselines, compute_confidence_scores,
    evaluate_classification, generalization_test, should_escalate, EscalationLevel, LinearProbe,
};
use entrenar::transformer::{EncoderModel, ModelArchitecture, TransformerConfig};

fn create_encoder() -> (TransformerConfig, EncoderModel) {
    let config = TransformerConfig {
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 4,
        intermediate_size: 64,
        vocab_size: 100,
        max_position_embeddings: 32,
        architecture: ModelArchitecture::Encoder,
        ..TransformerConfig::tiny()
    };
    let encoder = EncoderModel::new(&config);
    println!(
        "Encoder: {} layers, hidden={}, params={}",
        config.num_hidden_layers,
        config.hidden_size,
        encoder.num_parameters()
    );
    (config, encoder)
}

fn extract_embeddings(
    encoder: &EncoderModel,
    num_safe: usize,
    num_unsafe: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let total = num_safe + num_unsafe;
    let mut embeddings = Vec::with_capacity(total);
    let mut labels = Vec::with_capacity(total);

    for i in 0..total {
        let token_ids: Vec<u32> =
            if i < num_safe { vec![1, 2, 3, 4] } else { vec![50, 60, 70, 80] };
        let cls = encoder.cls_embedding(&token_ids);
        let data = cls.data();
        embeddings.push(data.as_slice().expect("contiguous").to_vec());
        labels.push(usize::from(i >= num_safe));
    }
    println!("  Extracted {total} embeddings (safe={num_safe}, unsafe={num_unsafe})");
    (embeddings, labels)
}

fn train_probe(
    hidden_size: usize,
    embeddings: &[Vec<f32>],
    labels: &[usize],
    num_safe: usize,
    num_unsafe: usize,
) -> LinearProbe {
    let mut probe = LinearProbe::new(hidden_size, 2);
    println!("  Probe params: {} (768*2+2=1538 for full CodeBERT)", probe.num_parameters());

    let total = (num_safe + num_unsafe) as f32;
    let class_weights = vec![total / (2.0 * num_safe as f32), total / (2.0 * num_unsafe as f32)];
    println!("  Class weights: safe={:.3}, unsafe={:.3}", class_weights[0], class_weights[1]);

    let final_loss = probe.train(embeddings, labels, 20, 0.1, Some(&class_weights));
    println!("  Final loss: {final_loss:.4}");
    probe
}

fn evaluate_and_report(probe: &LinearProbe, embeddings: &[Vec<f32>], labels: &[usize]) {
    let predictions: Vec<usize> = embeddings
        .iter()
        .map(|emb| {
            let t = entrenar::Tensor::from_vec(emb.clone(), false);
            probe.predict(&t)
        })
        .collect();

    let metrics = evaluate_classification(&predictions, labels, 2);
    println!("  Accuracy: {:.1}%", metrics.accuracy * 100.0);
    println!("  MCC: {:.4}", metrics.mcc);
    println!("  Safe recall: {:.3}, Unsafe recall: {:.3}", metrics.recall[0], metrics.recall[1]);

    let ci = bootstrap_mcc_ci(&predictions, labels, 2, 1000);
    println!("  MCC Bootstrap CI (95%): [{:.4}, {:.4}]", ci.lower, ci.upper);

    // Baselines
    println!("\nBaselines comparison:");
    let baselines = vec![("majority", 0.0_f32), ("keyword", 0.35), ("linter", 0.50)];
    for c in compare_baselines(metrics.mcc, &baselines) {
        let status = if c.beats_baseline { "BEATS" } else { "LOSES" };
        println!(
            "  {} {}: model={:.3} vs baseline={:.3}",
            status, c.name, c.model_mcc, c.baseline_mcc
        );
    }

    // Generalization
    println!("\nGeneralization test:");
    let novel_unsafe: Vec<Vec<f32>> = embeddings[80..].to_vec();
    let gen = generalization_test(probe, &novel_unsafe, 1);
    println!("  Detected: {}/{} ({:.1}%)", gen.detected, gen.total, gen.detection_rate * 100.0);

    // Confidence scores
    println!("\nConfidence scores (first 3):");
    for (i, s) in compute_confidence_scores(probe, &embeddings[..3]).iter().enumerate() {
        println!("  Sample {i}: class={} conf={:.3}", s.predicted_class, s.confidence);
    }

    // Ship gate
    println!("\n=== Ship Gate C-CLF-001 ===");
    let gate = check_ship_gate(&ci, metrics.accuracy, &gen, EscalationLevel::LinearProbe);
    println!(
        "  MCC: {} | Accuracy: {} | Generalization: {} | Ship: {}",
        if gate.mcc_passes { "PASS" } else { "FAIL" },
        if gate.accuracy_passes { "PASS" } else { "FAIL" },
        if gate.generalization_passes { "PASS" } else { "FAIL" },
        if gate.ship_ready { "YES" } else { "NO" },
    );

    if let Some(next) = should_escalate(EscalationLevel::LinearProbe, &ci, metrics.accuracy) {
        println!("  Escalation needed: {next}");
    } else {
        println!("  No escalation needed — ship gate met!");
    }
}

fn main() {
    println!("=== SSC v11 Stage 1: CodeBERT Linear Probe ===\n");

    let (config, encoder) = create_encoder();

    println!("\nExtracting [CLS] embeddings...");
    let (embeddings, labels) = extract_embeddings(&encoder, 80, 20);

    println!("\nTraining linear probe...");
    let probe = train_probe(config.hidden_size, &embeddings, &labels, 80, 20);

    println!("\nEvaluating...");
    evaluate_and_report(&probe, &embeddings, &labels);

    println!("\n=== Done ===");
}
