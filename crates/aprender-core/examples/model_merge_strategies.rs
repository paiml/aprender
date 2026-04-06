#![allow(clippy::disallowed_methods)]
//! Model Merge Strategies Example (GH-245)
//!
//! Demonstrates all 5 merge strategies available in `apr merge`:
//! - Average: Simple weight averaging (ensemble)
//! - Weighted: Weighted average with user-specified weights
//! - SLERP: Spherical Linear Interpolation (2 models only)
//! - TIES: Trim, Elect Sign, Merge (requires base model)
//! - DARE: Drop And Rescale (requires base model)
//!
//! These are the three dominant merge methods on HuggingFace — many
//! top-ranked open models are merges, not trained from scratch.
//!
//! Run with: `cargo run --example model_merge_strategies`

use aprender::format::{apr_merge, MergeOptions, MergeStrategy};
use aprender::serialization::safetensors::{load_safetensors, save_safetensors};
use std::collections::BTreeMap;
use tempfile::tempdir;

fn create_model(path: &std::path::Path, weight_diag: [f32; 4], bias: [f32; 4]) {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "layer.weight".to_string(),
        (
            vec![
                weight_diag[0],
                0.0,
                0.0,
                0.0,
                0.0,
                weight_diag[1],
                0.0,
                0.0,
                0.0,
                0.0,
                weight_diag[2],
                0.0,
                0.0,
                0.0,
                0.0,
                weight_diag[3],
            ],
            vec![4, 4],
        ),
    );
    tensors.insert("layer.bias".to_string(), (bias.to_vec(), vec![4]));
    save_safetensors(path, &tensors).expect("Failed to create model");
}

fn inspect(path: &std::path::Path, label: &str) {
    let (metadata, raw_data) = load_safetensors(path).expect("Failed to load model");
    println!("  {label}:");
    for (name, info) in &metadata {
        let [start, end] = info.data_offsets;
        let bytes = &raw_data[start..end];
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let preview: Vec<String> = floats.iter().take(8).map(|v| format!("{v:.3}")).collect();
        let suffix = if floats.len() > 8 { "..." } else { "" };
        println!(
            "    {name} {:?} = [{}{}]",
            info.shape,
            preview.join(", "),
            suffix
        );
    }
}

fn main() {
    println!("Model Merge Strategies (GH-245)");
    println!("================================\n");

    let dir = tempdir().expect("create temp dir");

    // Create test models
    let base_path = dir.path().join("base.safetensors");
    let model_a_path = dir.path().join("model_a.safetensors");
    let model_b_path = dir.path().join("model_b.safetensors");
    let model_c_path = dir.path().join("model_c.safetensors");

    create_model(&base_path, [0.0; 4], [0.0; 4]);
    create_model(&model_a_path, [1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]);
    create_model(&model_b_path, [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
    create_model(
        &model_c_path,
        [2.0, 2.0, 2.0, 2.0],
        [0.25, 0.25, 0.25, 0.25],
    );

    println!("Input models:");
    inspect(&base_path, "base (zeros)");
    inspect(&model_a_path, "model_a (diag 1,2,3,4)");
    inspect(&model_b_path, "model_b (diag 4,3,2,1)");
    inspect(&model_c_path, "model_c (diag 2,2,2,2)");

    // ── 1. Average ──────────────────────────────────────────────
    println!("\n1. Average Merge (A + B)");
    println!("   Formula: (model_a + model_b) / 2");
    let out = dir.path().join("avg.safetensors");
    let report = apr_merge(
        &[&model_a_path, &model_b_path],
        &out,
        MergeOptions::default(),
    )
    .expect("average merge");
    inspect(&out, &format!("result ({} tensors)", report.tensor_count));

    // ── 2. Weighted ─────────────────────────────────────────────
    println!("\n2. Weighted Merge (0.7*A + 0.3*B)");
    println!("   Formula: 0.7*model_a + 0.3*model_b");
    let out = dir.path().join("weighted.safetensors");
    let _report = apr_merge(
        &[&model_a_path, &model_b_path],
        &out,
        MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.7, 0.3]),
            ..Default::default()
        },
    )
    .expect("weighted merge");
    inspect(&out, "result");

    // ── 3. SLERP ────────────────────────────────────────────────
    println!("\n3. SLERP Merge (t=0.3)");
    println!("   Formula: spherical interpolation (curved, not linear)");
    let out = dir.path().join("slerp.safetensors");
    let _report = apr_merge(
        &[&model_a_path, &model_b_path],
        &out,
        MergeOptions {
            strategy: MergeStrategy::Slerp,
            weights: Some(vec![0.3]),
            ..Default::default()
        },
    )
    .expect("slerp merge");
    inspect(&out, "result");

    // ── 4. TIES ─────────────────────────────────────────────────
    println!("\n4. TIES Merge (3 task models, density=0.2)");
    println!("   Formula: base + elect_sign(trim(deltas))");
    let out = dir.path().join("ties.safetensors");
    let _report = apr_merge(
        &[&model_a_path, &model_b_path, &model_c_path],
        &out,
        MergeOptions {
            strategy: MergeStrategy::Ties,
            base_model: Some(base_path.clone()),
            density: 0.2,
            ..Default::default()
        },
    )
    .expect("ties merge");
    inspect(&out, "result");

    // ── 5. DARE ─────────────────────────────────────────────────
    println!("\n5. DARE Merge (drop_rate=0.5, seed=42)");
    println!("   Formula: base + rescale(drop(deltas))");
    let out = dir.path().join("dare.safetensors");
    let _report = apr_merge(
        &[&model_a_path, &model_b_path, &model_c_path],
        &out,
        MergeOptions {
            strategy: MergeStrategy::Dare,
            base_model: Some(base_path),
            drop_rate: 0.5,
            seed: 42,
            ..Default::default()
        },
    )
    .expect("dare merge");
    inspect(&out, "result");

    // ── CLI equivalents ─────────────────────────────────────────
    println!("\n--- CLI equivalents ---");
    println!("apr merge a.st b.st --strategy average -o merged.st");
    println!("apr merge a.st b.st --strategy weighted --weights 0.7,0.3 -o merged.st");
    println!("apr merge a.st b.st --strategy slerp --weights 0.3 -o merged.st");
    println!(
        "apr merge a.st b.st c.st --strategy ties --base-model base.st --density 0.2 -o merged.st"
    );
    println!("apr merge a.st b.st c.st --strategy dare --base-model base.st --drop-rate 0.5 --seed 42 -o merged.st");

    println!("\nAll 5 merge strategies completed successfully.");
}
