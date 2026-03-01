#![allow(clippy::disallowed_methods)]
//! Dogfood example: APR Checkpoint Lifecycle
//!
//! Demonstrates the full checkpoint save → load → filter → validate cycle
//! per the APR Checkpoint Specification v1.3.0.
//!
//! Exercises contracts:
//! - F-CKPT-002: Schema version present
//! - F-CKPT-003: Adapter has no __training__.* tensors
//! - F-CKPT-007: NaN/Inf rejected on checked read
//! - F-CKPT-009: Atomic writes (tmp+fsync+rename)
//! - F-CKPT-013: Post-load NaN scan
//! - F-CKPT-014: Shape-config validation
//! - F-CKPT-015: Canonical tensor ordering
//! - F-CKPT-016: Filtered reader skips training state
//! - F-CKPT-017: Provenance metadata present
//! - F-CKPT-018: Round-trip bit-identical

use aprender::serialization::apr::{AprReader, AprWriter};
use serde_json::json;

fn main() {
    println!("=== APR Checkpoint Lifecycle Dogfood ===\n");

    let dir = std::env::temp_dir().join("apr_checkpoint_lifecycle");
    let _ = std::fs::create_dir_all(&dir);

    // ── Step 1: Create a training checkpoint (.ckpt.apr) ─────────────
    println!("1. Creating training checkpoint...");
    let mut writer = AprWriter::new();

    // F-CKPT-002: schema version
    writer.set_metadata("__checkpoint__.schema_version", json!("1.3.0"));
    writer.set_metadata("model_type", json!("adapter"));
    writer.set_metadata("architecture", json!("qwen2_classify"));
    writer.set_metadata("num_classes", json!(5));
    writer.set_metadata("lora_rank", json!(8));
    writer.set_metadata("epoch", json!(10));
    writer.set_metadata("val_loss", json!(0.4231));

    // F-CKPT-017: provenance
    writer.set_metadata("data_hash", json!("sha256:abc123..."));
    writer.set_metadata("base_model_source", json!("hf://Qwen/Qwen2.5-Coder-0.5B"));
    writer.set_metadata("provenance", json!({
        "tool": "entrenar v0.7.5",
        "started_at": "2026-03-01T10:00:00Z",
    }));

    // Model tensors
    writer.add_tensor_f32("classifier.weight", vec![5, 896], &vec![0.01; 5 * 896]);
    writer.add_tensor_f32("classifier.bias", vec![5], &[0.1, 0.2, 0.3, 0.4, 0.5]);
    writer.add_tensor_f32("lora.0.q_proj.lora_a", vec![8, 896], &vec![0.001; 8 * 896]);
    writer.add_tensor_f32("lora.0.q_proj.lora_b", vec![896, 8], &vec![0.0; 896 * 8]);

    // Training state (should be filtered out by inference readers)
    writer.add_tensor_f32("__training__.optimizer.step", vec![1], &[500.0]);
    writer.add_tensor_f32("__training__.optimizer.m.0", vec![5 * 896], &vec![0.01; 5 * 896]);
    writer.add_tensor_f32("__training__.optimizer.v.0", vec![5 * 896], &vec![0.001; 5 * 896]);
    writer.add_tensor_f32("__training__.epoch", vec![1], &[10.0]);
    writer.add_tensor_f32("__training__.learning_rate", vec![1], &[0.00005]);

    // F-CKPT-009: atomic write
    let ckpt_path = dir.join("model.apr");
    writer.write(&ckpt_path).expect("write failed");
    let file_size = std::fs::metadata(&ckpt_path).unwrap().len();
    println!("   Written: {} ({} bytes)", ckpt_path.display(), file_size);

    // Verify no .tmp orphan
    assert!(!dir.join("model.apr.tmp").exists(), "F-CKPT-009: orphan .tmp!");
    println!("   F-CKPT-009: No orphan .tmp file ✓");

    // ── Step 2: Full read (training resume) ──────────────────────────
    println!("\n2. Loading full checkpoint (for resume)...");
    let full = AprReader::open(&ckpt_path).expect("open failed");
    println!("   Tensors: {} total", full.tensors.len());
    let training_count = full.tensors.iter().filter(|t| t.name.starts_with("__training__.")).count();
    let model_count = full.tensors.len() - training_count;
    println!("   Model tensors: {}, Training tensors: {}", model_count, training_count);
    assert_eq!(model_count, 4, "Should have 4 model tensors");
    assert_eq!(training_count, 5, "Should have 5 training tensors");

    // F-CKPT-002: verify schema version
    let schema = full.get_metadata("__checkpoint__.schema_version")
        .and_then(|v| v.as_str())
        .unwrap_or("MISSING");
    println!("   F-CKPT-002: schema_version = {schema} ✓");
    assert_eq!(schema, "1.3.0");

    // F-CKPT-015: canonical ordering
    let names: Vec<&str> = full.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut sorted_names = names.clone();
    sorted_names.sort();
    assert_eq!(names, sorted_names, "F-CKPT-015: not sorted!");
    println!("   F-CKPT-015: Canonical ordering ✓");

    // F-CKPT-017: provenance
    assert!(full.get_metadata("data_hash").is_some(), "missing data_hash");
    assert!(full.get_metadata("provenance").is_some(), "missing provenance");
    println!("   F-CKPT-017: Provenance present ✓");

    // F-CKPT-013: checked read (NaN scan)
    let w = full.read_tensor_f32_checked("classifier.weight").expect("NaN in weight!");
    println!("   F-CKPT-013: classifier.weight clean ({} values) ✓", w.len());

    // F-CKPT-014: shape validation
    full.validate_tensor_shape("classifier.weight", 5 * 896).expect("shape mismatch!");
    full.validate_tensor_shape("classifier.bias", 5).expect("bias shape mismatch!");
    println!("   F-CKPT-014: Shape validation ✓");

    // ── Step 3: Filtered read (inference deployment) ─────────────────
    println!("\n3. Loading for inference (filtered)...");
    let inference = AprReader::open_filtered(&ckpt_path, |name| {
        !name.starts_with("__training__.")
    }).expect("filtered open failed");
    println!("   Tensors: {} (training state excluded)", inference.tensors.len());
    assert_eq!(inference.tensors.len(), 4, "F-CKPT-016: training tensors leaked!");
    println!("   F-CKPT-016: Filtered reader ✓");

    // ── Step 4: Adapter-only file (F-CKPT-003) ──────────────────────
    println!("\n4. Creating adapter-only APR...");
    let mut adapter_writer = AprWriter::new();
    adapter_writer.set_metadata("__checkpoint__.schema_version", json!("1.3.0"));
    adapter_writer.set_metadata("model_type", json!("adapter"));
    // Only model tensors, NO __training__.*
    for t in &inference.tensors {
        let data = inference.read_tensor_f32(&t.name).unwrap();
        adapter_writer.add_tensor_f32(t.name.clone(), t.shape.clone(), &data);
    }
    let adapter_path = dir.join("model.adapter.apr");
    adapter_writer.write(&adapter_path).expect("adapter write failed");

    let adapter = AprReader::open(&adapter_path).expect("adapter read failed");
    let has_training = adapter.tensors.iter().any(|t| t.name.starts_with("__training__."));
    assert!(!has_training, "F-CKPT-003: adapter contains training tensors!");
    println!("   F-CKPT-003: Zero __training__.* tensors ✓");

    // ── Step 5: Round-trip (F-CKPT-018) ──────────────────────────────
    println!("\n5. Verifying round-trip...");
    let original_bytes = adapter_writer.to_bytes().unwrap();
    let rt_reader = AprReader::from_bytes(original_bytes.clone()).unwrap();
    let mut rt_writer = AprWriter::new();
    for (k, v) in &rt_reader.metadata {
        rt_writer.set_metadata(k.clone(), v.clone());
    }
    for t in &rt_reader.tensors {
        let data = rt_reader.read_tensor_f32(&t.name).unwrap();
        rt_writer.add_tensor_f32(t.name.clone(), t.shape.clone(), &data);
    }
    let rt_bytes = rt_writer.to_bytes().unwrap();
    assert_eq!(original_bytes, rt_bytes, "F-CKPT-018: round-trip not bit-identical!");
    println!("   F-CKPT-018: Round-trip bit-identical ✓");

    // ── Cleanup ──────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);

    println!("\n=== All checkpoint contracts verified ✓ ===");
}
