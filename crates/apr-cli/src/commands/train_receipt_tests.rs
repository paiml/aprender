//! Falsifiers for contract train-run-receipt-v1 (TRR-001..006) on the shared
//! writer and on `apr pretrain`, the first verb wired to it.

use super::*;
use tempfile::TempDir;

/// Minimal SafeTensors file: 8-byte LE header length, JSON header, f32 data.
fn write_safetensors(path: &Path, tensors: &[(&str, &[usize], &[f32])]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            (*name).to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let h = serde_json::Value::Object(header).to_string();
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(h.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).expect("write safetensors");
}

/// FALSIFY-TRR-002: the receipt's git sha is the one `apr --version` prints.
#[test]
fn falsify_trr_002_git_sha_is_the_version_sha() {
    let tmp = TempDir::new().expect("tempdir");
    let out = tmp.path().join("out.bin");
    std::fs::write(&out, b"weights").expect("write");
    let cfg = serde_json::json!({"lr": 1e-4});
    let r = TrainReceipt::build(&RunFacts {
        verb: "finetune",
        effective_config: &cfg,
        data_files: &[],
        seed: 7,
        base_models: &[],
        backend: "cpu",
        device: "cpu",
        started_at: now_utc(),
        steps: 1,
        final_loss: Some(1.0),
        output: &out,
    })
    .expect("build");
    let version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")");
    assert!(!r.apr_git_sha.is_empty(), "FALSIFY-TRR-002: empty git sha");
    assert_eq!(
        format!("{} ({})", r.apr_version, r.apr_git_sha),
        version,
        "FALSIFY-TRR-002: receipt sha differs from the --version sha"
    );
}

/// FALSIFY-TRR-003: reordering the data or flipping one byte changes data_sha256.
#[test]
fn falsify_trr_003_reorder_or_byte_flip_changes_data_hash() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.jsonl");
    let b = tmp.path().join("b.jsonl");
    std::fs::write(&a, b"{\"x\":1}\n").expect("write a");
    std::fs::write(&b, b"{\"x\":2}\n").expect("write b");
    let ab = data_sha256(&[a.clone(), b.clone()]).expect("hash");
    let ba = data_sha256(&[b.clone(), a.clone()]).expect("hash");
    assert_ne!(
        ab, ba,
        "FALSIFY-TRR-003: a reorder left data_sha256 unchanged"
    );

    std::fs::write(&a, b"{\"x\":3}\n").expect("flip a");
    let flipped = data_sha256(&[a, b]).expect("hash");
    assert_ne!(
        ab, flipped,
        "FALSIFY-TRR-003: a byte flip left data_sha256 unchanged"
    );
}

/// FALSIFY-TRR-005: a one-tensor mutant changes base_sha256; renaming the
/// model FILE does not (the hash is over tensors, not paths or bytes on disk).
#[test]
fn falsify_trr_005_tensor_mutant_changes_base_hash_rename_does_not() {
    let tmp = TempDir::new().expect("tempdir");
    let base = tmp.path().join("base.safetensors");
    let renamed = tmp.path().join("renamed.safetensors");
    let mutant = tmp.path().join("mutant.safetensors");
    write_safetensors(
        &base,
        &[
            ("w.a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            ("w.b", &[2], &[5.0, 6.0]),
        ],
    );
    // Same tensors, header written in the other order: still the same model.
    write_safetensors(
        &renamed,
        &[
            ("w.b", &[2], &[5.0, 6.0]),
            ("w.a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        ],
    );
    write_safetensors(
        &mutant,
        &[
            ("w.a", &[2, 2], &[1.0, 2.0, 3.0, 4.5]),
            ("w.b", &[2], &[5.0, 6.0]),
        ],
    );

    let h_base = model_canonical_sha256(&base).expect("hash base");
    assert_eq!(
        h_base,
        model_canonical_sha256(&renamed).expect("hash renamed"),
        "FALSIFY-TRR-005: a file rename / header reorder changed base_sha256"
    );
    assert_ne!(
        h_base,
        model_canonical_sha256(&mutant).expect("hash mutant"),
        "FALSIFY-TRR-005: a one-tensor mutant left base_sha256 unchanged"
    );
}

/// F-CKPT-017 layout: shape is part of the hash, so a reshape is a new model.
#[test]
fn canonical_tensor_hash_binds_shape_and_name() {
    let d = [0u8; 16];
    let h = canonical_tensor_hash([("w", DTYPE_F32, &[2usize, 2][..], &d[..])]);
    assert_ne!(
        h,
        canonical_tensor_hash([("w", DTYPE_F32, &[4usize][..], &d[..])])
    );
    assert_ne!(
        h,
        canonical_tensor_hash([("v", DTYPE_F32, &[2usize, 2][..], &d[..])])
    );
}

/// FALSIFY-TRR-006: spelling a default out gives the same recipe hash (the
/// effective config is identical, whatever key order it serializes in);
/// changing a value gives a different one.
#[test]
fn falsify_trr_006_explicit_default_same_hash_changed_value_differs() {
    let implicit =
        serde_json::json!({"lr": 1e-4, "opt": {"beta1": 0.9, "beta2": 0.999}, "seed": 42});
    let explicit: serde_json::Value =
        serde_json::from_str(r#"{"seed":42,"opt":{"beta2":0.999,"beta1":0.9},"lr":0.0001}"#)
            .expect("json");
    assert_eq!(
        recipe_sha256(&implicit),
        recipe_sha256(&explicit),
        "FALSIFY-TRR-006: the same effective config hashed differently"
    );
    let changed =
        serde_json::json!({"lr": 2e-4, "opt": {"beta1": 0.9, "beta2": 0.999}, "seed": 42});
    assert_ne!(
        recipe_sha256(&implicit),
        recipe_sha256(&changed),
        "FALSIFY-TRR-006: a changed value left recipe_sha256 unchanged"
    );
}

/// The receipt must not hash itself: writing it leaves output_sha256 stable.
#[test]
fn output_hash_of_a_directory_excludes_the_receipt() {
    let tmp = TempDir::new().expect("tempdir");
    std::fs::write(tmp.path().join("ckpt.bin"), b"abc").expect("write");
    let before = output_sha256(tmp.path()).expect("hash");
    std::fs::write(tmp.path().join(RECEIPT_FILE), b"{}").expect("write receipt");
    assert_eq!(before, output_sha256(tmp.path()).expect("hash"));
    std::fs::write(tmp.path().join("ckpt.bin"), b"abd").expect("mutate");
    assert_ne!(before, output_sha256(tmp.path()).expect("hash"));
}

fn run_synthetic_pretrain(run_dir: &Path, target_val_loss: f32) -> Result<()> {
    let tmp_in = run_dir.parent().expect("parent");
    super::super::pretrain::run(
        &tmp_in.join("data.jsonl"),
        &tmp_in.join("tok"),
        run_dir,
        crate::commands::pretrain::PretrainMode::Finetune,
        Some(5.0e-5),
        25,
        Some(5),
        2,
        4,
        5,
        1234,
        Some(target_val_loss),
        50257,
        true,
        "cpu",
        None,
        false,
        None,
        true,
    )
}

/// FALSIFY-TRR-001 (pretrain): a successful run writes a complete receipt
/// whose seed is the seed passed in.
#[test]
fn falsify_trr_001_pretrain_writes_complete_receipt() {
    let tmp = TempDir::new().expect("tempdir");
    let run_dir = tmp.path().join("run");
    run_synthetic_pretrain(&run_dir, 2.2).expect("synthetic pretrain succeeds");
    let path = run_dir.join(RECEIPT_FILE);
    let text = std::fs::read_to_string(&path).expect("FALSIFY-TRR-001: no train_receipt.json");
    let v: serde_json::Value = serde_json::from_str(&text).expect("receipt is JSON");
    for field in [
        "apr_version",
        "apr_git_sha",
        "recipe_sha256",
        "data_sha256",
        "seed",
        "base_sha256",
        "backend",
        "device",
        "started_at",
        "ended_at",
        "steps",
        "final_loss",
        "output_sha256",
    ] {
        assert!(
            v.get(field).is_some(),
            "FALSIFY-TRR-001: receipt lacks {field}"
        );
    }
    assert_eq!(v["verb"], "pretrain");
    assert_eq!(
        v["seed"], 1234,
        "FALSIFY-TRR-001: receipt seed is not the seed used"
    );
    assert!(v["steps"].as_u64().is_some_and(|s| s > 0));
    assert_eq!(v["recipe_sha256"].as_str().map(str::len), Some(64));
    assert_eq!(
        v["output_sha256"].as_str(),
        Some(output_sha256(&run_dir).expect("hash").as_str()),
        "FALSIFY-TRR-001: output_sha256 does not match the run directory"
    );
}
