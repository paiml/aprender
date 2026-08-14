//! End-to-end tests for `apr shard` / `apr unshard`.
//!
//! Covers CRUX-B-05 falsifiers — split→merge round-trip identity, weight-map
//! coverage, total_size invariant — plus parser unit tests.

use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use tempfile::TempDir;

use super::sharder::{shard_safetensors_file, ShardReport};
use super::unsharder::{unshard_safetensors_dir, UnshardReport};

fn write_safetensors(
    dir: &Path,
    name: &str,
    tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)],
) -> std::path::PathBuf {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(n, dt, shape, bytes)| {
            (
                *n,
                TensorView::new(*dt, shape.clone(), bytes).expect("TensorView"),
            )
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize");
    let path = dir.join(name);
    fs::write(&path, bytes).expect("write");
    path
}

fn fake_f32_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 1e-9;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// FALSIFY-CRUX-B-05-001 — weight_map covers exactly the input tensors,
/// every referenced shard file is present, total_size > 0.
#[test]
fn falsify_crux_b_05_001_weight_map_covers_all_tensors() {
    let tmp = TempDir::new().expect("tempdir");
    let mut spec = Vec::new();
    for i in 0..20 {
        spec.push((
            // SafeTensors crate refuses tensor names with '.' followed by digits
            // alone, but `t_0`, `t_1`, ... are valid. Match the contract intent.
            Box::leak(format!("t_{i}").into_boxed_str()) as &str,
            Dtype::F32,
            vec![32usize, 32],
            fake_f32_bytes(i as u32, 32 * 32),
        ));
    }
    let input = write_safetensors(tmp.path(), "model.safetensors", &spec);

    let out_dir = tmp.path().join("out");
    let ShardReport {
        index_path,
        shard_files,
        tensor_count,
        total_size,
    } = shard_safetensors_file(&input, 8 * 1024, &out_dir).expect("shard");

    assert!(index_path.is_file(), "index.json must exist");
    assert_eq!(tensor_count, 20);
    assert!(total_size > 0);
    assert!(
        shard_files.len() >= 2,
        "small limit should produce multiple shards"
    );

    let idx_text = fs::read_to_string(&index_path).expect("read index");
    // Spot-check the JSON shape — every tensor name appears as a key.
    for (name, _, _, _) in &spec {
        assert!(
            idx_text.contains(&format!("\"{name}\"")),
            "tensor {name} missing from weight_map"
        );
    }
    for shard in &shard_files {
        assert!(shard.is_file(), "shard file {} must exist", shard.display());
        let fname = shard.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            idx_text.contains(&format!("\"{fname}\"")),
            "weight_map references {fname} which is on disk"
        );
    }
}

/// FALSIFY-CRUX-B-05-002 — split(M) then merge produces tensors that are
/// byte-identical to the originals, including dtype and shape.
#[test]
fn falsify_crux_b_05_002_split_then_merge_identity() {
    let tmp = TempDir::new().expect("tempdir");
    let mut spec: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for i in 0..30 {
        spec.push((
            Box::leak(format!("layer_{i}").into_boxed_str()) as &str,
            Dtype::F32,
            vec![16usize, 16],
            fake_f32_bytes(i as u32 + 100, 16 * 16),
        ));
    }
    let input = write_safetensors(tmp.path(), "model.safetensors", &spec);

    let sharded = tmp.path().join("sharded");
    shard_safetensors_file(&input, 2 * 1024, &sharded).expect("shard");

    let rebuilt = tmp.path().join("rebuilt.safetensors");
    let UnshardReport { tensor_count, .. } =
        unshard_safetensors_dir(&sharded, &rebuilt).expect("unshard");
    assert_eq!(tensor_count, 30);

    // Compare tensor-by-tensor.
    let orig_bytes = fs::read(&input).expect("read orig");
    let reb_bytes = fs::read(&rebuilt).expect("read rebuilt");
    let orig = SafeTensors::deserialize(&orig_bytes).expect("deserialize orig");
    let reb = SafeTensors::deserialize(&reb_bytes).expect("deserialize reb");

    // safetensors 0.7: `names()` yields `&str` directly (it yielded `&String` in 0.4).
    let orig_names: std::collections::HashSet<&str> = orig.names().into_iter().collect();
    let reb_names: std::collections::HashSet<&str> = reb.names().into_iter().collect();
    assert_eq!(orig_names, reb_names, "tensor set must match");

    for name in orig.names() {
        let a = orig.tensor(name).expect("orig tensor");
        let b = reb.tensor(name).expect("reb tensor");
        assert_eq!(a.dtype(), b.dtype(), "dtype mismatch for {name}");
        assert_eq!(a.shape(), b.shape(), "shape mismatch for {name}");
        assert_eq!(a.data(), b.data(), "bytes mismatch for {name}");
    }
}

/// FALSIFY-CRUX-B-05-003 — `metadata.total_size` equals the sum of
/// `element_size × numel` across all tensors.
#[test]
fn falsify_crux_b_05_003_total_size_equals_sum_of_bytes() {
    let tmp = TempDir::new().expect("tempdir");

    // Mix of f32 (4B/elem) and f16 (2B/elem) to exercise dtype_size.
    let f32_bytes = fake_f32_bytes(7, 100 * 100); // 40_000 bytes
    let f16_bytes: Vec<u8> = (0..(50 * 50))
        .flat_map(|i| (i as u16).to_le_bytes())
        .collect(); // 5_000 bytes
    let spec = vec![
        ("a", Dtype::F32, vec![100usize, 100], f32_bytes),
        ("b", Dtype::F16, vec![50usize, 50], f16_bytes),
    ];
    let expected_total: u64 = 100 * 100 * 4 + 50 * 50 * 2;
    let input = write_safetensors(tmp.path(), "model.safetensors", &spec);

    let out_dir = tmp.path().join("out");
    let report = shard_safetensors_file(&input, 1024, &out_dir).expect("shard");
    assert_eq!(report.total_size, expected_total);

    let idx_text =
        fs::read_to_string(&out_dir.join("model.safetensors.index.json")).expect("read index");
    assert!(
        idx_text.contains(&format!("\"total_size\": {expected_total}")),
        "index.json must declare total_size = {expected_total}"
    );
}

/// Edge case — single tensor larger than the shard limit lands in its own shard.
#[test]
fn oversized_single_tensor_alone() {
    let tmp = TempDir::new().expect("tempdir");
    let spec = vec![
        (
            "small1",
            Dtype::F32,
            vec![16usize, 16],
            fake_f32_bytes(0, 256),
        ),
        (
            "big",
            Dtype::F32,
            vec![512usize, 512],
            fake_f32_bytes(1, 512 * 512),
        ),
        (
            "small2",
            Dtype::F32,
            vec![16usize, 16],
            fake_f32_bytes(2, 256),
        ),
    ];
    let input = write_safetensors(tmp.path(), "model.safetensors", &spec);

    let out_dir = tmp.path().join("out");
    let report = shard_safetensors_file(&input, 4 * 1024, &out_dir).expect("shard");
    assert!(report.shard_files.len() >= 2);

    let rebuilt = tmp.path().join("rebuilt.safetensors");
    unshard_safetensors_dir(&out_dir, &rebuilt).expect("unshard");

    let orig = fs::read(&input).unwrap();
    let reb = fs::read(&rebuilt).unwrap();
    let o = SafeTensors::deserialize(&orig).unwrap();
    let r = SafeTensors::deserialize(&reb).unwrap();
    for name in o.names() {
        assert_eq!(
            o.tensor(name).unwrap().data(),
            r.tensor(name).unwrap().data(),
            "tensor {name} corrupted on round-trip"
        );
    }
}

/// Round-trip identity holds for a small input with a huge shard size limit
/// (everything in one shard).
#[test]
fn single_shard_roundtrip() {
    let tmp = TempDir::new().expect("tempdir");
    let spec = vec![
        (
            "alpha",
            Dtype::F32,
            vec![10usize, 10],
            fake_f32_bytes(3, 100),
        ),
        (
            "beta",
            Dtype::F32,
            vec![10usize, 10],
            fake_f32_bytes(4, 100),
        ),
    ];
    let input = write_safetensors(tmp.path(), "model.safetensors", &spec);

    let out_dir = tmp.path().join("out");
    let report = shard_safetensors_file(&input, 100 * 1024 * 1024, &out_dir).expect("shard");
    assert_eq!(report.shard_files.len(), 1);

    let rebuilt = tmp.path().join("rebuilt.safetensors");
    unshard_safetensors_dir(&out_dir, &rebuilt).expect("unshard");

    let o = fs::read(&input).unwrap();
    let r = fs::read(&rebuilt).unwrap();
    let ost = SafeTensors::deserialize(&o).unwrap();
    let rst = SafeTensors::deserialize(&r).unwrap();
    for name in ost.names() {
        assert_eq!(
            ost.tensor(name).unwrap().data(),
            rst.tensor(name).unwrap().data()
        );
    }
}

/// Missing index.json → unshard returns a descriptive error.
#[test]
fn unshard_rejects_missing_index() {
    let tmp = TempDir::new().expect("tempdir");
    let err = unshard_safetensors_dir(tmp.path(), &tmp.path().join("out.safetensors"))
        .expect_err("must fail without index.json");
    let msg = format!("{err}");
    assert!(msg.contains("model.safetensors.index.json"), "error: {msg}");
}

/// Empty input → shard returns a descriptive error rather than producing
/// a 0-shard index.json that an unsharder cannot reconstruct.
#[test]
fn shard_rejects_empty_input() {
    let tmp = TempDir::new().expect("tempdir");
    let path = write_safetensors(tmp.path(), "empty.safetensors", &[]);
    let out_dir = tmp.path().join("out");
    let err =
        shard_safetensors_file(&path, 1024, &out_dir).expect_err("empty input should be rejected");
    assert!(format!("{err}").contains("no tensors"));
}
