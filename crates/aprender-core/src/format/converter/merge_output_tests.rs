//! merge-output-fidelity-v1: FALSIFY-MOF-002 / 003 / 005.

use super::*;
use crate::format::v2::MAGIC_V2;
use tempfile::tempdir;

fn tensors(scale: f32) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = BTreeMap::new();
    t.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        (vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale], vec![2, 2]),
    );
    t.insert(
        "model.norm.weight".to_string(),
        (vec![scale, -scale], vec![2]),
    );
    t
}

/// A tiny APR v2 fixture stamped with the given architecture metadata.
fn write_apr(path: &Path, arch: &str, hf_arch: &str, scale: f32) {
    let metadata = AprV2Metadata {
        architecture: Some(arch.to_string()),
        hf_architecture: Some(hf_arch.to_string()),
        ..AprV2Metadata::default()
    };
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors(scale) {
        writer.add_f32_tensor(name, shape, &data);
    }
    fs::write(path, writer.write().expect("encode fixture")).expect("write fixture");
}

const QWEN35: (&str, &str) = ("qwen3_5", "Qwen3_5ForConditionalGeneration");
const QWEN2: (&str, &str) = ("qwen2", "Qwen2ForCausalLM");

fn average() -> MergeOptions {
    MergeOptions {
        strategy: MergeStrategy::Average,
        ..MergeOptions::default()
    }
}

#[test]
fn falsify_mof_002_apr_output_is_an_apr_file() {
    let dir = tempdir().expect("tempdir");
    let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("merged.apr"));
    write_apr(&a, QWEN35.0, QWEN35.1, 1.0);
    write_apr(&b, QWEN35.0, QWEN35.1, 3.0);

    apr_merge(&[a, b], out.clone(), average()).expect("merge");

    let bytes = fs::read(&out).expect("read output");
    assert_eq!(bytes[0..4], MAGIC_V2, "an -o *.apr merge must write APR v2, got {:02x?}", &bytes[0..4]);
    // Values are the merge, not either input: average of 1x and 3x is 2x.
    let merged = load_model_tensors(&out).expect("load merged");
    assert_eq!(merged, tensors(2.0));
}

#[test]
fn falsify_mof_003_apr_output_keeps_input_architecture() {
    let dir = tempdir().expect("tempdir");
    let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("merged.apr"));
    write_apr(&a, QWEN35.0, QWEN35.1, 1.0);
    write_apr(&b, QWEN35.0, QWEN35.1, 3.0);

    apr_merge(&[a, b], out.clone(), average()).expect("merge");

    let meta = read_apr_metadata(&out)
        .expect("read metadata")
        .expect("output carries APR v2 metadata");
    assert_eq!(meta.architecture.as_deref(), Some(QWEN35.0));
    assert_eq!(meta.hf_architecture.as_deref(), Some(QWEN35.1));
    assert_eq!(
        meta.custom.get("merge_strategy").and_then(|v| v.as_str()),
        Some("Average")
    );
}

#[test]
fn falsify_mof_005_mismatched_architectures_are_refused() {
    let dir = tempdir().expect("tempdir");
    let (a, b, out) = (dir.path().join("qwen35.apr"), dir.path().join("qwen2.apr"), dir.path().join("merged.apr"));
    // Identical tensor names and shapes: only the metadata tells them apart.
    write_apr(&a, QWEN35.0, QWEN35.1, 1.0);
    write_apr(&b, QWEN2.0, QWEN2.1, 1.0);

    let err = apr_merge(&[a, b], out.clone(), average())
        .expect_err("different architectures must not merge")
        .to_string();
    assert!(err.contains(QWEN35.0) && err.contains(QWEN2.0), "must name both: {err}");
    assert!(!out.exists(), "a refused merge must not write an output");
}

#[test]
fn falsify_mof_005_base_model_architecture_is_checked() {
    let dir = tempdir().expect("tempdir");
    let (a, b, base, out) = (
        dir.path().join("a.apr"),
        dir.path().join("b.apr"),
        dir.path().join("base.apr"),
        dir.path().join("merged.apr"),
    );
    write_apr(&a, QWEN35.0, QWEN35.1, 1.0);
    write_apr(&b, QWEN35.0, QWEN35.1, 2.0);
    write_apr(&base, QWEN2.0, QWEN2.1, 0.5);

    let options = MergeOptions {
        strategy: MergeStrategy::Ties,
        base_model: Some(base),
        ..MergeOptions::default()
    };
    let err = apr_merge(&[a, b], out, options)
        .expect_err("a base of another architecture must not merge")
        .to_string();
    assert!(err.contains(QWEN2.0), "must name the base architecture: {err}");
}

#[test]
fn safetensors_output_stays_safetensors() {
    let dir = tempdir().expect("tempdir");
    let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("merged.safetensors"));
    write_apr(&a, QWEN35.0, QWEN35.1, 1.0);
    write_apr(&b, QWEN35.0, QWEN35.1, 3.0);

    apr_merge(&[a, b], out.clone(), average()).expect("merge");

    assert!(read_apr_metadata(&out).expect("probe").is_none());
    assert_eq!(load_model_tensors(&out).expect("load"), tensors(2.0));
}

#[test]
fn non_apr_file_has_no_apr_metadata() {
    let dir = tempdir().expect("tempdir");
    let short = dir.path().join("short.apr");
    fs::write(&short, b"APR").expect("write");
    assert!(read_apr_metadata(&short).expect("probe").is_none());
}
