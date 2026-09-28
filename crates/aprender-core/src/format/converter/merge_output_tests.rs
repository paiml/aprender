//! merge-output-fidelity-v1: FALSIFY-MOF-002 / 003 / 004 / 005.

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

/// A qwen3_5 APR fixture with one 128x128 weight stored as `dtype`.
/// Values are multiples of 1/8, exact in bf16, f16 and f32.
fn write_apr_dtype(path: &Path, dtype: TensorDType, scale: f32) -> Vec<f32> {
    let metadata = AprV2Metadata {
        architecture: Some(QWEN35.0.to_string()),
        hf_architecture: Some(QWEN35.1.to_string()),
        ..AprV2Metadata::default()
    };
    let data: Vec<f32> = (0..128 * 128).map(|i| scale * ((i % 64) as f32 - 32.0) / 8.0).collect();
    let mut writer = AprV2Writer::new(metadata);
    let name = "model.layers.0.mlp.down_proj.weight";
    match dtype {
        TensorDType::BF16 => writer.add_tensor(
            name,
            TensorDType::BF16,
            vec![128, 128],
            data.iter().flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes()).collect(),
        ),
        TensorDType::F16 => writer.add_f16_tensor(name, vec![128, 128], &data),
        _ => writer.add_f32_tensor(name, vec![128, 128], &data),
    }
    fs::write(path, writer.write().expect("encode fixture")).expect("write fixture");
    data
}

/// FALSIFY-MOF-004: bf16 inputs give a bf16 output of about the input size,
/// not an F32 file twice as large (S-R7: 4.51 GB F32 from 2.77 GB bf16).
#[test]
fn falsify_mof_004_bf16_inputs_give_bf16_output() {
    for dtype in [TensorDType::BF16, TensorDType::F16] {
        let dir = tempdir().expect("tempdir");
        let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("m.apr"));
        let one = write_apr_dtype(&a, dtype, 1.0);
        write_apr_dtype(&b, dtype, 3.0);

        apr_merge(&[a.clone(), b], out.clone(), average()).expect("merge");

        let dtypes = input_tensor_dtypes(&out).expect("read output index");
        assert_eq!(dtypes.values().copied().collect::<Vec<_>>(), vec![dtype], "{dtype:?} in, {dtypes:?} out");
        let (in_len, out_len) = (fs::metadata(&a).expect("a").len(), fs::metadata(&out).expect("out").len());
        assert!(
            (out_len as f64) < in_len as f64 * 1.02,
            "{dtype:?}: output {out_len} B vs input {in_len} B — widened?"
        );
        let merged = load_model_tensors(&out).expect("load merged");
        let two: Vec<f32> = one.iter().map(|v| 2.0 * v).collect();
        assert_eq!(merged.values().next().expect("one tensor").0, two);
    }
}

/// FALSIFY-MOF-004 "unless asked to widen": the same bf16/f16 inputs with
/// `widen` give an F32 output, values exact. Ignoring the flag turns this RED.
#[test]
fn falsify_mof_004_widen_writes_f32() {
    for dtype in [TensorDType::BF16, TensorDType::F16] {
        let dir = tempdir().expect("tempdir");
        let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("m.apr"));
        let one = write_apr_dtype(&a, dtype, 1.0);
        write_apr_dtype(&b, dtype, 3.0);
        let options = MergeOptions {
            widen: true,
            ..average()
        };

        apr_merge(&[a.clone(), b], out.clone(), options).expect("merge");

        let dtypes = input_tensor_dtypes(&out).expect("read output index");
        assert_eq!(dtypes.values().copied().collect::<Vec<_>>(), vec![TensorDType::F32], "{dtype:?} in, --widen");
        let (in_len, out_len) = (fs::metadata(&a).expect("a").len(), fs::metadata(&out).expect("out").len());
        assert!(out_len as f64 > in_len as f64 * 1.9, "{dtype:?}: output {out_len} B vs input {in_len} B — not widened?");
        let merged = load_model_tensors(&out).expect("load merged");
        let two: Vec<f32> = one.iter().map(|v| 2.0 * v).collect();
        assert_eq!(merged.values().next().expect("one tensor").0, two);
    }
}

#[test]
fn mixed_input_dtypes_widen_to_f32() {
    let dir = tempdir().expect("tempdir");
    let (a, b, out) = (dir.path().join("a.apr"), dir.path().join("b.apr"), dir.path().join("m.apr"));
    write_apr_dtype(&a, TensorDType::BF16, 1.0);
    write_apr_dtype(&b, TensorDType::F32, 3.0);
    apr_merge(&[a, b], out.clone(), average()).expect("merge");
    let dtypes = input_tensor_dtypes(&out).expect("read output index");
    assert_eq!(dtypes.values().copied().collect::<Vec<_>>(), vec![TensorDType::F32]);
}

#[test]
fn safetensors_header_dtypes_are_read() {
    let dir = tempdir().expect("tempdir");
    let p = dir.path().join("m.safetensors");
    let json = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json);
    bytes.extend_from_slice(&[0u8; 8]);
    fs::write(&p, bytes).expect("write");
    let dtypes = input_tensor_dtypes(&p).expect("read header");
    assert_eq!(dtypes.get("w"), Some(&TensorDType::BF16));
    assert_eq!(dtypes.get("b"), Some(&TensorDType::F32));
    assert_eq!(dtypes.len(), 2);
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16_bits(1.0), 0x3F80);
    assert_eq!(f32_to_bf16_bits(-2.5), 0xC020);
    // 1 + 2^-8 is halfway between bf16 1.0 and 1.0078125: ties to even (1.0).
    assert_eq!(f32_to_bf16_bits(1.0 + 1.0 / 256.0), 0x3F80);
    // 1 + 3*2^-8 is halfway above an odd mantissa: rounds up.
    assert_eq!(f32_to_bf16_bits(1.0 + 3.0 / 256.0), 0x3F82);
    assert!(f32::from_bits(u32::from(f32_to_bf16_bits(f32::NAN)) << 16).is_nan());
}
