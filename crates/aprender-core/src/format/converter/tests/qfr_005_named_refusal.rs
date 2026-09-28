//! FALSIFY-QFR-005 (contracts/qwen35-format-roundtrip-v1.yaml): `apr import`
//! of a Qwen3.5 GGUF is a named refusal. Before, the file carried QK norm, was
//! relabelled Qwen3 by tensor evidence, and died as "may be malformed" on the
//! hidden size the reader never parses for `qwen35.*` keys (#3733).

use crate::format::converter::{apr_import, Architecture, ImportOptions};
use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
use tempfile::{NamedTempFile, TempDir};

/// A pygmy GGUF with the shape a real Qwen3.5 file shows the importer:
/// `general.architecture = arch`, and a QK-norm tensor that triggers the
/// Qwen3 tensor-evidence override.
fn pygmy(arch: &str) -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".gguf").expect("temp gguf");
    let f32s = |n: usize| vec![0u8; 4 * n];
    let tensors = vec![
        GgufTensor {
            name: "token_embd.weight".into(),
            shape: vec![4, 8],
            dtype: GgmlType::F32,
            data: f32s(32),
        },
        GgufTensor {
            name: "blk.0.attn_q_norm.weight".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: f32s(4),
        },
    ];
    let header = vec![
        ("general.architecture".into(), GgufValue::String(arch.into())),
        (format!("{arch}.block_count"), GgufValue::Uint32(1)),
        (format!("{arch}.embedding_length"), GgufValue::Uint32(4)),
    ];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &header).expect("export gguf");
    std::fs::write(file.path(), &bytes).expect("write gguf");
    file
}

fn import(arch: &str, options: ImportOptions) -> (String, bool) {
    let src = pygmy(arch);
    let dir = TempDir::new().expect("tempdir");
    let out = dir.path().join("out.apr");
    let err = apr_import(src.path().to_str().expect("utf-8 path"), &out, options)
        .expect_err("pygmy import must not succeed")
        .to_string();
    (err, out.exists())
}

#[test]
fn falsify_qfr_005_qwen35_gguf_is_refused_by_name() {
    for arch in ["qwen35", "qwen35moe", "qwen3next"] {
        let (err, wrote) = import(arch, ImportOptions::default());
        assert!(
            err.contains(&format!("GGUF architecture '{arch}' is unsupported")),
            "{arch}: refusal must name the architecture, got: {err}"
        );
        assert!(!err.contains("may be malformed"), "{arch}: {err}");
        assert!(!wrote, "{arch}: a refused import must write no .apr");
    }
}

#[test]
fn falsify_qfr_005_explicit_arch_does_not_bypass_the_refusal() {
    let options = ImportOptions {
        architecture: Architecture::Qwen3,
        ..ImportOptions::default()
    };
    let (err, _) = import("qwen35", options);
    assert!(err.contains("'qwen35' is unsupported"), "--arch qwen3 must not import a qwen35 GGUF: {err}");
}

/// Control: the guard is keyed on the name, not on QK norm. A `qwen3` file
/// with the same tensors still reaches the ordinary import path.
#[test]
fn qfr_005_control_qwen3_gguf_is_not_refused_by_the_guard() {
    let (err, _) = import("qwen3", ImportOptions::default());
    assert!(!err.contains("is unsupported for import"), "qwen3 hit the qwen35 guard: {err}");
}
