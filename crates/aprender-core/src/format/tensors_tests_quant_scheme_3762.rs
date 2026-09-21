//! #3762: a file's `quantization` is the scheme it declares, else its dominant `>= 2`-D tensor
//! dtype by count, never the dtype holding the most bytes; and `inspect` reads every ggml type
//! `tensors` reads.

use super::*;
use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
use crate::format::rosetta::RosettaStone;
use std::collections::BTreeMap;

fn histogram(rows: &[(&str, &[usize])]) -> BTreeMap<String, usize> {
    dtype_histogram(rows.iter().copied())
}

#[test]
fn the_histogram_counts_tensors_of_two_or_more_dimensions_by_count() {
    let h = histogram(&[
        ("q4_k", &[256, 2]),
        ("Q4_K", &[256, 4]),
        ("F32", &[4096]),
        ("q6_k", &[256, 1000]),
    ]);
    let want: BTreeMap<String, usize> = [("Q4_K".to_string(), 2), ("Q6_K".to_string(), 1)].into();
    assert_eq!(
        h, want,
        "1-D tensors do not vote; names are upper-cased; size is irrelevant"
    );
    assert_eq!(dominant_dtypes(&h), vec!["Q4_K".to_string()]);

    let tie = histogram(&[("Q4_K", &[2, 2]), ("Q6_K", &[2, 2])]);
    assert_eq!(
        dominant_dtypes(&tie),
        vec!["Q4_K".to_string(), "Q6_K".to_string()]
    );
    assert!(dominant_dtypes(&BTreeMap::new()).is_empty());
}

#[test]
fn a_declared_scheme_wins_and_otherwise_the_dominant_dtype_is_named_as_such() {
    let h = histogram(&[("Q4_K", &[2, 2]), ("Q4_K", &[2, 2]), ("Q6_K", &[9, 9])]);
    assert_eq!(
        QuantScheme::resolve(Some(("Q4_K_M", "general.file_type")), &h),
        Some(QuantScheme {
            name: "Q4_K_M".to_string(),
            source: "general.file_type"
        })
    );
    assert_eq!(
        QuantScheme::resolve(None, &h),
        Some(QuantScheme {
            name: "Q4_K".to_string(),
            source: QuantScheme::DOMINANT
        })
    );
    let tie = histogram(&[("Q4_K", &[2, 2]), ("Q6_K", &[2, 2])]);
    assert_eq!(
        QuantScheme::resolve(None, &tie).map(|s| s.name),
        Some("Q4_K+Q6_K".to_string())
    );
    assert_eq!(QuantScheme::resolve(None, &BTreeMap::new()), None);
}

#[test]
fn general_file_type_is_decoded_by_the_upstream_table() {
    let meta = |v: &str| BTreeMap::from([("general.file_type".to_string(), v.to_string())]);
    assert_eq!(gguf_declared_scheme(&meta("15")), Some("Q4_K_M"));
    assert_eq!(gguf_declared_scheme(&meta("30")), Some("IQ4_XS"));
    assert_eq!(gguf_declared_scheme(&meta("4")), None, "removed upstream");
    assert_eq!(
        gguf_declared_scheme(&meta("1024")),
        None,
        "LLAMA_FTYPE_GUESSED is no scheme"
    );
    assert_eq!(gguf_declared_scheme(&meta("fifteen")), None);
    assert_eq!(gguf_declared_scheme(&BTreeMap::new()), None);
}

/// A GGUF of `tensors`, each a 2-D `[block, rows]` tensor of zeros, with `file_type` as
/// `general.file_type` when given; written to a temp file for `RosettaStone::inspect`.
fn gguf_file(
    dir: &std::path::Path,
    name: &str,
    tensors: &[(&str, GgmlType, usize)],
    file_type: Option<u32>,
) -> std::path::PathBuf {
    let tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|&(tensor_name, dtype, rows)| {
            let block = dtype.block_size();
            GgufTensor {
                name: tensor_name.to_string(),
                shape: vec![block as u64, rows as u64],
                dtype,
                data: vec![0u8; dtype.tensor_bytes(block * rows)],
            }
        })
        .collect();
    let mut metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("llama".to_string()),
    )];
    if let Some(ft) = file_type {
        metadata.push(("general.file_type".to_string(), GgufValue::Uint32(ft)));
    }
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &metadata).expect("export GGUF");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write GGUF");
    path
}

/// The measured defect: a Q4_K_M file whose one Q6_K tensor (the embedding) holds more
/// parameters than all its Q4_K matrices was reported as Q6_K.
#[test]
fn a_q4_k_m_gguf_is_q4_k_m_and_never_its_largest_tensors_dtype() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tensors = [
        ("token_embd.weight", GgmlType::Q6K, 64),
        ("blk.0.attn_q.weight", GgmlType::Q4K, 2),
        ("blk.0.ffn_up.weight", GgmlType::Q4K, 2),
    ];
    let declared = gguf_file(dir.path(), "declared.gguf", &tensors, Some(15));
    let report = RosettaStone::new().inspect(&declared).expect("inspect");
    assert_eq!(
        report.quantization.as_deref(),
        Some("Q4_K_M"),
        "general.file_type 15"
    );

    let undeclared = gguf_file(dir.path(), "undeclared.gguf", &tensors, None);
    let report = RosettaStone::new().inspect(&undeclared).expect("inspect");
    assert_eq!(
        report.quantization.as_deref(),
        Some("Q4_K"),
        "no declared scheme: the dominant >= 2-D dtype by count (2 x Q4_K vs 1 x Q6_K), \
         not the dtype holding the most parameters"
    );
}

/// done_when 3: `inspect` succeeds on every ggml type `tensors` reads. The list is upstream's
/// (`trueno_quant::ALL`, extracted), never typed here; a type added upstream joins this test.
#[test]
fn inspect_and_tensors_read_every_live_ggml_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut failures = Vec::new();
    for dtype in trueno_quant::ALL {
        let name = format!("{}.gguf", dtype.as_str());
        let path = gguf_file(dir.path(), &name, &[("blk.0.w.weight", dtype, 2)], None);
        let bytes = std::fs::read(&path).expect("read");
        match list_tensors_from_bytes(&bytes, TensorListOptions::default()) {
            Ok(listed)
                if listed.tensors.first().map(|t| t.dtype.as_str()) == Some(dtype.as_str()) => {}
            Ok(listed) => failures.push(format!(
                "tensors: {} listed as {:?}",
                dtype.as_str(),
                listed.tensors.first().map(|t| &t.dtype)
            )),
            Err(e) => failures.push(format!("tensors: {}: {e}", dtype.as_str())),
        }
        match RosettaStone::new().inspect(&path) {
            Ok(report) if report.quantization.as_deref() == Some(dtype.as_str()) => {}
            Ok(report) => failures.push(format!(
                "inspect: {} reported quantization {:?}",
                dtype.as_str(),
                report.quantization
            )),
            Err(e) => failures.push(format!("inspect: {}: {e}", dtype.as_str())),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} ggml types:\n{}",
        failures.len(),
        trueno_quant::ALL.len(),
        failures.join("\n")
    );
}
