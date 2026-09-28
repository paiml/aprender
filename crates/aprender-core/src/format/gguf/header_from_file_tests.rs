//! #4520 step 2: `GgufReader::header_from_file` reads the header, never the tensor data,
//! and sizes and refuses tensors exactly as the whole-file reader does.

use super::*;
use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
use std::io::Write;

fn gguf(metadata: &[(String, GgufValue)]) -> Vec<u8> {
    let tensors = vec![
        GgufTensor {
            name: "b.weight".into(),
            shape: vec![4, 8],
            dtype: GgmlType::F32,
            data: vec![1u8; 128],
        },
        GgufTensor {
            name: "a.weight".into(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: vec![2u8; 16],
        },
    ];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, metadata).expect("write GGUF");
    bytes
}

fn arch() -> Vec<(String, GgufValue)> {
    vec![(
        "general.architecture".to_string(),
        GgufValue::String("llama".into()),
    )]
}

fn file(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
    f.write_all(bytes).expect("write");
    f
}

#[test]
fn header_matches_the_whole_file_reader() {
    let f = file(&gguf(&arch()));
    let whole = load_gguf_raw(f.path()).expect("whole-file load");
    let (reader, len) = GgufReader::header_from_file(f.path()).expect("header");
    assert_eq!(len, std::fs::metadata(f.path()).expect("stat").len());
    assert_eq!(gguf_raw_metadata(&reader), whole.raw_metadata);
    assert_eq!(reader.architecture(), whole.model_config.architecture);
    let extents = reader.tensor_extents(len).expect("extents");
    let from_header: Vec<_> = extents
        .iter()
        .map(|(n, (shape, dtype, size))| (n.clone(), shape.clone(), *dtype, *size))
        .collect();
    let from_data: Vec<_> = whole
        .tensors
        .iter()
        .map(|(n, t)| (n.clone(), t.shape.clone(), t.dtype, t.data.len()))
        .collect();
    assert_eq!(from_header, from_data);
}

fn big_vocab_gguf() -> Vec<u8> {
    let mut metadata = arch();
    let tokens: Vec<String> = (0..700_000).map(|i| format!("token-{i:08}")).collect();
    metadata.push((
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(tokens),
    ));
    gguf(&metadata)
}

/// A header past the first prefix (a big vocabulary does this) still parses: the prefix
/// doubles instead of reporting the file as malformed.
#[test]
fn a_header_longer_than_the_first_prefix_still_parses() {
    let bytes = big_vocab_gguf();
    assert!(
        bytes.len() > 1 << 20,
        "the fixture must outgrow the first prefix"
    );
    let f = file(&bytes);
    let (reader, len) =
        GgufReader::header_from_file_within(f.path(), 1 << 20, 64 << 20).expect("header");
    assert_eq!(len, bytes.len() as u64);
    assert_eq!(reader.tensor_extents(len).expect("extents").len(), 2);
    assert_eq!(
        gguf_raw_metadata(&reader)
            .get("tokenizer.ggml.tokens")
            .map(String::as_str),
        Some("[len=700000]")
    );
}

/// A header still unparsed at the cap is refused by name, never read to EOF: without the
/// cap a corrupt 17 GB GGUF buffered all 17 GB before its parse error (quorum finding on
/// PMAT-3761, lane 2).
#[test]
fn a_header_past_the_cap_is_refused_not_read_whole() {
    let bytes = big_vocab_gguf();
    let f = file(&bytes);
    let err = GgufReader::header_from_file_within(f.path(), 64 << 10, 1 << 20)
        .expect_err("the header does not fit in 1 MiB");
    assert!(
        err.to_string().contains("refused rather than read whole"),
        "{err}"
    );
}

/// A file cut inside its tensor data is refused as the whole-file reader refuses it.
#[test]
fn truncated_tensor_data_is_refused_from_the_header() {
    let bytes = gguf(&arch());
    // 64 bytes: past the end-of-file alignment padding, into the last tensor.
    let f = file(&bytes[..bytes.len() - 64]);
    let whole = load_gguf_raw(f.path()).expect_err("whole-file load refuses it");
    let (reader, len) = GgufReader::header_from_file(f.path()).expect("the header is whole");
    let header = reader
        .tensor_extents(len)
        .expect_err("header-only refuses it");
    assert_eq!(header.to_string(), whole.to_string());
    assert!(
        header.to_string().contains("data exceeds file size"),
        "{header}"
    );
}

/// Not a GGUF: the parse error is final once the whole (small) file has been read.
#[test]
fn a_non_gguf_file_is_refused() {
    let f = file(&[0x42u8; 64]);
    let err = GgufReader::header_from_file(f.path()).expect_err("not a GGUF");
    assert!(err.to_string().contains("Invalid GGUF magic"), "{err}");
}
