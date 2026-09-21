//! #3733: `apr inspect --json` must show EVERY key the GGUF header carries.
//!
//! The reader's parse allowlist (`tokenizer.`, `general.`, six arch prefixes)
//! decided what inspect could show, so a Qwen3.5 file listed 24 of its 46 header
//! keys and none of the 18 `qwen35.*` ones: `qwen35.` does not start with
//! `qwen3.`. The fixture below uses that prefix, a prefix nothing has heard of,
//! and the `quantize.*` group real quantized files carry.

use super::*;
use crate::format::gguf::GGUF_MAGIC;
use crate::format::rosetta::RosettaStone;

/// Every key the fixture's header carries. The count assertions compare against
/// THIS list, which is exactly what `export_tensors_to_gguf` writes.
fn header() -> Vec<(String, GgufValue)> {
    vec![
        (
            "general.architecture".into(),
            GgufValue::String("qwen35".into()),
        ),
        (
            "general.name".into(),
            GgufValue::String("pygmy-hybrid".into()),
        ),
        ("qwen35.context_length".into(), GgufValue::Uint32(262_144)),
        ("qwen35.block_count".into(), GgufValue::Uint32(32)),
        (
            "qwen35.full_attention_interval".into(),
            GgufValue::Uint32(4),
        ),
        (
            "qwen35.attention.head_count_kv".into(),
            GgufValue::Uint32(4),
        ),
        ("qwen35.attention.key_length".into(), GgufValue::Uint32(256)),
        (
            "qwen35.rope.dimension_sections".into(),
            GgufValue::ArrayInt32(vec![11, 11, 10, 0]),
        ),
        ("qwen35.rope.freq_base".into(), GgufValue::Float32(1.0e7)),
        (
            "quantize.imatrix.file".into(),
            GgufValue::String("imatrix.gguf".into()),
        ),
        ("zzz_never_heard_of.depth".into(), GgufValue::Uint64(7)),
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        ),
    ]
}

fn fixture() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".gguf").expect("temp gguf");
    let tensors = vec![GgufTensor {
        name: "token_embd.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64],
    }];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &header()).expect("export gguf");
    std::fs::write(file.path(), &bytes).expect("write gguf");
    file
}

fn assert_every_header_key(shown: &BTreeMap<String, String>, layer: &str) {
    let header = header();
    let missing: Vec<&str> = header
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !shown.contains_key(*k))
        .collect();
    assert!(
        missing.is_empty(),
        "{layer} dropped header keys: {missing:?}"
    );
    assert_eq!(
        shown.len(),
        header.len(),
        "{layer} must show exactly the header's KV count"
    );
}

#[test]
fn raw_metadata_carries_every_header_key() {
    let file = fixture();
    let raw = load_gguf_raw(file.path()).expect("load");
    assert_every_header_key(&raw.raw_metadata, "load_gguf_raw");
    assert_eq!(raw.raw_metadata["qwen35.context_length"], "262144");
    assert_eq!(
        raw.raw_metadata["qwen35.rope.dimension_sections"],
        "[len=4]"
    );
    assert_eq!(raw.raw_metadata["zzz_never_heard_of.depth"], "7");
}

/// The path `apr inspect` takes for a GGUF (`run_rosetta_inspect`).
#[test]
fn rosetta_inspect_carries_every_header_key() {
    let file = fixture();
    let report = RosettaStone::new().inspect(file.path()).expect("inspect");
    assert_every_header_key(&report.metadata, "RosettaStone::inspect");
}

/// The keys are shown, not consumed: the config accessors still read only the
/// allowlisted map, so an architecture the loaders do not know gets no config it
/// never had (a widened accessor would hand `apr import` a qwen35 hidden size).
#[test]
fn display_only_keys_do_not_reach_the_config_accessors() {
    let file = fixture();
    let reader = GgufReader::from_file(file.path()).expect("read");
    assert!(reader
        .display_only_metadata
        .contains_key("qwen35.context_length"));
    assert!(!reader.metadata.contains_key("qwen35.context_length"));
    assert_eq!(reader.context_length(), None);
    assert_eq!(reader.num_layers(), None);

    // `keep_all` (the shard merge) still puts everything in `metadata`.
    let full = GgufReader::from_file_full(file.path()).expect("read full");
    assert!(full.display_only_metadata.is_empty());
    assert_eq!(full.metadata.len(), header().len());
}

/// Parsing (not skipping) keys outside the allowlist must not turn a hostile
/// array count into an allocation or an out-of-bounds index.
#[test]
fn an_array_count_past_the_end_of_the_file_is_an_error_not_a_panic() {
    for elem_type in [4u32, 5, 6, 8] {
        let key = "zzz.hostile";
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // one KV
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&9u32.to_le_bytes()); // array
        data.extend_from_slice(&elem_type.to_le_bytes());
        data.extend_from_slice(&(u64::MAX / 2).to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);
        let err = GgufReader::from_bytes(data).expect_err("a count past EOF must be refused");
        assert!(
            err.to_string().contains("overruns"),
            "type {elem_type}: {err}"
        );
    }
}
