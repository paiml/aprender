//! #7169 S1 (d): inference on `qwen4_exp` refuses by name, never produces output.
use realizar::safetensors_infer::SafetensorsToAprConverter;

#[test]
fn convert_refuses_qwen4_exp_naming_every_unsupported_op() {
    let dir = tempfile::tempdir().expect("tmp");
    // Minimal valid safetensors file: 8-byte LE header length + "{}".
    let mut bytes = 2u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    std::fs::write(dir.path().join("model.safetensors"), bytes).expect("write");
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen4_exp/config.json"
        ),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let err = SafetensorsToAprConverter::convert(&dir.path().join("model.safetensors"))
        .expect_err("must refuse");
    let msg = err.to_string();
    for n in [
        "qwen4_exp",
        "sparse-attention indexer",
        "ngram embedding",
        "ple",
        "hc residual",
        "sigmoid output gate",
        "mtp",
        "refusal, not a fallback",
    ] {
        assert!(msg.contains(n), "{n} missing from: {msg}");
    }
}
