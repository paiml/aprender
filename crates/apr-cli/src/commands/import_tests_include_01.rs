#[test]
fn t_gh267_reject_remote_bin_file() {
    // Remote .bin files should be rejected based on extension
    let result = reject_pytorch_format("hf://openai/whisper-tiny/pytorch_model.bin");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"), "Error must cite GH-267: {err}");
    assert!(
        err.contains("SafeTensors"),
        "Error must suggest conversion: {err}"
    );
}

#[test]
fn t_gh267_reject_pt_extension() {
    let result = reject_pytorch_format("model.pt");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"));
}

#[test]
fn t_gh267_allow_safetensors() {
    let result = reject_pytorch_format("model.safetensors");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_allow_gguf() {
    let result = reject_pytorch_format("model.gguf");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_allow_apr() {
    let result = reject_pytorch_format("model.apr");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_local_bin_with_pytorch_magic() {
    // Create a temp file with PyTorch ZIP magic
    let dir = tempfile::tempdir().expect("tempdir");
    let bin_path = dir.path().join("model.bin");
    std::fs::write(&bin_path, b"PK\x03\x04rest_of_zip_data").expect("write");
    let result = reject_pytorch_format(bin_path.to_str().expect("path"));
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"));
}

#[test]
fn t_gh267_local_bin_not_pytorch() {
    // A .bin file that doesn't have PyTorch magic should be allowed through
    let dir = tempfile::tempdir().expect("tempdir");
    let bin_path = dir.path().join("model.bin");
    std::fs::write(&bin_path, b"\x00\x00\x00\x08{\"__metadata__\":{}}").expect("write");
    let result = reject_pytorch_format(bin_path.to_str().expect("path"));
    assert!(result.is_ok());
}

#[test]
fn t_gh267_run_rejects_pytorch_bin() {
    let result = run(
        "pytorch_model.bin",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        false,
        false,
    );
    // Should fail with GH-267 PyTorch error (not file-not-found or other errors)
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("GH-267"),
        "Expected GH-267 PyTorch detection error, got: {err}"
    );
}
