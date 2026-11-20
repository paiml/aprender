//! #7294 (EG-5, I-3): `apr rerank` refuses a GGUF file by name in a build WITHOUT `inference`, before it
//! reads `--vocab`. The in-crate unit tests cannot run there (the lib's test target does not compile under
//! `--no-default-features`), so this drives the built `apr` binary.
#![cfg(not(feature = "inference"))]

use std::io::Write;
use std::process::Command;

#[test]
fn rerank_refuses_gguf_magic_without_inference_before_the_vocab() {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(b"GGUF\x03\0\0\0").expect("write");
    let out = Command::new(env!("CARGO_BIN_EXE_apr"))
        .arg("rerank")
        .arg(file.path())
        .args(["--query", "q", "--passage", "p"])
        .args(["--vocab", "/nonexistent/eg5/vocab.txt"])
        .output()
        .expect("run apr");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(6), "{stderr}");
    assert!(stderr.contains("not GGUF"), "{stderr}");
    assert!(!stderr.contains("vocab"), "{stderr}");
}
