//! FALSIFY-INSPECT-2519: `inspect_model` must never synthesise model facts.
//!
//! It used to build its tensor list from the file's SIZE and then run
//! architecture detection over the invented shapes:
//!
//!     // For real implementation, would parse the actual file
//!     // Here we return simulated data based on file size
//!     let estimated_params = estimate_params_from_size(metadata.len(), &format);
//!     let tensors = generate_mock_tensors(estimated_params);
//!
//! Measured before the fix: 5 KB of /dev/urandom named `.safetensors` exited 0
//! and reported `Architecture llama | Hidden Dimension 768 | Layers 1 |
//! Vocab 256 | Tensors 9`. A real one-tensor safetensors file got the SAME nine
//! tensors, because the answer never depended on the contents. This crate is
//! published to crates.io, so that reached users as an "inspection".
//!
//! These tests are black box: they only need a path and an exit condition.

use std::io::Write;

fn write_bytes(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    // Per-process unique so concurrent test binaries cannot collide.
    let dir = std::env::temp_dir().join(format!("apr-inspect-2519-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).expect("create fixture");
    f.write_all(bytes).expect("write fixture");
    p
}

#[test]
fn garbage_bytes_are_not_reported_as_a_model() {
    // Deliberately NOT a valid safetensors file. The extension is the only
    // thing suggesting it is one -- which is exactly what the old code keyed on.
    let junk: Vec<u8> = (0..5120u32).map(|i| (i % 251) as u8).collect();
    let path = write_bytes("garbage.safetensors", &junk);

    let result = entrenar_inspect::inspect::inspect_model(&path);

    assert!(
        result.is_err(),
        "inspect_model returned Ok for 5 KB of non-model bytes. It is \
         fabricating model facts again -- that is the #2519 defect."
    );
}

#[test]
fn two_different_files_do_not_get_the_same_invented_answer() {
    // The sharpest form of the old bug: the answer depended on SIZE, not
    // contents, so two unrelated files of similar size got identical
    // "architectures". Whatever inspect_model does, it must not succeed here
    // with equal results -- either it errors, or it genuinely read the files.
    let a = write_bytes("a.safetensors", &vec![0xAAu8; 5120]);
    let b = write_bytes("b.safetensors", &vec![0x55u8; 5120]);

    let ra = entrenar_inspect::inspect::inspect_model(&a);
    let rb = entrenar_inspect::inspect::inspect_model(&b);

    if let (Ok(ia), Ok(ib)) = (&ra, &rb) {
        assert_ne!(
            (
                ia.architecture.hidden_dim,
                ia.architecture.num_layers,
                ia.tensors.len()
            ),
            (
                ib.architecture.hidden_dim,
                ib.architecture.num_layers,
                ib.tensors.len()
            ),
            "two different files of equal size produced identical architecture \
             and tensor count -- the answer is derived from SIZE, not contents"
        );
    }
}

/// Non-vacuity companion. Both tests above are satisfied by a function that
/// errors unconditionally, including for reasons unrelated to fabrication. This
/// pins that a MISSING file still fails for its own distinct reason, so the
/// tests above are not merely observing a function that refuses everything for
/// one blanket cause.
#[test]
fn a_missing_file_fails_for_its_own_reason() {
    let missing = std::env::temp_dir()
        .join(format!("apr-inspect-2519-{}", std::process::id()))
        .join("does-not-exist.safetensors");

    let err = entrenar_inspect::inspect::inspect_model(&missing)
        .expect_err("a missing path must be an error");
    let text = format!("{err}");

    assert!(
        text.contains("does-not-exist") || text.to_lowercase().contains("not found"),
        "a missing file should fail by NAMING the path, not with the \
         cannot-parse message. Got: {text}"
    );
}
