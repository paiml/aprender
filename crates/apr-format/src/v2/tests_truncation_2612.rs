//! Falsifiers for the APR data-extent invariant (issue #2612).
//!
//! ```text
//! data_offset + max(tensor.offset + tensor.size) <= file_length
//! ```
//!
//! Measured on published `apr 0.63.0`: `head -c 50000000` of a 1,115,528,900-byte
//! `.apr` — 95.5% of the file gone — validated with **exit code 0**, while the same
//! truncation on a GGUF exited 5 with `Truncated GGUF: file is 50000000 bytes but
//! tensor data starts at 1117314624`. The APR container writes header, metadata and
//! tensor index in FRONT of the data section, so everything the reader parses
//! survives the truncation intact; nothing downstream ever compared the extent the
//! index declares against the bytes that exist.
//!
//! Each test here fails on the pre-#2612 tree because `required_file_len` did not
//! exist.

use super::{required_file_len, AprV2Metadata, AprV2Writer, V2FormatError, HEADER_SIZE_V2};

/// A small but structurally complete `.apr`: two tensors, sorted index, footer CRC.
fn known_good_apr() -> Vec<u8> {
    let metadata = AprV2Metadata::new("truncation-fixture");
    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("layer.0.weight", vec![8, 8], &[0.25_f32; 64]);
    writer.add_f32_tensor("layer.0.bias", vec![8], &[0.5_f32; 8]);
    writer
        .write()
        .expect("fixture writer must produce a valid .apr")
}

#[test]
fn required_len_of_a_complete_file_fits_inside_it() {
    let bytes = known_good_apr();
    let required = required_file_len(&bytes).expect("complete .apr must report an extent");

    assert!(
        required <= bytes.len() as u64,
        "a complete .apr must satisfy data_offset + max(offset+size) <= file_length, \
         got required={required} file_len={}",
        bytes.len()
    );
    // The invariant must not be vacuous: the extent has to actually cover the
    // tensor payload, not stop at the start of the data section.
    let header = super::AprV2Header::from_bytes(&bytes).expect("header parses");
    assert!(
        required > header.data_offset,
        "required extent {required} must include tensor bytes past data_offset {}",
        header.data_offset
    );
}

#[test]
fn truncated_file_is_shorter_than_its_own_declared_extent() {
    let bytes = known_good_apr();
    let required = required_file_len(&bytes).expect("complete .apr must report an extent");

    // 95.5% of the measured file was missing; here we cut everything after the
    // data section starts, which is the same class of damage.
    let header = super::AprV2Header::from_bytes(&bytes).expect("header parses");
    let cut = header.data_offset as usize + 16;
    assert!(
        cut < bytes.len(),
        "fixture must be longer than the cut point"
    );
    let truncated = &bytes[..cut];

    // The header, metadata and index all survive — this is why the defect was
    // invisible. Parsing still succeeds.
    let required_after =
        required_file_len(truncated).expect("truncated .apr still parses its own index");
    assert_eq!(
        required, required_after,
        "the declared extent comes from the index, which the truncation did not touch"
    );

    assert!(
        required_after > truncated.len() as u64,
        "truncation must be detectable: required={required_after} file_len={}",
        truncated.len()
    );
}

#[test]
fn non_apr_v2_buffer_is_an_error_not_a_silent_zero() {
    // A GGUF header must not be reported as a zero-extent APR (which would read
    // as "fits, therefore fine").
    let mut gguf = vec![0u8; HEADER_SIZE_V2 * 2];
    gguf[0..4].copy_from_slice(b"GGUF");
    assert!(matches!(
        required_file_len(&gguf),
        Err(V2FormatError::InvalidMagic(_))
    ));

    // Too short for a header at all.
    assert!(required_file_len(&[0u8; 8]).is_err());
}

#[test]
fn index_pointing_past_eof_is_reported_as_a_damaged_index() {
    let mut bytes = known_good_apr();
    // Point the tensor index past EOF, as a file truncated mid-index would.
    let past_eof = (bytes.len() as u64) + 4096;
    bytes[24..32].copy_from_slice(&past_eof.to_le_bytes());

    assert!(
        matches!(
            required_file_len(&bytes),
            Err(V2FormatError::InvalidTensorIndex(_))
        ),
        "an index outside the file is damage, not an unknown"
    );
}
