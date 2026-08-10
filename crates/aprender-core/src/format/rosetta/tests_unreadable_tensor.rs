//! Falsifier: a tensor the reader cannot decode must FAIL validation, not vanish.
//!
//! Dogfooding `cargo install aprender` 0.63.0 found that `apr validate` reported a
//! corrupt GGUF as `VALID: 338 tensors checked, 0 contract violations` with exit 0,
//! while `apr tensors` counted **339** on the same file and `inspect`, `debug`,
//! `tree` and `diff` all rejected it with
//! `Tensor 'output_norm.weight' data exceeds file size`.
//!
//! All three `validate_*` paths wrote `if let Ok(data) = reader.get_tensor(..)` and
//! silently dropped the tensor on the else branch, then reported
//! `tensor_count: tensors.len()` — the count of what SURVIVED. The gate was
//! arithmetically incapable of failing on that corruption class, because the
//! evidence was removed from the denominator before the comparison.
//!
//! This drives the real `RosettaStone::validate` over a real file on disk. The
//! SafeTensors format is used because it can be hand-written here without a
//! writer dependency; the defect and the fix are identical across GGUF, APR and
//! SafeTensors, which share `unreadable_tensor_validation`.

use super::super::*;

/// Write a SafeTensors file whose header declares two tensors: one readable and
/// entirely healthy, one whose declared extent runs a megabyte past EOF.
///
/// `good` carries real non-zero values on purpose. If it were zeros the file
/// would fail the all-zeros data-quality gate and the test would pass for the
/// wrong reason — the whole point is that the ONLY thing wrong with this file is
/// the tensor that cannot be read.
fn write_file_with_unreadable_tensor() -> std::path::PathBuf {
    use std::io::Write;

    let path = unique_temp_path("unreadable_tensor", "safetensors");
    let header = concat!(
        r#"{"good":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"#,
        r#""overrun":{"dtype":"F32","shape":[4],"data_offsets":[16,1048576]}}"#
    );
    let mut header = header.as_bytes().to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }

    let mut f = std::fs::File::create(&path).expect("create fixture");
    f.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write header len");
    f.write_all(&header).expect("write header");
    for v in [1.5f32, -2.25, 3.0, 0.75] {
        f.write_all(&v.to_le_bytes()).expect("write good tensor");
    }
    // Only 16 bytes of padding — nowhere near the 1 MiB `overrun` declares.
    f.write_all(&[0u8; 16]).expect("write padding");
    path
}

#[test]
fn a_tensor_that_cannot_be_read_fails_validation() {
    let path = write_file_with_unreadable_tensor();
    let report = RosettaStone::new().validate(&path).expect("validate ran");
    let _ = std::fs::remove_file(&path);

    assert!(
        !report.passed(),
        "a file with an undecodable tensor must not validate.\n\
         got: {}\n\
         0.63.0 reported `VALID: 1 tensors checked, 0 contract violations` here, \
         with exit 0, by dropping the unreadable tensor before counting.",
        report.summary()
    );
}

#[test]
fn the_reported_tensor_count_includes_the_unreadable_tensor() {
    let path = write_file_with_unreadable_tensor();
    let report = RosettaStone::new().validate(&path).expect("validate ran");
    let _ = std::fs::remove_file(&path);

    // This is the assertion that makes the defect unrepresentable rather than
    // merely fixed: the count must be what the FILE declares, not what happened
    // to decode. A dropped tensor cannot be invisible if the count still sees it.
    assert_eq!(
        report.tensor_count, 2,
        "header declares 2 tensors, so validate must report 2, not the number \
         that decoded. summary: {}",
        report.summary()
    );
    assert_eq!(
        report.failed_tensor_count, 1,
        "exactly one tensor is undecodable. summary: {}",
        report.summary()
    );
}

#[test]
fn the_failure_names_the_tensor_and_says_why() {
    let path = write_file_with_unreadable_tensor();
    let report = RosettaStone::new().validate(&path).expect("validate ran");
    let _ = std::fs::remove_file(&path);

    let failed: Vec<&TensorValidation> = report.tensors.iter().filter(|t| !t.is_valid).collect();
    assert_eq!(failed.len(), 1, "expected exactly one failing tensor");
    assert_eq!(
        failed[0].name, "overrun",
        "the failing tensor must be the unreadable one, not a bystander"
    );
    assert!(
        failed[0]
            .failures
            .iter()
            .any(|f| f.contains("could not be read")),
        "the failure should say the data could not be read, got {:?}",
        failed[0].failures
    );

    // And the healthy tensor must still be reported as healthy — the fix must not
    // turn one bad tensor into a blanket condemnation of the file.
    let good = report
        .tensors
        .iter()
        .find(|t| t.name == "good")
        .expect("good tensor present in the report");
    assert!(
        good.is_valid,
        "the readable, healthy tensor must still pass: {:?}",
        good.failures
    );
}
