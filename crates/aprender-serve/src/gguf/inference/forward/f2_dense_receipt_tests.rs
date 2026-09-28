//! #3602: the dense receipt's precision table, as tests that can fail.
//!
//! Each row plants a receipt that matches in model and build and differs, or
//! not, in the precision tag, and asserts what the run does. The row that
//! matters most is `fp8_rejected_receipt_forces_fp16`: without it a cached run
//! would serve FP8 on a model whose guard rejected FP8.

use super::super::f2_receipt::{F2Receipt, F2ReceiptKey, F2ValidateReason, F2_RECEIPT_SCHEMA};
use super::{decide_dense, dense_device_key, DenseDecision, DensePrecision};

const SHA: &str = "1d9614638d18024d1d9614638d18024d1d9614638d18024d1d9614638d18024d";
const APR: &str = "0.70.0 exe:0123456789abcdef";
const DEV: &str = "NVIDIA GeForce RTX 4090";

fn planted(precision: DensePrecision) -> Result<Option<F2Receipt>, String> {
    Ok(Some(F2Receipt {
        schema: F2_RECEIPT_SCHEMA,
        key: F2ReceiptKey {
            model_sha256: SHA.to_string(),
            apr_version: APR.to_string(),
            device: dense_device_key(DEV, precision),
        },
        validated_at: 1_758_900_000,
        positions_judged: 30,
    }))
}

fn run(found: Result<Option<F2Receipt>, String>, fp8_on: bool) -> DenseDecision {
    decide_dense(found, SHA, APR, DEV, fp8_on, false)
}

fn skip(d: &DenseDecision) -> Option<bool> {
    match d {
        DenseDecision::Skip { force_fp16, .. } => Some(*force_fp16),
        DenseDecision::Validate(_) => None,
    }
}

#[test]
fn precision_after_validation() {
    assert_eq!(
        DensePrecision::after_validation(true, true),
        DensePrecision::Fp8
    );
    assert_eq!(
        DensePrecision::after_validation(true, false),
        DensePrecision::Fp16Fp8Rejected
    );
    assert_eq!(
        DensePrecision::after_validation(false, false),
        DensePrecision::Fp16
    );
}

#[test]
fn device_keys_are_distinct_per_precision() {
    let k = [
        dense_device_key(DEV, DensePrecision::Fp8),
        dense_device_key(DEV, DensePrecision::Fp16),
        dense_device_key(DEV, DensePrecision::Fp16Fp8Rejected),
    ];
    assert_ne!(k[0], k[1]);
    assert_ne!(k[0], k[2]);
    assert_ne!(k[1], k[2]);
    assert!(k.iter().all(|s| s.starts_with(DEV)));
}

#[test]
fn exact_match_skips_without_forcing() {
    assert_eq!(skip(&run(planted(DensePrecision::Fp8), true)), Some(false));
    assert_eq!(
        skip(&run(planted(DensePrecision::Fp16), false)),
        Some(false)
    );
}

#[test]
fn fp8_rejected_receipt_forces_fp16() {
    assert_eq!(
        skip(&run(planted(DensePrecision::Fp16Fp8Rejected), true)),
        Some(true)
    );
}

#[test]
fn fp8_rejected_receipt_serves_fp16_run_without_forcing() {
    assert_eq!(
        skip(&run(planted(DensePrecision::Fp16Fp8Rejected), false)),
        Some(false)
    );
}

// A run under FP8_PREFILL=0 proved nothing about FP8: it must not switch the
// model's default precision off through the cache.
#[test]
fn plain_fp16_receipt_does_not_vouch_for_fp8() {
    let d = run(planted(DensePrecision::Fp16), true);
    assert!(
        matches!(
            d,
            DenseDecision::Validate(F2ValidateReason::DeviceMismatch { .. })
        ),
        "{d:?}"
    );
}

// And an FP8 receipt says nothing about the FP16 kernels.
#[test]
fn fp8_receipt_does_not_vouch_for_fp16() {
    assert!(skip(&run(planted(DensePrecision::Fp8), false)).is_none());
}

#[test]
fn a_different_device_validates_whatever_its_tag() {
    for p in [
        DensePrecision::Fp8,
        DensePrecision::Fp16,
        DensePrecision::Fp16Fp8Rejected,
    ] {
        for fp8_on in [true, false] {
            let d = decide_dense(planted(p), SHA, APR, "NVIDIA GB10", fp8_on, false);
            assert!(skip(&d).is_none(), "{p:?} fp8_on={fp8_on}: {d:?}");
        }
    }
}

#[test]
fn model_and_build_mismatch_validate_even_with_the_rejected_tag() {
    let other = "b".repeat(64);
    let d = decide_dense(
        planted(DensePrecision::Fp16Fp8Rejected),
        &other,
        APR,
        DEV,
        true,
        false,
    );
    assert!(
        matches!(
            d,
            DenseDecision::Validate(F2ValidateReason::ModelSha256Mismatch { .. })
        ),
        "{d:?}"
    );
    let d = decide_dense(
        planted(DensePrecision::Fp16Fp8Rejected),
        SHA,
        "0.70.0 exe:ffffffffffffffff",
        DEV,
        true,
        false,
    );
    assert!(
        matches!(
            d,
            DenseDecision::Validate(F2ValidateReason::AprVersionMismatch { .. })
        ),
        "{d:?}"
    );
}

#[test]
fn revalidate_and_absence_always_validate() {
    for p in [DensePrecision::Fp8, DensePrecision::Fp16Fp8Rejected] {
        let d = decide_dense(planted(p), SHA, APR, DEV, true, true);
        assert_eq!(d, DenseDecision::Validate(F2ValidateReason::Revalidate));
    }
    assert_eq!(
        run(Ok(None), true),
        DenseDecision::Validate(F2ValidateReason::NoReceipt)
    );
    assert!(matches!(
        run(Err("garbage".into()), true),
        DenseDecision::Validate(F2ValidateReason::Unreadable(_))
    ));
}

#[test]
fn streamed_file_hash_equals_the_in_memory_hash() {
    use super::super::f2_receipt::{model_sha256, model_sha256_file};
    let dir = std::env::temp_dir().join(format!("f2-dense-sha-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let p = dir.join("m.gguf");
    // Longer than one 1 MiB read, and not a multiple of it.
    let bytes: Vec<u8> = (0..(3 << 20) + 17)
        .map(|i: usize| (i * 31 % 251) as u8)
        .collect();
    std::fs::write(&p, &bytes).expect("write");
    assert_eq!(model_sha256_file(&p).expect("hash"), model_sha256(&bytes));
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_dir(&dir);
}
