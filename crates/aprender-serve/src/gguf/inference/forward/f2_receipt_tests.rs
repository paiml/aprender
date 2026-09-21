//! #3604 `done_when` 2, 3 and 4, as tests that can fail.
//!
//! The decision table is pure, so every case here is a plain call — no GPU,
//! no model file, no clock. The three PLANTED-RECEIPT falsifiers (`done_when`
//! 3) are the point of this file: each plants a receipt that is perfect in two
//! keys and wrong in the third, and asserts the decision is `Validate` naming
//! THAT key. A cache that skipped on any of them would be serving a GPU path
//! that was proved for a different model, a different build, or a different
//! device.

use super::{
    apr_version, decide, model_sha256, read_receipt, receipt_path, write_receipt, F2Decision,
    F2Receipt, F2ReceiptKey, F2ValidateReason, F2_RECEIPT_SCHEMA,
};

fn key() -> F2ReceiptKey {
    F2ReceiptKey {
        model_sha256: "a".repeat(64),
        apr_version: "0.68.2".to_string(),
        device: "NVIDIA GeForce RTX 4090".to_string(),
    }
}

fn receipt_for(key: &F2ReceiptKey) -> F2Receipt {
    F2Receipt {
        schema: F2_RECEIPT_SCHEMA,
        key: key.clone(),
        validated_at: 1_758_400_000,
        positions_judged: 64,
    }
}

// ---------------------------------------------------------------- done_when 1
#[test]
fn a_receipt_whose_three_keys_all_match_is_read_and_the_forward_is_skipped() {
    let k = key();
    let r = receipt_for(&k);
    assert_eq!(
        decide(Ok(Some(r.clone())), &k, false),
        F2Decision::Skip { receipt: r }
    );
}

// ---------------------------------------------------------------- done_when 3
// THE THREE PLANTED-RECEIPT FALSIFIERS. Two keys right, one wrong, every time.

#[test]
fn falsifier_a_planted_receipt_with_the_wrong_model_sha256_revalidates() {
    let k = key();
    let mut planted = receipt_for(&k);
    planted.key.model_sha256 = "b".repeat(64); // right apr, right device, wrong model
    match decide(Ok(Some(planted)), &k, false) {
        F2Decision::Validate(F2ValidateReason::ModelSha256Mismatch { found, expected }) => {
            assert_eq!(found, "b".repeat(64));
            assert_eq!(expected, "a".repeat(64));
        },
        other => panic!("a wrong-model receipt must re-validate naming the model, got {other:?}"),
    }
}

#[test]
fn falsifier_a_planted_receipt_from_a_different_apr_version_revalidates() {
    let k = key();
    let mut planted = receipt_for(&k);
    planted.key.apr_version = "0.61.0".to_string(); // right model, right device, older build
    match decide(Ok(Some(planted)), &k, false) {
        F2Decision::Validate(F2ValidateReason::AprVersionMismatch { found, expected }) => {
            assert_eq!(found, "0.61.0");
            assert_eq!(expected, "0.68.2");
        },
        other => {
            panic!("a wrong-version receipt must re-validate naming the version, got {other:?}")
        },
    }
}

#[test]
fn falsifier_a_planted_receipt_for_a_different_device_revalidates() {
    let k = key();
    let mut planted = receipt_for(&k);
    planted.key.device = "NVIDIA GB10".to_string(); // right model, right apr, other GPU
    match decide(Ok(Some(planted)), &k, false) {
        F2Decision::Validate(F2ValidateReason::DeviceMismatch { found, expected }) => {
            assert_eq!(found, "NVIDIA GB10");
            assert_eq!(expected, "NVIDIA GeForce RTX 4090");
        },
        other => panic!("a wrong-device receipt must re-validate naming the device, got {other:?}"),
    }
}

// ---------------------------------------------------------------- done_when 4
#[test]
fn a_missing_receipt_validates_because_absence_is_never_consent() {
    assert_eq!(
        decide(Ok(None), &key(), false),
        F2Decision::Validate(F2ValidateReason::NoReceipt)
    );
}

#[test]
fn an_unreadable_receipt_validates_and_says_so_distinctly_from_missing() {
    // "It was there and I could not read it" and "it was not there" both
    // validate, but the operator must be able to tell them apart.
    let d = decide(Err("permission denied".to_string()), &key(), false);
    assert_eq!(
        d,
        F2Decision::Validate(F2ValidateReason::Unreadable(
            "permission denied".to_string()
        ))
    );
    assert_ne!(d, F2Decision::Validate(F2ValidateReason::NoReceipt));
}

#[test]
fn a_receipt_from_an_older_schema_revalidates() {
    let k = key();
    let mut old = receipt_for(&k);
    old.schema = 0;
    assert_eq!(
        decide(Ok(Some(old)), &k, false),
        F2Decision::Validate(F2ValidateReason::SchemaMismatch {
            found: 0,
            expected: F2_RECEIPT_SCHEMA
        })
    );
}

// ---------------------------------------------------------------- done_when 2
#[test]
fn revalidate_forces_a_fresh_run_even_on_a_perfect_receipt() {
    let k = key();
    assert_eq!(
        decide(Ok(Some(receipt_for(&k))), &k, true),
        F2Decision::Validate(F2ValidateReason::Revalidate)
    );
}

#[test]
fn revalidate_wins_over_every_other_reason_so_the_message_names_the_flag() {
    // With --revalidate the user does not want to hear about a stale receipt;
    // they asked for a run. The reason is the flag, whatever the file says.
    let k = key();
    for found in [
        Ok(None),
        Err("boom".to_string()),
        Ok(Some({
            let mut r = receipt_for(&k);
            r.key.device = "other".to_string();
            r
        })),
    ] {
        assert_eq!(
            decide(found, &k, true),
            F2Decision::Validate(F2ValidateReason::Revalidate)
        );
    }
}

// ------------------------------------------------------------ the file layer
#[test]
fn a_written_receipt_reads_back_equal_and_a_missing_one_is_none_not_err() {
    let dir = std::env::temp_dir().join(format!("f2-receipt-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let k = key();
    let path = receipt_path(&dir, &k.model_sha256);

    assert_eq!(
        read_receipt(&path),
        Ok(None),
        "no file must read as None, not Err"
    );

    let r = receipt_for(&k);
    write_receipt(&path, &r).expect("write");
    assert_eq!(read_receipt(&path), Ok(Some(r)));
    // No temp file of any name survives a successful write.
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .expect("dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp files left beside the receipt: {leftovers:?}"
    );

    // Corrupt it: that is Err, and it names the file.
    std::fs::write(&path, "{not json").expect("corrupt");
    let err = read_receipt(&path).expect_err("garbage must not parse");
    assert!(err.contains("not a receipt"), "{err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_receipt_file_carries_all_three_keys_flat_so_a_human_can_read_them() {
    let k = key();
    let json = serde_json::to_string(&receipt_for(&k)).expect("serialize");
    for needle in [
        "\"model_sha256\"",
        "\"apr_version\"",
        "\"device\"",
        "\"schema\"",
        "\"positions_judged\"",
    ] {
        assert!(json.contains(needle), "receipt JSON lacks {needle}: {json}");
    }
}

#[test]
fn model_sha256_is_the_real_sha256_lowercase_hex() {
    // sha256("") and sha256("abc") are the two everyone can check by hand.
    assert_eq!(
        model_sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        model_sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn apr_version_is_this_crates_version() {
    assert_eq!(apr_version(), env!("CARGO_PKG_VERSION"));
    assert!(!apr_version().is_empty());
}

#[test]
fn two_writers_on_one_model_each_rename_a_whole_file() {
    // The race lane 1 named: a shared `<sha>.json.tmp` let writer B truncate
    // what writer A was about to rename. With per-writer temp names, whichever
    // rename lands last, the receipt on disk is a COMPLETE receipt from one of
    // them — never a partial. Exercised by racing threads through the same
    // path many times and parsing the survivor every time.
    let dir = std::env::temp_dir().join(format!("f2-receipt-race-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let k = key();
    let path = receipt_path(&dir, &k.model_sha256);
    let mut a = receipt_for(&k);
    a.positions_judged = 11;
    let mut b = receipt_for(&k);
    b.positions_judged = 22;

    for _ in 0..40 {
        let (pa, pb, ra, rb) = (path.clone(), path.clone(), a.clone(), b.clone());
        let ta = std::thread::spawn(move || write_receipt(&pa, &ra));
        let tb = std::thread::spawn(move || write_receipt(&pb, &rb));
        ta.join().expect("thread a").expect("write a");
        tb.join().expect("thread b").expect("write b");
        let survivor = read_receipt(&path)
            .expect("the receipt must always parse")
            .expect("the receipt must exist");
        assert!(
            survivor == a || survivor == b,
            "the survivor is neither writer's whole receipt: {survivor:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
