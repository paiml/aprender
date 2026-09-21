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
    apr_version, build_label, cached_sha256, decide, exe_sha256, file_identity, file_sha256,
    model_sha256, read_receipt, receipt_path, write_receipt, F2Decision, F2Receipt, F2ReceiptKey,
    F2ValidateReason, HashSource, F2_RECEIPT_SCHEMA,
};

fn key() -> F2ReceiptKey {
    F2ReceiptKey {
        model_sha256: "a".repeat(64),
        exe_sha256: "e".repeat(64),
        device: "NVIDIA GeForce RTX 4090".to_string(),
    }
}

fn receipt_for(key: &F2ReceiptKey) -> F2Receipt {
    F2Receipt {
        schema: F2_RECEIPT_SCHEMA,
        key: key.clone(),
        build: "apr 0.69.0 (5615e7afe)".to_string(),
        exe_path: "/usr/local/bin/apr".to_string(),
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
    planted.key.model_sha256 = "b".repeat(64); // right exe, right device, wrong model
    match decide(Ok(Some(planted)), &k, false) {
        F2Decision::Validate(F2ValidateReason::ModelSha256Mismatch { found, expected }) => {
            assert_eq!(found, "b".repeat(64));
            assert_eq!(expected, "a".repeat(64));
        },
        other => panic!("a wrong-model receipt must re-validate naming the model, got {other:?}"),
    }
}

#[test]
fn falsifier_a_planted_receipt_from_a_different_executable_revalidates() {
    let k = key();
    let mut planted = receipt_for(&k);
    planted.key.exe_sha256 = "f".repeat(64); // right model, right device, other binary
    match decide(Ok(Some(planted)), &k, false) {
        F2Decision::Validate(F2ValidateReason::ExeSha256Mismatch { found, expected }) => {
            assert_eq!(found, "f".repeat(64));
            assert_eq!(expected, "e".repeat(64));
        },
        other => {
            panic!("another executable's receipt must re-validate naming the exe, got {other:?}")
        },
    }
}

/// #3748 MUTANT "key on the version": two DIFFERENT executables of the SAME
/// apr version (two 0.69.0 builds — a clean one and a dirty tree, or a cuda
/// and a cuda-batch build) must never share a receipt. The build label they
/// record may be identical; it is not the key.
#[test]
fn falsifier_the_same_version_from_another_executable_never_skips() {
    let k = key();
    let mut other_build = receipt_for(&k);
    other_build.key.exe_sha256 = "d".repeat(64);
    assert_eq!(
        other_build.build,
        receipt_for(&k).build,
        "same version label"
    );
    assert!(
        matches!(
            decide(Ok(Some(other_build)), &k, false),
            F2Decision::Validate(F2ValidateReason::ExeSha256Mismatch { .. })
        ),
        "a receipt from another binary of the same version must re-validate"
    );
}

#[test]
fn falsifier_a_planted_receipt_for_a_different_device_revalidates() {
    let k = key();
    let mut planted = receipt_for(&k);
    planted.key.device = "NVIDIA GB10".to_string(); // right model, right exe, other GPU
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
    let path = receipt_path(&dir, &k);

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
        "\"exe_sha256\"",
        "\"device\"",
        "\"build\"",
        "\"exe_path\"",
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
    let path = receipt_path(&dir, &k);
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

// ------------------------------------------------------------------- #3748
/// A scratch dir unique to one test.
fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("f2-3748-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// #3748 done_when 1 — builds A, B, A on one model: B's receipt must not
/// overwrite A's, so A's second run skips. MUTANT "key on the model only"
/// (one file per model) turns this RED: B's write replaces A's, and A reads a
/// receipt from another executable.
#[test]
fn builds_a_then_b_then_a_never_overwrite_each_other() {
    let dir = scratch("aba");
    let a = key();
    let mut b = key();
    b.exe_sha256 = "b".repeat(64);
    let (pa, pb) = (receipt_path(&dir, &a), receipt_path(&dir, &b));
    assert_ne!(pa, pb, "two executables on one model must be two files");

    write_receipt(&pa, &receipt_for(&a)).expect("A validates and writes");
    write_receipt(&pb, &receipt_for(&b)).expect("B validates and writes");
    let again = read_receipt(&receipt_path(&dir, &a));
    assert!(
        matches!(decide(again, &a, false), F2Decision::Skip { .. }),
        "A's second run must read A's receipt and skip"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The file name carries the device too: one model and one executable on two
/// GPUs are two receipts.
#[test]
fn a_second_device_is_a_second_receipt_file() {
    let dir = std::path::Path::new("/r");
    let a = key();
    let mut gx = key();
    gx.device = "NVIDIA GB10".to_string();
    assert_ne!(receipt_path(dir, &a), receipt_path(dir, &gx));
    assert!(receipt_path(dir, &gx)
        .to_string_lossy()
        .ends_with("-nvidia-gb10.json"));
}

/// Hash through the cache, counting the reads.
fn hash_counting(
    dir: &std::path::Path,
    f: &std::path::Path,
    reads: &mut u32,
) -> (String, HashSource) {
    cached_sha256(Some(dir), f, || {
        *reads += 1;
        file_sha256(f)
    })
    .expect("hash")
}

/// #3748 done_when 2: a warm run re-reads nothing; a changed file is re-read.
/// Each of the four identity fields is changed ALONE, and each must miss —
/// MUTANT "drop mtime" (a rebuilt binary of the same size in place) and
/// MUTANT "drop inode" (replaced by rename with size and mtime restored) each
/// turn one row RED.
#[test]
fn the_hash_cache_hits_warm_and_misses_on_any_identity_change() {
    let dir = scratch("cache");
    let f = dir.join("model.bin");
    std::fs::write(&f, b"weights v1").expect("write");
    let mut reads = 0;

    let (h1, s1) = hash_counting(&dir, &f, &mut reads);
    assert_eq!((s1, reads), (HashSource::Hashed, 1), "cold: hashed");
    assert_eq!(h1, model_sha256(b"weights v1"));
    let (h2, s2) = hash_counting(&dir, &f, &mut reads);
    assert_eq!(
        (h2.as_str(), s2, reads),
        (h1.as_str(), HashSource::Cache, 1),
        "warm: nothing read"
    );

    // size alone
    std::fs::write(&f, b"weights v1 plus").expect("grow");
    let (h3, s3) = hash_counting(&dir, &f, &mut reads);
    assert_eq!(
        (s3, reads),
        (HashSource::Hashed, 2),
        "a size change must miss"
    );
    assert_eq!(h3, model_sha256(b"weights v1 plus"));

    // mtime alone: same size, same inode (an in-place rebuild)
    let before = file_identity(&f).expect("identity");
    {
        use std::io::Write as _;
        let mut w = std::fs::OpenOptions::new()
            .write(true)
            .open(&f)
            .expect("open");
        w.write_all(b"WEIGHTS").expect("overwrite in place");
        w.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .expect("mtime");
    }
    let after = file_identity(&f).expect("identity");
    assert_eq!((before.size, before.inode), (after.size, after.inode));
    assert_ne!(before.mtime_ns, after.mtime_ns);
    let (h4, s4) = hash_counting(&dir, &f, &mut reads);
    assert_eq!(
        (s4, reads),
        (HashSource::Hashed, 3),
        "an mtime change must miss"
    );
    assert_eq!(h4, model_sha256(b"WEIGHTS v1 plus"));

    // inode alone: replace by rename, size and mtime restored
    let before = file_identity(&f).expect("identity");
    let tmp = dir.join("model.bin.new");
    std::fs::write(&tmp, b"weights v2 plus").expect("new content, same size");
    std::fs::File::options()
        .write(true)
        .open(&tmp)
        .expect("open")
        .set_modified(
            std::time::UNIX_EPOCH
                + std::time::Duration::from_nanos(u64::try_from(before.mtime_ns).expect("mtime")),
        )
        .expect("restore mtime");
    std::fs::rename(&tmp, &f).expect("replace");
    let after = file_identity(&f).expect("identity");
    assert_eq!((before.size, before.mtime_ns), (after.size, after.mtime_ns));
    assert_ne!(before.inode, after.inode);
    let (h5, s5) = hash_counting(&dir, &f, &mut reads);
    assert_eq!(
        (s5, reads),
        (HashSource::Hashed, 4),
        "an inode change must miss"
    );
    assert_eq!(h5, model_sha256(b"weights v2 plus"));

    // path alone: the same bytes under another name are their own entry
    let g = dir.join("copy.bin");
    std::fs::copy(&f, &g).expect("copy");
    let (h6, s6) = hash_counting(&dir, &g, &mut reads);
    assert_eq!(
        (h6, s6, reads),
        (h5, HashSource::Hashed, 5),
        "another path must miss"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file that changes WHILE it is hashed is never cached: the next run
/// hashes again instead of trusting a hash of bytes that no longer exist.
#[test]
fn a_file_rewritten_mid_hash_is_not_cached() {
    let dir = scratch("midhash");
    let f = dir.join("model.bin");
    std::fs::write(&f, b"one").expect("write");
    let (_, s) = cached_sha256(Some(&dir), &f, || {
        std::fs::write(&f, b"three").expect("rewrite during the hash");
        Ok(model_sha256(b"one"))
    })
    .expect("hash");
    assert_eq!(s, HashSource::Hashed);
    let mut reads = 0;
    let (h, s) = hash_counting(&dir, &f, &mut reads);
    assert_eq!(
        (s, reads),
        (HashSource::Hashed, 1),
        "the mid-hash result must not be served"
    );
    assert_eq!(h, model_sha256(b"three"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// No cache dir: always hashed, never an error.
#[test]
fn without_a_cache_dir_every_run_hashes() {
    let dir = scratch("nodir");
    let f = dir.join("m");
    std::fs::write(&f, b"x").expect("write");
    for _ in 0..2 {
        let (h, s) = cached_sha256(None, &f, || file_sha256(&f)).expect("hash");
        assert_eq!((h, s), (model_sha256(b"x"), HashSource::Hashed));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The key is the RUNNING executable: `exe_sha256` equals a plain hash of
/// `current_exe()`, and a warm call reads it from the cache.
#[test]
fn exe_sha256_is_the_running_executables_hash_and_caches() {
    let dir = scratch("exe");
    let exe = std::env::current_exe().expect("current exe");
    let (h1, _, p) = exe_sha256(Some(&dir)).expect("hash self");
    assert_eq!(p, exe);
    assert_eq!(h1, file_sha256(&exe).expect("plain hash"));
    let (h2, s2, _) = exe_sha256(Some(&dir)).expect("hash self again");
    assert_eq!((h2, s2), (h1, HashSource::Cache));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The build label is for humans and defaults to the version when the binary
/// set none; it is not part of [`F2ReceiptKey`].
#[test]
fn the_build_label_names_the_version() {
    assert!(
        build_label().contains(env!("CARGO_PKG_VERSION")),
        "{}",
        build_label()
    );
}
