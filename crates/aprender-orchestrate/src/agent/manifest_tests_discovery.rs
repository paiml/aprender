//! Integration tests for model discovery with real temp files (PMAT-184).
//!
//! Tests create APR/GGUF files with proper magic bytes and controlled
//! mtimes, then validate sort_candidates, is_valid_model_file, and
//! the full discovery pipeline.

use super::*;
use crate::agent::driver::validate::is_valid_model_file;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// APR v2 magic bytes
const APR_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
/// GGUF magic bytes
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

fn write_valid_apr(path: &std::path::Path) {
    let mut data = Vec::new();
    data.extend_from_slice(&APR_MAGIC);
    data.extend_from_slice(br#"{"tokenizer.merges":["a b"],"tokenizer.vocabulary":["hi"]}"#);
    std::fs::write(path, &data).expect("write APR");
}

fn write_invalid_apr(path: &std::path::Path) {
    let mut data = Vec::new();
    data.extend_from_slice(&APR_MAGIC);
    // No tokenizer data — just architecture metadata
    data.extend_from_slice(br#"{"architecture":"qwen2","vocab_size":151936}"#);
    std::fs::write(path, &data).expect("write invalid APR");
}

fn write_valid_gguf(path: &std::path::Path) {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC);
    data.extend_from_slice(&[3, 0, 0, 0]); // version 3
    data.extend_from_slice(&[0u8; 100]);
    std::fs::write(path, &data).expect("write GGUF");
}

fn set_mtime(path: &std::path::Path, time: SystemTime) {
    let ft = filetime::FileTime::from_system_time(time);
    filetime::set_file_mtime(path, ft).expect("set mtime");
}

// ═══ PMAT-184: Model discovery integration tests ═══

/// FALSIFY-DISC-101: Real APR file with tokenizer passes is_valid_model_file.
#[test]
fn falsify_disc_101_valid_apr_passes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.apr");
    write_valid_apr(&path);
    assert!(is_valid_model_file(&path), "FALSIFY-DISC-101: valid APR with tokenizer must pass");
}

/// FALSIFY-DISC-102: Real APR without tokenizer fails is_valid_model_file.
#[test]
fn falsify_disc_102_invalid_apr_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.apr");
    write_invalid_apr(&path);
    assert!(!is_valid_model_file(&path), "FALSIFY-DISC-102: APR without tokenizer must fail");
}

/// FALSIFY-DISC-103: Real GGUF file with valid magic passes.
#[test]
fn falsify_disc_103_valid_gguf_passes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_valid_gguf(&path);
    assert!(is_valid_model_file(&path), "FALSIFY-DISC-103: valid GGUF must pass");
}

/// FALSIFY-DISC-104: sort_candidates with real files — newer GGUF beats older APR.
#[test]
fn falsify_disc_104_mtime_beats_format_real_files() {
    let dir = tempfile::tempdir().unwrap();
    let apr_path = dir.path().join("old.apr");
    let gguf_path = dir.path().join("new.gguf");

    write_valid_apr(&apr_path);
    write_valid_gguf(&gguf_path);

    let now = SystemTime::now();
    let yesterday = now - Duration::from_secs(86400);
    set_mtime(&apr_path, yesterday);
    set_mtime(&gguf_path, now);

    let mut candidates = vec![
        (apr_path.clone(), yesterday, true, is_valid_model_file(&apr_path)),
        (gguf_path.clone(), now, false, is_valid_model_file(&gguf_path)),
    ];
    ModelConfig::sort_candidates(&mut candidates);

    assert_eq!(
        candidates[0].0.file_name().unwrap().to_str().unwrap(),
        "new.gguf",
        "FALSIFY-DISC-104: newer GGUF must beat older APR"
    );
}

/// FALSIFY-DISC-105: sort_candidates — APR wins at same mtime with real files.
#[test]
fn falsify_disc_105_apr_tiebreak_real_files() {
    let dir = tempfile::tempdir().unwrap();
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    write_valid_apr(&apr_path);
    write_valid_gguf(&gguf_path);

    let now = SystemTime::now();
    set_mtime(&apr_path, now);
    set_mtime(&gguf_path, now);

    let mut candidates =
        vec![(gguf_path.clone(), now, false, true), (apr_path.clone(), now, true, true)];
    ModelConfig::sort_candidates(&mut candidates);

    assert_eq!(
        candidates[0].0.file_name().unwrap().to_str().unwrap(),
        "model.apr",
        "FALSIFY-DISC-105: APR preferred at same mtime"
    );
}

/// FALSIFY-DISC-106: Jidoka — invalid APR deprioritized behind valid GGUF with real files.
#[test]
fn falsify_disc_106_jidoka_real_files() {
    let dir = tempfile::tempdir().unwrap();
    let apr_path = dir.path().join("broken.apr");
    let gguf_path = dir.path().join("valid.gguf");

    write_invalid_apr(&apr_path); // APR without tokenizer
    write_valid_gguf(&gguf_path);

    let now = SystemTime::now();
    let yesterday = now - Duration::from_secs(86400);
    // APR is NEWER but INVALID
    set_mtime(&apr_path, now);
    set_mtime(&gguf_path, yesterday);

    let mut candidates = vec![
        (apr_path.clone(), now, true, is_valid_model_file(&apr_path)),
        (gguf_path.clone(), yesterday, false, is_valid_model_file(&gguf_path)),
    ];
    ModelConfig::sort_candidates(&mut candidates);

    assert_eq!(
        candidates[0].0.file_name().unwrap().to_str().unwrap(),
        "valid.gguf",
        "FALSIFY-DISC-106: valid GGUF must beat NEWER invalid APR (Jidoka)"
    );
    assert!(!candidates[1].3, "FALSIFY-DISC-106: broken APR marked invalid");
}

/// FALSIFY-DISC-107: Four-way sort — valid+newest wins across mixed formats.
#[test]
fn falsify_disc_107_four_way_sort() {
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now();

    // 4 candidates: valid APR (old), valid GGUF (new), invalid APR (newest), valid GGUF (oldest)
    let files: Vec<(&str, bool, SystemTime)> = vec![
        ("valid_old.apr", true, now - Duration::from_secs(7200)),
        ("valid_new.gguf", true, now - Duration::from_secs(60)),
        ("broken_newest.apr", false, now),
        ("valid_oldest.gguf", true, now - Duration::from_secs(86400)),
    ];

    let mut candidates: Vec<(PathBuf, SystemTime, bool, bool)> = files
        .iter()
        .map(|(name, make_valid, mtime)| {
            let path = dir.path().join(name);
            if name.ends_with(".apr") {
                if *make_valid {
                    write_valid_apr(&path);
                } else {
                    write_invalid_apr(&path);
                }
            } else {
                write_valid_gguf(&path);
            }
            set_mtime(&path, *mtime);
            let is_apr = name.ends_with(".apr");
            let is_valid = is_valid_model_file(&path);
            (path, *mtime, is_apr, is_valid)
        })
        .collect();

    ModelConfig::sort_candidates(&mut candidates);

    // Winner: valid_new.gguf (valid + newest among valid)
    assert_eq!(
        candidates[0].0.file_name().unwrap().to_str().unwrap(),
        "valid_new.gguf",
        "FALSIFY-DISC-107: valid+newest wins"
    );
    // Loser: broken_newest.apr (invalid, sorted last)
    assert_eq!(
        candidates[3].0.file_name().unwrap().to_str().unwrap(),
        "broken_newest.apr",
        "FALSIFY-DISC-107: invalid APR sorted last"
    );
}

/// FALSIFY-DISC-108: is_valid_model_file handles nonexistent path gracefully.
#[test]
fn falsify_disc_108_nonexistent_path() {
    assert!(
        !is_valid_model_file(std::path::Path::new("/nonexistent/model.apr")),
        "FALSIFY-DISC-108: nonexistent path must return false"
    );
}

/// FALSIFY-DISC-109: Extension detection — .apr is APR, .gguf is GGUF, both formats supported.
#[test]
fn falsify_disc_109_format_detection() {
    let dir = tempfile::tempdir().unwrap();
    let apr = dir.path().join("model.apr");
    let gguf = dir.path().join("model.gguf");
    write_valid_apr(&apr);
    write_valid_gguf(&gguf);

    // Both formats accepted
    assert!(is_valid_model_file(&apr), "FALSIFY-DISC-109: APR accepted");
    assert!(is_valid_model_file(&gguf), "FALSIFY-DISC-109: GGUF accepted");

    // Extension determines validation path
    let ext_apr = apr.extension().unwrap().to_str().unwrap();
    let ext_gguf = gguf.extension().unwrap().to_str().unwrap();
    assert_eq!(ext_apr, "apr", "FALSIFY-DISC-109: APR extension");
    assert_eq!(ext_gguf, "gguf", "FALSIFY-DISC-109: GGUF extension");
}
