//! #4087 case table. Every test uses its own temp cache dir and temp model; none reads a real `~/.cache`.

// apr-cli's lib.rs allows clippy::all/pedantic, unused_* and dead_code crate-wide (APR-MONO), which would make
// `cargo clippy -D warnings` vacuous for this module. Lint levels are scoped: this module is linted for real.
#![warn(
    clippy::all,
    clippy::pedantic,
    unused_variables,
    unused_imports,
    dead_code,
    unused_assignments
)]

use super::*;
use std::cell::Cell;
use std::time::Duration;

fn verdict(msg: &str) -> GateResult {
    GateResult::passed(
        "tensor_contract",
        msg,
        Some(3.0),
        Some(0.0),
        Duration::from_millis(5),
    )
}

fn model(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).expect("write model");
    p
}

/// Runs `cached_gate` with a gate that counts its calls and returns `msg`.
fn run(model: &Path, cache: &Path, calls: &Cell<u32>, msg: &str, completed: bool) -> GateResult {
    cached_gate(model, Some(cache.to_path_buf()), || {
        calls.set(calls.get() + 1);
        (verdict(msg), completed)
    })
}

fn source(r: &GateResult) -> &str {
    r.cache.as_ref().map_or("<none>", |c| c.source.as_str())
}

#[test]
fn a_miss_runs_and_stores_then_the_same_key_is_a_hit_that_does_not_run() {
    let d = tempfile::tempdir().expect("tempdir");
    let m = model(d.path(), "m.gguf", b"model-bytes");
    let calls = Cell::new(0);
    let first = run(&m, &d.path().join("c"), &calls, "3 tensors passed", true);
    assert_eq!((source(&first), calls.get()), ("run", 1));
    let second = run(
        &m,
        &d.path().join("c"),
        &calls,
        "a DIFFERENT verdict the gate would give",
        true,
    );
    assert_eq!(
        (source(&second), calls.get()),
        ("cache", 1),
        "a hit must not run the gate"
    );
    assert_eq!(
        second.message, "3 tensors passed",
        "a hit replays the STORED verdict"
    );
    let c = second.cache.expect("provenance");
    assert_eq!(
        c.model_sha256.as_deref(),
        Some(sha256_file(&m).expect("sha").as_str())
    );
    assert!(c.checker_sha256.is_some_and(|s| s.len() == 64));
}

/// Falsifier 1/3 of #4087: the key covers the CHECKER. The same model under a different checker hash is a
/// MISS. Under the mutant that drops the checker from the entry's file name, this lookup finds the old entry,
/// so this assertion goes RED.
#[test]
fn a_checker_change_is_a_miss() {
    let d = tempfile::tempdir().expect("tempdir");
    let old = CacheKey {
        model_sha256: "m".repeat(64),
        checker_sha256: "a".repeat(64),
    };
    store(d.path(), &old, &verdict("stored by checker A")).expect("store");
    let new = CacheKey {
        checker_sha256: "b".repeat(64),
        ..old.clone()
    };
    assert!(
        matches!(lookup(d.path(), &new), Lookup::Miss),
        "a changed checker must not see checker A's verdict"
    );
    assert!(matches!(lookup(d.path(), &old), Lookup::Hit(_)));
}

/// Falsifier 2 of #4087: one changed byte of the model is a miss, and the gate runs.
#[test]
fn a_one_byte_model_change_is_a_miss() {
    let d = tempfile::tempdir().expect("tempdir");
    let m = model(d.path(), "m.gguf", b"model-bytes");
    let calls = Cell::new(0);
    let _ = run(&m, &d.path().join("c"), &calls, "v", true);
    std::fs::write(&m, b"model-bytez").expect("mutate one byte");
    let r = run(&m, &d.path().join("c"), &calls, "v", true);
    assert_eq!((source(&r), calls.get()), ("run", 2));
}

/// A corrupted entry (valid JSON, content edited) is REFUSED, the gate runs, and the entry is rewritten.
#[test]
fn an_edited_entry_is_refused_the_gate_runs_and_the_entry_is_rewritten() {
    let d = tempfile::tempdir().expect("tempdir");
    let m = model(d.path(), "m.gguf", b"model-bytes");
    let c = d.path().join("c");
    let calls = Cell::new(0);
    let _ = run(&m, &c, &calls, "FAIL 7 violations", true);
    let entry = std::fs::read_dir(&c)
        .expect("dir")
        .next()
        .expect("one entry")
        .expect("entry")
        .path();
    let text = std::fs::read_to_string(&entry).expect("read");
    std::fs::write(
        &entry,
        text.replace("FAIL 7 violations", "0 violations, all good"),
    )
    .expect("tamper");
    let r = run(&m, &c, &calls, "FAIL 7 violations", true);
    assert_eq!(
        (source(&r), calls.get()),
        ("refused", 2),
        "an edited entry must never be trusted"
    );
    assert!(r
        .cache
        .as_ref()
        .and_then(|c| c.reason.clone())
        .is_some_and(|w| w.contains("integrity")));
    assert_eq!(
        source(&run(&m, &c, &calls, "x", true)),
        "cache",
        "the rewritten entry is valid again"
    );
}

#[test]
fn a_truncated_entry_is_refused() {
    let d = tempfile::tempdir().expect("tempdir");
    let key = CacheKey {
        model_sha256: "m".repeat(64),
        checker_sha256: "a".repeat(64),
    };
    store(d.path(), &key, &verdict("v")).expect("store");
    let p = d.path().join(key.file_name());
    let t = std::fs::read_to_string(&p).expect("read");
    std::fs::write(&p, &t[..t.len() / 2]).expect("truncate");
    assert!(matches!(lookup(d.path(), &key), Lookup::Refused(w) if w.contains("parse")));
}

/// An entry copied under another key's file name is refused: the key it carries must be the key asked for.
#[test]
fn an_entry_under_the_wrong_name_is_refused() {
    let d = tempfile::tempdir().expect("tempdir");
    let a = CacheKey {
        model_sha256: "m".repeat(64),
        checker_sha256: "a".repeat(64),
    };
    let b = CacheKey {
        model_sha256: "n".repeat(64),
        ..a.clone()
    };
    store(d.path(), &a, &verdict("v")).expect("store");
    std::fs::copy(d.path().join(a.file_name()), d.path().join(b.file_name())).expect("copy");
    assert!(matches!(lookup(d.path(), &b), Lookup::Refused(w) if w.contains("key")));
}

#[test]
fn an_entry_for_another_gate_is_refused() {
    let d = tempfile::tempdir().expect("tempdir");
    let key = CacheKey {
        model_sha256: "m".repeat(64),
        checker_sha256: "a".repeat(64),
    };
    let other = GateResult::passed("golden_output", "v", None, None, Duration::from_millis(1));
    store(d.path(), &key, &other).expect("store");
    assert!(matches!(lookup(d.path(), &key), Lookup::Refused(w) if w.contains("golden_output")));
}

/// A result from a validation that did not COMPLETE (an I/O error) is never stored: it may be transient.
#[test]
fn an_error_result_is_not_stored() {
    let d = tempfile::tempdir().expect("tempdir");
    let m = model(d.path(), "m.gguf", b"model-bytes");
    let calls = Cell::new(0);
    let _ = run(
        &m,
        &d.path().join("c"),
        &calls,
        "Failed to validate: EIO",
        false,
    );
    let r = run(&m, &d.path().join("c"), &calls, "v", true);
    assert_eq!((source(&r), calls.get()), ("run", 2));
}

/// A multi-file model is bypassed (run, nothing stored): one file's hash does not cover the other parts.
#[test]
fn multi_file_models_are_bypassed_and_never_stored() {
    let d = tempfile::tempdir().expect("tempdir");
    let c = d.path().join("c");
    for name in [
        "model.safetensors.index.json",
        "Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf",
    ] {
        let m = model(d.path(), name, b"x");
        let calls = Cell::new(0);
        let r = run(&m, &c, &calls, "v", true);
        assert_eq!((source(&r), calls.get()), ("bypassed", 1), "{name}");
    }
    assert!(
        !c.exists() || std::fs::read_dir(&c).expect("dir").next().is_none(),
        "nothing stored"
    );
}

#[test]
fn split_gguf_names_case_table() {
    for (stem, want) in [
        ("m-00001-of-00003", true),
        ("Qwen3.5-122B-A10B-UD-IQ4_XS-00003-of-00003", true),
        ("m-of-00003", false),
        ("m-00001-of-", false),
        ("qwen2.5-7b-q4_k_m", false),
        ("m-0001-x-0003", false),
    ] {
        assert_eq!(is_split_gguf(stem), want, "{stem}");
    }
}

#[test]
fn with_no_cache_dir_the_gate_runs_and_reports_no_provenance() {
    let d = tempfile::tempdir().expect("tempdir");
    let m = model(d.path(), "m.gguf", b"model-bytes");
    let r = cached_gate(&m, None, || (verdict("v"), true));
    assert!(r.cache.is_none());
}
