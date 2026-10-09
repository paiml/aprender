//! EXT-09 (aprender#4391): eval rows land in pacha keyed on the file's sha256.

use super::*;
use pacha::model::{ModelCard, ModelVersion};
use tempfile::TempDir;

fn outcome(suite: &str) -> EvalOutcome<'_> {
    EvalOutcome {
        suite,
        suite_manifest_sha: bytes_sha256(b"items"),
        score: 7.5,
        n: 42,
    }
}

fn model_file(dir: &TempDir, bytes: &[u8]) -> PathBuf {
    let p = dir.path().join("model.gguf");
    std::fs::write(&p, bytes).unwrap();
    p
}

fn register_with_sha(home: &Path, sha: &str) {
    let reg = Registry::open(RegistryConfig::new(home)).unwrap();
    let mut card = ModelCard::new("produced");
    card.extra.insert("sha256".into(), serde_json::json!(sha));
    reg.register_model("m", &ModelVersion::new(1, 0, 0), sha.as_bytes(), card)
        .unwrap();
}

#[test]
fn ext_09_unregistered_file_is_recorded_under_its_sha256_and_says_so() {
    let (home, files) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let model = model_file(&files, b"weights");
    // A registry that holds some OTHER model: "registered" must mean this file's sha256.
    register_with_sha(home.path(), &bytes_sha256(b"another model"));
    let got = attach_to(home.path(), &model, &outcome("perplexity/wikitext2")).unwrap();
    assert_eq!(got, Attached::Unregistered);

    let sha = file_sha256(&model).unwrap();
    assert_eq!(
        sha,
        bytes_sha256(b"weights"),
        "streamed hash equals the one-shot hash"
    );
    let rows = evals_for(home.path(), &sha).unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    let r = &rows[0];
    assert_eq!(
        (r.suite.as_str(), r.score, r.n),
        ("perplexity/wikitext2", 7.5, 42)
    );
    assert_eq!(r.suite_manifest_sha, bytes_sha256(b"items"));
    assert_eq!(r.engine_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(r.engine_sha, env!("APR_GIT_SHA"));
    assert!(!r.host.is_empty(), "host is recorded");
}

#[test]
fn ext_09_registered_file_is_recognised_by_its_card_sha256() {
    let (home, files) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let model = model_file(&files, b"produced weights");
    register_with_sha(home.path(), &file_sha256(&model).unwrap());
    assert_eq!(
        attach_to(home.path(), &model, &outcome("apr-qa")).unwrap(),
        Attached::Registered
    );
}

#[test]
fn ext_09_a_non_file_is_refused_before_pacha_is_touched() {
    let (home, files) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let err = attach_to(home.path(), files.path(), &outcome("apr-qa")).unwrap_err();
    assert!(err.contains("not a file"), "{err}");
    assert!(
        !RegistryConfig::new(home.path()).db_path().exists(),
        "a refused attach created a registry"
    );
    // The command-facing wrapper turns every failure into a warning: it returns, it never panics.
    attach(files.path(), &outcome("apr-qa"));
}

#[test]
fn ext_09_reading_evals_never_creates_a_registry() {
    let home = TempDir::new().unwrap();
    assert!(evals_for(home.path(), &bytes_sha256(b"x"))
        .unwrap()
        .is_empty());
    assert!(!RegistryConfig::new(home.path()).db_path().exists());
}

fn qa_report(gates: &[(&str, bool, bool)], registered: &[&str]) -> super::super::qa::QaReport {
    let gates: Vec<_> = gates
        .iter()
        .map(|(name, passed, skipped)| {
            serde_json::json!({"name": name, "passed": passed, "message": "", "duration_ms": 1,
                               "skipped": skipped})
        })
        .collect();
    serde_json::from_value(serde_json::json!({
        "model": "m.gguf", "passed": true, "gates": gates, "gates_registered": registered,
        "total_duration_ms": 3, "timestamp": "2026-09-25T00:00:00Z", "summary": ""
    }))
    .unwrap()
}

#[test]
fn ext_09_qa_score_is_passed_over_executed_and_skips_do_not_count() {
    let reg = ["golden", "throughput", "contract", "ollama"];
    let r = qa_report(
        &[
            ("golden", true, false),
            ("throughput", false, false),
            ("contract", true, false),
            ("ollama", false, true),
        ],
        &reg,
    );
    let o = qa_outcome(&r).unwrap();
    assert_eq!(o.suite, "apr-qa");
    assert_eq!(o.n, 3, "the skipped gate is not an item");
    assert!((o.score - 2.0 / 3.0).abs() < 1e-12, "score {}", o.score);
    assert_eq!(
        o.suite_manifest_sha,
        bytes_sha256(reg.join("\n").as_bytes())
    );

    let other = qa_outcome(&qa_report(&[("golden", true, false)], &["golden"])).unwrap();
    assert_ne!(
        other.suite_manifest_sha, o.suite_manifest_sha,
        "a different gate set is a different suite"
    );
    assert!(
        qa_outcome(&qa_report(&[("golden", true, true)], &reg)).is_none(),
        "nothing executed, no score"
    );
}

#[test]
fn ext_09_runs_show_block_lists_each_eval() {
    let (home, files) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let model = model_file(&files, b"w");
    let sha = file_sha256(&model).unwrap();
    assert!(evals_text(&sha, &[]).contains("evals: none recorded"));
    attach_to(home.path(), &model, &outcome("perplexity/wikitext2")).unwrap();
    attach_to(home.path(), &model, &outcome("apr-qa")).unwrap();
    let text = evals_text(&sha, &evals_for(home.path(), &sha).unwrap());
    assert!(text.contains(&sha), "{text}");
    assert!(
        text.contains("perplexity/wikitext2") && text.contains("apr-qa"),
        "{text}"
    );
    assert!(!text.contains("none recorded"), "{text}");
}
