//! CLI Integration Tests (EXTREME TDD - RED Phase)
//!
//! These tests define the expected behavior BEFORE implementation.

// APR-MONO §S #1976: workspace `.clippy.toml` disallowed-methods now applies; allow
// unwrap() in this integration-test crate per aprender-core convention.
#![allow(clippy::disallowed_methods)]

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Helper to get CLI command
fn cli() -> Command {
    Command::cargo_bin("aprender-rag").unwrap()
}

// ============================================================================
// INFO COMMAND TESTS
// ============================================================================

#[test]
fn test_info_shows_version() {
    cli()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Trueno-RAG Pipeline"))
        .stdout(predicate::str::contains("Version:"));
}

#[test]
fn test_info_shows_components() {
    cli()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Chunkers:"))
        .stdout(predicate::str::contains("Embedders:"))
        .stdout(predicate::str::contains("TF-IDF"));
}

// ============================================================================
// DEMO COMMAND TESTS
// ============================================================================

#[test]
fn test_demo_default_query() {
    cli()
        .arg("demo")
        .assert()
        .success()
        .stdout(predicate::str::contains("Trueno-RAG Demo"))
        .stdout(predicate::str::contains("Indexed"))
        .stdout(predicate::str::contains("Results"));
}

#[test]
fn test_demo_custom_query() {
    cli()
        .args(["demo", "--query", "What is deep learning?"])
        .assert()
        .success()
        .stdout(predicate::str::contains("What is deep learning?"));
}

#[test]
fn test_demo_custom_top_k() {
    cli()
        .args(["demo", "--top-k", "2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Results (2)").or(predicate::str::contains("Results (")));
}

#[test]
fn test_demo_shows_citations() {
    cli()
        .arg("demo")
        .assert()
        .success()
        .stdout(predicate::str::contains("Citations:"));
}

// ============================================================================
// INDEX COMMAND TESTS
// ============================================================================

#[test]
fn test_index_single_file() {
    let tmp = TempDir::new().unwrap();
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "This is a test document about machine learning.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed"));

    // Verify index was created
    assert!(index_path.exists(), "Index directory should be created");
}

#[test]
fn test_index_directory() {
    let tmp = TempDir::new().unwrap();
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();

    fs::write(docs_dir.join("doc1.txt"), "First document about AI.").unwrap();
    fs::write(docs_dir.join("doc2.txt"), "Second document about ML.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 documents"));
}

#[test]
fn test_index_with_chunk_size() {
    let tmp = TempDir::new().unwrap();
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "A ".repeat(500)).unwrap(); // Long document

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--chunk-size",
            "100",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("chunks"));
}

#[test]
fn test_index_nonexistent_path_fails() {
    cli()
        .args([
            "index",
            "--path",
            "/nonexistent/path",
            "--output",
            "/tmp/out",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("No such file")));
}

// ============================================================================
// QUERY COMMAND TESTS
// ============================================================================

#[test]
fn test_query_requires_index() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("nonexistent_index");

    cli()
        .args([
            "query",
            "test query",
            "--index",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("index").or(predicate::str::contains("not found")));
}

#[test]
fn test_query_with_index() {
    let tmp = TempDir::new().unwrap();

    // First, create an index
    let doc_path = tmp.path().join("test.txt");
    fs::write(
        &doc_path,
        "Machine learning is a field of artificial intelligence.",
    )
    .unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Now query
    cli()
        .args([
            "query",
            "What is machine learning?",
            "--index",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Results"));
}

#[test]
fn test_query_json_output() {
    let tmp = TempDir::new().unwrap();

    // Create index
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "Test document content.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Query with JSON output
    cli()
        .args([
            "query",
            "test",
            "--index",
            index_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("{").and(predicate::str::contains("}")));
}

#[test]
fn test_query_top_k() {
    let tmp = TempDir::new().unwrap();

    // Create index with multiple docs
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();
    for i in 0..5 {
        fs::write(
            docs_dir.join(format!("doc{i}.txt")),
            format!("Document {i} content."),
        )
        .unwrap();
    }

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Query with limited results
    cli()
        .args([
            "query",
            "document",
            "--index",
            index_path.to_str().unwrap(),
            "--top-k",
            "2",
        ])
        .assert()
        .success();
}

// ============================================================================
// TRANSCRIBE COMMAND TESTS
// ============================================================================

#[test]
fn test_transcribe_dry_run_no_media() {
    let tmp = TempDir::new().unwrap();
    // Empty directory — no media files
    cli()
        .args([
            "transcribe",
            "--path",
            tmp.path().to_str().unwrap(),
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("No media files found"));
}

#[test]
fn test_transcribe_dry_run_with_media() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("lecture.mp4"), b"").unwrap();
    fs::write(tmp.path().join("talk.wav"), b"").unwrap();

    cli()
        .args([
            "transcribe",
            "--path",
            tmp.path().to_str().unwrap(),
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 media files"))
        .stdout(predicate::str::contains("2 need transcription"))
        .stdout(predicate::str::contains("lecture.mp4"))
        .stdout(predicate::str::contains("talk.wav"));
}

#[test]
fn test_transcribe_skips_existing_sidecars() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("lecture.mp4"), b"").unwrap();
    fs::write(
        tmp.path().join("lecture.srt"),
        "1\n00:00:01,000 --> 00:00:05,000\nHello.\n",
    )
    .unwrap();
    fs::write(tmp.path().join("talk.wav"), b"").unwrap();

    cli()
        .args([
            "transcribe",
            "--path",
            tmp.path().to_str().unwrap(),
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("1 with .srt/.vtt"))
        .stdout(predicate::str::contains("1 need transcription"))
        .stdout(predicate::str::contains("talk.wav"));
}

#[test]
fn test_transcribe_all_have_sidecars() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("lecture.mp4"), b"").unwrap();
    fs::write(
        tmp.path().join("lecture.srt"),
        "1\n00:00:01,000 --> 00:00:05,000\nHello.\n",
    )
    .unwrap();

    cli()
        .args([
            "transcribe",
            "--path",
            tmp.path().to_str().unwrap(),
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Nothing to do"));
}

#[test]
fn test_transcribe_nonexistent_path() {
    cli()
        .args(["transcribe", "--path", "/nonexistent/path", "--dry-run"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_transcribe_help() {
    cli()
        .args(["transcribe", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Batch transcribe"));
}

#[test]
fn test_transcribe_backend_flag() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("lecture.wav"), b"").unwrap();

    cli()
        .args([
            "transcribe",
            "--path",
            tmp.path().to_str().unwrap(),
            "--backend",
            "gpu",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("1 media files"));
}

#[test]
fn test_index_with_manifest() {
    let tmp = TempDir::new().unwrap();
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "This is a test document about machine learning.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--manifest",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Manifest saved to"));

    // Verify manifest file was created
    let manifest_file = index_path.join("manifest.json");
    assert!(manifest_file.exists(), "Manifest file should exist");

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_file).unwrap()).unwrap();
    assert_eq!(manifest["file_count"], 1);
    assert!(manifest["chunk_count"].as_u64().unwrap() >= 1);
    assert!(manifest["files"].is_array());
}

#[test]
#[cfg(feature = "sqlite")]
fn test_index_with_sqlite_flag() {
    let tmp = TempDir::new().unwrap();
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();

    fs::write(
        docs_dir.join("doc1.txt"),
        "First document about Rust programming.",
    )
    .unwrap();
    fs::write(
        docs_dir.join("doc2.txt"),
        "Second document about Python scripting.",
    )
    .unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--sqlite",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SQLite index saved to"))
        .stdout(predicate::str::contains("2 docs"));

    // Verify both index.json and index.sqlite exist
    assert!(index_path.join("index.json").exists());
    assert!(index_path.join("index.sqlite").exists());
}

#[test]
#[cfg(feature = "sqlite")]
fn test_index_sqlite_with_dedup() {
    let tmp = TempDir::new().unwrap();
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();

    fs::write(docs_dir.join("a.txt"), "Identical content here.").unwrap();
    fs::write(docs_dir.join("b.txt"), "Identical content here.").unwrap(); // duplicate
    fs::write(docs_dir.join("c.txt"), "Unique content about SIMD.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--sqlite",
            "--dedup",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SQLite index saved to"));

    assert!(index_path.join("index.sqlite").exists());
}

#[test]
fn test_index_parallel_jobs() {
    let tmp = TempDir::new().unwrap();
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();

    fs::write(docs_dir.join("doc1.txt"), "First document about AI.").unwrap();
    fs::write(docs_dir.join("doc2.txt"), "Second document about ML.").unwrap();
    fs::write(docs_dir.join("doc3.txt"), "Third document about NLP.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--jobs",
            "2",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Loading with 2 parallel jobs"))
        .stdout(predicate::str::contains("3 documents"));
}

// ============================================================================
// HELP AND VERSION TESTS
// ============================================================================

#[test]
fn test_help() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Pure-Rust RAG pipeline CLI"));
}

#[test]
fn test_version() {
    cli()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-rag"));
}

#[test]
fn test_subcommand_help() {
    cli()
        .args(["index", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Index documents"));
}

#[test]
#[cfg(feature = "sqlite")]
fn test_incremental_from_scratch_then_fts_query_has_timecodes() {
    let tmp = TempDir::new().unwrap();
    let docs = tmp.path().join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(
        docs.join("lec.srt"),
        "1\n00:00:01,000 --> 00:00:05,000\nThe kubernetes scheduler places pods on nodes.\n\n\
         2\n00:00:06,000 --> 00:00:09,000\nNext we cover persistent volumes.\n",
    )
    .unwrap();
    fs::write(docs.join("note.txt"), "Plain text about zebra crossings.").unwrap();
    let index = tmp.path().join("idx");
    let run_index = || {
        cli()
            .args(["index", "--path", docs.to_str().unwrap(), "--output"])
            .arg(&index)
            .args(["--sqlite", "--incremental", "--chunk-strategy", "timestamp"])
            .assert()
            .success()
    };
    run_index();
    assert!(index.join("index.sqlite").exists());
    assert!(!index.join("index.json").exists());
    // second run: nothing changed
    run_index().stdout(predicate::str::contains("up to date"));

    let out = cli()
        .args(["query", "what does the kubernetes scheduler do", "--index"])
        .arg(&index)
        .args(["-f", "json", "-t", "3"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v[0]["source"].as_str().unwrap().ends_with("lec.srt"));
    assert!(v[0]["start_secs"].as_f64().is_some());
    assert!(v[0]["end_secs"].as_f64().is_some());

    // text format, untimed doc, no-hit query
    cli()
        .args(["query", "zebra", "--index"])
        .arg(&index)
        .assert()
        .success()
        .stdout(predicate::str::contains("note.txt").not())
        .stdout(predicate::str::contains("Plain text"));
    cli()
        .args(["query", "qqqzzz", "--index"])
        .arg(&index)
        .assert()
        .success()
        .stdout(predicate::str::contains("Results (0)"));
}

#[test]
#[cfg(feature = "sqlite")]
fn test_incremental_second_root_keeps_first_root_docs() {
    let tmp = TempDir::new().unwrap();
    let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
    fs::create_dir(&a).unwrap();
    fs::create_dir(&b).unwrap();
    fs::write(a.join("one.txt"), "alpha content about narwhals").unwrap();
    fs::write(b.join("two.txt"), "beta content about walruses").unwrap();
    let index = tmp.path().join("idx");
    for root in [&a, &b] {
        cli()
            .args(["index", "--path", root.to_str().unwrap(), "--output"])
            .arg(&index)
            .args(["--sqlite", "--incremental"])
            .assert()
            .success();
    }
    for (q, f) in [("narwhals", "one.txt"), ("walruses", "two.txt")] {
        cli()
            .args(["query", q, "--index"])
            .arg(&index)
            .args(["-f", "json"])
            .assert()
            .success()
            .stdout(predicate::str::contains(f));
    }
}
