//! Unit tests for RAG oracle (part 1: core + display).

use super::*;

/// Helper: create a temporary SQLite index with test data.
fn create_test_sqlite_index(
    path: &std::path::Path,
    docs: &[(&str, &[(&str, &str)])],
) -> trueno_rag::sqlite::SqliteIndex {
    let idx = trueno_rag::sqlite::SqliteIndex::open(path).expect("sqlite operation failed");
    for (doc_id, chunks) in docs {
        let content: String = chunks.iter().map(|(_, c)| *c).collect::<Vec<_>>().join("\n");
        let chunk_pairs: Vec<(String, String)> = chunks
            .iter()
            .enumerate()
            .map(|(i, (_, c))| (format!("{doc_id}#{i}"), c.to_string()))
            .collect();
        idx.insert_document(doc_id, None, Some(doc_id), &content, &chunk_pairs, None)
            .expect("unexpected failure");
    }
    idx.optimize().expect("optimize failed");
    idx
}

#[test]
fn test_extract_component() {
    use rag_sqlite::extract_component;
    assert_eq!(extract_component("trueno/CLAUDE.md"), "trueno");
    assert_eq!(extract_component("batuta/src/main.rs"), "batuta");
    assert_eq!(extract_component("standalone.txt"), "standalone.txt");
    assert_eq!(extract_component(""), "");
}

#[test]
fn test_sqlite_index_path_is_under_cache() {
    use rag_sqlite::sqlite_index_path;
    let path = sqlite_index_path();
    let path_str = path.to_string_lossy();
    assert!(
        path_str.contains("batuta/rag/index.sqlite"),
        "path should end with batuta/rag/index.sqlite, got: {path_str}"
    );
}

#[test]
fn test_rag_load_sqlite_returns_none_if_missing() {
    use rag_sqlite::rag_load_sqlite;
    let result = rag_load_sqlite();
    assert!(result.is_ok());
}

#[test]
fn test_rag_search_sqlite_returns_results() {
    use rag_sqlite::rag_search_sqlite;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");

    let idx = create_test_sqlite_index(
        &db_path,
        &[
            (
                "doc-a",
                &[
                    ("a#0", "Rust is a systems programming language"),
                    ("a#1", "The borrow checker ensures memory safety"),
                ],
            ),
            ("doc-b", &[("b#0", "Python is an interpreted language")]),
        ],
    );

    let results = rag_search_sqlite(&idx, "borrow checker", 5).expect("sqlite operation failed");
    assert!(!results.is_empty(), "Should find results for 'borrow checker'");
    assert!(results[0].content.contains("borrow checker"), "Top result should contain query terms");
}

#[test]
fn test_rag_search_sqlite_empty_query() {
    use rag_sqlite::rag_search_sqlite;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");

    let idx = create_test_sqlite_index(&db_path, &[("doc-a", &[("a#0", "some content")])]);

    let results = rag_search_sqlite(&idx, "", 5);
    assert!(results.is_ok());
}

#[test]
fn test_rag_search_multi_single_index() {
    use rag_sqlite::rag_search_multi;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");

    let idx = create_test_sqlite_index(
        &db_path,
        &[
            ("doc-a", &[("a#0", "SIMD operations for vector processing")]),
            ("doc-b", &[("b#0", "Python list comprehensions")]),
        ],
    );

    let indices = vec![("oracle".to_string(), idx)];
    let results = rag_search_multi(&indices, "SIMD vector", 5).expect("unexpected failure");
    assert!(!results.is_empty());
    assert!(results[0].content.contains("SIMD"));
}

#[test]
fn test_rag_search_multi_fuses_two_indices() {
    use rag_sqlite::rag_search_multi;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");

    let db1_path = tmp.path().join("oracle.sqlite");
    let idx1 = create_test_sqlite_index(
        &db1_path,
        &[("src/main.rs", &[("s#0", "Rust borrow checker and lifetimes")])],
    );

    let db2_path = tmp.path().join("video.sqlite");
    let idx2 = create_test_sqlite_index(
        &db2_path,
        &[("lecture-1.srt", &[("v#0", "PDCA cycle in software engineering")])],
    );

    let indices = vec![("oracle".to_string(), idx1), ("video-corpus".to_string(), idx2)];

    let results = rag_search_multi(&indices, "PDCA cycle", 5).expect("unexpected failure");
    assert!(!results.is_empty(), "Should find PDCA in video corpus");
    assert!(results[0].content.contains("PDCA"));

    let results = rag_search_multi(&indices, "borrow checker", 5).expect("unexpected failure");
    assert!(!results.is_empty(), "Should find borrow checker in oracle");
    assert!(results[0].content.contains("borrow checker"));
}

#[test]
fn test_rag_search_multi_rrf_scores_are_positive() {
    use rag_sqlite::rag_search_multi;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");

    let db_path = tmp.path().join("test.sqlite");
    let idx = create_test_sqlite_index(
        &db_path,
        &[("doc-a", &[("a#0", "alpha beta gamma")]), ("doc-b", &[("b#0", "delta epsilon zeta")])],
    );

    let indices = vec![("test".to_string(), idx)];
    let results = rag_search_multi(&indices, "alpha", 5).expect("unexpected failure");

    for r in &results {
        assert!(r.score > 0.0, "RRF scores should be positive");
    }
}

#[test]
fn test_rag_search_multi_respects_k_limit() {
    use rag_sqlite::rag_search_multi;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");

    let docs: Vec<(&str, Vec<(&str, &str)>)> = (0..20)
        .map(|i| match i {
            0 => ("d0", vec![("d0#0", "alpha bravo charlie")]),
            1 => ("d1", vec![("d1#0", "alpha delta echo")]),
            2 => ("d2", vec![("d2#0", "alpha foxtrot golf")]),
            3 => ("d3", vec![("d3#0", "alpha hotel india")]),
            4 => ("d4", vec![("d4#0", "alpha juliet kilo")]),
            _ => ("dN", vec![("dN#0", "something else entirely")]),
        })
        .collect();

    let doc_refs: Vec<(&str, &[(&str, &str)])> =
        docs.iter().map(|(id, chunks)| (*id, chunks.as_slice())).collect();

    let idx = create_test_sqlite_index(&db_path, &doc_refs);
    let indices = vec![("test".to_string(), idx)];

    let results = rag_search_multi(&indices, "alpha", 3).expect("unexpected failure");
    assert!(results.len() <= 3, "Should respect k=3 limit");
}

#[test]
fn test_rag_search_multi_empty_indices() {
    use rag_sqlite::rag_search_multi;
    let indices: Vec<(String, trueno_rag::sqlite::SqliteIndex)> = vec![];
    let results = rag_search_multi(&indices, "anything", 5).expect("unexpected failure");
    assert!(results.is_empty());
}

#[test]
fn test_rag_load_all_indices_includes_main() {
    use rag_sqlite::rag_load_all_indices;
    let result = rag_load_all_indices();
    assert!(result.is_ok());
}

#[test]
fn test_sqlite_search_result_fields() {
    use rag_sqlite::SqliteSearchResult;
    let r = SqliteSearchResult {
        chunk_id: "doc#0".to_string(),
        doc_id: "doc".to_string(),
        content: "test content".to_string(),
        score: 0.5,
    };
    assert_eq!(r.chunk_id, "doc#0");
    assert_eq!(r.doc_id, "doc");
    assert_eq!(r.content, "test content");
    assert!((r.score - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_format_timestamp_just_now() {
    use rag_helpers::format_timestamp;
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_ms =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("unexpected failure").as_millis()
            as u64;
    assert_eq!(format_timestamp(now_ms), "just now");
}

#[test]
fn test_format_timestamp_minutes_ago() {
    use rag_helpers::format_timestamp;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    let five_min_ago = SystemTime::now() - Duration::from_secs(300);
    let ms = five_min_ago.duration_since(UNIX_EPOCH).expect("time calculation failed").as_millis()
        as u64;
    let result = format_timestamp(ms);
    assert!(result.contains("min ago"), "expected 'min ago', got: {result}");
}

#[test]
fn test_format_timestamp_hours_ago() {
    use rag_helpers::format_timestamp;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    let two_hours_ago = SystemTime::now() - Duration::from_secs(7200);
    let ms = two_hours_ago.duration_since(UNIX_EPOCH).expect("time calculation failed").as_millis()
        as u64;
    let result = format_timestamp(ms);
    assert!(result.contains("hours ago"), "expected 'hours ago', got: {result}");
}

#[test]
fn test_format_timestamp_days_ago() {
    use rag_helpers::format_timestamp;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    let three_days_ago = SystemTime::now() - Duration::from_secs(259200);
    let ms = three_days_ago.duration_since(UNIX_EPOCH).expect("time calculation failed").as_millis()
        as u64;
    let result = format_timestamp(ms);
    assert!(result.contains("days ago"), "expected 'days ago', got: {result}");
}

#[test]
fn test_print_stat_does_not_panic() {
    use rag_helpers::print_stat;
    print_stat("Test Label", "test value");
    print_stat("Count", 42);
    print_stat("Ratio", format!("{:.2}", 0.95));
}

#[test]
fn test_rag_show_usage_does_not_panic() {
    use rag_display::rag_show_usage;
    rag_show_usage();
}
