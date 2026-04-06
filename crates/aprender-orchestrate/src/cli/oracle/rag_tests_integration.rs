//! Integration tests for RAG oracle (part 2: display + commands + stats).

use super::*;

use crate::cli::oracle::types::OracleOutputFormat;
use rag_display::rag_display_results;

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

fn make_result(
    id: &str,
    component: &str,
    source: &str,
    content: &str,
    score: f64,
) -> crate::oracle::rag::RetrievalResult {
    crate::oracle::rag::RetrievalResult {
        id: id.to_string(),
        component: component.to_string(),
        source: source.to_string(),
        content: content.to_string(),
        score,
        start_line: 1,
        end_line: 1,
        score_breakdown: crate::oracle::rag::ScoreBreakdown {
            bm25_score: score * 5.0,
            dense_score: 0.0,
            rrf_score: 0.0,
            rerank_score: None,
        },
    }
}

#[test]
fn test_rag_display_results_text() {
    let results =
        vec![make_result("doc#0", "trueno", "trueno/lib.rs", "SIMD tensor operations", 0.95)];
    let result = rag_display_results("test query", &results, OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_rag_display_results_json() {
    let results = vec![make_result("doc#0", "batuta", "batuta/main.rs", "test content", 0.8)];
    let result = rag_display_results("json query", &results, OracleOutputFormat::Json);
    assert!(result.is_ok());
}

#[test]
fn test_rag_display_results_markdown() {
    let results = vec![make_result("doc#0", "pmat", "pmat/analysis.rs", "code analysis", 0.7)];
    let result = rag_display_results("md query", &results, OracleOutputFormat::Markdown);
    assert!(result.is_ok());
}

#[test]
fn test_rag_format_multi_index_stats_text() {
    use rag_stats::rag_format_multi_index_stats;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");
    let idx = create_test_sqlite_index(&db_path, &[("doc-a", &[("a#0", "content")])]);
    let indices = vec![("oracle".to_string(), idx)];
    let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_rag_format_multi_index_stats_json() {
    use rag_stats::rag_format_multi_index_stats;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");
    let idx = create_test_sqlite_index(&db_path, &[("doc-a", &[("a#0", "content")])]);
    let indices = vec![("oracle".to_string(), idx)];
    let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Json);
    assert!(result.is_ok());
}

#[test]
fn test_rag_format_multi_index_stats_markdown() {
    use rag_stats::rag_format_multi_index_stats;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");
    let idx = create_test_sqlite_index(&db_path, &[("doc-a", &[("a#0", "content")])]);
    let indices = vec![("oracle".to_string(), idx)];
    let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Markdown);
    assert!(result.is_ok());
}

#[test]
fn test_rag_format_multi_index_stats_multiple() {
    use rag_stats::rag_format_multi_index_stats;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");

    let db1 = tmp.path().join("oracle.sqlite");
    let idx1 = create_test_sqlite_index(&db1, &[("doc-a", &[("a#0", "content alpha")])]);
    let db2 = tmp.path().join("video.sqlite");
    let idx2 = create_test_sqlite_index(&db2, &[("doc-b", &[("b#0", "content beta")])]);

    let indices = vec![("oracle".to_string(), idx1), ("video-corpus".to_string(), idx2)];
    let result = rag_format_multi_index_stats(&indices, OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_sqlite_with_query() {
    use rag_commands::cmd_oracle_rag_sqlite;
    let result =
        cmd_oracle_rag_sqlite(Some("test query".into()), OracleOutputFormat::Text, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_sqlite_no_query() {
    use rag_commands::cmd_oracle_rag_sqlite;
    let result = cmd_oracle_rag_sqlite(None, OracleOutputFormat::Text, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_sqlite_json_format() {
    use rag_commands::cmd_oracle_rag_sqlite;
    let result = cmd_oracle_rag_sqlite(Some("SIMD".into()), OracleOutputFormat::Json, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_sqlite_with_profiling() {
    use rag_commands::cmd_oracle_rag_sqlite;
    let result = cmd_oracle_rag_sqlite(
        Some("Rust programming".into()),
        OracleOutputFormat::Text,
        true,
        true,
    );
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_sqlite_markdown_format() {
    use rag_commands::cmd_oracle_rag_sqlite;
    let result =
        cmd_oracle_rag_sqlite(Some("Rust".into()), OracleOutputFormat::Markdown, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_rag_print_profiling_summary_does_not_panic() {
    use rag_display::rag_print_profiling_summary;
    rag_print_profiling_summary();
}

#[test]
fn test_cmd_oracle_rag_dispatch() {
    use rag_commands::cmd_oracle_rag;
    let result = cmd_oracle_rag(Some("dispatch test".into()), OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_with_profile_dispatch() {
    use rag_commands::cmd_oracle_rag_with_profile;
    let result = cmd_oracle_rag_with_profile(
        Some("profile dispatch".into()),
        OracleOutputFormat::Text,
        false,
        false,
    );
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_text() {
    use rag_stats::cmd_oracle_rag_stats;
    let result = cmd_oracle_rag_stats(OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_json() {
    use rag_stats::cmd_oracle_rag_stats;
    let result = cmd_oracle_rag_stats(OracleOutputFormat::Json);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_markdown() {
    use rag_stats::cmd_oracle_rag_stats;
    let result = cmd_oracle_rag_stats(OracleOutputFormat::Markdown);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_json_fallback_text() {
    use rag_stats::cmd_oracle_rag_stats_json;
    let result = cmd_oracle_rag_stats_json(OracleOutputFormat::Text);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_json_fallback_json() {
    use rag_stats::cmd_oracle_rag_stats_json;
    let result = cmd_oracle_rag_stats_json(OracleOutputFormat::Json);
    assert!(result.is_ok());
}

#[test]
fn test_cmd_oracle_rag_stats_json_fallback_markdown() {
    use rag_stats::cmd_oracle_rag_stats_json;
    let result = cmd_oracle_rag_stats_json(OracleOutputFormat::Markdown);
    assert!(result.is_ok());
}

#[test]
fn test_rag_dispatch_search_single_index() {
    use rag_sqlite::rag_dispatch_search;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db_path = tmp.path().join("test.sqlite");
    let idx = create_test_sqlite_index(
        &db_path,
        &[("doc-a", &[("a#0", "Rust borrow checker and ownership")])],
    );
    let indices = vec![("oracle".to_string(), idx)];
    let results = rag_dispatch_search(&indices, "borrow checker", 5).expect("unexpected failure");
    assert!(!results.is_empty());
}

#[test]
fn test_rag_dispatch_search_multi_index() {
    use rag_sqlite::rag_dispatch_search;
    let tmp = tempfile::TempDir::new().expect("tempdir creation failed");
    let db1 = tmp.path().join("a.sqlite");
    let db2 = tmp.path().join("b.sqlite");
    let idx1 = create_test_sqlite_index(&db1, &[("doc-a", &[("a#0", "Rust memory safety")])]);
    let idx2 = create_test_sqlite_index(&db2, &[("doc-b", &[("b#0", "Python type hints")])]);
    let indices = vec![("a".to_string(), idx1), ("b".to_string(), idx2)];
    let results = rag_dispatch_search(&indices, "memory safety", 5).expect("unexpected failure");
    assert!(!results.is_empty());
}

#[test]
fn test_cmd_oracle_rag_dashboard_without_tui() {
    use rag_stats::cmd_oracle_rag_dashboard;
    let result = cmd_oracle_rag_dashboard();
    let _ = result;
}
