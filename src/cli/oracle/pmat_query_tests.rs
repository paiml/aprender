//! Tests for PMAT Query integration.

use super::pmat_query_cache::{attach_rag_backlinks, cache_key};
use super::pmat_query_display::{grade_badge, parse_pmat_query_output, tdg_score_bar};
use super::pmat_query_fusion::{compute_quality_summary, format_summary_line, rrf_fuse_results};
use super::pmat_query_types::{FusedResult, PmatQueryOptions, PmatQueryResult, QualitySummary};

fn sample_json() -> &'static str {
    r#"[
        {
            "file_path": "src/pipeline.rs",
            "function_name": "validate_stage",
            "signature": "fn validate_stage(&self, stage: &Stage) -> Result<()>",
            "doc_comment": "Validates a pipeline stage.",
            "start_line": 142,
            "end_line": 185,
            "language": "rust",
            "tdg_score": 92.5,
            "tdg_grade": "A",
            "complexity": 4,
            "big_o": "O(n)",
            "satd_count": 0,
            "loc": 43,
            "relevance_score": 0.87,
            "source": null
        }
    ]"#
}

#[test]
fn test_parse_valid_json() {
    let results = parse_pmat_query_output(sample_json()).expect("unexpected failure");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].function_name, "validate_stage");
    assert_eq!(results[0].file_path, "src/pipeline.rs");
    assert_eq!(results[0].tdg_grade, "A");
    assert!((results[0].tdg_score - 92.5).abs() < f64::EPSILON);
    assert_eq!(results[0].complexity, 4);
    assert_eq!(results[0].big_o, "O(n)");
    assert_eq!(results[0].start_line, 142);
    assert_eq!(results[0].end_line, 185);
}

#[test]
fn test_parse_null_doc_comment() {
    let json = r#"[{
        "file_path": "src/lib.rs",
        "function_name": "foo",
        "signature": "fn foo()",
        "doc_comment": null,
        "start_line": 1,
        "end_line": 5,
        "language": "rust",
        "tdg_score": 50.0,
        "tdg_grade": "C",
        "complexity": 2,
        "big_o": "O(1)",
        "satd_count": 0,
        "loc": 4,
        "relevance_score": 0.5,
        "source": null
    }]"#;
    let results = parse_pmat_query_output(json).expect("unexpected failure");
    assert_eq!(results.len(), 1);
    assert!(results[0].doc_comment.is_none());
    assert!(results[0].source.is_none());
}

#[test]
fn test_parse_with_source() {
    let json = r#"[{
        "file_path": "src/main.rs",
        "function_name": "main",
        "signature": "fn main()",
        "doc_comment": null,
        "start_line": 1,
        "end_line": 3,
        "language": "rust",
        "tdg_score": 100.0,
        "tdg_grade": "A",
        "complexity": 1,
        "big_o": "O(1)",
        "satd_count": 0,
        "loc": 3,
        "relevance_score": 1.0,
        "source": "fn main() {\n    println!(\"hello\");\n}"
    }]"#;
    let results = parse_pmat_query_output(json).expect("unexpected failure");
    assert_eq!(results.len(), 1);
    assert!(results[0].source.is_some());
    assert!(results[0].source.as_ref().expect("unexpected failure").contains("println!"));
}

#[test]
fn test_parse_empty_array() {
    let results = parse_pmat_query_output("[]").expect("unexpected failure");
    assert!(results.is_empty());
}

#[test]
fn test_parse_invalid_json() {
    let result = parse_pmat_query_output("not json");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Failed to parse pmat query output"));
}

#[test]
fn test_parse_multiple_results() {
    let json = r#"[
        {
            "file_path": "a.rs",
            "function_name": "alpha",
            "tdg_score": 90.0,
            "tdg_grade": "A",
            "complexity": 2,
            "big_o": "O(1)",
            "satd_count": 0,
            "relevance_score": 0.9
        },
        {
            "file_path": "b.rs",
            "function_name": "beta",
            "tdg_score": 60.0,
            "tdg_grade": "C",
            "complexity": 12,
            "big_o": "O(n^2)",
            "satd_count": 3,
            "relevance_score": 0.6
        }
    ]"#;
    let results = parse_pmat_query_output(json).expect("unexpected failure");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].function_name, "alpha");
    assert_eq!(results[1].function_name, "beta");
    assert_eq!(results[1].complexity, 12);
    assert_eq!(results[1].satd_count, 3);
}

#[test]
fn test_serialization_roundtrip() {
    let result = PmatQueryResult {
        file_path: "src/lib.rs".to_string(),
        function_name: "process".to_string(),
        signature: "fn process(data: &[u8]) -> Vec<u8>".to_string(),
        doc_comment: Some("Process raw bytes.".to_string()),
        start_line: 10,
        end_line: 25,
        language: "rust".to_string(),
        tdg_score: 85.0,
        tdg_grade: "B".to_string(),
        complexity: 6,
        big_o: "O(n)".to_string(),
        satd_count: 1,
        loc: 15,
        relevance_score: 0.75,
        source: None,
        project: None,
        rag_backlinks: Vec::new(),
    };

    let json = serde_json::to_string(&result).expect("json serialize failed");
    let deserialized: PmatQueryResult =
        serde_json::from_str(&json).expect("json deserialize failed");
    assert_eq!(deserialized.function_name, "process");
    assert!((deserialized.tdg_score - 85.0).abs() < f64::EPSILON);
    assert_eq!(deserialized.tdg_grade, "B");
}

#[test]
fn test_grade_badge_a() {
    let badge = grade_badge("A");
    assert!(badge.contains('A'));
    assert!(badge.starts_with('['));
    assert!(badge.ends_with(']'));
}

#[test]
fn test_grade_badge_f() {
    let badge = grade_badge("F");
    assert!(badge.contains('F'));
}

#[test]
fn test_grade_badge_unknown() {
    let badge = grade_badge("X");
    assert!(badge.contains('X'));
}

#[test]
fn test_tdg_score_bar_full() {
    let bar = tdg_score_bar(100.0, 10);
    assert!(bar.contains("100.0"));
    assert_eq!(bar.matches('\u{2588}').count(), 10);
    assert_eq!(bar.matches('\u{2591}').count(), 0);
}

#[test]
fn test_tdg_score_bar_zero() {
    let bar = tdg_score_bar(0.0, 10);
    assert!(bar.contains("0.0"));
    assert_eq!(bar.matches('\u{2588}').count(), 0);
    assert_eq!(bar.matches('\u{2591}').count(), 10);
}

#[test]
fn test_tdg_score_bar_half() {
    let bar = tdg_score_bar(50.0, 10);
    assert!(bar.contains("50.0"));
    assert_eq!(bar.matches('\u{2588}').count(), 5);
    assert_eq!(bar.matches('\u{2591}').count(), 5);
}

#[test]
fn test_pmat_query_options_construction() {
    let opts = PmatQueryOptions {
        query: "error".to_string(),
        project_path: Some("/home/user/project".to_string()),
        limit: 5,
        min_grade: Some("B".to_string()),
        max_complexity: Some(10),
        include_source: true,
    };
    assert_eq!(opts.query, "error");
    assert_eq!(opts.limit, 5);
    assert_eq!(opts.min_grade.as_deref(), Some("B"));
    assert_eq!(opts.max_complexity, Some(10));
    assert!(opts.include_source);
}

#[test]
fn test_parse_minimal_fields() {
    let json = r#"[{
        "file_path": "x.rs",
        "function_name": "f"
    }]"#;
    let results = parse_pmat_query_output(json).expect("unexpected failure");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].file_path, "x.rs");
    assert_eq!(results[0].function_name, "f");
    assert!(results[0].signature.is_empty());
    assert_eq!(results[0].tdg_score, 0.0);
    assert_eq!(results[0].complexity, 0);
}

// ========================================================================
// v2.0 tests
// ========================================================================

#[test]
fn test_quality_summary_single() {
    let results = vec![PmatQueryResult {
        file_path: "a.rs".into(),
        function_name: "f".into(),
        tdg_grade: "A".into(),
        complexity: 5,
        satd_count: 1,
        ..Default::default()
    }];
    let s = compute_quality_summary(&results);
    assert_eq!(s.grades.get("A"), Some(&1));
    assert!((s.avg_complexity - 5.0).abs() < f64::EPSILON);
    assert_eq!(s.total_satd, 1);
    assert_eq!(s.complexity_range, (5, 5));
}

#[test]
fn test_quality_summary_multiple() {
    let results = vec![
        PmatQueryResult {
            file_path: "a.rs".into(),
            function_name: "f".into(),
            tdg_grade: "A".into(),
            complexity: 2,
            satd_count: 0,
            ..Default::default()
        },
        PmatQueryResult {
            file_path: "b.rs".into(),
            function_name: "g".into(),
            tdg_grade: "B".into(),
            complexity: 8,
            satd_count: 3,
            ..Default::default()
        },
        PmatQueryResult {
            file_path: "c.rs".into(),
            function_name: "h".into(),
            tdg_grade: "A".into(),
            complexity: 4,
            satd_count: 0,
            ..Default::default()
        },
    ];
    let s = compute_quality_summary(&results);
    assert_eq!(s.grades.get("A"), Some(&2));
    assert_eq!(s.grades.get("B"), Some(&1));
    assert!((s.avg_complexity - 14.0 / 3.0).abs() < 0.01);
    assert_eq!(s.total_satd, 3);
    assert_eq!(s.complexity_range, (2, 8));
}

#[test]
fn test_quality_summary_empty() {
    let s = compute_quality_summary(&[]);
    assert!(s.grades.is_empty());
    assert!((s.avg_complexity).abs() < f64::EPSILON);
    assert_eq!(s.total_satd, 0);
}

#[test]
fn test_format_summary_line() {
    let mut grades = std::collections::HashMap::new();
    grades.insert("A".to_string(), 3);
    grades.insert("B".to_string(), 1);
    let s =
        QualitySummary { grades, avg_complexity: 5.5, total_satd: 2, complexity_range: (1, 12) };
    let line = format_summary_line(&s);
    assert!(line.contains("3A"));
    assert!(line.contains("1B"));
    assert!(line.contains("5.5"));
    assert!(line.contains("SATD: 2"));
    assert!(line.contains("1-12"));
}

#[test]
fn test_rrf_fuse_pmat_only() {
    let pmat = vec![PmatQueryResult {
        file_path: "a.rs".into(),
        function_name: "f".into(),
        relevance_score: 0.9,
        tdg_grade: "A".into(),
        ..Default::default()
    }];
    let fused = rrf_fuse_results(&pmat, &[], 10);
    assert_eq!(fused.len(), 1);
    assert!(fused[0].1 > 0.0);
    assert!(matches!(fused[0].0, FusedResult::Function(_)));
}

#[test]
fn test_rrf_fuse_both() {
    use crate::oracle::rag::RetrievalResult;
    use crate::oracle::rag::ScoreBreakdown;

    let pmat = vec![PmatQueryResult {
        file_path: "a.rs".into(),
        function_name: "f".into(),
        relevance_score: 0.9,
        tdg_grade: "A".into(),
        ..Default::default()
    }];
    let rag = vec![RetrievalResult {
        id: "doc1".into(),
        component: "trueno".into(),
        source: "trueno/README.md".into(),
        content: "some content".into(),
        score: 0.8,
        start_line: 1,
        end_line: 10,
        score_breakdown: ScoreBreakdown::default(),
    }];
    let fused = rrf_fuse_results(&pmat, &rag, 10);
    assert_eq!(fused.len(), 2);
    // Both should have positive scores
    assert!(fused[0].1 > 0.0);
    assert!(fused[1].1 > 0.0);
}

#[test]
fn test_cache_key_deterministic() {
    let k1 = cache_key("error handling", Some("/home/user/project"));
    let k2 = cache_key("error handling", Some("/home/user/project"));
    assert_eq!(k1, k2);
}

#[test]
fn test_cache_key_different() {
    let k1 = cache_key("error", Some("/a"));
    let k2 = cache_key("serialize", Some("/a"));
    assert_ne!(k1, k2);
}

#[test]
fn test_attach_rag_backlinks() {
    let mut results = vec![PmatQueryResult {
        file_path: "src/pipeline.rs".into(),
        function_name: "f".into(),
        ..Default::default()
    }];
    let mut chunks = std::collections::HashMap::new();
    chunks.insert("batuta/src/pipeline.rs#42".to_string(), "chunk content".to_string());
    chunks.insert("trueno/src/simd.rs#1".to_string(), "other content".to_string());

    attach_rag_backlinks(&mut results, &chunks);
    assert_eq!(results[0].rag_backlinks.len(), 1);
    assert!(results[0].rag_backlinks[0].contains("pipeline.rs"));
}

#[test]
fn test_attach_rag_backlinks_no_match() {
    let mut results = vec![PmatQueryResult {
        file_path: "src/unique_file.rs".into(),
        function_name: "f".into(),
        ..Default::default()
    }];
    let mut chunks = std::collections::HashMap::new();
    chunks.insert("trueno/src/simd.rs#1".to_string(), "content".to_string());

    attach_rag_backlinks(&mut results, &chunks);
    assert!(results[0].rag_backlinks.is_empty());
}

#[test]
fn test_fused_result_serialization() {
    let fused = FusedResult::Function(Box::new(PmatQueryResult {
        file_path: "a.rs".into(),
        function_name: "f".into(),
        tdg_grade: "A".into(),
        ..Default::default()
    }));
    let json = serde_json::to_string(&fused).expect("json serialize failed");
    assert!(json.contains("\"type\":\"function\""));
    assert!(json.contains("a.rs"));
}

#[test]
fn test_fused_result_document_serialization() {
    let fused = FusedResult::Document {
        component: "trueno".into(),
        source: "trueno/README.md".into(),
        score: 0.85,
        content: "content".into(),
    };
    let json = serde_json::to_string(&fused).expect("json serialize failed");
    assert!(json.contains("\"type\":\"document\""));
    assert!(json.contains("trueno"));
}
