use super::*;
use crate::citl::diagnostic::DiagnosticSeverity;
use crate::citl::{Difficulty, ErrorCategory};
// ==================== ErrorEmbedding Tests ====================

#[test]
fn test_error_embedding_new() {
    let vector = vec![0.5; 256];
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding = ErrorEmbedding::new(vector.clone(), code, 12345);

    assert_eq!(embedding.dim(), 256);
    assert_eq!(embedding.context_hash, 12345);
}

#[test]
fn test_error_embedding_cosine_similarity() {
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0, 0.0];

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
    let e2 = ErrorEmbedding::new(v2, code.clone(), 0);
    let e3 = ErrorEmbedding::new(v3, code, 0);

    // Same vectors should have similarity 1.0
    assert!((e1.cosine_similarity(&e2) - 1.0).abs() < 0.001);

    // Orthogonal vectors should have similarity 0.0
    assert!(e1.cosine_similarity(&e3).abs() < 0.001);
}

#[test]
fn test_error_embedding_l2_distance() {
    let v1 = vec![0.0, 0.0];
    let v2 = vec![3.0, 4.0];

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
    let e2 = ErrorEmbedding::new(v2, code, 0);

    // Distance should be 5.0 (3-4-5 triangle)
    assert!((e1.l2_distance(&e2) - 5.0).abs() < 0.001);
}

// ==================== ErrorEncoder Tests ====================

#[test]
fn test_error_encoder_new() {
    let encoder = ErrorEncoder::new();
    assert_eq!(encoder.dim, 256);
}

#[test]
fn test_error_encoder_with_dim() {
    let encoder = ErrorEncoder::with_dim(128);
    assert_eq!(encoder.dim, 128);
}

#[test]
fn test_error_encoder_encode() {
    let encoder = ErrorEncoder::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let embedding = encoder.encode(&diagnostic, source);

    assert_eq!(embedding.dim(), 256);
    assert_eq!(embedding.error_code.code, "E0308");
}

#[test]
fn test_error_encoder_encode_with_types() {
    let encoder = ErrorEncoder::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
            .with_expected(crate::citl::diagnostic::TypeInfo::new("i32"))
            .with_found(crate::citl::diagnostic::TypeInfo::new("&str"));

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let embedding = encoder.encode(&diagnostic, source);

    assert_eq!(embedding.dim(), 256);
    // Type features should be populated
    let type_region: f32 = embedding.vector[128..192].iter().sum();
    assert!(type_region > 0.0);
}

#[test]
fn test_error_encoder_similar_errors_similar_embeddings() {
    let encoder = ErrorEncoder::new();

    // Two similar type mismatch errors
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span1 = SourceSpan::single_line("test.rs", 10, 5, 20);
    let span2 = SourceSpan::single_line("test.rs", 20, 5, 20);

    let diag1 = CompilerDiagnostic::new(
        code.clone(),
        DiagnosticSeverity::Error,
        "mismatched types",
        span1,
    );
    let diag2 = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span2);

    let source1 = "fn main() { let x: i32 = \"hello\"; }";
    let source2 = "fn foo() { let y: i32 = \"world\"; }";

    let e1 = encoder.encode(&diag1, source1);
    let e2 = encoder.encode(&diag2, source2);

    // Same error type should have high similarity
    let similarity = e1.cosine_similarity(&e2);
    assert!(
        similarity > 0.5,
        "Similar errors should have similarity > 0.5, got {similarity}"
    );
}

#[test]
fn test_error_encoder_different_errors_different_embeddings() {
    let encoder = ErrorEncoder::new();

    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);

    let diag1 = CompilerDiagnostic::new(
        code1,
        DiagnosticSeverity::Error,
        "mismatched types",
        span.clone(),
    );
    let diag2 = CompilerDiagnostic::new(
        code2,
        DiagnosticSeverity::Error,
        "borrow of moved value",
        span,
    );

    let source = "fn main() {}";

    let e1 = encoder.encode(&diag1, source);
    let e2 = encoder.encode(&diag2, source);

    // Different error types should have lower similarity
    let similarity = e1.cosine_similarity(&e2);
    assert!(
        similarity < 0.9,
        "Different errors should have similarity < 0.9, got {similarity}"
    );
}

// ==================== ProgramFeedbackGraph Tests ====================

#[test]
fn test_program_feedback_graph_new() {
    let graph = ProgramFeedbackGraph::new();
    assert_eq!(graph.num_nodes(), 0);
    assert_eq!(graph.num_edges(), 0);
}

#[test]
fn test_program_feedback_graph_add_node() {
    let mut graph = ProgramFeedbackGraph::new();
    let features = vec![1.0, 2.0, 3.0];
    let idx = graph.add_node(NodeType::Ast, features);

    assert_eq!(idx, 0);
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn test_program_feedback_graph_add_edge() {
    let mut graph = ProgramFeedbackGraph::new();
    graph.add_node(NodeType::Ast, vec![1.0]);
    graph.add_node(NodeType::Diagnostic, vec![2.0]);
    graph.add_edge(0, 1, EdgeType::DiagnosticRefers);

    assert_eq!(graph.num_edges(), 1);
    assert_eq!(graph.edges[0], (0, 1));
    assert_eq!(graph.edge_types[0], EdgeType::DiagnosticRefers);
}

#[test]
fn test_program_feedback_graph_complex() {
    let mut graph = ProgramFeedbackGraph::new();

    // Add AST nodes
    let ast1 = graph.add_node(NodeType::Ast, vec![1.0, 0.0]);
    let ast2 = graph.add_node(NodeType::Ast, vec![0.0, 1.0]);

    // Add diagnostic node
    let diag = graph.add_node(NodeType::Diagnostic, vec![1.0, 1.0]);

    // Add type nodes
    let expected = graph.add_node(NodeType::ExpectedType, vec![1.0, 0.0]);
    let found = graph.add_node(NodeType::FoundType, vec![0.0, 1.0]);

    // Add edges
    graph.add_edge(ast1, ast2, EdgeType::AstChild);
    graph.add_edge(diag, ast2, EdgeType::DiagnosticRefers);
    graph.add_edge(diag, expected, EdgeType::Expects);
    graph.add_edge(diag, found, EdgeType::Found);

    assert_eq!(graph.num_nodes(), 5);
    assert_eq!(graph.num_edges(), 4);
}

// ==================== NodeType and EdgeType Tests ====================

#[test]
fn test_node_types() {
    let types = [
        NodeType::Ast,
        NodeType::Diagnostic,
        NodeType::ExpectedType,
        NodeType::FoundType,
        NodeType::Suggestion,
    ];
    assert_eq!(types.len(), 5);
}

#[test]
fn test_edge_types() {
    let types = [
        EdgeType::AstChild,
        EdgeType::DataFlow,
        EdgeType::ControlFlow,
        EdgeType::DiagnosticRefers,
        EdgeType::Expects,
        EdgeType::Found,
    ];
    assert_eq!(types.len(), 6);
}

// ==================== GNNErrorEncoder Tests ====================

#[test]
fn test_gnn_encoder_new() {
    let encoder = GNNErrorEncoder::new(64, 256);
    assert_eq!(encoder.output_dim(), 256);
}

#[test]
fn test_gnn_encoder_default_config() {
    let encoder = GNNErrorEncoder::default_config();
    assert_eq!(encoder.output_dim(), 256);
}

#[test]
fn test_gnn_encoder_default_trait() {
    let encoder = GNNErrorEncoder::default();
    assert_eq!(encoder.output_dim(), 256);
}

#[test]
fn test_gnn_encoder_build_graph() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let graph = encoder.build_graph(&diagnostic, source);

    // Should have at least 1 node (diagnostic)
    assert!(graph.num_nodes() >= 1);
    // Should have diagnostic node
    assert!(graph.node_types.contains(&NodeType::Diagnostic));
}

#[test]
fn test_gnn_encoder_build_graph_with_types() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
            .with_expected(crate::citl::diagnostic::TypeInfo::new("i32"))
            .with_found(crate::citl::diagnostic::TypeInfo::new("&str"));

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let graph = encoder.build_graph(&diagnostic, source);

    // Should have expected and found type nodes
    assert!(graph.node_types.contains(&NodeType::ExpectedType));
    assert!(graph.node_types.contains(&NodeType::FoundType));
    // Should have edges to type nodes
    assert!(graph.num_edges() >= 2);
}

#[test]
fn test_gnn_encoder_encode_graph() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let graph = encoder.build_graph(&diagnostic, source);
    let embedding = encoder.encode_graph(&graph);

    // Should produce correct dimension
    assert_eq!(embedding.dim(), 128);
    // Should be normalized (unit length)
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be normalized, got norm {}",
        norm
    );
}

#[test]
fn test_gnn_encoder_encode_convenience() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    let source = "fn main() { let x: i32 = \"hello\"; }";
    let embedding = encoder.encode(&diagnostic, source);

    assert_eq!(embedding.dim(), 128);
}

#[test]
fn test_gnn_encoder_empty_graph() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let graph = ProgramFeedbackGraph::new();
    let embedding = encoder.encode_graph(&graph);

    // Should return zero embedding for empty graph
    assert_eq!(embedding.dim(), 128);
    assert!(embedding.vector.iter().all(|&x| x == 0.0));
}

#[test]
fn test_gnn_encoder_similar_errors_similar_embeddings() {
    let encoder = GNNErrorEncoder::new(32, 128);

    // Two similar type mismatch errors
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span1 = SourceSpan::single_line("test.rs", 10, 5, 20);
    let span2 = SourceSpan::single_line("test.rs", 20, 5, 20);

    let diag1 = CompilerDiagnostic::new(
        code.clone(),
        DiagnosticSeverity::Error,
        "mismatched types",
        span1,
    );
    let diag2 = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span2);

    let source1 = "fn main() { let x: i32 = \"hello\"; }";
    let source2 = "fn foo() { let y: i32 = \"world\"; }";

    let e1 = encoder.encode(&diag1, source1);
    let e2 = encoder.encode(&diag2, source2);

    // Same error type should have non-negative similarity
    let similarity = e1.cosine_similarity(&e2);
    assert!(
        similarity > 0.0,
        "Similar errors should have positive similarity, got {}",
        similarity
    );
}

#[test]
fn test_gnn_encoder_different_categories() {
    let encoder = GNNErrorEncoder::new(32, 128);

    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);

    let diag1 = CompilerDiagnostic::new(
        code1,
        DiagnosticSeverity::Error,
        "mismatched types",
        span.clone(),
    );
    let diag2 = CompilerDiagnostic::new(
        code2,
        DiagnosticSeverity::Error,
        "borrow of moved value",
        span,
    );

    let source = "fn main() { let x = 5; }";

    let e1 = encoder.encode(&diag1, source);
    let e2 = encoder.encode(&diag2, source);

    // Should produce different embeddings
    let diff: f32 = e1
        .vector
        .iter()
        .zip(e2.vector.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.1,
        "Different error categories should have different embeddings"
    );
}

#[test]
fn test_gnn_encoder_tokenize_rust() {
    let tokens = GNNErrorEncoder::tokenize_rust("let x: i32 = 5;");
    assert!(!tokens.is_empty());
    assert!(tokens.contains(&"let".to_string()));
}

#[test]
fn test_gnn_encoder_tokenize_rust_complex() {
    let tokens = GNNErrorEncoder::tokenize_rust("fn foo<T: Clone>(x: &mut T) -> Result<(), Error>");
    assert!(tokens.contains(&"fn".to_string()));
    // Should capture punctuation
    assert!(tokens.iter().any(|t| t.contains('<') || t == "<"));
}

#[test]
fn test_gnn_encoder_with_suggestion() {
    use crate::citl::diagnostic::{CodeReplacement, CompilerSuggestion, SuggestionApplicability};

    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let suggestion = CompilerSuggestion::new(
        "consider using .into() to convert",
        SuggestionApplicability::MachineApplicable,
        CodeReplacement::new(span.clone(), ".into()"),
    );
    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
            .with_suggestion(suggestion);

    let source = "fn main() { let x: String = \"hello\"; }";
    let graph = encoder.build_graph(&diagnostic, source);

    // Should have suggestion node
    assert!(graph.node_types.contains(&NodeType::Suggestion));
}

#[test]
fn test_gnn_encoder_graph_hash_consistency() {
    let encoder = GNNErrorEncoder::new(32, 128);

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);

    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    let source = "fn main() {}";

    // Same input should produce same hash
    let e1 = encoder.encode(&diag, source);
    let e2 = encoder.encode(&diag, source);

    assert_eq!(e1.context_hash, e2.context_hash);
}

#[test]
fn test_gnn_encoder_graph_hash_varies_with_structure() {
    let encoder = GNNErrorEncoder::new(32, 128);

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);

    // Simple diagnostic
    let diag1 = CompilerDiagnostic::new(
        code.clone(),
        DiagnosticSeverity::Error,
        "mismatched types",
        span.clone(),
    );

    // Diagnostic with type info (different structure)
    let diag2 = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
        .with_expected(crate::citl::diagnostic::TypeInfo::new("i32"))
        .with_found(crate::citl::diagnostic::TypeInfo::new("&str"));

    let source = "fn main() {}";
    let e1 = encoder.encode(&diag1, source);
    let e2 = encoder.encode(&diag2, source);

    // Different structures should have different hashes
    assert_ne!(e1.context_hash, e2.context_hash);
}

#[test]
fn test_gnn_encoder_large_source() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::new("test.rs", 1, 10, 1, 80);

    let diagnostic =
        CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    // Large source with many lines
    let source = (0..50)
        .map(|i| format!("    let var_{}: i32 = {};", i, i))
        .collect::<Vec<_>>()
        .join("\n");

    let graph = encoder.build_graph(&diagnostic, &source);

    // Should limit AST nodes to avoid huge graphs
    assert!(
        graph.num_nodes() <= 25,
        "Graph should limit nodes, got {}",
        graph.num_nodes()
    );

    // Should still produce valid embedding
    let embedding = encoder.encode_graph(&graph);
    assert_eq!(embedding.dim(), 128);
}

#[test]
fn test_gnn_encoder_extract_error_code_from_graph() {
    let encoder = GNNErrorEncoder::new(32, 128);
    let code = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic = CompilerDiagnostic::new(
        code,
        DiagnosticSeverity::Error,
        "borrow of moved value",
        span,
    );

    let source = "fn main() { let x = vec![1]; let y = x; x.push(1); }";
    let graph = encoder.build_graph(&diagnostic, source);
    let extracted = encoder.extract_error_code_from_graph(&graph);

    // Should extract ownership category
    assert_eq!(extracted.category, ErrorCategory::Ownership);
}
