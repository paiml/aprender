pub(crate) use super::*;

#[test]
fn test_ast_path_creation() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "x"),
        vec![
            AstNodeType::Parameter,
            AstNodeType::Function,
            AstNodeType::Return,
        ],
        Token::new(TokenType::Identifier, "result"),
    );

    assert_eq!(path.source().value(), "x");
    assert_eq!(path.target().value(), "result");
    assert_eq!(path.len(), 3);
    assert!(!path.is_empty());
}

#[test]
fn test_path_string() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "a"),
        vec![AstNodeType::Parameter, AstNodeType::Function],
        Token::new(TokenType::Identifier, "b"),
    );

    let path_str = path.to_path_string();
    assert!(path_str.contains("a"));
    assert!(path_str.contains("b"));
    assert!(path_str.contains("Param"));
    assert!(path_str.contains("Func"));
}

#[test]
fn test_path_extractor_simple() {
    // Create: Function("add") -> [Parameter("x"), Parameter("y"), Return("sum")]
    let mut func = AstNode::new(AstNodeType::Function, "add");
    func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "y"));
    func.add_child(AstNode::new(AstNodeType::Return, "sum"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&func);

    // Should have paths between: (x,y), (x,sum), (y,sum)
    assert_eq!(paths.len(), 3);
}

#[test]
fn test_path_extractor_max_length() {
    let mut func = AstNode::new(AstNodeType::Function, "test");
    func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "b"));

    // First, check actual path lengths
    let extractor_unlimited = PathExtractor::new(100);
    let all_paths = extractor_unlimited.extract(&func);
    assert!(!all_paths.is_empty(), "Should have at least one path");

    let actual_len = all_paths[0].len();

    // With max length less than actual, should get no paths
    let extractor = PathExtractor::new(actual_len - 1);
    let paths = extractor.extract(&func);
    assert!(
        paths.is_empty(),
        "Expected no paths with max_len={}, but got {} paths",
        actual_len - 1,
        paths.len()
    );

    // With max length equal to actual, should get paths
    let extractor = PathExtractor::new(actual_len);
    let paths = extractor.extract(&func);
    assert!(
        !paths.is_empty(),
        "Expected paths with max_len={}",
        actual_len
    );
}

#[test]
fn test_path_context() {
    let mut func = AstNode::new(AstNodeType::Function, "test");
    func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "b"));
    func.add_child(AstNode::new(AstNodeType::Return, "c"));

    let extractor = PathExtractor::new(10);
    let contexts = extractor.extract_with_context(&func);

    assert!(!contexts.is_empty());
    // Check that indices are valid
    for ctx in &contexts {
        assert!(ctx.source_index < ctx.target_index);
    }
}

#[test]
fn test_single_terminal() {
    let func = AstNode::new(AstNodeType::Function, "empty");
    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&func);

    // Single terminal - no paths possible
    assert!(paths.is_empty());
}

#[test]
fn test_deep_tree_paths() {
    // Create a deeper tree structure
    let mut func = AstNode::new(AstNodeType::Function, "compute");
    let mut block = AstNode::new(AstNodeType::Block, "body");
    let mut cond = AstNode::new(AstNodeType::Conditional, "if");
    cond.add_child(AstNode::new(AstNodeType::Variable, "condition"));
    cond.add_child(AstNode::new(AstNodeType::Return, "early"));
    block.add_child(cond);
    block.add_child(AstNode::new(AstNodeType::Return, "final"));
    func.add_child(block);
    func.add_child(AstNode::new(AstNodeType::Parameter, "input"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&func);

    // Should extract paths between all terminal pairs
    assert!(!paths.is_empty());
}

#[test]
fn test_max_paths_limit() {
    // Create many terminals
    let mut func = AstNode::new(AstNodeType::Function, "many");
    for i in 0..10 {
        func.add_child(AstNode::new(AstNodeType::Parameter, format!("p{i}")));
    }

    let extractor = PathExtractor::new(10).with_max_paths(5);
    let paths = extractor.extract(&func);

    assert_eq!(paths.len(), 5);
}

// ================================================================
// Additional coverage tests for missed branches
// ================================================================

#[test]
fn test_ast_path_empty() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "a"),
        vec![],
        Token::new(TokenType::Identifier, "b"),
    );

    assert!(path.is_empty());
    assert_eq!(path.len(), 0);
}

#[test]
fn test_ast_path_to_path_string_empty_nodes() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "src"),
        vec![],
        Token::new(TokenType::Identifier, "tgt"),
    );

    let s = path.to_path_string();
    assert_eq!(s, "src||tgt");
}

#[test]
fn test_path_extractor_default() {
    let extractor = PathExtractor::default();
    // Default uses MAX_PATH_LENGTH (8) and MAX_PATHS_PER_METHOD (200)
    let mut func = AstNode::new(AstNodeType::Function, "test");
    func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "b"));

    let paths = extractor.extract(&func);
    assert!(!paths.is_empty());
}

#[test]
fn test_extract_with_context_max_paths_limit() {
    let mut func = AstNode::new(AstNodeType::Function, "many");
    for i in 0..10 {
        func.add_child(AstNode::new(AstNodeType::Parameter, format!("p{i}")));
    }

    let extractor = PathExtractor::new(10).with_max_paths(3);
    let contexts = extractor.extract_with_context(&func);

    assert_eq!(contexts.len(), 3);
    // Verify context indices are correct
    for ctx in &contexts {
        assert!(ctx.source_index < ctx.target_index);
    }
}

#[test]
fn test_extract_with_context_single_terminal() {
    let func = AstNode::new(AstNodeType::Function, "single");
    let extractor = PathExtractor::new(10);
    let contexts = extractor.extract_with_context(&func);

    assert!(contexts.is_empty());
}

#[test]
fn test_node_to_token_literal_branch() {
    // Literal node without an explicit token triggers the Literal => Number branch
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::Literal, "42"));
    parent.add_child(AstNode::new(AstNodeType::Parameter, "x"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());

    // The literal node should produce a token
    let has_literal = paths
        .iter()
        .any(|p| p.source().value() == "42" || p.target().value() == "42");
    assert!(has_literal);
}

#[test]
fn test_node_to_token_type_annotation_branch() {
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::TypeAnnotation, "i32"));
    parent.add_child(AstNode::new(AstNodeType::Parameter, "x"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());
}

#[test]
fn test_node_to_token_binary_op_branch() {
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::BinaryOp, "+"));
    parent.add_child(AstNode::new(AstNodeType::Parameter, "a"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());
}

#[test]
fn test_node_to_token_unary_op_branch() {
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::UnaryOp, "-"));
    parent.add_child(AstNode::new(AstNodeType::Parameter, "a"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());
}

#[test]
fn test_node_to_token_generic_branch() {
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::Generic, "T"));
    parent.add_child(AstNode::new(AstNodeType::Parameter, "x"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());
}

#[test]
fn test_node_to_token_default_identifier_branch() {
    // Variable, Assignment, Call, etc. all fall into the default Identifier branch
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    parent.add_child(AstNode::new(AstNodeType::Variable, "x"));
    parent.add_child(AstNode::new(AstNodeType::Call, "foo"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());
}

#[test]
fn test_node_to_token_with_explicit_token() {
    // Terminal node that has an explicit token set
    let mut parent = AstNode::new(AstNodeType::Function, "test");
    let terminal = AstNode::terminal(Token::new(TokenType::Number, "99"));
    parent.add_child(terminal);
    parent.add_child(AstNode::new(AstNodeType::Parameter, "y"));

    let extractor = PathExtractor::new(10);
    let paths = extractor.extract(&parent);
    assert!(!paths.is_empty());

    let has_99 = paths
        .iter()
        .any(|p| p.source().value() == "99" || p.target().value() == "99");
    assert!(has_99);
}

#[test]
fn test_path_context_new() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "a"),
        vec![AstNodeType::Function],
        Token::new(TokenType::Identifier, "b"),
    );
    let ctx = PathContext::new(path, 0, 1);

    assert_eq!(ctx.source_index, 0);
    assert_eq!(ctx.target_index, 1);
    assert_eq!(ctx.path.source().value(), "a");
    assert_eq!(ctx.path.target().value(), "b");
}

#[test]
fn test_path_clone_and_debug() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "x"),
        vec![AstNodeType::Function, AstNodeType::Return],
        Token::new(TokenType::Identifier, "y"),
    );

    let cloned = path.clone();
    assert_eq!(cloned.source().value(), "x");
    assert_eq!(cloned.target().value(), "y");
    assert_eq!(cloned.len(), 2);

    let debug_str = format!("{:?}", path);
    assert!(debug_str.contains("AstPath"));
}

#[test]
fn test_path_context_clone_and_debug() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "a"),
        vec![AstNodeType::Parameter],
        Token::new(TokenType::Identifier, "b"),
    );
    let ctx = PathContext::new(path, 2, 5);
    let cloned = ctx.clone();
    assert_eq!(cloned.source_index, 2);
    assert_eq!(cloned.target_index, 5);

    let debug_str = format!("{:?}", ctx);
    assert!(debug_str.contains("PathContext"));
}

#[test]
fn test_path_extractor_clone_and_debug() {
    let extractor = PathExtractor::new(6).with_max_paths(10);
    let cloned = extractor.clone();

    let debug_str = format!("{:?}", extractor);
    assert!(debug_str.contains("PathExtractor"));

    // Cloned extractor should behave identically
    let mut func = AstNode::new(AstNodeType::Function, "f");
    func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "b"));

    let paths1 = extractor.extract(&func);
    let paths2 = cloned.extract(&func);
    assert_eq!(paths1.len(), paths2.len());
}
