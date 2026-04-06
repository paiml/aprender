//! Code Analysis and `Code2Vec` Embeddings
//!
//! This module provides statistical approximation techniques for code analysis
//! without state explosion, following the code2vec approach.
//!
//! # Architecture
//!
//! ```text
//! Source Code
//!      │
//!      ▼
//! ┌──────────────┐
//! │  AST Parser  │
//! └──────────────┘
//!      │
//!      ▼
//! ┌──────────────────────────────┐
//! │  Path Extractor              │
//! │  (leaf-to-leaf paths)        │
//! └──────────────────────────────┘
//!      │
//!      ▼
//! ┌──────────────────────────────┐
//! │  Code2Vec Encoder            │
//! │  (path → vector embedding)   │
//! └──────────────────────────────┘
//!      │
//!      ▼
//! ┌──────────────────────────────┐
//! │  GNN for Code Graphs         │
//! │  (type/lifetime propagation) │
//! └──────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **AST Representation**: Lightweight AST node types for code analysis
//! - **Path Extraction**: Extract paths between terminal nodes (code2vec style)
//! - **Embedding Encoder**: Map paths to dense vector representations
//! - **Code Graph Processing**: Use GNN layers for type inference
//!
//! # References
//!
//! - Alon et al. (2019), "code2vec: Learning distributed representations of code"
//! - Allamanis et al. (2018), "A survey of machine learning for big code"

mod ast;
mod embedding;
mod mpnn;
mod path;

pub use ast::{AstNode, AstNodeType, Token, TokenType};
pub use embedding::{Code2VecEncoder, CodeEmbedding};
pub use mpnn::{
    pooling, CodeEdgeType, CodeGraph, CodeGraphEdge, CodeGraphNode, CodeMPNN, CodeMPNNLayer,
};
pub use path::{AstPath, PathContext, PathExtractor};

/// Maximum path length for code2vec paths
pub const MAX_PATH_LENGTH: usize = 8;

/// Maximum number of paths to sample per method
pub const MAX_PATHS_PER_METHOD: usize = 200;

/// Default embedding dimension
pub const DEFAULT_EMBEDDING_DIM: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_constants() {
        assert!(MAX_PATH_LENGTH > 0);
        assert!(MAX_PATHS_PER_METHOD > 0);
        assert!(DEFAULT_EMBEDDING_DIM > 0);
    }

    #[test]
    fn test_ast_node_creation() {
        let node = AstNode::new(AstNodeType::Function, "calculate_sum");
        assert_eq!(node.node_type(), AstNodeType::Function);
        assert_eq!(node.value(), "calculate_sum");
        assert!(node.children().is_empty());
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenType::Identifier, "foo");
        assert_eq!(token.token_type(), TokenType::Identifier);
        assert_eq!(token.value(), "foo");
    }

    #[test]
    fn test_path_extractor_simple() {
        // Create a simple AST: Function -> [Param, Return]
        let mut func = AstNode::new(AstNodeType::Function, "add");
        func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
        func.add_child(AstNode::new(AstNodeType::Parameter, "y"));
        func.add_child(AstNode::new(AstNodeType::Return, "sum"));

        let extractor = PathExtractor::new(MAX_PATH_LENGTH);
        let paths = extractor.extract(&func);

        // Should extract paths between terminals
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_code2vec_encoder() {
        let encoder = Code2VecEncoder::new(DEFAULT_EMBEDDING_DIM);

        let path = AstPath::new(
            Token::new(TokenType::Identifier, "x"),
            vec![
                AstNodeType::Parameter,
                AstNodeType::Function,
                AstNodeType::Return,
            ],
            Token::new(TokenType::Identifier, "result"),
        );

        let embedding = encoder.encode_path(&path);
        assert_eq!(embedding.len(), DEFAULT_EMBEDDING_DIM);
    }

    #[test]
    fn test_code_embedding_aggregation() {
        let encoder = Code2VecEncoder::new(DEFAULT_EMBEDDING_DIM);

        let paths = vec![
            AstPath::new(
                Token::new(TokenType::Identifier, "a"),
                vec![AstNodeType::Parameter, AstNodeType::Function],
                Token::new(TokenType::Identifier, "b"),
            ),
            AstPath::new(
                Token::new(TokenType::Identifier, "c"),
                vec![AstNodeType::Return, AstNodeType::Function],
                Token::new(TokenType::Identifier, "d"),
            ),
        ];

        let embedding = encoder.aggregate_paths(&paths);
        assert_eq!(embedding.dim(), DEFAULT_EMBEDDING_DIM);
    }
}
