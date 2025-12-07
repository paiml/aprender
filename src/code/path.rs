//! Code2Vec path extraction from AST
//!
//! Implements the path extraction algorithm from the code2vec paper:
//! - Extract paths between terminal nodes (leaves) in the AST
//! - Paths go up from one terminal, through the LCA, down to another terminal
//! - Each path is represented as (source_token, path_nodes, target_token)

use super::ast::{AstNode, AstNodeType, Token, TokenType};

/// A path context representing a connection between two terminals
///
/// This is the core representation from code2vec: a path from one terminal
/// node to another through their lowest common ancestor.
#[derive(Debug, Clone)]
pub struct AstPath {
    /// Source terminal token
    source: Token,
    /// Sequence of node types along the path
    path_nodes: Vec<AstNodeType>,
    /// Target terminal token
    target: Token,
}

impl AstPath {
    /// Create a new AST path
    #[must_use]
    pub fn new(source: Token, path_nodes: Vec<AstNodeType>, target: Token) -> Self {
        Self {
            source,
            path_nodes,
            target,
        }
    }

    /// Get the source token
    #[must_use]
    pub fn source(&self) -> &Token {
        &self.source
    }

    /// Get the path node types
    #[must_use]
    pub fn path_nodes(&self) -> &[AstNodeType] {
        &self.path_nodes
    }

    /// Get the target token
    #[must_use]
    pub fn target(&self) -> &Token {
        &self.target
    }

    /// Get the path length (number of nodes in the path)
    #[must_use]
    pub fn len(&self) -> usize {
        self.path_nodes.len()
    }

    /// Check if path is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path_nodes.is_empty()
    }

    /// Convert path to a string representation for hashing/embedding
    #[must_use]
    pub fn to_path_string(&self) -> String {
        let path_str: String = self
            .path_nodes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("↑↓");
        format!(
            "{}|{}|{}",
            self.source.value(),
            path_str,
            self.target.value()
        )
    }
}

/// Context for a path including positional information
#[derive(Debug, Clone)]
pub struct PathContext {
    /// The AST path
    pub path: AstPath,
    /// Index of source terminal in the method
    pub source_index: usize,
    /// Index of target terminal in the method
    pub target_index: usize,
}

impl PathContext {
    /// Create a new path context
    #[must_use]
    pub fn new(path: AstPath, source_index: usize, target_index: usize) -> Self {
        Self {
            path,
            source_index,
            target_index,
        }
    }
}

/// Extracts paths from AST following the code2vec approach
#[derive(Debug, Clone)]
pub struct PathExtractor {
    /// Maximum path length to extract
    max_path_length: usize,
    /// Maximum number of paths to extract per method
    max_paths: usize,
}

impl PathExtractor {
    /// Create a new path extractor
    #[must_use]
    pub fn new(max_path_length: usize) -> Self {
        Self {
            max_path_length,
            max_paths: super::MAX_PATHS_PER_METHOD,
        }
    }

    /// Set maximum number of paths to extract
    #[must_use]
    pub fn with_max_paths(mut self, max_paths: usize) -> Self {
        self.max_paths = max_paths;
        self
    }

    /// Extract all paths from an AST node
    ///
    /// Returns paths between all pairs of terminal nodes,
    /// filtered by maximum path length.
    #[must_use]
    pub fn extract(&self, root: &AstNode) -> Vec<AstPath> {
        // Collect all terminals with their paths from root
        let terminals_with_paths = Self::collect_terminals_with_paths(root, Vec::new());

        if terminals_with_paths.len() < 2 {
            return Vec::new();
        }

        let mut paths = Vec::new();

        // Generate paths between all pairs of terminals
        for i in 0..terminals_with_paths.len() {
            for j in (i + 1)..terminals_with_paths.len() {
                let path =
                    Self::extract_path_between(&terminals_with_paths[i], &terminals_with_paths[j]);
                if path.len() <= self.max_path_length {
                    paths.push(path);
                    if paths.len() >= self.max_paths {
                        return paths;
                    }
                }
            }
        }

        paths
    }

    /// Extract paths with context information
    #[must_use]
    pub fn extract_with_context(&self, root: &AstNode) -> Vec<PathContext> {
        let terminals_with_paths = Self::collect_terminals_with_paths(root, Vec::new());

        if terminals_with_paths.len() < 2 {
            return Vec::new();
        }

        let mut contexts = Vec::new();

        for i in 0..terminals_with_paths.len() {
            for j in (i + 1)..terminals_with_paths.len() {
                let path =
                    Self::extract_path_between(&terminals_with_paths[i], &terminals_with_paths[j]);
                if path.len() <= self.max_path_length {
                    contexts.push(PathContext::new(path, i, j));
                    if contexts.len() >= self.max_paths {
                        return contexts;
                    }
                }
            }
        }

        contexts
    }

    /// Collect all terminal nodes with their paths from root
    fn collect_terminals_with_paths(
        node: &AstNode,
        current_path: Vec<AstNodeType>,
    ) -> Vec<TerminalWithPath<'_>> {
        let mut path = current_path;
        path.push(node.node_type());

        if node.is_terminal() {
            vec![TerminalWithPath {
                node,
                path_from_root: path,
            }]
        } else {
            node.children()
                .iter()
                .flat_map(|child| Self::collect_terminals_with_paths(child, path.clone()))
                .collect()
        }
    }

    /// Extract path between two terminals through their LCA
    fn extract_path_between(
        source: &TerminalWithPath<'_>,
        target: &TerminalWithPath<'_>,
    ) -> AstPath {
        // Find the lowest common ancestor
        let lca_depth = Self::find_lca_depth(&source.path_from_root, &target.path_from_root);

        // Build path: source → LCA → target
        // Go up from source to LCA, then down from LCA to target
        let mut path_nodes = Vec::new();

        // Path up from source to LCA (reverse order, excluding LCA)
        for node_type in source.path_from_root[lca_depth..].iter().rev() {
            path_nodes.push(*node_type);
        }

        // Path down from LCA to target (excluding LCA which is already included)
        for node_type in &target.path_from_root[(lca_depth + 1)..] {
            path_nodes.push(*node_type);
        }

        // Create tokens from terminal nodes
        let source_token = Self::node_to_token(source.node);
        let target_token = Self::node_to_token(target.node);

        AstPath::new(source_token, path_nodes, target_token)
    }

    /// Find the depth of the lowest common ancestor
    fn find_lca_depth(path1: &[AstNodeType], path2: &[AstNodeType]) -> usize {
        let mut lca_depth = 0;
        for (i, (n1, n2)) in path1.iter().zip(path2.iter()).enumerate() {
            if n1 == n2 {
                lca_depth = i;
            } else {
                break;
            }
        }
        lca_depth
    }

    /// Convert an AST node to a token
    fn node_to_token(node: &AstNode) -> Token {
        if let Some(token) = node.token() {
            token.clone()
        } else {
            // Create a token based on node type and value
            let token_type = match node.node_type() {
                AstNodeType::Literal => TokenType::Number,
                AstNodeType::TypeAnnotation | AstNodeType::Generic => TokenType::TypeName,
                AstNodeType::BinaryOp | AstNodeType::UnaryOp => TokenType::Operator,
                // Default to identifier for variables, parameters, functions, etc.
                _ => TokenType::Identifier,
            };
            Token::new(token_type, node.value())
        }
    }
}

/// Internal struct for tracking terminals with their paths
struct TerminalWithPath<'a> {
    node: &'a AstNode,
    path_from_root: Vec<AstNodeType>,
}

impl Default for PathExtractor {
    fn default() -> Self {
        Self::new(super::MAX_PATH_LENGTH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
