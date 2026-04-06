//! Python grammar definition
//!
//! Grammar rules for Python code generation, targeting depyler transpilation.
//! Uses tree-sitter for proper AST validation when the `tree-sitter` feature is enabled.

use crate::Language;

use super::Grammar;

/// Python grammar for code generation
///
/// When the `tree-sitter` feature is enabled, uses tree-sitter-python for
/// proper syntax validation. Otherwise, falls back to basic heuristics.
pub struct PythonGrammar {
    #[cfg(feature = "tree-sitter")]
    parser: std::sync::Mutex<tree_sitter::Parser>,
}

impl std::fmt::Debug for PythonGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonGrammar")
            .field("language", &"python")
            .finish()
    }
}

impl Default for PythonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonGrammar {
    /// Create a new Python grammar
    ///
    /// # Panics
    ///
    /// Panics if the tree-sitter Python grammar fails to load (should never happen
    /// with a correctly compiled tree-sitter-python dependency).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new() -> Self {
        #[cfg(feature = "tree-sitter")]
        {
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&tree_sitter_python::LANGUAGE.into())
                .expect("Failed to load Python grammar");
            Self {
                parser: std::sync::Mutex::new(parser),
            }
        }
        #[cfg(not(feature = "tree-sitter"))]
        {
            Self {}
        }
    }

    /// Parse Python code and return the AST tree
    ///
    /// Returns `None` if parsing fails or tree-sitter feature is disabled.
    #[cfg(feature = "tree-sitter")]
    pub fn parse(&self, code: &str) -> Option<tree_sitter::Tree> {
        let mut parser = self.parser.lock().ok()?;
        parser.parse(code, None)
    }

    /// Get the root node of parsed code
    #[cfg(feature = "tree-sitter")]
    pub fn root_node(&self, code: &str) -> Option<String> {
        self.parse(code)
            .map(|tree| tree.root_node().kind().to_string())
    }

    /// Check if the parsed code has any syntax errors
    #[cfg(feature = "tree-sitter")]
    pub fn has_errors(&self, code: &str) -> bool {
        self.parse(code)
            .map_or(true, |tree| tree.root_node().has_error())
    }

    /// Get the AST depth of parsed code
    #[cfg(feature = "tree-sitter")]
    pub fn ast_depth(&self, code: &str) -> usize {
        fn max_depth(node: tree_sitter::Node<'_>) -> usize {
            let child_depths = node
                .children(&mut node.walk())
                .map(max_depth)
                .max()
                .unwrap_or(0);
            1 + child_depths
        }

        self.parse(code)
            .map_or(0, |tree| max_depth(tree.root_node()))
    }

    /// Count the number of nodes in the AST
    #[cfg(feature = "tree-sitter")]
    pub fn node_count(&self, code: &str) -> usize {
        fn count_nodes(node: tree_sitter::Node<'_>) -> usize {
            1 + node
                .children(&mut node.walk())
                .map(count_nodes)
                .sum::<usize>()
        }

        self.parse(code)
            .map_or(0, |tree| count_nodes(tree.root_node()))
    }
}

impl Grammar for PythonGrammar {
    fn language(&self) -> Language {
        Language::Python
    }

    fn validate(&self, code: &str) -> bool {
        if code.is_empty() {
            return false;
        }

        #[cfg(feature = "tree-sitter")]
        {
            !self.has_errors(code)
        }

        #[cfg(not(feature = "tree-sitter"))]
        {
            // Basic fallback validation without tree-sitter
            // Check for obvious syntax issues
            let balanced_parens = code.chars().filter(|&c| c == '(').count()
                == code.chars().filter(|&c| c == ')').count();
            let balanced_brackets = code.chars().filter(|&c| c == '[').count()
                == code.chars().filter(|&c| c == ']').count();
            let balanced_braces = code.chars().filter(|&c| c == '{').count()
                == code.chars().filter(|&c| c == '}').count();

            balanced_parens && balanced_brackets && balanced_braces
        }
    }

    fn max_enumeration_depth(&self) -> usize {
        5 // Python ASTs get complex quickly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_grammar_language() {
        let grammar = PythonGrammar::new();
        assert_eq!(grammar.language(), Language::Python);
    }

    #[test]
    fn test_python_grammar_validate_basic() {
        let grammar = PythonGrammar::new();
        assert!(grammar.validate("x = 1"));
        assert!(!grammar.validate(""));
    }

    #[test]
    fn test_python_grammar_validate_function() {
        let grammar = PythonGrammar::new();
        assert!(grammar.validate("def foo():\n    pass"));
        assert!(grammar.validate("def add(a, b):\n    return a + b"));
    }

    #[test]
    fn test_python_grammar_validate_class() {
        let grammar = PythonGrammar::new();
        assert!(grammar.validate("class Foo:\n    pass"));
        assert!(grammar.validate("class Bar:\n    def __init__(self):\n        self.x = 1"));
    }

    #[test]
    fn test_python_grammar_validate_control_flow() {
        let grammar = PythonGrammar::new();
        assert!(grammar.validate("if x:\n    y = 1"));
        assert!(grammar.validate("for i in range(10):\n    print(i)"));
        assert!(grammar.validate("while True:\n    break"));
    }

    #[test]
    fn test_python_grammar_validate_unbalanced() {
        let grammar = PythonGrammar::new();
        // Unbalanced parentheses should fail
        assert!(!grammar.validate("x = (1 + 2"));
        assert!(!grammar.validate("x = [1, 2"));
    }

    #[test]
    fn test_python_grammar_max_depth() {
        let grammar = PythonGrammar::new();
        assert_eq!(grammar.max_enumeration_depth(), 5);
    }

    #[test]
    fn test_python_grammar_debug() {
        let grammar = PythonGrammar::new();
        let debug = format!("{:?}", grammar);
        assert!(debug.contains("PythonGrammar"));
        assert!(debug.contains("python"));
    }

    #[test]
    fn test_python_grammar_default() {
        let grammar = PythonGrammar::default();
        assert_eq!(grammar.language(), Language::Python);
    }

    #[test]
    fn test_python_grammar_validate_unbalanced_braces() {
        let grammar = PythonGrammar::new();
        // Unbalanced braces should fail
        assert!(!grammar.validate("x = {1, 2"));
    }

    #[cfg(feature = "tree-sitter")]
    mod tree_sitter_tests {
        use super::*;

        #[test]
        fn test_parse_simple() {
            let grammar = PythonGrammar::new();
            let tree = grammar.parse("x = 1");
            assert!(tree.is_some());
        }

        #[test]
        fn test_root_node() {
            let grammar = PythonGrammar::new();
            let root = grammar.root_node("x = 1");
            assert_eq!(root, Some("module".to_string()));
        }

        #[test]
        fn test_has_errors_valid() {
            let grammar = PythonGrammar::new();
            assert!(!grammar.has_errors("x = 1"));
            assert!(!grammar.has_errors("def foo(): pass"));
        }

        #[test]
        fn test_has_errors_invalid() {
            let grammar = PythonGrammar::new();
            assert!(grammar.has_errors("def foo("));
            assert!(grammar.has_errors("class :"));
        }

        #[test]
        fn test_ast_depth() {
            let grammar = PythonGrammar::new();
            let simple_depth = grammar.ast_depth("x = 1");
            let complex_depth = grammar.ast_depth("def foo():\n    if x:\n        return y + z");
            assert!(simple_depth > 0);
            assert!(complex_depth > simple_depth);
        }

        #[test]
        fn test_node_count() {
            let grammar = PythonGrammar::new();
            let simple_count = grammar.node_count("x = 1");
            let complex_count = grammar.node_count("x = 1\ny = 2\nz = 3");
            assert!(simple_count > 0);
            assert!(complex_count > simple_count);
        }
    }
}
