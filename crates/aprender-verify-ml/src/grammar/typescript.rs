//! TypeScript grammar definition
//!
//! Grammar rules for TypeScript code generation, targeting decy transpilation.
//! Uses tree-sitter for proper AST validation when the `tree-sitter` feature is enabled.

use crate::Language;

use super::Grammar;

/// TypeScript grammar for code generation
///
/// When the `tree-sitter` feature is enabled, uses tree-sitter-typescript for
/// proper syntax validation. Otherwise, falls back to basic heuristics.
pub struct TypeScriptGrammar {
    #[cfg(feature = "tree-sitter")]
    parser: std::sync::Mutex<tree_sitter::Parser>,
}

impl std::fmt::Debug for TypeScriptGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeScriptGrammar")
            .field("language", &"typescript")
            .finish()
    }
}

impl Default for TypeScriptGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeScriptGrammar {
    /// Create a new TypeScript grammar
    ///
    /// # Panics
    ///
    /// Panics if the tree-sitter TypeScript grammar fails to load (should never happen
    /// with a correctly compiled tree-sitter-typescript dependency).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new() -> Self {
        #[cfg(feature = "tree-sitter")]
        {
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
                .expect("Failed to load TypeScript grammar");
            Self {
                parser: std::sync::Mutex::new(parser),
            }
        }
        #[cfg(not(feature = "tree-sitter"))]
        {
            Self {}
        }
    }

    /// Parse TypeScript code and return the AST tree
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

impl Grammar for TypeScriptGrammar {
    fn language(&self) -> Language {
        Language::TypeScript
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
        5 // TypeScript ASTs get complex quickly like Python
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // RED PHASE: These tests should FAIL initially
    // ========================================

    #[test]
    fn test_typescript_grammar_language() {
        let grammar = TypeScriptGrammar::new();
        assert_eq!(grammar.language(), Language::TypeScript);
    }

    #[test]
    fn test_typescript_grammar_validate_basic() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("let x = 1;"));
        assert!(grammar.validate("const y: number = 42;"));
        assert!(!grammar.validate(""));
    }

    #[test]
    fn test_typescript_grammar_validate_function() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("function foo(): void {}"));
        assert!(grammar.validate("function add(a: number, b: number): number { return a + b; }"));
        assert!(grammar.validate("const arrow = (x: number) => x * 2;"));
    }

    #[test]
    fn test_typescript_grammar_validate_interface() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("interface Foo { x: number; }"));
        assert!(grammar.validate("interface Bar { name: string; age: number; }"));
    }

    #[test]
    fn test_typescript_grammar_validate_class() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("class Foo {}"));
        assert!(grammar.validate("class Bar { constructor(public x: number) {} }"));
        assert!(grammar.validate("class Baz extends Foo { private y: string = ''; }"));
    }

    #[test]
    fn test_typescript_grammar_validate_type_annotations() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("let x: number = 1;"));
        assert!(grammar.validate("let arr: number[] = [1, 2, 3];"));
        assert!(grammar.validate("let tuple: [string, number] = ['a', 1];"));
        assert!(grammar.validate("type MyType = string | number;"));
    }

    #[test]
    fn test_typescript_grammar_validate_generics() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("function identity<T>(x: T): T { return x; }"));
        assert!(grammar.validate("class Box<T> { value: T; }"));
        assert!(grammar.validate("let map: Map<string, number> = new Map();"));
    }

    #[test]
    fn test_typescript_grammar_validate_control_flow() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("if (x) { y = 1; }"));
        assert!(grammar.validate("for (let i = 0; i < 10; i++) { console.log(i); }"));
        assert!(grammar.validate("while (true) { break; }"));
        assert!(grammar.validate("switch (x) { case 1: break; default: break; }"));
    }

    #[test]
    fn test_typescript_grammar_validate_async() {
        let grammar = TypeScriptGrammar::new();
        assert!(grammar.validate("async function fetch(): Promise<void> {}"));
        assert!(grammar.validate("const result = await fetch();"));
    }

    #[test]
    fn test_typescript_grammar_validate_unbalanced() {
        let grammar = TypeScriptGrammar::new();
        // Unbalanced should fail
        assert!(!grammar.validate("let x = (1 + 2"));
        assert!(!grammar.validate("let x = [1, 2"));
        assert!(!grammar.validate("let x = {a: 1"));
    }

    #[test]
    fn test_typescript_grammar_max_depth() {
        let grammar = TypeScriptGrammar::new();
        assert_eq!(grammar.max_enumeration_depth(), 5);
    }

    #[test]
    fn test_typescript_grammar_debug() {
        let grammar = TypeScriptGrammar::new();
        let debug = format!("{:?}", grammar);
        assert!(debug.contains("TypeScriptGrammar"));
        assert!(debug.contains("typescript"));
    }

    #[test]
    fn test_typescript_grammar_default() {
        let grammar = TypeScriptGrammar::default();
        assert_eq!(grammar.language(), Language::TypeScript);
    }

    #[cfg(feature = "tree-sitter")]
    mod tree_sitter_tests {
        use super::*;

        #[test]
        fn test_parse_simple() {
            let grammar = TypeScriptGrammar::new();
            let tree = grammar.parse("let x = 1;");
            assert!(tree.is_some());
        }

        #[test]
        fn test_root_node() {
            let grammar = TypeScriptGrammar::new();
            let root = grammar.root_node("let x = 1;");
            assert_eq!(root, Some("program".to_string()));
        }

        #[test]
        fn test_has_errors_valid() {
            let grammar = TypeScriptGrammar::new();
            assert!(!grammar.has_errors("let x = 1;"));
            assert!(!grammar.has_errors("function foo(): void {}"));
        }

        #[test]
        fn test_has_errors_invalid() {
            let grammar = TypeScriptGrammar::new();
            assert!(grammar.has_errors("function foo("));
            assert!(grammar.has_errors("class {"));
        }

        #[test]
        fn test_ast_depth() {
            let grammar = TypeScriptGrammar::new();
            let simple_depth = grammar.ast_depth("let x = 1;");
            let complex_depth = grammar.ast_depth("function foo() { if (x) { return y + z; } }");
            assert!(simple_depth > 0);
            assert!(complex_depth > simple_depth);
        }

        #[test]
        fn test_node_count() {
            let grammar = TypeScriptGrammar::new();
            let simple_count = grammar.node_count("let x = 1;");
            let complex_count = grammar.node_count("let x = 1; let y = 2; let z = 3;");
            assert!(simple_count > 0);
            assert!(complex_count > simple_count);
        }

        #[test]
        fn test_typescript_specific_syntax() {
            let grammar = TypeScriptGrammar::new();
            // Type annotations should parse correctly
            assert!(!grammar.has_errors("let x: number = 1;"));
            assert!(
                !grammar.has_errors("function add(a: number, b: number): number { return a + b; }")
            );
            assert!(!grammar.has_errors("interface Foo { bar: string; }"));
            assert!(!grammar.has_errors("type MyType = string | number;"));
            assert!(!grammar.has_errors("enum Color { Red, Green, Blue }"));
        }
    }
}
