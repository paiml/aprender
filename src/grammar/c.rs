//! C grammar definition
//!
//! Grammar rules for C code generation, targeting decy transpilation (C-to-Rust).
//! Uses tree-sitter-c for proper AST validation when the `tree-sitter` feature is enabled.

use crate::Language;

use super::Grammar;

/// C grammar for code generation
///
/// When the `tree-sitter` feature is enabled, uses tree-sitter-c for
/// proper syntax validation. Otherwise, falls back to basic heuristics.
pub struct CGrammar {
    #[cfg(feature = "tree-sitter")]
    parser: std::sync::Mutex<tree_sitter::Parser>,
}

impl std::fmt::Debug for CGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CGrammar").field("language", &"c").finish()
    }
}

impl Default for CGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl CGrammar {
    /// Create a new C grammar
    ///
    /// # Panics
    ///
    /// Panics if the tree-sitter C grammar fails to load (should never happen
    /// with a correctly compiled tree-sitter-c dependency).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new() -> Self {
        #[cfg(feature = "tree-sitter")]
        {
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&tree_sitter_c::LANGUAGE.into())
                .expect("Failed to load C grammar");
            Self {
                parser: std::sync::Mutex::new(parser),
            }
        }
        #[cfg(not(feature = "tree-sitter"))]
        {
            Self {}
        }
    }

    /// Parse C code and return the AST tree
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

    /// Check if code consists only of preprocessor directives
    ///
    /// tree-sitter-c may report errors for preprocessor-only code, but these
    /// are valid C fragments for our generation purposes.
    fn is_preprocessor_only(code: &str) -> bool {
        let mut has_preprocessor = false;
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Check if it's a preprocessor directive or a continuation
            if trimmed.starts_with('#') || trimmed.ends_with('\\') {
                has_preprocessor = true;
            } else {
                // Non-preprocessor, non-empty line found
                return false;
            }
        }
        has_preprocessor
    }

    /// Check for balanced braces (fallback validation)
    #[cfg(not(feature = "tree-sitter"))]
    fn is_balanced_braces(code: &str) -> bool {
        let mut brace_count = 0i32;
        let mut paren_count = 0i32;
        let mut bracket_count = 0i32;
        let mut in_string = false;
        let mut in_char = false;
        let mut in_line_comment = false;
        let mut in_block_comment = false;
        let mut prev_char = '\0';

        for c in code.chars() {
            // Handle comments
            if !in_string && !in_char {
                if prev_char == '/' && c == '/' {
                    in_line_comment = true;
                } else if prev_char == '/' && c == '*' {
                    in_block_comment = true;
                } else if in_block_comment && prev_char == '*' && c == '/' {
                    in_block_comment = false;
                    prev_char = c;
                    continue;
                } else if in_line_comment && c == '\n' {
                    in_line_comment = false;
                }
            }

            if in_line_comment || in_block_comment {
                prev_char = c;
                continue;
            }

            // Handle strings and chars
            match c {
                '"' if !in_char && prev_char != '\\' => in_string = !in_string,
                '\'' if !in_string && prev_char != '\\' => in_char = !in_char,
                _ => {}
            }

            // Only count brackets outside of strings/chars
            if !in_string && !in_char {
                match c {
                    '{' => brace_count += 1,
                    '}' => brace_count -= 1,
                    '(' => paren_count += 1,
                    ')' => paren_count -= 1,
                    '[' => bracket_count += 1,
                    ']' => bracket_count -= 1,
                    _ => {}
                }

                if brace_count < 0 || paren_count < 0 || bracket_count < 0 {
                    return false;
                }
            }
            prev_char = c;
        }

        brace_count == 0 && paren_count == 0 && bracket_count == 0
    }

    /// Check for basic C syntax patterns (fallback validation)
    #[cfg(not(feature = "tree-sitter"))]
    fn has_valid_structure(code: &str) -> bool {
        // Check for unmatched preprocessor directives
        let has_include = code.contains("#include");
        let has_define = code.contains("#define");
        let has_ifdef = code.contains("#ifdef") || code.contains("#ifndef");
        let has_endif = code.contains("#endif");

        // If we have ifdef/ifndef, we need endif
        if has_ifdef && !has_endif {
            return false;
        }

        // Basic structure check - must have some content
        let trimmed = code.trim();
        !trimmed.is_empty()
            && (has_include || has_define || trimmed.contains(';') || trimmed.contains('{'))
    }
}

impl Grammar for CGrammar {
    fn language(&self) -> Language {
        Language::C
    }

    fn validate(&self, code: &str) -> bool {
        if code.is_empty() {
            return false;
        }

        // Preprocessor-only code is always valid (tree-sitter may report errors)
        if Self::is_preprocessor_only(code) {
            return true;
        }

        #[cfg(feature = "tree-sitter")]
        {
            !self.has_errors(code)
        }

        #[cfg(not(feature = "tree-sitter"))]
        {
            // Basic fallback validation without tree-sitter
            Self::is_balanced_braces(code) && Self::has_valid_structure(code)
        }
    }

    fn max_enumeration_depth(&self) -> usize {
        6 // C programs can have deep nesting (structs, unions, nested functions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic grammar tests
    #[test]
    fn test_c_grammar_language() {
        let grammar = CGrammar::new();
        assert_eq!(grammar.language(), Language::C);
    }

    #[test]
    fn test_c_grammar_debug() {
        let grammar = CGrammar::new();
        let debug = format!("{:?}", grammar);
        assert!(debug.contains("CGrammar"));
        assert!(debug.contains("c"));
    }

    #[test]
    fn test_c_grammar_default() {
        let grammar = CGrammar::default();
        assert_eq!(grammar.language(), Language::C);
    }

    #[test]
    fn test_c_grammar_max_depth() {
        let grammar = CGrammar::new();
        assert_eq!(grammar.max_enumeration_depth(), 6);
    }

    #[test]
    fn test_c_grammar_validate_empty() {
        let grammar = CGrammar::new();
        assert!(!grammar.validate(""));
    }

    // Variable declarations
    #[test]
    fn test_c_grammar_validate_int_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x;"));
        assert!(grammar.validate("int x = 42;"));
        assert!(grammar.validate("int x, y, z;"));
    }

    #[test]
    fn test_c_grammar_validate_float_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("float f;"));
        assert!(grammar.validate("double d = 3.14;"));
        assert!(grammar.validate("long double ld;"));
    }

    #[test]
    fn test_c_grammar_validate_char_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("char c;"));
        assert!(grammar.validate("char c = 'a';"));
        assert!(grammar.validate("unsigned char uc;"));
    }

    #[test]
    fn test_c_grammar_validate_pointer_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int *p;"));
        assert!(grammar.validate("int **pp;"));
        assert!(grammar.validate("char *str = \"hello\";"));
        assert!(grammar.validate("void *ptr;"));
    }

    #[test]
    fn test_c_grammar_validate_array_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int arr[10];"));
        assert!(grammar.validate("int arr[] = {1, 2, 3};"));
        assert!(grammar.validate("char str[] = \"hello\";"));
        assert!(grammar.validate("int matrix[3][3];"));
    }

    #[test]
    fn test_c_grammar_validate_const_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("const int x = 10;"));
        assert!(grammar.validate("const char *str = \"hello\";"));
        assert!(grammar.validate("static const int SIZE = 100;"));
    }

    // Functions
    #[test]
    fn test_c_grammar_validate_function_declaration() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int main(void);"));
        assert!(grammar.validate("void foo(int x, int y);"));
        assert!(grammar.validate("char *strdup(const char *s);"));
    }

    #[test]
    fn test_c_grammar_validate_function_definition() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int main(void) { return 0; }"));
        assert!(grammar.validate("void foo() {}"));
        assert!(grammar.validate("int add(int a, int b) { return a + b; }"));
    }

    #[test]
    fn test_c_grammar_validate_function_with_body() {
        let grammar = CGrammar::new();
        let code = r#"
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_variadic_function() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int printf(const char *fmt, ...);"));
    }

    // Control flow
    #[test]
    fn test_c_grammar_validate_if_statement() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; if (x > 0) { x = 1; }"));
        assert!(grammar.validate("int x; if (x) x++;"));
    }

    #[test]
    fn test_c_grammar_validate_if_else() {
        let grammar = CGrammar::new();
        let code = "int x; if (x > 0) { x = 1; } else { x = 0; }";
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_if_else_if() {
        let grammar = CGrammar::new();
        let code = r#"
int x;
if (x > 0) {
    x = 1;
} else if (x < 0) {
    x = -1;
} else {
    x = 0;
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_for_loop() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int i; for (i = 0; i < 10; i++) {}"));
        assert!(grammar.validate("for (int i = 0; i < 10; i++) {}"));
    }

    #[test]
    fn test_c_grammar_validate_while_loop() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = 10; while (x > 0) { x--; }"));
        assert!(grammar.validate("while (1) { break; }"));
    }

    #[test]
    fn test_c_grammar_validate_do_while() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = 0; do { x++; } while (x < 10);"));
    }

    #[test]
    fn test_c_grammar_validate_switch() {
        let grammar = CGrammar::new();
        let code = r#"
int x;
switch (x) {
    case 0: break;
    case 1: break;
    default: break;
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_goto() {
        let grammar = CGrammar::new();
        let code = r#"
int main() {
    goto end;
    end:
    return 0;
}
"#;
        assert!(grammar.validate(code));
    }

    // Structs and unions
    #[test]
    fn test_c_grammar_validate_struct() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("struct Point { int x; int y; };"));
    }

    #[test]
    fn test_c_grammar_validate_struct_typedef() {
        let grammar = CGrammar::new();
        let code = r#"
typedef struct {
    int x;
    int y;
} Point;
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_nested_struct() {
        let grammar = CGrammar::new();
        let code = r#"
struct Outer {
    struct Inner {
        int value;
    } inner;
    int other;
};
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_union() {
        let grammar = CGrammar::new();
        let code = r#"
union Data {
    int i;
    float f;
    char str[20];
};
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_enum() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("enum Color { RED, GREEN, BLUE };"));
        assert!(grammar.validate("enum { A = 1, B = 2, C = 4 };"));
    }

    // Preprocessor
    #[test]
    fn test_c_grammar_validate_include() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("#include <stdio.h>"));
        assert!(grammar.validate("#include \"myheader.h\""));
    }

    #[test]
    fn test_c_grammar_validate_define() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("#define MAX 100"));
        assert!(grammar.validate("#define SQUARE(x) ((x) * (x))"));
    }

    #[test]
    fn test_c_grammar_validate_ifdef() {
        let grammar = CGrammar::new();
        let code = r#"
#ifdef DEBUG
int debug = 1;
#endif
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_ifndef_guard() {
        let grammar = CGrammar::new();
        let code = r#"
#ifndef HEADER_H
#define HEADER_H
int x;
#endif
"#;
        assert!(grammar.validate(code));
    }

    // Operators
    #[test]
    fn test_c_grammar_validate_arithmetic() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = 1 + 2;"));
        assert!(grammar.validate("int x = 10 - 5;"));
        assert!(grammar.validate("int x = 3 * 4;"));
        assert!(grammar.validate("int x = 10 / 2;"));
        assert!(grammar.validate("int x = 10 % 3;"));
    }

    #[test]
    fn test_c_grammar_validate_bitwise() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = a & b;"));
        assert!(grammar.validate("int x = a | b;"));
        assert!(grammar.validate("int x = a ^ b;"));
        assert!(grammar.validate("int x = ~a;"));
        assert!(grammar.validate("int x = a << 2;"));
        assert!(grammar.validate("int x = a >> 2;"));
    }

    #[test]
    fn test_c_grammar_validate_logical() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = a && b;"));
        assert!(grammar.validate("int x = a || b;"));
        assert!(grammar.validate("int x = !a;"));
    }

    #[test]
    fn test_c_grammar_validate_comparison() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = a == b;"));
        assert!(grammar.validate("int x = a != b;"));
        assert!(grammar.validate("int x = a < b;"));
        assert!(grammar.validate("int x = a > b;"));
        assert!(grammar.validate("int x = a <= b;"));
        assert!(grammar.validate("int x = a >= b;"));
    }

    #[test]
    fn test_c_grammar_validate_assignment() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; x = 1;"));
        assert!(grammar.validate("int x; x += 1;"));
        assert!(grammar.validate("int x; x -= 1;"));
        assert!(grammar.validate("int x; x *= 2;"));
        assert!(grammar.validate("int x; x /= 2;"));
        assert!(grammar.validate("int x; x %= 2;"));
    }

    #[test]
    fn test_c_grammar_validate_increment_decrement() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; x++;"));
        assert!(grammar.validate("int x; x--;"));
        assert!(grammar.validate("int x; ++x;"));
        assert!(grammar.validate("int x; --x;"));
    }

    #[test]
    fn test_c_grammar_validate_ternary() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = a > b ? a : b;"));
    }

    #[test]
    fn test_c_grammar_validate_sizeof() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = sizeof(int);"));
        assert!(grammar.validate("int x = sizeof(x);"));
    }

    #[test]
    fn test_c_grammar_validate_cast() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x = (int)3.14;"));
        assert!(grammar.validate("void *p; int *ip = (int *)p;"));
    }

    // Pointers and memory
    #[test]
    fn test_c_grammar_validate_address_of() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; int *p = &x;"));
    }

    #[test]
    fn test_c_grammar_validate_dereference() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int *p; int x = *p;"));
    }

    #[test]
    fn test_c_grammar_validate_struct_access() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("struct Point p; int x = p.x;"));
        assert!(grammar.validate("struct Point *p; int x = p->x;"));
    }

    #[test]
    fn test_c_grammar_validate_array_access() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int arr[10]; int x = arr[0];"));
        assert!(grammar.validate("int arr[10]; arr[5] = 42;"));
    }

    // Comments
    #[test]
    fn test_c_grammar_validate_line_comment() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; // this is a comment"));
    }

    #[test]
    fn test_c_grammar_validate_block_comment() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("int x; /* block comment */"));
        assert!(grammar.validate("/* multi\nline\ncomment */ int x;"));
    }

    // String and char literals
    #[test]
    fn test_c_grammar_validate_string_literal() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("char *s = \"hello world\";"));
        assert!(grammar.validate("char *s = \"hello\\nworld\";"));
    }

    #[test]
    fn test_c_grammar_validate_char_literal() {
        let grammar = CGrammar::new();
        assert!(grammar.validate("char c = 'a';"));
        assert!(grammar.validate("char c = '\\n';"));
        assert!(grammar.validate("char c = '\\0';"));
    }

    // Complex programs
    #[test]
    fn test_c_grammar_validate_hello_world() {
        let grammar = CGrammar::new();
        let code = r#"
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_c_grammar_validate_linked_list() {
        let grammar = CGrammar::new();
        let code = r#"
struct Node {
    int data;
    struct Node *next;
};

struct Node *create_node(int data) {
    struct Node *node = malloc(sizeof(struct Node));
    node->data = data;
    node->next = NULL;
    return node;
}
"#;
        assert!(grammar.validate(code));
    }

    // Invalid code
    #[test]
    fn test_c_grammar_validate_unbalanced_braces() {
        let grammar = CGrammar::new();
        assert!(!grammar.validate("int main() {"));
        assert!(!grammar.validate("int main() { return 0; "));
    }

    #[test]
    fn test_c_grammar_validate_unbalanced_parens() {
        let grammar = CGrammar::new();
        assert!(!grammar.validate("int x = (1 + 2;"));
        assert!(!grammar.validate("if (x > 0 { }"));
    }

    // Fallback validation tests (only run when tree-sitter is disabled)
    #[cfg(not(feature = "tree-sitter"))]
    #[test]
    fn test_balanced_braces() {
        assert!(CGrammar::is_balanced_braces("{}"));
        assert!(CGrammar::is_balanced_braces("{ { } }"));
        assert!(CGrammar::is_balanced_braces("int main() { return 0; }"));
        assert!(!CGrammar::is_balanced_braces("{"));
        assert!(!CGrammar::is_balanced_braces("}"));
        assert!(!CGrammar::is_balanced_braces("{ { }"));
    }

    #[cfg(not(feature = "tree-sitter"))]
    #[test]
    fn test_balanced_braces_with_strings() {
        assert!(CGrammar::is_balanced_braces("char *s = \"{\";"));
        assert!(CGrammar::is_balanced_braces("char c = '{';"));
    }

    #[cfg(not(feature = "tree-sitter"))]
    #[test]
    fn test_balanced_braces_with_comments() {
        assert!(CGrammar::is_balanced_braces("int x; // { not counted"));
        assert!(CGrammar::is_balanced_braces("int x; /* { */ int y;"));
    }

    #[cfg(not(feature = "tree-sitter"))]
    #[test]
    fn test_has_valid_structure() {
        assert!(CGrammar::has_valid_structure("int x;"));
        assert!(CGrammar::has_valid_structure("#include <stdio.h>"));
        assert!(CGrammar::has_valid_structure("#define MAX 100"));
        assert!(CGrammar::has_valid_structure("int main() {}"));
        assert!(!CGrammar::has_valid_structure(""));
        assert!(!CGrammar::has_valid_structure("   "));
    }

    #[test]
    fn test_is_preprocessor_only() {
        // Valid preprocessor-only code
        assert!(CGrammar::is_preprocessor_only("#include <stdio.h>"));
        assert!(CGrammar::is_preprocessor_only("#define MAX 100"));
        assert!(CGrammar::is_preprocessor_only(
            "#include <stdio.h>\n#include <stdlib.h>"
        ));
        assert!(CGrammar::is_preprocessor_only("#ifdef DEBUG\n#endif"));
        assert!(CGrammar::is_preprocessor_only(
            "#ifndef HEADER_H\n#define HEADER_H\n#endif"
        ));

        // Mixed content (not preprocessor-only)
        assert!(!CGrammar::is_preprocessor_only("int x;"));
        assert!(!CGrammar::is_preprocessor_only(
            "#include <stdio.h>\nint main() {}"
        ));
        assert!(!CGrammar::is_preprocessor_only("int x;\n#define Y 1"));

        // Empty/whitespace
        assert!(!CGrammar::is_preprocessor_only(""));
        assert!(!CGrammar::is_preprocessor_only("   "));
    }

    #[cfg(feature = "tree-sitter")]
    mod tree_sitter_tests {
        use super::*;

        #[test]
        fn test_parse_simple() {
            let grammar = CGrammar::new();
            let tree = grammar.parse("int x;");
            assert!(tree.is_some());
        }

        #[test]
        fn test_root_node() {
            let grammar = CGrammar::new();
            let root = grammar.root_node("int x;");
            assert_eq!(root, Some("translation_unit".to_string()));
        }

        #[test]
        fn test_has_errors_valid() {
            let grammar = CGrammar::new();
            assert!(!grammar.has_errors("int x;"));
            assert!(!grammar.has_errors("int main() { return 0; }"));
        }

        #[test]
        fn test_has_errors_invalid() {
            let grammar = CGrammar::new();
            assert!(grammar.has_errors("int x")); // missing semicolon
            assert!(grammar.has_errors("int main() {")); // unclosed brace
        }

        #[test]
        fn test_ast_depth() {
            let grammar = CGrammar::new();
            let simple_depth = grammar.ast_depth("int x;");
            let complex_depth = grammar.ast_depth("int main() { if (x) { return y + z; } }");
            assert!(simple_depth > 0);
            assert!(complex_depth > simple_depth);
        }

        #[test]
        fn test_node_count() {
            let grammar = CGrammar::new();
            let simple_count = grammar.node_count("int x;");
            let complex_count = grammar.node_count("int x; int y; int z;");
            assert!(simple_count > 0);
            assert!(complex_count > simple_count);
        }
    }
}
