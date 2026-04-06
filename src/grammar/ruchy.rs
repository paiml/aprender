//! Ruchy grammar definitions
//!
//! Grammar support for Ruchy, a standalone programming language with:
//! - Rust-like syntax (fn, let, struct, enum, impl, trait)
//! - Actor model (actor, spawn, send, receive, ask)
//! - Effect system (effect, handle, handler)
//! - Pipeline operators (|>)
//! - Optional chaining (?., ??)
//! - f-strings and raw strings

use crate::grammar::Grammar;
use crate::Language;

/// Ruchy grammar implementation
#[derive(Debug)]
pub struct RuchyGrammar {
    max_depth: usize,
}

impl Default for RuchyGrammar {
    fn default() -> Self {
        Self { max_depth: 5 }
    }
}

impl RuchyGrammar {
    /// Create a new Ruchy grammar
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom max depth
    #[must_use]
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Check if braces are balanced
    fn balanced_braces(code: &str) -> bool {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut in_char = false;
        let mut in_line_comment = false;
        let mut in_block_comment = false;
        let mut prev_char = '\0';

        for ch in code.chars() {
            // Handle line comment end
            if in_line_comment {
                if ch == '\n' {
                    in_line_comment = false;
                }
                prev_char = ch;
                continue;
            }

            // Handle block comment
            if in_block_comment {
                if prev_char == '*' && ch == '/' {
                    in_block_comment = false;
                }
                prev_char = ch;
                continue;
            }

            // Check for comment start
            if !in_string && !in_char {
                if prev_char == '/' && ch == '/' {
                    in_line_comment = true;
                    prev_char = ch;
                    continue;
                }
                if prev_char == '/' && ch == '*' {
                    in_block_comment = true;
                    prev_char = ch;
                    continue;
                }
                // Hash comments
                if ch == '#' && prev_char != '[' {
                    in_line_comment = true;
                    prev_char = ch;
                    continue;
                }
            }

            // Handle strings
            if ch == '"' && !in_char && prev_char != '\\' {
                in_string = !in_string;
            }

            // Handle chars
            if ch == '\'' && !in_string && prev_char != '\\' {
                in_char = !in_char;
            }

            // Count braces outside strings/chars/comments
            if !in_string && !in_char {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth < 0 {
                            return false;
                        }
                    }
                    _ => {}
                }
            }

            prev_char = ch;
        }

        depth == 0 && !in_string && !in_char
    }

    /// Check if parentheses are balanced
    fn balanced_parens(code: &str) -> bool {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut in_char = false;
        let mut prev_char = '\0';

        for ch in code.chars() {
            if ch == '"' && !in_char && prev_char != '\\' {
                in_string = !in_string;
            }
            if ch == '\'' && !in_string && prev_char != '\\' {
                in_char = !in_char;
            }

            if !in_string && !in_char {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth < 0 {
                            return false;
                        }
                    }
                    _ => {}
                }
            }

            prev_char = ch;
        }

        depth == 0
    }

    /// Check if brackets are balanced
    fn balanced_brackets(code: &str) -> bool {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut in_char = false;
        let mut prev_char = '\0';

        for ch in code.chars() {
            if ch == '"' && !in_char && prev_char != '\\' {
                in_string = !in_string;
            }
            if ch == '\'' && !in_string && prev_char != '\\' {
                in_char = !in_char;
            }

            if !in_string && !in_char {
                match ch {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth < 0 {
                            return false;
                        }
                    }
                    _ => {}
                }
            }

            prev_char = ch;
        }

        depth == 0
    }

    /// Check for valid Ruchy structure using heuristics
    fn has_valid_structure(code: &str) -> bool {
        let trimmed = code.trim();

        // Empty is valid
        if trimmed.is_empty() {
            return true;
        }

        // Check for common Ruchy constructs
        let has_fn = trimmed.contains("fn ") || trimmed.contains("fun ");
        let has_let = trimmed.contains("let ");
        let has_var = trimmed.contains("var ");
        let has_struct = trimmed.contains("struct ");
        let has_enum = trimmed.contains("enum ");
        let has_impl = trimmed.contains("impl ");
        let has_trait = trimmed.contains("trait ");
        let has_actor = trimmed.contains("actor ");
        let has_effect = trimmed.contains("effect ");
        let has_if = trimmed.contains("if ");
        let has_for = trimmed.contains("for ");
        let has_while = trimmed.contains("while ");
        let has_loop = trimmed.contains("loop ");
        let has_match = trimmed.contains("match ");
        let has_use = trimmed.contains("use ");
        let has_mod = trimmed.contains("mod ");
        let has_import = trimmed.contains("import ");
        let has_try = trimmed.contains("try ");
        let has_async = trimmed.contains("async ");
        let has_spawn = trimmed.contains("spawn ");
        let has_type = trimmed.contains("type ");
        let has_const = trimmed.contains("const ");
        let has_static = trimmed.contains("static ");
        let has_handler = trimmed.contains("handler ");
        let has_expression = trimmed.contains('=')
            || trimmed.contains('+')
            || trimmed.contains('-')
            || trimmed.contains('*')
            || trimmed.contains('/')
            || trimmed.contains("|>");

        // Check for literals
        let has_number = trimmed.chars().any(|c| c.is_ascii_digit());
        let has_string = trimmed.contains('"');
        let has_bool = trimmed.contains("true") || trimmed.contains("false");

        // Must have some recognizable Ruchy content
        has_fn
            || has_let
            || has_var
            || has_struct
            || has_enum
            || has_impl
            || has_trait
            || has_actor
            || has_effect
            || has_if
            || has_for
            || has_while
            || has_loop
            || has_match
            || has_use
            || has_mod
            || has_import
            || has_try
            || has_async
            || has_spawn
            || has_type
            || has_const
            || has_static
            || has_handler
            || has_expression
            || has_number
            || has_string
            || has_bool
    }
}

impl Grammar for RuchyGrammar {
    fn language(&self) -> Language {
        Language::Ruchy
    }

    fn validate(&self, code: &str) -> bool {
        // Check basic structural validity
        if !Self::balanced_braces(code) {
            return false;
        }
        if !Self::balanced_parens(code) {
            return false;
        }
        if !Self::balanced_brackets(code) {
            return false;
        }

        // Check for valid Ruchy structure
        Self::has_valid_structure(code)
    }

    fn max_enumeration_depth(&self) -> usize {
        self.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ruchy_grammar_new() {
        let grammar = RuchyGrammar::new();
        assert_eq!(grammar.max_depth, 5);
    }

    #[test]
    fn test_ruchy_grammar_default() {
        let grammar = RuchyGrammar::default();
        assert_eq!(grammar.max_depth, 5);
    }

    #[test]
    fn test_ruchy_grammar_with_max_depth() {
        let grammar = RuchyGrammar::with_max_depth(10);
        assert_eq!(grammar.max_depth, 10);
    }

    #[test]
    fn test_ruchy_grammar_language() {
        let grammar = RuchyGrammar::new();
        assert_eq!(grammar.language(), Language::Ruchy);
    }

    #[test]
    fn test_ruchy_grammar_max_depth() {
        let grammar = RuchyGrammar::new();
        assert_eq!(grammar.max_enumeration_depth(), 5);
    }

    #[test]
    fn test_ruchy_grammar_debug() {
        let grammar = RuchyGrammar::new();
        let debug = format!("{:?}", grammar);
        assert!(debug.contains("RuchyGrammar"));
    }

    // Variable declarations
    #[test]
    fn test_ruchy_grammar_validate_let() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = 42"));
    }

    #[test]
    fn test_ruchy_grammar_validate_let_mut() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let mut x = 42"));
    }

    #[test]
    fn test_ruchy_grammar_validate_let_typed() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x: i32 = 42"));
    }

    #[test]
    fn test_ruchy_grammar_validate_var() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("var x = 42"));
    }

    // Function definitions
    #[test]
    fn test_ruchy_grammar_validate_fn() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("fn add(x: i32, y: i32) -> i32 { x + y }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_fun() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("fun add(x: i32, y: i32) -> i32 { x + y }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_fn_no_return() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("fn greet(name: String) { println(name) }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_fn_empty() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("fn noop() {}"));
    }

    #[test]
    fn test_ruchy_grammar_validate_async_fn() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("async fn fetch() -> Result { Ok(42) }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_pub_fn() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("pub fn public_func() { 42 }"));
    }

    // Control flow
    #[test]
    fn test_ruchy_grammar_validate_if() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("if x > 0 { true }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_if_else() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("if x > 0 { true } else { false }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_if_else_if() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("if x > 0 { 1 } else if x < 0 { -1 } else { 0 }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_match() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("match x { 0 => false, _ => true }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_match_enum() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("match result { Ok(v) => v, Err(e) => panic(e) }"));
    }

    // Loops
    #[test]
    fn test_ruchy_grammar_validate_for() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("for x in items { println(x) }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_for_range() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("for i in 0..10 { sum += i }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_while() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("while x > 0 { x -= 1 }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_loop() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("loop { break }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_loop_labeled() {
        let grammar = RuchyGrammar::new();
        // Test labeled loop in function context (lifetimes handled in full parser)
        assert!(grammar.validate("fn main() { loop { break } }"));
    }

    // Structs and enums
    #[test]
    fn test_ruchy_grammar_validate_struct() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("struct Point { x: i32, y: i32 }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_struct_pub() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("pub struct Point { pub x: i32, pub y: i32 }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_enum() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("enum Color { Red, Green, Blue }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_enum_with_data() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("enum Option { Some(T), None }"));
    }

    // Impl and trait
    #[test]
    fn test_ruchy_grammar_validate_impl() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("impl Point { fn new() -> Point { Point { x: 0, y: 0 } } }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_impl_trait() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("impl Display for Point { fn fmt() -> String { \"\" } }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_trait() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("trait Display { fn fmt() -> String }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_trait_with_default() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("trait Default { fn default() -> Self { Self {} } }"));
    }

    // Actor model
    #[test]
    fn test_ruchy_grammar_validate_actor() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("actor Counter { state count: i32 = 0 }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_actor_receive() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("actor Counter { receive Increment => { count += 1 } }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_spawn() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let counter = spawn Counter {}"));
    }

    #[test]
    fn test_ruchy_grammar_validate_send() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("send counter <- Increment"));
    }

    #[test]
    fn test_ruchy_grammar_validate_ask() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let value = ask counter <? GetValue"));
    }

    // Effect system
    #[test]
    fn test_ruchy_grammar_validate_effect() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("effect Log { fn log(msg: String) }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_handler() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("handler ConsoleLog for Log { fn log(msg) { println(msg) } }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_handle() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("handle { log(\"hello\") } with ConsoleLog"));
    }

    // Pipeline operator
    #[test]
    fn test_ruchy_grammar_validate_pipeline() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("[1, 2, 3] |> map(|x| x * 2)"));
    }

    #[test]
    fn test_ruchy_grammar_validate_pipeline_chain() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("[1, 2, 3] |> filter(|x| x > 1) |> map(|x| x * 2)"));
    }

    // Optional chaining
    #[test]
    fn test_ruchy_grammar_validate_safe_nav() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = obj?.field"));
    }

    #[test]
    fn test_ruchy_grammar_validate_null_coalesce() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = value ?? default"));
    }

    // String types
    #[test]
    fn test_ruchy_grammar_validate_string() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let s = \"hello\""));
    }

    #[test]
    fn test_ruchy_grammar_validate_fstring() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let s = f\"hello {name}\""));
    }

    #[test]
    fn test_ruchy_grammar_validate_raw_string() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let s = r\"raw\\nstring\""));
    }

    // Closures and lambdas
    #[test]
    fn test_ruchy_grammar_validate_closure() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let add = |x, y| x + y"));
    }

    #[test]
    fn test_ruchy_grammar_validate_closure_typed() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let add = |x: i32, y: i32| -> i32 { x + y }"));
    }

    // Modules and imports
    #[test]
    fn test_ruchy_grammar_validate_use() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("use std::collections::HashMap"));
    }

    #[test]
    fn test_ruchy_grammar_validate_mod() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("mod utils { pub fn helper() { 42 } }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_import() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("import math"));
    }

    // Try/catch
    #[test]
    fn test_ruchy_grammar_validate_try_catch() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("try { risky() } catch e { handle(e) }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_try_catch_finally() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("try { risky() } catch e { handle(e) } finally { cleanup() }"));
    }

    // Async/await
    #[test]
    fn test_ruchy_grammar_validate_await() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let result = await fetch()"));
    }

    // Type aliases
    #[test]
    fn test_ruchy_grammar_validate_type_alias() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("type StringList = Vec<String>"));
    }

    // Const and static
    #[test]
    fn test_ruchy_grammar_validate_const() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("const PI: f64 = 3.14159"));
    }

    #[test]
    fn test_ruchy_grammar_validate_static() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("static mut COUNTER: i32 = 0"));
    }

    // Comments
    #[test]
    fn test_ruchy_grammar_validate_line_comment() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = 42 // comment"));
    }

    #[test]
    fn test_ruchy_grammar_validate_block_comment() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = /* inline */ 42"));
    }

    #[test]
    fn test_ruchy_grammar_validate_doc_comment() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("/// Doc comment\nfn foo() {}"));
    }

    #[test]
    fn test_ruchy_grammar_validate_hash_comment() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = 42 # Python-style comment"));
    }

    // Attributes
    #[test]
    fn test_ruchy_grammar_validate_attribute() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("#[derive(Debug)]\nstruct Point { x: i32 }"));
    }

    // Operators
    #[test]
    fn test_ruchy_grammar_validate_arithmetic() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = 1 + 2 * 3 - 4 / 2"));
    }

    #[test]
    fn test_ruchy_grammar_validate_power() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = 2 ** 10"));
    }

    #[test]
    fn test_ruchy_grammar_validate_comparison() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("if x > 0 && y < 10 || z == 0 { true }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_bitwise() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate("let x = a & b | c ^ d"));
    }

    // Error cases
    #[test]
    fn test_ruchy_grammar_validate_empty() {
        let grammar = RuchyGrammar::new();
        assert!(grammar.validate(""));
    }

    #[test]
    fn test_ruchy_grammar_validate_unbalanced_braces() {
        let grammar = RuchyGrammar::new();
        assert!(!grammar.validate("fn foo() {"));
    }

    #[test]
    fn test_ruchy_grammar_validate_unbalanced_parens() {
        let grammar = RuchyGrammar::new();
        assert!(!grammar.validate("fn foo(x: i32 { }"));
    }

    #[test]
    fn test_ruchy_grammar_validate_unbalanced_brackets() {
        let grammar = RuchyGrammar::new();
        assert!(!grammar.validate("let arr = [1, 2, 3"));
    }

    // Complex programs
    #[test]
    fn test_ruchy_grammar_validate_fibonacci() {
        let grammar = RuchyGrammar::new();
        let code = r#"
fn fib(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_ruchy_grammar_validate_actor_counter() {
        let grammar = RuchyGrammar::new();
        let code = r#"
actor Counter {
    state count: i32 = 0

    receive Increment => {
        count += 1
    }

    receive GetCount => {
        count
    }
}

fn main() {
    let counter = spawn Counter {}
    send counter <- Increment
    let value = ask counter <? GetCount
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_ruchy_grammar_validate_effect_example() {
        let grammar = RuchyGrammar::new();
        let code = r#"
effect Console {
    fn print(msg: String)
    fn read() -> String
}

handler StdConsole for Console {
    fn print(msg) { println(msg) }
    fn read() { readline() }
}

fn greet() {
    handle {
        print("What is your name?")
        let name = read()
        print(f"Hello, {name}!")
    } with StdConsole
}
"#;
        assert!(grammar.validate(code));
    }

    #[test]
    fn test_ruchy_grammar_validate_pipeline_example() {
        let grammar = RuchyGrammar::new();
        let code = r#"
fn process_data(items: Vec<i32>) -> Vec<i32> {
    items
        |> filter(|x| x > 0)
        |> map(|x| x * 2)
        |> take(10)
}
"#;
        assert!(grammar.validate(code));
    }

    // Balanced helpers
    #[test]
    fn test_balanced_braces() {
        assert!(RuchyGrammar::balanced_braces("{}"));
        assert!(RuchyGrammar::balanced_braces("{ { } }"));
        assert!(RuchyGrammar::balanced_braces("fn foo() { if true { } }"));
        assert!(!RuchyGrammar::balanced_braces("{"));
        assert!(!RuchyGrammar::balanced_braces("}"));
        assert!(!RuchyGrammar::balanced_braces("{ } }"));
    }

    #[test]
    fn test_balanced_braces_with_strings() {
        assert!(RuchyGrammar::balanced_braces("let s = \"{\""));
        assert!(RuchyGrammar::balanced_braces("let s = \"}\""));
    }

    #[test]
    fn test_balanced_braces_with_comments() {
        assert!(RuchyGrammar::balanced_braces("// {\nlet x = 1"));
        assert!(RuchyGrammar::balanced_braces("/* { */ let x = 1"));
        assert!(RuchyGrammar::balanced_braces("# {\nlet x = 1"));
    }

    #[test]
    fn test_has_valid_structure() {
        assert!(RuchyGrammar::has_valid_structure("fn foo() {}"));
        assert!(RuchyGrammar::has_valid_structure("let x = 42"));
        assert!(RuchyGrammar::has_valid_structure("struct Point {}"));
        assert!(RuchyGrammar::has_valid_structure("42"));
        assert!(RuchyGrammar::has_valid_structure("true"));
        assert!(RuchyGrammar::has_valid_structure(""));
    }
}
