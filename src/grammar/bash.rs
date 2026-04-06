//! Bash grammar definition
//!
//! Grammar rules for Bash/shell code generation, targeting bashrs transpilation.
//! Uses heuristic validation since tree-sitter-bash is not commonly available.

use crate::Language;

use super::Grammar;

/// Bash grammar for code generation
///
/// Provides validation for Bash shell scripts with basic syntax checking.
/// Targets bashrs transpiler for Bash-to-Rust conversion.
#[derive(Debug, Default)]
pub struct BashGrammar;

impl BashGrammar {
    /// Create a new Bash grammar
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Check if a string literal is properly quoted
    fn is_balanced_quotes(code: &str) -> bool {
        let mut in_single = false;
        let mut in_double = false;
        let mut prev_char = '\0';

        for c in code.chars() {
            match c {
                '\'' if !in_double && prev_char != '\\' => in_single = !in_single,
                '"' if !in_single && prev_char != '\\' => in_double = !in_double,
                _ => {}
            }
            prev_char = c;
        }

        !in_single && !in_double
    }

    /// Check for balanced parentheses and braces
    fn is_balanced_brackets(code: &str) -> bool {
        let mut paren_count = 0i32;
        let mut brace_count = 0i32;
        let mut bracket_count = 0i32;
        let mut in_single = false;
        let mut in_double = false;
        let mut prev_char = '\0';

        for c in code.chars() {
            // Track quote state
            match c {
                '\'' if !in_double && prev_char != '\\' => in_single = !in_single,
                '"' if !in_single && prev_char != '\\' => in_double = !in_double,
                _ => {}
            }

            // Only count brackets outside of strings
            if !in_single && !in_double {
                match c {
                    '(' => paren_count += 1,
                    ')' => paren_count -= 1,
                    '{' => brace_count += 1,
                    '}' => brace_count -= 1,
                    '[' => bracket_count += 1,
                    ']' => bracket_count -= 1,
                    _ => {}
                }

                // Early exit if count goes negative
                if paren_count < 0 || brace_count < 0 || bracket_count < 0 {
                    return false;
                }
            }
            prev_char = c;
        }

        paren_count == 0 && brace_count == 0 && bracket_count == 0
    }

    /// Check for valid shebang if present
    fn has_valid_shebang(code: &str) -> bool {
        if code.starts_with("#!") {
            // Must have a valid interpreter path
            let first_line = code.lines().next().unwrap_or("");
            first_line.contains("/bin/") || first_line.contains("/usr/bin/env")
        } else {
            true // No shebang is also valid
        }
    }
}

impl Grammar for BashGrammar {
    fn language(&self) -> Language {
        Language::Bash
    }

    fn validate(&self, code: &str) -> bool {
        if code.is_empty() {
            return false;
        }

        // Basic syntax checks
        Self::is_balanced_quotes(code)
            && Self::is_balanced_brackets(code)
            && Self::has_valid_shebang(code)
    }

    fn max_enumeration_depth(&self) -> usize {
        4 // Bash scripts tend to be simpler in structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_grammar_language() {
        let grammar = BashGrammar::new();
        assert_eq!(grammar.language(), Language::Bash);
    }

    #[test]
    fn test_bash_grammar_validate_empty() {
        let grammar = BashGrammar::new();
        assert!(!grammar.validate(""));
    }

    #[test]
    fn test_bash_grammar_validate_simple_assignment() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("x=1"));
        assert!(grammar.validate("name=\"hello\""));
        assert!(grammar.validate("arr=(1 2 3)"));
    }

    #[test]
    fn test_bash_grammar_validate_echo() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("echo \"Hello, World!\""));
        assert!(grammar.validate("echo $PATH"));
        assert!(grammar.validate("echo ${HOME}"));
    }

    #[test]
    fn test_bash_grammar_validate_if_statement() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("if [ $x -eq 1 ]; then\n    echo \"one\"\nfi"));
        assert!(grammar.validate("if [[ $x == \"hello\" ]]; then echo yes; fi"));
    }

    #[test]
    fn test_bash_grammar_validate_for_loop() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("for i in 1 2 3; do\n    echo $i\ndone"));
        assert!(grammar.validate("for f in *.txt; do cat \"$f\"; done"));
    }

    #[test]
    fn test_bash_grammar_validate_while_loop() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("while true; do\n    echo \"loop\"\ndone"));
        assert!(grammar.validate("while read line; do echo \"$line\"; done"));
    }

    #[test]
    fn test_bash_grammar_validate_function() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("function greet() {\n    echo \"Hello\"\n}"));
        assert!(grammar.validate("greet() { echo \"Hello\"; }"));
    }

    #[test]
    fn test_bash_grammar_validate_case() {
        let grammar = BashGrammar::new();
        // Note: Full case statement validation requires more sophisticated parsing
        // as `)` is part of pattern syntax. Basic balanced bracket check passes
        // when patterns use the (pattern) form
        assert!(grammar.validate("case \"$x\" in\n    (yes) echo one;;\nesac"));
    }

    #[test]
    fn test_bash_grammar_validate_shebang() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("#!/bin/bash\necho hello"));
        assert!(grammar.validate("#!/usr/bin/env bash\necho hello"));
    }

    #[test]
    fn test_bash_grammar_validate_unbalanced_quotes() {
        let grammar = BashGrammar::new();
        assert!(!grammar.validate("echo \"hello"));
        assert!(!grammar.validate("echo 'world"));
    }

    #[test]
    fn test_bash_grammar_validate_unbalanced_brackets() {
        let grammar = BashGrammar::new();
        assert!(!grammar.validate("arr=(1 2 3"));
        assert!(!grammar.validate("echo ${HOME"));
        assert!(!grammar.validate("if [ $x -eq 1")); // Missing ]
    }

    #[test]
    fn test_bash_grammar_validate_pipe() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("cat file.txt | grep pattern | wc -l"));
        assert!(grammar.validate("ls -la | head -10"));
    }

    #[test]
    fn test_bash_grammar_validate_redirection() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("echo hello > output.txt"));
        assert!(grammar.validate("cat < input.txt"));
        assert!(grammar.validate("command 2>&1"));
    }

    #[test]
    fn test_bash_grammar_validate_command_substitution() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("result=$(ls -la)"));
        assert!(grammar.validate("result=`date`"));
    }

    #[test]
    fn test_bash_grammar_validate_arithmetic() {
        let grammar = BashGrammar::new();
        assert!(grammar.validate("((x = x + 1))"));
        assert!(grammar.validate("x=$((1 + 2))"));
    }

    #[test]
    fn test_bash_grammar_max_depth() {
        let grammar = BashGrammar::new();
        assert_eq!(grammar.max_enumeration_depth(), 4);
    }

    #[test]
    fn test_bash_grammar_debug() {
        let grammar = BashGrammar::new();
        let debug = format!("{:?}", grammar);
        assert!(debug.contains("BashGrammar"));
    }

    #[test]
    fn test_bash_grammar_default() {
        let grammar = BashGrammar::default();
        assert_eq!(grammar.language(), Language::Bash);
    }

    #[test]
    fn test_balanced_quotes() {
        assert!(BashGrammar::is_balanced_quotes("\"hello\""));
        assert!(BashGrammar::is_balanced_quotes("'world'"));
        assert!(BashGrammar::is_balanced_quotes("\"it's\""));
        assert!(!BashGrammar::is_balanced_quotes("\"hello"));
        assert!(!BashGrammar::is_balanced_quotes("'world"));
    }

    #[test]
    fn test_balanced_brackets() {
        assert!(BashGrammar::is_balanced_brackets("(1 + 2)"));
        assert!(BashGrammar::is_balanced_brackets("{a; b}"));
        assert!(BashGrammar::is_balanced_brackets("[[ $x ]]"));
        assert!(!BashGrammar::is_balanced_brackets("(1 + 2"));
        assert!(!BashGrammar::is_balanced_brackets("{a; b"));
    }

    #[test]
    fn test_escaped_quotes() {
        assert!(BashGrammar::is_balanced_quotes(
            "echo \"hello \\\"world\\\"\""
        ));
        assert!(BashGrammar::is_balanced_quotes("echo 'it\\'s'"));
    }
}
