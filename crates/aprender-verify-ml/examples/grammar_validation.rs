//! Grammar Validation Example
//!
//! Demonstrates how to validate code using language-specific grammars.
//!
//! Run with: cargo run --example grammar_validation

use verificar::grammar::{
    BashGrammar, CGrammar, Grammar, PythonGrammar, RuchyGrammar, TypeScriptGrammar,
};
use verificar::Language;

fn main() {
    println!("=== Verificar Grammar Validation Example ===\n");

    // Python validation
    println!("--- Python Grammar ---");
    let python = PythonGrammar::new();
    validate_and_print(&python, "def add(a, b):\n    return a + b");
    validate_and_print(&python, "class Foo:\n    pass");
    validate_and_print(&python, "if x:\n    y = 1");
    validate_and_print(&python, "def broken("); // Invalid

    // TypeScript validation
    println!("\n--- TypeScript Grammar ---");
    let typescript = TypeScriptGrammar::new();
    validate_and_print(&typescript, "let x: number = 42;");
    validate_and_print(&typescript, "interface User { name: string; age: number; }");
    validate_and_print(&typescript, "function add<T>(a: T, b: T): T { return a; }");
    validate_and_print(&typescript, "const arrow = (x: number) => x * 2;");
    validate_and_print(&typescript, "let broken = {"); // Invalid

    // C validation
    println!("\n--- C Grammar ---");
    let c = CGrammar::new();
    validate_and_print(&c, "int main() { return 0; }");
    validate_and_print(&c, "void foo(int x) { printf(\"%d\", x); }");
    validate_and_print(&c, "int broken("); // Invalid

    // Bash validation
    println!("\n--- Bash Grammar ---");
    let bash = BashGrammar::new();
    validate_and_print(&bash, "echo \"Hello, World!\"");
    validate_and_print(&bash, "for i in 1 2 3; do echo $i; done");
    validate_and_print(&bash, "if [ -f file ]; then cat file; fi");

    // Ruchy validation
    println!("\n--- Ruchy Grammar ---");
    let ruchy = RuchyGrammar::new();
    validate_and_print(&ruchy, "let x = 42");
    validate_and_print(&ruchy, "fn add(a: int, b: int) -> int { a + b }");

    // Using grammar_for factory
    println!("\n--- Using grammar_for() factory ---");
    let languages = [
        Language::Python,
        Language::TypeScript,
        Language::C,
        Language::Bash,
        Language::Ruchy,
    ];

    for lang in languages {
        let grammar = verificar::grammar::grammar_for(lang);
        println!(
            "{}: max_enumeration_depth = {}",
            lang,
            grammar.max_enumeration_depth()
        );
    }

    println!("\n=== Example Complete ===");
}

fn validate_and_print(grammar: &dyn Grammar, code: &str) {
    let valid = grammar.validate(code);
    let status = if valid { "✓" } else { "✗" };
    let preview = if code.len() > 50 {
        format!("{}...", &code[..47])
    } else {
        code.replace('\n', "\\n")
    };
    println!("  {} [{}] {}", status, grammar.language(), preview);
}
