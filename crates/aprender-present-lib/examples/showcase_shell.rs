//! Shell Autocomplete Showcase Demo
//!
//! Demonstrates the REAL trained aprender-shell-base.apr model
//! using N-gram Markov chain for command prediction.
//!
//! Run with: cargo run -p presentar --example `showcase_shell`

use presentar::browser::ShellAutocomplete;

fn main() {
    let ac = ShellAutocomplete::new().expect("Failed to load model");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Shell Autocomplete Showcase Demo                   ║");
    println!("║         Using REAL trained aprender-shell-base.apr           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Model Statistics:");
    println!("  • Type:    N-gram Markov (n={})", ac.ngram_size());
    println!("  • Vocab:   {} unique commands", ac.vocab_size());
    println!("  • N-grams: {} transitions", ac.ngram_count());
    println!(
        "  • Memory:  ~{:.1} KB\n",
        ac.estimated_memory_bytes() as f64 / 1024.0
    );

    let test_cases = [
        ("git ", "Git commands"),
        ("git c", "Git 'c' prefix"),
        ("cargo ", "Cargo commands"),
        ("docker ", "Docker commands"),
        ("npm ", "NPM commands"),
        ("kubectl ", "Kubernetes"),
        ("", "Top (empty)"),
    ];

    for (prefix, description) in test_cases {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!(
            "│ Input: {:20} ({:15})          │",
            format!("\"{}\"", prefix),
            description
        );
        println!("├─────────────────────────────────────────────────────────────┤");

        let suggestions = ac.suggest(prefix, 5);
        if suggestions.is_empty() {
            println!("│   (no suggestions)                                          │");
        } else {
            for (i, (text, score)) in suggestions.iter().enumerate() {
                let display = if text.len() > 45 {
                    format!("{}...", &text[..42])
                } else {
                    text.clone()
                };
                println!("│   {}. [{:.3}] {:48}│", i + 1, score, display);
            }
        }
        println!("└─────────────────────────────────────────────────────────────┘");
        println!();
    }

    println!("JSON API (for WASM):");
    println!("  {}\n", ac.suggest_json("git ", 3));
    println!("Model Info:");
    println!("  {}", ac.model_info_json());
}
