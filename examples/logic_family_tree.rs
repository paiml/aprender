//! TensorLogic Family Tree Demo (Section O3)
//!
//! Demonstrates neuro-symbolic reasoning with a family tree example.
//! Shows how TensorLogic can deduce grandparent relationships from parent facts.
//!
//! # Toyota Way Principles
//! - **Jidoka**: Boolean mode guarantees no hallucinations
//! - **Genchi Genbutsu**: Explicit tensor equations for auditability
//!
//! # Usage
//! ```bash
//! cargo run --example logic_family_tree
//! ```

use aprender::logic::{
    logical_join, logical_project, LogicMode, ProgramBuilder, Equation,
};

fn main() {
    println!("=== TensorLogic Family Tree Demo ===\n");

    // Define family: Alice -> Bob -> Charlie -> David
    println!("Family Tree:");
    println!("  Alice");
    println!("    └── Bob (child of Alice)");
    println!("        └── Charlie (child of Bob)");
    println!("            └── David (child of Charlie)\n");

    // Parent relation matrix (4 people: Alice=0, Bob=1, Charlie=2, David=3)
    let parent = vec![
        vec![0.0, 1.0, 0.0, 0.0], // Alice is parent of Bob
        vec![0.0, 0.0, 1.0, 0.0], // Bob is parent of Charlie
        vec![0.0, 0.0, 0.0, 1.0], // Charlie is parent of David
        vec![0.0, 0.0, 0.0, 0.0], // David has no children
    ];

    println!("Parent relation (as adjacency matrix):");
    print_matrix(&parent, &["Alice", "Bob", "Charlie", "David"]);

    // Compute grandparent = parent @ parent (matrix multiplication)
    println!("\nComputing grandparent relation (parent @ parent)...");
    let grandparent = logical_join(&parent, &parent, LogicMode::Boolean);

    println!("Grandparent relation:");
    print_matrix(&grandparent, &["Alice", "Bob", "Charlie", "David"]);

    // Interpret results
    println!("\nDeduced grandparent relationships:");
    let names = ["Alice", "Bob", "Charlie", "David"];
    for (i, row) in grandparent.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > 0.5 {
                println!("  {} is grandparent of {}", names[i], names[j]);
            }
        }
    }

    // Compute great-grandparent = grandparent @ parent
    println!("\nComputing great-grandparent relation...");
    let great_grandparent = logical_join(&grandparent, &parent, LogicMode::Boolean);

    println!("Great-grandparent relation:");
    print_matrix(&great_grandparent, &["Alice", "Bob", "Charlie", "David"]);

    println!("\nDeduced great-grandparent relationships:");
    for (i, row) in great_grandparent.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > 0.5 {
                println!("  {} is great-grandparent of {}", names[i], names[j]);
            }
        }
    }

    // Demonstrate existential projection (HasDescendant)
    println!("\n=== Existential Projection Demo ===");
    println!("Computing HasChild(X) = ∃Y: Parent(X,Y)...");

    let has_child = logical_project(&parent, 1, LogicMode::Boolean);
    println!("\nHasChild results:");
    for (i, &val) in has_child.iter().enumerate() {
        let has = if val > 0.5 { "YES" } else { "NO" };
        println!("  {} has child: {}", names[i], has);
    }

    // Demonstrate using ProgramBuilder for rule-based reasoning
    println!("\n=== Program-Based Reasoning Demo ===");

    let mut program = ProgramBuilder::new(LogicMode::Boolean)
        .add_fact("parent", parent.clone())
        .add_rule("grandparent", Equation::Join("parent".into(), "parent".into()))
        .add_rule("great_grandparent", Equation::Join("grandparent".into(), "parent".into()))
        .build();

    let results = program.forward();

    println!("Forward chaining results:");
    if let Some(gp) = results.get("grandparent") {
        println!("  grandparent relation computed: {}x{} matrix", gp.len(), gp[0].len());
    }
    if let Some(ggp) = results.get("great_grandparent") {
        println!("  great_grandparent relation computed: {}x{} matrix", ggp.len(), ggp[0].len());
    }

    // Demonstrate continuous mode for "soft" reasoning
    println!("\n=== Continuous Mode Demo ===");
    println!("Same computation but with continuous (probabilistic) values...");

    // Uncertain parent relationships (probabilities)
    let uncertain_parent = vec![
        vec![0.0, 0.9, 0.0, 0.0], // Alice is probably parent of Bob
        vec![0.0, 0.0, 0.8, 0.0], // Bob is probably parent of Charlie
        vec![0.0, 0.0, 0.0, 0.7], // Charlie might be parent of David
        vec![0.0, 0.0, 0.0, 0.0],
    ];

    let uncertain_grandparent = logical_join(&uncertain_parent, &uncertain_parent, LogicMode::Continuous);

    println!("\nUncertain grandparent probabilities:");
    for (i, row) in uncertain_grandparent.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > 0.1 {
                println!("  P({} is grandparent of {}) = {:.2}", names[i], names[j], val);
            }
        }
    }

    println!("\n=== Demo Complete ===");
    println!("TensorLogic enables differentiable logical reasoning!");
}

/// Print a matrix with row/column labels
fn print_matrix(matrix: &[Vec<f64>], labels: &[&str]) {
    // Header
    print!("        ");
    for label in labels {
        print!("{:>8}", label);
    }
    println!();

    // Rows
    for (i, row) in matrix.iter().enumerate() {
        print!("{:>8}", labels[i]);
        for &val in row {
            if val > 0.5 {
                print!("{:>8}", "1");
            } else if val > 0.0 {
                print!("{:>8.2}", val);
            } else {
                print!("{:>8}", ".");
            }
        }
        println!();
    }
}
