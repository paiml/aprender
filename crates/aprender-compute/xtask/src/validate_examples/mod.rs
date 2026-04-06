//! Book examples validation with EXTREME TDD quality
//!
//! This module validates that all book examples meet quality standards:
//! - Compile successfully
//! - Pass clippy lints
//! - Have module documentation
//! - Are runnable (complete with main function)
//! - Are referenced in the book
//! - Follow `snake_case` naming conventions

mod helpers;
mod results;
mod steps;

#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_extended;

use anyhow::{bail, Result};
use colored::Colorize;

use helpers::{collect_examples, get_project_root};
use results::ValidationResults;

const EXAMPLES_DIR: &str = "examples";
const BOOK_DIR: &str = "book";

/// Main entry point for example validation
pub fn run() -> Result<()> {
    println!("{}", "📚 Validating book examples...".bold());
    println!();

    let project_root = get_project_root()?;
    let examples_dir = project_root.join(EXAMPLES_DIR);
    let book_dir = project_root.join(BOOK_DIR);

    // Collect all example files
    let examples = collect_examples(&examples_dir)?;

    if examples.is_empty() {
        bail!("No examples found in {}", examples_dir.display());
    }

    println!("Found {} examples to validate", examples.len());
    println!();

    // Run validation steps
    let mut results = ValidationResults::new();

    // Step 1: Verify examples compile
    results.add_step(1, "Compile examples", || steps::step_compile_examples(&project_root));

    // Step 2: Run clippy on examples
    results.add_step(2, "Clippy lints", || steps::step_clippy_examples(&project_root));

    // Step 3: Verify module documentation
    results.add_step(3, "Module documentation", || steps::step_check_module_docs(&examples));

    // Step 4: Verify examples are runnable
    results
        .add_step(4, "Runnable examples", || steps::step_check_runnable(&examples, &project_root));

    // Step 5: Validate book references
    results
        .add_step(5, "Book references", || steps::step_check_book_references(&examples, &book_dir));

    // Step 6: Verify naming conventions
    results.add_step(6, "Naming conventions", || steps::step_check_naming_conventions(&examples));

    // Print summary
    results.print_summary();

    // Exit with error if any step failed
    if results.has_failures() {
        bail!("Validation failed");
    }

    Ok(())
}
