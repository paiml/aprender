//! Validation results tracking and summary formatting

use anyhow::Result;
use colored::Colorize;

/// Validation results tracker
pub struct ValidationResults {
    pub(crate) steps: Vec<StepResult>,
}

pub(crate) struct StepResult {
    pub(crate) number: usize,
    pub(crate) name: String,
    pub(crate) success: bool,
    #[allow(dead_code)]
    pub(crate) error: Option<String>,
}

impl ValidationResults {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step<F>(&mut self, number: usize, name: &str, f: F)
    where
        F: FnOnce() -> Result<()>,
    {
        print!("Step {}/6: {}... ", number, name);

        match f() {
            Ok(()) => {
                println!("{}", "✓".green());
                self.steps.push(StepResult {
                    number,
                    name: name.to_string(),
                    success: true,
                    error: None,
                });
            }
            Err(e) => {
                println!("{}", "✗".red());
                eprintln!("  {}", format!("{}", e).red());
                self.steps.push(StepResult {
                    number,
                    name: name.to_string(),
                    success: false,
                    error: Some(format!("{}", e)),
                });
            }
        }
    }

    pub fn has_failures(&self) -> bool {
        self.steps.iter().any(|s| !s.success)
    }

    pub fn print_summary(&self) {
        println!();
        println!("{}", "═══════════════════════════════════".bold());
        println!("{}", "Summary".bold());
        println!("{}", "═══════════════════════════════════".bold());

        let total = self.steps.len();
        let passed = self.steps.iter().filter(|s| s.success).count();
        let failed = total - passed;

        for step in &self.steps {
            let status = if step.success {
                "✓".green()
            } else {
                "✗".red()
            };
            println!("{} Step {}: {}", status, step.number, step.name);
        }

        println!();
        println!(
            "Total: {}, Passed: {}, Failed: {}",
            total,
            format!("{}", passed).green(),
            if failed > 0 {
                format!("{}", failed).red()
            } else {
                format!("{}", failed).green()
            }
        );

        if failed == 0 {
            println!();
            println!("{}", "✅ All validation checks passed!".green().bold());
        }
    }
}

/// Count validation errors in results
#[allow(dead_code)]
pub fn count_validation_errors(results: &ValidationResults) -> usize {
    results.steps.iter().filter(|s| !s.success).count()
}

/// Format validation summary as string
#[allow(dead_code)]
pub fn format_validation_summary(results: &ValidationResults) -> String {
    let total = results.steps.len();
    let passed = results.steps.iter().filter(|s| s.success).count();
    let failed = total - passed;
    format!("Total: {}, Passed: {}, Failed: {}", total, passed, failed)
}
