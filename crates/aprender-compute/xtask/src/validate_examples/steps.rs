//! Validation step implementations for book examples

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use super::helpers::{
    extract_file_names, extract_file_stems, find_markdown_files, format_error_list,
    has_main_function, has_module_doc, validate_snake_case,
};

const TIMEOUT_SECS: u64 = 5;

/// Step 1: Verify all examples compile
pub(crate) fn step_compile_examples(project_root: &Path) -> Result<()> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--examples")
        .arg("--quiet")
        .current_dir(project_root)
        .output()
        .context("Failed to execute cargo build")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Examples failed to compile:\n{}", stderr);
    }

    Ok(())
}

/// Step 2: Run clippy on examples
pub(crate) fn step_clippy_examples(project_root: &Path) -> Result<()> {
    let output = Command::new("cargo")
        .arg("clippy")
        .arg("--examples")
        .arg("--quiet")
        .arg("--")
        .arg("-D")
        .arg("warnings")
        .current_dir(project_root)
        .output()
        .context("Failed to execute cargo clippy")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Clippy found issues:\n{}", stderr);
    }

    Ok(())
}

/// Step 3: Verify examples have module documentation (//!)
pub(crate) fn step_check_module_docs(examples: &[std::path::PathBuf]) -> Result<()> {
    let mut missing_docs = Vec::new();

    for example in examples {
        if !has_module_doc(example)? {
            missing_docs.push(example.clone());
        }
    }

    if !missing_docs.is_empty() {
        let names = extract_file_names(&missing_docs);
        bail!("Examples missing module documentation (//!):\n  {}", names.join("\n  "));
    }

    Ok(())
}

/// Step 4: Verify examples are runnable (have main function, run without panic)
pub(crate) fn step_check_runnable(
    examples: &[std::path::PathBuf],
    project_root: &Path,
) -> Result<()> {
    let mut errors = Vec::new();

    for example in examples {
        // Check if has main function
        if !has_main_function(example)? {
            errors.push(format!(
                "{}: missing main() function",
                example.file_name().unwrap_or_default().to_str().unwrap_or("unknown")
            ));
            continue;
        }

        // Try to run it with timeout
        let example_name = example
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid example filename"))?;

        match run_example_with_timeout(example_name, project_root, TIMEOUT_SECS) {
            Ok(()) => {} // Success
            Err(e) => {
                errors.push(format!("{}: {}", example_name, e));
            }
        }
    }

    if !errors.is_empty() {
        bail!("Examples not runnable:\n  {}", errors.join("\n  "));
    }

    Ok(())
}

/// Run example with timeout
fn run_example_with_timeout(
    example_name: &str,
    project_root: &Path,
    timeout_secs: u64,
) -> Result<()> {
    // Spawn the example process
    let mut child = Command::new("cargo")
        .arg("run")
        .arg("--example")
        .arg(example_name)
        .arg("--quiet")
        .current_dir(project_root)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn example")?;

    // Wait with timeout
    let timeout = Duration::from_secs(timeout_secs);
    let start = std::time::Instant::now();

    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                let mut stderr = Vec::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_end(&mut stderr);
                }
                let stderr_str = String::from_utf8_lossy(&stderr);
                bail!("exited with error: {}", stderr_str);
            }
            return Ok(());
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            bail!("timed out after {}s", timeout_secs);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Step 5: Validate book references actual examples
pub(crate) fn step_check_book_references(
    examples: &[std::path::PathBuf],
    book_dir: &Path,
) -> Result<()> {
    if !book_dir.exists() {
        // Book directory is optional
        return Ok(());
    }

    // Extract example names from paths
    let example_names: std::collections::HashSet<String> =
        extract_file_stems(examples).into_iter().collect();

    // Find all markdown files in book
    let md_files = find_markdown_files(book_dir);

    // Extract referenced examples from markdown
    let mut referenced = HashSet::new();
    let example_ref_regex =
        Regex::new(r"examples/([a-z_]+)\.rs").expect("invariant: regex pattern is valid");

    for md_file in md_files {
        let content = fs::read_to_string(&md_file)
            .with_context(|| format!("Failed to read {}", md_file.display()))?;

        for cap in example_ref_regex.captures_iter(&content) {
            if let Some(name) = cap.get(1) {
                referenced.insert(name.as_str().to_string());
            }
        }
    }

    // Check for references to non-existent examples
    let mut invalid_refs = Vec::new();
    for ref_name in &referenced {
        if !example_names.contains(ref_name) {
            invalid_refs.push(ref_name.clone());
        }
    }

    if !invalid_refs.is_empty() {
        bail!("Book references non-existent examples:\n  {}", invalid_refs.join("\n  "));
    }

    Ok(())
}

/// Step 6: Verify `snake_case` naming conventions
pub(crate) fn step_check_naming_conventions(examples: &[std::path::PathBuf]) -> Result<()> {
    let mut invalid_names = Vec::new();

    for example in examples {
        let name = example
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid filename"))?;

        if !validate_snake_case(name) {
            invalid_names.push(name.to_string());
        }
    }

    if !invalid_names.is_empty() {
        let error_msg = format_error_list(&invalid_names, "Examples not in snake_case");
        bail!("{}", error_msg);
    }

    Ok(())
}
