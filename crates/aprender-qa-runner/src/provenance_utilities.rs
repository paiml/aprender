/// Save provenance to a model directory
///
/// # Errors
///
/// Returns error if provenance cannot be serialized or written.
pub fn save_provenance(model_dir: &Path, provenance: &Provenance) -> Result<()> {
    let provenance_path = model_dir.join(".provenance.json");
    let content = serde_json::to_string_pretty(provenance)?;
    std::fs::write(provenance_path, content)?;
    Ok(())
}

/// Get apr-cli version by running `apr --version`
///
/// Returns "unknown" if command fails.
#[must_use]
pub fn get_apr_cli_version() -> String {
    std::process::Command::new("apr")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| s.split_whitespace().nth(1).map(String::from))
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}


#[cfg(test)]
#[path = "provenance_tests.rs"]
mod provenance_tests;
