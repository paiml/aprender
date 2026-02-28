
/// List cached models
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
pub fn list(json: bool) -> Result<()> {
    let fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let models = fetcher.list();

    // GH-248: JSON output mode
    if json {
        let models_json: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "format": m.format.name(),
                    "path": m.path.display().to_string(),
                })
            })
            .collect();
        let stats = fetcher.stats();
        let output = serde_json::json!({
            "models": models_json,
            "total": models.len(),
            "total_size_bytes": stats.total_size_bytes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return Ok(());
    }

    println!("{}", "=== Cached Models ===".cyan().bold());
    println!();

    if models.is_empty() {
        println!("{}", "No cached models found.".dimmed());
        println!();
        println!("Pull a model with:");
        println!("  apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        println!();
        println!("Or run directly (auto-downloads):");
        println!("  apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        return Ok(());
    }

    // Print header
    println!(
        "{:<40} {:<12} {:<12} {}",
        "NAME".dimmed(),
        "SIZE".dimmed(),
        "FORMAT".dimmed(),
        "PATH".dimmed()
    );
    println!("{}", "-".repeat(104).dimmed());

    for model in &models {
        let size = format_bytes(model.size_bytes);
        let format = model.format.name();
        let name = if model.name.len() > 38 {
            format!("{}...", &model.name[..35])
        } else {
            model.name.clone()
        };

        println!(
            "{:<40} {:<12} {:<12} {}",
            name.cyan(),
            size.yellow(),
            format,
            model.path.display().to_string().dimmed()
        );
    }

    println!();

    // Print stats
    let stats = fetcher.stats();
    println!(
        "Total: {} models, {} used",
        models.len(),
        format_bytes(stats.total_size_bytes)
    );

    Ok(())
}
