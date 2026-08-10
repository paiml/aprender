
/// Scan the pacha cache directory for model files that are on disk but may not be
/// tracked in the manifest (e.g., downloaded before pacha GH-162 added manifest
/// persistence, or via direct writes outside the fetcher).
///
/// Contract: apr-list-disk-reconciliation-v1 F-LIST-DISK-001 (paiml/aprender#602).
fn scan_cache_dir_for_orphans(
    cache_dir: &Path,
    known_paths: &HashSet<std::path::PathBuf>,
) -> Vec<DiskModelEntry> {
    let Ok(read_dir) = std::fs::read_dir(cache_dir) else {
        return Vec::new();
    };
    let mut orphans = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if known_paths.contains(&path) {
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        let format = match ext {
            "gguf" | "ggml" => "GGUF",
            "apr" => "APR",
            "safetensors" => "SafeTensors",
            _ => continue,
        };
        let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        orphans.push(DiskModelEntry {
            name,
            size_bytes,
            format,
            path: path.clone(),
        });
    }
    orphans
}

struct DiskModelEntry {
    name: String,
    size_bytes: u64,
    format: &'static str,
    path: std::path::PathBuf,
}

/// List cached models
///
/// Contract: apr-list-quiet-wiring-v1 F-LIST-QUIET-001 (paiml/aprender#623).
/// When quiet=true, suppress help text and tabular decoration — emit one entry per line.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
pub fn list(json: bool, quiet: bool) -> Result<()> {
    let fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let models = fetcher.list();

    // Contract: apr-list-disk-reconciliation-v1 F-LIST-DISK-001 (paiml/aprender#602).
    // pacha's manifest may be missing or stale (e.g., downloads predating GH-162's
    // save_manifest fix). Augment with a disk scan of the cache dir so orphan files
    // are visible. Manifest entries take precedence; only files not already listed
    // are added as disk orphans.
    let known_paths: HashSet<std::path::PathBuf> =
        models.iter().map(|m| m.path.clone()).collect();
    let orphans = scan_cache_dir_for_orphans(fetcher.cache_dir(), &known_paths);

    // Contract: apr-list-quiet-wiring-v1 F-LIST-QUIET-001 (paiml/aprender#623).
    // Quiet mode: one identifier per line, no decoration, no help text.
    // #2401: `emitln!` bypasses the crate-wide `--quiet` stdout gate. This IS
    // the quiet output F-LIST-QUIET-001 requires, so it must survive the gate
    // that silences ordinary reporting.
    if quiet {
        for m in &models {
            emitln!("{}", m.name);
        }
        for o in &orphans {
            emitln!("{}", o.name);
        }
        return Ok(());
    }

    // GH-248: JSON output mode
    if json {
        let mut models_json: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "format": m.format.name(),
                    "path": m.path.display().to_string(),
                    "source": "manifest",
                })
            })
            .collect();
        // Contract: apr-list-disk-reconciliation-v1 F-LIST-DISK-001 (paiml/aprender#602).
        for o in &orphans {
            models_json.push(serde_json::json!({
                "name": o.name,
                "size_bytes": o.size_bytes,
                "format": o.format,
                "path": o.path.display().to_string(),
                "source": "disk_scan",
            }));
        }
        let stats = fetcher.stats();
        let orphan_bytes: u64 = orphans.iter().map(|o| o.size_bytes).sum();
        let output = serde_json::json!({
            "models": models_json,
            "total": models.len() + orphans.len(),
            "total_size_bytes": stats.total_size_bytes + orphan_bytes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return Ok(());
    }

    println!("{}", "=== Cached Models ===".cyan().bold());
    println!();

    if models.is_empty() && orphans.is_empty() {
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

    // Contract: apr-list-disk-reconciliation-v1 F-LIST-DISK-001 (paiml/aprender#602).
    for o in &orphans {
        let size = format_bytes(o.size_bytes);
        let name = if o.name.len() > 38 {
            format!("{}...", &o.name[..35])
        } else {
            o.name.clone()
        };
        println!(
            "{:<40} {:<12} {:<12} {} {}",
            name.cyan(),
            size.yellow(),
            o.format,
            o.path.display().to_string().dimmed(),
            "(orphan)".dimmed()
        );
    }

    println!();

    // Print stats
    let stats = fetcher.stats();
    let orphan_bytes: u64 = orphans.iter().map(|o| o.size_bytes).sum();
    let total_count = models.len() + orphans.len();
    let total_bytes = stats.total_size_bytes + orphan_bytes;
    if orphans.is_empty() {
        println!("Total: {} models, {} used", total_count, format_bytes(total_bytes));
    } else {
        println!(
            "Total: {} models ({} tracked + {} orphans), {} used",
            total_count,
            models.len(),
            orphans.len(),
            format_bytes(total_bytes)
        );
    }

    Ok(())
}
