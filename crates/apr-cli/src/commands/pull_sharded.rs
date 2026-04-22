// Sharded HuggingFace download path (extracted from `pull.rs` to
// keep the PMAT-689 file-size invariant).
//
// Inlined into `pull.rs` via `include!()`, so items keep private-to-parent
// visibility and share access to sibling helpers (`download_file`,
// `download_file_with_progress`, `format_bytes`, `fetch_safetensors_companions`,
// `convert_safetensors_formats`) that live in other `pull_*.rs` include files.

/// GH-213: Pull a sharded SafeTensors model (3B+ models with multiple shard files)
fn run_sharded(org: &str, repo: &str, shard_files: &[String], force: bool) -> Result<()> {
    println!(
        "Model: {}/{} ({} shards)",
        org.cyan(),
        repo.cyan(),
        shard_files.len().to_string().yellow()
    );

    let cache_dir = resolve_shard_cache_dir(org, repo)?;
    std::fs::create_dir_all(&cache_dir)?;

    let base_url = format!("https://huggingface.co/{org}/{repo}/resolve/main");
    let index_path = cache_dir.join("model.safetensors.index.json");

    download_index_if_needed(&base_url, &index_path, force)?;

    let manifest_path = cache_dir.join(".apr-manifest.json");
    let existing_manifest = load_existing_manifest(&manifest_path, force);

    let file_checksums = download_all_shards(
        &cache_dir,
        &base_url,
        shard_files,
        force,
        existing_manifest.as_ref(),
    )?;

    download_companion_files(&cache_dir, &base_url, force)?;
    write_shard_manifest(&manifest_path, org, repo, file_checksums)?;

    println!();
    println!("{} Downloaded successfully", "✓".green());
    println!("  Path: {}", index_path.display().to_string().green());
    println!("  Shards: {}", shard_files.len().to_string().yellow());

    convert_safetensors_formats(&index_path)?;

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", index_path.display());
    println!("  apr serve {}", index_path.display());
    Ok(())
}

/// Resolve the cache directory for a sharded model.
fn resolve_shard_cache_dir(org: &str, repo: &str) -> Result<std::path::PathBuf> {
    Ok(dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo))
}

/// Download the SafeTensors index.json if not already cached.
fn download_index_if_needed(base_url: &str, index_path: &Path, force: bool) -> Result<()> {
    if force || !index_path.exists() {
        println!();
        println!("  {} model.safetensors.index.json", "Downloading".yellow());
        download_file(
            &format!("{base_url}/model.safetensors.index.json"),
            index_path,
        )?;
    } else {
        println!("  {} model.safetensors.index.json (cached)", "✓".green());
    }
    Ok(())
}

/// Load existing shard manifest for cache-hit verification (GH-213).
fn load_existing_manifest(manifest_path: &Path, force: bool) -> Option<ShardManifest> {
    if force || !manifest_path.exists() {
        return None;
    }
    std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Download all shards, collecting checksums for the manifest.
fn download_all_shards(
    cache_dir: &Path,
    base_url: &str,
    shard_files: &[String],
    force: bool,
    existing_manifest: Option<&ShardManifest>,
) -> Result<HashMap<String, FileChecksum>> {
    let mut file_checksums: HashMap<String, FileChecksum> = HashMap::new();
    let total = shard_files.len();
    for (i, shard_file) in shard_files.iter().enumerate() {
        download_or_verify_shard(
            cache_dir,
            base_url,
            shard_file,
            i,
            total,
            force,
            existing_manifest,
            &mut file_checksums,
        )?;
    }
    Ok(file_checksums)
}

/// Download or verify a single shard file, updating the checksum map.
fn download_or_verify_shard(
    cache_dir: &Path,
    base_url: &str,
    shard_file: &str,
    index: usize,
    total: usize,
    force: bool,
    existing_manifest: Option<&ShardManifest>,
    checksums: &mut HashMap<String, FileChecksum>,
) -> Result<()> {
    let shard_path = cache_dir.join(shard_file);

    if !force && shard_path.exists() {
        if let Some(manifest) = existing_manifest {
            if let Some(expected) = manifest.files.get(shard_file) {
                let actual_size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                if actual_size == expected.size {
                    checksums.insert(
                        shard_file.to_string(),
                        FileChecksum {
                            size: expected.size,
                            blake3: expected.blake3.clone(),
                        },
                    );
                    println!(
                        "  {} [{}/{}] {} (cached, verified)",
                        "✓".green(),
                        index + 1,
                        total,
                        shard_file
                    );
                    return Ok(());
                }
                println!(
                    "  {} [{}/{}] {} (size mismatch: {} vs {} bytes, re-downloading)",
                    "⚠".yellow(),
                    index + 1,
                    total,
                    shard_file,
                    actual_size,
                    expected.size
                );
                // Fall through to re-download
            }
        } else {
            println!(
                "  {} [{}/{}] {} (cached)",
                "✓".green(),
                index + 1,
                total,
                shard_file
            );
            return Ok(());
        }
    }

    let shard_url = format!("{base_url}/{shard_file}");
    print!(
        "  {} [{}/{}] {}...",
        "↓".yellow(),
        index + 1,
        total,
        shard_file
    );
    io::stdout().flush().ok();

    let checksum = download_file_with_progress(&shard_url, &shard_path)?;
    checksums.insert(shard_file.to_string(), checksum);
    println!(" {}", "done".green());
    Ok(())
}

/// Download companion files (tokenizer, config) for sharded models.
///
/// GH-356: tokenizer.json is optional — some models only have tokenizer.model (SentencePiece)
/// or tokenizer_config.json. We validate that at least ONE tokenizer file was obtained.
fn download_companion_files(cache_dir: &Path, base_url: &str, force: bool) -> Result<()> {
    // (filename, is_required) — tokenizer files are individually optional but collectively required
    let companions = [
        ("tokenizer.json", false),
        ("config.json", true),
        ("tokenizer_config.json", false),
        ("tokenizer.model", false),
    ];
    for (filename, required) in &companions {
        let companion_path = cache_dir.join(filename);
        if !force && companion_path.exists() {
            println!("  {} {} (cached)", "✓".green(), filename);
            continue;
        }

        let url = format!("{base_url}/{filename}");
        match download_file(&url, &companion_path) {
            Ok(()) => println!("  {} {}", "✓".green(), filename),
            Err(CliError::HttpNotFound(_)) if *required => {
                return Err(CliError::ValidationFailed(format!(
                    "{filename} is required for inference but was not found (HTTP 404) at {url}"
                )));
            }
            Err(CliError::HttpNotFound(_)) => {
                println!("  {} {} (not found in repo)", "⚠".yellow(), filename);
            }
            Err(e) if *required => {
                return Err(CliError::ValidationFailed(format!(
                    "{filename} is required for inference but download failed: {e}"
                )));
            }
            Err(_) => println!("  {} {} (not available, optional)", "⚠".yellow(), filename),
        }
    }

    // GH-356: Validate at least one tokenizer file exists
    let tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"];
    let has_tokenizer = tokenizer_files.iter().any(|f| cache_dir.join(f).exists());
    if !has_tokenizer {
        return Err(CliError::ValidationFailed(format!(
            "No tokenizer found for this model. Tried: {}.\n\
             The model may require a custom tokenizer not hosted in the repository.",
            tokenizer_files.join(", ")
        )));
    }

    Ok(())
}

/// Write shard manifest with BLAKE3 checksums for integrity verification.
fn write_shard_manifest(
    manifest_path: &Path,
    org: &str,
    repo: &str,
    file_checksums: HashMap<String, FileChecksum>,
) -> Result<()> {
    if file_checksums.is_empty() {
        return Ok(());
    }
    let manifest = ShardManifest {
        version: 1,
        repo: format!("{org}/{repo}"),
        files: file_checksums,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize manifest: {e}")))?;
    std::fs::write(manifest_path, manifest_json)?;
    println!("  {} .apr-manifest.json (integrity checksums)", "✓".green());
    Ok(())
}
