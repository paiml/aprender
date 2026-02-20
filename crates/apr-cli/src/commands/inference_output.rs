
/// Download sharded model files from HuggingFace (GH-127)
///
/// Parses the index.json to get list of shard files and downloads each one.
/// Returns path to the index file which can be used to locate all shards.
fn download_sharded_model(cache_dir: &Path, index_path: &Path, base_url: &str) -> Result<PathBuf> {
    // Read and parse index file
    let index_content = std::fs::read_to_string(index_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read index file: {e}")))?;

    // Parse weight_map to get unique shard filenames
    // Format: {"metadata": {...}, "weight_map": {"tensor.name": "model-00001-of-00006.safetensors", ...}}
    let shard_files: HashSet<String> = extract_shard_files(&index_content);

    if shard_files.is_empty() {
        return Err(CliError::ValidationFailed(
            "Sharded model index contains no shard files".to_string(),
        ));
    }

    let total_shards = shard_files.len();
    eprintln!("  Found {} shard files to download", total_shards);

    // Download each shard
    for (i, shard_file) in shard_files.iter().enumerate() {
        let shard_url = format!("{base_url}/{shard_file}");
        let shard_path = cache_dir.join(shard_file);

        // Skip if already cached
        if shard_path.exists() {
            eprintln!("  [{}/{}] {} (cached)", i + 1, total_shards, shard_file);
            continue;
        }

        eprintln!(
            "  [{}/{}] Downloading {}...",
            i + 1,
            total_shards,
            shard_file
        );
        download_file(&shard_url, &shard_path)?;
    }

    // Return path to index file (caller uses this to locate shards)
    Ok(index_path.to_path_buf())
}

/// Find the content of a brace-delimited section, handling nesting.
fn find_brace_content(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let content = &text[start + 1..];
    let mut depth = 1usize;
    for (i, c) in content.char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 1 => return Some(&content[..i]),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Extract a shard filename from a "key": "value" pair.
fn extract_shard_filename(kv_pair: &str) -> Option<String> {
    let colon_pos = kv_pair.rfind(':')?;
    let value = kv_pair[colon_pos + 1..].trim();
    let filename = value.trim_matches(|c: char| c == '"' || c.is_whitespace());
    if filename.ends_with(".safetensors") && !filename.is_empty() {
        Some(filename.to_string())
    } else {
        None
    }
}

/// Extract unique shard filenames from index.json weight_map
fn extract_shard_files(json: &str) -> HashSet<String> {
    let Some(weight_map_start) = json.find("\"weight_map\"") else {
        return HashSet::new();
    };
    let Some(entries) = find_brace_content(&json[weight_map_start..]) else {
        return HashSet::new();
    };
    entries
        .split(',')
        .filter_map(extract_shard_filename)
        .collect()
}

/// Download model from arbitrary URL
///
/// Caches to ~/.apr/cache/url/<hash>/<filename>
fn download_url_model(url: &str) -> Result<PathBuf> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Hash URL for cache directory
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    let url_hash = format!("{:016x}", hasher.finish());

    // Extract filename from URL or use default
    let filename = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty() && s.contains('.'))
        .unwrap_or("model.safetensors");

    let cache_dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("url")
        .join(&url_hash);

    std::fs::create_dir_all(&cache_dir)?;

    let model_path = cache_dir.join(filename);

    // Download model
    eprintln!("  Downloading {}...", filename);
    download_file(url, &model_path)?;

    eprintln!("{}", "  Download complete!".green());

    Ok(model_path)
}

/// Download a file from URL to local path
fn download_file(url: &str, path: &Path) -> Result<()> {
    use std::io::Write;

    // Use ureq for simple HTTP requests (already a dependency via hf-hub)
    let response = ureq::get(url)
        .call()
        .map_err(|e| CliError::ValidationFailed(format!("Download failed: {e}")))?;

    if response.status() != 200 {
        return Err(CliError::ValidationFailed(format!(
            "Download failed with status {}: {}",
            response.status(),
            url
        )));
    }

    let mut file = std::fs::File::create(path)?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    Ok(())
}

/// Find model file in directory
#[allow(clippy::unnecessary_wraps)] // Consistent with error-returning callers
fn find_model_in_dir(dir: &Path) -> Result<PathBuf> {
    for ext in &["apr", "safetensors", "gguf"] {
        let pattern = dir.join(format!("*.{ext}"));
        if let Some(path) = glob_first(&pattern) {
            return Ok(path);
        }
    }
    // Return directory itself if no model found
    Ok(dir.to_path_buf())
}

/// Get first match from glob pattern
fn glob_first(pattern: &Path) -> Option<PathBuf> {
    glob::glob(pattern.to_str()?).ok()?.next()?.ok()
}

/// Inference output with text and metrics
/// BUG-RUN-001 FIX: Return actual token count from inference engine
/// GH-250: Enhanced with tok_per_sec and used_gpu for JSON output
struct InferenceOutput {
    text: String,
    tokens_generated: Option<usize>,
    inference_ms: Option<f64>,
    tok_per_sec: Option<f64>,
    used_gpu: Option<bool>,
    /// GH-250: Generated token IDs for parity checking
    generated_tokens: Option<Vec<u32>>,
}

/// Execute inference on model
/// BUG-RUN-001 FIX: Now returns InferenceOutput with actual token count
fn execute_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<InferenceOutput> {
    // Check model file size for mmap decision
    let metadata = std::fs::metadata(model_path)?;
    let use_mmap = metadata.len() > 50 * 1024 * 1024; // 50MB threshold

    // F-UX-26: Only show mmap info in verbose mode (NOISY-GUARD)
    if use_mmap && options.verbose {
        eprintln!(
            "{}",
            format!("Using mmap for {}MB model", metadata.len() / 1024 / 1024).dimmed()
        );
    }

    // Try realizar inference if feature enabled
    #[cfg(feature = "inference")]
    {
        return execute_with_realizar(model_path, input_path, options, use_mmap);
    }

    // Fallback: placeholder when realizar not available
    #[cfg(not(feature = "inference"))]
    {
        let input_desc =
            input_path.map_or_else(|| "stdin".to_string(), |p| p.display().to_string());

        Ok(InferenceOutput {
            text: format!(
                "[Inference requires --features inference]\nModel: {}\nInput: {}\nFormat: {}\nGPU: {}",
                model_path.display(),
                input_desc,
                options.output_format,
                if options.no_gpu { "disabled" } else { "auto" }
            ),
            tokens_generated: None,
            inference_ms: None,
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        })
    }
}

/// Execute inference using realizar engine
///
/// Per spec APR-CLI-DELEGATE-001: All inference delegates to realizar's
/// high-level API. This eliminates ~1500 lines of duplicated code.
/// BUG-RUN-001 FIX: Now returns InferenceOutput with actual token count
#[cfg(feature = "inference")]
fn execute_with_realizar(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
    _use_mmap: bool,
) -> Result<InferenceOutput> {
    use realizar::{run_inference, InferenceConfig};

    // Get prompt from options or input file
    let prompt = if let Some(ref p) = options.prompt {
        Some(p.clone())
    } else if let Some(path) = input_path {
        Some(std::fs::read_to_string(path)?)
    } else {
        None
    };

    // Build inference config
    let mut config = InferenceConfig::new(model_path);
    if let Some(ref p) = prompt {
        config = config.with_prompt(p);
    }
    config = config
        .with_max_tokens(options.max_tokens)
        .with_verbose(options.verbose); // NOISY-GUARD F-UX-27: explicit --verbose flag

    if options.no_gpu {
        config = config.without_gpu();
    }

    if options.trace {
        config = config.with_trace(true);
    }

    // Pass trace output path if specified (PMAT-SHOWCASE-METHODOLOGY-001)
    if let Some(ref trace_path) = options.trace_output {
        config = config.with_trace_output(trace_path);
    }

    // Run inference via realizar
    let result = run_inference(&config)
        .map_err(|e| CliError::InferenceFailed(format!("Inference failed: {e}")))?;

    // Report performance if benchmarking
    if options.benchmark {
        eprintln!(
            "{}",
            format!(
                "Generated {} tokens in {:.1}ms ({:.1} tok/s)",
                result.generated_token_count, result.inference_ms, result.tok_per_sec
            )
            .green()
        );
    }

    // BUG-RUN-001 FIX: Return actual token count from realizar instead of word approximation
    // GH-250: Include tok_per_sec, GPU usage, and generated token IDs for JSON output
    let generated_tokens = if result.tokens.len() > result.input_token_count {
        Some(result.tokens[result.input_token_count..].to_vec())
    } else {
        Some(Vec::new())
    };
    Ok(InferenceOutput {
        text: result.text,
        tokens_generated: Some(result.generated_token_count),
        inference_ms: Some(result.inference_ms),
        tok_per_sec: Some(result.tok_per_sec),
        used_gpu: Some(result.used_gpu),
        generated_tokens,
    })
}
