/// Convert model format with caching
///
/// Uses `apr rosetta convert` to convert between formats.
/// Caches result and skips conversion if cache is valid.
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `source_path` - Path to source model file
/// * `target_path` - Path to target model file
/// * `cache_hash_path` - Path to store source file hash for cache validation
///
/// # Errors
///
/// Returns an error if conversion fails.
pub fn convert_format_cached(
    apr_binary: &str,
    source_path: &Path,
    target_path: &Path,
    cache_hash_path: &Path,
) -> Result<FormatConversionResult> {
    let source_format = extension_or_unknown(source_path);
    let target_format = extension_or_unknown(target_path);
    let current_hash = compute_file_hash(source_path)?;

    if is_cache_valid(target_path, cache_hash_path, &current_hash) {
        return Ok(cached_result(source_format, target_format));
    }

    ensure_parent_dir(target_path);
    let start = std::time::Instant::now();
    let output = run_convert_command(apr_binary, source_path, target_path)?;
    let duration_ms = start.elapsed().as_millis() as u64;

    Ok(build_conversion_result(
        source_format,
        target_format,
        duration_ms,
        &output,
        cache_hash_path,
        &current_hash,
    ))
}

/// Path file-extension as owned String, or `"unknown"`.
fn extension_or_unknown(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Cache is valid when the target exists and stored hash matches current source hash.
fn is_cache_valid(target_path: &Path, cache_hash_path: &Path, current_hash: &str) -> bool {
    if !target_path.exists() || !cache_hash_path.exists() {
        return false;
    }
    std::fs::read_to_string(cache_hash_path)
        .map(|s| s.trim() == current_hash)
        .unwrap_or(false)
}

/// Construct the cache-hit `FormatConversionResult`.
fn cached_result(source_format: String, target_format: String) -> FormatConversionResult {
    FormatConversionResult {
        source_format,
        target_format,
        success: true,
        duration_ms: 0,
        error: None,
        cached: true,
    }
}

/// Best-effort `mkdir -p` on the parent directory; Jidoka-logs failures but
/// does not abort (the convert command will surface any real failure).
fn ensure_parent_dir(target_path: &Path) {
    let Some(parent) = target_path.parent() else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(parent) {
        eprintln!(
            "[JIDOKA] Failed to create target directory {}: {e}",
            parent.display()
        );
    }
}

/// Spawn `apr rosetta convert SRC DST` and return its `Output`.
fn run_convert_command(
    apr_binary: &str,
    source_path: &Path,
    target_path: &Path,
) -> Result<std::process::Output> {
    Command::new(apr_binary)
        .arg("rosetta")
        .arg("convert")
        .arg(source_path)
        .arg(target_path)
        .output()
        .map_err(|e| Error::ExecutionFailed {
            command: format!(
                "apr rosetta convert {} {}",
                source_path.display(),
                target_path.display()
            ),
            reason: e.to_string(),
        })
}

/// Convert `apr rosetta convert` output into a `FormatConversionResult`,
/// writing the cache hash on success.
fn build_conversion_result(
    source_format: String,
    target_format: String,
    duration_ms: u64,
    output: &std::process::Output,
    cache_hash_path: &Path,
    current_hash: &str,
) -> FormatConversionResult {
    if output.status.success() {
        if let Err(e) = std::fs::write(cache_hash_path, current_hash) {
            eprintln!("[JIDOKA] Failed to write cache hash: {e}");
        }
        return FormatConversionResult {
            source_format,
            target_format,
            success: true,
            duration_ms,
            error: None,
            cached: false,
        };
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    FormatConversionResult {
        source_format,
        target_format,
        success: false,
        duration_ms,
        error: Some(stderr.to_string()),
        cached: false,
    }
}

// ============================================================================
// Provenance-Aware Model Preparation (PMAT-PROV-001)
// ============================================================================

/// Result of model preparation with provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreparationResult {
    /// Provenance record
    pub provenance: Provenance,
    /// Path to SafeTensors source
    pub safetensors_path: std::path::PathBuf,
    /// Path to GGUF (if conversion succeeded)
    pub gguf_path: Option<std::path::PathBuf>,
    /// Path to APR (if conversion succeeded)
    pub apr_path: Option<std::path::PathBuf>,
    /// Conversion results
    pub conversions: Vec<FormatConversionResult>,
}

/// Resolve or create provenance for a model source
///
/// Loads existing provenance if available and matching the current source hash,
/// otherwise creates a new provenance record.
fn resolve_provenance(
    safetensors_path: &Path,
    hf_repo: &str,
    output_dir: &Path,
) -> Result<Provenance> {
    match load_provenance(output_dir) {
        Ok(existing) => {
            let current_hash = crate::provenance::compute_sha256(safetensors_path)?;
            if existing.source.sha256 == current_hash {
                Ok(existing)
            } else {
                create_source_provenance(safetensors_path, hf_repo)
            }
        }
        Err(_) => create_source_provenance(safetensors_path, hf_repo),
    }
}

/// Convert a single format and track it in provenance
///
/// Returns the conversion result and, if successful, the target path.
#[allow(clippy::too_many_arguments)]
fn convert_and_track(
    apr_binary: &str,
    safetensors_path: &Path,
    target_path: &Path,
    hash_path: &Path,
    format_name: &str,
    quantization: Option<&str>,
    cli_version: &str,
    provenance: &mut Provenance,
) -> Result<(FormatConversionResult, Option<std::path::PathBuf>)> {
    let conv = convert_format_cached(apr_binary, safetensors_path, target_path, hash_path)?;
    let result_path = if conv.success {
        let already_tracked = provenance
            .derived
            .iter()
            .any(|d| d.format == format_name && d.quantization.as_deref() == quantization);
        if !already_tracked {
            add_derived(provenance, format_name, target_path, quantization, cli_version)?;
        }
        Some(target_path.to_path_buf())
    } else {
        None
    };
    Ok((conv, result_path))
}

/// Prepare a model from SafeTensors source with full provenance tracking
///
/// Implements spec 7.4 (Ground Truth Policy) and 7.5 (Provenance Validation):
/// 1. SafeTensors is the canonical source (PROV-003)
/// 2. All conversions use apr-cli (PROV-002)
/// 3. Provenance tracks all derived formats
///
/// # Errors
///
/// Returns error if any conversion fails or provenance validation fails.
pub fn prepare_model_with_provenance(
    apr_binary: &str,
    safetensors_path: &Path,
    hf_repo: &str,
    output_dir: &Path,
    quantization: Option<&str>,
) -> Result<ModelPreparationResult> {
    let mut provenance = resolve_provenance(safetensors_path, hf_repo, output_dir)?;

    // Create output directories
    std::fs::create_dir_all(output_dir)?;

    // Convert SafeTensors -> GGUF
    let gguf_target = quantization.map_or_else(
        || output_dir.join("model.gguf"),
        |q| output_dir.join(format!("model-{q}.gguf")),
    );
    let gguf_hash_path = output_dir.join(".gguf_conversion_hash");
    let cli_version = get_apr_cli_version();
    let (gguf_conv, gguf_path) = convert_and_track(
        apr_binary, safetensors_path, &gguf_target, &gguf_hash_path,
        "gguf", quantization, &cli_version, &mut provenance,
    )?;

    // Convert SafeTensors -> APR
    let apr_target = quantization.map_or_else(
        || output_dir.join("model.apr"),
        |q| output_dir.join(format!("model-{q}.apr")),
    );
    let apr_hash_path = output_dir.join(".apr_conversion_hash");
    let (apr_conv, apr_path) = convert_and_track(
        apr_binary, safetensors_path, &apr_target, &apr_hash_path,
        "apr", quantization, &cli_version, &mut provenance,
    )?;

    // Validate and save provenance
    validate_provenance(&provenance)?;
    save_provenance(output_dir, &provenance)?;

    Ok(ModelPreparationResult {
        provenance,
        safetensors_path: safetensors_path.to_path_buf(),
        gguf_path,
        apr_path,
        conversions: vec![gguf_conv, apr_conv],
    })
}

/// Verify provenance before running comparisons
///
/// Checks PROV-005 (quantization parity) for format comparison.
///
/// # Errors
///
/// Returns error if provenance is invalid or formats can't be compared.
pub fn verify_comparison_provenance(
    model_dir: &Path,
    format_a: &str,
    format_b: &str,
) -> Result<Provenance> {
    let provenance = load_provenance(model_dir)?;
    validate_provenance(&provenance)?;
    crate::provenance::validate_comparison(&provenance, format_a, format_b)?;
    Ok(provenance)
}

/// Six-column throughput profile result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SixColumnProfile {
    /// GGUF CPU throughput (tok/s)
    pub tps_gguf_cpu: Option<f64>,
    /// GGUF GPU throughput (tok/s)
    pub tps_gguf_gpu: Option<f64>,
    /// APR CPU throughput (tok/s)
    pub tps_apr_cpu: Option<f64>,
    /// APR GPU throughput (tok/s)
    pub tps_apr_gpu: Option<f64>,
    /// SafeTensors CPU throughput (tok/s)
    pub tps_st_cpu: Option<f64>,
    /// SafeTensors GPU throughput (tok/s)
    pub tps_st_gpu: Option<f64>,
    /// Conversion results
    pub conversions: Vec<FormatConversionResult>,
    /// Total profiling duration in milliseconds
    pub total_duration_ms: u64,
    /// Failed assertions (format, backend, actual, threshold)
    pub failed_assertions: Vec<ProfileAssertion>,
}

/// A profile assertion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileAssertion {
    /// Format (gguf, apr, safetensors)
    pub format: String,
    /// Backend (cpu, gpu)
    pub backend: String,
    /// Actual throughput
    pub actual_tps: f64,
    /// Minimum threshold
    pub min_threshold: f64,
    /// Whether assertion passed
    pub passed: bool,
}

impl SixColumnProfile {
    /// Check if all assertions passed.
    ///
    /// Popperian: returns false when no throughput was measured — untested ≠ passed.
    #[must_use]
    pub fn all_assertions_passed(&self) -> bool {
        let any_measured = self.tps_gguf_cpu.is_some()
            || self.tps_gguf_gpu.is_some()
            || self.tps_apr_cpu.is_some()
            || self.tps_apr_gpu.is_some()
            || self.tps_st_cpu.is_some()
            || self.tps_st_gpu.is_some();
        any_measured && self.failed_assertions.is_empty()
    }

    /// Check throughput against thresholds and record failures
    pub fn check_assertions(&mut self, min_cpu: f64, min_gpu: f64) {
        const CPU: &str = "cpu";
        const GPU: &str = "gpu";
        let checks: [(Option<f64>, &str, &str, f64); 6] = [
            (self.tps_gguf_cpu, "gguf", CPU, min_cpu),
            (self.tps_gguf_gpu, "gguf", GPU, min_gpu),
            (self.tps_apr_cpu, "apr", CPU, min_cpu),
            (self.tps_apr_gpu, "apr", GPU, min_gpu),
            (self.tps_st_cpu, "safetensors", CPU, min_cpu),
            (self.tps_st_gpu, "safetensors", GPU, min_gpu),
        ];
        for (tps, format, backend, min) in checks {
            if let Some(assertion) = check_one_assertion(tps, format, backend, min) {
                self.failed_assertions.push(assertion);
            }
        }
    }
}

/// Return `Some(assertion)` when `tps` is measured and below `min`.
fn check_one_assertion(
    tps: Option<f64>,
    format: &str,
    backend: &str,
    min: f64,
) -> Option<ProfileAssertion> {
    let actual_tps = tps?;
    if actual_tps >= min {
        return None;
    }
    Some(ProfileAssertion {
        format: format.to_string(),
        backend: backend.to_string(),
        actual_tps,
        min_threshold: min,
        passed: false,
    })
}

/// Run full 6-column profiling for a model
///
/// 1. Converts GGUF to APR and SafeTensors (with caching)
/// 2. Benchmarks each format on CPU and GPU
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `model_cache_dir` - Directory containing model format subdirs
/// * `warmup` - Warmup iterations for benchmarks
/// * `iterations` - Measurement iterations for benchmarks
///
/// # Errors
///
/// Returns an error if profiling fails.
pub fn run_six_column_profile(
    apr_binary: &str,
    model_cache_dir: &Path,
    warmup: usize,
    iterations: usize,
) -> Result<SixColumnProfile> {
    let start = std::time::Instant::now();
    let mut profile = SixColumnProfile::default();

    let gguf_dir = model_cache_dir.join("gguf");
    let apr_dir = model_cache_dir.join("apr");
    let st_dir = model_cache_dir.join("safetensors");

    let gguf_path = find_model_file(&gguf_dir)?;

    let apr_path = apr_dir.join("model.apr");
    let apr_hash_path = apr_dir.join(".conversion_hash");
    let apr_conv = convert_format_cached(apr_binary, &gguf_path, &apr_path, &apr_hash_path)?;
    profile.conversions.push(apr_conv.clone());

    // Conversion may fail upstream (#190); ST benchmarks are gated on success below.
    let st_path = st_dir.join("model.safetensors");
    let st_hash_path = st_dir.join(".conversion_hash");
    let st_conv = convert_format_cached(apr_binary, &gguf_path, &st_path, &st_hash_path)?;
    profile.conversions.push(st_conv.clone());

    bench_cpu_gpu(
        apr_binary,
        &gguf_path,
        warmup,
        iterations,
        &mut profile.tps_gguf_cpu,
        &mut profile.tps_gguf_gpu,
    );
    if apr_conv.success {
        bench_cpu_gpu(
            apr_binary,
            &apr_path,
            warmup,
            iterations,
            &mut profile.tps_apr_cpu,
            &mut profile.tps_apr_gpu,
        );
    }
    if st_conv.success {
        bench_cpu_gpu(
            apr_binary,
            &st_path,
            warmup,
            iterations,
            &mut profile.tps_st_cpu,
            &mut profile.tps_st_gpu,
        );
    }

    profile.total_duration_ms = start.elapsed().as_millis() as u64;
    Ok(profile)
}

/// Benchmark one model file on both CPU and GPU, writing measured throughput
/// into the provided slots (left `None` on failure).
fn bench_cpu_gpu(
    apr_binary: &str,
    path: &Path,
    warmup: usize,
    iterations: usize,
    cpu_slot: &mut Option<f64>,
    gpu_slot: &mut Option<f64>,
) {
    if let Ok(result) = run_bench_throughput(apr_binary, path, false, warmup, iterations) {
        *cpu_slot = Some(result.throughput_tps);
    }
    if let Ok(result) = run_bench_throughput(apr_binary, path, true, warmup, iterations) {
        *gpu_slot = Some(result.throughput_tps);
    }
}

/// Find model file in a directory
fn find_model_file(dir: &Path) -> Result<std::path::PathBuf> {
    if !dir.exists() {
        return Err(Error::ExecutionFailed {
            command: format!("find model in {}", dir.display()),
            reason: "Directory does not exist".to_string(),
        });
    }

    std::fs::read_dir(dir)
        .map_err(|e| Error::ExecutionFailed {
            command: format!("read_dir {}", dir.display()),
            reason: e.to_string(),
        })?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_file() || p.is_symlink())
        .ok_or_else(|| Error::ExecutionFailed {
            command: format!("find model in {}", dir.display()),
            reason: "No model file found".to_string(),
        })
}

