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
    let source_format = source_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    let target_format = target_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Check cache validity
    let current_hash = compute_file_hash(source_path)?;

    if target_path.exists() && cache_hash_path.exists() {
        if let Ok(cached_hash) = std::fs::read_to_string(cache_hash_path) {
            if cached_hash.trim() == current_hash {
                return Ok(FormatConversionResult {
                    source_format,
                    target_format,
                    success: true,
                    duration_ms: 0,
                    error: None,
                    cached: true,
                });
            }
        }
    }

    // Create target directory if needed
    if let Some(parent) = target_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("[JIDOKA] Failed to create target directory {}: {e}", parent.display());
        }
    }

    let start = std::time::Instant::now();

    let output = Command::new(apr_binary)
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
        })?;

    let duration_ms = start.elapsed().as_millis() as u64;

    if output.status.success() {
        // Write hash for cache validation
        if let Err(e) = std::fs::write(cache_hash_path, &current_hash) {
            eprintln!("[JIDOKA] Failed to write cache hash: {e}");
        }

        Ok(FormatConversionResult {
            source_format,
            target_format,
            success: true,
            duration_ms,
            error: None,
            cached: false,
        })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(FormatConversionResult {
            source_format,
            target_format,
            success: false,
            duration_ms,
            error: Some(stderr.to_string()),
            cached: false,
        })
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
    #[allow(clippy::similar_names)]
    pub fn check_assertions(&mut self, min_cpu: f64, min_gpu: f64) {
        // Check GGUF CPU
        if let Some(tps) = self.tps_gguf_cpu {
            let passed = tps >= min_cpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "gguf".to_string(),
                    backend: "cpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_cpu,
                    passed,
                });
            }
        }

        // Check GGUF GPU
        if let Some(tps) = self.tps_gguf_gpu {
            let passed = tps >= min_gpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "gguf".to_string(),
                    backend: "gpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_gpu,
                    passed,
                });
            }
        }

        // Check APR CPU (if measured)
        if let Some(tps) = self.tps_apr_cpu {
            let passed = tps >= min_cpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "apr".to_string(),
                    backend: "cpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_cpu,
                    passed,
                });
            }
        }

        // Check APR GPU (if measured)
        if let Some(tps) = self.tps_apr_gpu {
            let passed = tps >= min_gpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "apr".to_string(),
                    backend: "gpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_gpu,
                    passed,
                });
            }
        }

        // Check SafeTensors CPU (if measured)
        if let Some(tps) = self.tps_st_cpu {
            let passed = tps >= min_cpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "safetensors".to_string(),
                    backend: "cpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_cpu,
                    passed,
                });
            }
        }

        // Check SafeTensors GPU (if measured)
        if let Some(tps) = self.tps_st_gpu {
            let passed = tps >= min_gpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "safetensors".to_string(),
                    backend: "gpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_gpu,
                    passed,
                });
            }
        }
    }
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

    // Paths
    let gguf_dir = model_cache_dir.join("gguf");
    let apr_dir = model_cache_dir.join("apr");
    let st_dir = model_cache_dir.join("safetensors");

    // Find GGUF source file
    let gguf_path = find_model_file(&gguf_dir)?;

    // Convert GGUF → APR (with caching)
    let apr_path = apr_dir.join("model.apr");
    let apr_hash_path = apr_dir.join(".conversion_hash");
    let apr_conv = convert_format_cached(apr_binary, &gguf_path, &apr_path, &apr_hash_path)?;
    profile.conversions.push(apr_conv.clone());

    // Convert GGUF → SafeTensors (with caching) - may fail due to #190
    let st_path = st_dir.join("model.safetensors");
    let st_hash_path = st_dir.join(".conversion_hash");
    let st_conv = convert_format_cached(apr_binary, &gguf_path, &st_path, &st_hash_path)?;
    profile.conversions.push(st_conv.clone());

    // Benchmark GGUF CPU
    if let Ok(result) = run_bench_throughput(apr_binary, &gguf_path, false, warmup, iterations) {
        profile.tps_gguf_cpu = Some(result.throughput_tps);
    }

    // Benchmark GGUF GPU
    if let Ok(result) = run_bench_throughput(apr_binary, &gguf_path, true, warmup, iterations) {
        profile.tps_gguf_gpu = Some(result.throughput_tps);
    }

    // Benchmark APR CPU (only if conversion succeeded)
    if apr_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &apr_path, false, warmup, iterations) {
            profile.tps_apr_cpu = Some(result.throughput_tps);
        }
    }

    // Benchmark APR GPU (only if conversion succeeded)
    if apr_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &apr_path, true, warmup, iterations) {
            profile.tps_apr_gpu = Some(result.throughput_tps);
        }
    }

    // Benchmark SafeTensors CPU (only if conversion succeeded)
    if st_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &st_path, false, warmup, iterations) {
            profile.tps_st_cpu = Some(result.throughput_tps);
        }
    }

    // Benchmark SafeTensors GPU (only if conversion succeeded)
    if st_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &st_path, true, warmup, iterations) {
            profile.tps_st_gpu = Some(result.throughput_tps);
        }
    }

    profile.total_duration_ms = start.elapsed().as_millis() as u64;
    Ok(profile)
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

