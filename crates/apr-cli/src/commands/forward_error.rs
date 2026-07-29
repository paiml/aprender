
/// Strip quantization suffixes from a GGUF stem to find the base model name.
fn strip_quant_suffix(stem: &str) -> &str {
    stem.trim_end_matches("-q4k")
        .trim_end_matches("-q4_k_m")
        .trim_end_matches("-q6k")
        .trim_end_matches("-q6_k")
        .trim_end_matches("-q5k")
        .trim_end_matches("-q5_k_m")
        .trim_end_matches("-q8_0")
        .trim_end_matches("-f16")
        .trim_end_matches("-f32")
}

/// True if `filename` is one of apr's OWN conversion outputs
/// (`<x>.converted.safetensors`, `<x>.converted.converted.safetensors`, …) rather
/// than an INDEPENDENT SafeTensors reference (PMAT-743).
///
/// The format-parity gate compares a GGUF forward pass against an *independent*
/// SafeTensors forward pass. A SafeTensors that was itself converted FROM the GGUF
/// under test is a circular reference — comparing a model against a derivative of
/// itself proves nothing — and in practice these artifacts are frequently stale or
/// double-converted (`.converted.converted.…`, see the qa-runner idempotency fix),
/// so converting them back fails with a confusing "conversion failed" message.
/// Discovery must skip them; genuine HF SafeTensors use `-`/`model-NNNNN-of-NNNNN`
/// naming and never contain a `.converted.` segment.
fn is_synthetic_conversion_artifact(filename: &str) -> bool {
    filename.contains(".converted.")
}

/// Strategy 2 helper: find a SafeTensors entry point in a sharded model directory.
/// Returns the first shard (sorted) for sharded models. The format parity gate
/// handles converter failures for sharded models gracefully.
fn find_sharded_safetensors(dir: &Path) -> Option<std::path::PathBuf> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return None;
    }
    // Collect all shards, sort, return first (lowest shard number = shard-00001)
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if name_str.ends_with(".safetensors")
                && name_str != "model.safetensors"
                && !is_synthetic_conversion_artifact(&name_str)
            {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards.into_iter().next()
}

/// Strategy 2: look in a sibling subdirectory (base model name without quant suffix)
/// for `model.safetensors` or a sharded safetensors file.
fn discover_sibling_subdir(parent: &Path, base_name: &str) -> Option<std::path::PathBuf> {
    let subdir = parent.join(base_name);
    if !subdir.is_dir() {
        return None;
    }
    let single = subdir.join("model.safetensors");
    if single.exists() {
        return Some(single);
    }
    find_sharded_safetensors(&subdir)
}

/// Find the SafeTensors entry point in a snapshot directory (single or sharded).
/// For sharded models, returns the first shard (sorted by name = shard-00001).
fn find_safetensors_in_snapshot(snap_path: &Path) -> Option<std::path::PathBuf> {
    let single = snap_path.join("model.safetensors");
    if single.exists() {
        return Some(single);
    }
    // Fallback: return first shard sorted (shard-00001)
    let mut shards: Vec<_> = std::fs::read_dir(snap_path)
        .ok()?
        .flatten()
        .filter_map(|f| {
            let fname = f.file_name();
            let name = fname.to_string_lossy();
            if name.ends_with(".safetensors")
                && name != "model.safetensors"
                && !is_synthetic_conversion_artifact(&name)
            {
                Some(f.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards.into_iter().next()
}

/// Check if a HF cache directory name matches the target model.
fn hf_cache_dir_matches(dir_name: &str, base_lower: &str) -> bool {
    if !dir_name.starts_with("models--") {
        return false;
    }
    let model_part = dir_name
        .trim_start_matches("models--")
        .replace("--", "/")
        .to_lowercase();
    model_part.contains(base_lower)
}

/// Search HF cache model directory snapshots for SafeTensors files.
fn search_hf_model_snapshots(model_dir: &Path) -> Option<std::path::PathBuf> {
    let snapshots = model_dir.join("snapshots");
    for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
        if let Some(found) = find_safetensors_in_snapshot(&snap.path()) {
            return Some(found);
        }
    }
    None
}

/// Strategy 3: search HuggingFace cache (`~/.cache/huggingface/hub/models--*`)
/// for a matching model directory containing safetensors files.
fn discover_hf_cache(base_name: &str) -> Option<std::path::PathBuf> {
    let hf_cache = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !hf_cache.is_dir() {
        return None;
    }
    let base_lower = base_name.to_lowercase();

    for entry in std::fs::read_dir(&hf_cache).ok()?.flatten() {
        let dir_name = entry.file_name();
        if !hf_cache_dir_matches(&dir_name.to_string_lossy(), &base_lower) {
            continue;
        }
        if let Some(found) = search_hf_model_snapshots(&entry.path()) {
            return Some(found);
        }
    }
    None
}

/// Search a single repo directory for SafeTensors files (sharded or single).
fn find_safetensors_in_repo(repo_path: &Path) -> Option<std::path::PathBuf> {
    find_sharded_safetensors(repo_path).or_else(|| {
        let single = repo_path.join("model.safetensors");
        single.exists().then_some(single)
    })
}

/// Strategy 4: search APR cache (`~/.apr/cache/hf/`) for SafeTensors files.
///
/// `apr pull` downloads sharded models to `~/.apr/cache/hf/{org}/{repo}/`.
/// This strategy searches for a repo directory whose name matches the GGUF
/// base name, then returns the first SafeTensors file found.
fn discover_apr_cache(base_name: &str) -> Option<std::path::PathBuf> {
    let apr_cache = dirs::home_dir()?.join(".apr").join("cache").join("hf");
    if !apr_cache.is_dir() {
        return None;
    }
    let base_lower = base_name.to_lowercase();
    for org_entry in std::fs::read_dir(&apr_cache).ok()?.flatten() {
        if !org_entry.path().is_dir() {
            continue;
        }
        if let Some(found) = search_org_for_model(&org_entry.path(), &base_lower) {
            return Some(found);
        }
    }
    None
}

/// Search an org directory for a repo matching `base_lower` that contains SafeTensors.
fn search_org_for_model(org_path: &Path, base_lower: &str) -> Option<std::path::PathBuf> {
    for repo_entry in std::fs::read_dir(org_path).ok()?.flatten() {
        let repo_name = repo_entry.file_name().to_string_lossy().to_lowercase();
        if repo_name.contains(base_lower) {
            if let Some(found) = find_safetensors_in_repo(&repo_entry.path()) {
                return Some(found);
            }
        }
    }
    None
}

/// Gate 5: Cross-Format Parity Test (F-QUAL-032)
///
/// Compares argmax output between GGUF and SafeTensors for the same model.
/// P0-QA-001: Auto-discover SafeTensors model for format parity gate.
///
/// Search strategy (in order):
/// 1. Sibling directory of GGUF file (same name but .safetensors)
/// 2. Sibling subdirectories containing .safetensors files
/// 3. HuggingFace cache (~/.cache/huggingface/hub/models--*)
/// 4. APR cache (~/.apr/cache/hf/) — sharded models from `apr pull` (GH-279-2)
///
/// Returns the first found SafeTensors path, or None.
fn auto_discover_safetensors(gguf_path: &Path) -> Option<std::path::PathBuf> {
    let parent = gguf_path.parent()?;
    let stem = gguf_path.file_stem()?.to_str()?;

    // Strategy 1: Sibling file with .safetensors extension
    let sibling = parent.join(format!("{stem}.safetensors"));
    if sibling.exists() {
        return Some(sibling);
    }

    // Strategy 2: Sibling subdirectory containing model.safetensors
    let base_name = strip_quant_suffix(stem);
    if let Some(found) = discover_sibling_subdir(parent, base_name) {
        return Some(found);
    }

    // Strategy 3: HuggingFace cache
    if let Some(found) = discover_hf_cache(base_name) {
        return Some(found);
    }

    // Strategy 4: APR cache (~/.apr/cache/hf/) — sharded models from `apr pull`
    discover_apr_cache(base_name)
}

/// Compute the argmax index from a slice of logits.
fn compute_argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
}

/// Minimum decode steps the cross-format parity gate must compare.
///
/// PMAT-QA-FMTPARITY-DECODE-001. The gate previously ran ONE prefill forward per
/// format and compared a single final-position argmax, so every cached-decode
/// divergence — the class that actually ships (PMAT-810 'CertainlyCertainly',
/// GH-1864) — was structurally invisible to it. 64 is the roadmap floor, chosen
/// because cache-indexing bugs typically surface tens of tokens in, not at
/// step 0.
#[cfg(feature = "inference")]
const FORMAT_PARITY_MIN_DECODE_STEPS: usize = 64;

/// Cosine similarity of two logit vectors.
///
/// Used only to exempt NEAR-TIES: when the two formats' top-1 differ but the
/// full distributions are essentially the same vector, the disagreement is
/// floating-point noise between two legitimately-close candidates rather than a
/// structural divergence. Without this the gate would flake on models with tied
/// logits while telling us nothing about cache correctness.
#[cfg(feature = "inference")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Cosine floor above which a top-1 disagreement counts as a near-tie.
#[cfg(feature = "inference")]
const FORMAT_PARITY_NEAR_TIE_COSINE: f32 = 0.98;

/// Outcome of a lockstep cross-format decode comparison.
#[cfg(feature = "inference")]
struct DecodeParityReport {
    /// Decode steps actually compared (excludes prefill).
    steps: usize,
    /// Steps whose top-1 agreed outright.
    agreed: usize,
    /// Steps whose top-1 differed but whose logit vectors were near-identical.
    near_ties: usize,
    /// First structural divergence: (step, gguf_argmax, st_argmax, cosine).
    first_divergence: Option<(usize, u32, u32, f32)>,
}

/// Greedy-decode both formats in lockstep through their PRODUCTION cache paths
/// and compare top-1 at every step.
///
/// Two design points that decide whether this proves anything:
///
/// 1. **Teacher forcing.** Both sides are fed the SAME next token (the GGUF
///    argmax) rather than each following its own. If each free-ran, the first
///    disagreement would put them on different sequences and every later step
///    would compare unrelated distributions — the comparison would degrade into
///    noise exactly when it matters most. Teacher forcing keeps all N steps on
///    one shared sequence, so a divergence at step k is attributable to step k.
///
/// 2. **The cache path, not a re-prefill.** Both sides use the same
///    single-token-plus-cache entry point production uses
///    (`forward_single_with_cache` / `forward_with_cache`), so a KV-cache
///    indexing or write bug is inside the system under test. Re-running a full
///    prefill per step would be simpler and would test the wrong thing.
#[cfg(feature = "inference")]
fn lockstep_decode_parity(
    gguf: &realizar::gguf::OwnedQuantizedModel,
    st: &realizar::ValidatedAprTransformer,
    prompt_tokens: &[u32],
    steps: usize,
) -> Result<DecodeParityReport> {
    use realizar::apr_transformer::AprKVCache;
    use realizar::gguf::OwnedQuantizedKVCache;

    let max_seq = prompt_tokens.len() + steps + 1;

    // from_config (not ::new) — it derives kv_dim as num_kv_heads * head_dim.
    // Passing hidden_dim would over-allocate and mis-stride on any GQA model,
    // and Qwen2.5-1.5B (12 query heads, 2 KV heads) is exactly that.
    let mut gguf_cache = OwnedQuantizedKVCache::from_config(gguf.config(), max_seq);
    let mut st_cache = AprKVCache::new(&st.config);

    let mut gguf_logits = Vec::new();
    let mut st_logits = Vec::new();

    // Prefill: feed the prompt one token at a time so both caches are populated
    // by the same code path that will serve the decode steps.
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        gguf_logits = gguf
            .forward_single_with_cache(tok, &mut gguf_cache, pos)
            .map_err(|e| {
                CliError::ValidationFailed(format!("GGUF prefill failed at pos {pos}: {e}"))
            })?;
        st_logits = st
            .forward_with_cache(tok, &mut st_cache, pos)
            .map_err(|e| {
                CliError::ValidationFailed(format!("SafeTensors prefill failed at pos {pos}: {e}"))
            })?;
    }

    let mut report = DecodeParityReport {
        steps: 0,
        agreed: 0,
        near_ties: 0,
        first_divergence: None,
    };

    for step in 0..steps {
        let Some(g_tok) = compute_argmax(&gguf_logits) else {
            return Err(CliError::ValidationFailed(format!(
                "GGUF produced no argmax at decode step {step}"
            )));
        };
        let Some(s_tok) = compute_argmax(&st_logits) else {
            return Err(CliError::ValidationFailed(format!(
                "SafeTensors produced no argmax at decode step {step}"
            )));
        };

        report.steps += 1;
        if g_tok == s_tok {
            report.agreed += 1;
        } else {
            let cos = cosine_similarity(&gguf_logits, &st_logits);
            if cos >= FORMAT_PARITY_NEAR_TIE_COSINE {
                report.near_ties += 1;
            } else if report.first_divergence.is_none() {
                report.first_divergence = Some((step, g_tok, s_tok, cos));
            }
        }

        // Teacher-force both sides with the same token (see doc comment).
        let pos = prompt_tokens.len() + step;
        gguf_logits = gguf
            .forward_single_with_cache(g_tok, &mut gguf_cache, pos)
            .map_err(|e| {
                CliError::ValidationFailed(format!("GGUF decode failed at step {step}: {e}"))
            })?;
        st_logits = st
            .forward_with_cache(g_tok, &mut st_cache, pos)
            .map_err(|e| {
                CliError::ValidationFailed(format!("SafeTensors decode failed at step {step}: {e}"))
            })?;
    }

    Ok(report)
}

/// Turn a lockstep decode comparison into the gate verdict.
#[cfg(feature = "inference")]
fn compare_decode_parity(report: &DecodeParityReport, duration: Duration) -> GateResult {
    let matched = report.agreed + report.near_ties;
    let total = report.steps as f64;

    if let Some((step, g_tok, s_tok, cos)) = report.first_divergence {
        return GateResult::failed(
            "format_parity",
            &format!(
                "Cross-format decode DIVERGED at step {step}/{}: GGUF argmax={g_tok} != SafeTensors argmax={s_tok} (cosine={cos:.4} < {FORMAT_PARITY_NEAR_TIE_COSINE}). \
                 {} of {} steps agreed ({} near-tie). A divergence this far into decode implicates the KV cache path, not the weights.",
                report.steps, matched, report.steps, report.near_ties
            ),
            Some(matched as f64),
            Some(total),
            duration,
        );
    }

    GateResult::passed(
        "format_parity",
        &format!(
            "Cross-format parity VERIFIED over {} greedy decode steps through the cache path \
             ({} exact top-1 matches, {} near-tie exemptions at cosine >= {FORMAT_PARITY_NEAR_TIE_COSINE})",
            report.steps, report.agreed, report.near_ties
        ),
        Some(matched as f64),
        Some(total),
        duration,
    )
}

/// Compare argmax values from GGUF and SafeTensors forward passes and produce
/// the appropriate `GateResult`.
#[allow(dead_code)]
fn compare_argmax_results(
    gguf_argmax: Option<u32>,
    st_argmax: Option<u32>,
    duration: Duration,
) -> GateResult {
    match (gguf_argmax, st_argmax) {
        (Some(gguf_token), Some(st_token)) if gguf_token == st_token => GateResult::passed(
            "format_parity",
            &format!(
                "GGUF argmax={} == SafeTensors argmax={} (Cross-format parity VERIFIED)",
                gguf_token, st_token
            ),
            Some(gguf_token as f64),
            Some(st_token as f64),
            duration,
        ),
        (Some(gguf_token), Some(st_token)) => GateResult::failed(
            "format_parity",
            &format!(
                "GGUF argmax={} != SafeTensors argmax={} (Cross-format parity BROKEN)",
                gguf_token, st_token
            ),
            Some(gguf_token as f64),
            Some(st_token as f64),
            duration,
        ),
        _ => GateResult::failed(
            "format_parity",
            "Failed to get argmax from one or both formats",
            None,
            None,
            duration,
        ),
    }
}

/// Resolve the SafeTensors path from config or auto-discovery.
/// Returns `Ok(PathBuf)` on success, or `Err(GateResult)` carrying the gate
/// outcome to short-circuit with.
///
/// PMAT-815: A genuinely ABSENT reference (no `--safetensors-path` AND nothing
/// auto-discovered) is a missing OPTIONAL input, not a format divergence — so the
/// gate SKIPs, exactly like `run_ollama_parity_gate` SKIPs when Ollama is not
/// available. A diagnostic must not hard-FAIL on the absence of the thing it
/// compares against (PMAT-743 class). The critical distinction is preserved: an
/// EXPLICIT `--safetensors-path` that does not exist still FAILs downstream (the
/// user asked for a specific reference), and a reference that IS present but whose
/// outputs diverge still FAILs — the SKIP only covers genuine absence.
fn resolve_safetensors_path(
    gguf_path: &Path,
    config: &QaConfig,
    _elapsed: Duration,
) -> std::result::Result<std::path::PathBuf, GateResult> {
    if let Some(p) = &config.safetensors_path {
        return Ok(p.clone());
    }
    match auto_discover_safetensors(gguf_path) {
        Some(p) => {
            if !config.json {
                println!(
                    "  {} Auto-discovered SafeTensors: {}",
                    "INFO".cyan(),
                    p.display()
                );
            }
            Ok(p)
        }
        None => Err(GateResult::skipped(
            "format_parity",
            "No SafeTensors reference available for parity comparison \
             (provide --safetensors-path or download: \
             huggingface-cli download <model> --include '*.safetensors')",
        )),
    }
}

/// Invariant: argmax(forward_gguf(M, tokens)) == argmax(forward_safetensors(M, tokens))
///
/// This is the cornerstone of the architecture's logical validity - it demonstrates
/// that independent binary format readers can reach the same logical conclusion.
fn run_format_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running cross-format parity test...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};

        // Peek the primary's magic bytes first (cheap — no full-file read) so that
        // non-GGUF inputs SKIP cleanly instead of churning through SafeTensors
        // resolution that is meaningless when the primary isn't the GGUF side of
        // the comparison. Matches peer gates (ollama_parity / ptx_parity /
        // capability_match) which all SKIP on non-GGUF. The P0-QA-001 "never
        // silently skip" invariant was scoped to missing-reference failures, not
        // to category-mismatched inputs.
        {
            let header = std::fs::read(path)
                .map(|b| b.into_iter().take(8).collect::<Vec<u8>>())
                .map_err(|e| {
                    CliError::ValidationFailed(format!("Failed to read primary model: {e}"))
                })?;
            let header_format = detect_format(&header).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to detect primary format: {e}"))
            })?;
            if header_format != ModelFormat::Gguf {
                return Ok(GateResult::skipped(
                    "format_parity",
                    "Non-GGUF format (format parity test compares GGUF vs SafeTensors forward passes)",
                ));
            }
        }

        // Primary is GGUF — now resolve the SafeTensors reference or FAIL with
        // an actionable message (P0-QA-001).
        let safetensors_path = match resolve_safetensors_path(path, config, start.elapsed()) {
            Ok(p) => p,
            Err(gate_result) => return Ok(gate_result),
        };

        let gguf_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read GGUF: {e}")))?;

        // Verify SafeTensors model exists
        if !safetensors_path.exists() {
            return Ok(GateResult::failed(
                "format_parity",
                &format!(
                    "SafeTensors not found: {}. Download with: huggingface-cli download <model> --include '*.safetensors'",
                    safetensors_path.display()
                ),
                None,
                None,
                start.elapsed(),
            ));
        }

        // Load GGUF model and get tokenizer
        let gguf = GGUFModel::from_bytes(&gguf_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        // Test prompt - use simple arithmetic for deterministic output
        let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
        let bos = aprender::demo::SpecialTokens::qwen2().bos_id;
        let prompt_tokens: Vec<u32> = gguf.encode(prompt).unwrap_or_else(|| vec![bos, 9707]);

        let mapped = MappedGGUFModel::from_path(path)
            .map_err(|e| CliError::ValidationFailed(format!("GGUF map failed: {e}")))?;
        let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)
            .map_err(|e| CliError::ValidationFailed(format!("GGUF model failed: {e}")))?;

        let st_model = match load_safetensors_transformer(&safetensors_path) {
            Ok(m) => m,
            Err(ForwardError::ConversionFailed(path)) => {
                return Ok(GateResult::failed(
                    "format_parity",
                    &format!("SafeTensors conversion failed: {}", path),
                    None,
                    None,
                    start.elapsed(),
                ));
            }
            Err(ForwardError::Cli(e)) => return Err(e),
        };

        // At least FORMAT_PARITY_MIN_DECODE_STEPS regardless of --max-tokens:
        // the contract's claim is about decode-depth coverage, so a smaller
        // --max-tokens must not be able to quietly shrink the gate back into the
        // single-forward shape this replaced.
        let steps = config.max_tokens.max(FORMAT_PARITY_MIN_DECODE_STEPS);
        let report = lockstep_decode_parity(&gguf_model, &st_model, &prompt_tokens, steps)?;

        Ok(compare_decode_parity(&report, start.elapsed()))
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "format_parity",
            "Requires 'inference' feature",
        ))
    }
}

/// Check if Ollama is available by pinging the API
/// Internal error type for SafeTensors forward pass (avoids early-return from parent).
#[cfg(feature = "inference")]
enum ForwardError {
    ConversionFailed(String),
    Cli(CliError),
}

/// Load the reference SafeTensors model, handling sharded and single-file layouts.
///
/// Split out of the old `run_safetensors_forward` so the caller can drive a
/// multi-step cached decode against the SAME transformer instead of paying the
/// (multi-second, multi-gigabyte) conversion once per forward pass.
#[cfg(feature = "inference")]
fn load_safetensors_transformer(
    safetensors_path: &Path,
) -> std::result::Result<realizar::ValidatedAprTransformer, ForwardError> {
    use realizar::safetensors_infer::SafetensorsToAprConverter;
    use realizar::{SafetensorsConfig, ShardedSafeTensorsModel};

    let parent_dir = safetensors_path.parent().unwrap_or(Path::new("."));
    let index_path = parent_dir.join("model.safetensors.index.json");

    let transformer = if index_path.exists() {
        let sharded = ShardedSafeTensorsModel::load_from_index(&index_path)
            .map_err(|e| ForwardError::Cli(CliError::ValidationFailed(format!("Sharded load failed: {e}"))))?;
        let config = SafetensorsConfig::load_from_sibling(safetensors_path)
            .ok_or_else(|| ForwardError::Cli(CliError::ValidationFailed("config.json not found for sharded model".to_string())))?;
        SafetensorsToAprConverter::convert_sharded(&sharded, &config)
    } else {
        SafetensorsToAprConverter::convert(safetensors_path)
    };

    match transformer {
        Ok(t) => Ok(t),
        Err(e) => {
            // PMAT-743: ANY failure to load/convert the REFERENCE SafeTensors means
            // the parity comparison cannot run — that is a graceful GATE failure with
            // an actionable message, NEVER a hard crash of `apr qa`. Previously only
            // "Tensor not found"/"not supported" were handled gracefully; other
            // errors (e.g. a corrupt reference whose down_proj is all zeros, caught
            // by F-DATA-QUALITY-001) propagated as a hard CliError and aborted the
            // whole `apr qa` run (exit 5). The diagnostic tool must survive a bad
            // reference and report it, not die. The error detail is preserved so the
            // user knows WHY (missing tensor, unsupported arch, corrupt weights, …).
            Err(ForwardError::ConversionFailed(format!(
                "{} ({e})",
                safetensors_path.display()
            )))
        }
    }
}

fn check_ollama_available() -> bool {
    // Try to connect to Ollama API
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://localhost:11434/api/tags",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
        .unwrap_or(false)
}

/// Detect model size label from a lowercased filename.
/// Returns None if no known size pattern is found.
fn detect_size_from_filename(filename_lower: &str) -> Option<&'static str> {
    // Match size patterns with boundary checks to avoid false positives from
    // random hex in temp filenames (e.g., ".tmp97bF2a1.gguf" must NOT match "7b",
    // ".tmp3bF2a1.gguf" must NOT match "3b").
    //
    // Rule: BOTH the char BEFORE and the char AFTER the pattern must be word
    // boundaries (start/end of string, '-', '_', '.', or non-alphanumeric).
    // This prevents "7b" matching in "97bF2a1" but allows it in "model7b.gguf"
    // or "model-7b-chat.gguf". Without the BEFORE check, NamedTempFile names
    // with random hex like ".tmp97bXXX" trip the file-size-heuristic test.
    const SIZE_PATTERNS: &[(&str, &str)] = &[
        ("0.5b", "0.5b"),
        ("0_5b", "0.5b"),
        ("1.5b", "1.5b"),
        ("1_5b", "1.5b"),
        ("3b", "3b"),
        ("7b", "7b"),
        ("14b", "14b"),
        ("32b", "32b"),
    ];
    SIZE_PATTERNS.iter().find_map(|(pattern, label)| {
        if let Some(pos) = filename_lower.find(pattern) {
            let bytes = filename_lower.as_bytes();
            let end = pos + pattern.len();
            // BEFORE check: rejects random hex like "97b" / "a7b1" but allows
            // "model3b" / "qwen3b". Rule: char before must NOT be an ASCII digit
            // (the size patterns 3b/7b/14b/32b are digit+letter, so a preceding
            // digit means we're inside a longer hex/numeric token, not at a
            // size-prefix boundary).
            let has_boundary_before = pos == 0 || !bytes[pos - 1].is_ascii_digit();
            // AFTER check: rejects "3bF2a1" but allows "3b.gguf" / "3b-chat".
            // Rule: char after must be end of string or non-alphanumeric.
            let has_boundary_after = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            if has_boundary_before && has_boundary_after {
                return Some(*label);
            }
        }
        None
    })
}

/// Estimate model size from file size on disk (for hash-named pacha-cached files).
/// GGUF Q4_K sizes: 0.5B~400MB, 1.5B~1GB, 3B~2GB, 7B~4.5GB
fn estimate_size_from_file(path: &Path) -> &'static str {
    match std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) {
        0..=800_000_000 => "0.5b",
        800_000_001..=2_000_000_000 => "1.5b",
        2_000_000_001..=4_000_000_000 => "3b",
        _ => "7b",
    }
}

include!("ollama.rs");
include!("forward_error_tests.rs");
