
/// Check if decoded text looks like garbage (BUG-GGUF-001)
fn is_likely_garbage(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();

    if has_repeated_words(&words) || has_unusual_chars(text) || has_garbage_pattern(&text_lower) {
        return true;
    }

    let has_normal_words = [
        "the", "is", "are", "and", "to", "of", "in", "that", "it", "for",
    ]
    .iter()
    .any(|w| text_lower.contains(w));
    let has_numbers = text.chars().any(|c| c.is_ascii_digit());

    !has_numbers && !has_normal_words && words.len() > 2
}

/// Check if more than 50% of words are consecutive repeats.
fn has_repeated_words(words: &[&str]) -> bool {
    if words.len() <= 2 {
        return false;
    }
    let repeated = words.windows(2).filter(|w| w[0] == w[1]).count();
    repeated * 2 > words.len()
}

/// Check if more than 1/3 of characters are unusual Unicode (replacement, PUA, etc).
fn has_unusual_chars(text: &str) -> bool {
    let total = text.chars().count();
    if total == 0 {
        return false;
    }
    let unusual = text
        .chars()
        .filter(|c| {
            *c == '\u{FFFD}'
                || ('\u{E000}'..='\u{F8FF}').contains(c)
                || ('\u{20000}'..='\u{2FFFF}').contains(c)
        })
        .count();
    unusual * 3 > total
}

/// Check for known garbage output patterns.
fn has_garbage_pattern(text_lower: &str) -> bool {
    const GARBAGE_PATTERNS: &[&str] = &[
        "random random",
        "random_",
        "domain domain",
        "domainuster",
        "pandas pandas",
        "olumbia",
        "localents",
        "nunca",
        ".mult",
    ];
    GARBAGE_PATTERNS.iter().any(|p| text_lower.contains(p))
}

/// Traced inference for SafeTensors models
#[cfg(feature = "inference")]
fn run_traced_inference_safetensors(path: &Path) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::safetensors::{SafetensorsConfig, SafetensorsModel};

    println!("{}", "Format: SafeTensors (float)".cyan());
    println!();

    // Load SafeTensors
    let data = std::fs::read(path)?;
    let model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    println!("Tensors: {}", model.tensors.len());

    // Load config if available
    if let Some(config) = SafetensorsConfig::load_from_sibling(path) {
        println!("Architecture: {}", config.architecture());
        println!("  Layers: {}", config.num_hidden_layers.unwrap_or(0));
        println!("  Hidden: {}", config.hidden_size.unwrap_or(0));
        println!("  Vocab: {}", config.vocab_size.unwrap_or(0));
    } else {
        println!("{}", "No config.json found".yellow());
    }

    println!();
    println!("{}", "SafeTensors traced inference:".green().bold());
    println!("  For SafeTensors, use `apr run --trace` for full tracing.");
    println!("  SafeTensors path uses realizar's optimized inference.");

    Ok(())
}

/// Stub for SafeTensors inference when inference feature is disabled
#[cfg(not(feature = "inference"))]
fn run_traced_inference_safetensors(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference for SafeTensors models requires the 'inference' feature. Build with --features inference".to_string(),
    ))
}

/// Read and parse model metadata from an APR file.
fn read_model_metadata(path: &Path) -> Result<(String, Vec<u8>), CliError> {
    validate_path(path)?;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let format_name = validate_header(&mut reader)?;

    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(8))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut metadata_bytes = vec![0u8; metadata_size];
    reader.read_exact(&mut metadata_bytes)?;

    Ok((format_name, metadata_bytes))
}

/// Compute trace summary from layer information.
/// BUG-TRACE-001 FIX: Accept total_params from caller since weight_stats are not populated
fn compute_trace_summary(layers: &[LayerTrace], total_params: usize) -> TraceSummary {
    let all_anomalies: Vec<String> = layers.iter().flat_map(|l| l.anomalies.clone()).collect();

    TraceSummary {
        total_layers: layers.len(),
        total_parameters: total_params,
        anomaly_count: all_anomalies.len(),
        anomalies: all_anomalies,
    }
}

/// Run the trace command
#[allow(clippy::too_many_arguments)] // CLI command needs these distinct options
#[allow(clippy::fn_params_excessive_bools)] // CLI flags are naturally boolean
pub(crate) fn run(
    path: &Path,
    layer_filter: Option<&str>,
    reference: Option<&Path>,
    json_output: bool,
    verbose: bool,
    payload: bool,
    diff: bool,
    interactive: bool,
) -> Result<(), CliError> {
    if let Some(result) = handle_special_modes(path, reference, payload, diff, interactive) {
        return result;
    }

    // Detect format via Rosetta Stone dispatch
    // BUG-TRACE-001 FIX: Now returns total_params computed from tensor shapes
    let (format_name, layers, total_params) = detect_and_trace(path, layer_filter, verbose)?;
    let summary = compute_trace_summary(&layers, total_params);

    if let Some(ref_path) = reference {
        return compare_with_reference(path, ref_path, &layers, json_output);
    }

    if json_output {
        output_json(path, &format_name, &layers, &summary);
    } else {
        output_text(path, &format_name, &layers, &summary, verbose);
    }

    Ok(())
}

/// Detect format and trace layers from any supported format.
/// BUG-TRACE-001 FIX: Now returns total_params computed from tensor shapes
fn detect_and_trace(
    path: &Path,
    layer_filter: Option<&str>,
    verbose: bool,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::rosetta::FormatType;

    validate_path(path)?;

    let format = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    match format {
        FormatType::Apr => {
            let (format_name, metadata_bytes) = read_model_metadata(path)?;
            let layers = trace_layers(&metadata_bytes, layer_filter, verbose);
            // BUG-TRACE-003 FIX: Use RosettaStone to compute total_params from tensor shapes
            // Previously hardcoded to 0, now properly computed like GGUF/SafeTensors
            let rosetta = aprender::format::rosetta::RosettaStone::new();
            let total_params = rosetta
                .inspect(path)
                .map(|report| report.total_params)
                .unwrap_or(0);
            Ok((format_name, layers, total_params))
        }
        FormatType::Gguf => trace_gguf(path, layer_filter),
        FormatType::SafeTensors => trace_safetensors(path, layer_filter),
    }
}

/// Extract a u32 value from GGUF metadata (handles Uint32 and Uint64 variants).
fn gguf_meta_u32(
    metadata: &BTreeMap<String, aprender::format::gguf::GgufValue>,
    key: &str,
) -> Option<u32> {
    use aprender::format::gguf::GgufValue;
    match metadata.get(key)? {
        GgufValue::Uint32(v) => Some(*v),
        GgufValue::Uint64(v) => Some(*v as u32),
        GgufValue::Int32(v) => Some(*v as u32),
        _ => None,
    }
}

/// Trace layers from GGUF format by extracting architecture from KV metadata.
/// BUG-TRACE-001 FIX: Now computes total_params from tensor shapes
fn trace_gguf(
    path: &Path,
    layer_filter: Option<&str>,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::gguf::reader::GgufReader;
    use aprender::format::gguf::GgufValue;

    let data = std::fs::read(path)?;
    let reader = GgufReader::from_bytes(data)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse GGUF: {e}")))?;

    // BUG-TRACE-001 FIX: Compute total params from tensor dimensions
    let total_params: usize = reader
        .tensors
        .iter()
        .map(|t| t.dims.iter().map(|&d| d as usize).product::<usize>())
        .sum();

    // Extract architecture info from GGUF KV metadata
    let arch = match reader.metadata.get("general.architecture") {
        Some(GgufValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    let n_layers = gguf_meta_u32(&reader.metadata, &format!("{arch}.block_count"))
        .or_else(|| gguf_meta_u32(&reader.metadata, "general.block_count"))
        .unwrap_or(0) as usize;
    let n_embd =
        gguf_meta_u32(&reader.metadata, &format!("{arch}.embedding_length")).unwrap_or(0) as usize;

    let format_name = format!("GGUF ({arch})");

    let mut layers = vec![create_embedding_layer(n_embd)];
    layers.extend(create_transformer_layers(n_layers, layer_filter));
    layers.push(create_final_layer_norm());

    // Add tensor count info as anomaly note if verbose
    if layers.len() <= 2 && !reader.tensors.is_empty() {
        // No layers detected from metadata but tensors exist
        layers.clear();
        layers.extend(infer_layers_from_tensor_names(
            &reader
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            layer_filter,
        ));
    }

    if layers.is_empty() {
        layers.push(create_default_layer());
    }

    Ok((format_name, layers, total_params))
}

/// Trace layers from SafeTensors format by inferring architecture from tensor names.
/// BUG-TRACE-001 FIX: Now returns total_params from Rosetta inspection
fn trace_safetensors(
    path: &Path,
    layer_filter: Option<&str>,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::rosetta::RosettaStone;

    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect SafeTensors: {e}")))?;

    let format_name = "SafeTensors".to_string();
    let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut layers = infer_layers_from_tensor_names(&tensor_names, layer_filter);

    if layers.is_empty() {
        layers.push(create_default_layer());
    }

    // BUG-TRACE-001 FIX: Use total_params from Rosetta inspection
    Ok((format_name, layers, report.total_params))
}

/// Infer layer structure from tensor naming conventions.
/// Supports patterns like: `model.layers.N.*`, `encoder.layer.N.*`, `h.N.*`
fn infer_layers_from_tensor_names(
    tensor_names: &[&str],
    layer_filter: Option<&str>,
) -> Vec<LayerTrace> {
    let mut layer_indices: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut has_embedding = false;
    let mut has_lm_head = false;

    for &name in tensor_names {
        let lower = name.to_lowercase();

        if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") {
            has_embedding = true;
        }
        if lower.contains("lm_head") || lower.contains("output") {
            has_lm_head = true;
        }

        // Extract layer index from common patterns
        if let Some(idx) = extract_layer_index(name) {
            layer_indices.entry(idx).or_default().push(name.to_string());
        }
    }

    let mut layers = Vec::new();

    if has_embedding {
        maybe_push_layer(&mut layers, "embedding", None, layer_filter);
    }

    for &idx in layer_indices.keys() {
        let name = format!("transformer_block_{idx}");
        maybe_push_layer(&mut layers, &name, Some(idx), layer_filter);
    }

    if has_lm_head {
        maybe_push_layer(&mut layers, "lm_head", None, layer_filter);
    }

    layers
}

/// Push a layer trace if it passes the optional filter.
fn maybe_push_layer(
    layers: &mut Vec<LayerTrace>,
    name: &str,
    index: Option<usize>,
    filter: Option<&str>,
) {
    if filter.is_some_and(|f| !name.contains(f)) {
        return;
    }
    layers.push(LayerTrace {
        name: name.to_string(),
        index,
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec![],
    });
}

/// Extract layer index from tensor name patterns.
/// Matches: `model.layers.N.`, `encoder.layer.N.`, `h.N.`, `blk.N.`
fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns: layers.N, layer.N, h.N, blk.N, blocks.N
    let patterns = ["layers.", "layer.", "h.", "blk.", "blocks.", "block."];

    for pattern in &patterns {
        if let Some(pos) = name.find(pattern) {
            let after = &name[pos + pattern.len()..];
            let num_str: String = after.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(idx) = num_str.parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn validate_header(reader: &mut BufReader<File>) -> Result<String, CliError> {
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    // BUG-TRACE-002 FIX: Error message now mentions GGUF (matches is_valid_magic)
    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic: expected APRN, APR1, APR2, APR\\0, or GGUF, got {magic:?}"
        )));
    }

    Ok(output::format_name(&magic).to_string())
}

/// Extract layer count from hyperparameters.
fn extract_layer_count(hp: &serde_json::Map<String, serde_json::Value>) -> usize {
    hp.get("n_layer")
        .or_else(|| hp.get("n_layers"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

/// Extract model dimension from hyperparameters.
fn extract_model_dimension(hp: &serde_json::Map<String, serde_json::Value>) -> usize {
    hp.get("n_embd")
        .or_else(|| hp.get("d_model"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}
