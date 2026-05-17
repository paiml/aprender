
// ============================================================================
// Output Formatting
// ============================================================================

fn output_json(path: &Path, file_size: u64, header: &HeaderData, metadata: MetadataInfo) {
    let (v_maj, v_min) = header.version;
    // GH-249: Promote key metadata fields to top level for parity checker compatibility
    let architecture = metadata.architecture.clone();
    let num_layers = metadata.num_layers;
    let num_heads = metadata.num_heads;
    let hidden_size = metadata.hidden_size;
    let vocab_size = metadata.vocab_size;
    let result = InspectResult {
        file: path.display().to_string(),
        valid: true,
        format: "APR v2".to_string(),
        version: format!("{v_maj}.{v_min}"),
        tensor_count: header.tensor_count,
        size_bytes: file_size,
        checksum_valid: header.checksum_valid,
        architecture,
        num_layers,
        num_heads,
        hidden_size,
        vocab_size,
        flags: flags_from_header(header),
        metadata,
    };
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

/// PMAT-690 P3-A: JSON output with optional quality block.
///
/// Per SPEC §84 P3-A / AC-SHIP2-007, ship-ready models MUST score ≥ 90.
/// The score is a transparent sum of weighted sub-scores (see
/// `compute_quality_score` below). When `--quality` is absent, output
/// is unchanged. When present, a `quality` block is appended with the
/// numeric score plus the sub-score breakdown so operators can see
/// exactly which dimension dragged the score down.
fn output_json_with_quality(
    path: &Path,
    file_size: u64,
    header: &HeaderData,
    metadata: MetadataInfo,
    show_quality: bool,
) {
    if !show_quality {
        return output_json(path, file_size, header, metadata);
    }

    let quality = compute_quality_score(&metadata, header);
    let (v_maj, v_min) = header.version;
    let architecture = metadata.architecture.clone();
    let num_layers = metadata.num_layers;
    let num_heads = metadata.num_heads;
    let hidden_size = metadata.hidden_size;
    let vocab_size = metadata.vocab_size;
    let result = InspectResult {
        file: path.display().to_string(),
        valid: true,
        format: "APR v2".to_string(),
        version: format!("{v_maj}.{v_min}"),
        tensor_count: header.tensor_count,
        size_bytes: file_size,
        checksum_valid: header.checksum_valid,
        architecture,
        num_layers,
        num_heads,
        hidden_size,
        vocab_size,
        flags: flags_from_header(header),
        metadata,
    };
    if let Ok(mut json) = serde_json::to_value(&result) {
        if let Some(obj) = json.as_object_mut() {
            obj.insert("quality".to_string(), quality.to_json());
        }
        if let Ok(pretty) = serde_json::to_string_pretty(&json) {
            println!("{pretty}");
        }
    }
}

/// PMAT-690 P3-A: model quality score (0-100).
///
/// Weighted sum of five sub-scores. The weights reflect SPEC §84 P3-A
/// ship-blocker priorities — provenance + HF identity are weighted
/// heaviest because their absence is the exact §81-§83 cascade root
/// cause we just shipped (P0-K).
///
/// | Sub-score    | Weight | What it checks                                |
/// |--------------|--------|-----------------------------------------------|
/// | physics      | 20     | header.checksum_valid                         |
/// | structural   | 20     | arch + hidden_size + num_layers + num_heads   |
/// | provenance   | 25     | license + data_source + data_license non-null |
/// | hf_identity  | 20     | hf_architecture + hf_model_type non-null      |
/// | tokenizer    | 15     | has_vocab flag (from header) is true          |
///
/// Total: 100. The ≥ 90 ship gate per AC-SHIP2-007 requires at most
/// one sub-score missing — usually `has_vocab` (15 pts) is the
/// recoverable one for distilled / from-scratch models without an
/// embedded tokenizer. A model missing both HF identity AND
/// provenance scores ≤ 55, well below the ship threshold.
fn compute_quality_score(meta: &MetadataInfo, header: &HeaderData) -> QualityReport {
    let physics = if header.checksum_valid { 20 } else { 0 };
    let structural = {
        let mut score = 0;
        if meta
            .architecture
            .as_deref()
            .is_some_and(|s| !s.is_empty() && s != "unknown")
        {
            score += 5;
        }
        if meta.hidden_size.is_some() {
            score += 5;
        }
        if meta.num_layers.is_some() {
            score += 5;
        }
        if meta.num_heads.is_some() {
            score += 5;
        }
        score
    };
    let provenance = {
        let mut score = 0;
        if meta.license.as_deref().is_some_and(|s| !s.is_empty()) {
            score += 9;
        }
        if meta.data_source.as_deref().is_some_and(|s| !s.is_empty()) {
            score += 8;
        }
        if meta
            .data_license
            .as_deref()
            .is_some_and(|s| !s.is_empty())
        {
            score += 8;
        }
        score
    };
    let hf_identity = {
        let mut score = 0;
        if meta
            .hf_architecture
            .as_deref()
            .is_some_and(|s| !s.is_empty())
        {
            score += 12;
        }
        if meta
            .hf_model_type
            .as_deref()
            .is_some_and(|s| !s.is_empty())
        {
            score += 8;
        }
        score
    };
    let tokenizer = if flags_from_header(header).has_vocab {
        15
    } else {
        0
    };
    let total = physics + structural + provenance + hf_identity + tokenizer;
    QualityReport {
        score: total,
        physics,
        structural,
        provenance,
        hf_identity,
        tokenizer,
        ship_ready: total >= 90,
    }
}

#[derive(Debug)]
struct QualityReport {
    score: u32,
    physics: u32,
    structural: u32,
    provenance: u32,
    hf_identity: u32,
    tokenizer: u32,
    ship_ready: bool,
}

impl QualityReport {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "score": self.score,
            "ship_ready": self.ship_ready,
            "threshold": 90,
            "breakdown": {
                "physics": self.physics,
                "structural": self.structural,
                "provenance": self.provenance,
                "hf_identity": self.hf_identity,
                "tokenizer": self.tokenizer,
            },
        })
    }
}

/// PMAT-690 P3-A: text rendering of the quality block.
fn output_quality_text(meta: &MetadataInfo, header: &HeaderData) {
    let q = compute_quality_score(meta, header);
    println!("\n  Quality (0-100):");
    println!("    Score: {} / 100", q.score);
    println!(
        "    Ship-ready (≥90 per AC-SHIP2-007): {}",
        if q.ship_ready { "YES" } else { "NO" }
    );
    println!("    Breakdown:");
    println!("      physics:     {} / 20", q.physics);
    println!("      structural:  {} / 20", q.structural);
    println!("      provenance:  {} / 25", q.provenance);
    println!("      hf_identity: {} / 20", q.hf_identity);
    println!("      tokenizer:   {} / 15", q.tokenizer);
}

fn output_text(
    path: &Path,
    file_size: u64,
    header: &HeaderData,
    metadata: &MetadataInfo,
    show_vocab: bool,
    show_filters: bool,
    show_weights: bool,
) {
    output::header(&path.display().to_string());

    // Header info as kv_table
    let (v_maj, v_min) = header.version;
    let checksum_str = if header.checksum_valid {
        output::badge_pass("VALID")
    } else {
        output::badge_fail("INVALID")
    };

    let header_pairs = vec![
        ("Format", "APR v2".to_string()),
        ("Version", format!("{v_maj}.{v_min}")),
        ("Size", output::format_size(file_size)),
        ("Tensors", output::count_fmt(header.tensor_count as usize)),
        ("Checksum", checksum_str),
        (
            "Data Offset",
            format!(
                "0x{:X} ({})",
                header.data_offset,
                output::format_size(header.data_offset)
            ),
        ),
    ];
    println!("{}", output::kv_table(&header_pairs));

    // Flags
    output_flags(header);

    // Architecture section
    output_architecture(metadata);

    // General metadata
    output_metadata_text(metadata);

    if show_vocab {
        println!("\n  Vocabulary: (use `apr tensors` for detailed view)");
    }
    if show_filters {
        println!("\n  Filters: (not applicable for v2 format)");
    }
    if show_weights {
        println!("\n  Weights: (use `apr tensors` for detailed view)");
    }
}

fn flags_from_header(header: &HeaderData) -> FlagsInfo {
    FlagsInfo {
        lz4_compressed: header.flags.is_lz4_compressed(),
        zstd_compressed: header.flags.is_zstd_compressed(),
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.contains(AprV2Flags::SIGNED),
        sharded: header.flags.is_sharded(),
        quantized: header.flags.is_quantized(),
        has_vocab: header.flags.contains(AprV2Flags::HAS_VOCAB),
    }
}

fn output_flags(header: &HeaderData) {
    let flag_list = collect_flag_labels(header);
    if flag_list.is_empty() {
        output::kv("Flags", "(none)");
    } else {
        output::kv("Flags", flag_list.join(" | "));
    }
}

/// Ordered list of (predicate, label) pairs. Order here is the display order.
fn collect_flag_labels(header: &HeaderData) -> Vec<&'static str> {
    type Pred = fn(&HeaderData) -> bool;
    const TABLE: &[(Pred, &str)] = &[
        (|h| h.flags.is_lz4_compressed(), "LZ4"),
        (|h| h.flags.is_zstd_compressed(), "ZSTD"),
        (|h| h.flags.is_encrypted(), "ENCRYPTED"),
        (|h| h.flags.contains(AprV2Flags::SIGNED), "SIGNED"),
        (|h| h.flags.is_sharded(), "SHARDED"),
        (|h| h.flags.is_quantized(), "QUANTIZED"),
        (|h| h.flags.contains(AprV2Flags::HAS_VOCAB), "HAS_VOCAB"),
        (|h| h.flags.contains(AprV2Flags::HAS_FILTERBANK), "HAS_FILTERBANK"),
        (|h| h.flags.contains(AprV2Flags::HAS_MODEL_CARD), "HAS_MODEL_CARD"),
        (|h| h.flags.contains(AprV2Flags::STREAMING), "STREAMING"),
    ];
    TABLE
        .iter()
        .filter_map(|(pred, label)| pred(header).then_some(*label))
        .collect()
}

fn output_architecture(metadata: &MetadataInfo) {
    let has_arch_info = metadata.architecture.is_some()
        || metadata.hidden_size.is_some()
        || metadata.num_layers.is_some();
    if !has_arch_info {
        return;
    }

    println!("\n  Architecture:");
    if let Some(arch) = &metadata.architecture {
        println!("    Family: {arch}");
    }
    // PMAT-690 P0-K: surface the HF identity fields so operators can
    // verify upstream `apr convert` stamping without `--json | jq`.
    // Distinct from `Family`: `HF Class` is the canonical class name
    // (e.g. "Qwen2ForCausalLM"); `HF model_type` mirrors
    // `config.json::model_type`. Per C-APR-CONVERT-HF-ARCH and the
    // §84 P0-K root-cause analysis, missing fields here mean the
    // import skipped stamping — operators can route the failure
    // back to `apr convert` rather than chasing downstream symptoms.
    if let Some(hf) = &metadata.hf_architecture {
        println!("    HF Class: {hf}");
    }
    if let Some(mt) = &metadata.hf_model_type {
        println!("    HF model_type: {mt}");
    }
    if let Some(p) = metadata.param_count {
        println!("    Parameters: {}", format_param_count(p));
    }
    print_arch_numeric_fields(metadata);
}

fn print_arch_numeric_fields(metadata: &MetadataInfo) {
    for (label, value) in [
        ("Hidden Size", metadata.hidden_size),
        ("Layers", metadata.num_layers),
        ("Attention Heads", metadata.num_heads),
        ("KV Heads", metadata.num_kv_heads),
        ("Intermediate Size", metadata.intermediate_size),
        ("Vocab Size", metadata.vocab_size),
        ("Max Position", metadata.max_position_embeddings),
    ] {
        if let Some(v) = value {
            println!("    {label}: {v}");
        }
    }
    if let Some(r) = metadata.rope_theta {
        println!("    RoPE Theta: {r}");
    }
}

/// Print chat template section if present.
fn output_chat_template_info(metadata: &MetadataInfo) {
    if metadata.chat_template.is_none() && metadata.chat_format.is_none() {
        return;
    }
    println!("\n  Chat Template:");
    if let Some(format) = &metadata.chat_format {
        println!("    Format: {format}");
    }
    if let Some(template) = &metadata.chat_template {
        let display_template = if template.len() > 100 {
            format!("{}... ({} chars)", &template[..100], template.len())
        } else {
            template.clone()
        };
        println!("    Template: {display_template}");
    }
    if let Some(tokens) = &metadata.special_tokens {
        print_json_object("    Special Tokens:", tokens, "      ");
    }
}

/// Print a JSON object's non-null key-value pairs.
fn print_json_object(header: &str, value: &serde_json::Value, indent: &str) {
    println!("{header}");
    let Some(obj) = value.as_object() else { return };
    for (k, v) in obj {
        if !v.is_null() {
            if let Some(s) = v.as_str() {
                println!("{indent}{k}: {s}");
            } else {
                println!("{indent}{k}: {v}");
            }
        }
    }
}

fn output_metadata_text(metadata: &MetadataInfo) {
    // General metadata fields
    let fields: &[(&str, &Option<String>)] = &[
        ("Name", &metadata.name),
        ("Model Type", &metadata.model_type),
        ("Description", &metadata.description),
        ("Author", &metadata.author),
        ("Source", &metadata.source),
        ("Original Format", &metadata.original_format),
        ("Created", &metadata.created_at),
    ];
    for (label, value) in fields {
        if let Some(v) = value {
            output::kv(label, v);
        }
    }

    // C-APR-PROVENANCE / AC-SHIP2-012 / FALSIFY-SHIP-022:
    // always emit the three provenance keys so auditors never see a
    // silent skip (INV-APR-PROV-002 / FM-APR-PROV-SILENT-SKIP).
    print!("{}", format_provenance_block(metadata));

    output_chat_template_info(metadata);

    if let Some(source_meta) = &metadata.source_metadata {
        print_json_object("\n  Source Metadata (PMAT-223):", source_meta, "    ");
    }
}

/// Render the provenance block for `apr inspect` text output.
///
/// C-APR-PROVENANCE / INV-APR-PROV-002: always emits a "Provenance:" header
/// followed by `license`, `data_source`, `data_license`, each as either the
/// stored value or the literal `(missing)` — NEVER silently skipped.
fn format_provenance_block(metadata: &MetadataInfo) -> String {
    let mut out = String::new();
    out.push('\n');
    out.push_str("  Provenance:\n");
    for (label, value) in [
        ("license", &metadata.license),
        ("data_source", &metadata.data_source),
        ("data_license", &metadata.data_license),
    ] {
        let display = value.as_deref().unwrap_or("(missing)");
        out.push_str(&format!("    {label}: {display}\n"));
    }
    out
}

fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B ({count})", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M ({count})", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K ({count})", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}
