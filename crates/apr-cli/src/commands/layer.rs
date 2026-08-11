
/// Create an embedding layer trace.
///
/// #2407: this used to attach an `output_stats` block of all zeros with
/// `count = d_model`. Nothing was ever measured — no tensor is read on this
/// path — so a client checking `l2_norm == 0.0` saw a dead embedding layer
/// on a healthy model. The declared width is reported as `hidden_dim`, which
/// is what it actually is.
fn create_embedding_layer(d_model: usize) -> LayerTrace {
    LayerTrace {
        name: "embedding".to_string(),
        index: None,
        hidden_dim: (d_model > 0).then_some(d_model),
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec![],
    }
}

/// Create transformer layer traces.
fn create_transformer_layers(n_layers: usize) -> Vec<LayerTrace> {
    (0..n_layers)
        .map(|i| LayerTrace {
            name: format!("transformer_block_{i}"),
            index: Some(i),
            hidden_dim: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        })
        .collect()
}

/// Create final layer norm trace.
fn create_final_layer_norm() -> LayerTrace {
    LayerTrace {
        name: "final_layer_norm".to_string(),
        index: None,
        hidden_dim: None,
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec![],
    }
}

/// Create default layer trace when no metadata available.
fn create_default_layer() -> LayerTrace {
    LayerTrace {
        name: "(layer trace metadata not available)".to_string(),
        index: None,
        hidden_dim: None,
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec!["No layer information in metadata".to_string()],
    }
}

/// Apply the `--layer` filter to an already-built layer list.
///
/// #2407: filtering used to happen inside every layer constructor, so a
/// filter that matched nothing left the list empty and the caller's
/// `if layers.is_empty()` fallback then fabricated a
/// "(layer trace metadata not available)" entry with the anomaly
/// "No layer information in metadata" — a false alarm on a model whose
/// unfiltered trace, in the same run, listed 25 real layers. Filtering once,
/// after the list exists, keeps "matched nothing" distinguishable from
/// "there is nothing to match".
///
/// Returns the surviving layers plus any notes describing the filter outcome.
fn apply_layer_filter(
    layers: Vec<LayerTrace>,
    filter: Option<&str>,
) -> (Vec<LayerTrace>, Vec<String>) {
    let Some(pattern) = filter else {
        return (layers, Vec::new());
    };
    let total = layers.len();
    let kept: Vec<LayerTrace> = layers
        .into_iter()
        .filter(|l| l.name.contains(pattern))
        .collect();
    if kept.is_empty() {
        let note = format!("layer filter {pattern:?} matched 0 of {total} layers");
        return (kept, vec![note]);
    }
    (kept, Vec::new())
}

/// Extract layers from hyperparameters metadata.
fn extract_layers_from_hyperparameters(
    hp: &serde_json::Map<String, serde_json::Value>,
) -> Vec<LayerTrace> {
    let n_layers = extract_layer_count(hp);
    let d_model = extract_model_dimension(hp);

    let mut layers = vec![create_embedding_layer(d_model)];
    layers.extend(create_transformer_layers(n_layers));
    layers.push(create_final_layer_norm());
    layers
}

/// Build the unfiltered layer skeleton for an APR file's metadata blob.
///
/// The `--layer` filter is applied afterwards by [`apply_layer_filter`], so
/// that "the filter matched nothing" cannot be mistaken for "the file has no
/// layer metadata" (#2407).
#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe here for empty vec
fn trace_layers(metadata_bytes: &[u8], verbose: bool) -> Vec<LayerTrace> {
    // GH-529: Warn that verbose layer tracing is not yet implemented
    if verbose {
        eprintln!("Warning: --verbose is not yet implemented for layer tracing. Flag ignored.");
    }
    let metadata: BTreeMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_default();

    let layers: Vec<LayerTrace> = metadata
        .get("hyperparameters")
        .and_then(|hp| hp.as_object())
        .map(extract_layers_from_hyperparameters)
        .unwrap_or_default();

    if layers.is_empty() {
        vec![create_default_layer()]
    } else {
        layers
    }
}

/// Message returned when `--reference` is supplied.
///
/// #2407: `--reference` printed `{"comparison": "reference comparison not
/// yet implemented"}` on stdout, wrote its only real signal to stderr, and
/// exited 0. The MCP wrapper drops stderr on success, so an MCP client saw a
/// plain success result for an operation that compared nothing — and
/// `--reference` also silently voided the `--layer` filter and the whole
/// trace payload. Until layer comparison exists, saying so and failing is
/// the only honest answer.
const REFERENCE_UNIMPLEMENTED: &str =
    "`apr trace --reference` is not implemented: no layer-by-layer comparison is performed. \
     Re-run without --reference to get the trace of a single model.";

fn compare_with_reference(ref_path: &Path) -> Result<(), CliError> {
    // Report a bad reference path as such rather than hiding it behind the
    // unimplemented message.
    validate_path(ref_path)?;
    Err(CliError::NotImplemented(
        REFERENCE_UNIMPLEMENTED.to_string(),
    ))
}

/// Assemble the `--json` payload. Split out from [`output_json`] so the
/// #2407 falsifiers can assert on the payload instead of on stdout.
fn build_trace_result(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    notes: &[String],
) -> TraceResult {
    let mut all_notes = vec![METADATA_ONLY_NOTE.to_string()];
    all_notes.extend(notes.iter().cloned());

    TraceResult {
        file: path.display().to_string(),
        format: format.to_string(),
        stats_source: STATS_SOURCE_METADATA_ONLY,
        notes: all_notes,
        layers: layers.to_vec(),
        summary: TraceSummary {
            total_layers: summary.total_layers,
            total_parameters: summary.total_parameters,
            anomaly_count: summary.anomaly_count,
            anomalies: summary.anomalies.clone(),
        },
    }
}

fn output_json(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    notes: &[String],
) {
    let result = build_trace_result(path, format, layers, summary, notes);
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    notes: &[String],
    verbose: bool,
) {
    output::header(&format!("Layer Trace: {}", path.display()));

    println!(
        "{}",
        output::kv_table(&[
            ("Format", format.to_string()),
            ("Layers", summary.total_layers.to_string()),
            ("Parameters", output::count_fmt(summary.total_parameters)),
            ("Stats", STATS_SOURCE_METADATA_ONLY.to_string()),
        ])
    );

    println!();
    println!("  {}", METADATA_ONLY_NOTE.yellow());
    for note in notes {
        println!("  {}", note.yellow());
    }

    if !summary.anomalies.is_empty() {
        println!();
        println!(
            "  {} {} anomalies detected:",
            output::badge_warn("ANOMALY"),
            summary.anomaly_count
        );
        for anomaly in &summary.anomalies {
            println!("    - {}", anomaly.red());
        }
    }

    println!();
    output::subheader("Layer Breakdown");

    // Build layer table
    let mut rows: Vec<Vec<String>> = Vec::new();
    for layer in layers {
        let idx_str = layer.index.map_or(String::new(), |i| format!("{i}"));
        let anomaly_str = if layer.anomalies.is_empty() {
            String::new()
        } else {
            layer.anomalies.join("; ")
        };

        if verbose {
            let weight_info = layer.weight_stats.as_ref().map_or(String::from("-"), |s| {
                format!("{} params, mean={:.4}, std={:.4}", s.count, s.mean, s.std)
            });
            let output_info = layer.output_stats.as_ref().map_or(String::from("-"), |s| {
                format!(
                    "mean={:.4}, std={:.4}, [{:.4}, {:.4}]",
                    s.mean, s.std, s.min, s.max
                )
            });
            rows.push(vec![
                idx_str,
                layer.name.clone(),
                weight_info,
                output_info,
                anomaly_str,
            ]);
        } else {
            rows.push(vec![idx_str, layer.name.clone(), anomaly_str]);
        }
    }

    if verbose {
        println!(
            "{}",
            output::table(&["#", "Layer", "Weights", "Output", "Anomalies"], &rows,)
        );
    } else {
        println!("{}", output::table(&["#", "Layer", "Anomalies"], &rows));
    }
}
