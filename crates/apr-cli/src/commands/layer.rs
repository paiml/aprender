
/// Create an embedding layer trace.
fn create_embedding_layer(d_model: usize) -> LayerTrace {
    LayerTrace {
        name: "embedding".to_string(),
        index: None,
        input_stats: None,
        output_stats: Some(TensorStats {
            count: d_model,
            mean: 0.0,
            std: 0.0,
            l2_norm: 0.0,
            min: 0.0,
            max: 0.0,
            max_abs: 0.0,
            nan_count: 0,
            inf_count: 0,
        }),
        weight_stats: None,
        anomalies: vec![],
    }
}

/// Create transformer layer traces with optional filtering.
fn create_transformer_layers(n_layers: usize, filter: Option<&str>) -> Vec<LayerTrace> {
    (0..n_layers)
        .filter_map(|i| {
            let layer_name = format!("transformer_block_{i}");
            if filter.is_some_and(|f| !layer_name.contains(f)) {
                return None;
            }
            Some(LayerTrace {
                name: layer_name,
                index: Some(i),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec![],
            })
        })
        .collect()
}

/// Create final layer norm trace.
fn create_final_layer_norm() -> LayerTrace {
    LayerTrace {
        name: "final_layer_norm".to_string(),
        index: None,
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
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec!["No layer information in metadata".to_string()],
    }
}

/// Extract layers from hyperparameters metadata.
fn extract_layers_from_hyperparameters(
    hp: &serde_json::Map<String, serde_json::Value>,
    filter: Option<&str>,
) -> Vec<LayerTrace> {
    let n_layers = extract_layer_count(hp);
    let d_model = extract_model_dimension(hp);

    let mut layers = vec![create_embedding_layer(d_model)];
    layers.extend(create_transformer_layers(n_layers, filter));
    layers.push(create_final_layer_norm());
    layers
}

#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe here for empty vec
fn trace_layers(metadata_bytes: &[u8], filter: Option<&str>, _verbose: bool) -> Vec<LayerTrace> {
    let metadata: BTreeMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_default();

    let layers: Vec<LayerTrace> = metadata
        .get("hyperparameters")
        .and_then(|hp| hp.as_object())
        .map(|hp_obj| extract_layers_from_hyperparameters(hp_obj, filter))
        .unwrap_or_default();

    if layers.is_empty() {
        vec![create_default_layer()]
    } else {
        layers
    }
}

fn compare_with_reference(
    model_path: &Path,
    ref_path: &Path,
    _layers: &[LayerTrace],
    json_output: bool,
) -> Result<(), CliError> {
    validate_path(ref_path)?;

    if json_output {
        println!("{{\"comparison\": \"reference comparison not yet implemented\"}}");
    } else {
        output::section(&format!(
            "Layer Comparison: {} vs {}",
            model_path.display(),
            ref_path.display()
        ));
        println!();
        println!("{}", "Reference comparison coming soon...".yellow());
        println!();
        println!("Future features:");
        println!("  - Layer-by-layer output comparison");
        println!("  - Cosine similarity between activations");
        println!("  - Probar visual diff generation");
    }

    Ok(())
}

fn output_json(path: &Path, format: &str, layers: &[LayerTrace], summary: &TraceSummary) {
    let result = TraceResult {
        file: path.display().to_string(),
        format: format.to_string(),
        layers: layers.to_vec(),
        summary: TraceSummary {
            total_layers: summary.total_layers,
            total_parameters: summary.total_parameters,
            anomaly_count: summary.anomaly_count,
            anomalies: summary.anomalies.clone(),
        },
    };

    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    verbose: bool,
) {
    output::header(&format!("Layer Trace: {}", path.display()));

    println!(
        "{}",
        output::kv_table(&[
            ("Format", format.to_string()),
            ("Layers", summary.total_layers.to_string()),
            ("Parameters", output::count_fmt(summary.total_parameters)),
        ])
    );

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
