/// Does this pruning method REMOVE parameters, or only set them to zero?
///
/// Only depth pruning drops tensors. Magnitude / Wanda / SparseGPT — and
/// `Structured` / `Width`, which both dispatch to `prune_magnitude` — zero
/// weights in place: `apply_pruning` returns the same tensors with the same
/// shapes, so the serialized file cannot shrink.
fn method_removes_parameters(method: PruneMethod) -> bool {
    matches!(method, PruneMethod::Depth)
}

/// What `apr prune --plan` predicts about the file the run will write.
struct PrunePlanEstimate {
    params_in: u64,
    params_kept: u64,
    /// Bytes the pruned model is expected to occupy on disk.
    estimated_output: u64,
    /// True when the method only zeroes weights, so nothing is removed.
    zeroes_only: bool,
}

/// Predict the output size of a prune run.
///
/// The old estimate was `input_size * (1 - target_ratio)` — arithmetic that
/// ignored what the command does. On the dogfood fixture
/// `--target-ratio 0.9 --plan` promised 474.61 KiB and the run that followed
/// wrote 9.27 MiB: 20x out, and out in the direction that under-sizes a
/// deployment. Prune writes a dense f32 APR of the SURVIVING parameters
/// (4 bytes each), and only depth pruning removes any of them.
fn estimate_prune_output(
    file: &Path,
    method: PruneMethod,
    remove_layers: Option<&str>,
) -> Result<PrunePlanEstimate> {
    let (params_in, params_kept) = plan_param_counts(file, method, remove_layers)?;
    Ok(PrunePlanEstimate {
        params_in,
        params_kept,
        estimated_output: params_kept * 4,
        zeroes_only: !method_removes_parameters(method),
    })
}

/// Count parameters in the model, and how many survive the given method.
///
/// Reads only the tensor index (shapes), never the tensor data.
fn plan_param_counts(
    file: &Path,
    method: PruneMethod,
    remove_layers: Option<&str>,
) -> Result<(u64, u64)> {
    use aprender::format::tensors::{list_tensors, TensorListOptions};

    let listing = list_tensors(file, TensorListOptions::default())
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read tensor index: {e}")))?;

    let mut total: u64 = 0;
    let mut kept: u64 = 0;
    let removed_layers = if method_removes_parameters(method) {
        remove_layers.map(parse_layer_spec).transpose()?
    } else {
        None
    };

    for t in &listing.tensors {
        let n: u64 = t.shape.iter().map(|d| *d as u64).product::<u64>();
        total += n;
        let dropped = removed_layers.as_ref().is_some_and(|layers| {
            layers.iter().any(|idx| {
                [
                    format!("layers.{idx}."),
                    format!("blk.{idx}."),
                    format!("h.{idx}."),
                ]
                .iter()
                .any(|p| t.name.contains(p.as_str()))
            })
        });
        if !dropped {
            kept += n;
        }
    }
    Ok((total, kept))
}

/// Plan pruning (estimate only)
#[allow(clippy::disallowed_methods)]
fn run_plan(
    file: &Path,
    method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    remove_layers: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    let PrunePlanEstimate {
        params_in,
        params_kept,
        estimated_output,
        zeroes_only,
    } = estimate_prune_output(file, method, remove_layers)?;
    let peak_memory = file_size + estimated_output;
    let zeroed_pct = f64::from(effective_prune_fraction(target_ratio, sparsity)) * 100.0;

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "input": file.display().to_string(),
            "input_size": file_size,
            "method": format!("{method:?}"),
            "target_ratio": target_ratio,
            "sparsity": sparsity,
            "parameters_in": params_in,
            "parameters_kept": params_kept,
            "removes_parameters": !zeroes_only,
            "weights_zeroed_pct": zeroed_pct,
            "estimated_output_size": estimated_output,
            "peak_memory": peak_memory,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Prune — Plan");
        let mut rows = vec![
            ("Input", file.display().to_string()),
            (
                "Input size",
                humansize::format_size(file_size, humansize::BINARY),
            ),
            ("Method", format!("{method:?}")),
            ("Target ratio", format!("{target_ratio:.2}")),
            ("Parameters", format!("{params_in} → {params_kept}")),
        ];
        if zeroes_only {
            rows.push(("Weights zeroed", format!("{zeroed_pct:.1}%")));
        }
        rows.push((
            "Est. output",
            humansize::format_size(estimated_output, humansize::BINARY),
        ));
        rows.push((
            "Peak memory",
            humansize::format_size(peak_memory, humansize::BINARY),
        ));
        println!("{}", output::kv_table(&rows));
        println!();
        if zeroes_only {
            println!(
                "  {} {method:?} pruning ZEROES weights in place — no parameter is removed, \
                 so the output is not smaller than the input (it is written as dense f32).",
                output::badge_info("NOTE"),
            );
        }
        println!(
            "  {} Run without --plan to execute.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

/// Magnitude pruning: zero out weights below a threshold derived from the sparsity ratio
fn prune_magnitude(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    sparsity: f32,
) -> std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut result = std::collections::BTreeMap::new();

    for (name, (data, shape)) in tensors {
        // Collect absolute values and find the threshold
        let mut abs_vals: Vec<f32> = data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff_idx = ((abs_vals.len() as f64 * sparsity as f64) as usize)
            .min(abs_vals.len().saturating_sub(1));
        let threshold = abs_vals[cutoff_idx];

        // Zero out values below threshold
        let pruned: Vec<f32> = data
            .iter()
            .map(|v| if v.abs() < threshold { 0.0 } else { *v })
            .collect();
        result.insert(name.clone(), (pruned, shape.clone()));
    }

    result
}

/// Parse a `--remove-layers` spec: a range like `20-24` or a list like `5,10,15`.
///
/// Shared by `prune_depth` (which does the removal) and `run_plan` (which must
/// estimate the same removal), so a plan and its run cannot disagree.
fn parse_layer_spec(layer_spec: &str) -> Result<Vec<usize>> {
    if layer_spec.contains('-') {
        let parts: Vec<&str> = layer_spec.split('-').collect();
        if parts.len() != 2 {
            return Err(CliError::ValidationFailed(format!(
                "Invalid layer range: {layer_spec}"
            )));
        }
        let start: usize = parts[0].parse().map_err(|_| {
            CliError::ValidationFailed(format!("Invalid layer number: {}", parts[0]))
        })?;
        let end: usize = parts[1].parse().map_err(|_| {
            CliError::ValidationFailed(format!("Invalid layer number: {}", parts[1]))
        })?;
        Ok((start..=end).collect())
    } else {
        layer_spec
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| CliError::ValidationFailed(format!("Invalid layer number: {s}")))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
    }
}

/// Depth pruning: remove entire layers by name pattern
#[allow(clippy::type_complexity)]
fn prune_depth(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    layer_spec: &str,
) -> Result<std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let layers_to_remove = parse_layer_spec(layer_spec)?;

    let mut result = std::collections::BTreeMap::new();
    for (name, (data, shape)) in tensors {
        // Check if tensor belongs to a removed layer (e.g., "model.layers.20.self_attn.q_proj.weight")
        let should_remove = layers_to_remove.iter().any(|layer_idx| {
            let patterns = [
                format!("layers.{layer_idx}."),
                format!("blk.{layer_idx}."),
                format!("h.{layer_idx}."),
            ];
            patterns.iter().any(|p| name.contains(p))
        });

        if !should_remove {
            result.insert(name.clone(), (data.clone(), shape.clone()));
        }
    }

    Ok(result)
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a minimal f32 SafeTensors file with the given `(name, shape)`
    /// tensors so the plan estimator has a real tensor index to read.
    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut payload: Vec<u8> = Vec::new();
        for (name, shape) in tensors {
            let n: usize = shape.iter().product();
            let bytes = n * 4;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            payload.extend(std::iter::repeat_n(0u8, bytes));
            offset += bytes;
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header))
            .expect("serialize safetensors header");
        let mut out = Vec::new();
        out.extend((header_json.len() as u64).to_le_bytes());
        out.extend(&header_json);
        out.extend(&payload);
        std::fs::write(path, out).expect("write safetensors fixture");
    }

    /// `--plan` promised `input_size * (1 - target_ratio)`, which described
    /// nothing the command does: magnitude pruning ZEROES weights, it does not
    /// remove them. On the 4.63 MiB dogfood fixture `--target-ratio 0.9 --plan`
    /// promised 474.61 KiB and the run wrote 9.27 MiB — 20x out, and small.
    /// The estimate must track the bytes actually written.
    #[test]
    fn plan_estimate_matches_the_bytes_pruning_actually_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("m.safetensors");
        write_safetensors(
            &input,
            &[
                ("model.layers.0.self_attn.q_proj.weight", vec![128, 128]),
                ("model.layers.1.self_attn.q_proj.weight", vec![128, 128]),
                ("model.norm.weight", vec![128]),
            ],
        );

        for ratio in [0.2_f32, 0.5, 0.9] {
            let est = estimate_prune_output(&input, PruneMethod::Magnitude, None)
                .expect("plan estimate");
            assert_eq!(est.params_in, 128 * 128 * 2 + 128);
            assert_eq!(
                est.params_kept, est.params_in,
                "magnitude pruning removes no parameters"
            );
            assert!(est.zeroes_only, "magnitude only zeroes weights");
            let estimated = est.estimated_output;

            let out = dir.path().join(format!("p_{ratio}.apr"));
            run(
                &input,
                "magnitude",
                ratio,
                0.0,
                Some(&out),
                None,
                false,
                false,
                None,
                true,
            )
            .expect("prune run");
            let actual = std::fs::metadata(&out).expect("output written").len();

            let err = (estimated as f64 - actual as f64).abs() / actual as f64;
            assert!(
                err < 0.05,
                "target-ratio {ratio}: plan said {estimated} bytes, prune wrote {actual} \
                 ({:.1}% off)",
                err * 100.0
            );
        }
    }

    /// Depth pruning is the one method that DOES remove parameters, and the
    /// estimate must follow the same layer spec the run uses.
    #[test]
    fn plan_estimate_drops_removed_layers_for_depth_pruning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("d.safetensors");
        write_safetensors(
            &input,
            &[
                ("model.layers.0.mlp.up_proj.weight", vec![128, 128]),
                ("model.layers.1.mlp.up_proj.weight", vec![128, 128]),
                ("model.layers.2.mlp.up_proj.weight", vec![128, 128]),
                ("model.norm.weight", vec![128]),
            ],
        );
        let est =
            estimate_prune_output(&input, PruneMethod::Depth, Some("1-2")).expect("plan estimate");
        assert_eq!(est.params_in, 128 * 128 * 3 + 128);
        assert_eq!(
            est.params_kept,
            128 * 128 + 128,
            "layers 1 and 2 must be estimated away"
        );
        assert!(!est.zeroes_only, "depth pruning removes parameters");
        let kept = est.params_kept;

        // Same spec, executed: the estimate must equal the real output size.
        let out = dir.path().join("d.apr");
        run(
            &input,
            "depth",
            0.5,
            0.0,
            Some(&out),
            Some("1-2"),
            false,
            false,
            None,
            true,
        )
        .expect("depth prune run");
        let actual = std::fs::metadata(&out).expect("output written").len();
        let err = (est.estimated_output as f64 - actual as f64).abs() / actual as f64;
        assert!(
            err < 0.05,
            "plan said {} bytes, prune wrote {actual}",
            est.estimated_output
        );
        assert_eq!(est.estimated_output, kept * 4);
    }

    /// The estimate must not move with `--target-ratio` for a method that only
    /// zeroes weights — that linear scaling was the whole defect.
    #[test]
    fn plan_estimate_is_invariant_to_target_ratio_for_unstructured_methods() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("i.safetensors");
        write_safetensors(&input, &[("w", vec![8, 8])]);
        let mut seen = Vec::new();
        for method in [
            PruneMethod::Magnitude,
            PruneMethod::Structured,
            PruneMethod::Width,
            PruneMethod::Wanda,
            PruneMethod::SparseGpt,
        ] {
            assert!(
                !method_removes_parameters(method),
                "{method:?} zeroes weights, it does not remove them"
            );
            let est = estimate_prune_output(&input, method, None).expect("plan estimate");
            assert_eq!(est.estimated_output, 64 * 4, "{method:?}");
            seen.push(est.params_kept);
        }
        assert!(seen.iter().all(|k| *k == 64), "got {seen:?}");
    }

    #[test]
    fn test_prune_method_parse() {
        assert!(matches!(
            "magnitude".parse::<PruneMethod>(),
            Ok(PruneMethod::Magnitude)
        ));
        assert!(matches!(
            "mag".parse::<PruneMethod>(),
            Ok(PruneMethod::Magnitude)
        ));
        assert!(matches!(
            "structured".parse::<PruneMethod>(),
            Ok(PruneMethod::Structured)
        ));
        assert!(matches!(
            "depth".parse::<PruneMethod>(),
            Ok(PruneMethod::Depth)
        ));
        assert!(matches!(
            "width".parse::<PruneMethod>(),
            Ok(PruneMethod::Width)
        ));
        assert!(matches!(
            "wanda".parse::<PruneMethod>(),
            Ok(PruneMethod::Wanda)
        ));
        assert!(matches!(
            "sparsegpt".parse::<PruneMethod>(),
            Ok(PruneMethod::SparseGpt)
        ));
        assert!("unknown".parse::<PruneMethod>().is_err());
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent.apr"),
            "magnitude",
            0.5,
            0.0,
            Some(Path::new("/tmp/out.apr")),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_invalid_target_ratio_zero() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            "magnitude",
            0.0,
            0.0,
            Some(Path::new("/tmp/out.apr")),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Target ratio")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_invalid_target_ratio_one() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            "magnitude",
            1.0,
            0.0,
            Some(Path::new("/tmp/out.apr")),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_sparsity() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            1.5,
            Some(Path::new("/tmp/out.apr")),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Sparsity")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_depth_requires_layers() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            input.path(),
            "depth",
            0.5,
            0.0,
            Some(Path::new("/tmp/out.apr")),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("remove-layers")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_output() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            0.0,
            None,
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Output path")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_analyze_mode() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            0.0,
            None,
            None,
            true,
            false,
            None,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_json() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            0.0,
            None,
            None,
            true,
            false,
            None,
            true,
        );
        assert!(result.is_ok());
    }

    /// These two used to feed 2048 zero bytes — not a model in any format —
    /// and assert `is_ok()`. That passed only because the estimate was
    /// `file_size * (1 - target_ratio)`, arithmetic that never opened the
    /// file; the assertion therefore LOCKED IN a plan that could describe a
    /// model it had not read. A plan for a file that is not a model must fail.
    #[test]
    fn test_plan_mode_rejects_a_file_that_is_not_a_model() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            input.path(),
            "structured",
            0.3,
            0.0,
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            result.is_err(),
            "planned a prune for 2048 zero bytes without reading them"
        );
    }

    #[test]
    fn test_plan_json_rejects_a_file_that_is_not_a_model() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            0.2,
            None,
            None,
            false,
            true,
            None,
            true,
        );
        assert!(
            result.is_err(),
            "planned a prune for 2048 zero bytes without reading them"
        );
    }

    /// A plan on a REAL model still succeeds (the error above is about the
    /// input, not about planning being broken).
    #[test]
    fn test_plan_mode_succeeds_on_a_real_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("ok.safetensors");
        write_safetensors(&input, &[("w", vec![32, 32])]);
        for json in [false, true] {
            run(
                &input,
                "magnitude",
                0.3,
                0.0,
                None,
                None,
                false,
                true,
                None,
                json,
            )
            .expect("plan on a real model");
        }
    }

    #[test]
    fn test_run_with_valid_input() {
        // Create a valid APR file with real tensors
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        writer.add_tensor_f32("layers.0.self_attn.q_proj.weight", vec![8, 8], &weights);
        let bias: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        writer.add_tensor_f32("layers.0.self_attn.q_proj.bias", vec![8], &bias);
        let bytes = writer.to_bytes().expect("serialize");

        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        std::fs::write(input.path(), &bytes).expect("write apr");

        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            input.path(),
            "magnitude",
            0.5,
            0.0,
            Some(output.path()),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(result.is_ok(), "prune failed: {:?}", result.err());
        // Verify output file was actually created with content
        let meta = std::fs::metadata(output.path()).expect("output exists");
        assert!(meta.len() > 0, "Output file should not be empty");
    }

    /// FALSIFY-PRUNE-SPARSITY-001 (PMAT-830): `--sparsity` below the 0.5 `--target-ratio`
    /// default was silently raised by `sparsity.max(target_ratio)`, so
    /// `apr prune --method magnitude --sparsity 0.25` zeroed 50% of weights, not 25% — and
    /// stamped pruning_sparsity=0.25 into the output (a self-contradicting, over-pruned model).
    /// Prior tests only used sparsity=0.0, where `0.0.max(0.5) = 0.5` hid the override.
    #[test]
    fn test_magnitude_sparsity_below_target_ratio_not_overridden() {
        let mut writer = aprender::serialization::apr::AprWriter::new();
        // 64 distinct non-zero magnitudes so the zeroed count == round(64 * fraction).
        let weights: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        writer.add_tensor_f32("layers.0.self_attn.q_proj.weight", vec![8, 8], &weights);
        let bytes = writer.to_bytes().expect("serialize");
        let input = NamedTempFile::with_suffix(".apr").expect("input");
        std::fs::write(input.path(), &bytes).expect("write");
        let output = NamedTempFile::with_suffix(".apr").expect("output");

        // User asks for 25% sparsity; --target-ratio left at its CLI default of 0.5.
        let r = run(
            input.path(),
            "magnitude",
            0.5,  // target_ratio (default)
            0.25, // sparsity (explicit user request)
            Some(output.path()),
            None,
            false,
            false,
            None,
            false,
        );
        assert!(r.is_ok(), "{:?}", r.err());

        let tensors =
            aprender::format::converter::load_model_tensors(output.path()).expect("load output");
        let zeros = tensors["layers.0.self_attn.q_proj.weight"]
            .0
            .iter()
            .filter(|v| **v == 0.0)
            .count();
        // EXPECT 16 (25% of 64). RED pre-fix produced 32 (50%) via sparsity.max(0.5).
        assert_eq!(zeros, 16, "expected 25% pruned (16/64), got {zeros}/64");
    }
}
