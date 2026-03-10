// ============================================================================
// Advanced Merge Strategies (GH-442)
// ============================================================================
// Task Arithmetic, NuSLERP, MultiSLERP, DELLA, Breadcrumbs, SCE

/// Task Arithmetic: linear combination of task vectors.
///
/// result = base + Σ(scale_i * (model_i - base))
///
/// Reference: Ilharco et al. 2023, "Editing Models with Task Arithmetic"
pub(crate) fn task_arithmetic_merge(
    base_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    task_models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    scales: &[f32],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    for (name, (base_data, shape)) in base_tensors {
        let mut result = base_data.clone();

        for (model_idx, model_tensors) in task_models.iter().enumerate() {
            let (model_data, _) = model_tensors.get(name).expect("validated above");
            let scale = scales.get(model_idx).copied().unwrap_or(1.0);

            for (i, (&m_val, r_val)) in model_data.iter().zip(result.iter_mut()).enumerate() {
                let _ = i; // suppress unused warning
                *r_val += scale * (m_val - base_data[i]);
            }
        }

        merged.insert(name.clone(), (result, shape.clone()));
    }

    merged
}

/// NuSLERP: enhanced SLERP with nlerp fallback for near-parallel vectors.
///
/// Uses normalized linear interpolation (nlerp) when angle is very small,
/// full SLERP otherwise. Faster than standard SLERP with equivalent quality.
pub(crate) fn nuslerp_tensors(
    model_a: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    model_b: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    t: f32,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    for (name, (data_a, shape)) in model_a {
        let (data_b, _) = model_b.get(name).expect("validated above");
        let merged_data = nuslerp_vectors(data_a, data_b, t);
        merged.insert(name.clone(), (merged_data, shape.clone()));
    }

    merged
}

/// NuSLERP between two vectors with nlerp fallback.
fn nuslerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let norm_a = vector_norm(a);
    let norm_b = vector_norm(b);

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return lerp_vectors(a, b, t);
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);

    // NuSLERP threshold: use nlerp when nearly parallel (within ~5 degrees)
    if cos_omega.abs() > 0.9995 {
        return nlerp_vectors(a, b, t);
    }

    let omega = cos_omega.acos();
    let sin_omega = omega.sin();
    let t64 = f64::from(t);
    let coeff_a = ((1.0 - t64) * omega).sin() / sin_omega;
    let coeff_b = (t64 * omega).sin() / sin_omega;

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (coeff_a * f64::from(x) + coeff_b * f64::from(y)) as f32)
        .collect()
}

/// Normalized linear interpolation: lerp then normalize to preserve magnitude.
fn nlerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let lerped: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * (1.0 - t) + y * t)
        .collect();

    let norm = vector_norm(&lerped);
    if norm < 1e-12 {
        return lerped;
    }

    // Target norm: interpolate between input norms
    let target_norm = f64::from(1.0 - t) * vector_norm(a) + f64::from(t) * vector_norm(b);
    let scale = (target_norm / norm) as f32;

    lerped.iter().map(|&x| x * scale).collect()
}

/// MultiSLERP: barycentric SLERP for >2 models.
///
/// Iteratively applies SLERP to pairs, accumulating the result.
/// For N models with weights w_i, normalizes weights to sum to 1,
/// then iteratively interpolates: result = slerp(result, model_i, w_i / running_sum).
pub(crate) fn multi_slerp_tensors(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    weights: &[f32],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    assert!(
        all_tensors.len() >= 2,
        "MultiSLERP requires at least 2 models"
    );
    assert_eq!(all_tensors.len(), weights.len());

    let sum: f32 = weights.iter().sum();
    let norm_weights: Vec<f32> = weights.iter().map(|w| w / sum).collect();

    // Start with the first model
    let mut result = all_tensors[0].clone();
    let mut accum_weight = norm_weights[0];

    // Iteratively SLERP each subsequent model
    for i in 1..all_tensors.len() {
        let w_i = norm_weights[i];
        // Interpolation parameter: fraction of new model in accumulated result
        let t = w_i / (accum_weight + w_i);
        result = nuslerp_tensors(&result, &all_tensors[i], t);
        accum_weight += w_i;
    }

    result
}

/// DELLA: Task arithmetic + adaptive magnitude pruning.
///
/// Like DARE but with magnitude-adaptive drop rates: elements with smaller
/// magnitude deltas are dropped more aggressively.
///
/// Reference: Adaptive DARE variant with magnitude-proportional retention.
pub(crate) fn della_merge(
    base_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    task_models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    drop_rate: f32,
    seed: u64,
    weights: Option<&[f32]>,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();
    let num_models = task_models.len();
    let default_weights: Vec<f32> = vec![1.0 / num_models as f32; num_models];
    let w = weights.unwrap_or(&default_weights);

    for (tensor_idx, (name, (base_data, shape))) in base_tensors.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tensor_idx as u64));
        let mut merged_delta = vec![0.0f32; base_data.len()];

        for (model_idx, model_tensors) in task_models.iter().enumerate() {
            let (model_data, _) = model_tensors.get(name).expect("validated above");
            let weight = w[model_idx];

            // Compute per-tensor max magnitude for adaptive scaling
            let max_mag: f32 = model_data
                .iter()
                .zip(base_data.iter())
                .map(|(&m, &b)| (m - b).abs())
                .fold(0.0f32, f32::max);

            if max_mag < 1e-12 {
                continue;
            }

            for (i, (&m_val, &b_val)) in model_data.iter().zip(base_data.iter()).enumerate() {
                let delta = m_val - b_val;
                // Adaptive drop rate: smaller deltas have higher drop probability
                let magnitude_ratio = delta.abs() / max_mag;
                let adaptive_drop = drop_rate * (1.0 - magnitude_ratio);
                let keep = rng.random::<f32>() >= adaptive_drop;
                if keep {
                    let rescale = 1.0 / (1.0 - adaptive_drop).max(1e-6);
                    merged_delta[i] += delta * rescale * weight;
                }
            }
        }

        let result: Vec<f32> = base_data
            .iter()
            .zip(merged_delta.iter())
            .map(|(&b, &d)| b + d)
            .collect();

        merged.insert(name.clone(), (result, shape.clone()));
    }

    merged
}

/// Breadcrumbs: Task arithmetic + outlier removal.
///
/// Removes outlier deltas (elements where |delta| > k * std(delta)) before
/// summing task vectors. Prevents extreme weight shifts from dominating.
///
/// Reference: Davari & Belilovsky 2023, "Model Breadcrumbs"
pub(crate) fn breadcrumbs_merge(
    base_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    task_models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    scales: &[f32],
    outlier_k: f32,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    for (name, (base_data, shape)) in base_tensors {
        let mut result = base_data.clone();

        for (model_idx, model_tensors) in task_models.iter().enumerate() {
            let (model_data, _) = model_tensors.get(name).expect("validated above");
            let scale = scales.get(model_idx).copied().unwrap_or(1.0);

            // Compute delta statistics for outlier detection
            let deltas: Vec<f32> = model_data
                .iter()
                .zip(base_data.iter())
                .map(|(&m, &b)| m - b)
                .collect();

            let (mean, std) = delta_mean_std(&deltas);
            let threshold = outlier_k * std;

            // Apply task arithmetic with outlier removal
            for (i, &delta) in deltas.iter().enumerate() {
                if (delta - mean).abs() <= threshold {
                    result[i] += scale * delta;
                }
                // Outliers are dropped (breadcrumbs removed)
            }
        }

        merged.insert(name.clone(), (result, shape.clone()));
    }

    merged
}

/// Compute mean and standard deviation of deltas.
fn delta_mean_std(deltas: &[f32]) -> (f32, f32) {
    if deltas.is_empty() {
        return (0.0, 0.0);
    }
    let n = deltas.len() as f64;
    let sum: f64 = deltas.iter().map(|&x| f64::from(x)).sum();
    let mean = sum / n;
    let var: f64 = deltas
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean as f32, var.sqrt() as f32)
}

/// SCE: Adaptive matrix-level weighting based on variance.
///
/// For each tensor, computes the variance of weights across models
/// and uses high-variance tensors' dominant model more strongly.
/// Low-variance tensors (models agree) use equal weights.
///
/// Reference: Stoica et al. 2024, "ZipIt! Merging Models from Different Tasks
/// without Training"
pub(crate) fn sce_merge(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    base_weights: &[f32],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();
    let reference = &all_tensors[0];
    let num_models = all_tensors.len();

    let sum: f32 = base_weights.iter().sum();
    let norm_weights: Vec<f32> = base_weights.iter().map(|w| w / sum).collect();

    for (name, (_, shape)) in reference {
        // Collect all model data for this tensor
        let model_data: Vec<&Vec<f32>> = all_tensors
            .iter()
            .map(|t| &t.get(name).expect("validated above").0)
            .collect();

        let data_len = model_data[0].len();

        // Compute per-model variance contribution for this tensor
        let variances: Vec<f64> = (0..num_models)
            .map(|m| {
                model_data[m]
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
                    / data_len as f64
            })
            .collect();

        // Adapt weights: models with higher variance (more distinctive) get more weight
        let total_var: f64 = variances.iter().sum();
        let adaptive_weights: Vec<f32> = if total_var < 1e-12 {
            norm_weights.clone()
        } else {
            // Blend: 50% base weights + 50% variance-proportional weights
            (0..num_models)
                .map(|m| {
                    let var_weight = (variances[m] / total_var) as f32;
                    0.5 * norm_weights[m] + 0.5 * var_weight
                })
                .collect()
        };

        // Renormalize adaptive weights
        let w_sum: f32 = adaptive_weights.iter().sum();
        let final_weights: Vec<f32> = adaptive_weights.iter().map(|w| w / w_sum).collect();

        // Weighted merge with adaptive weights
        let mut merged_data = vec![0.0f32; data_len];
        for (m, data) in model_data.iter().enumerate() {
            let weight = final_weights[m];
            for (i, &val) in data.iter().enumerate() {
                merged_data[i] += val * weight;
            }
        }

        merged.insert(name.clone(), (merged_data, shape.clone()));
    }

    merged
}

// ============================================================================
// Passthrough / Frankenmerge (GH-443)
// ============================================================================

/// Passthrough merge: direct tensor copy for layer stacking (frankenmerge).
///
/// Takes specific layers from specific models and concatenates them.
/// Each layer range specifies (model_index, start_layer, end_layer_exclusive).
/// Non-layer tensors (embed, lm_head, norm) are taken from the first model.
///
/// Reference: Inspired by MergeKit's passthrough strategy for creating
/// "frankenmerge" models with custom layer compositions.
pub(crate) fn passthrough_merge(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    layer_ranges: &[(usize, usize, usize)],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    // Build output layer mapping: output_layer -> (model_idx, source_layer)
    let mut layer_map: Vec<(usize, usize)> = Vec::new();
    for &(model_idx, start, end) in layer_ranges {
        for layer in start..end {
            layer_map.push((model_idx, layer));
        }
    }

    // Collect all tensor names from all models
    let mut all_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for model in all_tensors {
        for name in model.keys() {
            all_names.insert(name.clone());
        }
    }

    for name in &all_names {
        if let Some((layer_num, prefix, suffix)) = parse_layer_tensor_name(name) {
            // Find which output position maps from this source layer
            for (out_idx, &(model_idx, src_layer)) in layer_map.iter().enumerate() {
                if src_layer == layer_num {
                    if let Some(model) = all_tensors.get(model_idx) {
                        if let Some((data, shape)) = model.get(name) {
                            let out_name = format!("{prefix}{out_idx}{suffix}");
                            merged.insert(out_name, (data.clone(), shape.clone()));
                        }
                    }
                }
            }
        } else {
            // Non-layer tensor: take from first model that has it
            for model in all_tensors {
                if let Some((data, shape)) = model.get(name) {
                    merged.insert(name.clone(), (data.clone(), shape.clone()));
                    break;
                }
            }
        }
    }

    merged
}

/// Parse a layer tensor name into (layer_number, prefix, suffix).
/// Returns None for non-layer tensors.
fn parse_layer_tensor_name(name: &str) -> Option<(usize, &str, &str)> {
    // Try "layers.N." pattern
    if let Some(pos) = name.find("layers.") {
        let after_layers = &name[pos + 7..];
        if let Some(dot_pos) = after_layers.find('.') {
            if let Ok(num) = after_layers[..dot_pos].parse::<usize>() {
                let prefix = &name[..pos + 7];
                let suffix = &after_layers[dot_pos..];
                return Some((num, prefix, suffix));
            }
        }
    }
    // Try "blk.N." pattern (GGUF style)
    if let Some(pos) = name.find("blk.") {
        let after_blk = &name[pos + 4..];
        if let Some(dot_pos) = after_blk.find('.') {
            if let Ok(num) = after_blk[..dot_pos].parse::<usize>() {
                let prefix = &name[..pos + 4];
                let suffix = &after_blk[dot_pos..];
                return Some((num, prefix, suffix));
            }
        }
    }
    None
}
