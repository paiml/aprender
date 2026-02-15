
// ============================================================================
// TIES (Trim, Elect Sign, Merge)
// ============================================================================

/// TIES merge: Trim, Elect Sign, Merge.
///
/// 1. Compute task vectors: delta_i = model_i - base
/// 2. Trim: zero elements where |delta| < density * max(|delta|) per tensor
/// 3. Elect sign: majority vote across models per element
/// 4. Merge: average values agreeing with elected sign
/// 5. Result: base + merged_delta
fn ties_merge(
    base_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    task_models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    density: f32,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    for (name, (base_data, shape)) in base_tensors {
        // Step 1: Compute task vectors (deltas)
        let deltas: Vec<Vec<f32>> = task_models
            .iter()
            .map(|model| {
                let (model_data, _) = model.get(name).expect("validated above");
                model_data
                    .iter()
                    .zip(base_data.iter())
                    .map(|(&m, &b)| m - b)
                    .collect()
            })
            .collect();

        // Step 2: Trim small values per delta
        let trimmed: Vec<Vec<f32>> = deltas
            .iter()
            .map(|delta| ties_trim(delta, density))
            .collect();

        // Step 3 + 4: Elect sign and merge
        let merged_delta = ties_elect_and_merge(&trimmed, base_data.len());

        // Step 5: base + merged_delta
        let result: Vec<f32> = base_data
            .iter()
            .zip(merged_delta.iter())
            .map(|(&b, &d)| b + d)
            .collect();

        merged.insert(name.clone(), (result, shape.clone()));
    }

    merged
}

/// Trim elements with magnitude below density * max(|delta|).
fn ties_trim(delta: &[f32], density: f32) -> Vec<f32> {
    let max_abs = delta
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);

    if max_abs < 1e-12 {
        return vec![0.0; delta.len()];
    }

    let threshold = density * max_abs;
    delta
        .iter()
        .map(|&x| if x.abs() >= threshold { x } else { 0.0 })
        .collect()
}

/// Elect sign per element (majority vote) and merge agreeing values.
fn ties_elect_and_merge(trimmed_deltas: &[Vec<f32>], len: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; len];
    let num_models = trimmed_deltas.len();

    for i in 0..len {
        // Count positive vs negative votes (ignoring zeros from trimming)
        let mut pos_count = 0i32;
        let mut neg_count = 0i32;
        for delta in trimmed_deltas {
            let val = delta[i];
            if val > 0.0 {
                pos_count += 1;
            } else if val < 0.0 {
                neg_count += 1;
            }
        }

        // Elected sign: positive if pos_count >= neg_count, else negative
        let elected_positive = pos_count >= neg_count;

        // Average values that agree with the elected sign
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for delta in trimmed_deltas {
            let val = delta[i];
            let agrees = (elected_positive && val > 0.0) || (!elected_positive && val < 0.0);
            if agrees {
                sum += val;
                count += 1;
            }
        }

        if count > 0 {
            // Scale by num_models (not count) to preserve magnitude relative to all models
            result[i] = sum / num_models as f32;
        }
    }

    result
}

// ============================================================================
// DARE (Drop And Rescale)
// ============================================================================

/// DARE merge: Drop And Rescale.
///
/// 1. Compute task vectors: delta_i = model_i - base
/// 2. Randomly drop elements with probability `drop_rate` (using seeded RNG)
/// 3. Rescale remaining by 1/(1 - drop_rate)
/// 4. Average rescaled deltas (with optional weights)
/// 5. Result: base + avg(rescaled_deltas)
fn dare_merge(
    base_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    task_models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    drop_rate: f32,
    seed: u64,
    weights: Option<&[f32]>,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();
    let rescale = 1.0 / (1.0 - drop_rate);
    let num_models = task_models.len();

    // Equal weights if not specified
    let default_weights: Vec<f32> = vec![1.0 / num_models as f32; num_models];
    let w = weights.unwrap_or(&default_weights);

    // Each tensor gets its own RNG seeded from base seed + tensor index
    // to ensure determinism independent of tensor iteration order
    for (tensor_idx, (name, (base_data, shape))) in base_tensors.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tensor_idx as u64));

        let mut merged_delta = vec![0.0f32; base_data.len()];

        for (model_idx, model_tensors) in task_models.iter().enumerate() {
            let (model_data, _) = model_tensors.get(name).expect("validated above");
            let weight = w[model_idx];

            for (i, (&m_val, &b_val)) in model_data.iter().zip(base_data.iter()).enumerate() {
                let delta = m_val - b_val;
                // Drop with probability drop_rate; keep with probability (1 - drop_rate)
                let keep: bool = rng.gen::<f32>() >= drop_rate;
                if keep {
                    merged_delta[i] += delta * rescale * weight;
                }
            }
        }

        // Result: base + merged_delta
        let result: Vec<f32> = base_data
            .iter()
            .zip(merged_delta.iter())
            .map(|(&b, &d)| b + d)
            .collect();

        merged.insert(name.clone(), (result, shape.clone()));
    }

    merged
}

// ============================================================================
// MAIN MERGE ENTRY POINT
// ============================================================================

/// Merge multiple models into one
///
/// # Arguments
///
/// * `inputs` - Input model paths (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Merge options
///
/// # Returns
///
/// Merge report with statistics
///
/// # Errors
///
/// Returns error if:
/// - Less than 2 input files
/// - Input files don't exist
/// - Models have incompatible tensor shapes
/// - Strategy not supported
/// - SLERP with != 2 models
/// - TIES/DARE without base model
pub fn apr_merge<P: AsRef<Path>>(
    inputs: &[P],
    output: P,
    options: MergeOptions,
) -> Result<MergeReport> {
    // Validate inputs and options
    validate_merge_options(inputs, &options)?;

    // Load all models
    let all_tensors = load_all_models(inputs)?;

    // Verify tensor compatibility
    verify_tensor_compatibility(&all_tensors)?;

    // Dispatch to strategy-specific merge
    let merged = match options.strategy {
        MergeStrategy::Average | MergeStrategy::Weighted => {
            let weights = calculate_merge_weights(inputs.len(), &options)?;
            merge_tensors(&all_tensors, &weights)
        }
        MergeStrategy::Slerp => {
            // Use first weight as interpolation t, default 0.5
            let t = options
                .weights
                .as_ref()
                .and_then(|w| w.first().copied())
                .unwrap_or(0.5);
            slerp_tensors(&all_tensors[0], &all_tensors[1], t)
        }
        MergeStrategy::Ties => {
            let base_path = options.base_model.as_ref().expect("validated above");
            let base_tensors = load_model_tensors(base_path)?;
            // Verify base is compatible with task models
            verify_tensor_compatibility(&[base_tensors.clone(), all_tensors[0].clone()])?;
            ties_merge(&base_tensors, &all_tensors, options.density)
        }
        MergeStrategy::Dare => {
            let base_path = options.base_model.as_ref().expect("validated above");
            let base_tensors = load_model_tensors(base_path)?;
            verify_tensor_compatibility(&[base_tensors.clone(), all_tensors[0].clone()])?;
            dare_merge(
                &base_tensors,
                &all_tensors,
                options.drop_rate,
                options.seed,
                options.weights.as_deref(),
            )
        }
    };

    // Save merged model
    let output_path = output.as_ref();
    save_safetensors(output_path, &merged).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save merged model: {e}"),
    })?;

    // Get output file size
    let output_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(MergeReport {
        model_count: inputs.len(),
        tensor_count: merged.len(),
        output_size,
        strategy: options.strategy,
        weights_used: options.weights,
    })
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod tests;
