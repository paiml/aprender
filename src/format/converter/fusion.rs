/// Collect and optionally transpose source tensors for one fusion rule + layer.
/// Returns `(concatenated_data, per_source_shapes)` or `None` if any source is missing.
fn collect_fusion_sources(
    rule: &FusionExportRule,
    layer: usize,
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    needs_transpose: bool,
) -> Option<(Vec<f32>, Vec<Vec<usize>>)> {
    let is_weight = rule.gguf_suffix.ends_with(".weight");
    let mut all_data: Vec<f32> = Vec::new();
    let mut all_shapes: Vec<Vec<usize>> = Vec::new();

    for apr_suffix in &rule.apr_suffixes {
        let apr_name = format!("model.layers.{layer}.{apr_suffix}");
        let (data, shape) = tensors.get(&apr_name)?;

        if needs_transpose && is_weight && shape.len() == 2 {
            let transposed = transpose_2d_f32(data, shape[0], shape[1]);
            all_data.extend_from_slice(&transposed);
            all_shapes.push(vec![shape[1], shape[0]]);
        } else {
            all_data.extend_from_slice(data);
            all_shapes.push(shape.clone());
        }
    }
    Some((all_data, all_shapes))
}

/// GH-277: Build fused tensors for the F32 export path.
///
/// For each fusion rule and each layer, looks up source tensors by APR name,
/// concatenates their f32 data, and returns the fused GGUF tensors.
fn build_fused_tensors_f32(
    mapper: &GgufNameMapper,
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    use_q4k: bool,
) -> Vec<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::{GgmlType, GgufTensor};

    let rules = mapper.fusion_rules();
    if rules.is_empty() {
        return Vec::new();
    }

    let num_layers = detect_num_layers_from_names(tensors.keys().map(|s| s.as_str()));
    let needs_transpose = mapper.needs_transpose();
    let mut fused = Vec::new();

    for rule in rules {
        for layer in 0..num_layers {
            let Some((all_data, all_shapes)) =
                collect_fusion_sources(rule, layer, tensors, needs_transpose)
            else {
                continue;
            };

            let Some(fused_shape) = compute_fused_shape(&all_shapes) else {
                continue;
            };

            let gguf_shape = shape_to_gguf(&fused_shape);
            let gguf_name = format!("blk.{layer}.{}", rule.gguf_suffix);

            let (dtype, bytes) = if use_q4k && fused_shape.len() == 2 && all_data.len() >= 256 {
                let gguf_shape_usize = vec![fused_shape[1], fused_shape[0]];
                let q4k_bytes = super::quantize_q4_k_matrix(&all_data, &gguf_shape_usize);
                (GgmlType::Q4K, q4k_bytes)
            } else {
                let f32_bytes: Vec<u8> = all_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (GgmlType::F32, f32_bytes)
            };

            eprintln!(
                "[GH-277] Fused `{}` from {} sources ({} elements)",
                gguf_name,
                rule.apr_suffixes.len(),
                all_data.len()
            );

            fused.push(GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: bytes,
            });
        }
    }

    fused
}

/// GH-277: Build fused tensors for the raw APR→GGUF export path.
///
/// For each fusion rule and each layer, reads raw tensor bytes from the APR reader,
/// concatenates them, and returns fused GGUF tensors.
/// Map APR tensor dtype to GGML type for raw byte fusion.
fn apr_dtype_to_ggml(dtype: crate::format::v2::TensorDType) -> crate::format::gguf::GgmlType {
    use crate::format::gguf::GgmlType;
    use crate::format::v2::TensorDType;
    match dtype {
        TensorDType::F32 => GgmlType::F32,
        TensorDType::F16 => GgmlType::F16,
        TensorDType::Q4K => GgmlType::Q4K,
        TensorDType::Q6K => GgmlType::Q6K,
        TensorDType::Q8 => GgmlType::Q8_0,
        _ => GgmlType::F32,
    }
}

