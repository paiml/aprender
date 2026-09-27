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

            // PMAT-690 defects 2+3 (2026-05-17): same divisibility +
            // shape-passing rules as encode_gguf_data. Q4_K needs K
            // (= fused_shape[1] = APR's inner dim) to be 256-divisible,
            // and the function must receive the APR-native shape directly
            // (no swap) so it pads/slices along the correct dim.
            let q4k_eligible = use_q4k
                && fused_shape.len() == 2
                && all_data.len() >= 256
                && fused_shape[1] % 256 == 0;
            let (dtype, bytes) = if q4k_eligible {
                let q4k_bytes = super::quantize_q4_k_matrix(&all_data, &fused_shape);
                (GgmlType::Q4K, q4k_bytes)
            } else {
                if use_q4k && fused_shape.len() == 2 && all_data.len() >= 256 {
                    eprintln!(
                        "[GH-277-Q4K-FALLBACK] fused blk.{}.{} (shape {:?}) — \
                         K={} not divisible by 256; falling back to F32",
                        layer,
                        rule.gguf_suffix,
                        fused_shape,
                        fused_shape[1]
                    );
                }
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
///
/// GH-439 (poka-yoke): Returns `None` for dtypes with no GGUF equivalent,
/// instead of silently falling back to F32 (the GH-186 pattern).
fn apr_dtype_to_ggml(dtype: crate::format::v2::TensorDType) -> Option<crate::format::gguf::GgmlType> {
    use crate::format::gguf::GgmlType;
    use crate::format::v2::TensorDType;
    match dtype {
        TensorDType::F32 => Some(GgmlType::F32),
        TensorDType::F16 => Some(GgmlType::F16),
        TensorDType::Q4K => Some(GgmlType::Q4K),
        TensorDType::Q6K => Some(GgmlType::Q6K),
        // AprQ8 (APR-native single-scale 8-bit, 4+N bytes) is NOT GGML Q8_0
        // (per-32-block, ceil(N/32)*34 bytes) — relabeling produces corrupt
        // GGUF. Reject alongside AprQ4 (symmetric with metadata.rs export arm).
        TensorDType::AprQ8
        | TensorDType::BF16
        | TensorDType::F64
        | TensorDType::I32
        | TensorDType::I64
        | TensorDType::I8
        | TensorDType::U8
        | TensorDType::AprQ4 => {
            eprintln!(
                "[GH-439] apr_dtype_to_ggml: unsupported dtype {:?} — \
                 no GGUF equivalent, skipping tensor",
                dtype
            );
            None
        }
    }
}


/// ONT-10 L4 simulation witness for `apr_dtype_to_ggml` (fusion.rs:118).
///
/// The Lean model `exportDtype` (`ProvableContracts/Theorems/GgufExportSymmetry/Roundtrip.lean`)
/// transcribes the match above arm for arm; `exportDtypeTable_eq` there PROVES the table below.
/// This test runs the Rust fn on every `TensorDType` variant and requires the same table, so the
/// Lean model and the Rust item cannot drift apart silently.
#[cfg(test)]
mod export_tests_ges_l4_witness {
    use super::apr_dtype_to_ggml;
    use crate::format::v2::TensorDType;

    /// `#eval exportDtypeTable` — Lean output, verbatim.
    const LEAN_EXPORT_DTYPE_TABLE: &str = "F32=>F32;F16=>F16;BF16=>none;F64=>none;I32=>none;\
I64=>none;I8=>none;U8=>none;AprQ4=>none;AprQ8=>none;Q4K=>Q4K;Q6K=>Q6K";

    /// Every variant, in declaration order. The match makes adding a variant a compile error
    /// here until the domain (and the Lean `allAprDTypes`) grows with it.
    fn all_dtypes() -> [TensorDType; 12] {
        let _exhaustive = |d: TensorDType| match d {
            TensorDType::F32
            | TensorDType::F16
            | TensorDType::BF16
            | TensorDType::F64
            | TensorDType::I32
            | TensorDType::I64
            | TensorDType::I8
            | TensorDType::U8
            | TensorDType::AprQ4
            | TensorDType::AprQ8
            | TensorDType::Q4K
            | TensorDType::Q6K => (),
        };
        [
            TensorDType::F32,
            TensorDType::F16,
            TensorDType::BF16,
            TensorDType::F64,
            TensorDType::I32,
            TensorDType::I64,
            TensorDType::I8,
            TensorDType::U8,
            TensorDType::AprQ4,
            TensorDType::AprQ8,
            TensorDType::Q4K,
            TensorDType::Q6K,
        ]
    }

    #[test]
    fn apr_dtype_to_ggml_matches_lean_export_dtype_on_all_variants() {
        let rust_table = all_dtypes()
            .iter()
            .map(|&d| {
                let g = apr_dtype_to_ggml(d).map_or_else(|| "none".to_string(), |g| format!("{g:?}"));
                format!("{d:?}=>{g}")
            })
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(rust_table, LEAN_EXPORT_DTYPE_TABLE);
    }
}
