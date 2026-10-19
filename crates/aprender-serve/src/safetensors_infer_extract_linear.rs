// =========================================================================
// extract_linear_layer_generic (GH-278 / ALB-010, Refs #5534)
// =========================================================================

/// In-memory `TensorSource` keyed by tensor name.
struct MapSource(std::collections::HashMap<String, Vec<f32>>);

impl TensorSource for MapSource {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("missing tensor {name}"),
            })
    }
    fn has_tensor(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn tensor_names(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }
    fn tensor_shape(&self, _name: &str) -> Option<Vec<usize>> {
        None
    }
}

// hidden 4, key_dim 2 (1 head x 2), value_dim 2 (1 head x 2), intermediate 3.
const GDN_HIDDEN: usize = 4;
const GDN_INTER: usize = 3;

fn gdn_config() -> SafetensorsConfig {
    SafetensorsConfig {
        linear_key_head_dim: Some(2),
        linear_value_head_dim: Some(2),
        linear_num_key_heads: Some(1),
        linear_num_value_heads: Some(1),
        ..Default::default()
    }
}

fn ramp(len: usize, start: f32) -> Vec<f32> {
    (0..len).map(|i| start + i as f32).collect()
}

/// Tensors every layer needs regardless of the projection convention.
fn gdn_common(attn_sub: &str) -> Vec<(String, Vec<f32>)> {
    let p = "model.layers.0";
    vec![
        (format!("{p}.input_layernorm.weight"), vec![1.0; GDN_HIDDEN]),
        (
            format!("{p}.post_attention_layernorm.weight"),
            vec![2.0; GDN_HIDDEN],
        ),
        (
            format!("{p}.{attn_sub}.out_proj.weight"),
            ramp(GDN_HIDDEN * 2, 500.0),
        ),
        (format!("{p}.{attn_sub}.conv1d.weight"), vec![0.5; 6]),
        (format!("{p}.{attn_sub}.A_log"), vec![0.1]),
        (format!("{p}.{attn_sub}.dt_bias"), vec![0.2]),
        (format!("{p}.{attn_sub}.norm.weight"), vec![1.0; 2]),
    ]
}

fn gdn_combined_source(qkvz_len: usize) -> MapSource {
    let p = "model.layers.0";
    let mut m: std::collections::HashMap<_, _> = gdn_common("self_attn").into_iter().collect();
    m.insert(
        format!("{p}.self_attn.in_proj_qkvz.weight"),
        ramp(qkvz_len, 0.0),
    );
    // BA: b = first num_value_heads*hidden (4), a = the rest (4)
    m.insert(format!("{p}.self_attn.in_proj_ba.weight"), ramp(8, 100.0));
    MapSource(m)
}

fn gdn_separate_source(qkv_len: usize, dense_ffn: bool) -> MapSource {
    let p = "model.layers.0";
    let mut m: std::collections::HashMap<_, _> = gdn_common("linear_attn").into_iter().collect();
    m.insert(
        format!("{p}.linear_attn.in_proj_qkv.weight"),
        ramp(qkv_len, 0.0),
    );
    m.insert(format!("{p}.linear_attn.in_proj_z.weight"), ramp(8, 200.0));
    m.insert(format!("{p}.linear_attn.in_proj_b.weight"), ramp(4, 300.0));
    m.insert(format!("{p}.linear_attn.in_proj_a.weight"), ramp(4, 400.0));
    if dense_ffn {
        m.insert(
            format!("{p}.mlp.gate_proj.weight"),
            ramp(GDN_INTER * GDN_HIDDEN, 10.0),
        );
        m.insert(
            format!("{p}.mlp.up_proj.weight"),
            ramp(GDN_INTER * GDN_HIDDEN, 20.0),
        );
        m.insert(
            format!("{p}.mlp.down_proj.weight"),
            ramp(GDN_HIDDEN * GDN_INTER, 30.0),
        );
    }
    MapSource(m)
}

fn extract(src: &MapSource) -> Result<AprTransformerLayer> {
    SafetensorsToAprConverter::extract_linear_layer_generic(
        src,
        0,
        GDN_HIDDEN,
        GDN_INTER,
        &gdn_config(),
        "model",
    )
}

#[test]
fn extract_linear_combined_qkvz_splits_qkv_and_ba() {
    // qkvz = 2*key_dim + 2*value_dim = 8 rows x hidden 4 = 32
    let layer = extract(&gdn_combined_source(32)).expect("combined layout extracts");
    // qkv = Q|K|V = first (2+2+2)*4 = 24 values in order; z (8) is split off, not in qkv
    assert_eq!(layer.qkv_weight, ramp(24, 0.0));
    // No dense FFN in the source: gate absent, up/down zero-filled to their shapes
    assert!(layer.ffn_gate_weight.is_none());
    assert_eq!(layer.ffn_up_weight, vec![0.0; GDN_INTER * GDN_HIDDEN]);
    assert_eq!(layer.ffn_down_weight, vec![0.0; GDN_HIDDEN * GDN_INTER]);
    assert_eq!(layer.attn_norm_weight, vec![1.0; GDN_HIDDEN]);
}

#[test]
fn extract_linear_combined_qkvz_size_mismatch_is_refused() {
    let err = extract(&gdn_combined_source(31)).expect_err("short qkvz must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("GH-278") && msg.contains("expected 32, got 31"),
        "{msg}"
    );
}

#[test]
fn extract_linear_separate_layout_moves_qkv_whole() {
    // qkv = 2*key_dim + value_dim = 6 rows x hidden 4 = 24
    let layer = extract(&gdn_separate_source(24, false)).expect("separate layout extracts");
    assert_eq!(layer.qkv_weight, ramp(24, 0.0));
    assert!(layer.ffn_gate_weight.is_none());
    assert_eq!(layer.ffn_up_weight.len(), GDN_INTER * GDN_HIDDEN);
}

#[test]
fn extract_linear_separate_layout_size_mismatch_is_refused() {
    let err = extract(&gdn_separate_source(23, false)).expect_err("short qkv must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("ALB-010") && msg.contains("expected 24, got 23"),
        "{msg}"
    );
}

#[test]
fn extract_linear_separate_layout_with_dense_ffn_keeps_gate_up_down() {
    let layer = extract(&gdn_separate_source(24, true)).expect("dense ffn extracts");
    let gate = layer.ffn_gate_weight.expect("dense gate present");
    assert_eq!(gate.len(), GDN_INTER * GDN_HIDDEN);
    // Values are carried through, not zero-filled
    assert!(gate.iter().any(|v| *v != 0.0));
    assert!(layer.ffn_up_weight.iter().any(|v| *v != 0.0));
    assert!(layer.ffn_down_weight.iter().any(|v| *v != 0.0));
}

#[test]
fn extract_linear_missing_tensor_is_an_error_not_a_default() {
    let mut src = gdn_separate_source(24, false);
    src.0.remove("model.layers.0.linear_attn.A_log");
    let err = extract(&src).expect_err("missing A_log must fail");
    assert!(err.to_string().contains("A_log"), "{err}");
}
