use super::super::super::super::*;

// =========================================================================
// F276-F278: QuantType, LayerActivationTrace anomalies, ModelActivationTrace
// =========================================================================

/// F276: All QuantType variants bits_per_element coverage
#[test]
fn test_f276_quant_type_all_variants() {
    // Test all QuantType variants for bits_per_element
    assert_eq!(QuantType::F32.bits_per_element(), 32.0);
    assert_eq!(QuantType::F16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Bf16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Q8_0.bits_per_element(), 8.0);
    assert_eq!(QuantType::Q6_K.bits_per_element(), 6.5);
    assert_eq!(QuantType::Q5_K.bits_per_element(), 5.5);
    assert_eq!(QuantType::Q4_0.bits_per_element(), 4.5);
    assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);
    assert_eq!(QuantType::Q3_K.bits_per_element(), 3.5);
    assert_eq!(QuantType::Q2_K.bits_per_element(), 2.5);

    // Compression ratios for all types
    assert!((QuantType::Bf16.compression_ratio() - 2.0).abs() < 0.01);
    assert!((QuantType::Q8_0.compression_ratio() - 4.0).abs() < 0.01);
    assert!((QuantType::Q6_K.compression_ratio() - 4.92).abs() < 0.1);
    assert!((QuantType::Q5_K.compression_ratio() - 5.82).abs() < 0.1);
    assert!((QuantType::Q3_K.compression_ratio() - 9.14).abs() < 0.1);
    assert!((QuantType::Q2_K.compression_ratio() - 12.8).abs() < 0.1);
}

/// F277: LayerActivationTrace all anomaly paths
#[test]
fn test_f277_layer_anomaly_all_paths() {
    // Test post_norm anomaly
    let mut layer = LayerActivationTrace::new(0);
    layer.post_norm_stats = TensorStats::from_slice(&[f32::NAN]);
    assert!(layer.has_anomaly());
    let desc = layer.anomaly_description().unwrap();
    assert!(desc.contains("post_norm"));

    // Test post_attn anomaly
    let mut layer2 = LayerActivationTrace::new(1);
    layer2.post_attn_stats = TensorStats::from_slice(&[f32::INFINITY]);
    assert!(layer2.has_anomaly());
    let desc2 = layer2.anomaly_description().unwrap();
    assert!(desc2.contains("post_attn"));

    // Test post_ffn anomaly
    let mut layer3 = LayerActivationTrace::new(2);
    layer3.post_ffn_stats = TensorStats::from_slice(&[f32::NAN]);
    assert!(layer3.has_anomaly());
    let desc3 = layer3.anomaly_description().unwrap();
    assert!(desc3.contains("post_ffn"));

    // Test output anomaly
    let mut layer4 = LayerActivationTrace::new(3);
    layer4.output_stats = TensorStats::from_slice(&[1e7]);
    assert!(layer4.has_anomaly());
    let desc4 = layer4.anomaly_description().unwrap();
    assert!(desc4.contains("output"));

    // Test residual dominance
    let mut layer5 = LayerActivationTrace::new(4);
    layer5.residual_ratio = 0.995;
    assert!(layer5.has_anomaly());
    let desc5 = layer5.anomaly_description().unwrap();
    assert!(desc5.contains("residual"));
}

/// F278: ModelActivationTrace full workflow
#[test]
fn test_f278_model_activation_trace_workflow() {
    // Test with_capacity
    let mut trace = ModelActivationTrace::with_capacity(32);
    assert_eq!(trace.layers.capacity(), 32);

    // Add normal layers
    for i in 0..3 {
        let layer = LayerActivationTrace::new(i);
        trace.add_layer(layer);
    }
    assert!(!trace.has_anomaly);

    // Add layer with anomaly
    let mut bad_layer = LayerActivationTrace::new(3);
    bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN, 1.0, 2.0]);
    trace.add_layer(bad_layer);
    assert!(trace.has_anomaly);
    assert!(trace.anomaly_desc.is_some());

    // Test finalize with embedding anomaly
    let mut trace2 = ModelActivationTrace::with_capacity(2);
    trace2.embedding_stats = TensorStats::from_slice(&[f32::INFINITY]);
    trace2.finalize();
    assert!(trace2.has_anomaly);
    assert!(trace2.anomaly_desc.as_ref().unwrap().contains("Embedding"));

    // Test finalize with logits anomaly
    let mut trace3 = ModelActivationTrace::with_capacity(2);
    trace3.logits_stats = TensorStats::from_slice(&[f32::NAN]);
    trace3.finalize();
    assert!(trace3.has_anomaly);
    assert!(trace3.anomaly_desc.as_ref().unwrap().contains("Logits"));
}
