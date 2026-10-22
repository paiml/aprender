// Advanced merge strategy tests (GH-442)

use super::{apr_merge, load_model_tensors, tempdir, MergeOptions, MergeStrategy};
use std::collections::BTreeMap;

fn make_tensor_map(
    name: &str,
    data: Vec<f32>,
    shape: Vec<usize>,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut m = BTreeMap::new();
    m.insert(name.to_string(), (data, shape));
    m
}

fn make_two_tensor_models() -> (
    BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>>,
) {
    let base = make_tensor_map("w", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let model_a = make_tensor_map("w", vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
    let model_b = make_tensor_map("w", vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    (base, vec![model_a, model_b])
}

// ============================================================================
// Task Arithmetic tests
// ============================================================================

#[test]
fn test_task_arithmetic_basic() {
    let (base, models) = make_two_tensor_models();
    let scales = vec![1.0, 1.0];
    let result = super::task_arithmetic_merge(&base, &models, &scales);
    let (data, _) = result.get("w").unwrap();
    // base + 1*(model_a - base) + 1*(model_b - base) = base + (1,1,1,1) + (2,2,2,2) = (4,5,6,7)
    assert_eq!(data, &[4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_task_arithmetic_scaled() {
    let (base, models) = make_two_tensor_models();
    let scales = vec![0.5, 0.5];
    let result = super::task_arithmetic_merge(&base, &models, &scales);
    let (data, _) = result.get("w").unwrap();
    // base + 0.5*(1,1,1,1) + 0.5*(2,2,2,2) = (1,2,3,4) + (0.5,0.5,0.5,0.5) + (1,1,1,1) = (2.5,3.5,4.5,5.5)
    for (i, expected) in [2.5, 3.5, 4.5, 5.5].iter().enumerate() {
        assert!(
            (data[i] - expected).abs() < 1e-6,
            "idx {}: got {}, expected {}",
            i,
            data[i],
            expected
        );
    }
}

#[test]
fn test_task_arithmetic_zero_scale() {
    let (base, models) = make_two_tensor_models();
    let scales = vec![0.0, 0.0];
    let result = super::task_arithmetic_merge(&base, &models, &scales);
    let (data, _) = result.get("w").unwrap();
    // No deltas applied: result = base
    assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// NuSLERP tests
// ============================================================================

#[test]
fn test_nuslerp_basic() {
    let model_a = make_tensor_map("w", vec![1.0, 0.0, 0.0], vec![3]);
    let model_b = make_tensor_map("w", vec![0.0, 1.0, 0.0], vec![3]);
    let result = super::nuslerp_tensors(&model_a, &model_b, 0.5);
    let (data, _) = result.get("w").unwrap();
    // At t=0.5 between orthogonal unit vectors, magnitude should be ~1
    let norm: f64 = data
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "SLERP midpoint should have unit norm, got {}",
        norm
    );
}

#[test]
fn test_nuslerp_nearly_parallel() {
    // Nearly parallel vectors should use nlerp fallback
    let model_a = make_tensor_map("w", vec![1.0, 0.001, 0.0], vec![3]);
    let model_b = make_tensor_map("w", vec![1.0, 0.002, 0.0], vec![3]);
    let result = super::nuslerp_tensors(&model_a, &model_b, 0.5);
    let (data, _) = result.get("w").unwrap();
    assert!((data[0] - 1.0).abs() < 0.01);
    assert!(data[1] > 0.001 && data[1] < 0.002);
}

#[test]
fn test_nuslerp_t0() {
    let model_a = make_tensor_map("w", vec![1.0, 2.0, 3.0], vec![3]);
    let model_b = make_tensor_map("w", vec![4.0, 5.0, 6.0], vec![3]);
    let result = super::nuslerp_tensors(&model_a, &model_b, 0.0);
    let (data, _) = result.get("w").unwrap();
    for (i, &expected) in [1.0, 2.0, 3.0].iter().enumerate() {
        assert!(
            (data[i] - expected).abs() < 0.01,
            "t=0 should return model_a"
        );
    }
}

// ============================================================================
// MultiSLERP tests
// ============================================================================

#[test]
fn test_multi_slerp_two_models() {
    let model_a = make_tensor_map("w", vec![1.0, 0.0], vec![2]);
    let model_b = make_tensor_map("w", vec![0.0, 1.0], vec![2]);
    let models = vec![model_a, model_b];
    let weights = vec![0.5, 0.5];
    let result = super::multi_slerp_tensors(&models, &weights);
    let (data, _) = result.get("w").unwrap();
    // Equal weight SLERP between orthogonal unit vectors
    assert!(
        (data[0] - data[1]).abs() < 0.01,
        "Equal weight should give balanced result"
    );
}

#[test]
fn test_multi_slerp_three_models() {
    let m1 = make_tensor_map("w", vec![1.0, 0.0, 0.0, 0.0], vec![4]);
    let m2 = make_tensor_map("w", vec![0.0, 1.0, 0.0, 0.0], vec![4]);
    let m3 = make_tensor_map("w", vec![0.0, 0.0, 1.0, 0.0], vec![4]);
    let models = vec![m1, m2, m3];
    let weights = vec![1.0, 1.0, 1.0];
    let result = super::multi_slerp_tensors(&models, &weights);
    let (data, _) = result.get("w").unwrap();
    // All three should contribute
    assert!(data[0] > 0.0 && data[1] > 0.0 && data[2] > 0.0);
}

// ============================================================================
// DELLA tests
// ============================================================================

#[test]
fn test_della_basic() {
    let (base, models) = make_two_tensor_models();
    let result = super::della_merge(&base, &models, 0.5, 42, None);
    let (data, _) = result.get("w").unwrap();
    // Should be different from base (deltas applied)
    assert!(data
        .iter()
        .zip([1.0, 2.0, 3.0, 4.0].iter())
        .any(|(a, b)| (a - b).abs() > 0.01));
    // All values should be finite
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn test_della_adaptive_keeps_large_deltas() {
    // Large deltas should have low adaptive drop rate
    let base = make_tensor_map("w", vec![0.0; 100], vec![100]);
    let model = make_tensor_map("w", (0..100).map(|i| i as f32).collect(), vec![100]);
    let result = super::della_merge(&base, &[model], 0.9, 42, None);
    let (data, _) = result.get("w").unwrap();
    // The largest delta (99.0) should almost always be kept
    // (adaptive_drop = 0.9 * (1 - 99/99) = 0.0)
    assert!(data[99] > 0.0, "Largest delta should be kept");
}

#[test]
fn test_della_deterministic() {
    let (base, models) = make_two_tensor_models();
    let r1 = super::della_merge(&base, &models, 0.5, 42, None);
    let r2 = super::della_merge(&base, &models, 0.5, 42, None);
    let (d1, _) = r1.get("w").unwrap();
    let (d2, _) = r2.get("w").unwrap();
    assert_eq!(d1, d2, "Same seed should give identical results");
}

// ============================================================================
// Breadcrumbs tests
// ============================================================================

#[test]
fn test_breadcrumbs_basic() {
    let (base, models) = make_two_tensor_models();
    let scales = vec![1.0, 1.0];
    let result = super::breadcrumbs_merge(&base, &models, &scales, 3.0);
    let (data, _) = result.get("w").unwrap();
    // Uniform deltas: no outliers, result should equal task arithmetic
    let ta_result = super::task_arithmetic_merge(&base, &models, &scales);
    let (ta_data, _) = ta_result.get("w").unwrap();
    for (i, (&a, &b)) in data.iter().zip(ta_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "No outliers: breadcrumbs should equal task arithmetic at idx {}",
            i
        );
    }
}

#[test]
fn test_breadcrumbs_removes_outliers() {
    let base = make_tensor_map("w", vec![0.0; 10], vec![10]);
    // One extreme outlier at index 9, rest are small
    let mut model_data = vec![1.0; 10];
    model_data[9] = 1000.0; // Very extreme outlier
    let model = make_tensor_map("w", model_data, vec![10]);
    let scales = vec![1.0];
    // k=1.5 should remove the extreme outlier
    let result = super::breadcrumbs_merge(&base, &[model], &scales, 1.5);
    let (data, _) = result.get("w").unwrap();
    // The extreme outlier at index 9 should be removed (value remains at base=0)
    assert!(
        data[9].abs() < 1e-6,
        "Outlier should be removed, got {}",
        data[9]
    );
    // Normal values should be kept
    assert!((data[0] - 1.0).abs() < 1e-6, "Normal values should be kept");
}

// ============================================================================
// SCE tests
// ============================================================================

#[test]
fn test_sce_equal_variance() {
    let m1 = make_tensor_map("w", vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let m2 = make_tensor_map("w", vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let models = vec![m1, m2];
    let weights = vec![0.5, 0.5];
    let result = super::sce_merge(&models, &weights);
    let (data, _) = result.get("w").unwrap();
    // Identical models: result = same as input
    assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_sce_different_variance() {
    let m1 = make_tensor_map("w", vec![10.0, 10.0, 10.0, 10.0], vec![4]);
    let m2 = make_tensor_map("w", vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let models = vec![m1, m2];
    let weights = vec![0.5, 0.5];
    let result = super::sce_merge(&models, &weights);
    let (data, _) = result.get("w").unwrap();
    // m1 has higher variance → should get more weight → result > 5.5 (simple average)
    let simple_avg = 5.5;
    assert!(
        data[0] > simple_avg,
        "SCE should weight high-variance model more, got {} vs avg {}",
        data[0],
        simple_avg
    );
}

#[test]
fn test_sce_weights_sum_to_one() {
    // Verify the result is a proper weighted average (between min and max)
    let m1 = make_tensor_map("w", vec![2.0], vec![1]);
    let m2 = make_tensor_map("w", vec![8.0], vec![1]);
    let models = vec![m1, m2];
    let weights = vec![0.5, 0.5];
    let result = super::sce_merge(&models, &weights);
    let (data, _) = result.get("w").unwrap();
    assert!(
        data[0] >= 2.0 && data[0] <= 8.0,
        "Result should be between inputs: {}",
        data[0]
    );
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-MERGE-ADV-001: All advanced strategies produce finite results.
#[test]
fn falsify_merge_adv_001_finite_results() {
    let (base, models) = make_two_tensor_models();

    // Task Arithmetic
    let r = super::task_arithmetic_merge(&base, &models, &[1.0, 1.0]);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "TaskArithmetic: non-finite"
    );

    // NuSLERP
    let r = super::nuslerp_tensors(&models[0], &models[1], 0.5);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "NuSLERP: non-finite"
    );

    // MultiSLERP
    let r = super::multi_slerp_tensors(&models, &[0.5, 0.5]);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "MultiSLERP: non-finite"
    );

    // DELLA
    let r = super::della_merge(&base, &models, 0.5, 42, None);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "DELLA: non-finite"
    );

    // Breadcrumbs
    let r = super::breadcrumbs_merge(&base, &models, &[1.0, 1.0], 3.0);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "Breadcrumbs: non-finite"
    );

    // SCE
    let r = super::sce_merge(&models, &[0.5, 0.5]);
    assert!(
        r.get("w").unwrap().0.iter().all(|x| x.is_finite()),
        "SCE: non-finite"
    );
}

/// FALSIFY-MERGE-ADV-002: Task arithmetic with scale=0 returns base.
#[test]
fn falsify_merge_adv_002_zero_scale_is_identity() {
    let (base, models) = make_two_tensor_models();
    let result = super::task_arithmetic_merge(&base, &models, &[0.0, 0.0]);
    let (data, _) = result.get("w").unwrap();
    let (base_data, _) = base.get("w").unwrap();
    assert_eq!(data, base_data, "Zero scale must return base model");
}

/// FALSIFY-MERGE-ADV-003: SCE weights produce result bounded by inputs.
#[test]
fn falsify_merge_adv_003_sce_bounded() {
    for val_a in [0.0, 1.0, 5.0, 10.0] {
        for val_b in [0.0, 1.0, 5.0, 10.0] {
            let m1 = make_tensor_map("w", vec![val_a], vec![1]);
            let m2 = make_tensor_map("w", vec![val_b], vec![1]);
            let models = vec![m1, m2];
            let result = super::sce_merge(&models, &[0.5, 0.5]);
            let (data, _) = result.get("w").unwrap();
            let lo = val_a.min(val_b);
            let hi = val_a.max(val_b);
            assert!(
                data[0] >= lo - 1e-6 && data[0] <= hi + 1e-6,
                "SCE({}, {}): result {} not in [{}, {}]",
                val_a,
                val_b,
                data[0],
                lo,
                hi
            );
        }
    }
}

// ============================================================================
// Passthrough / Frankenmerge tests (GH-443)
// ============================================================================

fn make_layered_model(layer_count: usize, value: f32) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut m = BTreeMap::new();
    m.insert("embed.weight".to_string(), (vec![value; 4], vec![2, 2]));
    for i in 0..layer_count {
        m.insert(
            format!("model.layers.{}.self_attn.q_proj.weight", i),
            (vec![value + i as f32; 4], vec![2, 2]),
        );
        m.insert(
            format!("model.layers.{}.mlp.weight", i),
            (vec![value * 10.0 + i as f32; 4], vec![2, 2]),
        );
    }
    m.insert("lm_head.weight".to_string(), (vec![value; 4], vec![2, 2]));
    m
}

#[test]
fn test_passthrough_basic() {
    let model_a = make_layered_model(4, 1.0);
    let model_b = make_layered_model(4, 2.0);
    let models = vec![model_a, model_b];

    // Take layers 0-2 from model A, layers 2-4 from model B
    let ranges = vec![(0, 0, 2), (1, 2, 4)];
    let result = super::passthrough_merge(&models, &ranges);

    // Output should have 4 layers (0,1 from A; 2,3 from B)
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.1.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.2.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.3.self_attn.q_proj.weight"));

    // Layer 0 should be from model A (value ~1.0)
    let (d0, _) = result
        .get("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert!((d0[0] - 1.0).abs() < 0.01, "Layer 0 should be from model A");

    // Layer 2 should be from model B layer 2 (value ~2.0 + 2 = 4.0)
    let (d2, _) = result
        .get("model.layers.2.self_attn.q_proj.weight")
        .unwrap();
    assert!(
        (d2[0] - 4.0).abs() < 0.01,
        "Layer 2 should be from model B layer 2, got {}",
        d2[0]
    );
}

#[test]
fn test_passthrough_non_layer_tensors() {
    let model_a = make_layered_model(2, 1.0);
    let model_b = make_layered_model(2, 2.0);
    let models = vec![model_a, model_b];

    let ranges = vec![(0, 0, 1), (1, 0, 1)];
    let result = super::passthrough_merge(&models, &ranges);

    // Non-layer tensors should come from first model
    let (embed, _) = result.get("embed.weight").unwrap();
    assert!(
        (embed[0] - 1.0).abs() < 0.01,
        "embed should come from model A"
    );
    let (lm_head, _) = result.get("lm_head.weight").unwrap();
    assert!(
        (lm_head[0] - 1.0).abs() < 0.01,
        "lm_head should come from model A"
    );
}

#[test]
fn test_parse_layer_tensor_name() {
    // Standard format
    let r = super::parse_layer_tensor_name("model.layers.5.self_attn.q_proj.weight");
    assert_eq!(r, Some((5, "model.layers.", ".self_attn.q_proj.weight")));

    // GGUF format
    let r = super::parse_layer_tensor_name("blk.12.attn_q.weight");
    assert_eq!(r, Some((12, "blk.", ".attn_q.weight")));

    // Non-layer tensor
    let r = super::parse_layer_tensor_name("embed.weight");
    assert_eq!(r, None);

    // lm_head
    let r = super::parse_layer_tensor_name("lm_head.weight");
    assert_eq!(r, None);
}

#[test]
fn test_passthrough_gguf_style() {
    let mut m1 = BTreeMap::new();
    m1.insert(
        "blk.0.attn_q.weight".to_string(),
        (vec![1.0; 4], vec![2, 2]),
    );
    m1.insert(
        "blk.1.attn_q.weight".to_string(),
        (vec![2.0; 4], vec![2, 2]),
    );
    m1.insert("token_embd.weight".to_string(), (vec![0.5; 4], vec![2, 2]));

    let ranges = vec![(0, 0, 2)];
    let result = super::passthrough_merge(&[m1], &ranges);

    assert!(result.contains_key("blk.0.attn_q.weight"));
    assert!(result.contains_key("blk.1.attn_q.weight"));
    assert!(result.contains_key("token_embd.weight"));
}

/// FALSIFY-PASSTHROUGH-001: Passthrough preserves tensor data exactly.
#[test]
fn falsify_passthrough_001_data_preservation() {
    let model = make_layered_model(3, 7.5);
    let ranges = vec![(0, 0, 3)];
    let result = super::passthrough_merge(&[model.clone()], &ranges);

    // Every layer tensor should be exactly preserved
    for (name, (data, shape)) in &model {
        if let Some((_, prefix, suffix)) = super::parse_layer_tensor_name(name) {
            // Find the renamed tensor in result
            let found = result.iter().any(|(_, (rd, rs))| rd == data && rs == shape);
            assert!(found, "Tensor {} data should be preserved exactly", name);
            let _ = (prefix, suffix);
        } else {
            // Non-layer tensor preserved with same name
            let (rd, rs) = result.get(name).unwrap();
            assert_eq!(rd, data, "Non-layer tensor {} data mismatch", name);
            assert_eq!(rs, shape, "Non-layer tensor {} shape mismatch", name);
        }
    }
}

// ============================================================================
// TIES / DARE tests (coverage of ties_merge.rs; Refs #5530)
// ============================================================================

fn ties_fixture() -> (
    BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>>,
) {
    let base = make_tensor_map("w", vec![0.0; 4], vec![4]);
    let a = make_tensor_map("w", vec![1.0, -1.0, 0.05, 2.0], vec![4]);
    let b = make_tensor_map("w", vec![1.0, 1.0, 0.05, -2.0], vec![4]);
    (base, vec![a, b])
}

#[test]
fn test_ties_trim_zeroes_below_density_threshold() {
    let trimmed = super::ties_trim(&[1.0, -0.4, 0.5, -2.0], 0.5);
    // threshold = 0.5 * 2.0 = 1.0; only |x| >= 1.0 survive
    assert_eq!(trimmed, vec![1.0, 0.0, 0.0, -2.0]);
    // all-zero delta stays all-zero (no division by max)
    assert_eq!(super::ties_trim(&[0.0, 0.0], 0.5), vec![0.0, 0.0]);
}

#[test]
fn test_ties_merge_elects_sign_and_scales_by_model_count() {
    let (base, models) = ties_fixture();
    let result = super::ties_merge(&base, &models, 0.5);
    let (data, shape) = result.get("w").unwrap();
    assert_eq!(shape, &vec![4]);
    // [0] both +1 agree: 2/2; [1] tie elects positive, only b agrees: 1/2;
    // [2] trimmed by both: 0; [3] tie elects positive, only a agrees: 2/2
    assert_eq!(data, &[1.0, 0.5, 0.0, 1.0]);
}

#[test]
fn test_ties_merge_majority_sign_discards_minority() {
    let base = make_tensor_map("w", vec![10.0], vec![1]);
    let models = vec![
        make_tensor_map("w", vec![12.0], vec![1]),
        make_tensor_map("w", vec![13.0], vec![1]),
        make_tensor_map("w", vec![9.0], vec![1]),
    ];
    let result = super::ties_merge(&base, &models, 0.0);
    let (data, _) = result.get("w").unwrap();
    // elected +, agreeing deltas 2 and 3, scaled by 3 models: 10 + 5/3
    assert!((data[0] - (10.0 + 5.0 / 3.0)).abs() < 1e-6, "{:?}", data);
}

#[test]
fn test_dare_merge_zero_drop_is_weighted_mean_of_deltas() {
    let base = make_tensor_map("w", vec![0.0, 0.0], vec![2]);
    let models = vec![
        make_tensor_map("w", vec![1.0, 2.0], vec![2]),
        make_tensor_map("w", vec![3.0, 4.0], vec![2]),
    ];
    let result = super::dare_merge(&base, &models, 0.0, 7, None);
    assert_eq!(result.get("w").unwrap().0, vec![2.0, 3.0]);
    let weighted = super::dare_merge(&base, &models, 0.0, 7, Some(&[1.0, 0.0]));
    assert_eq!(weighted.get("w").unwrap().0, vec![1.0, 2.0]);
}

#[test]
fn test_dare_merge_is_seed_deterministic_and_rescales_kept_deltas() {
    let base = make_tensor_map("w", vec![0.0; 64], vec![64]);
    let models = vec![make_tensor_map("w", vec![1.0; 64], vec![64])];
    let r1 = super::dare_merge(&base, &models, 0.5, 42, Some(&[1.0]));
    let r2 = super::dare_merge(&base, &models, 0.5, 42, Some(&[1.0]));
    assert_eq!(r1, r2, "same seed must reproduce the same drop mask");
    let data = &r1.get("w").unwrap().0;
    // kept elements are rescaled by 1/(1-0.5) = 2; dropped are 0
    assert!(data.iter().all(|&v| v == 0.0 || v == 2.0), "{:?}", data);
    assert!(data.contains(&0.0) && data.contains(&2.0));
}

fn merge_end_to_end(
    strategy: MergeStrategy,
    configure: impl FnOnce(&mut MergeOptions),
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let dir = tempdir().expect("tempdir");
    let (base, models) = ties_fixture();
    let base_path = dir.path().join("base.safetensors");
    let a_path = dir.path().join("a.safetensors");
    let b_path = dir.path().join("b.safetensors");
    let out_path = dir.path().join("out.safetensors");
    super::create_test_model(&base_path, &base).expect("base");
    super::create_test_model(&a_path, &models[0]).expect("a");
    super::create_test_model(&b_path, &models[1]).expect("b");
    let mut options = MergeOptions {
        strategy,
        base_model: Some(base_path),
        ..MergeOptions::default()
    };
    configure(&mut options);
    let report = apr_merge(&[a_path, b_path], out_path.clone(), options).expect("merge");
    assert_eq!(report.model_count, 2);
    assert_eq!(report.tensor_count, 1);
    assert!(report.output_size > 0);
    load_model_tensors(&out_path).expect("load merged")
}

#[test]
fn test_apr_merge_ties_end_to_end() {
    let merged = merge_end_to_end(MergeStrategy::Ties, |o| o.density = 0.5);
    assert_eq!(merged.get("w").unwrap().0, vec![1.0, 0.5, 0.0, 1.0]);
}

#[test]
fn test_apr_merge_dare_end_to_end() {
    let run = || merge_end_to_end(MergeStrategy::Dare, |o| o.drop_rate = 0.5);
    let merged = run();
    assert_eq!(merged, run(), "fixed seed must reproduce the merge");
    let data = &merged.get("w").unwrap().0;
    // each model keeps a delta with p=0.5, rescaled x2, then weighted 1/2:
    // element 0 (deltas 1, 1) is a sum of {0, 1} contributions
    assert!([0.0f32, 1.0, 2.0].contains(&data[0]), "{:?}", data);
    // element 3 (deltas 2, -2) is a sum of {0, 2} and {0, -2}
    assert!([-2.0f32, 0.0, 2.0].contains(&data[3]), "{:?}", data);
}

#[test]
fn test_apr_merge_dare_rejects_zero_drop_rate() {
    let dir = tempdir().expect("tempdir");
    let (base, models) = ties_fixture();
    let paths: Vec<_> = ["base", "a", "b"]
        .iter()
        .map(|n| dir.path().join(format!("{n}.safetensors")))
        .collect();
    super::create_test_model(&paths[0], &base).expect("base");
    super::create_test_model(&paths[1], &models[0]).expect("a");
    super::create_test_model(&paths[2], &models[1]).expect("b");
    let options = MergeOptions {
        strategy: MergeStrategy::Dare,
        base_model: Some(paths[0].clone()),
        drop_rate: 0.0,
        ..MergeOptions::default()
    };
    let err = apr_merge(&paths[1..], dir.path().join("o.safetensors"), options).unwrap_err();
    assert!(err.to_string().contains("drop_rate"), "{err}");
}
