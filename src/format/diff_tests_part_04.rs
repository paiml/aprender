
#[test]
fn test_diff_inspections_no_metadata() {
    use std::collections::BTreeMap;

    let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
    r1.metadata = {
        let mut m = BTreeMap::new();
        m.insert("key".to_string(), "val1".to_string());
        m
    };
    let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
    r2.metadata = {
        let mut m = BTreeMap::new();
        m.insert("key".to_string(), "val2".to_string());
        m
    };

    // With metadata comparison disabled, no diff
    let report = diff_inspections(
        &r1,
        &r2,
        "a.apr",
        "b.apr",
        DiffOptions::new().without_metadata(),
    );
    assert!(report.is_identical());

    // With metadata comparison enabled, diff present
    let report2 = diff_inspections(
        &r1,
        &r2,
        "a.apr",
        "b.apr",
        DiffOptions::new().with_metadata(),
    );
    assert!(!report2.is_identical());
}

// ====================================================================
// Coverage: compare_tensor_stats None->Some branch
// ====================================================================

#[test]
fn test_compare_tensor_stats_none_some() {
    use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
    let mut diffs = Vec::new();
    let t1 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: None,
    };
    let t2 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    compare_tensor_stats(&t1, &t2, &mut diffs);
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("stats"));
    assert_eq!(diffs[0].value1, "(none)");
    assert_eq!(diffs[0].value2, "present");
}

// ====================================================================
// Coverage: compare_tensor_stats std differs alone
// ====================================================================

#[test]
fn test_compare_tensor_stats_std_differs_only() {
    use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
    let mut diffs = Vec::new();
    let t1 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    let t2 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.9,
        }),
    };
    compare_tensor_stats(&t1, &t2, &mut diffs);
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("std"));
}

// ====================================================================
// Coverage: cross-format tensor comparison with GGUF name mapping
// ====================================================================

#[test]
fn test_compare_tensors_cross_format_gguf_to_apr() {
    use crate::format::rosetta::TensorInfo;

    // Model 1 uses GGUF naming
    let t1 = vec![
        TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![4096, 4096],
            size_bytes: 1000,
            stats: None,
        },
        TensorInfo {
            name: "token_embd.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![32000, 4096],
            size_bytes: 2000,
            stats: None,
        },
    ];

    // Model 2 uses APR/HF naming with same shapes
    let t2 = vec![
        TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![4096, 4096],
            size_bytes: 1000,
            stats: None,
        },
        TensorInfo {
            name: "model.embed_tokens.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![32000, 4096],
            size_bytes: 2000,
            stats: None,
        },
    ];

    let mut diffs = Vec::new();
    let options = DiffOptions::default();
    compare_tensors(&t1, &t2, &options, &mut diffs);

    // Cross-format mapping should make these match
    assert!(
        diffs.is_empty(),
        "Expected no diffs for cross-format name mapping, got: {diffs:?}"
    );
}

#[test]
fn test_compare_tensors_transposed_shapes_compatible() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![100, 200],
        size_bytes: 800,
        stats: None,
    }];
    let t2 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![200, 100], // Transposed
        size_bytes: 800,
        stats: None,
    }];

    let mut diffs = Vec::new();
    compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

    // Transposed 2D shapes are considered compatible
    assert!(diffs.is_empty());
}

#[test]
fn test_compare_tensors_compatible_quant_no_diff() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "Q5_0".to_string(),
        shape: vec![10, 20],
        size_bytes: 400,
        stats: None,
    }];
    let t2 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "Q6_K".to_string(),
        shape: vec![10, 20],
        size_bytes: 400,
        stats: None,
    }];

    let mut diffs = Vec::new();
    compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

    // Q5_0 and Q6_K are considered compatible
    assert!(
        diffs.is_empty(),
        "Expected no dtype diff for compatible quants Q5_0 and Q6_K, got: {diffs:?}"
    );
}

#[test]
fn test_compare_tensors_only_in_model1() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "unique_tensor".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: None,
    }];
    let t2: Vec<TensorInfo> = vec![];

    let mut diffs = Vec::new();
    compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

    assert!(diffs
        .iter()
        .any(|d| d.field.contains("unique_tensor") && d.value2 == "(missing)"));
}

#[test]
fn test_compare_tensors_only_in_model2() {
    use crate::format::rosetta::TensorInfo;

    let t1: Vec<TensorInfo> = vec![];
    let t2 = vec![TensorInfo {
        name: "extra_tensor".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: None,
    }];

    let mut diffs = Vec::new();
    compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

    assert!(diffs
        .iter()
        .any(|d| d.field.contains("extra_tensor") && d.value1 == "(missing)"));
}
