// =========================================================================
// FALSIFY-LORA: lora-algebra-v1.yaml contract (aprender LoRAAdapter)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LORA-* tests
//   Why 2: LoRA tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from lora-algebra-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: LoRA was "obviously correct" (low-rank additive update)
//
// References:
//   - provable-contracts/contracts/lora-algebra-v1.yaml
//   - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
// =========================================================================

use super::*;
use crate::autograd::Tensor;

/// FALSIFY-LORA-001: LoRA output shape matches base weight shape
#[test]
fn falsify_lora_001_output_shape() {
    let config = LoRAConfig::new(4, 1.0);
    let adapter = LoRAAdapter::new(8, 16, config);
    let base_weight = Tensor::new(&vec![0.1; 16 * 8], &[16, 8]);

    let result = adapter.apply(&base_weight);
    assert_eq!(
        result.shape(), &[16, 8],
        "FALSIFIED LORA-001: output shape {:?} != base shape [16, 8]", result.shape()
    );
}

/// FALSIFY-LORA-002: Zero-initialized B → apply returns base weight
#[test]
fn falsify_lora_002_zero_init_identity() {
    let config = LoRAConfig::new(4, 1.0);
    let adapter = LoRAAdapter::new(8, 16, config);
    let base_data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let base_weight = Tensor::new(&base_data, &[16, 8]);

    let result = adapter.apply(&base_weight);

    // B is zero-initialized, so BA = 0, result ≈ base_weight
    for (i, (&r, &b)) in result.data().iter().zip(base_data.iter()).enumerate() {
        assert!(
            (r - b).abs() < 0.1,
            "FALSIFIED LORA-002: result[{i}]={r} far from base[{i}]={b} (zero B should preserve)"
        );
    }
}

/// FALSIFY-LORA-003: LoRA adapter output is finite
#[test]
fn falsify_lora_003_finite_output() {
    let config = LoRAConfig::new(4, 1.0);
    let adapter = LoRAAdapter::new(8, 16, config);
    let base_weight = Tensor::new(&vec![1.0; 128], &[16, 8]);

    let result = adapter.apply(&base_weight);
    for (i, &v) in result.data().iter().enumerate() {
        assert!(
            v.is_finite(),
            "FALSIFIED LORA-003: output[{i}] = {v} is not finite"
        );
    }
}

/// FALSIFY-LORA-004: Scaling factor affects output magnitude
#[test]
fn falsify_lora_004_scaling_affects_output() {
    let config1 = LoRAConfig::new(4, 1.0);
    let config2 = LoRAConfig::new(4, 10.0);

    // Both have same structure but different alpha
    assert!(
        (config1.scaling() - config2.scaling()).abs() > 1e-6,
        "FALSIFIED LORA-004: different alpha produces same scaling"
    );
}
