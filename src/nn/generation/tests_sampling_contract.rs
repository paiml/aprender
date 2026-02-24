// =========================================================================
// FALSIFY-SA: sampling-algorithms-v1.yaml contract (aprender samplers)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-SA-* tests for sampling
//   Why 2: sampler tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from sampling-algorithms-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Nucleus sampling was "obviously correct" (top-p filtering)
//
// References:
//   - provable-contracts/contracts/sampling-algorithms-v1.yaml
//   - Holtzman et al. (2019) "The Curious Case of Neural Text Degeneration"
// =========================================================================

use super::*;
use crate::autograd::Tensor;

/// FALSIFY-SA-001: Sampled token index is within vocab range
#[test]
fn falsify_sa_001_sample_in_range() {
    let logits = Tensor::new(&[1.0, 2.0, 3.0, 0.5, 1.5], &[5]);

    let sampler = NucleusSampler::new(0.9);
    let token = sampler.sample(&logits);
    assert!(
        token < 5,
        "FALSIFIED SA-001: sampled token {token} >= vocab size 5"
    );
}

/// FALSIFY-SA-002: TopK sampler returns valid index
#[test]
fn falsify_sa_002_topk_in_range() {
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 0.1, 3.0], &[5]);

    let sampler = TopKSampler::new(3);
    let token = sampler.sample(&logits);
    assert!(
        token < 5,
        "FALSIFIED SA-002: TopK sampled token {token} >= vocab size 5"
    );
}

/// FALSIFY-SA-003: Greedy decoding returns argmax
#[test]
fn falsify_sa_003_greedy_argmax() {
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 0.1, 3.0], &[5]);

    let decoder = GreedyDecoder::new();
    let token = decoder.decode(&logits);
    assert_eq!(
        token, 1,
        "FALSIFIED SA-003: greedy returned {token}, expected 1 (argmax)"
    );
}

/// FALSIFY-SA-004: Nucleus sampler respects top_p constructor
#[test]
fn falsify_sa_004_nucleus_top_p_stored() {
    let sampler = NucleusSampler::new(0.85);
    assert!(
        (sampler.top_p() - 0.85).abs() < 1e-6,
        "FALSIFIED SA-004: top_p={}, expected 0.85",
        sampler.top_p()
    );
}
