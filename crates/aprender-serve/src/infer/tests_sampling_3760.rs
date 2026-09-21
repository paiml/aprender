//! #3760: `apr run` on a SafeTensors model samples.
//!
//! The SafeTensors CPU loop was `greedy_decode_with_transformer`, which never read
//! `temperature`, `top_k`, `top_p` or `seed`, so every sampling flag on a
//! `.safetensors` model did nothing. These rows run the tiny SafeTensors fixture
//! through `run_inference`, the path `apr run model.safetensors` takes.

use super::*;
use crate::fixtures::{ModelConfig, ModelFixture};

fn generate(fixture: &ModelFixture, temperature: f32, top_k: usize, seed: u64) -> Vec<u32> {
    let config = InferenceConfig::new(fixture.path())
        .with_input_tokens(vec![5, 6, 7])
        .with_max_tokens(12)
        .with_temperature(temperature)
        .with_top_k(top_k)
        .with_seed(seed)
        .without_gpu();
    let result = run_inference(&config).expect("run");
    result.tokens[result.input_token_count..].to_vec()
}

/// done_when 2's first control: sampled differs from greedy for some seed. RED at
/// 0.69.0, where the loop was greedy whatever the flags said.
#[test]
fn a_sampled_safetensors_run_differs_from_greedy() {
    let fixture = ModelFixture::safetensors("sample3760", ModelConfig::tiny());
    let greedy = generate(&fixture, 0.0, 1, 1);
    assert!(!greedy.is_empty(), "control: the greedy run must generate");
    let differs = (1..=8).any(|seed| generate(&fixture, 5.0, 0, seed) != greedy);
    assert!(
        differs,
        "temperature 5.0 matched greedy on all 8 seeds: the loop does not sample"
    );
}

/// The same seed reproduces a sampled run byte for byte; another seed changes it.
#[test]
fn a_sampled_safetensors_run_is_seeded() {
    let fixture = ModelFixture::safetensors("seed3760", ModelConfig::tiny());
    let first = generate(&fixture, 5.0, 0, 7);
    assert_eq!(
        generate(&fixture, 5.0, 0, 7),
        first,
        "same seed, different output"
    );
    let seed_used = (8..=15).any(|seed| generate(&fixture, 5.0, 0, seed) != first);
    assert!(
        seed_used,
        "8 other seeds all reproduced seed 7: the seed is not used"
    );
}

/// `--top-k 1` or temperature 0 stays byte-greedy on any seed.
#[test]
fn top_k_one_and_temperature_zero_stay_greedy_on_safetensors() {
    let fixture = ModelFixture::safetensors("greedy3760", ModelConfig::tiny());
    let greedy = generate(&fixture, 0.0, 1, 1);
    for seed in 1..=4 {
        assert_eq!(
            generate(&fixture, 5.0, 1, seed),
            greedy,
            "top_k 1, seed {seed}"
        );
        assert_eq!(
            generate(&fixture, 0.0, 40, seed),
            greedy,
            "temperature 0, seed {seed}"
        );
    }
}
