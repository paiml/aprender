//! #3754: a sampling flag given alone samples.
//!
//! `apr run --top-k` defaulted to 1, which every decode loop reads as greedy, so
//! `--temperature` alone got greedy output and no warning. `apr run` forwards its `--top-k`
//! to realizar explicitly (`with_top_k`), and its default is now `DEFAULT_TOP_K`, which
//! apr-cli's `run_tests_top_k_default_3754` pins. These rows run the pygmy GGUF through
//! `run_inference`, the path `apr run` takes, with the top-k `apr run` now sends.

use super::*;
use std::io::Write;

fn pygmy() -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp gguf");
    file.write_all(&crate::gguf::test_factory::build_executable_pygmy_gguf())
        .expect("write gguf");
    file.flush().expect("flush gguf");
    file
}

/// The generated tokens for one run. `top_k: None` sends what `apr run` sends when the
/// user names no `--top-k`: `DEFAULT_TOP_K` (0.69.0 sent 1).
fn generate(
    file: &tempfile::NamedTempFile,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
) -> Vec<u32> {
    let mut config = InferenceConfig::new(file.path())
        .with_input_tokens(vec![1, 2, 3, 4])
        .with_max_tokens(12)
        .with_temperature(temperature)
        .with_seed(seed)
        .without_gpu();
    config = config.with_top_k(top_k.unwrap_or(DEFAULT_TOP_K));
    let result = run_inference(&config).expect("run");
    result.tokens[result.input_token_count..].to_vec()
}

#[test]
fn sampling_top_k_resolves_greedy_the_request_or_the_default() {
    assert!(
        DEFAULT_TOP_K > 1,
        "a default of 1 is greedy, which is the defect"
    );
    assert_eq!(sampling_top_k(0.0, None), 1);
    assert_eq!(sampling_top_k(0.0, Some(40)), 1);
    assert_eq!(sampling_top_k(0.8, None), DEFAULT_TOP_K);
    assert_eq!(sampling_top_k(0.8, Some(5)), 5);
    // PMAT-823: the library default stays byte-identical greedy.
    assert_eq!(InferenceConfig::new("m.gguf").top_k, 1);
    assert_eq!(
        InferenceConfig::new("m.gguf").temperature,
        0.0,
        "greedy by default"
    );
}

/// done_when 1: `temperature > 0` with no explicit `--top-k` samples. Its output differs from
/// greedy for at least one of 8 fixed seeds. At 0.69.0 `apr run` sent top-k 1 here, which is
/// greedy on every seed (the `DEFAULT_TOP_K = 1` mutant turns this row RED).
#[test]
fn a_temperature_given_alone_samples() {
    let file = pygmy();
    let greedy = generate(&file, 0.0, None, 1);
    assert_eq!(
        greedy.len(),
        12,
        "control: the run must produce its whole budget"
    );
    let differs = (1..=8).any(|seed| generate(&file, 1.5, None, seed) != greedy);
    assert!(
        differs,
        "temperature 1.5 with no top-k matched greedy on all 8 seeds"
    );
}

/// done_when 2: an explicit `top_k 1`, or `temperature 0`, stays greedy and byte-identical
/// whatever the seed.
#[test]
fn explicit_top_k_one_and_temperature_zero_stay_greedy() {
    let file = pygmy();
    let greedy = generate(&file, 0.0, None, 1);
    for seed in 1..=4 {
        assert_eq!(
            generate(&file, 1.5, Some(1), seed),
            greedy,
            "top_k 1, seed {seed}"
        );
        assert_eq!(
            generate(&file, 0.0, None, seed),
            greedy,
            "temperature 0, seed {seed}"
        );
        assert_eq!(
            generate(&file, 0.0, Some(40), seed),
            greedy,
            "temperature 0 wins over top_k"
        );
    }
}

/// done_when 4: the same seed reproduces the sample byte for byte, and some other seed
/// changes it. Without the second half, a seed nobody reads would pass the first.
#[test]
fn a_seeded_sample_is_reproducible_and_the_seed_is_used() {
    let file = pygmy();
    let first = generate(&file, 1.5, None, 7);
    assert_eq!(
        generate(&file, 1.5, None, 7),
        first,
        "same seed, different output"
    );
    let seed_used = (8..=15).any(|seed| generate(&file, 1.5, None, seed) != first);
    assert!(seed_used, "8 other seeds all reproduced seed 7's sample");
}
