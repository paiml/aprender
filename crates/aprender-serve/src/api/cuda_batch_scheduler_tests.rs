//! #4280: the scheduler's single-request path runs through the one engine and
//! answers what the loop it replaced (`generate_gpu_resident_streaming`)
//! answered, token for token, on a real dense file on the GPU.
//!
//! Both tests need the file and a CUDA device; without either they print why
//! and return, as the other real-file GPU tests in this crate do.

use super::*;
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use crate::session::{entries_for, EntryKind};

/// A dense Q4_K file (qwen2): the arch the ported path serves.
const MODEL_PATH: &str = "/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";

fn cuda_model() -> Option<OwnedQuantizedModelCuda> {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP #4280: {MODEL_PATH} is not on this host");
        return None;
    }
    let mapped = MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("CPU model");
    // As `apr serve` builds it (`prepare_gguf_cuda_state`).
    match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 4096) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("SKIP #4280: no CUDA model: {}", e.error);
            None
        },
    }
}

/// What the scheduler sends for `prompt`: the tokens, or the error line.
fn scheduled(
    model: &mut OwnedQuantizedModelCuda,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
    non_streaming: bool,
) -> Vec<std::result::Result<u32, String>> {
    let (token_tx, mut rx) = tokio::sync::mpsc::channel(4096);
    generate_single_request(
        model,
        CudaBatchRequest {
            prompt_ids: prompt.to_vec(),
            config: config.clone(),
            token_tx,
            non_streaming,
            enqueue_time: std::time::Instant::now(),
            timing_tx: None,
        },
    );
    let mut out = Vec::new();
    while let Ok(t) = rx.try_recv() {
        out.push(t);
    }
    out
}

/// What the pre-port loop emitted for `prompt`.
fn pre_port(
    model: &mut OwnedQuantizedModelCuda,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
) -> Vec<std::result::Result<u32, String>> {
    let mut out = Vec::new();
    match model.generate_gpu_resident_streaming(prompt, config, |t| {
        out.push(Ok(t));
        true
    }) {
        Ok(_) => out,
        Err(e) => {
            out.push(Err(e.to_string()));
            out
        },
    }
}

fn greedy(max_tokens: usize, stop_tokens: Vec<u32>) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens,
        ..Default::default()
    }
}

/// Temperature 0, four prompts (a single-token one, which prefills nothing),
/// with and without a repetition penalty and with a stop token the answer
/// reaches: the ported path sends exactly the pre-port tokens, streamed and
/// bulk, and never sends the stop token.
#[test]
fn single_request_is_token_identical_to_the_pre_port_loop() {
    let Some(mut model) = cuda_model() else {
        return;
    };
    let prompts: [&[u32]; 4] = [
        &[151_644, 872, 198, 3838, 374, 220, 17, 10, 17, 30],
        &[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13],
        &[750, 10309, 1445, 982, 262, 470],
        &[9707],
    ];
    let mut compared = 0;
    for prompt in prompts {
        let plain = greedy(24, Vec::new());
        let reference = pre_port(&mut model, prompt, &plain);
        assert!(
            reference.len() >= 4 && reference.iter().all(Result::is_ok),
            "fixture: the pre-port loop answered {reference:?} for {prompt:?}"
        );
        // A stop token the answer reaches at its third token.
        let stop = *reference[2].as_ref().expect("checked above");
        let penalised = QuantizedGenerateConfig {
            repeat_penalty: 1.3,
            repeat_last_n: 64,
            ..greedy(24, Vec::new())
        };
        for config in [plain, greedy(24, vec![stop]), penalised] {
            let want = pre_port(&mut model, prompt, &config);
            for non_streaming in [true, false] {
                let got = scheduled(&mut model, prompt, &config, non_streaming);
                assert_eq!(
                    got, want,
                    "prompt {prompt:?}, stop {:?}, penalty {}, non_streaming {non_streaming}",
                    config.stop_tokens, config.repeat_penalty
                );
                if config.stop_tokens.contains(&stop) {
                    assert!(!got.contains(&Ok(stop)), "the stop token was sent");
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 24);
}

/// The engine-identity entry for serve × dense CUDA: the single-request path
/// is one [`Session::generate`](crate::session::Session::generate) on the GPU.
/// A scheduler that calls the model's own loop again records nothing here.
#[test]
fn tests_engine_identity_serve_dense_cuda_single_request() {
    let Some(mut model) = cuda_model() else {
        return;
    };
    // A prompt no other test uses, so the witness entries are this test's.
    let prompt = [4280_u32, 17, 4280, 99, 7];
    assert!(
        entries_for(&prompt).is_empty(),
        "fixture: the prompt is not unique"
    );
    let sent = scheduled(&mut model, &prompt, &greedy(4, Vec::new()), false);
    assert!(
        sent.iter().all(Result::is_ok) && !sent.is_empty(),
        "sent {sent:?}"
    );
    let entries = entries_for(&prompt);
    assert_eq!(entries.len(), 1, "one turn, one engine entry: {entries:?}");
    assert_eq!(entries[0].kind, EntryKind::Generate);
    assert_eq!(entries[0].arch, "qwen2");
    assert!(
        entries[0].on_gpu,
        "the serve turn left the GPU: {entries:?}"
    );
}
