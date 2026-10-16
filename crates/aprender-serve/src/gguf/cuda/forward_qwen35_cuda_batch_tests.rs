//! #5289: [`Qwen35CudaModel::forward_batch_greedy`] against each slot decoded
//! alone, on the real Qwen3.5-0.8B file.

use super::{check_batch_shape, Qwen35StepOut};
use crate::gguf::cuda::Qwen35CudaModel;
use crate::gguf::forward_qwen35::Qwen35Model;

/// The real hybrid file, as in the per-layer parity tests.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Greedy tokens each slot generates after its prompt.
const STEPS: usize = 8;

/// Four ragged prompts (1, 4, 7 and 2 tokens), all distinct, so a slot that
/// read another slot's state or scratch would decode a different stream.
const PROMPTS: [&[u32]; 4] = [
    &[9707],
    &[785, 6722, 315, 9625],
    &[22043, 374, 264, 3974, 11, 1879, 13],
    &[16, 10],
];

/// `prompt` then `steps` greedy tokens through `forward_single_greedy` on a
/// fresh state: the stream the slot must reproduce inside a batch.
fn decode_alone(gpu: &mut Qwen35CudaModel<'_>, prompt: &[u32], steps: usize) -> Vec<u32> {
    let mut state = gpu.new_state().expect("device state");
    let mut next = 0;
    for (pos, &token) in prompt.iter().enumerate() {
        next = gpu
            .forward_single_greedy(token, &mut state, pos)
            .expect("prompt token");
    }
    let mut out = vec![next];
    for pos in prompt.len()..prompt.len() + steps - 1 {
        next = gpu
            .forward_single_greedy(next, &mut state, pos)
            .expect("decode token");
        out.push(next);
    }
    out
}

/// FALSIFY (#5289): for m = 1..4 ragged slots, the streams from interleaved
/// batched steps are bitwise equal to each prompt decoded alone. The model's
/// scratch buffers (hidden, logits, embedding row) are shared by every slot,
/// so a step that left anything behind would change the next slot's token.
#[test]
#[serial_test::serial]
fn batched_greedy_streams_equal_each_slot_decoded_alone() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");

    let alone: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| decode_alone(&mut gpu, p, STEPS))
        .collect();
    assert!(
        alone.windows(2).any(|w| w[0] != w[1]),
        "the solo streams are all equal, so they cannot tell slots apart: {alone:?}"
    );

    for m in 1..=PROMPTS.len() {
        let prompts = &PROMPTS[..m];
        let mut states: Vec<_> = (0..m)
            .map(|_| gpu.new_state().expect("device state"))
            .collect();
        // Prefill position by position across the slots whose prompt still has
        // a token there, so the slots interleave from the first token.
        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut first = vec![0u32; m];
        for pos in 0..longest {
            for i in (0..m).filter(|&i| pos < prompts[i].len()) {
                first[i] = gpu
                    .forward_batch_greedy(
                        &[prompts[i][pos]],
                        std::slice::from_mut(&mut states[i]),
                        &[pos],
                    )
                    .expect("prefill token")[0];
            }
        }
        let mut streams: Vec<Vec<u32>> = first.iter().map(|&t| vec![t]).collect();
        let mut positions: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        for _ in 1..STEPS {
            let tokens: Vec<u32> = streams.iter().map(|s| s[s.len() - 1]).collect();
            let next = gpu
                .forward_batch_greedy(&tokens, &mut states, &positions)
                .expect("decode step");
            for ((s, p), t) in streams.iter_mut().zip(&mut positions).zip(next) {
                s.push(t);
                *p += 1;
            }
        }
        for (i, s) in streams.iter().enumerate() {
            assert_eq!(
                s, &alone[i],
                "m={m}: slot {i} diverged from its solo stream"
            );
        }
        // #5303: past one slot the step is the batched one, not the loop.
        assert_eq!(
            gpu.batch_bufs.is_some(),
            m > 1,
            "m={m}: the batched step ran iff m > 1"
        );
        eprintln!("[batch] m={m}: {m} slots x {STEPS} tokens equal their solo streams");
    }
}

/// The logits `forward_single` gives `prompt`'s last token and then each of
/// `stream`'s tokens but the last, on a fresh state: row `k` is the logits
/// the slot's `k`-th generated token is picked from.
fn logits_alone(gpu: &mut Qwen35CudaModel<'_>, prompt: &[u32], stream: &[u32]) -> Vec<Vec<f32>> {
    let mut state = gpu.new_state().expect("device state");
    let mut rows = Vec::new();
    let fed = prompt.iter().chain(&stream[..stream.len() - 1]);
    for (pos, &token) in fed.enumerate() {
        let logits = gpu
            .forward_single(token, &mut state, pos)
            .expect("solo token");
        if pos + 1 >= prompt.len() {
            rows.push(logits);
        }
    }
    rows
}

/// FALSIFY (#5279): a slot that asks for its logits stays in the batched
/// step, and gets the logits bitwise as `forward_single` gives them decoded
/// alone; the slots beside it still get their greedy tokens. Logprobs, a
/// penalty or sampling no longer take a slot out of the batch.
#[test]
#[serial_test::serial]
fn host_logit_slots_step_in_the_batch_bitwise_their_solo_logits() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");

    let alone: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| decode_alone(&mut gpu, p, STEPS))
        .collect();
    let want: Vec<Vec<Vec<f32>>> = PROMPTS
        .iter()
        .zip(&alone)
        .map(|(p, s)| logits_alone(&mut gpu, p, s))
        .collect();

    let m = PROMPTS.len();
    let host: Vec<bool> = (0..m).map(|i| i % 2 == 0).collect();
    let mut states: Vec<_> = (0..m)
        .map(|_| gpu.new_state().expect("device state"))
        .collect();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        for (pos, &token) in prompt.iter().enumerate() {
            gpu.forward_batch_greedy(&[token], std::slice::from_mut(&mut states[i]), &[pos])
                .expect("prefill token");
        }
    }
    let mut positions: Vec<usize> = PROMPTS.iter().map(|p| p.len()).collect();
    for k in 1..STEPS {
        let tokens: Vec<u32> = alone.iter().map(|s| s[k - 1]).collect();
        let outs = gpu
            .forward_batch(&tokens, &mut states, &positions, &host)
            .expect("decode step");
        for (i, out) in outs.into_iter().enumerate() {
            match out {
                Qwen35StepOut::Logits(got) => {
                    assert!(host[i], "step {k}: slot {i} got logits it did not ask for");
                    let same = got.len() == want[i][k].len()
                        && got
                            .iter()
                            .zip(&want[i][k])
                            .all(|(g, w)| g.to_bits() == w.to_bits());
                    assert!(
                        same,
                        "step {k}: slot {i}'s logits differ from its solo logits"
                    );
                },
                Qwen35StepOut::Token(t) => {
                    assert!(!host[i], "step {k}: slot {i} got a token, not its logits");
                    assert_eq!(t, alone[i][k], "step {k}: slot {i} left its solo stream");
                },
            }
        }
        for p in &mut positions {
            *p += 1;
        }
    }
    assert!(gpu.batch_bufs.is_some(), "the m={m} steps ran batched");
    eprintln!("[batch] m={m}: host-logit slots bitwise their solo logits for {STEPS} steps");
}

/// Unequal lengths, and an empty batch, are refused by name.
#[test]
fn a_batch_whose_inputs_disagree_is_refused() {
    assert!(check_batch_shape(2, 2, 2).is_ok());
    for (t, s, p) in [(0, 0, 0), (2, 1, 2), (2, 2, 3), (1, 2, 2)] {
        let err = check_batch_shape(t, s, p).expect_err("refused").to_string();
        assert!(
            err.contains("qwen35_cuda_batch"),
            "({t}, {s}, {p}) refused without naming the op: {err}"
        );
    }
}
