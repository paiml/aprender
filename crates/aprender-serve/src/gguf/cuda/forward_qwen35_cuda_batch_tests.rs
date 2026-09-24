//! #4234: [`Qwen35CudaModel::forward_batch`] against the per-sequence
//! [`Qwen35CudaModel::forward_single`] it batches, on the real Qwen3.5-0.8B file.
//!
//! The claim is bitwise: the batch runs the same kernels on the same inputs in the
//! same per-sequence order, so every logit of every sequence must have the same BITS
//! as the per-sequence reference. A tolerance here would hide the defects a batch can
//! introduce — one sequence reading another's scratch, conv window or KV rows — which
//! perturb logits by amounts a cosine floor forgives.

use super::super::{Qwen35CudaModel, Qwen35CudaState};
use crate::gguf::forward_qwen35::Qwen35Model;

const MODEL_0_8B: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Four prompts of four different lengths, so the four sequences decode at four
/// different positions every step. Sequences 1 and 3 share a prompt: two streams of
/// the same request must still come out identical to each other and to the reference.
const PROMPTS: [&[u32]; 4] = [
    &[9707],
    &[9707, 11, 1879],
    &[760, 6511, 314, 9338, 369],
    &[9707, 11, 1879],
];

/// Greedy decode steps after each prompt.
const STEPS: usize = 8;

/// Positions each test state holds: the longest prompt plus every decode step.
const STATE_LEN: usize = 16;

fn argmax(v: &[f32]) -> u32 {
    let (i, _) = v
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        });
    u32::try_from(i).expect("vocab fits u32")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Feed `prompt` through `forward_single` and return the last position's logits.
fn prefill(gpu: &mut Qwen35CudaModel<'_>, state: &mut Qwen35CudaState, prompt: &[u32]) -> Vec<f32> {
    let mut last = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        last = gpu
            .forward_single(token, state, pos)
            .expect("prefill token");
    }
    last
}

/// Every decode step's logits for one sequence, run alone through `forward_single`.
fn reference(gpu: &mut Qwen35CudaModel<'_>, prompt: &[u32]) -> Vec<Vec<f32>> {
    let mut state = gpu.new_state_with_len(STATE_LEN).expect("state");
    let mut logits = prefill(gpu, &mut state, prompt);
    let mut steps = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let token = argmax(&logits);
        logits = gpu
            .forward_single(token, &mut state, prompt.len() + step)
            .expect("reference decode");
        steps.push(logits.clone());
    }
    steps
}

#[test]
#[serial_test::serial]
fn qwen35_cuda_forward_batch_is_bitwise_forward_single_per_sequence() {
    if !std::path::Path::new(MODEL_0_8B).exists() {
        eprintln!("SKIP: {MODEL_0_8B} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_0_8B).expect("map");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");

    let t0 = std::time::Instant::now();
    let want: Vec<Vec<Vec<f32>>> = PROMPTS.iter().map(|p| reference(&mut gpu, p)).collect();
    let sequential_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut states: Vec<Qwen35CudaState> = (0..PROMPTS.len())
        .map(|_| gpu.new_state_with_len(STATE_LEN).expect("state"))
        .collect();
    let mut last: Vec<Vec<f32>> = PROMPTS
        .iter()
        .zip(states.iter_mut())
        .map(|(p, s)| prefill(&mut gpu, s, p))
        .collect();

    let t1 = std::time::Instant::now();
    for step in 0..STEPS {
        let tokens: Vec<u32> = last.iter().map(|l| argmax(l)).collect();
        let positions: Vec<usize> = PROMPTS.iter().map(|p| p.len() + step).collect();
        let mut refs: Vec<&mut Qwen35CudaState> = states.iter_mut().collect();
        last = gpu
            .forward_batch(&tokens, &mut refs, &positions)
            .expect("batched decode");
        assert_eq!(last.len(), PROMPTS.len(), "one logits row per sequence");
        for (b, got) in last.iter().enumerate() {
            let expect = &want[b][step];
            let differing = bits(got)
                .iter()
                .zip(bits(expect))
                .filter(|(g, w)| **g != *w)
                .count();
            assert_eq!(
                differing,
                0,
                "sequence {b} step {step} (position {}): {differing} of {} logits differ in their \
                 bits from forward_single — argmax {} vs {}",
                positions[b],
                got.len(),
                argmax(got),
                argmax(expect),
            );
        }
    }
    let batched_ms = t1.elapsed().as_secs_f64() * 1e3;

    for (b, (state, prompt)) in states.iter().zip(PROMPTS).enumerate() {
        assert_eq!(
            state.kv_len(),
            prompt.len() + STEPS,
            "sequence {b}: the KV length must have advanced once per step"
        );
    }
    // Not vacuous: the sequences at different positions must actually differ, or a
    // batch that returned sequence 0's row for everyone could only pass by accident.
    assert_ne!(
        bits(&last[0]),
        bits(&last[2]),
        "sequences 0 and 2 have different prompts and must not share logits"
    );
    // Printed, never asserted: a timing assertion in a correctness test is a flake.
    eprintln!(
        "[4234] {} sequences x {STEPS} steps: reference (prefill + decode, one at a time) \
         {sequential_ms:.1} ms; batched decode {batched_ms:.1} ms",
        PROMPTS.len()
    );
}

#[test]
#[serial_test::serial]
fn qwen35_cuda_forward_batch_refuses_before_touching_any_state() {
    if !std::path::Path::new(MODEL_0_8B).exists() {
        eprintln!("SKIP: {MODEL_0_8B} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_0_8B).expect("map");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");

    let mut a = gpu.new_state_with_len(4).expect("state");
    let mut b = gpu.new_state_with_len(4).expect("state");
    prefill(&mut gpu, &mut a, &[9707, 11]);
    let conv_before = {
        let mut host = vec![0.0f32; a.conv_len()];
        a.conv[0].copy_to_host(&mut host).expect("download");
        host
    };

    let cases: [(&str, Vec<u32>, Vec<usize>); 4] = [
        ("empty batch", vec![], vec![]),
        (
            "more tokens than states",
            vec![9707, 11, 1879],
            vec![2, 0, 0],
        ),
        (
            "a token outside the vocabulary",
            vec![9707, u32::MAX],
            vec![2, 0],
        ),
        ("a position past the state", vec![9707, 11], vec![2, 4]),
    ];
    for (what, tokens, positions) in cases {
        let mut refs: Vec<&mut Qwen35CudaState> = if tokens.is_empty() {
            Vec::new()
        } else {
            vec![&mut a, &mut b]
        };
        let err = gpu
            .forward_batch(&tokens, &mut refs, &positions)
            .expect_err(what);
        eprintln!("[4234] {what}: {err}");
        assert!(
            err.to_string().contains("qwen35_cuda_forward_batch"),
            "{what}: the refusal must name the batch step, got: {err}"
        );
    }

    assert_eq!(a.kv_len(), 2, "a refused batch must not advance a state");
    assert_eq!(b.kv_len(), 0, "a refused batch must not advance a state");
    let mut conv_after = vec![0.0f32; a.conv_len()];
    a.conv[0].copy_to_host(&mut conv_after).expect("download");
    assert_eq!(
        bits(&conv_before),
        bits(&conv_after),
        "a refused batch must not write a conv window"
    );
}
