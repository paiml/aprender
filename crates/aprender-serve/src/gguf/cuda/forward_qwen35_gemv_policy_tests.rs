//! #4485 (RCA-SRV-001 fix 3): the decode GEMV precision policy of a Qwen3.5
//! CUDA model — production runs the DP4A (Q8_1-activation) kernels, the
//! kernel-isolation tests pin the float ones.
//!
//! The float pin was the production default from PMAT-3477 / #3090. Its stated
//! reason, "DP4A is catastrophic through the recurrence", was #4258 (a stale
//! Q8_1 activation reused by every GEMV); with that fixed, what was left was a
//! kernel-isolation tolerance (1e-3 of the CPU reference) standing in as the
//! production precision policy. llama.cpp's `mul_mat_vec_q` quantizes the
//! activation to q8_1 the same way.

use super::{cosine, Qwen35CudaModel, MODEL_PATH, MODEL_PATH_4B, PROMPT_4B};
use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
use crate::gguf::forward_qwen35::Qwen35Model;

/// The policy table, without a device: DP4A from Turing up, float below it
/// (the DP4A kernels are validated from sm_75, `GpuProfile::detect`).
#[test]
fn production_gemv_is_dp4a_from_turing_and_float_below() {
    let dp4a = (Q4kVariant::HwDp4a, Q6kVariant::HwDp4a);
    let float = (Q4kVariant::Mwv, Q6kVariant::Mwv);
    for (cc, want) in [
        (61, float),
        (70, float),
        (72, float),
        (75, dp4a),
        (86, dp4a),
        (89, dp4a),
        (90, dp4a),
        (121, dp4a),
    ] {
        assert_eq!(
            Qwen35CudaModel::production_gemv(cc),
            want,
            "sm_{cc}: wrong production GEMV pair"
        );
    }
}

/// A model built the ordinary way runs the production pair, and the
/// kernel-isolation pin still takes it back to float.
#[test]
#[serial_test::serial]
fn qwen35_cuda_a_fresh_4b_model_runs_the_production_gemv() {
    let executor = qwen35_cuda_file_or_skip!(MODEL_PATH_4B);
    let cc = executor.gpu_profile.cc;
    if gemv_env_is_set() {
        eprintln!("SKIP: a Q4K/Q6K variant env var is set; the policy defers to it");
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH_4B).expect("map the 4B GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("4B hybrid layers");

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the 4B CUDA model");
    assert_eq!(
        gpu.gemv_variants(),
        Qwen35CudaModel::production_gemv(cc),
        "Qwen35CudaModel::new must arm the production GEMV pair for sm_{cc}"
    );
    if cc >= 75 {
        assert_eq!(
            gpu.gemv_variants(),
            (Q4kVariant::HwDp4a, Q6kVariant::HwDp4a),
            "a DP4A-capable device must decode on the DP4A GEMVs"
        );
    }
    gpu.pin_reference_gemv();
    assert_eq!(
        gpu.gemv_variants(),
        (Q4kVariant::Mwv, Q6kVariant::Mwv),
        "pin_reference_gemv must restore the float pair for the isolation tests"
    );
}

fn gemv_env_is_set() -> bool {
    Qwen35CudaModel::GEMV_ENV_OVERRIDES
        .iter()
        .any(|v| std::env::var_os(v).is_some())
}

/// Teacher-forced positions after the prompt.
const STEPS: usize = 32;

/// The e2e quality falsifier for the DP4A production pair, on the 4B file.
///
/// PRE-REGISTERED (written before this test first ran): over the prompt plus
/// [`STEPS`] positions, teacher-forced on the CPU reference's own greedy
/// tokens, the DP4A pair
///
/// 1. disagrees with the CPU argmax at most **2** more positions than the float
///    pair does, and
/// 2. keeps every position's logit cosine to the CPU at or above **0.99**.
///
/// Greedy TEXT identity is not the bar: measured on 17 prompts it held on 10,
/// and each of the 7 divergences was a near-tie between two fluent
/// continuations ("every direction" / "all directions"). Any change in
/// activation precision moves those; this test bounds how often it moves the
/// argmax against an independent reference, and how far the logits move.
#[test]
#[serial_test::serial]
fn qwen35_cuda_4b_dp4a_gemv_stays_within_the_float_pair_of_cpu() {
    let executor = qwen35_cuda_file_or_skip!(MODEL_PATH_4B);
    if executor.gpu_profile.cc < 75 {
        eprintln!("SKIP: no DP4A on sm_{}", executor.gpu_profile.cc);
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH_4B).expect("map the 4B GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("4B hybrid layers");

    // The CPU reference picks the tokens; both GPU arms are forced along them.
    let mut cpu_state = qwen.new_state(PROMPT_4B.len() + STEPS);
    let mut tokens = PROMPT_4B.to_vec();
    let mut cpu_logits = Vec::new();
    for pos in 0..PROMPT_4B.len() + STEPS {
        let logits = qwen
            .forward_single_qwen35(tokens[pos], &mut cpu_state, pos)
            .expect("cpu 4B forward");
        if pos + 1 >= tokens.len() && pos + 1 < PROMPT_4B.len() + STEPS {
            tokens.push(crate::gguf::ops::argmax(&logits));
        }
        cpu_logits.push(logits);
    }

    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the 4B CUDA model");
    let mut arm = |q4k, q6k, name: &str| -> (usize, f32) {
        gpu.executor_mut().gpu_profile.q4k = q4k;
        gpu.executor_mut().gpu_profile.q6k = q6k;
        assert_eq!(gpu.gemv_variants(), (q4k, q6k), "{name}: arm not engaged");
        let mut state = gpu.new_state().expect("device state");
        let (mut wrong, mut min_cos) = (0usize, 1.0f32);
        for (pos, want) in cpu_logits.iter().enumerate() {
            let got = gpu
                .forward_single(tokens[pos], &mut state, pos)
                .expect("gpu 4B forward");
            let cos = cosine(&got, want);
            let (g, c) = (
                crate::gguf::ops::argmax(&got),
                crate::gguf::ops::argmax(want),
            );
            if g != c {
                wrong += 1;
            }
            min_cos = min_cos.min(cos);
            eprintln!("[4485 {name}] pos {pos}: argmax gpu {g} cpu {c} cosine {cos:.6}");
        }
        (wrong, min_cos)
    };
    let (float_wrong, float_cos) = arm(Q4kVariant::Mwv, Q6kVariant::Mwv, "float");
    let (dp4a_wrong, dp4a_cos) = arm(Q4kVariant::HwDp4a, Q6kVariant::HwDp4a, "dp4a");
    eprintln!(
        "[4485] {} positions: float {float_wrong} wrong, min cosine {float_cos:.6}; \
         dp4a {dp4a_wrong} wrong, min cosine {dp4a_cos:.6}",
        cpu_logits.len()
    );

    assert!(
        dp4a_wrong <= float_wrong + 2,
        "DP4A moved the argmax off the CPU reference at {dp4a_wrong} positions against the \
         float pair's {float_wrong}: more than the pre-registered +2"
    );
    assert!(
        dp4a_cos >= 0.99,
        "DP4A logit cosine to the CPU fell to {dp4a_cos:.6} < the pre-registered 0.99"
    );
}

/// Teacher-forced decode positions for the graph arm.
const GRAPH_STEPS: usize = 48;

/// #4485: the decode graph under the production DP4A GEMVs.
///
/// The DP4A kernels replay bit-exactly from the manual graph
/// (`tests_dp4a_graph_replay`). The graph STEP is not the eager step, though: it
/// runs the indirect RoPE and attention kernels, ~5e-6 off on the float logits.
/// DP4A re-rounds every GEMV input onto a Q8_1 grid, so that difference lands
/// on different grid points; measured, the logits then differ by ~0.15.
///
/// PRE-REGISTERED (written before this test first ran): teacher-forced on the
/// eager DP4A greedy tokens, from the capture token on,
///
/// 1. the largest graph-vs-eager logit difference is no larger than the largest
///    DP4A-vs-float eager difference (the replay moves the logits by no more
///    than the precision policy already does), and
/// 2. every position's graph-vs-eager logit cosine is at or above **0.999**.
fn qwen35_dp4a_graph_stays_within_the_dp4a_precision_band(
    path: &str,
    executor: crate::cuda::CudaExecutor,
) {
    if executor.gpu_profile.cc < 75 {
        eprintln!("SKIP: no DP4A on sm_{}", executor.gpu_profile.cc);
        return;
    }
    let mapped = crate::gguf::MappedGGUFModel::from_path(path).expect("map the GGUF");
    let base = super::load_cpu_model(&mapped);
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let prompt = super::LONG_PROMPT;
    let total = prompt.len() + GRAPH_STEPS;
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, total + 1)
        .expect("build the CUDA model");

    let mut run = |q4k, q6k, graph: bool, forced: Option<&[u32]>| -> (Vec<u32>, Vec<Vec<f32>>) {
        gpu.executor_mut().gpu_profile.q4k = q4k;
        gpu.executor_mut().gpu_profile.q6k = q6k;
        gpu.set_decode_graph(graph);
        let before = gpu.decode_graph_replays();
        let mut state = gpu.new_state().expect("device state");
        let mut tokens = prompt.to_vec();
        let mut all = Vec::with_capacity(total);
        for pos in 0..total {
            let token = forced.map_or_else(|| tokens[pos], |f| f[pos]);
            let logits = gpu.forward_single(token, &mut state, pos).expect("forward");
            if forced.is_none() && pos + 1 == tokens.len() && tokens.len() < total {
                tokens.push(crate::gguf::ops::argmax(&logits) as u32);
            }
            all.push(logits);
        }
        let replays = gpu.decode_graph_replays() - before;
        let want = if graph { total as u64 - 1 } else { 0 };
        assert_eq!(
            replays, want,
            "graph={graph}: the decode graph was not engaged as asked"
        );
        (forced.map_or(tokens, <[u32]>::to_vec), all)
    };
    let dp4a = (Q4kVariant::HwDp4a, Q6kVariant::HwDp4a);
    let float = (Q4kVariant::Mwv, Q6kVariant::Mwv);
    let (tokens, eager) = run(dp4a.0, dp4a.1, false, None);
    let (_, graphed) = run(dp4a.0, dp4a.1, true, Some(&tokens));
    let (_, eager_float) = run(float.0, float.1, false, Some(&tokens));

    let max_abs =
        |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let (mut replay_max, mut policy_max, mut min_cos, mut flips) = (0f32, 0f32, 1f32, 0usize);
    for pos in 0..total {
        let (r, p) = (
            max_abs(&graphed[pos], &eager[pos]),
            max_abs(&eager_float[pos], &eager[pos]),
        );
        let cos = cosine(&graphed[pos], &eager[pos]);
        let (g, e) = (
            crate::gguf::ops::argmax(&graphed[pos]),
            crate::gguf::ops::argmax(&eager[pos]),
        );
        flips += usize::from(g != e);
        replay_max = replay_max.max(r);
        policy_max = policy_max.max(p);
        min_cos = min_cos.min(cos);
        eprintln!(
            "[4485 graph] pos {pos}: replay max|d| {r:.4} dp4a-vs-float max|d| {p:.4} \
             cosine {cos:.6} argmax graph {g} eager {e}"
        );
    }
    eprintln!(
        "[4485 graph] {path}: {total} positions, replay max|d| {replay_max:.4} vs precision \
         band {policy_max:.4}, min cosine {min_cos:.6}, {flips} argmax flips"
    );
    assert!(
        replay_max <= policy_max,
        "the DP4A graph moved a logit by {replay_max:.4}, more than DP4A itself moves it off \
         float ({policy_max:.4})"
    );
    assert!(
        min_cos >= 0.999,
        "DP4A graph-vs-eager logit cosine fell to {min_cos:.6} < the pre-registered 0.999"
    );
}

#[test]
#[serial_test::serial]
fn qwen35_cuda_dp4a_graph_stays_within_the_dp4a_precision_band_0_8b() {
    let executor = qwen35_cuda_fixture_or_skip!();
    qwen35_dp4a_graph_stays_within_the_dp4a_precision_band(MODEL_PATH, executor);
}

#[test]
#[serial_test::serial]
fn qwen35_cuda_dp4a_graph_stays_within_the_dp4a_precision_band_4b() {
    let executor = qwen35_cuda_file_or_skip!(MODEL_PATH_4B);
    qwen35_dp4a_graph_stays_within_the_dp4a_precision_band(MODEL_PATH_4B, executor);
}
