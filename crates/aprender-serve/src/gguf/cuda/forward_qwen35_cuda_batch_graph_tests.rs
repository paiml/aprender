//! #5279: the captured batched step against the eager one, on the real
//! Qwen3.5-0.8B file.

use super::super::Qwen35StepOut;
use crate::gguf::cuda::Qwen35CudaModel;
use crate::gguf::forward_qwen35::Qwen35Model;

const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Ragged, distinct prompts, as in the batch tests.
const PROMPTS: [&[u32]; 4] = [
    &[9707],
    &[785, 6722, 315, 9625],
    &[22043, 374, 264, 3974, 11, 1879, 13],
    &[16, 10],
];

/// The token a caller takes next from one slot's output: its greedy token,
/// or the argmax of the logits it asked for on the host.
fn next_token(out: &Qwen35StepOut) -> u32 {
    match out {
        Qwen35StepOut::Token(t) => *t,
        Qwen35StepOut::Logits(l) => {
            let mut best = (0, f32::NEG_INFINITY);
            for (j, &v) in l.iter().enumerate() {
                if v > best.1 {
                    best = (j, v);
                }
            }
            best.0 as u32
        },
    }
}

/// Every step's outputs: the four slots step together for six steps, then
/// slot 3 leaves and three step for six more, so the run needs two graphs.
/// Slot 1 takes its logits on the host, the others their greedy tokens.
fn run(gpu: &mut Qwen35CudaModel<'_>) -> Vec<Vec<Qwen35StepOut>> {
    let mut states: Vec<_> = PROMPTS
        .iter()
        .map(|_| gpu.new_state().expect("device state"))
        .collect();
    let mut next = vec![0u32; PROMPTS.len()];
    for (i, prompt) in PROMPTS.iter().enumerate() {
        for (pos, &token) in prompt.iter().enumerate() {
            next[i] = gpu
                .forward_batch_greedy(&[token], std::slice::from_mut(&mut states[i]), &[pos])
                .expect("prefill token")[0];
        }
    }
    let mut positions: Vec<usize> = PROMPTS.iter().map(|p| p.len()).collect();
    let mut steps = Vec::new();
    for step in 0..12 {
        let m = if step < 6 { 4 } else { 3 };
        let host: Vec<bool> = (0..m).map(|i| i == 1).collect();
        let outs = gpu
            .forward_batch(&next[..m], &mut states[..m], &positions[..m], &host)
            .expect("batched step");
        for (i, out) in outs.iter().enumerate() {
            next[i] = next_token(out);
            positions[i] += 1;
        }
        steps.push(outs);
    }
    steps
}

/// One slot's eager and graphed outputs are the same kind and bitwise equal.
fn assert_same(step: usize, slot: usize, e: &Qwen35StepOut, g: &Qwen35StepOut) {
    match (e, g) {
        (Qwen35StepOut::Token(a), Qwen35StepOut::Token(b)) => {
            assert_eq!(a, b, "step {step}, slot {slot}: token");
        },
        (Qwen35StepOut::Logits(a), Qwen35StepOut::Logits(b)) => {
            let same =
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "step {step}, slot {slot}: logits differ");
        },
        _ => panic!("step {step}, slot {slot}: output kinds differ"),
    }
}

/// FALSIFY (#5279): every step through the captured graphs (two shapes,
/// captured once each and replayed after) is bitwise the eager batched step:
/// the same greedy tokens and the same host logits.
#[test]
#[serial_test::serial]
fn the_captured_batched_step_is_bitwise_the_eager_one() {
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

    gpu.set_batch_graph(false);
    let eager = run(&mut gpu);
    assert_eq!(gpu.batch_graph_counts(), (0, 0), "graphs off, yet one ran");

    gpu.set_batch_graph(true);
    let graphed = run(&mut gpu);
    let (replays, captures) = gpu.batch_graph_counts();
    assert_eq!(captures, 2, "one capture per shape (m = 4, then m = 3)");
    assert_eq!(replays, 10, "every other step replayed its shape's graph");

    assert_eq!(eager.len(), graphed.len(), "step count");
    for (step, (e, g)) in eager.iter().zip(&graphed).enumerate() {
        assert_eq!(e.len(), g.len(), "step {step}: slot count");
        for (slot, (e, g)) in e.iter().zip(g).enumerate() {
            assert_same(step, slot, e, g);
        }
    }
    eprintln!("[batch graph] 12 steps, 2 shapes: graphed == eager bitwise");
}
