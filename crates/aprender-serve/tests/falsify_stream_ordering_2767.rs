//! PERF-053 / aprender#2767: batched decode must answer the same greedy question the same way.
//!
//! `CudaStream::new` used to create this crate's streams with `CU_STREAM_NON_BLOCKING`, which
//! CUDA explicitly EXCLUDES from legacy default-stream ordering, while `GpuBuffer::copy_from_host`
//! / `copy_to_host` are `cuMemcpyHtoD` / `cuMemcpyDtoH` -- LEGACY-stream transfers. A legacy
//! transfer does not order against a non-blocking stream, so every host transfer in the batched
//! decode path raced the kernels in flight, and the surrounding code is written throughout as
//! though the transfers were ordered against it.
//!
//! The measurement in #2767 is the shape this test reproduces in-process. Ten rounds of four
//! concurrent IDENTICAL greedy requests against a live server returned:
//!
//! ```text
//!   CU_STREAM_NON_BLOCKING   40 responses, 11 DISTINCT continuations, most of them garbage
//!   CU_STREAM_DEFAULT        40 responses,  1 distinct continuation, the correct answer
//! ```
//!
//! Two properties of that data drive the design here, and getting either wrong makes the test
//! blind:
//!
//! 1. **The corruption is per-BATCH, not per-slot.** Within one round all four slots agreed;
//!    the answer changed ACROSS rounds. So comparing the M slots of a single batch against each
//!    other cannot see this defect. The test must run several rounds and compare across them.
//! 2. **It is a determinism property, not an oracle property.** This asserts only that repeated
//!    identical greedy batches agree with EACH OTHER. It deliberately does NOT compare against
//!    the M=1 path: FALSIFY-CB-006 has a separate, still-open residual gap between batched and
//!    M=1, and folding that in would make this test RED for a defect it is not about.
//!
//! Controls, because an all-equal assertion is the easiest kind to pass vacuously:
//!
//! - the batched path is engaged BY CONSTRUCTION (m prompts handed to the batched entry point),
//!   not inferred from firing m HTTP requests and hoping a batch formed;
//! - every slot must actually generate tokens, so "all equal" is never equality of empties;
//! - a DISCRIMINATION round with a different prompt must produce a different continuation. If the
//!   comparator cannot tell two different answers apart, the main assertion proves nothing, and
//!   that failure is reported as such.
//!
//! Requires a GPU and a local model, so it self-skips loudly rather than reporting a code verdict
//! on a host it cannot evaluate.
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::Path;

use realizar::apr::MappedAprModel;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};

/// Number of identical batches. MUST be > 1: the corruption is per-batch, so a single round
/// cannot see it (all slots within a round agreed even on the racy tree).
const ROUNDS: usize = 4;
/// Slots per batch. 4 is the concurrency the #2767 measurement used.
const SLOTS: usize = 4;
const MAX_TOKENS: usize = 24;
/// Below this, "every slot agrees" is a statement about near-empty outputs.
const MIN_GENERATED: usize = 6;

/// Models this can run on, in preference order. Override with `APR_FALSIFY_MODEL`.
///
/// The 1.5B GGUF is FIRST because it is the model #2767 measured. The order matters more than
/// it looks: the 0.5B `.apr` here FAILS the GPU/CPU parity gate on sm_89 (cosine 0.9636 against
/// a 0.98 floor, a pre-existing and unrelated defect), and `OwnedQuantizedModelCuda::new`
/// refuses it. Trying it first made this test SKIP -- a green that proved nothing. So a
/// candidate that fails to load is skipped over rather than ending the run.
const CANDIDATES: &[&str] = &[
    "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
    "/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-q4k.apr",
];

fn candidates() -> Vec<String> {
    match std::env::var("APR_FALSIFY_MODEL") {
        Ok(p) => vec![p],
        Err(_) => CANDIDATES.iter().map(|p| (*p).to_string()).collect(),
    }
}

fn load(path: &str) -> Option<OwnedQuantizedModel> {
    if path.ends_with(".apr") {
        let mapped = MappedAprModel::from_path(path)
            .map_err(|e| eprintln!("[perf053] {path}: MappedAprModel::from_path: {e}"))
            .ok()?;
        OwnedQuantizedModel::from_apr(&mapped)
            .map_err(|e| eprintln!("[perf053] {path}: from_apr: {e}"))
            .ok()
    } else {
        let mapped = MappedGGUFModel::from_path(path)
            .map_err(|e| eprintln!("[perf053] {path}: MappedGGUFModel::from_path: {e}"))
            .ok()?;
        OwnedQuantizedModel::from_mapped(&mapped)
            .map_err(|e| eprintln!("[perf053] {path}: from_mapped: {e}"))
            .ok()
    }
}

/// First candidate that exists, loads, AND is accepted by the GPU. Returns its path too, so the
/// run says which model produced the verdict.
fn open_gpu_model() -> Option<(String, OwnedQuantizedModelCuda)> {
    for path in candidates() {
        if !Path::new(&path).exists() {
            continue;
        }
        let Some(model) = load(&path) else { continue };
        match OwnedQuantizedModelCuda::new(model, 0) {
            Ok(cuda) => return Some((path, cuda)),
            Err(e) => eprintln!("[perf053] {path}: GPU refused it: {e}"),
        }
    }
    None
}

fn greedy(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 1,
        repeat_penalty: 1.0,
        stop_tokens: vec![],
        ..Default::default()
    }
}

/// Run ONE batch of `m` identical greedy prompts; return each slot's GENERATED suffix.
fn one_round(
    cuda: &mut OwnedQuantizedModelCuda,
    prompt: &[u32],
    m: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let prompts: Vec<Vec<u32>> = (0..m).map(|_| prompt.to_vec()).collect();
    let configs: Vec<QuantizedGenerateConfig> = (0..m).map(|_| greedy(MAX_TOKENS)).collect();
    let on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>> =
        (0..m).map(|_| Box::new(|_: u32| true) as _).collect();

    let seqs = cuda
        .generate_batched_streaming(&prompts, &configs, on_tokens)
        .map_err(|e| format!("batched generation failed: {e}"))?;
    if seqs.len() != m {
        return Err(format!("expected {m} slot sequences, got {}", seqs.len()));
    }
    Ok(seqs
        .into_iter()
        .map(|s| s.get(prompt.len()..).unwrap_or(&[]).to_vec())
        .collect())
}

#[test]
fn perf053_identical_greedy_batches_return_one_continuation() {
    // Keep m >= 4 off the cuBLAS decode route. That route's divergence at m >= 4 is a
    // SEPARATE, named defect (contracts/continuous-batching-v1.yaml, CB-006 round 9); leaving
    // it live here would confound this test with a different bug. Set before any CUDA work,
    // since the threshold is read on first use.
    if std::env::var("CUBLAS_GEMM_THRESHOLD").is_err() {
        std::env::set_var("CUBLAS_GEMM_THRESHOLD", "32");
    }

    let Some((path, mut cuda)) = open_gpu_model() else {
        eprintln!(
            "SKIP perf053: no usable GPU model on this host. Set APR_FALSIFY_MODEL. \
             Tried {:?}",
            candidates()
        );
        return;
    };
    eprintln!("[perf053] model={path} rounds={ROUNDS} slots={SLOTS} max_tokens={MAX_TOKENS}");

    // Two prompts that must produce DIFFERENT answers. `alt` is the discrimination control.
    let prompt: Vec<u32> = vec![785, 11, 1879, 374];
    let alt: Vec<u32> = vec![7985, 12, 264, 1985, 374, 264];

    // distinct continuation -> the (round, slot) labels that produced it.
    let mut seen: BTreeMap<Vec<u32>, Vec<String>> = BTreeMap::new();
    let mut short = Vec::new();

    for round in 0..ROUNDS {
        let outs = match one_round(&mut cuda, &prompt, SLOTS) {
            Ok(o) => o,
            Err(e) => panic!("perf053: round {round} could not run: {e}"),
        };
        for (slot, out) in outs.into_iter().enumerate() {
            if out.len() < MIN_GENERATED {
                short.push(format!("r{round}s{slot} len={}", out.len()));
            }
            eprintln!(
                "[perf053] r{round} s{slot} len={} head={:?}",
                out.len(),
                &out[..out.len().min(8)]
            );
            seen.entry(out)
                .or_default()
                .push(format!("r{round}s{slot}"));
        }
    }

    assert!(
        short.is_empty(),
        "perf053 UNUSABLE: {} slot(s) generated fewer than {MIN_GENERATED} tokens ({}), so \
         'every batch agrees' would be agreement between near-empty outputs, not evidence. \
         Pick a prompt/model that decodes.",
        short.len(),
        short.join(", ")
    );

    // Discrimination control: a DIFFERENT prompt must give a DIFFERENT continuation. Without
    // this, an assertion that N outputs are equal would also pass on a comparator that cannot
    // tell any two outputs apart.
    let alt_out = match one_round(&mut cuda, &alt, SLOTS) {
        Ok(o) => o,
        Err(e) => panic!("perf053: discrimination round could not run: {e}"),
    };
    let reference = seen
        .keys()
        .next()
        .cloned()
        .expect("perf053: no continuations were collected at all");
    assert!(
        alt_out.iter().any(|o| *o != reference),
        "perf053 UNUSABLE: a different prompt produced the SAME continuation as the reference, \
         so this comparator cannot distinguish two answers and the equality assertion below \
         would pass vacuously. reference={:?}",
        &reference[..reference.len().min(8)]
    );

    // THE ASSERTION. #2767: 11 distinct out of 40 on the racy tree, 1 out of 40 once the streams
    // are ordered against the legacy stream the host transfers actually use.
    if seen.len() > 1 {
        let mut detail = String::new();
        for (i, (cont, who)) in seen.iter().enumerate() {
            detail.push_str(&format!(
                "\n  [{i}] x{:<2} {} head={:?}",
                who.len(),
                who.join(","),
                &cont[..cont.len().min(12)]
            ));
        }
        panic!(
            "perf053 RED (aprender#2767): {} identical greedy batches of {SLOTS} produced {} \
             DISTINCT continuations, expected 1. Host transfers (cuMemcpyHtoD/DtoH) are \
             legacy-stream and do not order against a CU_STREAM_NON_BLOCKING stream, so they \
             race the kernels in flight.{detail}",
            ROUNDS,
            seen.len()
        );
    }
    eprintln!(
        "[perf053] GREEN: {} outputs, 1 distinct continuation",
        ROUNDS * SLOTS
    );
}
