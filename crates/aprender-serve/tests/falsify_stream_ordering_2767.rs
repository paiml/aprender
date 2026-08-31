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
//! Requires a GPU and a real model, named by `APR_FALSIFY_MODEL`. It does **not** skip when that
//! is missing -- it FAILS, saying what is absent. A skip is indistinguishable from a pass in CI,
//! and this is the only falsifier for a defect that answered one deterministic question 11
//! different ways. See the note on `MODEL_ENV` for why a committed fixture cannot stand in here,
//! and `.github/workflows/cuda-nightly.yml` for the lane that actually runs it.
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

/// The model this runs against. **No default, no fallback, and no skip.**
///
/// `APR_FALSIFY_MODEL` must name a GGUF or `.apr` model that this GPU accepts. If it is unset,
/// missing, or refused, this test goes RED. That is deliberate, and it is the narrow choice
/// among the three shapes `scripts/check_test_fixture_paths.sh` accepts:
///
/// - **A committed fixture would be strictly better, and is MEASURED NOT TO WORK for this
///   defect.** The synthetic, model-free falsifier for exactly this race already exists:
///   `aprender-gpu`'s `test_sync_upload_visible_to_nonblocking_stream`, which uploads
///   synchronously and reads back from a `CudaStream`. It caught 15 stale reads in 200 rounds
///   when it was written; it now passes 128/128 under `APR_STREAM_NONBLOCKING=1
///   APR_ORD9_DRAIN_SKIP=1` -- the exact pre-fix configuration it was built to catch. A
///   committed fixture that stays green under the defect is the "green proving nothing" this
///   file exists to eliminate. What a fixture CAN pin here is the flag, and it already does:
///   the `perf053_*` unit tests in `aprender-gpu/src/driver/stream.rs` assert
///   `CU_STREAM_DEFAULT` is the default with no GPU at all. The behavioural half needs enough
///   kernels genuinely in flight to open the race window, and that is a real decode.
/// - **A hardcoded path** under `/home` is green on every machine that lacks it, which is
///   every machine but one. That is what this constant used to be, and it is the population
///   `check_test_fixture_paths.sh` exists to stop growing.
/// - **An env var that fails loudly when unset** is therefore the correct shape. On a
///   `--features cuda` host that has not been provisioned this is RED and says what is
///   missing, rather than passing quietly. `.github/workflows/cuda-nightly.yml` resolves a
///   model, fails the job if it cannot, and sets this variable -- so the test does run
///   somewhere, which is the other half of not being theatre.
///
/// Note that the fallback list this replaced was itself a scar: the 0.5B `.apr` in it FAILS
/// the GPU/CPU parity gate on sm_89 (cosine 0.9636 against a 0.98 floor, a pre-existing and
/// unrelated defect), so `OwnedQuantizedModelCuda::new` refuses it and the run walked on to
/// the next entry. Naming exactly one model removes the silent walk-on: the model you asked
/// for is the model that must produce the verdict.
const MODEL_ENV: &str = "APR_FALSIFY_MODEL";

fn model_path() -> String {
    let Ok(path) = std::env::var(MODEL_ENV) else {
        panic!(
            "perf053 UNRUNNABLE: {MODEL_ENV} is not set.\n\
             This falsifier needs a real decode on a real GPU -- see the note on MODEL_ENV for \
             why a synthetic fixture cannot see this race. It fails rather than skips because a \
             skip is indistinguishable from a pass, and this is the only falsifier for a defect \
             that returned 11 distinct continuations to 40 identical greedy requests.\n\
             Set {MODEL_ENV} to a GGUF or .apr model this GPU accepts, e.g.\n  \
             {MODEL_ENV}=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \\\n    \
             cargo test -p aprender-serve --features cuda --release \\\n      \
             --test falsify_stream_ordering_2767 -- --nocapture"
        )
    };
    assert!(
        Path::new(&path).exists(),
        "perf053 UNRUNNABLE: {MODEL_ENV}={path} does not exist."
    );
    path
}

fn load(path: &str) -> Result<OwnedQuantizedModel, String> {
    if path.ends_with(".apr") {
        let mapped = MappedAprModel::from_path(path)
            .map_err(|e| format!("MappedAprModel::from_path: {e}"))?;
        OwnedQuantizedModel::from_apr(&mapped).map_err(|e| format!("from_apr: {e}"))
    } else {
        let mapped = MappedGGUFModel::from_path(path)
            .map_err(|e| format!("MappedGGUFModel::from_path: {e}"))?;
        OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| format!("from_mapped: {e}"))
    }
}

/// The model named by `APR_FALSIFY_MODEL`, loaded and accepted by the GPU. Returns its path
/// too, so the run says which model produced the verdict. Every failure here PANICS: an
/// unrunnable falsifier must be distinguishable from a passing one.
fn open_gpu_model() -> (String, OwnedQuantizedModelCuda) {
    let path = model_path();
    let model = match load(&path) {
        Ok(m) => m,
        Err(e) => panic!("perf053 UNRUNNABLE: {MODEL_ENV}={path} could not be loaded: {e}"),
    };
    match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(cuda) => (path, cuda),
        Err(e) => panic!(
            "perf053 UNRUNNABLE: the GPU refused {MODEL_ENV}={path}: {e}\n\
             This is reported as a failure, not a skip: a model the GPU will not take cannot \
             produce a verdict on stream ordering, and pretending otherwise is how the \
             previous fallback list turned a refusal into a green."
        ),
    }
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

    // No `else { return }` here on purpose. This used to skip when it found no usable model,
    // and a skip reads exactly like a pass in CI. `open_gpu_model` panics with what is missing.
    let (path, mut cuda) = open_gpu_model();
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
