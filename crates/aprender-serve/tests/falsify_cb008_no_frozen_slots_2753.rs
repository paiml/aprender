//! FALSIFY-CB-008 / aprender#2753: a batched decode slot must not freeze on one token.
//!
//! `contracts/continuous-batching-v1.yaml` has forbidden this since v1.0 — *"No frozen slots —
//! all M slots produce distinct tokens per decode step (not constant)"* — and #2753 is that
//! rule failing in production for four releases. The contract's `test:` field named a
//! `BATCHED_DECODE_TRACE` log; **the variable did not exist**, so the evidence it pointed at
//! could never have been read. This file is that field made executable.
//!
//! The measurement, on this branch, RTX 4090 sm_89, `qwen2.5-coder-1.5b-instruct-q4_k_m`,
//! greedy, 120-token cap, `[PMAT-044] Batch m=3 done` in the log proving the batched path
//! engaged and `[CB-008] step=N …` printing the ids:
//!
//! ```text
//!   origin/main 745fa8588   [CB-008] step=0..115  token_ids=[151662, 151662, 151662]
//!                           all three slots frozen, finish_reason=length at the cap
//!   this branch             varied ids every step, finish_reason=stop where the m=1
//!                           reference stops, byte-identical output at m=2/4/8
//! ```
//!
//! ## What decides the verdict
//!
//! `realizar::cb008_frozen_slots` — a pure function over one slot's generated stream, whose
//! discrimination controls run in the ordinary workspace `--lib` line on hosts with no CUDA at
//! all. They assert it rejects the two recorded signatures verbatim: token id 0 to the 400-token
//! cap (the `!!!!…` output #2753 opened with) and id 151662 for 116 of 120 steps (what this
//! branch measured on main). So "can this checker fire?" is answered on every PR, not only on a
//! nightly GPU lane, and this file is left to answer the question only a GPU can: *does the real
//! batched decode freeze?*
//!
//! ## Controls, because an all-varied assertion is easy to pass vacuously
//!
//! - the batched path is engaged **by construction** — M prompts handed to the batched entry
//!   point, not M HTTP requests and a hope that a batch formed;
//! - every slot must generate at least `MIN_GENERATED` tokens, enforced by the checker itself,
//!   which reports too-short as a FAILURE and never as a pass;
//! - a **harness positive control** runs the same assertion path over a synthetic frozen stream
//!   inside this process, so a refactor that turned the assertion into a no-op is caught here
//!   and not only in the unit tests;
//! - **two batch sizes**, m=3 and m=8, because they take different GEMM routes (m>=4 reaches
//!   the cuBLAS/FP8 decode route and m<4 does not), and a freeze on one of them is not visible
//!   from the other;
//! - an **M=1 reference control**, which is the reason this file has one at all: the test went
//!   RED on `qwen2.5-coder-0.5b-instruct-q4_k_m`, and that model's M=1 path is degenerate too
//!   (`@1\n@1\n@1`, "```" repeated, for four ordinary prompts). A frozen batched slot beside a
//!   frozen M=1 reference is not evidence about batching, so that case exits non-zero as
//!   UNMEASURABLE and explicitly refuses to name a code cause.
//!
//! ## Mutation
//!
//! One binary, one env var. `APR_STREAM_NONBLOCKING=1` restores the pre-#2767 stream flag —
//! host transfers (`cuMemcpyHtoD`/`DtoH`, legacy stream) then race the kernels in flight, the
//! decode argmaxes a partially written logits buffer, and the slots freeze. That is the RED.
//!
//! Requires a GPU and a real model named by `APR_FALSIFY_MODEL`. It does **not** skip when that
//! is missing — it FAILS, saying what is absent, for the reason spelled out at length in
//! `falsify_stream_ordering_2767.rs`: a skip is indistinguishable from a pass, and a synthetic
//! fixture cannot open the window this defect lives in. `.github/workflows/cuda-nightly.yml`
//! resolves a model and runs it.
#![cfg(feature = "cuda")]

use std::path::Path;

use realizar::apr::MappedAprModel;
use realizar::cb008_frozen_slots::{batch_frozen_verdict, frozen_slot_verdict, MIN_GENERATED};
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};

/// Slot counts to exercise. 3 is the batch size in #2753's evidence table; 8 crosses
/// `cublas_gemm_threshold()` (4) and the `m >= 5` FP8 shortcut, so both decode GEMM routes are
/// covered. A freeze on one is invisible from the other.
const BATCH_SIZES: [usize; 2] = [3, 8];

/// Comfortably above `MIN_GENERATED` so a slot that decodes normally is never judged
/// UNMEASURABLE, while keeping the nightly lane's cost to a few seconds per batch.
const MAX_TOKENS: usize = 64;

const MODEL_ENV: &str = "APR_FALSIFY_MODEL";

fn model_path() -> String {
    let Ok(path) = std::env::var(MODEL_ENV) else {
        panic!(
            "CB-008 UNRUNNABLE: {MODEL_ENV} is not set.\n\
             This falsifier needs a real batched decode on a real GPU. It fails rather than \
             skips because a skip is indistinguishable from a pass, and CB-008 spent four \
             releases as a sentence nobody ran while the defect it forbids was shipping.\n\
             Set {MODEL_ENV} to a GGUF or .apr model this GPU accepts, e.g.\n  \
             {MODEL_ENV}=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \\\n    \
             cargo test -p aprender-serve --features cuda --release \\\n      \
             --test falsify_cb008_no_frozen_slots_2753 -- --nocapture"
        )
    };
    assert!(
        Path::new(&path).exists(),
        "CB-008 UNRUNNABLE: {MODEL_ENV}={path} does not exist."
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

fn open_gpu_model() -> (String, OwnedQuantizedModelCuda) {
    let path = model_path();
    let model = match load(&path) {
        Ok(m) => m,
        Err(e) => panic!("CB-008 UNRUNNABLE: {MODEL_ENV}={path} could not be loaded: {e}"),
    };
    match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(cuda) => (path, cuda),
        Err(e) => panic!(
            "CB-008 UNRUNNABLE: the GPU refused {MODEL_ENV}={path}: {e}\n\
             Reported as a failure, not a skip: a model the GPU will not take cannot produce a \
             verdict about frozen slots."
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

/// One batch of `m` identical greedy prompts; each slot's GENERATED suffix.
fn one_batch(
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
        return Err(format!(
            "CB-008 UNMEASURABLE: asked for {m} slots and got {} sequences back, so the batched \
             path did not run the batch this test is about",
            seqs.len()
        ));
    }
    Ok(seqs
        .into_iter()
        .map(|s| s.get(prompt.len()..).unwrap_or(&[]).to_vec())
        .collect())
}

#[test]
fn cb008_batched_decode_slots_are_not_frozen() {
    // HARNESS POSITIVE CONTROL, before any GPU work. The assertion path used below must reject
    // the recorded #2753 signature in THIS process. Without it, a refactor that made
    // `batch_frozen_verdict` return Ok unconditionally would turn every run below green and
    // nothing here would notice.
    let synthetic_frozen = vec![vec![151_662u32; MAX_TOKENS]; BATCH_SIZES[0]];
    let control = batch_frozen_verdict(&synthetic_frozen).expect_err(
        "CB-008 HARNESS BROKEN: the verdict function accepted a batch of slots each emitting \
         one token id for the whole generation — the exact output aprender#2753 reported. \
         Every result below would be meaningless.",
    );
    assert!(
        control.contains("FROZEN"),
        "CB-008 HARNESS BROKEN: the verdict fired but does not say what it found: {control}"
    );

    let (path, mut cuda) = open_gpu_model();
    eprintln!(
        "[CB-008] model={path} batch_sizes={BATCH_SIZES:?} max_tokens={MAX_TOKENS} \
         min_generated={MIN_GENERATED}"
    );

    // A prompt that keeps generating past MAX_TOKENS, so a healthy slot is never judged
    // UNMEASURABLE for stopping early. Same token ids the #2767 falsifier uses.
    let prompt: Vec<u32> = vec![785, 11, 1879, 374];

    // M=1 REFERENCE CONTROL. CB-008 is a statement about the BATCHED path, so it may only be
    // decided on a model whose M=1 path decodes. This control is not decoration: it was added
    // because the test went RED on qwen2.5-coder-0.5b-instruct-q4_k_m, and the m=1 path on that
    // model is degenerate too — its greedy output for four ordinary prompts is
    // `# This is the first\n# This is the second`, "```" repeated, `@1\n@1\n@1`. A frozen batched
    // slot next to a frozen m=1 reference says nothing about batching, and reporting it as a
    // batching defect is exactly the "named a code cause for a box it could not evaluate" failure
    // this repo has hit repeatedly. So that case is UNMEASURABLE — still a non-zero exit, never a
    // silent skip, but never a code verdict either.
    let m1 = cuda
        .generate_gpu_resident(&prompt, &greedy(MAX_TOKENS))
        .unwrap_or_else(|e| panic!("CB-008 UNMEASURABLE: the M=1 reference could not run: {e}"));
    let m1_generated: Vec<u32> = m1.get(prompt.len()..).unwrap_or(&[]).to_vec();
    eprintln!(
        "[CB-008] m=1 reference: len={} distinct={} longest_run={} head={:?}",
        m1_generated.len(),
        realizar::cb008_frozen_slots::distinct_count(&m1_generated),
        realizar::cb008_frozen_slots::longest_run(&m1_generated),
        &m1_generated[..m1_generated.len().min(8)]
    );
    if let Err(v) = frozen_slot_verdict(0, &m1_generated) {
        panic!(
            "CB-008 UNMEASURABLE on {path}: the M=1 fast path is ITSELF frozen on this \
             model/prompt, so a frozen batched slot cannot be attributed to batching. This is \
             NOT a verdict about the batched decode.\n  {v}\n\
             Pick a model whose greedy M=1 output varies (the reference model for this \
             falsifier is qwen2.5-coder-1.5b-instruct-q4_k_m; the 0.5b sibling is degenerate on \
             this GPU and is measured to fail here), or fix the M=1 path first."
        );
    }

    let mut failures: Vec<String> = Vec::new();
    for &m in &BATCH_SIZES {
        let slots = match one_batch(&mut cuda, &prompt, m) {
            Ok(s) => s,
            Err(e) => panic!("CB-008: the m={m} batch could not run: {e}"),
        };
        for (i, s) in slots.iter().enumerate() {
            eprintln!(
                "[CB-008] m={m} slot {i}: len={} distinct={} longest_run={} head={:?}",
                s.len(),
                realizar::cb008_frozen_slots::distinct_count(s),
                realizar::cb008_frozen_slots::longest_run(s),
                &s[..s.len().min(8)]
            );
        }
        if let Err(v) = batch_frozen_verdict(&slots) {
            failures.push(format!("m={m}: {v}"));
        }
    }

    assert!(
        failures.is_empty(),
        "CB-008 RED (aprender#2753) on {path}:\n{}\n\n\
         Mechanism, if this is a regression of the original: host transfers in the batched \
         decode path are legacy-stream cuMemcpy and do not order against a non-blocking \
         stream, so the decode argmaxes a partially written logits buffer and every slot \
         locks onto one id — a freshly allocated buffer reads as zero, which argmaxes to \
         token 0, the `!!!!` output. Reproduce the RED in ONE binary with \
         APR_STREAM_NONBLOCKING=1.",
        failures.join("\n")
    );

    eprintln!("[CB-008] GREEN: no frozen slots at m={BATCH_SIZES:?}, {MAX_TOKENS} tokens per slot");
}
