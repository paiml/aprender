//! #3595: a resident session must produce exactly what a one-shot generate of
//! the same prompt produces — on the state it reused, on a state it reset in
//! place, on either backend. On the CPU the one-shot is the per-token
//! reference loop (an independent oracle, #4263); on the GPU it is a fresh
//! `apr run` load, so reuse is compared with no reuse.

use super::*;
use crate::gguf::forward_qwen35::qwen35_reference_generate;
use crate::gguf::QuantizedGenerateConfig;
use crate::session::turn_budget;

/// The real hybrid file the rest of the Qwen3.5 tests are specified against.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

macro_rules! mapped_or_skip {
    () => {{
        if !std::path::Path::new(MODEL_PATH).exists() {
            eprintln!("SKIP: {MODEL_PATH} is absent");
            return;
        }
        MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF")
    }};
}

fn encode(mapped: &MappedGGUFModel, text: &str) -> Vec<u32> {
    mapped
        .model
        .encode(text)
        .expect("the GGUF tokenizer encodes")
}

/// Greedy, no stop tokens: every turn generates exactly `max_tokens`.
fn greedy(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
        ..Default::default()
    }
}

fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// The per-token CPU reference (test-only since #4263).
fn one_shot_cpu(
    mapped: &MappedGGUFModel,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
) -> Vec<u32> {
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    qwen35_reference_generate(mapped, &base, prompt, config).expect("one-shot generate")
}

#[test]
fn a_session_can_move_between_threads() {
    // `apr serve` holds the session across requests on a multi-threaded runtime.
    fn assert_send<T: Send>() {}
    assert_send::<Qwen35Session>();
}

#[test]
fn the_budget_is_max_tokens_until_the_declared_context_ends_it() {
    assert_eq!(turn_budget(10, 5, 100), (5, false));
    assert_eq!(
        turn_budget(95, 5, 100),
        (5, false),
        "exactly enough room is not a cap"
    );
    assert_eq!(
        turn_budget(97, 5, 100),
        (3, true),
        "the context ends first, and says so"
    );
    assert_eq!(
        turn_budget(99, 0, 100),
        (0, false),
        "asking for nothing is not capped"
    );
}

#[test]
fn a_prompt_the_declared_context_cannot_hold_is_refused_whole() {
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    let ctx = session.context_length();
    assert!(ctx > 1, "the GGUF declares a context length");
    let prompt = vec![0u32; ctx];
    let err = session
        .generate(&prompt, &greedy(4), &mut |_| true)
        .expect_err("a prompt as long as the context has no room to answer");
    let msg = err.to_string();
    assert!(msg.contains("refused whole rather than truncated"), "{msg}");
    assert!(
        msg.contains(&ctx.to_string()),
        "names the declared context: {msg}"
    );
    assert_eq!(session.processed_len(), 0, "nothing was prefilled");
}

#[test]
fn cpu_a_turn_that_extends_the_last_one_reuses_the_state_and_matches_one_shot() {
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    assert!(!session.on_gpu(), "--no-gpu is the CPU route");
    let config = greedy(6);

    let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
    let t1 = session
        .generate(&p1, &config, &mut |_| true)
        .expect("turn 1");
    assert_eq!(t1.reused, 0);
    assert_eq!(
        t1.tokens,
        one_shot_cpu(&mapped, &p1, &config),
        "turn 1 is the one-shot"
    );

    let mut p2 = t1.tokens.clone();
    p2.extend(encode(
        &mapped,
        &format!("<|im_end|>\n{}", user_turn("And of Chile?")),
    ));
    let t2 = session
        .generate(&p2, &config, &mut |_| true)
        .expect("turn 2");
    // The last token of turn 1 was chosen, never forwarded: the state holds
    // everything before it.
    assert_eq!(
        t2.reused,
        t1.tokens.len() - 1,
        "turn 2 prefilled only its new suffix"
    );
    assert_eq!(
        t2.tokens,
        one_shot_cpu(&mapped, &p2, &config),
        "a reused state must decode exactly what a fresh one does"
    );
}

#[test]
fn cpu_a_turn_that_does_not_extend_resets_in_place_and_matches_one_shot() {
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    let config = greedy(6);
    let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
    session
        .generate(&p1, &config, &mut |_| true)
        .expect("turn 1");

    // Unrelated: the recurrent state from turn 1 must not leak into it.
    let p2 = encode(&mapped, &user_turn("Count from one to five."));
    let t2 = session
        .generate(&p2, &config, &mut |_| true)
        .expect("turn 2");
    assert_eq!(t2.reused, 0);
    assert_eq!(t2.tokens, one_shot_cpu(&mapped, &p2, &config));
}

#[test]
fn on_token_returning_false_ends_the_turn_after_that_token() {
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
    let mut seen = Vec::new();
    let t = session
        .generate(&p1, &greedy(8), &mut |tok| {
            seen.push(tok);
            seen.len() < 3
        })
        .expect("turn");
    assert_eq!(seen.len(), 3);
    assert_eq!(
        &t.tokens[p1.len()..],
        seen.as_slice(),
        "every emitted token is in the turn"
    );
}

/// Where a chat prompt's checkpoint goes: before its last `<|im_start|>`.
fn header_at(mapped: &MappedGGUFModel, prompt: &[u32]) -> usize {
    let im_start = encode(mapped, "<|im_start|>");
    assert_eq!(im_start.len(), 1, "<|im_start|> is one token");
    prompt
        .iter()
        .rposition(|&t| t == im_start[0])
        .expect("a chat prompt holds <|im_start|>")
}

/// Turn 2 as a chat server renders it: turn 1's reply re-rendered from text
/// (here, not the tokens turn 1 generated), then the new user turn.
fn re_rendered_turn_two(mapped: &MappedGGUFModel, p1: &[u32]) -> Vec<u32> {
    let mut p2 = p1.to_vec();
    p2.extend(encode(
        mapped,
        &format!(
            "The capital of Peru is Lima.<|im_end|>\n{}",
            user_turn("And of Chile?")
        ),
    ));
    p2
}

#[test]
fn cpu_an_identical_prompt_again_resumes_at_its_generation_header_and_matches_one_shot() {
    // #4214: the same prompt twice cost the whole prefill twice.
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    let config = greedy(6);
    let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
    let t1 = session
        .generate(&p1, &config, &mut |_| true)
        .expect("turn 1");
    let t2 = session
        .generate(&p1, &config, &mut |_| true)
        .expect("turn 2");
    assert_eq!(
        t2.reused,
        header_at(&mapped, &p1),
        "resumed at the checkpoint"
    );
    assert!(t2.reused > 0);
    assert_eq!(t2.tokens, t1.tokens);
    assert_eq!(t2.tokens, one_shot_cpu(&mapped, &p1, &config));
}

#[test]
fn cpu_a_re_rendered_turn_two_resumes_at_turn_ones_header_and_matches_one_shot() {
    // #4274: serve re-renders the history, so turn 2 never extended turn 1.
    let mapped = mapped_or_skip!();
    let mut session = Qwen35Session::load(&mapped, true).expect("load");
    let config = greedy(6);
    let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
    session
        .generate(&p1, &config, &mut |_| true)
        .expect("turn 1");
    let p2 = re_rendered_turn_two(&mapped, &p1);
    let t2 = session
        .generate(&p2, &config, &mut |_| true)
        .expect("turn 2");
    assert_eq!(
        t2.reused,
        header_at(&mapped, &p1),
        "resumed at turn 1's checkpoint"
    );
    assert_eq!(
        t2.tokens,
        one_shot_cpu(&mapped, &p2, &config),
        "a resumed state must decode exactly what a fresh one does"
    );
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;

    /// The one-shot GPU path `apr run --gpu` takes: a fresh load sized to the
    /// one call, one turn.
    fn one_shot_gpu(
        mapped: &MappedGGUFModel,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Vec<u32> {
        let qwen =
            Qwen35Forward::cached_host(std::path::Path::new(MODEL_PATH), mapped).expect("host");
        let positions = prompt.len() + config.max_tokens;
        let mut one = Qwen35Session::load_for_run(qwen, mapped, false, positions).expect("load");
        let turn = one
            .generate(prompt, config, &mut |_| true)
            .expect("one-shot");
        assert!(
            turn.used_gpu,
            "the one-shot reference must itself be a GPU run"
        );
        turn.tokens
    }

    fn gpu_session_or_skip(mapped: &MappedGGUFModel) -> Option<Qwen35Session> {
        if !crate::cuda::CudaExecutor::is_available() {
            eprintln!("SKIP: no CUDA device");
            return None;
        }
        let session = Qwen35Session::load(mapped, false).expect("load");
        assert!(
            session.on_gpu(),
            "a CUDA build on a CUDA host serves from the GPU"
        );
        Some(session)
    }

    #[test]
    fn gpu_a_turn_that_extends_the_last_one_reuses_the_state_and_matches_one_shot() {
        let mapped = mapped_or_skip!();
        let Some(mut session) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let config = greedy(6);

        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        let t1 = session
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");
        assert!(t1.used_gpu);
        assert_eq!(
            t1.tokens,
            one_shot_gpu(&mapped, &p1, &config),
            "turn 1 is the one-shot"
        );

        let mut p2 = t1.tokens.clone();
        p2.extend(encode(
            &mapped,
            &format!("<|im_end|>\n{}", user_turn("And of Chile?")),
        ));
        let t2 = session
            .generate(&p2, &config, &mut |_| true)
            .expect("turn 2");
        assert!(
            t2.used_gpu,
            "turn 2 stays on the GPU: no rebuild, no fallback"
        );
        assert_eq!(
            t2.reused,
            t1.tokens.len() - 1,
            "turn 2 prefilled only its new suffix"
        );
        assert_eq!(
            t2.tokens,
            one_shot_gpu(&mapped, &p2, &config),
            "a reused device state must decode exactly what a fresh one does"
        );
    }

    /// Built on one thread, driven from another — the shape `apr serve` has. A
    /// CUDA context is current per thread; without binding it the first
    /// allocation fails with CUDA_ERROR_INVALID_CONTEXT and the session falls
    /// back to the CPU (measured on #3571 before the fix).
    #[test]
    fn gpu_a_session_built_on_one_thread_serves_from_another() {
        let mapped = mapped_or_skip!();
        let Some(session) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let config = greedy(6);
        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        let want = one_shot_gpu(&mapped, &p1, &config);

        let prompt = p1.clone();
        let cfg = config.clone();
        let turn = std::thread::spawn(move || {
            let mut session = session;
            let turn = session
                .generate(&prompt, &cfg, &mut |_| true)
                .expect("turn");
            (turn, session.on_gpu())
        })
        .join()
        .expect("the worker thread");
        let (turn, still_on_gpu) = turn;
        assert!(
            turn.used_gpu && still_on_gpu,
            "fell back on the worker thread"
        );
        assert_eq!(turn.tokens, want);
    }

    /// 0.69.3: `apr serve` prefills through the batched prefill, not one token at a
    /// time. Engagement is read off the session's counter (a speed would only look
    /// like it), and the tokens must be exactly the one-token path's — on a first
    /// turn (position 0) and on an extending turn (a nonzero start, which reads the
    /// first call's KV rows and recurrent state).
    #[test]
    fn gpu_serve_prefill_is_batched_and_token_identical_to_the_one_token_path() {
        let mapped = mapped_or_skip!();
        let Some(mut batched) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let Some(mut one_token) = gpu_session_or_skip(&mapped) else {
            return;
        };
        one_token.engine_mut().per_token_prefill = true;
        let config = greedy(8);

        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        let b1 = batched
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");
        let o1 = one_token
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");
        // #4214: a prompt is prefilled in two spans, split at its generation header so the
        // checkpoint can be taken there; each span goes through the batched prefill.
        assert_eq!(
            batched.batched_prefills(),
            2,
            "both spans of turn 1's prompt went through the batched prefill"
        );
        assert_eq!(
            one_token.batched_prefills(),
            0,
            "the control prefilled per token"
        );
        assert!(b1.used_gpu && o1.used_gpu, "neither fell back to the CPU");
        assert_eq!(b1.tokens, o1.tokens, "turn 1: batched == one-token");

        let mut p2 = b1.tokens.clone();
        p2.extend(encode(
            &mapped,
            &format!("<|im_end|>\n{}", user_turn("And of Chile?")),
        ));
        let b2 = batched
            .generate(&p2, &config, &mut |_| true)
            .expect("turn 2");
        let o2 = one_token
            .generate(&p2, &config, &mut |_| true)
            .expect("turn 2");
        assert_eq!(b2.reused, b1.tokens.len() - 1, "turn 2 extended the state");
        assert_eq!(
            batched.batched_prefills(),
            4,
            "both spans of turn 2's new suffix went through the batched prefill, from a nonzero position"
        );
        assert!(b2.used_gpu && o2.used_gpu, "neither fell back to the CPU");
        assert_eq!(b2.tokens, o2.tokens, "turn 2: batched == one-token");
        assert_eq!(
            b2.tokens,
            one_shot_gpu(&mapped, &p2, &config),
            "and both are what `apr run` decodes"
        );
    }

    #[test]
    fn gpu_a_turn_that_does_not_extend_resets_in_place_and_matches_one_shot() {
        let mapped = mapped_or_skip!();
        let Some(mut session) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let config = greedy(6);
        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        session
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");

        // reset_state must zero the conv windows and recurrent states: turn 1's
        // must not leak into an unrelated prompt.
        let p2 = encode(&mapped, &user_turn("Count from one to five."));
        let t2 = session
            .generate(&p2, &config, &mut |_| true)
            .expect("turn 2");
        assert!(t2.used_gpu);
        assert_eq!(t2.reused, 0);
        assert_eq!(t2.tokens, one_shot_gpu(&mapped, &p2, &config));
    }

    #[test]
    fn gpu_an_identical_prompt_again_resumes_at_its_generation_header_and_matches_one_shot() {
        let mapped = mapped_or_skip!();
        let Some(mut session) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let config = greedy(6);
        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        let t1 = session
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");
        let t2 = session
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 2");
        assert!(t1.used_gpu && t2.used_gpu, "no fallback");
        assert_eq!(t2.reused, header_at(&mapped, &p1));
        assert!(t2.reused > 0);
        assert_eq!(t2.tokens, t1.tokens);
        assert_eq!(t2.tokens, one_shot_gpu(&mapped, &p1, &config));
    }

    /// A second save on the device writes into the first save's buffers
    /// rather than allocating a fresh copy per turn (#4214). Allocations are
    /// counted, not addresses compared: freeing the old copy first and
    /// allocating anew hands back the same addresses.
    #[test]
    fn gpu_a_second_checkpoint_allocates_nothing_on_the_device() {
        use crate::session::ArchForward;
        use trueno_gpu::driver::device_allocs_total;
        let mapped = mapped_or_skip!();
        if !crate::cuda::CudaExecutor::is_available() {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        let mut f = Qwen35Forward::load(&mapped, false).expect("load");
        assert!(
            f.on_gpu(),
            "a CUDA build on a CUDA host serves from the GPU"
        );
        let prompt = encode(&mapped, &user_turn("Name the capital of Peru."));
        f.reserve(prompt.len() + 1).expect("reserve");
        f.forward(&prompt[..prompt.len() - 1], 0).expect("prefill");
        let before = device_allocs_total();
        f.save_checkpoint().expect("save 1");
        assert!(
            device_allocs_total() > before,
            "the first save allocates the copy"
        );
        f.forward(&prompt[prompt.len() - 1..], prompt.len() - 1)
            .expect("one more");
        let before = device_allocs_total();
        f.save_checkpoint().expect("save 2");
        assert_eq!(
            device_allocs_total(),
            before,
            "the second save reused the first one's buffers"
        );
        assert!(f.on_gpu(), "no fallback");
        assert!(matches!(f.checkpoint, Some(Checkpoint::Gpu(_))));
    }

    #[test]
    fn gpu_a_re_rendered_turn_two_resumes_at_turn_ones_header_and_matches_one_shot() {
        let mapped = mapped_or_skip!();
        let Some(mut session) = gpu_session_or_skip(&mapped) else {
            return;
        };
        let config = greedy(6);
        let p1 = encode(&mapped, &user_turn("Name the capital of Peru."));
        session
            .generate(&p1, &config, &mut |_| true)
            .expect("turn 1");
        let p2 = re_rendered_turn_two(&mapped, &p1);
        let t2 = session
            .generate(&p2, &config, &mut |_| true)
            .expect("turn 2");
        assert!(t2.used_gpu, "no fallback");
        assert_eq!(t2.reused, header_at(&mapped, &p1));
        assert_eq!(t2.tokens, one_shot_gpu(&mapped, &p2, &config));
    }
}

// #4255: the two #4247 branches no GPU test reaches — the prefill that does not
// fit, and the prefill that fails. Pure, so they run in CPU `workspace-test`.

const MIB: u64 = 1 << 20;

fn plan(
    free: u64,
    need: impl FnMut(&'static str, usize) -> u64,
) -> std::result::Result<(&'static str, usize), String> {
    choose_prefill_plan(free, &["cublas", "flash"], &[512, 128], need, |a| a)
}

#[test]
fn prefill_plan_that_fits_nowhere_is_refused_naming_every_candidate() {
    let why = plan(100 * MIB, |_, rows| rows as u64 * MIB).expect_err("512 and 128 MiB > 100 MiB");
    assert_eq!(
        why,
        "cublas at 512 rows needs 512 MiB, cublas at 128 rows needs 128 MiB, \
         flash at 512 rows needs 512 MiB, flash at 128 rows needs 128 MiB; 100 MiB free"
    );
}

#[test]
fn prefill_plan_takes_the_first_fit_attention_major() {
    // Every candidate fits: the first one is apr run's choice.
    assert_eq!(plan(u64::MAX, |_, _| MIB), Ok(("cublas", 512)));
    // Only the smaller chunk fits under cuBLAS: fewer rows beat a different attention.
    assert_eq!(
        plan(200 * MIB, |_, rows| rows as u64 * MIB),
        Ok(("cublas", 128))
    );
    // cuBLAS fits nowhere; flash at the big chunk does.
    let need = |a: &str, rows: usize| if a == "flash" { rows as u64 } else { u64::MAX };
    assert_eq!(plan(1024, need), Ok(("flash", 512)));
}

#[test]
fn prefill_plan_boundary_need_equal_to_free_fits() {
    assert_eq!(plan(512 * MIB, |_, _| 512 * MIB), Ok(("cublas", 512)));
    assert!(plan(512 * MIB - 1, |_, _| 512 * MIB).is_err());
}

#[test]
fn prefill_plan_with_no_candidates_is_refused() {
    let why = choose_prefill_plan::<&str>(u64::MAX, &[], &[512], |_, _| 0, |a| a)
        .expect_err("no attention to try");
    assert!(why.ends_with("MiB free"), "{why}");
}

#[test]
fn failed_batched_prefill_is_a_gpu_step_so_the_session_moves_to_the_cpu() {
    match batched_prefill_outcome::<&str>(Err("CUDA_ERROR_OUT_OF_MEMORY"), 851, 17) {
        Err(Step::Gpu(why)) => assert_eq!(
            why,
            "the GPU batched prefill of 851 tokens at position 17 failed: CUDA_ERROR_OUT_OF_MEMORY"
        ),
        Err(Step::Fatal(e)) => {
            panic!("a prefill failure must fall back to the CPU, not end the turn: {e}")
        },
        Ok(logits) => panic!("a failed prefill returned {} logits", logits.len()),
    }
}

#[test]
fn successful_batched_prefill_returns_its_logits_unchanged() {
    match batched_prefill_outcome::<&str>(Ok(vec![0.5, -1.0, 2.0]), 3, 0) {
        Ok(logits) => assert_eq!(logits, [0.5, -1.0, 2.0]),
        Err(Step::Gpu(why)) => panic!("an Ok prefill became a GPU failure: {why}"),
        Err(Step::Fatal(e)) => panic!("an Ok prefill became a fatal error: {e}"),
    }
}

/// #4274: a turn past MIN_CAPACITY must leave headroom for the next turn,
/// or that turn reallocates and re-prefills from position 0. Rows are
/// (positions, current, min, context) -> capacity.
#[test]
fn a_state_that_outgrows_min_capacity_leaves_headroom_for_the_next_turn() {
    let cases = [
        // the #4274 repro: turn 1 at 4197 of an 8192 context
        ((4197, 0, MIN_CAPACITY, 8192), 8192),
        ((850, 0, MIN_CAPACITY, 8192), MIN_CAPACITY),
        ((MIN_CAPACITY, 0, MIN_CAPACITY, 8192), MIN_CAPACITY),
        ((5000, MIN_CAPACITY, MIN_CAPACITY, 32768), 8192),
        ((9000, 8192, MIN_CAPACITY, 32768), 16384),
        ((20000, 0, MIN_CAPACITY, 32768), 32768),
        // clamped to the context, never below the turn
        ((4197, 0, MIN_CAPACITY, 6000), 6000),
        ((100, 0, MIN_CAPACITY, 64), 100),
        // a one-call state is sized to the call
        ((4197, 0, 0, 8192), 4197),
    ];
    for ((positions, current, min, context), want) in cases {
        assert_eq!(
            grown_capacity(positions, current, min, context),
            want,
            "grown_capacity({positions}, {current}, {min}, {context})"
        );
    }
    // The e2e shape: turn 2 (4279 positions) fits what turn 1 allocated.
    let first = grown_capacity(4197, 0, MIN_CAPACITY, 8192);
    assert!(4279 <= first, "turn 2 would reallocate from {first}");
}

// ---- checkpoint save/restore on the CPU backend, with no GGUF file ----------
//
// The tests above skip when the real hybrid file is absent (CI), so the CPU
// checkpoint arms were unexercised there. This builds the smallest host model
// in memory (no layers) and drives `save_checkpoint` / `restore_checkpoint`
// directly on its `Backend::Cpu` state.

fn empty_tensor() -> crate::gguf::OwnedQuantizedTensor {
    crate::gguf::OwnedQuantizedTensor {
        data: Vec::new(),
        in_dim: 0,
        out_dim: 0,
        qtype: crate::gguf::types::GGUF_TYPE_F32,
    }
}

fn synthetic_forward() -> Qwen35Forward {
    let config = crate::gguf::GGUFConfig {
        architecture: "qwen35".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("qwen35"),
        hidden_dim: 4,
        num_layers: 0,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 1,
        intermediate_dim: 4,
        context_length: 32,
        rope_theta: 10_000.0,
        eps: 1.0e-6,
        rope_type: 2,
        explicit_head_dim: Some(4),
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let base = OwnedQuantizedModel {
        config,
        token_embedding: vec![0.0; 4],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None,
        lm_head_weight: empty_tensor(),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };
    let base: &'static OwnedQuantizedModel = Box::leak(Box::new(base));
    let qwen: &'static Qwen35Model<'static> = Box::leak(Box::new(Qwen35Model {
        base,
        layers: vec![],
        head_dim: 4,
        num_kv_heads: 1,
        num_v_heads: 1,
        head_v_dim: 1,
        num_k_heads: 1,
        head_k_dim: 1,
        conv_kernel: 4,
        rope_sections: [1, 1, 0, 0],
    }));
    let mut state = qwen.new_state(8);
    state.conv_states = vec![vec![0.0; 3]];
    state.ssm_states = vec![vec![0.0; 2]];
    Qwen35Forward {
        qwen,
        backend: Backend::Cpu(Some(state)),
        capacity: 8,
        turn_positions: 0,
        allocations: 1,
        min_capacity: 0,
        context_length: 32,
        notices: Vec::new(),
        per_token_prefill: false,
        batched_prefills: 0,
        checkpoint: None,
        im_start: None,
    }
}

fn cpu_state(f: &mut Qwen35Forward) -> &mut Qwen35State {
    match &mut f.backend {
        Backend::Cpu(Some(state)) => state,
        _ => panic!("the synthetic forward runs on the CPU with a state"),
    }
}

#[test]
fn a_cpu_checkpoint_is_saved_and_restored_into_the_state() {
    use crate::session::ArchForward;
    let mut f = synthetic_forward();
    assert!(!f.on_gpu(), "the synthetic session is on the CPU");
    assert!(
        !f.restore_checkpoint().expect("restore with nothing saved"),
        "nothing saved: nothing to restore"
    );

    cpu_state(&mut f).conv_states[0][1] = 3.0;
    cpu_state(&mut f).ssm_states[0][0] = 5.0;
    cpu_state(&mut f).kv_cache.advance();
    f.save_checkpoint().expect("save");
    assert!(
        matches!(f.checkpoint, Some(Checkpoint::Cpu(_))),
        "saving on the CPU keeps a CPU checkpoint"
    );

    // Move the state on, then return to the checkpoint.
    cpu_state(&mut f).conv_states[0][1] = 9.0;
    cpu_state(&mut f).ssm_states[0][0] = 9.0;
    cpu_state(&mut f).kv_cache.advance();
    cpu_state(&mut f).kv_cache.advance();
    assert!(
        f.restore_checkpoint().expect("restore"),
        "a saved CPU checkpoint restores"
    );
    let state = cpu_state(&mut f);
    assert_eq!(state.conv_states[0][1], 3.0, "conv window restored");
    assert_eq!(state.ssm_states[0][0], 5.0, "recurrent state restored");
    assert_eq!(state.kv_cache.len(), 1, "KV rows truncated to the save");
}

/// A failed restore: on the device it is a GPU failure the session survives on the CPU, loudly, prefilling
/// from 0 (`Ok(false)`, checkpoint dropped); on the CPU the error is the caller's. Both halves run here on the
/// CPU build, since a device restore cannot fail without a device.
#[test]
fn a_failed_restore_falls_back_on_the_device_and_propagates_on_the_cpu() {
    use crate::session::ArchForward;
    let refused = || RealizarError::UnsupportedOperation {
        operation: "restore".to_string(),
        reason: "planted".to_string(),
    };

    let mut f = synthetic_forward();
    f.save_checkpoint().expect("save");
    let err = f
        .after_failed_restore(false, refused())
        .expect_err("a CPU restore failure is the caller's error");
    assert!(err.to_string().contains("planted"), "{err}");
    assert!(f.checkpoint.is_some(), "propagating keeps the checkpoint");
    assert!(f.notices.is_empty(), "no fallback was announced");

    let mut f = synthetic_forward();
    f.save_checkpoint().expect("save");
    assert!(
        !f.after_failed_restore(true, refused())
            .expect("a device restore failure falls back"),
        "after the fallback the turn prefills from 0"
    );
    assert!(f.checkpoint.is_none(), "the device checkpoint is dropped");
    assert!(!f.on_gpu());
    assert_eq!(f.notices.len(), 1, "{:?}", f.notices);
    assert!(
        f.notices[0].contains("the device checkpoint would not restore: ")
            && f.notices[0].contains("planted"),
        "{:?}",
        f.notices
    );
}
