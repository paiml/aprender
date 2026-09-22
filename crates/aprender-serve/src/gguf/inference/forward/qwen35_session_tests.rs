//! #3595: a resident session must produce exactly what a one-shot generate of
//! the same prompt produces — on the state it reused, on a state it reset in
//! place, on either backend. Every assertion here compares against the path
//! `apr run` takes; a session that drifts from it is serving a different model.

use super::*;
use crate::gguf::forward_qwen35::run_qwen35_generate;

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

/// The one-shot CPU path `apr run --no-gpu` takes.
fn one_shot_cpu(
    mapped: &MappedGGUFModel,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
) -> Vec<u32> {
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    run_qwen35_generate(mapped, &base, prompt, config).expect("one-shot generate")
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

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::gguf::forward_qwen35::run_qwen35_generate_dispatch;

    /// The one-shot GPU path `apr run --gpu` takes.
    fn one_shot_gpu(
        mapped: &MappedGGUFModel,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Vec<u32> {
        let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
        let (tokens, used_gpu) =
            run_qwen35_generate_dispatch(mapped, &base, prompt, config, false).expect("one-shot");
        assert!(used_gpu, "the one-shot reference must itself be a GPU run");
        tokens
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
}
