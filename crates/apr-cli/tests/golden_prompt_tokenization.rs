//! Does the APR file's EMBEDDED BPE tokenizer encode ChatML control tokens as
//! single special-token IDs, or as their literal characters?
//!
//! WHY THIS EXISTS (#2350). `apr qa`'s Golden Output gate fails on GPU and passes
//! on CPU for `qwen2.5-coder-1.5b-instruct-q4k.apr`, producing
//! "I'm sorry, but I'm not sure what you're asking" for the prompt
//! `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n`.
//!
//! Everything else was ruled out by measurement: GGUF+GPU passes, prompt length
//! is not the trigger, sampling config is identical to `apr run`'s defaults
//! (temperature 0.0 / top_k 1), and the model file has been unchanged since
//! 2026-03-08. The one path `apr run` cannot exercise is the gate's:
//! `golden_output_apr` (output_verification.rs:505) calls
//! `load_embedded_bpe_tokenizer().encode(prompt)` and passes the result through
//! `with_input_tokens`, deliberately bypassing `prepare_tokens`' ChatML
//! auto-wrap.
//!
//! If that encode emits `<`, `|`, `im`, `_start`, `|`, `>` instead of the single
//! id 151644, then the gate has been asserting on a prompt it never intended,
//! and the model is being asked to continue malformed text. That would be a
//! defect in the gate, not only in the GPU path — and it is a one-assert
//! question, so it should not be guessed at.
//!
//! Qwen2.5 control ids: <|endoftext|> 151643, <|im_start|> 151644, <|im_end|> 151645.

#![cfg(feature = "inference")]

use std::path::PathBuf;

const IM_START: u32 = 151_644;
const IM_END: u32 = 151_645;

/// The exact prompt from `golden_test_cases()` case 2 (golden_output.rs:86).
const GOLDEN_PROMPT: &str = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?)
        .join("models/qwen2.5-coder-1.5b-instruct-q4k.apr");
    p.exists().then_some(p)
}

#[test]
#[ignore = "needs the 1.5B APR model on disk; run with --ignored"]
fn embedded_tokenizer_encodes_chatml_controls_as_single_ids() {
    use realizar::apr::AprV2Model;

    let Some(path) = model_path() else {
        eprintln!("SKIP: model not present");
        return;
    };

    let model = AprV2Model::load(&path).expect("load APR");
    let tokenizer = model
        .load_embedded_bpe_tokenizer()
        .expect("APR has an embedded BPE tokenizer");

    let ids = tokenizer.encode(GOLDEN_PROMPT);
    eprintln!("prompt : {GOLDEN_PROMPT:?}");
    eprintln!("n_tokens: {}", ids.len());
    eprintln!("ids     : {ids:?}");

    // Round-trip is the readable form of the same question.
    let decoded = tokenizer.decode(&ids);
    eprintln!("decoded : {decoded:?}");

    let n_start = ids.iter().filter(|&&t| t == IM_START).count();
    let n_end = ids.iter().filter(|&&t| t == IM_END).count();
    eprintln!("<|im_start|> ({IM_START}) x{n_start}, <|im_end|> ({IM_END}) x{n_end}");

    // The prompt contains <|im_start|> twice and <|im_end|> once.
    assert_eq!(
        n_start, 2,
        "expected <|im_start|> to encode as the single id {IM_START} twice; \
         got {n_start}. If this is 0 the embedded tokenizer is emitting the \
         LITERAL characters, so the golden gate has been feeding the model \
         malformed text (#2350). ids={ids:?}"
    );
    assert_eq!(
        n_end, 1,
        "expected <|im_end|> to encode as the single id {IM_END} once; got {n_end}. ids={ids:?}"
    );

    // A correctly-tokenised ChatML prompt of this length is ~10 tokens. Literal
    // character encoding would balloon it well past 20.
    assert!(
        ids.len() < 20,
        "prompt encoded to {} tokens, which is far more than ChatML with proper \
         control ids should need — strong evidence of literal-character encoding. ids={ids:?}",
        ids.len()
    );
}

/// ANSWER-FIRST DIAGNOSTIC, not an assertion.
///
/// Tokenisation is clean (test above), so the gate feeds a well-formed 9-token
/// prompt and GPU still diverges from CPU. The remaining difference between the
/// gate and `apr run` is the CONFIG ENTRY: the gate uses
/// `InferenceConfig::with_input_tokens(...)`, `apr run` uses the prompt path
/// which goes through `prepare_tokens`. This prints what each entry actually
/// produces so the divergence is observed rather than reasoned about.
///
/// Run with CUDA_VISIBLE_DEVICES="" to get the CPU column.
#[test]
#[ignore = "diagnostic; needs the 1.5B APR model. Run with --ignored --nocapture"]
fn compare_input_tokens_entry_vs_prompt_entry() {
    use realizar::apr::AprV2Model;
    use realizar::{run_inference, InferenceConfig};

    let Some(path) = model_path() else {
        eprintln!("SKIP: model not present");
        return;
    };

    let model = AprV2Model::load(&path).expect("load APR");
    let tokenizer = model
        .load_embedded_bpe_tokenizer()
        .expect("embedded tokenizer");
    let ids = tokenizer.encode(GOLDEN_PROMPT);

    let gpu = std::env::var("CUDA_VISIBLE_DEVICES").map_or(true, |v| !v.is_empty());
    eprintln!("=== device: {} ===", if gpu { "GPU" } else { "CPU" });

    // (a) EXACTLY what the golden gate does.
    let cfg_tokens = InferenceConfig::new(&path)
        .with_input_tokens(ids.clone())
        .with_max_tokens(24)
        .with_temperature(0.0)
        .with_top_k(1);
    match run_inference(&cfg_tokens) {
        Ok(r) => eprintln!("with_input_tokens -> {:?}\n  tokens={:?}", r.text, r.tokens),
        Err(e) => eprintln!("with_input_tokens -> ERROR {e}"),
    }

    // (b) The same text through the prompt entry (auto-wrap applies).
    let cfg_prompt = InferenceConfig::new(&path)
        .with_prompt("Hello")
        .with_max_tokens(24)
        .with_temperature(0.0)
        .with_top_k(1);
    match run_inference(&cfg_prompt) {
        Ok(r) => eprintln!(
            "with_prompt(\"Hello\") -> {:?}\n  tokens={:?}",
            r.text, r.tokens
        ),
        Err(e) => eprintln!("with_prompt -> ERROR {e}"),
    }
}

/// SEQUENCE dependence — the last untested difference.
///
/// A single `run_inference` call with the gate's exact tokens returns the RIGHT
/// answer on GPU (test above). `apr qa` calling the same thing returns the wrong
/// one. The difference is that `apr qa` runs golden case 1 ("What is 2+2?")
/// FIRST, in the same process, on the same GPU — and `apr qa`'s own
/// "GPU State Isolation" gate is SKIPPED for APR format ("Only GGUF format
/// supported"), so nothing checks for cross-inference contamination on this path.
///
/// This replays both cases in order, exactly as the gate does. If case 2 alone is
/// correct but case-2-after-case-1 is wrong, the defect is GPU state leaking
/// between inferences, not decode numerics.
#[test]
#[ignore = "diagnostic; needs the 1.5B APR model. Run with --ignored --nocapture"]
fn golden_cases_run_in_sequence_like_the_gate_does() {
    use realizar::apr::AprV2Model;
    use realizar::{run_inference, InferenceConfig};

    let Some(path) = model_path() else {
        eprintln!("SKIP: model not present");
        return;
    };
    let model = AprV2Model::load(&path).expect("load APR");
    let tok = model.load_embedded_bpe_tokenizer().expect("tokenizer");

    let cases = [
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4"),
        (GOLDEN_PROMPT, "Hello/Hi/hey"),
        ("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "greeting(shorter)"),
        ("<|im_start|>user\nHello there, how are you doing today my friend?<|im_end|>\n<|im_start|>assistant\n", "greeting(longer)"),
        ("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n", "Paris"),
    ];

    let gpu = std::env::var("CUDA_VISIBLE_DEVICES").map_or(true, |v| !v.is_empty());
    eprintln!(
        "=== device: {} — running BOTH cases in order ===",
        if gpu { "GPU" } else { "CPU" }
    );

    for (i, (prompt, want)) in cases.iter().enumerate() {
        let cfg = InferenceConfig::new(&path)
            .with_input_tokens(tok.encode(prompt))
            .with_max_tokens(512)
            .with_temperature(0.0)
            .with_top_k(1);
        match run_inference(&cfg) {
            Ok(r) => eprintln!("case {} (want {want}) -> {:?}", i + 1, r.text),
            Err(e) => eprintln!("case {} -> ERROR {e}", i + 1),
        }
    }
}
