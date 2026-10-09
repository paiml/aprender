//! #3793: which loop a constrained run takes, what it reports, and when it refuses.

use super::*;
use crate::constrain::{ConstraintError, ConstraintRequest};
use crate::session::ConstrainedStop;

fn refusal_kind(e: RealizarError) -> ConstraintError {
    match e {
        RealizarError::Constraint(c) => c,
        other => panic!("not a constraint refusal: {other}"),
    }
}

/// apr's default Qwen3 template prefills a CLOSED think block: no reasoning comes first, so a
/// constraint may apply from the first token. An OPEN block means reasoning comes first.
#[test]
fn only_a_prompt_that_leaves_think_open_is_a_thinking_prompt() {
    let no_think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let open = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let plain = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    assert!(!prompt_opens_thinking(no_think));
    assert!(prompt_opens_thinking(open));
    assert!(!prompt_opens_thinking(plain));
    // An earlier turn's closed block does not close a new open one
    assert!(prompt_opens_thinking(&format!(
        "{no_think}ok<|im_end|>\n{open}"
    )));
}

#[test]
fn a_constrained_finish_is_reported_as_itself() {
    use run_report::FinishReason;
    assert_eq!(
        gguf_finish_reason(Some(ConstrainedStop::Complete), &[1, 2], &[9], 64),
        FinishReason::ConstraintComplete
    );
    assert_eq!(
        gguf_finish_reason(Some(ConstrainedStop::Length), &[1, 2], &[9], 2),
        FinishReason::Length
    );
    // Unconstrained runs keep #3718's rule
    assert_eq!(
        gguf_finish_reason(None, &[1, 9], &[9], 64),
        FinishReason::Stop
    );
    assert_eq!(
        gguf_finish_reason(None, &[1, 2], &[9], 2),
        FinishReason::Length
    );
}

#[test]
fn a_format_that_applies_no_constraint_refuses_a_constrained_run_only() {
    let plain = InferenceConfig::new("m.apr");
    assert!(refuse_constraint_on(&plain, "apr", "x").is_ok());
    let constrained = InferenceConfig::new("m.apr")
        .with_constraint(ConstraintRequest::Lark(r#"start: "a""#.to_string()));
    match refusal_kind(
        refuse_constraint_on(&constrained, "apr", "not scheduled").expect_err("refused"),
    ) {
        ConstraintError::UnsupportedPath { path, removed_by } => {
            assert_eq!(path, "apr");
            assert_eq!(removed_by, "not scheduled");
        },
        other => panic!("{other}"),
    }
}

/// The mock backend refuses too: a constraint is never silently dropped on any path.
#[test]
fn the_mock_backend_refuses_a_constrained_run() {
    let mut config = InferenceConfig::new("m.gguf")
        .with_prompt("hi")
        .with_constraint(ConstraintRequest::Lark(r#"start: "a""#.to_string()));
    config.use_mock_backend = true;
    let e = run_inference_report(&config).expect_err("refused");
    assert!(
        matches!(refusal_kind(e), ConstraintError::UnsupportedPath { path, .. } if path == "mock")
    );
}

fn pygmy_model() -> (tempfile::NamedTempFile, crate::gguf::OwnedQuantizedModel) {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
    f.write_all(&build_executable_pygmy_gguf()).expect("write");
    let mapped = crate::gguf::MappedGGUFModel::from_path(f.path()).expect("map");
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    (f, model)
}

fn path_of(
    config: &InferenceConfig,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    model: &crate::gguf::OwnedQuantizedModel,
    is_qwen35: bool,
) -> Option<&'static str> {
    unconstrained_gguf_path(config, gen_config, model, is_qwen35).map(|(path, _)| path)
}

#[test]
fn the_moe_loop_refuses_and_no_gpu_takes_the_engine() {
    let (_f, mut model) = pygmy_model();
    let cpu = InferenceConfig::new("m.gguf").without_gpu();
    let greedy = crate::gguf::QuantizedGenerateConfig::default();
    assert_eq!(path_of(&cpu, &greedy, &model, false), None);
    assert_eq!(path_of(&cpu, &greedy, &model, true), None);
    // The MoE loop refuses even with --no-gpu: it applies no constraint on any device
    model.config.architecture = "qwen3moe".to_string();
    assert_eq!(path_of(&cpu, &greedy, &model, false), Some("qwen3-moe"));
}

#[test]
fn trace_refuses_on_the_dense_loop_only() {
    // #4268: `--trace` runs the instrumented dense loop; the hybrid's engine ignores it
    let (_f, model) = pygmy_model();
    let cpu = InferenceConfig::new("m.gguf").without_gpu();
    let traced = crate::gguf::QuantizedGenerateConfig {
        trace: true,
        ..Default::default()
    };
    assert_eq!(path_of(&cpu, &traced, &model, false), Some("gguf-trace"));
    assert_eq!(path_of(&cpu, &traced, &model, true), None);
}

#[test]
fn the_engine_runs_wherever_the_dispatch_takes_it_and_only_other_loops_refuse() {
    let (_f, model) = pygmy_model();
    let greedy = crate::gguf::QuantizedGenerateConfig::default();
    let sampled = crate::gguf::QuantizedGenerateConfig {
        temperature: 0.8,
        top_k: 40,
        ..Default::default()
    };
    let traced = crate::gguf::QuantizedGenerateConfig {
        trace: true,
        ..Default::default()
    };
    let bare = InferenceConfig::new("m.gguf");
    let forced = InferenceConfig::new("m.gguf").with_accel_forced(true);
    // #3568 PR 3: the engine applies the constraint on the device as on the CPU, so a bare run
    // takes it on every build: CUDA on a cuda build, the CPU otherwise (#3757: never wgpu)
    assert_eq!(path_of(&bare, &greedy, &model, false), None);
    assert_eq!(path_of(&bare, &greedy, &model, true), None);
    // The hybrid has no wgpu forward and no instrumented loop
    assert_eq!(path_of(&forced, &greedy, &model, true), None);
    assert_eq!(path_of(&bare, &traced, &model, true), None);
    assert_eq!(path_of(&forced, &sampled, &model, false), None);
    if cfg!(feature = "cuda") {
        // CUDA is entered before wgpu, so `--gpu` takes the engine on the device
        assert_eq!(path_of(&forced, &greedy, &model, false), None);
        // `--trace` on the device keeps the instrumented device loop
        assert_eq!(
            path_of(&bare, &traced, &model, false),
            Some("gguf-cuda-trace")
        );
    } else {
        // `--gpu` enters wgpu for a greedy request (#3757) and never for a sampled one (#3760)
        let wgpu = cfg!(feature = "gpu").then_some("gguf-wgpu");
        assert_eq!(path_of(&forced, &greedy, &model, false), wgpu);
        assert_eq!(path_of(&bare, &traced, &model, false), Some("gguf-trace"));
    }
}

/// #3826: the constrained run reports `gpu_attempted` from the dispatch's own facts: a cuda
/// build, no `--no-gpu`, no legacy quant. Never from a device probe.
#[test]
fn the_dense_cuda_attempt_is_the_build_and_the_request() {
    let bare = InferenceConfig::new("m.gguf");
    let cpu = InferenceConfig::new("m.gguf").without_gpu();
    assert_eq!(dense_cuda_attempted(&bare, false), cfg!(feature = "cuda"));
    assert!(!dense_cuda_attempted(&cpu, false));
    // Legacy quant never enters CUDA, in either run
    assert!(!dense_cuda_attempted(&bare, true));
    // The hybrid's attempt (no legacy-quant rule) is the same build-and-request fact
    assert_eq!(session_cuda_attempted(&bare), cfg!(feature = "cuda"));
    assert!(!session_cuda_attempted(&cpu));
}

/// A Q4_K model, which the dense dispatch sends to CUDA on a cuda build (the pygmy model is
/// Q4_0, legacy, and never enters it), with a tokenizer the constraint compiles against.
#[cfg(feature = "structured-output")]
fn q4k_model_with_vocab() -> (tempfile::NamedTempFile, crate::gguf::MappedGGUFModel) {
    use crate::gguf::test_factory::build_minimal_llama_gguf_with;
    use std::io::Write;
    let vocab: Vec<String> = ["<unk>", "</s>", "a", "b"]
        .into_iter()
        .map(String::from)
        .chain((4..32).map(|i| format!("x{i}")))
        .collect();
    let vocab: Vec<&str> = vocab.iter().map(String::as_str).collect();
    // intermediate = hidden: the builder sizes each Q4_K tensor from its first dim as rows, so
    // a non-square FFN tensor gets half the bytes its out dim needs and the forward refuses it.
    let bytes = build_minimal_llama_gguf_with(32, 64, 64, 4, 4, |b| {
        b.add_string("tokenizer.ggml.model", "llama")
            .add_string_array("tokenizer.ggml.tokens", &vocab)
            .add_u32("tokenizer.ggml.eos_token_id", 1)
    });
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
    f.write_all(&bytes).expect("write");
    let mapped = crate::gguf::MappedGGUFModel::from_path(f.path()).expect("map");
    (f, mapped)
}

/// #3826, end to end through the function the report reads: a dense constrained run returns
/// the turn's `used_gpu` beside the dispatch's attempt, whatever the device did. On a cuda
/// build a device that refuses the model (no driver, a failed upload, an F2 mismatch) leaves
/// the CPU engine serving the turn and the pair (false, true), which the report reads as
/// fell_back; a run that reports the attempt only when the device served it hides that.
#[cfg(feature = "structured-output")]
#[test]
fn a_dense_constrained_run_reports_the_attempt_the_dispatch_made() {
    let (_f, mapped) = q4k_model_with_vocab();
    let request = ConstraintRequest::Lark(r#"start: "a" | "b""#.to_string());
    let gen_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![1],
        ..Default::default()
    };
    for (config, attempted) in [
        (InferenceConfig::new("m.gguf"), cfg!(feature = "cuda")),
        (InferenceConfig::new("m.gguf").without_gpu(), false),
    ] {
        let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("load");
        assert!(!model_has_legacy_quant(&model), "Q4_K is not legacy");
        let (tokens, used_gpu, gpu_attempted, stop) = generate_gguf_constrained(
            &request,
            &config,
            &mapped,
            Some(model),
            None,
            &[2],
            &gen_config,
        )
        .expect("a constrained turn");
        assert_eq!(gpu_attempted, attempted, "no_gpu={}", config.no_gpu);
        assert!(
            !used_gpu || gpu_attempted,
            "the device served an unattempted turn"
        );
        assert_eq!(stop, ConstrainedStop::Complete);
        assert!(matches!(tokens.get(1), Some(2 | 3)), "{tokens:?}");
    }
}
