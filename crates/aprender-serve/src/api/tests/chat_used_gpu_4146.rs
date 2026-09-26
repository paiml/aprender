//! aprender#4146 / E5 #4000: every non-streamed `/v1/chat/completions` reply names the
//! device that served it (`used_gpu`), as `/v1/completions` does.
//!
//! Before: only the MoE arm passed `Some(..)` to `build_chat_response`; the other eight
//! chat backends passed `None`, so the key was absent from their replies.
//!
//! Two halves. The runtime rows serve a chat on each CPU backend and read the key off
//! the reply. The source row covers every `build_chat_response` call site, the CUDA
//! arms included, which need a GPU and a real model to serve and are not run here.

use axum::http::StatusCode;

use super::native_routes_2376::post;
use crate::api::AppState;

const CHAT: &str =
    r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":2}"#;

/// POST one chat turn and return the reply's `used_gpu`, requiring a 200.
async fn used_gpu(state: AppState, backend: &str) -> serde_json::Value {
    let (status, body) = post(state, "/v1/chat/completions", CHAT).await;
    assert_eq!(status, StatusCode::OK, "{backend}: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("JSON reply");
    json["used_gpu"].clone()
}

#[cfg(feature = "gpu")]
fn llama_config() -> crate::gguf::GGUFConfig {
    crate::gguf::GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

/// `registry_fallback`: the dense registry `Model` decodes on the CPU.
#[tokio::test]
async fn registry_chat_reports_used_gpu_false() {
    let state = AppState::demo().expect("demo AppState");
    assert_eq!(
        used_gpu(state, "registry").await,
        serde_json::Value::Bool(false)
    );
}

/// `try_apr_transformer_backend`: the f32 `AprTransformer` decodes on the CPU.
#[tokio::test]
async fn apr_transformer_chat_reports_used_gpu_false() {
    let state = super::apr_model_routes_2609::apr_transformer_state();
    assert_eq!(
        used_gpu(state, "apr_transformer").await,
        serde_json::Value::Bool(false)
    );
}

/// `try_quantized_backend`: `OwnedQuantizedModel` decodes on the CPU.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn quantized_chat_reports_used_gpu_false() {
    let model = crate::api::test_helpers::create_test_quantized_model(&llama_config());
    let state = AppState::with_quantized_model(model).expect("quantized AppState");
    assert_eq!(
        used_gpu(state, "quantized").await,
        serde_json::Value::Bool(false)
    );
}

/// `try_cached_backend`: the cached model delegates to the CPU model.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn cached_chat_reports_used_gpu_false() {
    let state = crate::api::test_helpers::create_test_cached_state();
    assert_eq!(
        used_gpu(state, "cached").await,
        serde_json::Value::Bool(false)
    );
}

/// `try_gpu_backend`: a `GpuModel` without a CUDA scheduler decodes every m=1 matmul
/// on the CPU (IMP-097), so the reply says false even when wgpu found an adapter.
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
#[tokio::test]
async fn hybrid_gpu_model_chat_reports_used_gpu_false() {
    use crate::gpu::{GpuModel, GpuModelConfig};
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };
    let state = AppState::with_gpu_model(GpuModel::new(config).expect("GpuModel"))
        .expect("GpuModel AppState");
    assert_eq!(
        used_gpu(state, "gpu_model").await,
        serde_json::Value::Bool(false)
    );
}

/// The top-level arguments of the call whose `(` is at `open`, with `//` comments
/// already stripped from `src`.
fn call_args(src: &str, open: usize) -> Vec<String> {
    let (mut depth, mut args, mut cur) = (0usize, Vec::new(), String::new());
    for c in src[open..].chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                if depth == 1 {
                    continue;
                }
            },
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    args.push(cur.trim().to_string());
                    return args.into_iter().filter(|a| !a.is_empty()).collect();
                }
            },
            ',' if depth == 1 => {
                args.push(std::mem::take(&mut cur).trim().to_string());
                continue;
            },
            _ => {},
        }
        cur.push(c);
    }
    panic!("unterminated call at byte {open}");
}

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `build_chat_response` call passes `Some(..)` as its last argument, `used_gpu`.
/// The CUDA arms are only reachable here: serving them needs a GPU and a real model.
#[test]
fn every_chat_response_builder_names_its_device() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api");
    let mut calls = 0;
    for file in [
        "cuda_chat_backend.rs",
        "openai_handlers.rs",
        "qwen35_chat_backend.rs",
    ] {
        let raw = std::fs::read_to_string(dir.join(file)).expect("read source");
        let src = strip_line_comments(&raw);
        for (at, _) in src.match_indices("build_chat_response(") {
            if src[..at].ends_with("fn ") {
                continue;
            }
            let args = call_args(&src, at + "build_chat_response".len());
            let last = args.last().expect("arguments");
            assert!(
                last.starts_with("Some("),
                "{file}: a build_chat_response call passes `{last}` for used_gpu (#4146)"
            );
            calls += 1;
        }
    }
    // The count guards the scan itself: a scan that matched nothing would pass.
    assert_eq!(calls, 9, "build_chat_response call sites found");

    // `try_safetensors_cuda_backend` writes its JSON inline instead.
    let raw = std::fs::read_to_string(dir.join("cuda_chat_backend.rs")).expect("read source");
    let body = raw
        .split("fn try_safetensors_cuda_backend(")
        .nth(1)
        .and_then(|s| s.split("\n}\n").next())
        .expect("try_safetensors_cuda_backend");
    assert!(
        body.contains(r#","used_gpu":true}}"#),
        "the safetensors CUDA chat body names its device (#4146)"
    );
}

#[test]
fn call_args_splits_only_top_level_commas() {
    let src = "f(a, g(b, c), [d, e], Some(x))";
    assert_eq!(call_args(src, 1), ["a", "g(b, c)", "[d, e]", "Some(x)"]);
    let src = "f(\n    a,\n    None,\n)";
    assert_eq!(call_args(src, 1), ["a", "None"]);
    assert_eq!(strip_line_comments("a, // b, c\nd"), "a, \nd");
}
