//! APR-EMBED-001 EG-3 (#4902): `apr serve` for an EmbeddingGemma 2 file.
//!
//! Tests cover:
//! - Every embedding route answers with the vector `apr embed` computes for the same text
//! - `dimensions` is `--dim`: a Matryoshka width truncates, any other width is a 400
//! - Every generative route on an embedding model is a 400 that names the kind
//! - I-7: media is refused, never dropped. Three different images (and an audio and a video
//!   part) to the embedding routes; any 200 is RED, on an embedding and a generative server
//! - A generative model's `dimensions` must be its hidden size
//! - The embedding state refuses a generative file by kind

use super::native_routes_2376::post;
use crate::api::AppState;
use crate::gguf::embedding_gemma2_input::{EmbedTokenizer, Prompt};
use crate::gguf::forward_embedding_gemma2::{embed_text, tests::served_fixture, EmbeddingGemma2};
use crate::gguf::MappedGGUFModel;
use axum::http::StatusCode;
use std::io::Write;

/// The served width: the real file's 768, so every Matryoshka width truncates.
const WIDTH: usize = 768;

fn mapped(bytes: &[u8]) -> MappedGGUFModel {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(bytes).expect("write the fixture");
    file.flush().expect("flush the fixture");
    MappedGGUFModel::from_path(file.path()).expect("the fixture maps")
}

fn embedding_state() -> AppState {
    AppState::with_embedding_gemma2(&mapped(&served_fixture(WIDTH)))
        .expect("the embedding state builds")
}

/// What `apr embed <file> --text <text> [--dim <dim>]` computes: its own load of the file,
/// then the call its GGUF arm makes with no `--task`.
fn cli_vector(text: &str, dim: Option<usize>) -> Vec<f32> {
    let mapped = mapped(&served_fixture(WIDTH));
    let encoder = EmbeddingGemma2::from_mapped(&mapped).expect("the encoder loads");
    let tokenizer = EmbedTokenizer::from_gguf(&mapped.model).expect("the tokenizer loads");
    embed_text(&encoder, &tokenizer, &Prompt::Raw, text, dim)
        .expect("the CLI path embeds")
        .1
}

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"))
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    serde_json::from_value(value.clone()).expect("an array of numbers")
}

/// The first vector of an OpenAI-shaped embeddings response.
fn first_vector(body: &str) -> Vec<f32> {
    floats(&json(body)["data"][0]["embedding"])
}

// ============================================================================
// The embedding routes
// ============================================================================

#[tokio::test]
async fn every_embedding_route_returns_the_cli_vector_eg3() {
    let state = embedding_state();
    assert!(state.model_loaded(), "an embedding model is a loaded model");
    for text in ["ab ab", "a b", "ba"] {
        let want = cli_vector(text, None);
        assert_eq!(want.len(), WIDTH);
        let input = serde_json::json!({ "input": text }).to_string();
        for route in ["/v1/embeddings", "/realize/embed"] {
            let (status, body) = post(state.clone(), route, &input).await;
            assert_eq!(status, StatusCode::OK, "{route}: {body}");
            assert_eq!(first_vector(&body), want, "{route} {text:?}");
        }
        let prompt = serde_json::json!({ "prompt": text }).to_string();
        let (status, body) = post(state.clone(), "/api/embeddings", &prompt).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(floats(&json(&body)["embedding"]), want, "{text:?}");
    }
    // The vector depends on the text: the routes are not answering a constant.
    assert_ne!(cli_vector("ab ab", None), cli_vector("ba", None));
}

#[tokio::test]
async fn a_batch_returns_one_cli_vector_per_input_in_order_eg3() {
    let (status, body) = post(
        embedding_state(),
        "/v1/embeddings",
        r#"{"input": ["ba", "ab ab"]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = &json(&body)["data"];
    assert_eq!(floats(&data[0]["embedding"]), cli_vector("ba", None));
    assert_eq!(floats(&data[1]["embedding"]), cli_vector("ab ab", None));
}

#[tokio::test]
async fn dimensions_is_the_cli_dim_eg3() {
    let state = embedding_state();
    for dim in [768, 512, 256, 128] {
        let input = serde_json::json!({ "input": "ab ab", "dimensions": dim }).to_string();
        let (status, body) = post(state.clone(), "/v1/embeddings", &input).await;
        assert_eq!(status, StatusCode::OK, "dimensions {dim}: {body}");
        let got = first_vector(&body);
        assert_eq!(got.len(), dim);
        assert_eq!(got, cli_vector("ab ab", Some(dim)), "dimensions {dim}");
    }
    for dim in [0, 6, 300, 1024] {
        let input = serde_json::json!({ "input": "ab ab", "dimensions": dim }).to_string();
        let (status, body) = post(state.clone(), "/v1/embeddings", &input).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "dimensions {dim}: {body}");
        assert!(body.contains(&format!("dimensions {dim}")), "{body}");
    }
}

#[tokio::test]
async fn every_generative_route_names_the_embedding_kind_eg3() {
    let state = embedding_state();
    // `/v1/batch/completions` is absent: it is mounted only with a wgpu `cached_model`
    // (#3991), which an embedding state never has, so it is the 404 index there.
    for route in [
        "/generate",
        "/batch/generate",
        "/stream/generate",
        "/realize/generate",
        "/realize/batch",
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/chat/completions/stream",
        "/api/chat",
        "/api/generate",
    ] {
        let (status, body) = post(state.clone(), route, r#"{"prompt": "ab"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{route}: {body}");
        assert!(
            body.contains("is an embedding model") && body.contains("/v1/embeddings"),
            "{route}: {body}"
        );
    }
}

// ============================================================================
// I-7: media is refused, never dropped
// ============================================================================

/// Three different images, then an audio and a video part, as `input` values.
fn media_inputs() -> [(serde_json::Value, &'static str); 5] {
    [
        (
            serde_json::json!("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="),
            "image",
        ),
        (
            serde_json::json!([{ "type": "image_url", "image_url": { "url": "https://example.com/cat.jpg" } }]),
            "image",
        ),
        (
            serde_json::json!(["ab", "  DATA:IMAGE/GIF;base64,R0lGODlhAQABAAAAACw="]),
            "image",
        ),
        (
            serde_json::json!([{ "type": "input_audio", "input_audio": { "data": "UklGRg==", "format": "wav" } }]),
            "audio",
        ),
        (
            serde_json::json!({ "type": "video_url", "video_url": { "url": "https://example.com/a.mp4" } }),
            "video",
        ),
    ]
}

async fn assert_media_refused(state: &AppState, server: &str) {
    for (input, kind) in media_inputs() {
        for route in ["/v1/embeddings", "/realize/embed"] {
            let request = serde_json::json!({ "input": input }).to_string();
            let (status, body) = post(state.clone(), route, &request).await;
            assert_eq!(
                status,
                StatusCode::UNPROCESSABLE_ENTITY,
                "{server} {route} {input}: {body}"
            );
            assert!(body.contains(&format!("no {kind} encoder")), "{body}");
        }
    }
    for request in [
        r#"{"prompt": "ab", "images": ["iVBORw0KGgoAAAANSUhEUgAAAAE="]}"#,
        r#"{"prompt": "data:image/webp;base64,UklGRhYAAABXRUJQ"}"#,
    ] {
        let (status, body) = post(state.clone(), "/api/embeddings", request).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{server} {request}: {body}");
        assert!(body.contains("no image encoder"), "{body}");
    }
}

#[tokio::test]
async fn media_is_refused_on_an_embedding_server_eg3() {
    assert_media_refused(&embedding_state(), "embedding").await;
}

#[tokio::test]
async fn media_is_refused_on_a_generative_server_eg3() {
    let state = AppState::demo().expect("the demo state builds");
    assert_media_refused(&state, "generative").await;
}

#[tokio::test]
async fn an_empty_images_field_is_not_media_eg3() {
    let (status, body) = post(
        embedding_state(),
        "/api/embeddings",
        r#"{"prompt": "ab ab", "images": []}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(floats(&json(&body)["embedding"]), cli_vector("ab ab", None));
}

#[tokio::test]
async fn an_input_that_is_neither_text_nor_media_is_a_400_eg3() {
    let (status, body) = post(embedding_state(), "/v1/embeddings", r#"{"input": [1, 2, 3]}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("string or an array of strings"), "{body}");
}

// ============================================================================
// The generative kind
// ============================================================================

#[tokio::test]
async fn a_generative_model_answers_dimensions_only_at_its_hidden_size_eg3() {
    let state = AppState::demo().expect("the demo state builds");
    let (status, body) = post(state.clone(), "/v1/embeddings", r#"{"input": "hello"}"#).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hidden = first_vector(&body).len();

    let at_hidden = serde_json::json!({ "input": "hello", "dimensions": hidden }).to_string();
    let (status, body) = post(state.clone(), "/v1/embeddings", &at_hidden).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(first_vector(&body).len(), hidden);

    let narrower = serde_json::json!({ "input": "hello", "dimensions": hidden - 1 }).to_string();
    let (status, body) = post(state, "/v1/embeddings", &narrower).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("cannot be truncated"), "{body}");
}

#[test]
fn the_embedding_state_refuses_a_generative_file_by_kind_eg3() {
    let llama = crate::gguf::test_factory::build_minimal_llama_gguf(16, 8, 16, 2, 2);
    let Err(err) = AppState::with_embedding_gemma2(&mapped(&llama)) else {
        panic!("a generative file must be refused");
    };
    assert!(err.to_string().contains("ModelKind::Generative"), "{err}");
}
