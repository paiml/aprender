//! aprender#4804: a GGUF server must tokenize a prompt the way the model was trained,
//! which is what `GGUFModel::encode` and llama.cpp do.
//!
//! `AppState::with_quantized_model_and_vocab` built `BPETokenizer::new(vocab, [], unk)`:
//! greedy longest match with no merges and no special tokens. On Qwen, `?<` is a vocabulary
//! entry, so `?<|im_end|>\n` came out `?<` `|i` `m` `_end` `|` `>\n` and the model never saw
//! its end-of-turn token (+3 prompt tokens per turn against llama.cpp, #4662).
//!
//! The fixture is a small Qwen-shaped byte-level vocabulary: the 256 byte glyphs, a `?<`
//! trap, two control tokens and a few merges. The expected ids come from the canonical
//! encoder itself (`ByteLevelBpe::from_gguf`), so the test pins the route to it.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};
use crate::gguf::byte_level_bpe::ByteLevelBpe;
use crate::gguf::GGUFValue;

const PROMPT: &str = "<|im_start|>user\nhi?<|im_end|>\n<|im_start|>assistant\n";

/// GPT-2 `bytes_to_unicode`: printable Latin-1 maps to itself, every other byte to
/// U+0100 onwards, in byte order.
fn byte_glyphs() -> Vec<String> {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    let mut next = 0x100u32;
    (0u32..256)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(c).expect("glyph").to_string()
        })
        .collect()
}

fn fixture() -> (Vec<String>, HashMap<String, GGUFValue>) {
    let mut vocab = byte_glyphs();
    let extra = [
        "?<", "<|im_start|>", "<|im_end|>", "us", "use", "user", "hi", "as", "ass",
    ];
    vocab.extend(extra.iter().map(|s| (*s).to_string()));
    let types: Vec<GGUFValue> = vocab
        .iter()
        .map(|t| GGUFValue::Int32(if t.starts_with("<|im_") { 3 } else { 1 }))
        .collect();
    let merges = ["u s", "us e", "use r", "h i", "a s", "as s"];
    let mut md = HashMap::new();
    md.insert(
        "tokenizer.ggml.model".to_string(),
        GGUFValue::String("gpt2".to_string()),
    );
    md.insert(
        "tokenizer.ggml.pre".to_string(),
        GGUFValue::String("qwen2".to_string()),
    );
    md.insert(
        "tokenizer.ggml.merges".to_string(),
        GGUFValue::Array(
            merges
                .iter()
                .map(|m| GGUFValue::String((*m).to_string()))
                .collect(),
        ),
    );
    md.insert("tokenizer.ggml.token_type".to_string(), GGUFValue::Array(types));
    (vocab, md)
}

fn id_of(vocab: &[String], t: &str) -> u32 {
    vocab.iter().position(|v| v == t).expect("in vocab") as u32
}

fn state(vocab: Vec<String>, bpe: Option<Arc<ByteLevelBpe>>) -> AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let config = GGUFConfig {
        architecture: "qwen2".to_string(),
        constraints: ArchConstraints::from_architecture("qwen2"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: vocab.len(),
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let model = create_test_quantized_model(&config);
    AppState::with_quantized_model_and_vocab(model, vocab)
        .expect("build quantized AppState")
        .with_byte_level_bpe(bpe)
}

async fn tokenize(state: AppState, text: &str) -> Vec<u32> {
    let body = serde_json::json!({ "text": text });
    let resp = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    json["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

#[tokio::test]
async fn gguf_tokenize_route_matches_the_canonical_encoder() {
    let (vocab, md) = fixture();
    let bpe = ByteLevelBpe::from_gguf(&md, &vocab).expect("qwen2 byte-level BPE");
    let want = bpe.encode(PROMPT);
    // The canonical encoding itself, so a broken fixture cannot pass vacuously.
    assert_eq!(want.iter().filter(|&&t| t == id_of(&vocab, "<|im_end|>")).count(), 1);
    assert!(want.contains(&id_of(&vocab, "user")), "merges applied");
    assert!(!want.contains(&id_of(&vocab, "?<")), "the special claims its `<`");

    let got = tokenize(state(vocab.clone(), Some(bpe)), PROMPT).await;
    assert_eq!(got, want, "serve /tokenize must equal GGUFModel::encode (#4804)");
}

#[tokio::test]
async fn without_a_canonical_encoder_the_route_keeps_its_tokenizer() {
    // Positive control: the greedy tokenizer really does split `?<|im_end|>`, so the test
    // above would fail without the encoder wired in.
    let (vocab, _) = fixture();
    let im_end = id_of(&vocab, "<|im_end|>");
    let got = tokenize(state(vocab.clone(), None), "hi?<|im_end|>\n").await;
    assert!(got.contains(&id_of(&vocab, "?<")), "greedy takes `?<`: {got:?}");
    assert!(!got.contains(&im_end), "greedy loses the end-of-turn token: {got:?}");
}
