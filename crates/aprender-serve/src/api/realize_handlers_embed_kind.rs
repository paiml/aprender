// APR-EMBED-001 EG-3 (#4902): what the embedding routes refuse, and how an
// EmbeddingGemma 2 file answers them.

/// The encoder a media input would need, read off a string: `"image"`, `"audio"` or
/// `"video"` for a `data:` URI of that type, `None` for text.
fn media_kind_of_str(s: &str) -> Option<&'static str> {
    const PREFIXES: [(&str, &str); 3] = [
        ("data:image/", "image"),
        ("data:audio/", "audio"),
        ("data:video/", "video"),
    ];
    let head = s.trim_start().as_bytes();
    PREFIXES.iter().find_map(|(prefix, kind)| {
        head.get(..prefix.len())
            .filter(|h| h.eq_ignore_ascii_case(prefix.as_bytes()))
            .map(|_| *kind)
    })
}

/// The encoder a structured input part would need: an OpenAI-style `image_url` or
/// `input_audio` part, a part whose `type` names a medium, or a `data:` URI anywhere
/// inside it. `None` when nothing in it is media.
fn media_kind_of_value(v: &serde_json::Value) -> Option<&'static str> {
    const KEYS: [(&str, &str); 9] = [
        ("image_url", "image"),
        ("image", "image"),
        ("images", "image"),
        ("input_image", "image"),
        ("audio", "audio"),
        ("audio_url", "audio"),
        ("input_audio", "audio"),
        ("video", "video"),
        ("video_url", "video"),
    ];
    match v {
        serde_json::Value::String(s) => media_kind_of_str(s),
        serde_json::Value::Array(items) => items.iter().find_map(media_kind_of_value),
        serde_json::Value::Object(map) => {
            if let Some((_, kind)) = KEYS.iter().find(|(key, _)| map.contains_key(*key)) {
                return Some(kind);
            }
            if let Some(serde_json::Value::String(t)) = map.get("type") {
                let t = t.to_ascii_lowercase();
                if let Some(kind) = ["image", "audio", "video"]
                    .into_iter()
                    .find(|k| t.contains(k))
                {
                    return Some(kind);
                }
            }
            map.values().find_map(media_kind_of_value)
        },
        _ => None,
    }
}

/// I-7: media is refused, never dropped.
///
/// No model `apr serve` loads has an image, audio or video encoder, so an input that
/// carries one is answered with a 4xx naming the encoder it would need. Before this,
/// a `data:image/…` string was tokenized as text and embedded: a 200 with a vector
/// that described the base64, not the picture. An input that is neither text nor
/// media (a token array, say) is a 400 too.
fn refuse_unsupported_input(inputs: &EmbeddingInput, route: &str) -> Result<(), RErr> {
    let refuse = |index: usize, kind: &str| {
        Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "{route}: input at index {index} is {kind} media, and this model has no \
                     {kind} encoder; media is refused, never dropped"
                ),
            }),
        ))
    };
    match inputs {
        EmbeddingInput::Unsupported(value) => match media_kind_of_value(value) {
            Some(kind) => refuse(0, kind),
            None => Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("{route}: input must be a string or an array of strings"),
                }),
            )),
        },
        _ => match inputs
            .iter()
            .enumerate()
            .find_map(|(index, text)| media_kind_of_str(text).map(|kind| (index, kind)))
        {
            Some((index, kind)) => refuse(index, kind),
            None => Ok(()),
        },
    }
}

/// A `dimensions` request field the generative path can honour: only its hidden size.
///
/// A generative model's vector is a mean-pooled hidden state, not a Matryoshka
/// embedding, so a prefix of it is not a smaller embedding; any other width is a 400.
fn check_generative_dimensions(dimensions: Option<usize>, hidden_dim: usize) -> Result<(), RErr> {
    match dimensions {
        Some(dim) if dim != hidden_dim => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "dimensions {dim}: a generative model's embedding is its {hidden_dim}-wide \
                     hidden state and cannot be truncated; ask for {hidden_dim} or omit the field"
                ),
            }),
        )),
        _ => Ok(()),
    }
}

/// Embed each input with the resident EmbeddingGemma 2 encoder (EG-3).
///
/// Each text goes through [`embed_text`](crate::gguf::forward_embedding_gemma2::embed_text)
/// with `Prompt::Raw`, the call `apr embed` makes with no `--task`, so the HTTP routes and
/// the CLI return equal vectors for the same text. `dimensions` is `--dim`: a Matryoshka
/// width no wider than the vector.
fn embed_with_gemma2(
    served: &super::EmbeddingServed,
    inputs: &EmbeddingInput,
    dimensions: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), RErr> {
    use crate::gguf::embedding_gemma2_input::Prompt;
    use crate::gguf::forward_embedding_gemma2::{embed_text, MATRYOSHKA_DIMS};

    let bad_request = |error: String| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }));
    let width = served.encoder.output_dim();
    if let Some(dim) = dimensions {
        if !(MATRYOSHKA_DIMS.contains(&dim) && dim <= width) {
            return Err(bad_request(format!(
                "dimensions {dim}: this embedding model returns {width} values and truncates to \
                 {:?}",
                MATRYOSHKA_DIMS
                    .iter()
                    .filter(|d| **d <= width)
                    .collect::<Vec<_>>()
            )));
        }
    }

    let mut out = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0usize;
    for (index, text) in inputs.iter().enumerate() {
        if text.is_empty() {
            return Err(bad_request(format!(
                "Input at index {index} cannot be empty"
            )));
        }
        let (tokens, vector) = embed_text(
            &served.encoder,
            &served.tokenizer,
            &Prompt::Raw,
            text,
            dimensions,
        )
        .map_err(|e| {
            let status = super::generation_error_status(&e);
            let error = if status == StatusCode::BAD_REQUEST {
                e.to_string()
            } else {
                format!("Embedding forward pass failed: {e}")
            };
            (status, Json(ErrorResponse { error }))
        })?;
        prompt_tokens += tokens;
        out.push(vector);
    }
    Ok((out, prompt_tokens))
}
