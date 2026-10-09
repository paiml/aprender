//! #3568 PR 4: OpenAI's `response_format` on the chat servers apr-cli runs itself (GGUF on
//! wgpu, APR on the CPU, APR on CUDA, SafeTensors on the CPU).
//!
//! None of their loops applies a constraint. Until #3568 they read the fields they knew and
//! dropped `response_format`, so a client asking for JSON got free text with a 200. A request
//! that asks for a constraint is now refused by name, with the status and body the realizar
//! router answers on its own unconstrained backends, before any model work: "a constraint that
//! is silently ignored is decoration".

use axum::{http::StatusCode, response::IntoResponse, Json};
use realizar::constrain::{ConstraintError, OUTSIDE_3568};

/// The apr-cli chat loop that answered: the `path` a refusal names.
pub(crate) const GGUF_WGPU: &str = "serve-gguf-wgpu";
pub(crate) const APR_CPU: &str = "serve-apr-cpu";
pub(crate) const APR_CUDA: &str = "serve-apr-cuda";
pub(crate) const SAFETENSORS_CPU: &str = "serve-safetensors-cpu";

/// The refusal for a chat request whose `response_format` asks for a constraint `path`
/// cannot apply: 400, `{"error": <SchemaUnsupportedPath line>, "type":
/// "schema_unsupported_path"}`. `None` when it asks for none: the field absent, `null`, or
/// `{"type": "text"}`, which is what an unconstrained reply already is. Any other value,
/// including a `type` apr does not know, is refused rather than dropped.
pub(crate) fn refuse_response_format(
    req: &serde_json::Value,
    path: &str,
) -> Option<axum::response::Response> {
    let format = req.get("response_format").filter(|f| !f.is_null())?;
    if format.get("type").and_then(serde_json::Value::as_str) == Some("text") {
        return None;
    }
    let refusal = ConstraintError::UnsupportedPath {
        path: path.to_string(),
        removed_by: OUTSIDE_3568.to_string(),
    };
    Some(
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": refusal.to_string(),
                "type": "schema_unsupported_path",
            })),
        )
            .into_response(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body_of(resp: axum::response::Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .expect("a refusal body is small");
        (
            status,
            serde_json::from_slice(&bytes).expect("a refusal body is JSON"),
        )
    }

    #[test]
    fn a_request_without_a_constraint_is_served_as_before() {
        for req in [
            serde_json::json!({"messages": []}),
            serde_json::json!({"messages": [], "response_format": null}),
            serde_json::json!({"messages": [], "response_format": {"type": "text"}}),
        ] {
            assert!(refuse_response_format(&req, APR_CPU).is_none(), "{req}");
        }
    }

    #[tokio::test]
    async fn every_constraint_a_client_can_ask_for_is_refused_by_name() {
        let schema = serde_json::json!({"type": "object"});
        for (format, path) in [
            (serde_json::json!({"type": "json_object"}), GGUF_WGPU),
            (
                serde_json::json!({"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}),
                APR_CPU,
            ),
            (serde_json::json!({"type": "json_schema"}), APR_CUDA),
            (serde_json::json!({"type": "grammar"}), SAFETENSORS_CPU),
            (serde_json::json!({"type": 7}), APR_CPU),
            (serde_json::json!("json_object"), APR_CPU),
        ] {
            let req = serde_json::json!({"messages": [], "response_format": format});
            let resp =
                refuse_response_format(&req, path).unwrap_or_else(|| panic!("{format} served"));
            let (status, body) = body_of(resp).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{format}");
            assert_eq!(body["type"], "schema_unsupported_path", "{format}");
            let error = body["error"].as_str().expect("the error is one line");
            assert!(error.starts_with("SchemaUnsupportedPath: "), "{error}");
            assert!(
                error.contains(&format!("the {path} generation path")),
                "{error}"
            );
            assert!(error.contains(OUTSIDE_3568), "{error}");
        }
    }

    #[test]
    fn the_paths_are_four_distinct_names_none_of_them_a_realizar_one() {
        let paths = [GGUF_WGPU, APR_CPU, APR_CUDA, SAFETENSORS_CPU];
        let unique: std::collections::BTreeSet<_> = paths.iter().collect();
        assert_eq!(unique.len(), paths.len());
        for realizar_path in [
            "serve-wgpu",
            "serve-apr",
            "serve-apr-q4k-cuda",
            "serve-safetensors-cuda",
        ] {
            assert!(
                !paths.contains(&realizar_path),
                "{realizar_path} is the realizar router's"
            );
        }
    }
}
