//! `GET /v1/capability`: the model-capability registry on the HTTP surface
//! (#3856 Row 3).
//!
//! `apr serve` has one router per model path (APR CPU, APR GPU, GGUF, SafeTensors,
//! wgpu, batch), and each one reaches the socket through its own `axum::serve`
//! call. This route is attached at every one of those calls, not inside any one
//! router, so a path added later cannot ship without it.
//! `every_axum_serve_site_attaches_the_capability_route` fails if a serve site
//! skips it.
//!
//! The body comes from [`crate::commands::capability::registry_json`], the same
//! function behind `apr capability --json` and the `apr.capability` MCP tool. The
//! three transports therefore answer from one embedded contract.

use axum::{http::StatusCode, response::IntoResponse, routing::get, Json, Router};

/// The route path. `pub(crate)` so the e2e and the guard name one string.
pub(crate) const CAPABILITY_PATH: &str = "/v1/capability";

/// Attach `GET /v1/capability` to a fully built router.
pub(crate) fn with_capability_route(app: Router) -> Router {
    app.route(CAPABILITY_PATH, get(capability_handler))
}

async fn capability_handler() -> impl IntoResponse {
    match crate::commands::capability::registry_json() {
        Ok(doc) => (StatusCode::OK, Json(doc)).into_response(),
        // A broken embedded contract is a server fault. Answering with an empty
        // registry would read as "nothing is supported", a confident wrong answer.
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `axum::serve(listener, …)` under `commands/serve/` must hand the
    /// socket a router wrapped by [`with_capability_route`].
    #[test]
    fn every_axum_serve_site_attaches_the_capability_route() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands/serve");
        let mut sites = 0;
        let mut bare = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("read commands/serve") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.ends_with(".rs") || name.starts_with("tests") || name == "capability_route.rs"
            {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read source");
            for (i, line) in src.lines().enumerate() {
                if let Some(rest) = line.trim_start().strip_prefix("axum::serve(listener, ") {
                    sites += 1;
                    if !rest.starts_with("with_capability_route(") {
                        bare.push(format!("{name}:{}: {}", i + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            sites >= 7,
            "found {sites} axum::serve sites; the scan no longer sees the serve paths"
        );
        assert!(
            bare.is_empty(),
            "serve sites without GET {CAPABILITY_PATH}:\n{}",
            bare.join("\n")
        );
    }

    #[test]
    fn the_route_answers_the_registry() {
        use tower::ServiceExt;
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        rt.block_on(async {
            let app = with_capability_route(Router::new());
            let req = axum::http::Request::builder()
                .uri(CAPABILITY_PATH)
                .body(axum::body::Body::empty())
                .expect("request");
            let resp = app.oneshot(req).await.expect("oneshot");
            assert_eq!(resp.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
                .await
                .expect("body");
            let doc: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
            let expected = crate::commands::capability::registry_json().expect("registry");
            assert_eq!(
                doc, expected,
                "HTTP must serve the CLI's registry byte-for-byte"
            );
            assert!(
                doc["quant_types"].as_array().is_some_and(|q| !q.is_empty()),
                "an empty quant table is not a registry"
            );
        });
    }
}
