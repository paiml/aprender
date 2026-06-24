//! PMAT-928 — `apr serve` streams Ollama NDJSON for `/api/chat` + `/api/generate`.
//!
//! The PROVEN GAP this falsifier closes: after PMAT-923 (#2216) the Ollama
//! endpoints on the REAL `apr serve` routers always returned a SINGLE coalesced
//! (`done:true`) JSON object — even though Ollama clients default to
//! `stream:true` and expect newline-delimited JSON: a sequence of
//! `{...,message:{role,content:<token>},done:false}` chunks then a terminal
//! `{...,done:true,...}` object. A real Ollama client streaming from `apr serve`
//! therefore saw a non-streaming body.
//!
//! This test exercises the REAL apr-cli APR-CPU serve router (the exact router
//! `apr serve <model.apr>` mounts) via the
//! `build_demo_streaming_apr_cpu_router_for_test` seam — whose streaming path is
//! driven by a deterministic scripted token sequence through the SAME mpsc
//! channel + NDJSON reshape pipeline the production transformer uses (only the
//! token *source* is faked, never the wire framing).
//!
//! It POSTs Ollama requests and asserts:
//!   * `stream:true` → `Content-Type: application/x-ndjson` AND the body parses
//!     as MULTIPLE newline-delimited JSON objects: intermediate `done:false`
//!     token chunks PLUS a final `done:true` object (NOT a single object).
//!   * `stream:false` → exactly one coalesced JSON object (`done:true`).
//!
//! RED on the old PMAT-923 code: `stream:true` returned a single coalesced
//! object (the adapter forced `stream:false`), so `lines.len() == 1` and the
//! content-type was `application/json`. GREEN once the NDJSON path is wired.
//!
//! Mutation-verified: forcing `stream:false` internally (e.g. clamping the
//! `req.stream` branch off in the router handler) collapses the body back to a
//! single object → the multi-line assertion flips RED.
//!
//! Contract: OBLIG-OLLAMA-NDJSON-STREAMING in
//! `contracts/apr-serve-openai-compat-v1.yaml`.

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

#[cfg(feature = "inference")]
mod tests {
    use apr_cli::serve_test_support::build_demo_streaming_apr_cpu_router_for_test;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use serde_json::Value;
    use tower::ServiceExt;

    /// POST `body` to `path` on the REAL streaming apr-cli APR-CPU serve router.
    /// Returns `(status, content_type, raw_body_text)`.
    async fn post_raw(path: &str, body: Value) -> (StatusCode, String, String) {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = build_demo_streaming_apr_cpu_router_for_test()
            .oneshot(req)
            .await
            .unwrap();
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap_or_default();
        (status, content_type, text)
    }

    /// Parse an NDJSON body into one JSON object per non-empty line.
    fn parse_ndjson(text: &str) -> Vec<Value> {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Value>(l).expect("each NDJSON line is one JSON object"))
            .collect()
    }

    #[tokio::test]
    async fn api_chat_stream_true_is_ndjson_multi_chunk() {
        let (status, content_type, text) = post_raw(
            "/api/chat",
            serde_json::json!({
                "model": "apr",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        )
        .await;

        assert_eq!(
            status,
            StatusCode::OK,
            "stream chat returns 200. body={text}"
        );

        // PROOF #1: streaming wire content-type, not application/json.
        assert_eq!(
            content_type, "application/x-ndjson",
            "stream:true must be NDJSON, not a coalesced JSON object. ct={content_type} body={text}"
        );

        // PROOF #2: MULTIPLE newline-delimited objects (RED on coalesced code:
        // a single object would yield exactly one line).
        let objs = parse_ndjson(&text);
        assert!(
            objs.len() >= 2,
            "stream:true must yield multiple NDJSON objects (>=1 token chunk + final), got {}. body={text}",
            objs.len()
        );

        // PROOF #3: at least one intermediate done:false token chunk.
        let intermediate: Vec<&Value> = objs.iter().filter(|o| o["done"] == false).collect();
        assert!(
            !intermediate.is_empty(),
            "must carry intermediate done:false token chunks. body={text}"
        );
        for chunk in &intermediate {
            assert_eq!(
                chunk["message"]["role"], "assistant",
                "chat token chunk nests message.role. body={text}"
            );
            assert!(
                chunk["message"]["content"].is_string(),
                "chat token chunk carries message.content string. body={text}"
            );
        }

        // PROOF #4: the LAST object is the terminal done:true with stats.
        let last = objs.last().expect("at least one object");
        assert_eq!(
            last["done"], true,
            "final NDJSON object must be done:true. body={text}"
        );
        assert!(
            last["eval_count"].is_number(),
            "terminal object carries eval_count. body={text}"
        );

        // PROOF #5: reassembled content matches the scripted tokens.
        let assembled: String = intermediate
            .iter()
            .filter_map(|o| o["message"]["content"].as_str())
            .collect();
        assert_eq!(
            assembled, "Hello, world!",
            "per-token chunks reassemble to the full generation. body={text}"
        );
    }

    #[tokio::test]
    async fn api_generate_stream_true_is_ndjson_multi_chunk() {
        let (status, content_type, text) = post_raw(
            "/api/generate",
            serde_json::json!({
                "model": "apr",
                "prompt": "hi",
                "stream": true
            }),
        )
        .await;

        assert_eq!(
            status,
            StatusCode::OK,
            "stream generate returns 200. body={text}"
        );
        assert_eq!(
            content_type, "application/x-ndjson",
            "stream:true generate is NDJSON. ct={content_type} body={text}"
        );

        let objs = parse_ndjson(&text);
        assert!(
            objs.len() >= 2,
            "generate stream yields multiple NDJSON objects, got {}. body={text}",
            objs.len()
        );

        // Intermediate generate chunks use the flat `response` field.
        let intermediate: Vec<&Value> = objs.iter().filter(|o| o["done"] == false).collect();
        assert!(
            !intermediate.is_empty(),
            "intermediate done:false chunks. body={text}"
        );
        for chunk in &intermediate {
            assert!(
                chunk["response"].is_string(),
                "generate token chunk carries flat response string. body={text}"
            );
            assert!(
                chunk.get("message").is_none(),
                "generate uses flat response, not nested message. body={text}"
            );
        }

        let last = objs.last().expect("at least one object");
        assert_eq!(last["done"], true, "final object is done:true. body={text}");

        let assembled: String = intermediate
            .iter()
            .filter_map(|o| o["response"].as_str())
            .collect();
        assert_eq!(assembled, "Hello, world!", "tokens reassemble. body={text}");
    }

    #[tokio::test]
    async fn api_chat_stream_false_is_single_coalesced_object() {
        let (status, content_type, text) = post_raw(
            "/api/chat",
            serde_json::json!({
                "model": "apr",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }),
        )
        .await;

        assert_eq!(
            status,
            StatusCode::OK,
            "non-stream chat returns 200. body={text}"
        );

        // stream:false must remain a single coalesced JSON object.
        let objs = parse_ndjson(&text);
        assert_eq!(
            objs.len(),
            1,
            "stream:false must be exactly ONE coalesced object, got {}. body={text}",
            objs.len()
        );
        assert_eq!(
            objs[0]["done"], true,
            "coalesced object is done:true. body={text}"
        );
        // Coalesced path uses application/json (axum Json), not NDJSON.
        assert!(
            content_type.starts_with("application/json"),
            "coalesced path is application/json. ct={content_type}"
        );
    }

    #[tokio::test]
    async fn api_chat_default_stream_is_streaming() {
        // Ollama default: a request that OMITS `stream` must STREAM (NDJSON),
        // matching ollama's wire default (stream:true).
        let (status, content_type, text) = post_raw(
            "/api/chat",
            serde_json::json!({
                "model": "apr",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;

        assert_eq!(
            status,
            StatusCode::OK,
            "default chat returns 200. body={text}"
        );
        assert_eq!(
            content_type, "application/x-ndjson",
            "absent stream field defaults to streaming (Ollama default). ct={content_type} body={text}"
        );
        let objs = parse_ndjson(&text);
        assert!(
            objs.len() >= 2,
            "default stream is multi-chunk. body={text}"
        );
        assert_eq!(objs.last().unwrap()["done"], true);
    }
}
