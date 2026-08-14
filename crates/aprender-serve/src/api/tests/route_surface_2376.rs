//! Falsifiers for aprender#2376 findings 7 and 8 — what the server TELLS a client
//! must match what the server DOES.
//!
//! Both are black-box: every assertion is something a client can observe from
//! outside with curl. Nothing here inspects a handler or counts a function call.
//!
//! - Finding 8 (route drift): the surface a client is told about must be exactly
//!   the surface that answers. Probed across every `RouterConfig`, in both
//!   directions — advertised ⇒ mounted, and unadvertised ⇒ absent.
//! - Finding 7 (error-body shape): no error may leave this server as `text/plain`,
//!   and none may name something a client cannot act on (a serde cursor position,
//!   a Rust constructor).

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{
    advertised_routes, create_router_with_config, AppState, RouterConfig,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Every `RouterConfig` a client can be served by. `cors` does not gate a route,
/// so varying it here would only slow the probe down.
fn all_configs() -> Vec<RouterConfig> {
    let mut configs = Vec::new();
    for openai_api in [true, false] {
        for metrics in [true, false] {
            configs.push(RouterConfig {
                openai_api,
                cors: true,
                metrics,
            });
        }
    }
    configs
}

fn router(config: &RouterConfig) -> axum::Router {
    create_router_with_config(AppState::with_cache(10), config.clone())
}

async fn body_string(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Dispatch one request against a fresh router built from `config`.
async fn probe(
    config: &RouterConfig,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &str,
) -> (StatusCode, Option<String>, String) {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(content_type) = content_type {
        request = request.header("content-type", content_type);
    }
    let response = router(config)
        .oneshot(
            request
                .body(Body::from(body.to_string()))
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    (status, content_type, body_string(response).await)
}

/// The routes named in the live 404 body — i.e. what a client is actually told.
async fn advertised_over_http(config: &RouterConfig) -> Vec<String> {
    let (status, _, body) = probe(config, "GET", "/no/such/route", None, "").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json 404 body");
    parsed["routes"]
        .as_array()
        .expect("routes array in 404 body")
        .iter()
        .map(|r| r.as_str().unwrap_or_default().to_string())
        .collect()
}

/// Axum path params are placeholders; substitute something concrete to probe.
fn concrete(path: &str) -> String {
    path.replace(":request_id", "not-a-uuid")
}

// ---------------------------------------------------------------------------
// Finding 8: the advertised surface IS the mounted surface
// ---------------------------------------------------------------------------

/// Every route the server advertises must answer, under every configuration.
///
/// The 0.63.0+#2449 tree passes this for the default config and FAILS it for
/// `--no-metrics`: the 404 body advertised `GET /metrics`, `GET /metrics/dispatch`
/// and `POST /metrics/dispatch/reset` while the router had unmounted all three.
#[tokio::test]
async fn advertised_routes_answer_under_every_config() {
    for config in all_configs() {
        for route in advertised_over_http(&config).await {
            let (method, path) = route.split_once(' ').expect("METHOD /path");
            let (status, _, _) = probe(
                &config,
                method,
                &concrete(path),
                Some("application/json"),
                "{}",
            )
            .await;
            assert_ne!(
                status,
                StatusCode::NOT_FOUND,
                "openai_api={} metrics={}: `{route}` is advertised to clients but does not answer",
                config.openai_api,
                config.metrics,
            );
        }
    }
}

/// The other direction: a route the server does NOT advertise must not answer.
///
/// Without this, "advertise everything" would pass the test above. The candidate
/// universe is every route any configuration mounts, so a route silently dropped
/// from one config's list while still being mounted is caught here.
#[tokio::test]
async fn unadvertised_routes_do_not_answer() {
    let universe: std::collections::BTreeSet<String> = all_configs()
        .iter()
        .flat_map(advertised_routes)
        .collect();

    for config in all_configs() {
        let advertised: std::collections::BTreeSet<String> =
            advertised_over_http(&config).await.into_iter().collect();

        for route in &universe {
            if advertised.contains(route) {
                continue;
            }
            let (method, path) = route.split_once(' ').expect("METHOD /path");
            let (status, _, _) = probe(
                &config,
                method,
                &concrete(path),
                Some("application/json"),
                "{}",
            )
            .await;
            assert_eq!(
                status,
                StatusCode::NOT_FOUND,
                "openai_api={} metrics={}: `{route}` answers but is advertised to nobody",
                config.openai_api,
                config.metrics,
            );
        }
    }
}

/// The specific regression: `--no-metrics` must not leave telemetry endpoints in
/// the list a client is handed.
#[tokio::test]
async fn no_metrics_stops_advertising_metrics() {
    let config = RouterConfig {
        openai_api: true,
        cors: true,
        metrics: false,
    };
    let advertised = advertised_over_http(&config).await;
    assert!(
        !advertised.iter().any(|r| r == "GET /metrics"),
        "a server that unmounted /metrics must not advertise it: {advertised:?}"
    );

    // And the baseline: with metrics on, it IS advertised — so the assertion above
    // cannot be satisfied by never advertising anything.
    let config = RouterConfig {
        metrics: true,
        ..config
    };
    let advertised = advertised_over_http(&config).await;
    assert!(
        advertised.iter().any(|r| r == "GET /metrics"),
        "a server serving /metrics must advertise it: {advertised:?}"
    );
}

/// `advertised_routes` is what the CLI banner prints. It must agree with the list
/// the running server hands out, or the banner is a second, independent claim.
#[tokio::test]
async fn banner_source_agrees_with_live_server() {
    for config in all_configs() {
        assert_eq!(
            advertised_routes(&config),
            advertised_over_http(&config).await,
            "openai_api={} metrics={}: the banner list and the 404 list disagree",
            config.openai_api,
            config.metrics,
        );
    }
}

// ---------------------------------------------------------------------------
// Finding 7: every error is a JSON envelope, and leaks nothing
// ---------------------------------------------------------------------------

/// Requests that every body-taking route rejects, one way or another.
const MALFORMED: &[(&str, Option<&str>, &str)] = &[
    // Unparseable JSON -> axum JsonSyntaxError (400)
    ("POST", Some("application/json"), "{not json"),
    // Truncated JSON -> axum JsonSyntaxError (400)
    ("POST", Some("application/json"), "{\"model\":"),
    // No Content-Type -> axum MissingJsonContentType (415)
    ("POST", None, "{\"prompt\":\"hi\",\"max_tokens\":2}"),
    // Wrong field types -> axum JsonDataError (422)
    (
        "POST",
        Some("application/json"),
        "{\"prompt\":[1,2,3],\"max_tokens\":\"lots\"}",
    ),
];

const BODY_ROUTES: &[&str] = &[
    "/generate",
    "/tokenize",
    "/batch/generate",
    "/batch/tokenize",
    "/stream/generate",
    "/realize/embed",
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/predict",
    "/api/generate",
];

/// No error response may reach a client as anything but JSON.
///
/// 0.63.0 answered 400 and 415 with `text/plain; charset=utf-8` on every route
/// here; only 422 was enveloped.
#[tokio::test]
async fn every_error_body_is_json() {
    let config = RouterConfig::default();
    for path in BODY_ROUTES {
        for (method, content_type, body) in MALFORMED {
            let (status, response_type, response_body) =
                probe(&config, method, path, *content_type, body).await;
            if status.is_success() {
                continue;
            }
            let response_type = response_type.unwrap_or_default();
            assert!(
                response_type.starts_with("application/json"),
                "{method} {path} answered {status} as `{response_type}`, not JSON: {response_body}"
            );
            let parsed: serde_json::Value = serde_json::from_str(&response_body)
                .unwrap_or_else(|e| panic!("{method} {path} {status}: body is not JSON ({e}): {response_body}"));
            assert!(
                parsed.get("error").is_some(),
                "{method} {path} {status}: JSON error body must carry an `error` field: {response_body}"
            );
        }
    }
}

/// No error response may hand a client our internals: the serde parser's cursor,
/// a Rust type or constructor, or a source path.
#[tokio::test]
async fn no_error_body_leaks_internals() {
    // Substrings a client must never receive. Each was observed in 0.63.0.
    const LEAKS: &[&str] = &[
        "at line 1 column",   // "key must be a string at line 1 column 2"
        "EOF while parsing",  // serde's own diagnostic
        "AppState",           // a Rust type no HTTP client can construct
        "::demo()",           // ...and the constructor it was told to call
        ".rs:",               // any source location
    ];

    let config = RouterConfig::default();
    let mut cases: Vec<(&str, &str, Option<&str>, &str)> = Vec::new();
    for path in BODY_ROUTES {
        for (method, content_type, body) in MALFORMED {
            cases.push((method, path, *content_type, body));
        }
    }
    // The 503 from an APR-less server, which told clients to "Use AppState::demo()".
    cases.push((
        "POST",
        "/v1/predict",
        Some("application/json"),
        "{\"features\":[1.0,2.0]}",
    ));

    for (method, path, content_type, body) in cases {
        let (status, _, response_body) = probe(&config, method, path, content_type, body).await;
        if status.is_success() {
            continue;
        }
        for leak in LEAKS {
            assert!(
                !response_body.contains(leak),
                "{method} {path} {status} leaked `{leak}` to the client: {response_body}"
            );
        }
    }
}

/// The envelope must not become a shredder: a handler's own diagnostic — which is
/// already JSON and is the whole value of a 400 — has to survive verbatim.
///
/// This is what separates "envelope what axum rejected" from "replace every error
/// with a generic string".
#[tokio::test]
async fn handler_diagnostics_survive_the_envelope() {
    let (status, content_type, body) = probe(
        &RouterConfig::default(),
        "POST",
        "/v1/predict",
        Some("application/json"),
        "{\"features\":[]}",
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        content_type
            .unwrap_or_default()
            .starts_with("application/json"),
        "handler errors are JSON too"
    );
    assert!(
        body.contains("features"),
        "the caller must still be told WHICH field they got wrong, got: {body}"
    );

    // A second, differently-worded handler error: two distinct diagnostics must
    // stay distinct. A middleware that flattened both to one generic string would
    // satisfy either assertion alone but not this comparison.
    let (other_status, _, other_body) = probe(
        &RouterConfig::default(),
        "POST",
        "/v1/predict",
        Some("application/json"),
        "{\"features\":[1.0,2.0]}",
    )
    .await;
    assert_ne!(
        (status, body.clone()),
        (other_status, other_body.clone()),
        "two different rejections must not collapse to one message: {body}"
    );
}

/// A 404 is an error status too, and its body carries the route list — the
/// envelope must not eat it.
#[tokio::test]
async fn route_list_survives_the_envelope() {
    let config = RouterConfig::default();
    let (status, content_type, body) = probe(&config, "GET", "/no/such/route", None, "").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        content_type.unwrap_or_default().starts_with("application/json"),
        "404 must stay JSON"
    );
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json 404 body");
    assert!(
        !parsed["routes"]
            .as_array()
            .expect("routes array")
            .is_empty(),
        "the 404 must still list the routes it promises: {body}"
    );
}

/// Every route this router mounts must come from `route_table`.
///
/// `unadvertised_routes_do_not_answer` above builds its candidate universe by
/// unioning the ADVERTISED lists. That was structurally unable to catch the
/// defect it was written for: `/api/tags`, `/api/show` and `/api/version` were
/// mounted, named in no list, and so never entered the universe and were never
/// probed. Its doc comment claims the universe is "every route any configuration
/// mounts"; the code says advertises, and those differed by exactly three routes.
///
/// Folding the index and the mount into one table closed that gap by
/// construction — advertised and mounted are now the same list, because the
/// mount loop consumes the table the index was built from. This test is the
/// REINTRODUCTION guard: a hand-written `.route("/path", ...)` would restore the
/// two-copy pattern and silently reopen the hole, since such a route is once
/// again mounted and advertised to nobody.
///
/// Source-level because axum exposes no way to enumerate a `Router`'s paths.
#[test]
fn every_mounted_route_comes_from_the_route_table() {
    let src = include_str!("../router.rs");

    // `GET /` is mounted by hand and advertised by hand: its body IS the index
    // derived from the table, so it cannot be a row of the table it prints.
    const HAND_MOUNTED: &[&str] = &["/"];

    // Vacuity control FIRST: prove we are reading the real router with a real
    // table before concluding anything from an absence. Rows look like
    // `("GET", "/health", get(handler))`.
    let table_rows = ["(\"GET\", \"/", "(\"POST\", \"/"]
        .iter()
        .map(|pat| src.matches(pat).count())
        .sum::<usize>();
    assert!(
        table_rows > 30,
        "found only {table_rows} route-table rows — this test is parsing the wrong \
         thing, or the table was dismantled. Fix the parser, not this number."
    );

    // Any `.route(` whose first argument is a string literal is a direct mount
    // that bypasses the table. `.route(path, handler)` in the fold loop is not.
    let literal_mounts: Vec<&str> = src
        .match_indices(".route(")
        .filter_map(|(i, m)| {
            let rest = &src[i + m.len()..];
            let arg = rest.trim_start();
            let arg = arg.strip_prefix('"')?;
            let end = arg.find('"')?;
            Some(&arg[..end])
        })
        .collect();

    for path in &literal_mounts {
        assert!(
            HAND_MOUNTED.contains(path),
            "`{path}` is mounted by a hand-written .route() call rather than from \
             route_table(), so it is advertised to nobody — neither the 404 body nor \
             the `apr serve` startup banner will name it. Add it to a *_routes() table."
        );
    }

    // And the hand-mounted allowlist must not rot: `/` really is still mounted.
    assert!(
        literal_mounts.contains(&"/"),
        "`GET /` is no longer mounted directly; the allowlist is stale: {literal_mounts:?}"
    );
}

/// The Ollama discovery surface must EXIST, not merely be self-consistent.
///
/// The table fold makes the index and the mount agree by construction — but
/// agreement with nothing is still agreement. Deleting `/api/tags`, `/api/show`
/// and `/api/version` from `route_table` removes them from both sides at once
/// and every consistency test above stays green. Verified: that mutation passes
/// all nine of them.
///
/// So this asserts the capability instead of the coherence. An Ollama client
/// calls `/api/tags` before it will chat and `/api/show` to probe capabilities;
/// without them the "drop-in Ollama replacement" claim in `openai_routes` is
/// unreachable in practice, however tidy the routing table is.
#[tokio::test]
async fn ollama_discovery_routes_answer() {
    let config = RouterConfig::default();
    for (method, path) in [
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("GET", "/api/version"),
    ] {
        let (status, _, _) = probe(&config, method, path, Some("application/json"), "{}").await;
        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "{method} {path} does not answer — an Ollama client probes this before it \
             will chat, so the drop-in replacement claim is dead without it"
        );
    }
}
