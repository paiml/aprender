//! aprender#2465: the advertised route list and the MOUNTED route surface must be
//! the same set, checked in both directions.
//!
//! The #2376(8) falsifiers (`route_surface_2376.rs`) check one arrow only:
//! *advertised ⇒ answers*. Their sibling `unadvertised_routes_do_not_answer` looks
//! like the other arrow but is not — it builds its candidate universe by calling
//! `advertised_routes` over every config, so a route that is advertised by NO
//! config is never in the universe and is never probed. Delete an entry from
//! `OPENAI_ROUTES` and it leaves the 404 body and the startup banner while the
//! route keeps answering, and the whole 15,706-test crate suite stays green.
//!
//! That blindness was not hypothetical: `/api/tags`, `/api/show` and
//! `/api/version` were mounted by #2396(2) and never added to `OPENAI_ROUTES`.
//! Three routes a client is never told about, shipped in 0.63.0+, past a test
//! named "advertised routes are all mounted".
//!
//! The missing arrow needs a source of truth for "mounted" that is NOT the
//! advertised list. This module reads the `.route(...)` calls out of the text of
//! `create_router_with_config` itself. Two independently-derived sets, compared
//! exhaustively, so drift in either direction turns a test red.

use std::collections::BTreeSet;

use crate::api::{advertised_routes, RouterConfig};

/// The source text of the module that both mounts and advertises the routes.
const ROUTER_SRC: &str = include_str!("../router.rs");

/// The method-router constructors a route may be mounted with.
const METHOD_CTORS: &[&str] = &["get", "post", "put", "delete", "patch", "head", "options"];

// ---------------------------------------------------------------------------
// Extracting the mounted surface from source
// ---------------------------------------------------------------------------

/// Strip whole-line `//` comments. Prose in this file names routes constantly
/// ("leaving it unmounted would advertise a route"), and a commented-out
/// `.route(...)` must not count as mounted.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Is the byte at `idx` the start of a whole identifier (not the tail of one)?
fn on_token_boundary(haystack: &str, idx: usize) -> bool {
    haystack[..idx]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_alphanumeric() && c != '_')
}

/// Every `(METHOD, path)` mounted by a literal `.route(path, method(..))` call in
/// `src`, as uppercase method and path-with-params.
///
/// Handles all three shapes rustfmt produces here: one line, a split argument
/// list, and a handler written as an inline closure.
fn extract_mounted(src: &str) -> BTreeSet<(String, String)> {
    let src = strip_line_comments(src);
    let mut out = BTreeSet::new();
    let mut rest = src.as_str();

    while let Some(at) = rest.find(".route(") {
        rest = &rest[at + ".route(".len()..];

        // First string literal after `.route(` is the path.
        let Some(open) = rest.find('"') else { break };
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find('"') else {
            break;
        };
        let path = after_open[..close].to_string();
        let tail = &after_open[close + 1..];

        // First method-router constructor after the path is the method.
        let method = METHOD_CTORS
            .iter()
            .filter_map(|ctor| {
                let pattern = format!("{ctor}(");
                tail.find(&pattern)
                    .filter(|at| on_token_boundary(tail, *at))
                    .map(|at| (at, *ctor))
            })
            .min_by_key(|(at, _)| *at);

        if let Some((_, ctor)) = method {
            out.insert((ctor.to_uppercase(), path));
        }
        rest = tail;
    }

    out
}

/// The body of `create_router_with_config` — the one function that mounts routes.
///
/// Bounded to that function so that a `.route(...)` in a doc example or a
/// neighbouring helper cannot be mistaken for a mount.
fn mount_fn_body() -> &'static str {
    const SIGNATURE: &str = "pub fn create_router_with_config";
    let at = ROUTER_SRC
        .find(SIGNATURE)
        .expect("router.rs must define create_router_with_config");
    let body = &ROUTER_SRC[at..];
    let end = body
        .find("\n}\n")
        .expect("create_router_with_config must be brace-closed at column 0");
    &body[..end]
}

/// The routes `create_router_with_config` mounts, read from its source.
///
/// The CUDA block is counted only when the feature that compiles it is on, so
/// this set is comparable with `advertised_routes` under either build.
fn mounted_routes() -> BTreeSet<(String, String)> {
    let body = mount_fn_body();
    let (before, cuda_onward) = body
        .split_once("#[cfg(feature = \"cuda\")]")
        .unwrap_or((body, ""));
    let (cuda_block, after) = cuda_onward
        .find("\n    }\n")
        .map_or((cuda_onward, ""), |at| cuda_onward.split_at(at));

    let mut mounted = extract_mounted(before);
    mounted.extend(extract_mounted(after));
    if cfg!(feature = "cuda") {
        mounted.extend(extract_mounted(cuda_block));
    }
    mounted
}

/// Everything `advertised_routes` will ever name, as `(METHOD, path)` pairs.
///
/// `RouterConfig::default()` is the maximal config: both gated groups on.
fn advertised_everywhere() -> BTreeSet<(String, String)> {
    advertised_routes(&RouterConfig::default())
        .into_iter()
        .map(|route| {
            let (method, path) = route.split_once(' ').expect("METHOD /path");
            (method.to_string(), path.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The falsifier
// ---------------------------------------------------------------------------

/// The extractor must be able to succeed before its silence means anything.
///
/// A parser that matched nothing would make `mounted_routes()` empty, and an
/// empty set trivially contains no unadvertised route — the exact shape of a
/// guard that passes because it looked wrong rather than because it found
/// nothing wrong.
#[test]
fn source_scan_finds_the_routes_it_is_scanning_for() {
    let mounted = mounted_routes();
    for anchor in [
        ("POST", "/generate"),
        ("GET", "/health"),
        ("GET", "/"),
        ("POST", "/v1/chat/completions"),
    ] {
        let anchor = (anchor.0.to_string(), anchor.1.to_string());
        assert!(
            mounted.contains(&anchor),
            "the source scan missed `{} {}`, which create_router_with_config \
             demonstrably mounts — the extractor is broken, not the router. Found: {:?}",
            anchor.0,
            anchor.1,
            mounted,
        );
    }
}

/// Arrow 1 — every MOUNTED route is ADVERTISED.
///
/// This is the arrow nothing checked. `/api/tags`, `/api/show` and `/api/version`
/// answered here while being absent from the 404 body and the `apr serve run`
/// banner, so an Ollama client that discovered the surface the way the server
/// told it to could not find the endpoint it must call first.
#[test]
fn every_mounted_route_is_advertised() {
    let unadvertised: Vec<(String, String)> = mounted_routes()
        .difference(&advertised_everywhere())
        .cloned()
        .collect();

    assert!(
        unadvertised.is_empty(),
        "create_router_with_config mounts routes that no config advertises, so they \
         are absent from the 404 body and the startup banner: {unadvertised:?}",
    );
}

/// Arrow 2 — every ADVERTISED route is MOUNTED.
///
/// Held by `advertised_routes_answer_under_every_config` over HTTP as well; held
/// here against the source so that deleting a `.route(...)` call is caught by the
/// same comparison that catches deleting a list entry.
#[test]
fn every_advertised_route_is_mounted() {
    let unmounted: Vec<(String, String)> = advertised_everywhere()
        .difference(&mounted_routes())
        .cloned()
        .collect();

    assert!(
        unmounted.is_empty(),
        "advertised_routes names routes create_router_with_config does not mount, so \
         the 404 body and the startup banner promise a surface that 404s: {unmounted:?}",
    );
}

// ---------------------------------------------------------------------------
// The extractor's own case table
// ---------------------------------------------------------------------------

/// Every pattern-matching guard in this repo that shipped wrong shipped wrong
/// because the pattern was reviewed instead of exercised. Must-match and
/// must-not-match, run rather than read.
#[test]
fn extractor_case_table() {
    let must_match: &[(&str, (&str, &str))] = &[
        // One line, the common shape.
        (r#".route("/health", get(health_handler))"#, ("GET", "/health")),
        (r#".route("/generate", post(generate_handler))"#, ("POST", "/generate")),
        // rustfmt splits the argument list once the line is long enough.
        (
            ".route(\n    \"/v1/chat/completions\",\n    post(openai_chat_completions_handler),\n)",
            ("POST", "/v1/chat/completions"),
        ),
        // An inline closure handler, as `GET /` is written.
        (
            ".route(\n    \"/\",\n    get(move || {\n        async move { \"hi\" }\n    }),\n)",
            ("GET", "/"),
        ),
        // A path parameter is part of the path, not a separate token.
        (
            r#".route("/v1/audit/:request_id", get(apr_audit_handler))"#,
            ("GET", "/v1/audit/:request_id"),
        ),
    ];
    for (source, (method, path)) in must_match {
        let found = extract_mounted(source);
        let expected = ((*method).to_string(), (*path).to_string());
        assert!(
            found.contains(&expected),
            "extractor missed `{method} {path}` in:\n{source}\nfound: {found:?}",
        );
    }

    let must_not_match: &[&str] = &[
        // A commented-out mount is not a mount.
        r#"// .route("/ghost", get(ghost_handler))"#,
        // Indented comment, the shape this file's prose actually takes.
        r#"        // leaving it unmounted would .route("/ghost", get(h)) advertise a lie"#,
        // The fallback is not a route.
        r#".fallback(move || async { "nope" })"#,
        // Neither is a layer.
        r#".layer(axum::middleware::from_fn(cancel_on_disconnect))"#,
        // No method-router constructor: not a mount we can classify.
        r#".route("/orphan")"#,
    ];
    for source in must_not_match {
        let found = extract_mounted(source);
        assert!(
            found.is_empty(),
            "extractor treated a non-mount as a mount in:\n{source}\nfound: {found:?}",
        );
    }

    // The token-boundary rule earns its keep: `forget(` ends in `get(` and must
    // not be read as a GET mount.
    let found = extract_mounted(r#".route("/x", std::mem::forget(handler))"#);
    assert!(
        found.is_empty(),
        "`forget(` was read as `get(` — the token-boundary check is not working: {found:?}",
    );
}
