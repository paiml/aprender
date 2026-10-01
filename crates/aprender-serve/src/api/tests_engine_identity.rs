//! #4263: ONE engine. Every verb that generates for an architecture the engine
//! serves enters [`crate::session::Session`] — `apr run`, and every `apr serve`
//! route — and the witness proves it per row. `apr chat`'s row lives in
//! apr-cli beside `ChatSession` (`commands/chat_engine_identity_4263.rs`).
//!
//! A row is judged by the session's witness id, not by re-deriving the verb's
//! prompt tokens: a route that decoded through anything but the session leaves
//! the session's entry count where it was, whatever tokens it built.

use super::*;
use crate::gguf::qwen35_session::Qwen35Session;
use crate::gguf::MappedGGUFModel;
use crate::session::{entries_for, entries_of_session, EntryKind};
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// A model under `$HOME/models` (host-local; the tests SKIP when it is absent).
fn home_model(name: &str) -> String {
    format!(
        "{}/models/{name}",
        std::env::var("HOME").unwrap_or_default()
    )
}

fn model_path() -> String {
    home_model("Qwen3.5-0.8B-Q4_K_M.gguf")
}

/// Which architectures the engine serves, and for the rest the ticket that
/// ports them. When a port lands its `impl ArchForward` appears and
/// [`every_arch_forward_is_a_session_row`] turns red until its row says
/// `Session` — and then the verb rows must cover it too.
const ARCHES: &[(&str, Arch)] = &[
    ("qwen35", Arch::Session("Qwen35Forward")),
    ("qwen3", Arch::Session("DenseForward")),
    ("qwen2", Arch::Session("DenseForward")),
    ("llama", Arch::Session("DenseForward")),
    // #4280 (#4308): `apr serve`'s dense CUDA batch scheduler drives qwen2/qwen3/llama
    // through the session on a borrowed CUDA model; its witness is api/cuda_batch_scheduler_tests.rs.
    ("dense-cuda-batch", Arch::Session("BorrowedCudaForward")),
    // #4269 M (#4305): `apr serve`/`apr chat` on APR CPU and SafeTensors CPU. These rows are
    // formats, not GGUF archs; their verb witnesses are tests_st_serve_session_4269.rs and the
    // tests in apr_transformer/apr_cpu_forward.rs.
    ("apr-cpu", Arch::Session("AprCpuForward")),
    ("safetensors-cpu", Arch::Session("StCpuForward")),
    ("qwen3_moe", Arch::Session("Qwen3MoeForward")),
    // #4269 M2b: `apr serve`'s APR Q4K CUDA scheduler (a format row, like apr-cpu); its
    // witnesses are api/tests/apr_q4k_cancel_2465.rs and apr_q4k_scheduler::generate_q4k.
    ("apr-q4k", Arch::Session("AprQ4kForward")),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    /// Served by the engine through this `ArchForward` impl.
    Session(&'static str),
    /// Not ported yet; names the owning ticket.
    NotYet(&'static str),
}

/// One `apr serve` route: the request, and how many engine entries it must
/// leave (one per prompt the request carries).
struct Route {
    uri: &'static str,
    body: fn() -> serde_json::Value,
    entries: usize,
}

const ROUTES: &[Route] = &[
    Route {
        uri: "/generate",
        body: || serde_json::json!({"prompt": "Engine row generate.", "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/batch/generate",
        body: || serde_json::json!({"prompts": ["Engine row batch one.", "Engine row batch two."], "max_tokens": 2, "temperature": 0.0}),
        entries: 2,
    },
    Route {
        uri: "/realize/batch",
        body: || serde_json::json!({"prompts": ["Engine row realize batch."], "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/stream/generate",
        body: || serde_json::json!({"prompt": "Engine row stream.", "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/realize/generate",
        body: || serde_json::json!({"prompt": "Engine row realize.", "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/v1/completions",
        body: || serde_json::json!({"model": "x", "prompt": "Engine row completions.", "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/v1/completions",
        body: || serde_json::json!({"model": "x", "prompt": "Engine row completions stream.", "max_tokens": 2, "temperature": 0.0, "stream": true}),
        entries: 1,
    },
    Route {
        uri: "/v1/chat/completions",
        body: || serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "Engine row chat."}], "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/v1/chat/completions",
        body: || serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "Engine row chat stream."}], "max_tokens": 2, "temperature": 0.0, "stream": true}),
        entries: 1,
    },
    Route {
        uri: "/v1/chat/completions/stream",
        body: || serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "Engine row chat stream route."}], "max_tokens": 2, "temperature": 0.0}),
        entries: 1,
    },
    Route {
        uri: "/api/chat",
        body: || serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "Engine row ollama chat."}], "stream": false, "options": {"num_predict": 2, "temperature": 0.0}}),
        entries: 1,
    },
    Route {
        uri: "/api/generate",
        body: || serde_json::json!({"model": "x", "prompt": "Engine row ollama generate.", "stream": false, "options": {"num_predict": 2, "temperature": 0.0}}),
        entries: 1,
    },
];

/// Routes that generate text and are deliberately NOT rows, each with why.
/// A generating route in the router that is in neither list fails
/// [`every_generating_route_is_a_row_or_excused`].
const EXCUSED: &[(&str, &str)] = &[
    ("/batch/tokenize", "tokenization only, no forward"),
    (
        "/api/embeddings",
        "embeddings, not generation — the engine has no embed entry",
    ),
    (
        "/realize/embed",
        "embeddings, not generation — the engine has no embed entry",
    ),
    (
        "/v1/embeddings",
        "embeddings, not generation — the engine has no embed entry",
    ),
    ("/v1/predict", "classical-ML .apr predict, no LLM forward"),
    ("/v1/explain", "classical-ML .apr explain, no LLM forward"),
];

async fn post(app: axum::Router, uri: &str, body: serde_json::Value) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    // Streams are drained whole: the entry is written when generate starts,
    // and the row is judged only after the route has finished.
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test(flavor = "multi_thread")]
async fn every_serve_route_enters_the_one_engine() {
    let model = model_path();
    if !std::path::Path::new(&model).exists() {
        eprintln!("SKIP: {model} is absent");
        return;
    }
    let mapped = Arc::new(MappedGGUFModel::from_path(&model).expect("map the GGUF"));
    let vocab = mapped.model.vocabulary().expect("vocabulary");
    let session = Qwen35Session::load(&mapped, true).expect("load the hybrid");
    let id = session.id();
    let app =
        create_router(AppState::with_qwen35_session(session, mapped, vocab).expect("app state"));

    let mut failures = Vec::new();
    for route in ROUTES {
        let before = entries_of_session(id).len();
        let body = (route.body)();
        let (status, text) = post(app.clone(), route.uri, body.clone()).await;
        let after = entries_of_session(id);
        let added = &after[before.min(after.len())..];
        let label = format!(
            "{} {}",
            route.uri,
            if body["stream"] == true {
                "(stream)"
            } else {
                ""
            }
        );
        if status != StatusCode::OK {
            failures.push(format!("{label}: HTTP {status}: {text}"));
        } else if added.len() != route.entries
            || added
                .iter()
                .any(|e| e.arch != "qwen35" || e.kind != EntryKind::Generate)
        {
            failures.push(format!(
                "{label}: the engine saw {} entries {added:?}, the route carries {} prompt(s) — it decoded outside the session",
                added.len(),
                route.entries
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "#4263: routes that bypass the one engine:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn apr_run_enters_the_one_engine() {
    let model = model_path();
    if !std::path::Path::new(&model).exists() {
        eprintln!("SKIP: {model} is absent");
        return;
    }
    // Ids no other test uses, so the witness answers for this call alone.
    let prompt: Vec<u32> = vec![9_901, 9_902, 9_903, 9_904, 9_905];
    let mut config = crate::infer::InferenceConfig::new(&model);
    config.input_tokens = Some(prompt.clone());
    config.max_tokens = 2;
    config.temperature = 0.0;
    config.top_k = 1;
    config.no_gpu = true;
    crate::infer::run_inference(&config).expect("apr run's inference");
    let entries = entries_for(&prompt);
    assert_eq!(
        entries.len(),
        1,
        "apr run left {entries:?}: it decoded outside the session"
    );
    assert_eq!(entries[0].arch, "qwen35");
    assert_eq!(entries[0].kind, EntryKind::Generate);
}

/// The dense verb rows (#4268 D1, #4293): `apr run` on a dense GGUF enters the
/// engine through `DenseForward`. One model per dense arch that is present on
/// this box; an absent file skips its row, never the others.
const DENSE_RUN_ROWS: &[(&str, &str)] = &[("qwen3", "Qwen3-1.7B-Q4_K_M.gguf")];

#[test]
fn apr_run_on_a_dense_gguf_enters_the_one_engine() {
    for (i, (arch, file)) in DENSE_RUN_ROWS.iter().enumerate() {
        let path = &home_model(file);
        assert!(
            ARCHES.contains(&(*arch, Arch::Session("DenseForward"))),
            "{arch} has a dense verb row but its ARCHES row is not DenseForward"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} is absent");
            continue;
        }
        // Ids no other test uses, so the witness answers for this call alone.
        let base = 9_801 + 10 * i as u32;
        let prompt: Vec<u32> = (base..base + 5).collect();
        let mut config = crate::infer::InferenceConfig::new(path);
        config.input_tokens = Some(prompt.clone());
        config.max_tokens = 2;
        config.temperature = 0.0;
        config.top_k = 1;
        config.no_gpu = true;
        crate::infer::run_inference(&config).expect("apr run's inference");
        let entries = entries_for(&prompt);
        assert_eq!(
            entries.len(),
            1,
            "apr run on {path} left {entries:?}: it decoded outside the session"
        );
        assert_eq!(entries[0].arch, *arch);
        assert_eq!(entries[0].kind, EntryKind::Generate);
    }
}

/// The MoE verb row (PMAT-4269 M1): `apr run --no-gpu` on a qwen3_moe GGUF
/// enters the engine through `Qwen3MoeForward`. Skipped when the file is absent.
fn moe_run_path() -> String {
    home_model("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
}

#[test]
fn apr_run_on_a_moe_gguf_enters_the_one_engine() {
    assert!(
        ARCHES.contains(&("qwen3_moe", Arch::Session("Qwen3MoeForward"))),
        "qwen3_moe has a verb row but its ARCHES row is not Qwen3MoeForward"
    );
    let moe = moe_run_path();
    if !std::path::Path::new(&moe).exists() {
        eprintln!("SKIP: {moe} is absent");
        return;
    }
    // Ids no other test uses, so the witness answers for this call alone.
    let prompt: Vec<u32> = (9_701..9_706).collect();
    let mut config = crate::infer::InferenceConfig::new(&moe);
    config.input_tokens = Some(prompt.clone());
    config.max_tokens = 2;
    config.temperature = 0.0;
    config.top_k = 1;
    config.no_gpu = true;
    crate::infer::run_inference(&config).expect("apr run's inference");
    let entries = entries_for(&prompt);
    assert_eq!(
        entries.len(),
        1,
        "apr run on {moe} left {entries:?}: it decoded outside the session"
    );
    assert_eq!(entries[0].arch, "qwen3_moe");
    assert_eq!(entries[0].kind, EntryKind::Generate);
}

/// Every `impl ArchForward for X` in production source is exactly the set of
/// `Arch::Session` rows — a port cannot land without the table (and so the
/// verb rows) knowing.
/// `impl<S: Q4kStep> ArchForward for X<S>` → `impl ArchForward for X<S>`: a
/// generic impl must not escape the scan below (#4269 M2b, the first one).
fn strip_impl_generics(line: &str) -> std::borrow::Cow<'_, str> {
    let Some(rest) = line.strip_prefix("impl<") else {
        return std::borrow::Cow::Borrowed(line);
    };
    let mut depth = 1usize;
    for (i, c) in rest.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return std::borrow::Cow::Owned(format!("impl {}", rest[i + 1..].trim_start()));
                }
            },
            _ => {},
        }
    }
    std::borrow::Cow::Borrowed(line)
}

#[test]
fn strip_impl_generics_case_table() {
    let cases = [
        (
            "impl<S: Q4kStep> crate::session::ArchForward for AprQ4kForward<S> {",
            "impl crate::session::ArchForward for AprQ4kForward<S> {",
        ),
        (
            "impl<'a, T: Iterator<Item = u32>> ArchForward for X<'a, T> {",
            "impl ArchForward for X<'a, T> {",
        ),
        (
            "impl crate::session::ArchForward for DenseForward<'_> {",
            "impl crate::session::ArchForward for DenseForward<'_> {",
        ),
        ("impl<S> Q4kStep for Y {", "impl Q4kStep for Y {"),
        ("fn impl_thing() {}", "fn impl_thing() {}"),
    ];
    for (input, want) in cases {
        assert_eq!(strip_impl_generics(input), want, "{input}");
    }
}

/// The type named by an `impl ArchForward for <Type>` line, if `line` is one.
fn arch_forward_impl_type(line: &str) -> Option<String> {
    let line = strip_impl_generics(line.trim_start());
    let rest = line
        .as_ref()
        .strip_prefix("impl crate::session::ArchForward for ")
        .or_else(|| line.as_ref().strip_prefix("impl ArchForward for "))?;
    Some(
        rest.chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect(),
    )
}

/// Every non-test `.rs` file under `root`, recursively.
fn non_test_sources(root: std::path::PathBuf) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read src") {
            let path = entry.expect("entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") && !name.contains("test") {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn every_arch_forward_is_a_session_row() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let found: std::collections::BTreeSet<String> = non_test_sources(root)
        .iter()
        .flat_map(|path| {
            let text = std::fs::read_to_string(path).expect("read source");
            text.lines()
                .filter_map(arch_forward_impl_type)
                .collect::<Vec<_>>()
        })
        .collect();
    let rows: std::collections::BTreeSet<String> = ARCHES
        .iter()
        .filter_map(|(_, a)| match a {
            Arch::Session(ty) => Some((*ty).to_string()),
            Arch::NotYet(_) => None,
        })
        .collect();
    assert!(
        !found.is_empty(),
        "the scan found no ArchForward impl at all — vacuous"
    );
    assert_eq!(
        found, rows,
        "#4263: the ArchForward impls in src/ and the Session rows of ARCHES differ — \
         a ported architecture must flip its row (and get verb rows)"
    );
}

/// Every POST route in the router that is not a row is excused by name, so a
/// new generating route cannot land outside the guard.
#[test]
fn every_generating_route_is_a_row_or_excused() {
    let router = include_str!("router.rs");
    let generating = [
        "generate",
        "completions",
        "chat",
        "batch",
        "predict",
        "explain",
        "embed",
    ];
    let mut missing = Vec::new();
    for line in router.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("(\"POST\", \"") else {
            continue;
        };
        let uri = rest.split('"').next().unwrap_or("");
        if !generating.iter().any(|g| uri.contains(g)) {
            continue;
        }
        let covered = ROUTES.iter().any(|r| r.uri == uri) || EXCUSED.iter().any(|(u, _)| *u == uri);
        if !covered {
            missing.push(uri.to_string());
        }
    }
    // The multi-line entries (`"/v1/chat/completions"` on its own line).
    for uri in ["/v1/chat/completions", "/v1/chat/completions/stream"] {
        assert!(
            router.contains(&format!("\"{uri}\"")),
            "{uri} left the router"
        );
        assert!(ROUTES.iter().any(|r| r.uri == uri), "{uri} has no row");
    }
    assert!(
        missing.is_empty(),
        "#4263: generating routes with no engine row and no excuse: {missing:?}"
    );
}
