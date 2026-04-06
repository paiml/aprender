//! Browser UI handlers — serve the embedded chat UI.
//!
//! Two modes:
//! - `/` — Zero-JS HTML form that POSTs to `/ui/chat` (SSR)
//! - `/ui/chat` — Server-side rendering: accepts form POST, returns full page with response

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    Form,
};
use serde::Deserialize;

use super::state::BancoState;

/// GET / — serve the zero-JS chat UI.
pub async fn index_handler(State(state): State<BancoState>) -> Html<String> {
    Html(render_chat_page(&state, None, None))
}

/// Form data from the chat HTML form.
#[derive(Deserialize)]
pub struct ChatFormData {
    pub message: String,
}

/// POST /ui/chat — server-rendered chat (zero JavaScript).
pub async fn chat_form_handler(
    State(state): State<BancoState>,
    Form(form): Form<ChatFormData>,
) -> Html<String> {
    let prompt = form.message.trim().to_string();
    if prompt.is_empty() {
        return Html(render_chat_page(&state, None, None));
    }

    // Call the chat completion API internally
    let request = super::types::BancoChatRequest {
        model: None,
        messages: vec![crate::serve::templates::ChatMessage::user(&prompt)],
        max_tokens: 256,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        conversation_id: None,
        response_format: None,
        rag: false,
        rag_config: None,
        attachments: vec![],
        tools: None,
        tool_choice: None,
    };

    let response = super::handlers_inference::try_inference(&state, &request)
        .map(|(text, _, _)| text)
        .unwrap_or_else(|| {
            if state.model.is_loaded() {
                "Model loaded but inference unavailable. Build with --features banco.".to_string()
            } else {
                "No model loaded. Use: POST /api/v1/models/load {\"model\": \"./model.gguf\"}"
                    .to_string()
            }
        });

    Html(render_chat_page(&state, Some(&prompt), Some(&response)))
}

/// Render the full chat page HTML (zero JavaScript).
fn render_chat_page(state: &BancoState, prompt: Option<&str>, response: Option<&str>) -> String {
    let model_status = if let Some(info) = state.model.info() {
        format!("<span style='color:#4caf50'>&#9679;</span> {}", info.model_id)
    } else {
        "<span style='color:#f44336'>&#9679;</span> No model".to_string()
    };

    let messages_html = match (prompt, response) {
        (Some(p), Some(r)) => format!(
            "<div class='msg user'>{}</div><div class='msg assistant'>{}</div>",
            html_escape(p),
            html_escape(r)
        ),
        _ => "<div class='msg system'>Send a message to start chatting.</div>".to_string(),
    };

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banco — Local AI Workbench</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{--bg:#1a1a2e;--surface:#16213e;--border:#0f3460;--accent:#e94560;--text:#e0e0e0;--dim:#888}}
body{{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}}
header{{background:var(--surface);padding:12px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}}
header h1{{font-size:16px;font-weight:600;color:var(--accent)}}
.model-info{{font-size:12px;color:var(--dim)}}
.messages{{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:8px}}
.msg{{padding:10px 14px;border-radius:8px;max-width:80%;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}}
.msg.user{{align-self:flex-end;background:var(--accent);color:white}}
.msg.assistant{{align-self:flex-start;background:var(--surface);border:1px solid var(--border)}}
.msg.system{{align-self:center;font-size:12px;color:var(--dim);font-style:italic}}
.chat-form{{display:flex;gap:8px;padding:12px 20px;background:var(--surface);border-top:1px solid var(--border)}}
.chat-form input[type=text]{{flex:1;padding:10px 14px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-size:14px}}
.chat-form button{{padding:10px 20px;background:var(--accent);color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px}}
.chat-form button:hover{{opacity:0.9}}
footer{{padding:8px 20px;font-size:11px;color:var(--dim);background:var(--surface);border-top:1px solid var(--border);text-align:center}}
footer a{{color:var(--accent);text-decoration:none}}
</style>
</head>
<body>
<header>
  <h1>Banco</h1>
  <span class="model-info">{model_status}</span>
</header>
<div class="messages">{messages_html}</div>
<form class="chat-form" method="POST" action="/ui/chat">
  <input name="message" type="text" placeholder="Type a message..." autocomplete="off" autofocus>
  <button type="submit">Send</button>
</form>
<footer>
  Zero-JS UI &middot; API: <a href="/api/v1/system">/api/v1/system</a> &middot;
  Chat: POST <a href="/api/v1/chat/completions">/api/v1/chat/completions</a>
</footer>
</body>
</html>"##,
        model_status = model_status,
        messages_html = messages_html
    )
}

/// Minimal HTML escaping for user content.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// GET /assets/* — serve static assets (CSS/JS/WASM).
/// Currently returns 404 — scaffold for presentar WASM bundles.
pub async fn assets_handler(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    let _ = path;
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/plain")],
        "Asset not found — UI is self-contained in index.html",
    )
}
