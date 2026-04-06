//! Browser UI — zero-JavaScript, server-rendered.
//!
//! The chat UI is rendered server-side in `handlers_ui.rs` using pure HTML forms.
//! No JavaScript, no WASM (yet). The `<form method="POST" action="/ui/chat">`
//! pattern handles chat submission via standard HTTP.
//!
//! Previously this module contained 41 lines of inline JavaScript (INDEX_HTML).
//! That was eliminated in PMAT-125 (zero-JS policy compliance).
//! Full presentar WASM widget is planned for Phase 5b.
