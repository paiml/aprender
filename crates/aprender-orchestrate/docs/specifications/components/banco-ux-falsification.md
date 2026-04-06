# Banco UX Falsification Report

> Methodology: Attempt to break every claim in banco-ux.md against what actually ships.
> Date: 2026-03-21
> Verdict: **FAIL — massive gap between spec and reality**

---

## Executive Summary

The UX spec promises a **sovereign AI studio with 7 screens, 80+ widgets, dual-surface (WASM + TUI), YAML-driven layout, WebGPU rendering, theming, accessibility, and presentar integration**.

What actually ships: **a minified chat bubble — 41 lines of inline JavaScript, no presentar, no TUI, no screens beyond chat, no widgets, no theming, no accessibility, no keyboard navigation beyond Enter.**

This is not an AI studio. This is a chat window.

---

## Claim-by-Claim Falsification

### "Dual-Surface Architecture (WASM + TUI)"

| Claim | Reality | Verdict |
|-------|---------|---------|
| WASM UI via presentar | 41 lines inline JavaScript, zero presentar | **FAIL** |
| TUI via presentar-terminal | Not implemented. No TUI binary. | **FAIL** |
| `batuta banco --tui` command | Does not exist | **FAIL** |
| "Both are first-class" | Neither exists. JS chat bubble is a scaffold. | **FAIL** |

### 7 Screens: Chat, Arena, Models, Data, Training, Experiments, System

| Screen | Spec Widget Count | What Actually Exists | Verdict |
|--------|------------------|---------------------|---------|
| **Chat** | 8 widgets (list, input, streaming, model selector, temp slider, token counter, system prompt, conversation list) | Text input + send button + message list. No model selector, no token counter, no system prompt editor. | **PARTIAL** |
| **Arena** | 4 widgets (side-by-side, latency chart, model labels, winner vote) | Does not exist | **FAIL** |
| **Models** | 4 widgets (data table, load progress, VRAM gauge, status card) | Model name shown in sidebar dot. No table, no progress, no VRAM. | **FAIL** |
| **Data** | 5 widgets (file upload, file list, recipe steps, dataset preview, recipe progress) | Upload button (file dialog). No file list, no recipe UI, no dataset preview. | **FAIL** |
| **Training** | 8 widgets (loss curve, GPU gauge, grad norm, LR schedule, config form, preset selector, run status, metrics table) | Not in UI at all. API-only. | **FAIL** |
| **Experiments** | 5 widgets (run list, loss comparison, hyperparam diff, eval metrics, export button) | Not in UI at all. API-only. | **FAIL** |
| **System** | 7 widgets (CPU, memory, GPU, network, request log, circuit breaker, uptime) | System info shown as one-line text message on load. | **FAIL** |

**Score: 1/7 screens partially implemented. 0/7 fully implemented.**

### Presentar Integration

| Claim | Reality | Verdict |
|-------|---------|---------|
| presentar-widgets for all UI | Zero presentar widgets used | **FAIL** |
| presentar-layout for layout engine | Inline CSS flexbox, not presentar | **FAIL** |
| presentar YAML scenes (`banco-wasm.prs`) | No `.prs` files exist | **FAIL** |
| presentar DataStream + StreamSubscription | Raw JS WebSocket, not presentar | **FAIL** |
| presentar theme system (WCAG AA) | Hardcoded CSS variables, no theme file | **FAIL** |
| presentar AccessibilityTree | Zero ARIA roles, zero keyboard nav beyond Enter | **FAIL** |
| presentar FormValidator | No form validation | **FAIL** |

### Zero-JavaScript Policy

| Claim | Reality | Verdict |
|-------|---------|---------|
| "All UI is Rust" | 41 lines of inline JavaScript | **FAIL** |
| No `<script>` tags | One `<script>` tag with 41 lines | **FAIL** |
| presentar compiled to wasm32 | No WASM binary exists | **FAIL** |
| cargo build --target wasm32 | Never invoked for banco | **FAIL** |

### TUI Features

| Claim | Reality | Verdict |
|-------|---------|---------|
| MockTty + FrameAssertion testing | Zero TUI tests | **FAIL** |
| Keybindings (Tab, 1-7, q, ?, /) | No TUI exists | **FAIL** |
| Panel-based grid layout | No TUI exists | **FAIL** |
| ComputeBlock layout | No TUI exists | **FAIL** |
| 60fps frame budget | No TUI exists | **FAIL** |

### Real-Time Data Flow

| Claim | Reality | Verdict |
|-------|---------|---------|
| WebSocket StreamMessage protocol | JS WebSocket connects, receives events | **PARTIAL** |
| Subscribe to training metrics | Events received but no visualization | **PARTIAL** |
| SSE streaming for TUI | No TUI exists | **FAIL** |

### Theming

| Claim | Reality | Verdict |
|-------|---------|---------|
| `~/.banco/theme.yaml` | Does not exist | **FAIL** |
| WCAG AA verified colors | Hardcoded dark theme, not verified | **FAIL** |
| Light/dark/solarized/monokai | Single hardcoded dark theme | **FAIL** |

### Accessibility

| Claim | Reality | Verdict |
|-------|---------|---------|
| ARIA roles on all interactive elements | Zero ARIA roles in HTML | **FAIL** |
| Keyboard navigation for every action | Only Enter to send. No Tab cycling, no shortcuts. | **FAIL** |
| Screen reader announcements | Zero LiveRegion elements | **FAIL** |
| Focus management across tabs | No tabs exist | **FAIL** |

---

## What Actually Works in the Browser

1. Page loads at `/` (200 OK, HTML served)
2. Dark-themed chat interface renders
3. Model status dot shows green/red
4. Text input + Send button (Enter key works)
5. Messages display in bubble format
6. Temperature + max_tokens sliders exist
7. RAG toggle button toggles class
8. File upload button opens file dialog
9. WebSocket status indicator shows connected/disconnected
10. Conversation sidebar lists conversations
11. New Chat / Refresh buttons work

**That's it.** It's a chat window with a few controls. Not an AI studio.

---

## Root Cause

The spec was written as a **vision document** but treated as a **status report**. Features were marked as "Phase 2/3/4" in the widget mapping but the implementation only built the API layer (82 endpoints, 353 tests). The UI layer was never built beyond the inline JS scaffold.

The 353 L1 unit tests all pass because they test the HTTP API via `tower::oneshot()`. None of them test the UI. The probar L4 browser tests that would have caught "7 screens don't exist" were never written.

---

## Honest Status

| Component | Spec Claims | Actually Exists | Gap |
|-----------|-------------|-----------------|-----|
| API (HTTP) | 82 endpoints | 82 endpoints | **None** — API is complete |
| Browser UI | 7 screens, 80+ widgets, presentar WASM | 1 chat screen, inline JS | **Massive** |
| TUI | Full dashboard, keybindings, 60fps | Nothing | **Total** |
| Theming | YAML config, 4 palettes, WCAG AA | Hardcoded CSS | **Total** |
| Accessibility | Full ARIA, keyboard nav, screen reader | Zero | **Total** |
| Testing (L4) | probar CDP browser tests | Zero | **Total** |

---

## What the Spec Should Say

The banco-ux.md should be rewritten to honestly separate:
1. **What exists now** (chat scaffold)
2. **What is planned** (presentar AI studio)
3. **What the gap is** (everything except chat)

The API is genuinely impressive (82 endpoints, 10 stack crates, 4 protocols). The UI is not.
