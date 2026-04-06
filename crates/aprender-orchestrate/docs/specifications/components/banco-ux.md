# Banco UX: Dual-Surface Architecture (WASM + TUI)

> Parent: [banco-spec.md](banco-spec.md) §5
> Depends on: presentar (0.3.x), presentar-terminal (0.3.x)

---

## Principle

Every Banco feature has two surfaces: a browser UI (presentar WASM) and a terminal UI (presentar-terminal TUI). Both are first-class. Neither is a degraded fallback. The API server (Phases 1-3) is the single source of truth — both surfaces are pure clients.

### Zero-JavaScript UI Constraint

**All UI is Rust.** The browser surface uses presentar compiled to `wasm32-unknown-unknown`. The terminal surface uses `presentar-terminal`. No JavaScript, no npm, no Node.js anywhere in the UI stack.

```
❌ FORBIDDEN                              ✅ REQUIRED
────────────────────────────────────────────────────────────
• React / Vue / Svelte                    • presentar-widgets (Rust → WASM)
• JavaScript bundlers (webpack, vite)     • presentar-cli bundle (Rust → .wasm)
• Inline <script> tags                    • presentar event handlers (Rust)
• npm / node_modules                      • cargo build --target wasm32
• CSS-in-JS                               • presentar-layout (Rust layout engine)
```

**Current state:** `src/serve/banco/ui.rs` contains a 41-line inline JavaScript scaffold. This is **tech debt** that MUST be replaced by a presentar WASM widget. See banco-spec.md §Zero-JS Policy for the replacement plan.

**Testing:** All browser testing via probar (jugar-probar) — a Playwright replacement in pure Rust using Chrome DevTools Protocol. Zero JavaScript test runners. See [banco-testing.md](banco-testing.md).

```
┌──────────────────────────────────┐
│  batuta serve --banco --port 8090│
│  (axum HTTP + SSE + WebSocket)   │
├──────────┬───────────────────────┤
│  WASM UI │  TUI                  │
│  Browser │  Terminal             │
│  :8090/  │  batuta banco --tui   │
├──────────┴───────────────────────┤
│  Same API, same state, same data │
└──────────────────────────────────┘
```

## Feature → Widget Mapping

### Chat (Phase 2)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Message list | `List` + `Text` | `Table` rows with role column |
| User input | `TextInput` (multiline) | `TextInput` with validation |
| Streaming response | `Text` (append on SSE) | `Text` (append on SSE) |
| Model selector | `Select` dropdown | `Select` |
| Temperature/top_p | `Slider` | `Slider` |
| Token counter | `ProgressBar` (context window %) | `SegmentedMeter` |
| System prompt | `TextInput` (expandable) | `TextInput` |
| Conversation list | `List` (sidebar) | `List` with selection |

### Arena (Phase 2)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Side-by-side responses | Two `Column` panels | Two `FlexCell` panels |
| Latency comparison | `BarChart` (2 bars) | `BarChart` |
| Model labels | `Text` + `DataCard` | `TitleBar` + `InfoDense` |
| Winner indicator | `Button` (vote) | `HealthStatus` |

### Model Management (Phase 2)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Model list | `DataTable` (id, format, size, quant) | `Table` sortable |
| Load progress | `ProgressBar` (determinate) | `ProgressBar` |
| VRAM usage | `Gauge` (circular) | `Gauge` (linear) + `MemoryBar` |
| Model status | `DataCard` with `HealthStatus` | `InfoDense` + `HealthStatus` |

### Training (Phase 3)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Loss curve | `Chart` (Line, EMA smoothed) | `LineChart` + `LossCurve` |
| GPU utilization | `Gauge` + `Chart` (Area) | `Gauge` + `Sparkline` |
| Grad norm | `Chart` (Line) | `Sparkline` |
| Learning rate schedule | `Chart` (Line) | `LineChart` |
| Training config | Form (`TextInput`, `Select`, `Slider`) | Form widgets |
| Preset selector | `Select` | `Select` |
| Run status | `ProgressBar` + `Text` (ETA) | `ProgressBar` + `format_duration` |
| Metrics table | `DataTable` (step, loss, lr, GPU) | `ProcessDataFrame` |

### Data Recipes (Phase 3)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| File upload | `Button` + drag-drop | `FilePanel` |
| File list | `DataTable` | `Table` |
| Recipe steps | `List` (reorderable) | `List` |
| Dataset preview | `DataTable` (rows + columns) | `Table` with scroll |
| Recipe progress | `ProgressBar` per step | `SegmentedMeter` |

### Experiment Tracking (Phase 3)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Run list | `DataTable` (sortable by loss, time) | `Table` sortable |
| Loss comparison | `Chart` (multi-line overlay) | `LineChart` multi-series |
| Hyperparams diff | `DataTable` (highlight diffs) | `Table` with diff colors |
| Eval metrics | `DataCard` (PPL before/after) | `DataCard` |
| Export button | `Button` + `Select` (format) | `Menu` |

### Model Evaluation (Phase 3)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Perplexity result | `DataCard` | `DataCard` |
| Confusion matrix | `Chart` (Heatmap) | `ConfusionMatrix` |
| ROC/PR curves | `Chart` (Line) | `ROC/PR curve` widget |
| Feature importance | `Chart` (Horizontal Bar) | `FeatureImportance` |

### RAG (Phase 3)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Source citations | `List` with score bars | `Table` (file, chunk, score) |
| Index stats | `DataCard` | `InfoDense` |
| Document chunks | `DataTable` | `Table` with scroll |

### System Monitoring (Phase 2+)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| CPU usage | `Chart` (Area) | `CpuGrid` |
| Memory | `Gauge` | `MemoryBar` |
| GPU | `Gauge` + `Chart` | `GpuPanel` |
| Network | `Chart` (Line, bytes/sec) | `NetworkPanel` |
| Request log | `DataTable` (live) | `Table` (tail) |
| Circuit breaker | `HealthStatus` | `HealthStatus` |
| Uptime / version | `Text` | `InfoDense` |

### Agent Mode (Phase 4)

| Feature | WASM Widget | TUI Widget |
|---------|-------------|------------|
| Step stream | `List` (plan/tool/reasoning/answer) | `Table` with type column |
| Tool calls | `DataCard` (tool name, input, output) | `Collapsible` panels |
| Memory viewer | `Tree` (hierarchical) | `Tree` |
| Plan display | `Text` (markdown) | `Text` |

---

## Screen Layouts

### WASM: Tab-Based SPA

```yaml
# Presentar scene: banco-wasm.prs
layout:
  type: flex
  direction: column
children:
  - widget: tabs
    id: main-nav
    items: [Chat, Arena, Models, Data, Training, Experiments, System]
  - widget: container
    id: content
    flex: 1
    bind: main-nav.selected
```

Tabs map to Phase 2-4 endpoint groups. Each tab is a sub-scene loaded on demand.

### TUI: Panel-Based Dashboard

```yaml
# Presentar scene: banco-tui.prs
layout:
  type: grid
  columns: [1fr, 2fr]
  rows: [auto, 1fr, auto]
panels:
  - id: sidebar
    position: [0, 0, 1, 3]
    content: [model-status, nav-menu]
  - id: main
    position: [1, 0, 1, 2]
    content: [chat | training | data]  # switches on nav
  - id: status-bar
    position: [0, 2, 2, 1]
    content: [health, uptime, gpu, circuit-breaker]
```

Uses presentar-terminal's `ComputeBlock` layout with grid snapping.

### TUI: Keybindings

| Key | Action |
|-----|--------|
| `Tab` | Cycle panels |
| `1`-`7` | Jump to screen (Chat, Arena, Models, Data, Training, Experiments, System) |
| `q` | Quit |
| `?` | Help overlay |
| `/` | Search / filter |
| `Enter` | Submit input / select |
| `Esc` | Cancel / back |
| `F5` | Refresh data |

---

## Real-Time Data Flow

Both surfaces use the same streaming protocol.

### WASM

```
Browser → WebSocket /api/v1/ws → StreamMessage protocol
  - Subscribe to training metrics
  - Subscribe to system stats
  - Receive model load/unload events
```

Uses presentar's `DataStream` + `StreamSubscription` + `ReconnectConfig` with exponential backoff.

### TUI

```
Terminal → HTTP SSE /api/v1/train/runs/{id}/metrics
Terminal → HTTP poll /health (1s interval)
Terminal → HTTP poll /api/v1/system (5s interval)
```

TUI uses SSE for training metrics (same as WASM) and polling for slower-changing data. The ptop async collector pattern applies.

---

## Theming

Both surfaces share a theme definition:

```yaml
# ~/.banco/theme.yaml
palette: dark    # or: light, solarized, monokai
primary: "#6C63FF"
surface: "#1E1E2E"
error: "#F44336"
warning: "#FF9800"
success: "#4CAF50"
```

- WASM: `ColorPalette` applied via presentar's theme system, WCAG AA verified
- TUI: Mapped to terminal 256-color palette with `Color` lerp approximation

---

## Validation

Forms in both surfaces use presentar's validation system:

```rust
FormValidator::new()
    .field("temperature", vec![Range(0.0, 2.0)])
    .field("max_tokens", vec![Required, Range(1.0, 32768.0)])
    .field("model_path", vec![Required, Pattern(r".*\.(gguf|apr|safetensors)$")])
```

Validation runs client-side (instant feedback) with server-side double-check on submit.

---

## Accessibility

WASM surface provides full accessibility via presentar's `AccessibilityTree`:
- All interactive elements have ARIA roles (Button, TextBox, List, etc.)
- Keyboard navigation for every action
- Screen reader announcements via `LiveRegion` for streaming content
- Focus management across tab transitions

TUI is inherently keyboard-accessible. Terminal screen readers (e.g., BRLTTY) work via crossterm's raw output.

---

## Feature Flag Matrix

| Feature | `banco` | `banco` + `tui` | `banco` + `banco-ui` |
|---------|---------|-----------------|---------------------|
| HTTP API | Yes | Yes | Yes |
| TUI dashboard | No | Yes | Yes |
| WASM browser UI | No | No | Yes |
| Training metrics TUI | No | Yes (SSE) | Yes (WS) |
| Chart rendering | No | Terminal charts | WebGPU/Canvas2D |

```toml
# Cargo.toml [features]
banco-tui = ["banco", "presentar-terminal", "crossterm"]
banco-ui = ["banco-tui", "presentar", "wasm-bindgen"]
```

---

## CLI Entry Points

```bash
# API only (headless)
batuta serve --banco --port 8090

# API + TUI dashboard
batuta serve --banco --tui --port 8090

# API + browser UI
batuta serve --banco --port 8090
# (browser UI always served when banco-ui feature is compiled in)
```

TUI connects to the same API server (localhost). It's a client, not a separate mode.

---

## Implementation Order

1. **Phase 2**: Chat TUI (TextInput + Table + streaming) — minimal viable TUI
2. **Phase 2**: Model management TUI (Table + ProgressBar + Gauge)
3. **Phase 3**: Training metrics TUI (LossCurve + Sparkline + Gauge)
4. **Phase 3**: Data/experiment TUI (Table + DataCard)
5. **Phase 4**: WASM SPA scaffold (Tabs + routing + theme)
6. **Phase 4**: WASM chat (full widget set)
7. **Phase 4**: WASM training dashboard (Chart + streaming)
8. **Phase 4**: WASM data management (upload + DataTable)
