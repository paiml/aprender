# SPEC-TRAINMON: Design-by-Provable-Contract Training Monitoring

Version: 1.0
Status: proposed
Date: 2026-04-10

**Document ID:** SPEC-TRAINMON-001
**Version:** 1.0.0
**Status:** PROPOSED
**Author:** PAIML Engineering
**Date:** 2026-04-10
**Priority:** P0 — v29 running now with no live dashboard
**Parent:** SPEC-SHIP-TWO-001 (monitoring required for both models)
**Crate:** `crates/aprender-train/src/monitor/`
**Contracts:** `contracts/aprender/training-monitor-v1.yaml`
**Citations:**
- [C1] Li et al. (2018) "Visualizing the Loss Landscape" arXiv:1712.09913
- [C2] Pascanu et al. (2013) "Training Recurrent Neural Networks" arXiv:1211.5063
- [C3] Hoffmann et al. (2022) "Training Compute-Optimal LLMs" arXiv:2203.15556
- [C4] Muennighoff et al. (2023) "Scaling Data-Constrained LLMs" arXiv:2305.16264
- [C5] Zhang et al. (2022) "OPT Training Logbook" arXiv:2205.01068
- [C6] Toneva et al. (2019) "Forgetting Events During DNN Learning" arXiv:1812.05159
- [C7] Zaharia et al. (2018) "Accelerating the ML Lifecycle with MLflow" IEEE Data Eng.

---

## 1. Abstract

A dual TUI+WASM training monitor with mathematical layout precision, provable
contracts on every widget boundary, and loss spike detection as a differentiator.

**Key finding from prior art research:** 0/6 major frameworks (Burn, Unsloth,
TensorBoard, PyTorch, H2O, MLflow) have native loss spike detection with automatic
response. All are retrospective — you look at charts after something went wrong.
This spec makes anomaly detection a first-class, contract-enforced feature.

**Key design from rmedia:** Integer-grid layout (fixed columns × rows, constant
cell size) produces pixel-deterministic output testable via snapshot. No constraint
solver needed.

**Existing primitives:** Presentar has 62 terminal widgets (including `LossCurve`,
`Sparkline`, `GpuPanel`, `Gauge`), dual TUI+WASM backends, `RecordingCanvas` for
snapshot testing, and `#[interface_test]` macros. The `DashboardSource` trait and
`TrainingDashboard` widget already exist in entrenar. This spec composes them under
contract.

---

## 2. Five Whys

1. Why can't we use wandb/TensorBoard? → Python, cloud accounts, browsers. We train
   on SSH GPU boxes with the sovereign Rust stack.
2. Why not just use existing `apr monitor`? → No provable layout contract. Widgets
   manually placed, untestable across TUI/WASM.
3. Why does layout matter? → Dual target means same dashboard must render correctly
   in two backends. Without constraints, visual regressions are invisible.
4. Why not just test visually? → rmedia proved integer-grid layouts produce
   deterministic output testable via snapshot comparison.
5. Why now? → v29 running with no live dashboard. SHIP-TWO needs monitoring.

---

## 3. Prior Art Feature Matrix

| Feature | Burn | Unsloth | TensorBoard | PyTorch | H2O | MLflow | **Ours** |
|---------|------|---------|-------------|---------|-----|--------|----------|
| Loss curve | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| Throughput | Yes | No | Profiler | Profiler | No | No | **Yes** |
| GPU monitoring | Yes | Yes | Yes | Yes | smi | No | **Yes** |
| Gradient norms | No | Yes | Yes | Yes | No | No | **Yes** |
| Loss spike detection | No | No | No | No | No | No | **YES** |
| TUI dashboard | **Yes** | No | No | tqdm | No | No | **Yes** |
| WASM dashboard | Inference | No | No | No | No | No | **Yes** |
| Provable layout | No | No | No | No | No | No | **YES** |
| Snapshot tests | No | No | No | No | No | No | **YES** |
| JSON agent output | No | No | No | No | No | No | **Yes** |

---

## 4. Layout Specification (rmedia-style Integer Grid)

### 4.1 Grid Protocol

8 columns × 6 rows. Cell size determined by terminal width ÷ 8.
All positions are integer grid coordinates — zero floating-point layout math.

```
Col:  0    1    2    3    4    5    6    7
Row 0: [  HEADER: model │ step │ elapsed │ ETA   ]
Row 1: [  Loss Curve     │ Throughput  │ GPU VRAM ]
Row 2: [  (EMA + raw)    │ tok/s spark │ GPU Util ]
Row 3: [  (val overlay)  │ MFU gauge   │ GPU Temp ]
Row 4: [  LR Schedule    │ Config / Hyperparameters ]
Row 5: [  FOOTER: alerts │ checkpoint │ ZClip     ]
```

### 4.2 Cell Assignment (Provable)

| Region | Columns | Rows | Widget | Contract |
|--------|---------|------|--------|----------|
| header | 0..8 | 0 | `TitleBar` | F-TM-LAYOUT-001 |
| loss | 0..3 | 1..4 | `LossCurve` | F-TM-LAYOUT-002 |
| throughput | 3..5 | 1..2 | `Sparkline` | F-TM-LAYOUT-003 |
| mfu | 3..5 | 3..4 | `Gauge` | F-TM-LAYOUT-004 |
| gradient | 3..5 | 2..3 | `Sparkline` | F-TM-LAYOUT-005 |
| gpu_vram | 5..8 | 1..2 | `MemoryBar` | F-TM-LAYOUT-006 |
| gpu_util | 5..8 | 2..3 | `Gauge` | F-TM-LAYOUT-007 |
| gpu_temp | 5..8 | 3..4 | `Meter` | F-TM-LAYOUT-008 |
| lr_schedule | 0..3 | 4..5 | `Sparkline` | F-TM-LAYOUT-009 |
| config | 3..8 | 4..5 | `Text` | F-TM-LAYOUT-010 |
| footer | 0..8 | 5 | `Text` (alerts) | F-TM-LAYOUT-011 |

### 4.3 Layout Invariants

- **No cell overlap:** Each (col, row) owned by exactly one region.
  Enforced by `HashSet<(u8, u8)>` occupancy tracking (rmedia pattern).
- **No gaps:** All 48 cells (8×6) assigned. Verified at compile time.
- **Deterministic sizing:** Cell pixel width = `terminal_width / 8`.
  Cell pixel height = `terminal_height / 6`. Integer division, no rounding error.
- **WASM parity:** Same grid renders in Canvas2D with `canvas_width / 8` cell size.
  Pixel-identical layout between TUI and WASM (modulo font rendering).

---

## 5. Data Contract: `DashboardSource` Trait

Already exists at `crates/aprender-train/src/dashboard/source.rs`. Formalize:

```rust
pub trait DashboardSource: Send + Sync {
    fn status(&self) -> TrainingStatus;
    fn recent_metrics(&self, limit: usize) -> Vec<MetricSnapshot>;
    fn subscribe(&self, callback: Box<dyn Fn(MetricSnapshot) + Send>) -> SubscriptionId;
    fn resource_usage(&self) -> ResourceUsage;
}
```

### 5.1 `MetricSnapshot` Fields (Contract)

| Field | Type | Unit | Required | Source |
|-------|------|------|----------|--------|
| step | u64 | steps | YES | Training loop |
| loss | f32 | nats | YES | Forward pass |
| loss_ema | f32 | nats | YES | Welford / EMA(α=0.99) |
| lr | f32 | — | YES | Scheduler |
| throughput | f32 | tok/s | YES | Batch timer |
| mfu | f32 | % | YES | FLOPs / theoretical peak |
| gradient_norm | f32 | L2 | YES | Gradient clipper |
| gpu_vram_used | u64 | bytes | YES | CUDA meminfo |
| gpu_vram_total | u64 | bytes | YES | CUDA meminfo |
| gpu_utilization | f32 | % | YES | NVML |
| gpu_temperature | f32 | °C | YES | NVML |
| val_ppl | Option<f32> | — | NO | Eval loop (periodic) |
| zclip_events | u32 | count | YES | ZClip tracker |
| elapsed_secs | f64 | seconds | YES | Wall clock |
| tokens_seen | u64 | tokens | YES | Batch counter |

---

## 6. Anomaly Detection (Differentiator)

### 6.1 Loss Spike Detection

```
spike(t) := loss(t) > 2.0 × ema_loss(t) AND t > warmup_steps
```

When detected:
1. Footer alert bar turns red: `[SPIKE] loss=X.XX at step N (2.1× EMA)`
2. JSON event emitted: `{"event": "loss_spike", "step": N, "loss": X, "ema": Y}`
3. If `--auto-checkpoint` enabled: save emergency checkpoint before potential divergence

### 6.2 NaN/Inf Detection

```
nan_detected(t) := is_nan(loss(t)) OR is_inf(loss(t))
```

When detected:
1. Footer alert: `[NaN] training diverged at step N`
2. If `--auto-rollback` enabled: restore last checkpoint + reduce LR by 10×

### 6.3 Gradient Explosion

```
exploding(t) := gradient_norm(t) > 100 × ema_gradient_norm(t)
```

Alerts via footer + JSON event.

---

## 7. Rendering Modes

| Mode | Backend | Command | Use Case |
|------|---------|---------|----------|
| TUI | presentar-terminal | `apr monitor <dir>` | SSH into GPU box |
| WASM | presentar (Canvas2D) | Browser at `localhost:8091` | Remote monitoring |
| JSON | stdout | `apr monitor <dir> --json` | CI, LLM agents, piping |
| Text | stdout | `apr monitor <dir> --format text` | Logs, non-interactive |

All modes consume the same `DashboardSource` data. Layout grid is identical
for TUI and WASM. JSON mode emits one `MetricSnapshot` per line (JSONL).

---

## 8. Acceptance Criteria

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| AC-TM-001 | TUI renders 11 regions without overlap | All 48 cells assigned | `#[interface_test]` snapshot |
| AC-TM-002 | WASM renders identical grid layout | Pixel-comparison TUI vs WASM | Canvas snapshot |
| AC-TM-003 | Loss curve updates at ≤500ms refresh | Timestamp delta between frames | Timer assertion |
| AC-TM-004 | Loss spike detection fires on 2× EMA | Inject synthetic spike | Unit test |
| AC-TM-005 | NaN detection fires on NaN loss | Inject NaN | Unit test |
| AC-TM-006 | JSON mode emits valid JSONL | Parse each line as JSON | Integration test |
| AC-TM-007 | `apr monitor` attaches to running v29 | Live test | Manual verification |
| AC-TM-008 | GPU metrics read from NVML | VRAM, util, temp present | Integration test |
| AC-TM-009 | Throughput matches training log | tok/s within 1% of stderr | Comparison test |
| AC-TM-010 | Grid occupancy = 48/48 cells | Compile-time assertion | `const_assert!` |

---

## 9. Falsification Tests

| ID | Hypothesis Falsified If... | Mitigation |
|----|---------------------------|------------|
| FALSIFY-TM-001 | Any two regions share a cell | Fix cell assignment in grid protocol |
| FALSIFY-TM-002 | TUI layout differs from WASM layout | Sync grid constants between backends |
| FALSIFY-TM-003 | Loss spike not detected when loss > 2× EMA | Fix spike detection threshold |
| FALSIFY-TM-004 | NaN loss does not trigger alert | Fix NaN check in metric ingestion |
| FALSIFY-TM-005 | JSON output is not valid JSONL | Fix serialization; add `serde_json::to_string` test |
| FALSIFY-TM-006 | Refresh rate > 1s in TUI mode | Profile render loop; optimize widget draw |
| FALSIFY-TM-007 | GPU metrics return 0 when GPU active | Fix NVML bindings; check permission |
| FALSIFY-TM-008 | Monitor crashes when training finishes | Handle EOF on metrics stream gracefully |
| FALSIFY-TM-009 | `MetricSnapshot` missing required field | `#[requires]` on DashboardSource methods |
| FALSIFY-TM-010 | WASM monitor fails to connect | WebSocket fallback to polling |

---

## 10. Implementation Plan

### Phase 1: Grid Protocol + Data Contract (2h)

Create `crates/aprender-train/src/monitor/grid.rs`:
- `MonitorGrid` struct with 8×6 occupancy `HashSet<(u8, u8)>`
- `Region` enum (Header, Loss, Throughput, Mfu, Gradient, GpuVram, GpuUtil, GpuTemp, LrSchedule, Config, Footer)
- `assign_region()` with overlap check
- `const GRID_COLS: u8 = 8; const GRID_ROWS: u8 = 6;`
- Compile-time assertion: all 48 cells assigned

Formalize `MetricSnapshot` struct with `#[contract]` on all required fields.

### Phase 2: TUI Dashboard Composition (3h)

Wire presentar-terminal widgets into grid regions:
- `LossCurve` for loss region (EMA + raw dual series)
- `Sparkline` for throughput + gradient norm
- `Gauge` for MFU
- `MemoryBar` / `Gauge` / `Meter` for GPU panel
- `Sparkline` for LR schedule
- `Text` for config + footer alerts
- `TitleBar` for header

### Phase 3: Anomaly Detection (2h)

- Loss spike: `loss > 2.0 * ema_loss && step > warmup`
- NaN/Inf: `loss.is_nan() || loss.is_infinite()`
- Gradient explosion: `gnorm > 100 * ema_gnorm`
- Wire alerts to footer region + JSON events

### Phase 4: JSON + Text Modes (1h)

- `--json`: JSONL output (one `MetricSnapshot` per line via serde)
- `--format text`: human-readable single-line per step

### Phase 5: WASM Dashboard (3h)

- Same `MonitorGrid` renders to Canvas2D via presentar WASM backend
- WebSocket connection to `DashboardSource` (or HTTP polling fallback)
- Cell size = `canvas_width / 8`

### Phase 6: Snapshot Tests (1h)

- `#[interface_test]` for each region
- `RecordingCanvas` captures TUI output
- Golden snapshot comparison for layout regression

---

## 11. Files to Create/Modify

| File | Change |
|------|--------|
| `crates/aprender-train/src/monitor/grid.rs` | NEW: Grid protocol + Region enum + occupancy |
| `crates/aprender-train/src/monitor/tui/dashboard.rs` | MODIFY: Compose widgets into grid regions |
| `crates/aprender-train/src/monitor/tui/anomaly.rs` | NEW: Spike/NaN/explosion detection |
| `crates/aprender-train/src/monitor/wasm/dashboard.rs` | MODIFY: Same grid for Canvas2D |
| `crates/aprender-train/src/dashboard/source.rs` | MODIFY: Formalize `MetricSnapshot` |
| `crates/apr-cli/src/commands/monitor.rs` | MODIFY: Add `--json`, `--format` |
| `contracts/aprender/training-monitor-v1.yaml` | NEW: Provable contract |

---

## 12. References

| Reference | Location |
|-----------|----------|
| Existing TUI monitor | `crates/aprender-train/src/monitor/tui/app.rs` |
| DashboardSource trait | `crates/aprender-train/src/dashboard/source.rs` |
| WASM MetricsCollector | `crates/aprender-train-wasm/src/lib.rs` |
| Presentar terminal widgets (62) | `crates/aprender-present-terminal/` |
| Presentar WASM backend | `crates/aprender-present/` |
| RecordingCanvas (snapshots) | `crates/aprender-present-terminal/` |
| rmedia grid protocol | `rmedia-semantic/src/grid.rs` |
| ZClip gradient clipper | `crates/aprender-train/src/` (grep for ZClip) |
| SHIP-TWO parent spec | `docs/specifications/aprender-train/ship-two-models-spec.md` |

---

*End of specification SPEC-TRAINMON-001.*
