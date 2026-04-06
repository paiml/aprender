# Falsification Protocol: cbtop v1.0 Release Candidate

**To**: External QA / Red Team
**From**: Trueno Engineering
**Subject**: Falsification Orders for `cbtop` (Compute Block Top)

## 🛑 Mission Directive
**Do not verify that it works. Verify that it fails.**
Your goal is to falsify the claim: *"cbtop is a strictly architected, Brick-native TUI that visualizes real hardware metrics without external TUI frameworks."*

If you find *one* violation of the criteria below, the release is **REJECTED**.

---

## 🔍 Falsification Vectors

### Vector 1: The "Potemkin Village" Architecture
**Claim**: "All UI panels are independent `Brick` components, not monolithic code."
**Falsification Attack**:
1.  Navigate to `crates/cbtop/src/bricks/panels/`.
2.  **VERIFY** that `overview.rs`, `cpu.rs`, `gpu.rs`, and `help.rs` exist.
3.  **INSPECT** `overview.rs`:
    *   Does it implement `impl Brick for OverviewPanelBrick`?
    *   Does it implement `assertions()` returning a `Vec<BrickAssertion>`?
    *   Does it have a `paint()` method?
4.  **INSPECT** `crates/cbtop/src/app.rs`:
    *   Search for `self.overview_panel.paint`.
    *   **FAILURE CONDITION**: If `app.rs` contains raw drawing logic (e.g., `canvas.draw_text`) inside `fn render()` instead of delegating to `self.panel.paint()`.

### Vector 2: The "Simulation" Simulacrum (Genchi Genbutsu)
**Claim**: "Metrics are derived from real hardware sources (/proc, /sys, NVML), not simulation."
**Falsification Attack**:
1.  **CPU Collector**:
    *   File: `crates/cbtop/src/bricks/collectors/cpu.rs`
    *   **SEARCH**: `grep "rand" crates/cbtop/src/bricks/collectors/cpu.rs`
    *   **SEARCH**: `grep "/proc/stat" crates/cbtop/src/bricks/collectors/cpu.rs`
    *   **FAILURE CONDITION**: Presence of `rand`, `sin()`, or hardcoded numbers instead of file IO.
2.  **GPU Collector**:
    *   File: `crates/cbtop/src/bricks/collectors/gpu.rs`
    *   **CHECK**: Does it import `trueno_gpu::monitor`?
    *   **CHECK**: Does it call `CudaDeviceInfo::query` or `CudaMemoryInfo::query`?
    *   **FAILURE CONDITION**: Returns hardcoded struct (e.g., `utilization: 45`).
3.  **Memory Collector**:
    *   File: `crates/cbtop/src/bricks/collectors/memory.rs`
    *   **CHECK**: Reads `/proc/meminfo`.

### Vector 3: The "Hidden External TUI" (Dependency Contamination)
**Claim**: "Zero reliance on `ratatui`, `tui-rs`, or external TUI rendering engines. All TUI rendering uses `presentar`."
**Falsification Attack**:
1.  Run: `cargo tree -p cbtop --invert ratatui`
2.  Run: `grep "ratatui" crates/cbtop/Cargo.toml`
3.  **FAILURE CONDITION**: Any output indicating `ratatui` is present in the dependency graph of `cbtop`. All TUI must use `presentar`.

### Vector 4: The "Paper Tiger" Tests
**Claim**: "Tests validate logic, they are not just empty assertions."
**Falsification Attack**:
1.  Run: `cargo test -p cbtop`
2.  **INSPECT** Output: Look for `test bricks::collectors::cpu::tests::test_cpu_metrics_in_range`.
3.  **FAILURE CONDITION**: Test suite passes but contains 0 tests, or tests merely `assert!(true)`.

---

## 🛠 Execution Script

Copy and paste this script to execute the Falsification Ritual:

```bash
#!/bin/bash
set -e

echo "🔥 INITIATING FALSIFICATION RITUAL..."

# 1. Architecture Check
echo "Checking Architecture..."
if grep -q "impl Brick for OverviewPanelBrick" crates/cbtop/src/bricks/panels/overview.rs; then
    echo "✅ OverviewPanelBrick implements Brick"
else
    echo "❌ FAIL: OverviewPanelBrick missing Brick trait"
    exit 1
fi

# 2. Data Fidelity Check (CPU)
echo "Checking Genchi Genbutsu (CPU)..."
if grep -q "/proc/stat" crates/cbtop/src/bricks/collectors/cpu.rs; then
    echo "✅ CPU Collector reads /proc/stat"
else
    echo "❌ FAIL: CPU Collector does not read /proc/stat"
    exit 1
fi

if grep -q "rand::" crates/cbtop/src/bricks/collectors/cpu.rs; then
    echo "❌ FAIL: CPU Collector uses randomization"
    exit 1
else
    echo "✅ No randomization detected in CPU collector"
fi

# 3. Data Fidelity Check (GPU)
echo "Checking Genchi Genbutsu (GPU)..."
if grep -q "trueno_gpu::monitor" crates/cbtop/src/bricks/collectors/gpu.rs; then
    echo "✅ GPU Collector uses trueno_gpu monitor"
else
    echo "❌ FAIL: GPU Collector does not use trueno_gpu"
    exit 1
fi

# 4. Zero-External-TUI Check (must use presentar, not ratatui)
echo "Checking Dependencies..."
if cargo tree -p cbtop | grep -q "ratatui"; then
    echo "❌ FAIL: ratatui detected in dependencies (should use presentar)"
    exit 1
else
    echo "✅ Zero-External-TUI confirmed (presentar only)"
fi

# 5. Run Tests
echo "Running Tests..."
cargo test -p cbtop

echo "🎉 FALSIFICATION SURVIVED. RELEASE CANDIDATE VALID."
```
