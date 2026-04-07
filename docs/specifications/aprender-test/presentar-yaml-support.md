# SPEC-030: Presentar YAML Support

**Status**: DRAFT
**Author**: probar team
**Created**: 2025-01-12
**Target**: probar 0.3.0

## Abstract

This specification defines probar's native support for testing presentar YAML configurations, enabling automated validation of TUI dashboards, terminal rendering, and falsification protocols.

## 1. Background and Motivation

### 1.1 Problem Statement

Presentar defines declarative YAML configurations for terminal dashboards (ptop, dashboards, charts). Currently, testing these configurations requires:
1. Manual visual inspection
2. Custom test harnesses per project
3. No standardized falsification protocol

### 1.2 Related Work

#### Peer-Reviewed Citations

1. **Model-Based Testing of Reactive Systems** (Tretmans, 2008)
   - Formal methods for state machine testing
   - Input-Output Conformance (IOCO) theory
   - *J. Logic and Algebraic Programming*, 77(1-2), pp. 38-73
   - DOI: 10.1016/j.jlap.2008.03.003

2. **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs** (Claessen & Hughes, 2000)
   - Property-based testing foundations
   - Shrinking strategies for minimal counterexamples
   - *ACM SIGPLAN Notices*, 35(9), pp. 268-279
   - DOI: 10.1145/351240.351266

3. **Mutation Testing: From Theory to Practice** (Jia & Harman, 2011)
   - Mutation operators and equivalent mutant problem
   - Falsification as mutation survival analysis
   - *IEEE Trans. Software Engineering*, 37(5), pp. 649-678
   - DOI: 10.1109/TSE.2010.62

4. **Visual GUI Testing: A Survey** (Alegroth et al., 2013)
   - Image-based vs. DOM-based GUI testing
   - Terminal rendering as visual regression subset
   - *Software Testing, Verification and Reliability*, 25(5-7), pp. 521-560
   - DOI: 10.1002/stvr.1547

5. **Statecharts: A Visual Formalism for Complex Systems** (Harel, 1987)
   - Hierarchical state machines
   - Foundation for SCXML (W3C) used in probar playbooks
   - *Science of Computer Programming*, 8(3), pp. 231-274
   - DOI: 10.1016/0167-6423(87)90035-9

6. **The Science of Conjecture: Evidence and Probability Before Pascal** (Franklin, 2001)
   - Falsificationism foundations (Popper)
   - Scientific method applied to software testing
   - Johns Hopkins University Press, ISBN: 978-0801871092

7. **An Empirical Study of the Reliability of UNIX Utilities** (Miller et al., 1990)
   - Fuzz testing origins
   - Random input generation for robustness testing
   - *Communications of the ACM*, 33(12), pp. 32-44
   - DOI: 10.1145/96267.96279

8. **Metamorphic Testing: A Review of Challenges and Opportunities** (Chen et al., 2018)
   - Testing without oracle problem
   - Metamorphic relations for UI testing
   - *ACM Computing Surveys*, 51(1), Article 4
   - DOI: 10.1145/3143561

### 1.3 Design Principles

Following Toyota Production System (TPS) principles:

| Principle | Application |
|-----------|-------------|
| **Poka-Yoke** | Schema validation prevents invalid YAML at parse time |
| **Jidoka** | Fail-fast on first falsification failure |
| **Genchi Genbutsu** | Test actual terminal output, not mocks |
| **Kaizen** | Incremental falsification coverage improvement |

---

## 2. Architecture

### 2.1 Module Structure

```
probar/
├── src/
│   ├── presentar/
│   │   ├── mod.rs              # Public API
│   │   ├── schema.rs           # YAML schema types
│   │   ├── validator.rs        # Config validation
│   │   ├── terminal.rs         # CellBuffer assertions
│   │   ├── falsification.rs    # F001-F100 generator
│   │   └── snapshot.rs         # ANSI snapshot diffing
│   └── playbook/
│       └── presentar_ext.rs    # Playbook extensions
```

### 2.2 Schema Types

```rust
/// Presentar configuration schema (ptop.yaml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentarConfig {
    /// Refresh interval in milliseconds
    pub refresh_ms: u32,
    /// Layout configuration
    pub layout: LayoutConfig,
    /// Panel configurations
    pub panels: PanelConfigs,
    /// Keybindings
    pub keybindings: KeybindingConfig,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    pub snap_to_grid: bool,
    pub grid_size: u8,
    pub min_panel_width: u16,
    pub min_panel_height: u16,
}

/// Panel type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelType {
    Cpu, Memory, Disk, Network, Process,
    Gpu, Battery, Sensors, SensorsCompact,
    Psi, System, Connections, Treemap, Files,
}
```

### 2.3 Integration with Playbook Schema

```yaml
# Extended playbook schema for presentar
version: "1.0"
name: "ptop-validation"
presentar:
  schema: "ptop"           # Schema type: ptop | dashboard | chart
  config: "./ptop.yaml"    # Config file to validate
  terminal:
    width: 120
    height: 40
machine:
  id: "ptop_states"
  initial: "normal"
  states:
    normal:
      id: "normal"
      invariants:
        - description: "All enabled panels visible"
          condition: "terminal.contains_all(enabled_panels)"
    exploded:
      id: "exploded"
      invariants:
        - description: "Single panel fills screen"
          condition: "terminal.panel_count == 1"
  transitions:
    - id: "explode"
      from: "normal"
      to: "exploded"
      event: "Enter"
      assertions:
        - type: terminal_contains
          text: "BREAKDOWN"
```

---

## 3. Falsification Protocol (F001-F100)

### 3.1 Panel Existence (F001-F014)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F001 | CPU panel exists | `panels.cpu.enabled = false` | "CPU panel must be visible" |
| F002 | Memory panel exists | `panels.memory.enabled = false` | "Memory panel must be visible" |
| F003 | Disk panel exists | `panels.disk.enabled = false` | "Disk panel must be visible" |
| F004 | Network panel exists | `panels.network.enabled = false` | "Network panel must be visible" |
| F005 | Process panel exists | `panels.process.enabled = false` | "Process panel must be visible" |
| F006 | GPU panel exists | `panels.gpu.enabled = false` | "GPU panel must be visible" |
| F007 | Battery panel exists | `panels.battery.enabled = false` | "Battery panel must be visible" |
| F008 | Sensors panel exists | `panels.sensors.enabled = false` | "Sensors panel must be visible" |
| F009 | SensorsCompact exists | `panels.sensors_compact.enabled = false` | "SensorsCompact must be visible" |
| F010 | PSI panel exists | `panels.psi.enabled = false` | "PSI panel must be visible" |
| F011 | System panel exists | `panels.system.enabled = false` | "System panel must be visible" |
| F012 | Connections exists | `panels.connections.enabled = false` | "Connections must be visible" |
| F013 | Treemap panel exists | `panels.treemap.enabled = false` | "Treemap panel must be visible" |
| F014 | Files panel exists | `panels.files.enabled = false` | "Files panel must be visible" |

### 3.2 Panel Content (F015-F028)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F015 | CPU shows percentage | `cpu.show_percent = false` | "CPU % must be visible" |
| F016 | CPU shows cores | `cpu.show_cores = false` | "Core count must be visible" |
| F017 | CPU shows frequency | `cpu.show_frequency = false` | "Frequency must be visible" |
| F018 | CPU shows temperature | `cpu.show_temperature = false` | "Temperature must be visible" |
| F019 | Memory shows used/total | `memory.show_usage = false` | "Memory usage must be visible" |
| F020 | Memory shows ZRAM | `memory.show_zram = false` | "ZRAM ratio must be visible" |
| F021 | Disk shows R/W rates | `disk.show_io = false` | "I/O rates must be visible" |
| F022 | Network shows RX/TX | `network.show_rates = false` | "RX/TX must be visible" |
| F023 | Process shows PID | `process.columns -= pid` | "PID column must exist" |
| F024 | Process shows CPU% | `process.columns -= cpu` | "CPU% column must exist" |
| F025 | Process shows MEM% | `process.columns -= mem` | "MEM% column must exist" |
| F026 | GPU shows utilization | `gpu.show_util = false` | "GPU util must be visible" |
| F027 | GPU shows VRAM | `gpu.show_vram = false` | "VRAM must be visible" |
| F028 | Battery shows charge | `battery.show_charge = false` | "Charge must be visible" |

### 3.3 Color Consistency (F029-F042)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F029 | CPU border color | `theme.cpu_color = #000000` | "CPU border must be #64C8FF" |
| F030 | Memory border color | `theme.memory_color = #000000` | "Memory border must be #B478FF" |
| F031 | Disk border color | `theme.disk_color = #000000` | "Disk border must be #64B4FF" |
| F032 | Network border color | `theme.network_color = #000000` | "Network border must be #FF9664" |
| F033 | Process border color | `theme.process_color = #000000` | "Process border must be #DCC464" |
| F034 | GPU border color | `theme.gpu_color = #000000` | "GPU border must be #64FF96" |
| F035 | Battery border color | `theme.battery_color = #000000` | "Battery border must be #FFDC64" |
| F036 | Sensors border color | `theme.sensors_color = #000000` | "Sensors border must be #FF6496" |
| F037 | PSI border color | `theme.psi_color = #000000` | "PSI border must be #C85050" |
| F038 | Connections border | `theme.conn_color = #000000` | "Connections border must be #78B4DC" |
| F039 | Files border color | `theme.files_color = #000000` | "Files border must be #B48C64" |
| F040 | Percent 0-25% cyan | `percent_color(10) != cyan` | "0-25% must be cyan" |
| F041 | Percent 50-75% yellow | `percent_color(60) != yellow` | "50-75% must be yellow" |
| F042 | Percent 90-100% red | `percent_color(95) != red` | "90-100% must be red" |

### 3.4 Layout Consistency (F043-F056)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F043 | Top panels 45% height | `layout.top_height = 0.2` | "Top must be 45% height" |
| F044 | Bottom row 55% height | `layout.bottom_height = 0.8` | "Bottom must be 55% height" |
| F045 | Process 40% width | `layout.process_width = 0.1` | "Process must be 40% width" |
| F046 | Connections 30% width | `layout.conn_width = 0.1` | "Connections must be 30% width" |
| F047 | Treemap 30% width | `layout.tree_width = 0.1` | "Treemap must be 30% width" |
| F048 | Grid snap enabled | `layout.snap_to_grid = false` | "Grid snap must work" |
| F049 | Min panel width | `layout.min_panel_width = 5` | "Min width must be 30" |
| F050 | Min panel height | `layout.min_panel_height = 2` | "Min height must be 6" |
| F051 | Rounded borders | `layout.border_style = sharp` | "Borders must be rounded" |
| F052 | Title left-aligned | `layout.title_align = center` | "Title must be left" |
| F053 | 1-char padding | `layout.padding = 0` | "Padding must be 1" |
| F054 | Responsive resize | `resize(80, 24)` fails | "Must handle resize" |
| F055 | Graceful degradation | `resize(40, 12)` crashes | "Must degrade gracefully" |
| F056 | 2-column adaptive | `columns != 2 when width > 100` | "Must use 2 columns" |

### 3.5 Keybinding Consistency (F057-F070)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F057 | 'q' quits | `keybindings.quit = x` | "'q' must quit" |
| F058 | '?' shows help | `keybindings.help = x` | "'?' must show help" |
| F059 | 'f' toggles FPS | `keybindings.toggle_fps = x` | "'f' must toggle FPS" |
| F060 | '/' filters | `keybindings.filter = x` | "'/' must filter" |
| F061 | 'c' sorts by CPU | `keybindings.sort_cpu = x` | "'c' must sort CPU" |
| F062 | 'm' sorts by MEM | `keybindings.sort_mem = x` | "'m' must sort MEM" |
| F063 | 'p' sorts by PID | `keybindings.sort_pid = x` | "'p' must sort PID" |
| F064 | 'k' kills process | `keybindings.kill = x` | "'k' must kill" |
| F065 | Enter explodes | `keybindings.explode = x` | "Enter must explode" |
| F066 | Escape collapses | `keybindings.collapse = x` | "Escape must collapse" |
| F067 | Tab navigates | `keybindings.navigate = x` | "Tab must navigate" |
| F068 | '1' toggles CPU | `keybindings.toggle_1 = x` | "'1' must toggle CPU" |
| F069 | '0' resets all | `keybindings.reset = x` | "'0' must reset" |
| F070 | No key conflicts | `keybindings.quit = keybindings.help` | "Keys must be unique" |

### 3.6 Data Binding (F071-F084)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F071 | CPU data updates | `cpu.update_interval = 0` | "CPU must update" |
| F072 | Memory data updates | `memory.update_interval = 0` | "Memory must update" |
| F073 | Disk data updates | `disk.update_interval = 0` | "Disk must update" |
| F074 | Network data updates | `network.update_interval = 0` | "Network must update" |
| F075 | Process data updates | `process.update_interval = 0` | "Process must update" |
| F076 | GPU data updates | `gpu.update_interval = 0` | "GPU must update" |
| F077 | Sparkline history | `sparkline_history = 0` | "History must exist" |
| F078 | Async data race | `async_delay = 5000ms` | "Must handle slow data" |
| F079 | Missing data fallback | `data.cpu = null` | "Must show N/A" |
| F080 | NaN handling | `data.cpu.percent = NaN` | "Must handle NaN" |
| F081 | Negative values | `data.memory.used = -1` | "Must clamp to 0" |
| F082 | Overflow values | `data.cpu.percent = 150` | "Must clamp to 100" |
| F083 | Empty process list | `data.processes = []` | "Must show empty state" |
| F084 | 1000+ processes | `data.processes.len = 5000` | "Must paginate" |

### 3.7 Performance (F085-F092)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F085 | 60 FPS render | `render_time > 16ms` | "Must render in 16ms" |
| F086 | Memory stable | `memory_growth > 1MB/min` | "Must not leak" |
| F087 | CPU < 5% idle | `cpu_usage > 5%` | "Must be efficient" |
| F088 | Startup < 100ms | `startup_time > 100ms` | "Must start fast" |
| F089 | Resize < 16ms | `resize_time > 16ms` | "Must resize fast" |
| F090 | Filter O(n) | `filter_complexity != O(n)` | "Filter must be O(n)" |
| F091 | Sort O(n log n) | `sort_complexity != O(n log n)` | "Sort must be O(n log n)" |
| F092 | Render O(panels) | `render_complexity != O(p)` | "Render must be O(p)" |

### 3.8 Accessibility (F093-F100)

| ID | Description | Mutation | Expected Failure |
|----|-------------|----------|------------------|
| F093 | High contrast mode | `theme.high_contrast = false` | "Must support high contrast" |
| F094 | Colorblind safe | `theme.colorblind = false` | "Must be colorblind safe" |
| F095 | Screen reader text | `aria.labels = null` | "Must have labels" |
| F096 | Keyboard-only nav | `mouse_only = true` | "Must work keyboard-only" |
| F097 | Focus visible | `focus.visible = false` | "Focus must be visible" |
| F098 | No flashing | `animation.flash = true` | "No flashing content" |
| F099 | Text scalable | `text.scalable = false` | "Text must scale" |
| F100 | Error messages clear | `error.verbose = false` | "Errors must be clear" |

---

## 4. Implementation

### 4.1 Phase 1: Schema Parsing (Week 1)

```rust
// src/presentar/schema.rs
impl PresentarConfig {
    pub fn from_yaml(yaml: &str) -> Result<Self, PresentarError> {
        let config: Self = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), PresentarError> {
        if self.refresh_ms < 16 {
            return Err(PresentarError::InvalidRefreshRate(self.refresh_ms));
        }
        if self.layout.grid_size < 2 || self.layout.grid_size > 16 {
            return Err(PresentarError::InvalidGridSize(self.layout.grid_size));
        }
        Ok(())
    }
}
```

### 4.2 Phase 2: Terminal Assertions (Week 2)

```rust
// src/presentar/terminal.rs
pub struct TerminalSnapshot {
    cells: Vec<Cell>,
    width: u16,
    height: u16,
}

impl TerminalSnapshot {
    pub fn contains(&self, text: &str) -> bool {
        self.to_string().contains(text)
    }

    pub fn cell_color(&self, x: u16, y: u16) -> Option<Color> {
        self.cells.get((y * self.width + x) as usize)
            .map(|c| c.fg)
    }

    pub fn assert_color(&self, x: u16, y: u16, expected: Color) -> Result<(), AssertionError> {
        match self.cell_color(x, y) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(AssertionError::ColorMismatch { x, y, expected, actual }),
            None => Err(AssertionError::OutOfBounds { x, y }),
        }
    }
}
```

### 4.3 Phase 3: Falsification Generator (Week 3)

```rust
// src/presentar/falsification.rs
pub fn generate_falsification_playbook(config: &PresentarConfig) -> Playbook {
    let mut mutations = Vec::new();

    // F001-F014: Panel existence
    for (panel, enabled) in config.panels.iter_enabled() {
        if enabled {
            mutations.push(MutationDef {
                id: format!("F{:03}", panel.index() + 1),
                description: format!("{} panel exists", panel.name()),
                mutate: format!("panels.{}.enabled = false", panel.key()),
                expected_failure: format!("{} panel must be visible", panel.name()),
            });
        }
    }

    // F029-F042: Color consistency
    for (panel, color) in config.theme.panel_colors() {
        mutations.push(MutationDef {
            id: format!("F{:03}", 28 + panel.index()),
            description: format!("{} border color", panel.name()),
            mutate: format!("theme.{}_color = #000000", panel.key()),
            expected_failure: format!("{} border must be {}", panel.name(), color),
        });
    }

    Playbook {
        version: "1.0".into(),
        name: "presentar-falsification".into(),
        falsification: Some(FalsificationConfig { mutations }),
        ..Default::default()
    }
}
```

### 4.4 Phase 4: PMAT Integration (Week 4)

```toml
# .pmat-gates.toml
[gates.presentar]
coverage = 95
mutation_score = 80
falsification_score = 100  # All F001-F100 must pass

[gates.presentar.complexity]
max_cyclomatic = 10
max_cognitive = 15

[gates.presentar.performance]
max_render_ms = 16
max_memory_mb = 50
```

```bash
# PMAT workflow
pmat check --gate presentar
pmat falsify --checklist F001-F100
pmat report --format junit > presentar-results.xml
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_ptop_config() {
        let yaml = include_str!("../../testdata/ptop.yaml");
        let config = PresentarConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.refresh_ms, 1000);
        assert!(config.panels.cpu.enabled);
    }

    #[test]
    fn test_reject_invalid_refresh_rate() {
        let yaml = "refresh_ms: 5\nlayout: {}\npanels: {}";
        let result = PresentarConfig::from_yaml(yaml);
        assert!(matches!(result, Err(PresentarError::InvalidRefreshRate(5))));
    }

    #[test]
    fn test_terminal_contains() {
        let snapshot = TerminalSnapshot::from_string("CPU 45%\nMEM 60%", 80, 24);
        assert!(snapshot.contains("CPU"));
        assert!(snapshot.contains("45%"));
        assert!(!snapshot.contains("GPU"));
    }
}
```

### 5.2 Integration Tests

```rust
#[test]
fn test_falsification_f001_cpu_panel() {
    let config = PresentarConfig::default();
    let playbook = generate_falsification_playbook(&config);

    let f001 = playbook.falsification.unwrap().mutations
        .iter()
        .find(|m| m.id == "F001")
        .unwrap();

    assert_eq!(f001.mutate, "panels.cpu.enabled = false");
    assert!(f001.expected_failure.contains("CPU"));
}
```

### 5.3 Property Tests

```rust
proptest! {
    #[test]
    fn prop_all_panels_have_falsification(
        enabled_panels in prop::collection::vec(any::<PanelType>(), 1..14)
    ) {
        let mut config = PresentarConfig::default();
        for panel in &enabled_panels {
            config.panels.set_enabled(*panel, true);
        }

        let playbook = generate_falsification_playbook(&config);
        let mutations = playbook.falsification.unwrap().mutations;

        for panel in enabled_panels {
            assert!(mutations.iter().any(|m| m.description.contains(panel.name())));
        }
    }
}
```

---

## 6. Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Schema validation | Parse valid YAML | 100% |
| Schema rejection | Reject invalid YAML | 100% |
| Falsification coverage | F001-F100 implemented | 100% |
| Terminal assertions | All assertion types work | 100% |
| Performance | Falsification run < 10s | 100% |
| PMAT integration | All gates pass | 100% |

---

## 7. References

1. W3C SCXML Specification: https://www.w3.org/TR/scxml/
2. Presentar ptop specification: `../presentar/docs/specifications/ptop-panel-falsification-checklist.md`
3. Probar playbook schema: `crates/probar/src/playbook/schema.rs`
4. Toyota Production System: Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.

---

## Appendix A: Full Falsification YAML

```yaml
# Generated falsification playbook for presentar
version: "1.0"
name: "presentar-f001-f100"
description: "100-point falsification protocol for presentar YAML"

falsification:
  mutations:
    # F001-F014: Panel Existence
    - id: "F001"
      description: "CPU panel exists"
      mutate: "panels.cpu.enabled = false"
      expected_failure: "CPU panel must be visible"
    # ... F002-F014 ...

    # F015-F028: Panel Content
    - id: "F015"
      description: "CPU shows percentage"
      mutate: "panels.cpu.show_percent = false"
      expected_failure: "CPU % must be visible"
    # ... F016-F028 ...

    # F029-F042: Color Consistency
    - id: "F029"
      description: "CPU border color"
      mutate: "theme.cpu_color = #000000"
      expected_failure: "CPU border must be #64C8FF"
    # ... F030-F042 ...

    # F043-F056: Layout Consistency
    - id: "F043"
      description: "Top panels 45% height"
      mutate: "layout.top_height = 0.2"
      expected_failure: "Top must be 45% height"
    # ... F044-F056 ...

    # F057-F070: Keybinding Consistency
    - id: "F057"
      description: "'q' quits"
      mutate: "keybindings.quit = x"
      expected_failure: "'q' must quit"
    # ... F058-F070 ...

    # F071-F084: Data Binding
    - id: "F071"
      description: "CPU data updates"
      mutate: "cpu.update_interval = 0"
      expected_failure: "CPU must update"
    # ... F072-F084 ...

    # F085-F092: Performance
    - id: "F085"
      description: "60 FPS render"
      mutate: "render_time > 16ms"
      expected_failure: "Must render in 16ms"
    # ... F086-F092 ...

    # F093-F100: Accessibility
    - id: "F093"
      description: "High contrast mode"
      mutate: "theme.high_contrast = false"
      expected_failure: "Must support high contrast"
    # ... F094-F100 ...
```
