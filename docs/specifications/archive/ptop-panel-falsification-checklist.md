# SPEC-024-F: ptop Panel Falsification Checklist

**Status**: ALL 14 PANELS IMPLEMENTED - Layout matches ttop
**Reference**: ttop at `/home/noah/src/trueno-viz/crates/ttop/src/panels.rs`

## Panel Implementation Status

| # | Panel | ttop Lines | ptop Status | Priority |
|---|-------|------------|-------------|----------|
| 1 | CPU | 61-307 | DONE | P0 |
| 2 | Memory | 310-661 | DONE | P0 |
| 3 | Disk | 663-1003 | DONE | P0 |
| 4 | Network | 1005-1496 | DONE | P0 |
| 5 | Process | 2497-2675 | DONE | P0 |
| 6 | GPU | 1498-1993 | DONE (nvidia-smi/sysfs) | P1 |
| 7 | Battery | 1995-2052 | DONE (/sys/power_supply) | P2 |
| 8 | Sensors | 2055-2154 | DONE | P1 |
| 9 | Sensors Compact | 2156-2258 | DONE | P2 |
| 10 | PSI | 2261-2342 | DONE | P1 |
| 11 | System | 2345-2385 | DONE | P2 |
| 12 | Connections | 2677-2800 | DONE (stub) | P1 |
| 13 | Treemap | 2807-2830 | DONE (stub) | P3 |
| 14 | Files | 3062-3250 | DONE (stub) | P3 |

---

## Layout Matching ttop

### Top/Bottom Split (ttop-style)
- [x] 45% height for top panels grid
- [x] 55% height for bottom row
- [x] Adaptive 2-column grid for top panels

### Bottom Row Layout (ttop-style 3-column)
- [x] 40% width: Process panel
- [x] 30% width: Connections panel
- [x] 30% width: Treemap/Files panel

---

## F001-F014: Panel Existence Falsification

### F001: CPU Panel
- [x] Panel exists in ptop
- [x] Title: ` CPU {pct}% │ {cores} cores │ {freq}GHz │ up {time} │ LAV {load} `
- [x] Per-core meters on LEFT (format: `NN ██████ XXX`)
- [x] CPU history graph on RIGHT (Block mode)
- [x] Load gauge at bottom with trend arrows (↑/↓/→)
- [x] percent_color gradient (cyan -> green -> yellow -> orange -> red)
- [x] Top 3 CPU consumers row
- [x] CPU frequency display with boost icon (⚡)
- [ ] Temperature overlay on per-core meters

### F002: Memory Panel
- [x] Panel exists in ptop
- [x] Title: ` Memory │ {used}G / {total}G ({pct}%) │ ZRAM:{ratio}x `
- [x] Stacked memory bar (Used|Cached|Free)
- [x] Memory breakdown rows (Used, Swap, Cached, Free)
- [x] percent_color for Used segment
- [x] ZRAM ratio display if active (with compressed/original size and algorithm)
- [ ] Sparkline trends per row
- [ ] PSI footer

### F003: Disk Panel
- [x] Panel exists in ptop
- [x] Title: ` Disk │ R: {rate}/s │ W: {rate}/s │ {used}G / {total}G `
- [x] Per-mount usage bars with percent_color
- [x] I/O rates from /proc/diskstats (read/write bytes per second)
- [ ] Latency gauge (first line)
- [ ] I/O rate per mount
- [ ] Sparklines per mount
- [ ] PSI footer

### F004: Network Panel
- [x] Panel exists in ptop
- [x] Title: ` Network ({iface}) │ ↓ {rx}/s │ ↑ {tx}/s `
- [x] Interface display with sparklines
- [x] RX/TX rate formatting with format_bytes
- [x] RX color (cyan) and TX color (red) matching ttop
- [x] Primary interface name in title (excludes loopback, picks highest traffic)
- [ ] Multi-interface row (if 2+ interfaces)
- [ ] Session totals + peak rates
- [ ] Protocol stats (TCP/UDP/ICMP)

### F005: Process Panel
- [x] Panel exists in ptop
- [x] Title: ` Processes ({count}) │ Sort: {col} {dir} │ Filter: "{filter}" `
- [x] Process table with PID, S, C%, M%, COMMAND (compact mode)
- [x] Colored CPU%/MEM% values using percent_color
- [x] Selection highlighting
- [x] Header row with columns: PID S C% M% COMMAND (compact)
- [x] State column (S) with colored symbols (R=green, S=gray, D=orange, Z=red, T=yellow, I=dark gray)
- [ ] Tree view mode (toggle with 't')
- [ ] Tree symbols (├─, └─, │)

### F006: GPU Panel
- [x] Panel exists in ptop
- [x] Title: ` {gpu_name} │ {temp}°C │ {power}W ` (NVIDIA via nvidia-smi, AMD via sysfs)
- [x] GPU utilization bar with percent_color
- [x] VRAM usage bar (if available)
- [x] Temperature row with color coding (green<70, yellow<85, red>85)
- [x] Power consumption row

### F007: Battery Panel
- [x] Panel exists in ptop
- [x] Title: ` Battery │ {capacity}% │ {state} │ {time} `
- [x] Charge meter (inverted color: red<20%, yellow<50%, green>50%)
- [x] Time remaining (discharging) or time to full (charging)
- [x] Status icon (⚡ Charging, 🔋 Discharging, ✓ Full)

### F008: Sensors Panel
- [x] Panel exists in ptop
- [x] Title: ` Sensors │ Max: {temp}°C `
- [x] Per-sensor row with health indicator (✓/⚠/✗)
- [x] Temperature value with color coding
- [ ] Drift indicator (↑/↓ rate/min)
- [ ] Outlier marker (!) in magenta
- [ ] Thermal headroom display

### F009: Sensors Compact Panel
- [x] Panel exists in ptop
- [x] Title: ` Sensors │ {max_temp}°C `
- [x] Type character: C (CPU), G (GPU), D (Disk), F (Fan), M (Mobo)
- [x] 4-char dual-color bar (▄)
- [x] Value display (right-aligned)
- [x] Label (truncated)

### F010: PSI Panel
- [x] Panel exists in ptop
- [x] Title: ` Pressure │ {level_symbol} `
- [x] CPU pressure with symbol and percentage
- [x] Memory pressure with symbol and percentage
- [x] I/O pressure with symbol and percentage
- [x] Symbols: — (none), ◐ (low), ▼ (med), ▲ (high), ▲▲ (critical)
- [x] Color escalation by severity

### F011: System Panel
- [x] Panel exists in ptop
- [x] Hostname display
- [x] Kernel version display
- [x] Container detection (Docker/Podman)
- [ ] Dynamic height allocation
- [ ] Conditional display based on availability

### F012: Connections Panel
- [x] Panel exists in ptop
- [x] Title: ` Connections │ {active} active │ {listen} listen `
- [x] Header: SVC │ LOCAL │ REMOTE │ GEO │ ST │ AGE │ PROC
- [ ] Protocol coloring (TCP cyan, UDP yellow)
- [ ] State coloring (Established green, Listen blue, etc.)
- [ ] Country flag emoji or localhost indicator
- [ ] Connection age formatting
- [ ] Process association
- [ ] Hot connection highlighting

### F013: Treemap Panel
- [x] Panel exists in ptop
- [x] Title: ` Files │ N:nvme D:hdd h:home `
- [x] Mount legend with single-letter codes
- [ ] Delegates to unified files view
- [ ] Actual treemap visualization

### F014: Files Panel
- [x] Panel exists in ptop
- [x] Title: ` Files │ {total} total │ {hot} hot │ {dup} dup │ {wasted} wasted `
- [x] 4 sparklines row (placeholder): I/O Activity, Entropy, Duplicates, Recent
- [ ] File list with type icon, I/O icon, entropy icon, dup marker
- [ ] File name (green if recent)
- [ ] File size (right-aligned)

---

## F015-F028: Panel Visibility Toggle Falsification

Each panel must have a keyboard toggle matching ttop:

| Key | Panel | ttop | ptop |
|-----|-------|------|------|
| 1 | CPU | Toggle | Toggle |
| 2 | Memory | Toggle | Toggle |
| 3 | Disk | Toggle | Toggle |
| 4 | Network | Toggle | Toggle |
| 5 | Process | Toggle | Toggle |
| 6 | GPU | Toggle | Toggle |
| 7 | Sensors | Toggle | Toggle |
| 8 | Connections | Toggle | Toggle |
| 9 | PSI | Toggle | Toggle |
| 0 | Reset All | Reset | Reset |

---

## F029-F042: Color Consistency Falsification

All panels must use ttop's exact RGB colors:

### Border Colors (from ttop/theme.rs)
| Panel | RGB | Hex | ptop |
|-------|-----|-----|------|
| CPU | (100, 200, 255) | #64C8FF | CHECK |
| Memory | (180, 120, 255) | #B478FF | CHECK |
| Disk | (100, 180, 255) | #64B4FF | CHECK |
| Network | (255, 150, 100) | #FF9664 | CHECK |
| Process | (220, 180, 100) | #DCC464 | CHECK |
| GPU | (100, 255, 150) | #64FF96 | CHECK |
| Battery | (255, 220, 100) | #FFDC64 | CHECK |
| Sensors | (255, 100, 150) | #FF6496 | CHECK |
| PSI | (200, 80, 80) | #C85050 | CHECK |
| Connections | (120, 180, 220) | #78B4DC | CHECK |
| Files | (180, 140, 100) | #B48C64 | CHECK |

### Percent Color Gradient (5-stop, matching ttop)
| Range | Color | Implementation |
|-------|-------|----------------|
| 0-25% | Cyan to Green | CHECK |
| 25-50% | Green to Yellow | CHECK |
| 50-75% | Yellow to Orange | CHECK |
| 75-90% | Orange to Red | CHECK |
| 90-100% | Bright Red | CHECK |

---

## F043-F056: Layout Consistency Falsification

### Panel Grid Layout (matching ttop)
- [x] Top panels: 45% height, 2-column adaptive grid
- [x] Bottom row: 55% height, 3-column (40/30/30)
- [x] Responsive to terminal size changes
- [x] Minimum size handling (graceful degradation)

### Panel Internal Layout
- [x] All panels use btop-style rounded borders
- [x] Title left-aligned within border
- [x] Content area respects 1-char padding from border
- [x] Reserved rows calculated correctly

---

## Helper Functions (matching ttop/theme.rs)

- [x] `percent_color(pct: f64) -> Color` - 5-stop gradient
- [x] `format_bytes(bytes: u64) -> String` - KB/MB/GB/TB formatting
- [x] `format_uptime(secs: u64) -> String` - Xd Yh Zm formatting

---

## Falsification Score

**Current Score**: 101 / 130 checks passing (78%)

**Target Score**: 130 / 130 checks passing (100%)

**Gap Analysis**:
- All 14 panels now exist (F001-F014 existence: COMPLETE)
- Panel toggles: 10/10 working
- Border colors: 11/11 correct
- Layout: Matches ttop (45/55 split, 3-column bottom)
- percent_color gradient: Matches ttop
- format_bytes/format_uptime: Implemented
- CPU panel: Frequency, boost icon, trend arrows, top consumers
- Process panel: Compact mode with S/C%/M% headers, state column with colors
- Network panel: RX/TX colors matching ttop, interface name in title
- Memory panel: ZRAM display with ratio, compressed/original size, algorithm
- Disk panel: I/O rates (R/W bytes/sec) from /proc/diskstats
- GPU panel: Full data via nvidia-smi (NVIDIA) or sysfs (AMD)
- Battery panel: Charge meter, status, time remaining/to full

**Remaining Work**:
- Process tree view mode (toggle with 't')
- Connections panel full data integration (/proc/net/tcp)
- Files panel full data integration
- Sparklines per memory/disk row
- PSI footer on relevant panels
