//! ptop v2 - Compositional Design
//!
//! ZERO draw_text calls. All rendering through widgets.
//! Target: Match ttop's USEFUL layout, not feature bloat.

use crate::widgets::*;
use crate::Layout;

/// The entire ptop UI as a widget composition
pub struct Ptop {
    // Data sources (updated each tick)
    cpu: CpuData,
    memory: MemoryData,
    disks: Vec<DiskMount>,
    network: NetworkData,
    pressure: PsiData,
    processes: Vec<Process>,
}

impl Ptop {
    /// Build the UI - THIS IS THE ENTIRE LAYOUT
    pub fn view(&self) -> impl Widget {
        Layout::rows([
            // === TOP HALF: System Overview (50%) ===
            Layout::columns([
                self.cpu_panel(),      // 33%
                self.memory_panel(),   // 33%
                self.disk_panel(),     // 33%
            ]).height(Percent(25)),

            Layout::columns([
                self.network_panel(),  // 50%
                self.pressure_panel(), // 50%
            ]).height(Percent(25)),

            // === BOTTOM HALF: Processes (50%) ===
            self.process_panel().expanded(),
        ])
    }

    fn cpu_panel(&self) -> impl Widget {
        Border::rounded("CPU")
            .child(
                Layout::rows([
                    // Per-core grid (compact, like ttop)
                    CpuGrid::new(&self.cpu.cores)
                        .columns(2)
                        .show_percentages(true),

                    // Load averages
                    Text::new(format!(
                        "Load {} {} {}",
                        self.cpu.load_1,
                        self.cpu.load_5,
                        self.cpu.load_15
                    )),
                ])
            )
    }

    fn memory_panel(&self) -> impl Widget {
        Border::rounded("Memory")
            .child(
                // Stacked bar showing Used/Swap/Cached/Free
                MemoryBar::new()
                    .segment("Used", self.memory.used, Color::MAGENTA)
                    .segment("Swap", self.memory.swap, Color::YELLOW)
                    .segment("Cached", self.memory.cached, Color::CYAN)
                    .segment("Free", self.memory.free, Color::GREEN)
                    .show_labels(true)
            )
    }

    fn disk_panel(&self) -> impl Widget {
        Border::rounded("Disk")
            .child(
                // Each mount point as a meter
                Layout::rows(
                    self.disks.iter().map(|d| {
                        Meter::new(d.used_percent)
                            .label(&d.mount_point)
                            .width(Percent(100))
                    })
                )
            )
    }

    fn network_panel(&self) -> impl Widget {
        Border::rounded("Network")
            .child(
                // Interface list with sparklines
                NetworkPanel::new(&self.network.interfaces)
                    .show_sparklines(true)
            )
    }

    fn pressure_panel(&self) -> impl Widget {
        Border::rounded("Pressure")
            .child(
                Layout::rows([
                    Meter::new(self.pressure.cpu).label("CPU"),
                    Meter::new(self.pressure.memory).label("MEM"),
                    Meter::new(self.pressure.io).label("I/O"),
                ])
            )
    }

    fn process_panel(&self) -> impl Widget {
        Border::rounded(format!("Processes ({})", self.processes.len()))
            .child(
                // THIS GETS 50% OF SCREEN
                Table::new()
                    .headers(["PID", "S", "CPU%", "MEM%", "COMMAND"])
                    .rows(self.processes.iter().map(|p| {
                        [
                            p.pid.to_string(),
                            p.state.to_string(),
                            format!("{:.1}", p.cpu),
                            format!("{:.1}", p.mem),
                            p.command.clone(),
                        ]
                    }))
                    .sortable(true)
                    .scrollable(true)
            )
    }
}

// === DATA STRUCTURES (filled by collectors) ===

pub struct CpuData {
    pub cores: Vec<f64>,  // Per-core percentages
    pub load_1: f64,
    pub load_5: f64,
    pub load_15: f64,
}

pub struct MemoryData {
    pub used: u64,
    pub swap: u64,
    pub cached: u64,
    pub free: u64,
    pub total: u64,
}

pub struct DiskMount {
    pub mount_point: String,
    pub used_percent: f64,
}

pub struct NetworkData {
    pub interfaces: Vec<NetworkInterface>,
}

pub struct PsiData {
    pub cpu: f64,
    pub memory: f64,
    pub io: f64,
}

pub struct Process {
    pub pid: u32,
    pub state: char,
    pub cpu: f64,
    pub mem: f64,
    pub command: String,
}

// === WHAT THIS GIVES US ===
//
// 1. ~100 lines instead of 2800
// 2. Zero draw_text calls
// 3. Each widget testable in isolation
// 4. Layout matches ttop (useful, not bloated)
// 5. Process table gets 50% of screen
// 6. No empty panels wasting space
// 7. Data/View separation (widgets don't fetch data)
