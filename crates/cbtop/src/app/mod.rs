//! cbtop Application State Machine
//!
//! Manages the TUI application lifecycle using presentar-terminal.
//! Implements REAL load generation using trueno compute bricks.
//!
//! Citations:
//! - [Gregg 2020] "Systems Performance" 2nd ed. Addison-Wesley. ISBN:978-0-13-682015-4
//! - [Hennessy & Patterson 2017] "Computer Architecture" 6th ed. ISBN:978-0-12-811905-1

mod hardware;
mod input;
mod metrics;
mod panels;
mod render;

pub use hardware::{DiskMetrics, HardwareInfo, LoadMetrics, MemoryBreakdown, NetworkMetrics};
pub use panels::ActivePanel;

use std::time::{Duration, Instant};

use presentar_terminal::direct::{CellBuffer, DiffRenderer};
use presentar_terminal::{ColorMode, Theme};

use crate::bricks::generators::SimdLoadBrick;
use crate::bricks::panels::{
    ConfigPanelBrick, CpuPanelBrick, GpuPanelBrick, HelpPanelBrick, LoadControlPanelBrick,
    MemoryPanelBrick, OverviewPanelBrick, PciePanelBrick, ThermalPanelBrick,
};
use crate::config::{ComputeBackend, Config, WorkloadType};
use crate::error::CbtopError;
use crate::ring_buffer::RingBuffer;

/// Application state
pub struct CbtopApp {
    /// Configuration
    pub(crate) config: Config,
    /// Active panel
    pub(crate) active_panel: ActivePanel,
    /// Is load generation running
    pub(crate) is_running: bool,
    /// Current load intensity
    pub(crate) intensity: f64,
    /// Current backend
    pub(crate) backend: ComputeBackend,
    /// Current workload
    pub(crate) workload: WorkloadType,
    /// Problem size
    pub(crate) problem_size: usize,

    /// Hardware information
    pub(crate) hardware: HardwareInfo,
    /// SIMD load generator (real compute)
    pub(crate) simd_generator: SimdLoadBrick,
    /// Real-time load metrics
    pub(crate) load_metrics: LoadMetrics,

    /// CPU usage history (REAL from /proc/stat)
    pub(crate) cpu_history: RingBuffer<f64>,
    /// Bricks/sec history
    pub(crate) bricks_history: RingBuffer<f64>,
    /// Frame times for FPS calculation
    pub(crate) frame_times: RingBuffer<f64>,
    /// Last CPU stat for delta calculation
    pub(crate) last_cpu_stat: Option<(u64, u64)>,
    /// Last network stat for rate calculation (rx_bytes, tx_bytes, time)
    pub(crate) last_network_stat: Option<(u64, u64, Instant)>,

    /// Panel Bricks
    pub(crate) overview_panel: OverviewPanelBrick,
    pub(crate) cpu_panel: CpuPanelBrick,
    pub(crate) gpu_panel: GpuPanelBrick,
    pub(crate) help_panel: HelpPanelBrick,
    pub(crate) memory_panel: MemoryPanelBrick,
    pub(crate) thermal_panel: ThermalPanelBrick,
    pub(crate) pcie_panel: PciePanelBrick,
    pub(crate) load_panel: LoadControlPanelBrick,
    pub(crate) config_panel: ConfigPanelBrick,

    /// Terminal buffer
    pub(crate) buffer: CellBuffer,
    /// Renderer
    pub(crate) renderer: DiffRenderer,
    /// Theme for colors
    pub(crate) theme: Theme,
    /// Last frame time
    pub(crate) last_frame: Instant,
    /// Frame count
    pub(crate) frame_count: u64,

    /// Should quit
    pub(crate) should_quit: bool,
}

impl CbtopApp {
    /// Create new application with config
    pub fn new(config: Config) -> Result<Self, CbtopError> {
        let (width, height) = crossterm::terminal::size()?;

        // Detect hardware at startup
        let hardware = HardwareInfo::detect();

        // Create SIMD load generator with configured problem size
        let mut simd_generator = SimdLoadBrick::new(config.problem_size);
        simd_generator.set_intensity(config.load_profile.intensity());

        Ok(Self {
            config: config.clone(),
            active_panel: ActivePanel::Overview,
            is_running: false,
            intensity: config.load_profile.intensity(),
            backend: config.backend,
            workload: config.workload,
            problem_size: config.problem_size,
            hardware,
            simd_generator,
            load_metrics: LoadMetrics::default(),
            cpu_history: RingBuffer::new(120),
            bricks_history: RingBuffer::new(120),
            frame_times: RingBuffer::new(60),
            last_cpu_stat: None,
            last_network_stat: None,
            overview_panel: OverviewPanelBrick::new(),
            cpu_panel: CpuPanelBrick::new(),
            gpu_panel: GpuPanelBrick::new(),
            help_panel: HelpPanelBrick::new(),
            memory_panel: MemoryPanelBrick::new(),
            thermal_panel: ThermalPanelBrick::new(),
            pcie_panel: PciePanelBrick::new(),
            load_panel: LoadControlPanelBrick::new(),
            config_panel: ConfigPanelBrick::new(),
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::with_color_mode(ColorMode::TrueColor),
            theme: Theme::tokyo_night(),
            last_frame: Instant::now(),
            frame_count: 0,
            should_quit: false,
        })
    }

    /// Run the application main loop
    pub fn run(&mut self) -> Result<(), CbtopError> {
        use crossterm::{
            event::{self, Event, KeyEventKind},
            terminal::{
                disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
            },
            ExecutableCommand,
        };
        use std::io::stdout;

        // Enter TUI mode
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        stdout().execute(crossterm::cursor::Hide)?;

        let refresh = Duration::from_millis(self.config.refresh_ms);

        while !self.should_quit {
            // Handle events
            if event::poll(refresh)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }

            // Run REAL load generation if enabled
            if self.is_running {
                self.run_load_iteration();
            }

            // Collect REAL metrics
            self.collect_real_metrics();

            // Render
            self.render()?;
        }

        // Cleanup
        stdout().execute(crossterm::cursor::Show)?;
        stdout().execute(LeaveAlternateScreen)?;
        disable_raw_mode()?;

        Ok(())
    }
}
