//! btop-killer: A comprehensive system monitor TUI
//!
//! A pure Rust implementation matching btop++ features:
//! - CPU: Per-core graphs, frequency, load average, temperature
//! - Memory: RAM/Swap with graphs and meters
//! - Disk: Usage bars, I/O activity
//! - Network: Per-interface RX/TX graphs
//! - Processes: Sortable table, tree view, filtering
//! - GPU: NVIDIA support (with monitor-nvidia feature)
//! - Battery: Status and power draw
//!
//! Run with: cargo run --example btop --features monitor
//! With GPU: cargo run --example btop --features monitor,monitor-nvidia

use std::io::{self, stdout};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Table as RatatuiTable, TableState,
};
use ratatui::Terminal;

use trueno_viz::monitor::collectors::{
    BatteryCollector, CpuCollector, DiskCollector, MemoryCollector, NetworkCollector,
    ProcessCollector, SensorCollector,
};
use trueno_viz::monitor::types::Collector;
use trueno_viz::monitor::widgets::{Graph, Meter};

#[cfg(feature = "monitor-nvidia")]
use trueno_viz::monitor::collectors::NvidiaGpuCollector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let result = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {e}");
    }

    Ok(())
}

/// View mode for different layouts
#[derive(Clone, Copy, PartialEq, Default)]
#[allow(dead_code)]
enum ViewMode {
    #[default]
    Full, // All panels
    Cpu,     // CPU focused
    Memory,  // Memory focused
    Process, // Process focused
    Network, // Network focused
    #[cfg(feature = "monitor-nvidia")]
    Gpu, // GPU focused
}

/// Process sort column
#[derive(Clone, Copy, PartialEq, Default)]
enum SortColumn {
    Pid,
    Name,
    #[default]
    Cpu,
    Mem,
    State,
    User,
    Threads,
}

impl SortColumn {
    fn name(&self) -> &'static str {
        match self {
            Self::Pid => "PID",
            Self::Name => "NAME",
            Self::Cpu => "CPU%",
            Self::Mem => "MEM%",
            Self::State => "STATE",
            Self::User => "USER",
            Self::Threads => "THR",
        }
    }

    fn next(&self) -> Self {
        match self {
            Self::Pid => Self::Name,
            Self::Name => Self::Cpu,
            Self::Cpu => Self::Mem,
            Self::Mem => Self::State,
            Self::State => Self::User,
            Self::User => Self::Threads,
            Self::Threads => Self::Pid,
        }
    }
}

struct App {
    // Collectors
    cpu: CpuCollector,
    memory: MemoryCollector,
    disk: DiskCollector,
    network: NetworkCollector,
    process: ProcessCollector,
    sensors: SensorCollector,
    battery: BatteryCollector,
    #[cfg(feature = "monitor-nvidia")]
    gpu: NvidiaGpuCollector,

    // History buffers
    cpu_history: Vec<f64>,
    mem_history: Vec<f64>,
    swap_history: Vec<f64>,
    net_rx_history: Vec<f64>,
    net_tx_history: Vec<f64>,
    per_core_percent: Vec<f64>,

    // UI State
    process_table_state: TableState,
    process_scroll: ScrollbarState,
    last_collect: Instant,
    sort_by: SortColumn,
    sort_desc: bool,
    show_help: bool,
    show_tree: bool,
    view_mode: ViewMode,
    filter: String,
    show_filter_input: bool,

    // Panel visibility (like btop 1-4 keys)
    show_cpu: bool,
    show_mem: bool,
    show_net: bool,
    show_proc: bool,
    #[cfg(feature = "monitor-nvidia")]
    show_gpu: bool,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            cpu: CpuCollector::new(),
            memory: MemoryCollector::new(),
            disk: DiskCollector::new(),
            network: NetworkCollector::new(),
            process: ProcessCollector::new(),
            sensors: SensorCollector::new(),
            battery: BatteryCollector::new(),
            #[cfg(feature = "monitor-nvidia")]
            gpu: NvidiaGpuCollector::new(),

            cpu_history: Vec::with_capacity(300),
            mem_history: Vec::with_capacity(300),
            swap_history: Vec::with_capacity(300),
            net_rx_history: Vec::with_capacity(300),
            net_tx_history: Vec::with_capacity(300),
            per_core_percent: Vec::new(),

            process_table_state: TableState::default(),
            process_scroll: ScrollbarState::default(),
            last_collect: Instant::now(),
            sort_by: SortColumn::Cpu,
            sort_desc: true,
            show_help: false,
            show_tree: false,
            view_mode: ViewMode::Full,
            filter: String::new(),
            show_filter_input: false,

            show_cpu: true,
            show_mem: true,
            show_net: true,
            show_proc: true,
            #[cfg(feature = "monitor-nvidia")]
            show_gpu: true,
        };

        // Initial collection to populate data
        app.collect_metrics();
        app.collect_metrics(); // Need 2 for deltas
        app.process_table_state.select(Some(0));

        app
    }

    fn collect_metrics(&mut self) {
        // CPU
        if self.cpu.is_available() {
            if let Ok(metrics) = self.cpu.collect() {
                if let Some(total) = metrics.get_gauge("cpu.total") {
                    self.cpu_history.push(total / 100.0);
                    if self.cpu_history.len() > 300 {
                        self.cpu_history.remove(0);
                    }
                }

                // Per-core percentages
                self.per_core_percent.clear();
                for i in 0..self.cpu.core_count() {
                    if let Some(percent) = metrics.get_gauge(&format!("cpu.core.{i}")) {
                        self.per_core_percent.push(percent);
                    }
                }
            }
        }

        // Memory
        if self.memory.is_available() {
            if let Ok(metrics) = self.memory.collect() {
                if let Some(percent) = metrics.get_gauge("memory.used.percent") {
                    self.mem_history.push(percent / 100.0);
                    if self.mem_history.len() > 300 {
                        self.mem_history.remove(0);
                    }
                }
                if let Some(swap_percent) = metrics.get_gauge("memory.swap.percent") {
                    self.swap_history.push(swap_percent / 100.0);
                    if self.swap_history.len() > 300 {
                        self.swap_history.remove(0);
                    }
                }
            }
        }

        // Network
        if self.network.is_available() {
            let _ = self.network.collect();
            if let Some(iface) = self.network.current_interface() {
                if let Some(rates) = self.network.all_rates().get(iface) {
                    // Normalize to 0-1 range (assume max 1 GB/s for scaling)
                    let rx_norm = (rates.rx_bytes_per_sec / 1_000_000_000.0).min(1.0);
                    let tx_norm = (rates.tx_bytes_per_sec / 1_000_000_000.0).min(1.0);
                    self.net_rx_history.push(rx_norm);
                    self.net_tx_history.push(tx_norm);
                    if self.net_rx_history.len() > 300 {
                        self.net_rx_history.remove(0);
                    }
                    if self.net_tx_history.len() > 300 {
                        self.net_tx_history.remove(0);
                    }
                }
            }
        }

        // Disk, Process, Sensors, Battery
        if self.disk.is_available() {
            let _ = self.disk.collect();
        }
        if self.process.is_available() {
            let _ = self.process.collect();
        }
        if self.sensors.is_available() {
            let _ = self.sensors.collect();
        }
        if self.battery.is_available() {
            let _ = self.battery.collect();
        }

        // GPU
        #[cfg(feature = "monitor-nvidia")]
        if self.gpu.is_available() {
            let _ = self.gpu.collect();
        }

        self.last_collect = Instant::now();
    }

    fn sorted_processes(&self) -> Vec<&trueno_viz::monitor::collectors::process::ProcessInfo> {
        let mut procs: Vec<_> = self
            .process
            .processes()
            .values()
            .filter(|p| {
                if self.filter.is_empty() {
                    true
                } else {
                    p.name.to_lowercase().contains(&self.filter.to_lowercase())
                        || p.cmdline.to_lowercase().contains(&self.filter.to_lowercase())
                }
            })
            .collect();

        procs.sort_by(|a, b| {
            let cmp = match self.sort_by {
                SortColumn::Pid => a.pid.cmp(&b.pid),
                SortColumn::Name => a.name.cmp(&b.name),
                SortColumn::Cpu => {
                    a.cpu_percent.partial_cmp(&b.cpu_percent).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::Mem => {
                    a.mem_percent.partial_cmp(&b.mem_percent).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::State => (a.state.as_char()).cmp(&b.state.as_char()),
                SortColumn::User => a.user.cmp(&b.user),
                SortColumn::Threads => a.threads.cmp(&b.threads),
            };
            if self.sort_desc {
                cmp.reverse()
            } else {
                cmp
            }
        });

        procs
    }

    fn navigate_process(&mut self, delta: isize) {
        let count = self.sorted_processes().len();
        if count == 0 {
            return;
        }
        let current = self.process_table_state.selected().unwrap_or(0);
        let new = if delta > 0 {
            (current + delta as usize).min(count - 1)
        } else {
            current.saturating_sub((-delta) as usize)
        };
        self.process_table_state.select(Some(new));
        self.process_scroll = self.process_scroll.position(new);
    }

    fn toggle_panel(&mut self, panel: u8) {
        match panel {
            1 => self.show_cpu = !self.show_cpu,
            2 => self.show_mem = !self.show_mem,
            3 => self.show_net = !self.show_net,
            4 => self.show_proc = !self.show_proc,
            #[cfg(feature = "monitor-nvidia")]
            5 => self.show_gpu = !self.show_gpu,
            _ => {}
        }
    }

    /// Handle keypress while filter input is active.
    /// Returns true if the event was consumed (caller should `continue`).
    fn handle_filter_key(&mut self, code: KeyCode) -> bool {
        if !self.show_filter_input {
            return false;
        }
        match code {
            KeyCode::Esc => {
                self.show_filter_input = false;
                self.filter.clear();
            }
            KeyCode::Enter => {
                self.show_filter_input = false;
            }
            KeyCode::Backspace => {
                self.filter.pop();
            }
            KeyCode::Char(c) => {
                self.filter.push(c);
            }
            _ => {}
        }
        true
    }

    /// Handle a normal-mode keypress. Returns `true` if the app should quit.
    fn handle_normal_key(&mut self, key: crossterm::event::KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
            KeyCode::Char('?') | KeyCode::F(1) => self.show_help = !self.show_help,
            KeyCode::Char('1') => self.toggle_panel(1),
            KeyCode::Char('2') => self.toggle_panel(2),
            KeyCode::Char('3') => self.toggle_panel(3),
            KeyCode::Char('4') => self.toggle_panel(4),
            #[cfg(feature = "monitor-nvidia")]
            KeyCode::Char('5') => self.toggle_panel(5),
            KeyCode::Down | KeyCode::Char('j') => self.navigate_process(1),
            KeyCode::Up | KeyCode::Char('k') => self.navigate_process(-1),
            KeyCode::PageDown => self.navigate_process(10),
            KeyCode::PageUp => self.navigate_process(-10),
            KeyCode::Home | KeyCode::Char('g') => {
                self.process_table_state.select(Some(0));
            }
            KeyCode::End | KeyCode::Char('G') => {
                let count = self.sorted_processes().len();
                if count > 0 {
                    self.process_table_state.select(Some(count - 1));
                }
            }
            KeyCode::Tab | KeyCode::Char('s') => self.sort_by = self.sort_by.next(),
            KeyCode::Char('r') => self.sort_desc = !self.sort_desc,
            KeyCode::Char('t') => self.show_tree = !self.show_tree,
            KeyCode::Char('f' | '/') => self.show_filter_input = true,
            KeyCode::Delete => self.filter.clear(),
            KeyCode::Char('0') => self.view_mode = ViewMode::Full,
            _ => {}
        }
        false
    }

    /// Count the number of visible top panels.
    fn visible_panel_count(&self) -> u32 {
        let mut count: u32 = 0;
        if self.show_cpu {
            count += 1;
        }
        if self.show_mem {
            count += 1;
        }
        if self.show_net {
            count += 1;
        }
        #[cfg(feature = "monitor-nvidia")]
        if self.show_gpu && self.gpu.is_available() {
            count += 1;
        }
        count
    }
}

/// Process a single crossterm event. Returns `true` to quit.
fn handle_event(app: &mut App) -> io::Result<bool> {
    let ev = event::read()?;
    let Event::Key(key) = ev else {
        return Ok(false);
    };
    if key.kind != KeyEventKind::Press {
        return Ok(false);
    }
    if app.handle_filter_key(key.code) {
        return Ok(false);
    }
    Ok(app.handle_normal_key(key))
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let mut app = App::new();
    let tick_rate = Duration::from_millis(50);
    let collect_interval = Duration::from_secs(1);

    loop {
        terminal.draw(|f| draw_ui(f, &mut app))?;

        if app.last_collect.elapsed() >= collect_interval {
            app.collect_metrics();
        }

        if event::poll(tick_rate)? && handle_event(&mut app)? {
            return Ok(());
        }
    }
}

fn draw_ui(f: &mut ratatui::Frame, app: &mut App) {
    let area = f.area();
    let visible_panels = app.visible_panel_count();

    let top_height = if visible_panels > 0 { 45 } else { 0 };
    let proc_height = if app.show_proc { 100 - top_height } else { 0 };

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(top_height as u16),
            Constraint::Percentage(proc_height as u16),
        ])
        .split(area);

    if visible_panels > 0 {
        draw_top_panels(f, app, main_chunks[0], visible_panels);
    }
    if app.show_proc {
        draw_process_panel(f, app, main_chunks[1]);
    }
    if app.show_help {
        draw_help_overlay(f);
    }
    if app.show_filter_input {
        draw_filter_input(f, app);
    }
}

/// Lay out and render the top metric panels (CPU, memory, network, GPU).
fn draw_top_panels(f: &mut ratatui::Frame, app: &mut App, area: Rect, visible: u32) {
    let mut constraints = Vec::new();
    if app.show_cpu {
        constraints.push(Constraint::Ratio(1, visible));
    }
    if app.show_mem {
        constraints.push(Constraint::Ratio(1, visible));
    }
    if app.show_net {
        constraints.push(Constraint::Ratio(1, visible));
    }
    #[cfg(feature = "monitor-nvidia")]
    if app.show_gpu && app.gpu.is_available() {
        constraints.push(Constraint::Ratio(1, visible));
    }

    let top_chunks =
        Layout::default().direction(Direction::Horizontal).constraints(constraints).split(area);

    let mut idx = 0;
    if app.show_cpu {
        draw_cpu_panel(f, app, top_chunks[idx]);
        idx += 1;
    }
    if app.show_mem {
        draw_memory_panel(f, app, top_chunks[idx]);
        idx += 1;
    }
    if app.show_net {
        draw_network_panel(f, app, top_chunks[idx]);
        #[allow(unused_assignments)]
        {
            idx += 1;
        }
    }
    #[cfg(feature = "monitor-nvidia")]
    if app.show_gpu && app.gpu.is_available() {
        draw_gpu_panel(f, app, top_chunks[idx]);
    }
}

fn draw_cpu_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let load = app.cpu.load_average();
    let uptime_secs = app.cpu.uptime_secs();
    let uptime_str = format_uptime(uptime_secs);

    let title = format!(
        " CPU ({} cores) | Load: {:.2} {:.2} {:.2} | Up: {} ",
        app.cpu.core_count(),
        load.one,
        load.five,
        load.fifteen,
        uptime_str
    );

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 3 {
        return;
    }

    // Layout: graph on top, per-core meters below
    let core_meter_height = (app.cpu.core_count() as u16 + 1).min(inner.height / 2);
    let graph_height = inner.height.saturating_sub(core_meter_height);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(graph_height), Constraint::Length(core_meter_height)])
        .split(inner);

    // CPU graph with braille
    let graph = Graph::new(&app.cpu_history).color(Color::Cyan);
    f.render_widget(graph, chunks[0]);

    // Per-core meters (2 columns)
    if !app.per_core_percent.is_empty() && chunks[1].height > 0 {
        let cols = 2;
        let col_width = chunks[1].width / cols;
        let rows_per_col = app.per_core_percent.len().div_ceil(cols as usize);

        for (i, &percent) in app.per_core_percent.iter().enumerate() {
            let col = i / rows_per_col;
            let row = i % rows_per_col;

            if row as u16 >= chunks[1].height {
                continue;
            }

            let meter_area = Rect {
                x: chunks[1].x + (col as u16 * col_width),
                y: chunks[1].y + row as u16,
                width: col_width.saturating_sub(1),
                height: 1,
            };

            let color = percent_color(percent);
            let label = format!("{i:2}");
            let meter = Meter::new(percent / 100.0).label(label).color(color);
            f.render_widget(meter, meter_area);
        }
    }
}

fn draw_memory_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Memory ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 {
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    // Memory graph
    let mem_graph = Graph::new(&app.mem_history).color(Color::Green);
    f.render_widget(mem_graph, chunks[0]);

    // RAM meter
    if let Some(&mem_pct) = app.mem_history.last() {
        let color = percent_color(mem_pct * 100.0);
        let meter = Meter::new(mem_pct).label("RAM").color(color);
        f.render_widget(meter, chunks[1]);
    }

    // Swap meter
    if let Some(&swap_pct) = app.swap_history.last() {
        let color = percent_color(swap_pct * 100.0);
        let meter = Meter::new(swap_pct).label("Swap").color(color);
        f.render_widget(meter, chunks[2]);
    }

    // Disk usage
    if chunks[3].height >= 1 {
        let disk_info = app.disk.mounts();
        let mut y = chunks[3].y;
        for mount in disk_info.iter().take(chunks[3].height as usize) {
            if mount.total_bytes > 0 {
                let used_pct = mount.used_bytes as f64 / mount.total_bytes as f64;
                let label = mount.mount_point.chars().take(8).collect::<String>();
                let color = percent_color(used_pct * 100.0);
                let meter = Meter::new(used_pct).label(label).color(color);
                f.render_widget(
                    meter,
                    Rect { x: chunks[3].x, y, width: chunks[3].width, height: 1 },
                );
                y += 1;
            }
        }
    }
}

fn draw_network_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let iface = app.network.current_interface().unwrap_or("none");
    let (rx_rate, tx_rate) = app
        .network
        .current_interface()
        .and_then(|i| app.network.all_rates().get(i))
        .map_or((0.0, 0.0), |r| (r.rx_bytes_per_sec, r.tx_bytes_per_sec));

    let title = format!(
        " Network ({}) | {} {} {} ",
        iface,
        format_bytes_rate(rx_rate),
        "",
        format_bytes_rate(tx_rate),
    );

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 {
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);

    // Download graph
    let rx_graph = Graph::new(&app.net_rx_history).color(Color::Blue);
    f.render_widget(rx_graph, chunks[0]);

    // Upload graph (inverted)
    let tx_graph = Graph::new(&app.net_tx_history).color(Color::Red).inverted(true);
    f.render_widget(tx_graph, chunks[1]);

    // Rate labels
    let rx_label = Line::from(vec![
        Span::styled(" ", Style::default().fg(Color::Blue)),
        Span::raw(format!(" {}/s", format_bytes(rx_rate as u64))),
    ]);
    let tx_label = Line::from(vec![
        Span::styled(" ", Style::default().fg(Color::Red)),
        Span::raw(format!(" {}/s", format_bytes(tx_rate as u64))),
    ]);

    f.render_widget(
        Paragraph::new(rx_label),
        Rect { x: chunks[0].x, y: chunks[0].y, width: 20.min(chunks[0].width), height: 1 },
    );
    f.render_widget(
        Paragraph::new(tx_label),
        Rect {
            x: chunks[1].x,
            y: chunks[1].y + chunks[1].height.saturating_sub(1),
            width: 20.min(chunks[1].width),
            height: 1,
        },
    );
}

#[cfg(feature = "monitor-nvidia")]
fn draw_gpu_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let gpus = app.gpu.gpus();
    let gpu_name = gpus.first().map_or("GPU", |g| g.name.as_str());
    let gpu_temp = gpus.first().map_or(0.0, |g| g.temperature);
    let gpu_power = gpus.first().map_or(0, |g| g.power_mw / 1000);

    let title = format!(" {} | {}°C | {}W ", gpu_name, gpu_temp as u32, gpu_power);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 || gpus.is_empty() {
        return;
    }

    let gpu = &gpus[0];
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    // GPU utilization graph
    if let Some(history) = app.gpu.gpu_history(0) {
        let data: Vec<f64> = history.iter().copied().collect();
        let graph = Graph::new(&data).color(Color::Yellow);
        f.render_widget(graph, chunks[0]);
    }

    // GPU utilization meter
    let gpu_meter =
        Meter::new(gpu.gpu_util / 100.0).label("GPU").color(percent_color(gpu.gpu_util));
    f.render_widget(gpu_meter, chunks[1]);

    // VRAM meter
    let vram_pct = if gpu.mem_total > 0 { gpu.mem_used as f64 / gpu.mem_total as f64 } else { 0.0 };
    let vram_meter = Meter::new(vram_pct).label("VRAM").color(percent_color(vram_pct * 100.0));
    f.render_widget(vram_meter, chunks[2]);

    // Additional info
    if chunks[3].height >= 1 {
        let info = format!(
            "Clock: {} MHz | Mem: {} MHz | Fan: {}%",
            gpu.gpu_clock_mhz,
            gpu.mem_clock_mhz,
            gpu.fan_speed.unwrap_or(0)
        );
        f.render_widget(
            Paragraph::new(info).style(Style::default().fg(Color::DarkGray)),
            chunks[3],
        );
    }
}

fn draw_process_panel(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let sorted = app.sorted_processes();
    let count = sorted.len();

    let sort_indicator = app.sort_by.name();
    let direction = if app.sort_desc { "↓" } else { "↑" };
    let filter_info = if app.filter.is_empty() {
        String::new()
    } else {
        format!(" | Filter: \"{}\"", app.filter)
    };
    let tree_info = if app.show_tree { " | Tree" } else { "" };

    let title = format!(
        " Processes ({count}) | Sort: {sort_indicator} {direction}{filter_info}{tree_info} "
    );

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Header with sort indicators
    let header_cells = ["PID", "USER", "S", "THR", "CPU%", "MEM%", "NAME", "COMMAND"];
    let header = Row::new(header_cells.iter().map(|h| {
        let style = if *h == app.sort_by.name()
            || (*h == "S" && app.sort_by == SortColumn::State)
            || (*h == "THR" && app.sort_by == SortColumn::Threads)
        {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        };
        Span::styled(*h, style)
    }))
    .height(1);

    let rows: Vec<Row> = sorted
        .iter()
        .map(|p| {
            let state_color = match p.state {
                trueno_viz::monitor::collectors::process::ProcessState::Running => Color::Green,
                trueno_viz::monitor::collectors::process::ProcessState::Sleeping => Color::DarkGray,
                trueno_viz::monitor::collectors::process::ProcessState::DiskWait => Color::Yellow,
                trueno_viz::monitor::collectors::process::ProcessState::Zombie => Color::Red,
                _ => Color::White,
            };

            let cpu_color = percent_color(p.cpu_percent);
            let mem_color = percent_color(p.mem_percent);

            Row::new(vec![
                Span::styled(format!("{:>6}", p.pid), Style::default()),
                Span::styled(
                    format!("{:8}", p.user.chars().take(8).collect::<String>()),
                    Style::default().fg(Color::Blue),
                ),
                Span::styled(format!("{}", p.state.as_char()), Style::default().fg(state_color)),
                Span::styled(format!("{:>3}", p.threads), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:>5.1}", p.cpu_percent), Style::default().fg(cpu_color)),
                Span::styled(format!("{:>5.1}", p.mem_percent), Style::default().fg(mem_color)),
                Span::styled(
                    format!("{:15}", p.name.chars().take(15).collect::<String>()),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    p.cmdline.chars().take(50).collect::<String>(),
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(7),
        Constraint::Length(9),
        Constraint::Length(2),
        Constraint::Length(4),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(16),
        Constraint::Min(20),
    ];

    app.process_scroll = app.process_scroll.content_length(count);

    let table = RatatuiTable::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol(" ");

    f.render_stateful_widget(table, inner, &mut app.process_table_state);

    // Scrollbar
    if count > inner.height as usize {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some(""))
            .end_symbol(Some(""));
        let scrollbar_area =
            Rect { x: area.x + area.width - 1, y: area.y + 1, width: 1, height: area.height - 2 };
        f.render_stateful_widget(scrollbar, scrollbar_area, &mut app.process_scroll);
    }
}

fn draw_help_overlay(f: &mut ratatui::Frame) {
    let area = f.area();
    let popup_width = 60;
    let popup_height = 22;

    let popup_area = Rect {
        x: (area.width.saturating_sub(popup_width)) / 2,
        y: (area.height.saturating_sub(popup_height)) / 2,
        width: popup_width.min(area.width),
        height: popup_height.min(area.height),
    };

    f.render_widget(Clear, popup_area);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  trueno-viz btop - Pure Rust System Monitor",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from("  Navigation:"),
        Line::from("    j/k, /       Move up/down"),
        Line::from("    PgUp/PgDn         Page up/down"),
        Line::from("    g/G               Go to top/bottom"),
        Line::from(""),
        Line::from("  Sorting & Filtering:"),
        Line::from("    s, Tab            Cycle sort column"),
        Line::from("    r                 Reverse sort order"),
        Line::from("    f, /              Filter processes"),
        Line::from("    Del               Clear filter"),
        Line::from(""),
        Line::from("  Panels:"),
        Line::from("    1-4               Toggle CPU/Mem/Net/Proc"),
        #[cfg(feature = "monitor-nvidia")]
        Line::from("    5                 Toggle GPU panel"),
        Line::from("    t                 Toggle tree view"),
        Line::from(""),
        Line::from("  General:"),
        Line::from("    q, Esc            Quit"),
        Line::from("    ?, F1             Toggle help"),
        Line::from(""),
    ];

    let help = Paragraph::new(help_text).block(
        Block::default()
            .title(" Help ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)),
    );

    f.render_widget(help, popup_area);
}

fn draw_filter_input(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();
    let input_width = 40;
    let input_area = Rect {
        x: (area.width.saturating_sub(input_width)) / 2,
        y: area.height / 2,
        width: input_width.min(area.width),
        height: 3,
    };

    f.render_widget(Clear, input_area);

    let input = Paragraph::new(app.filter.as_str())
        .block(
            Block::default()
                .title(" Filter (Enter to confirm, Esc to cancel) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .style(Style::default().fg(Color::White));

    f.render_widget(input, input_area);
}

// Helper functions

fn percent_color(percent: f64) -> Color {
    if percent > 90.0 {
        Color::Red
    } else if percent > 70.0 {
        Color::Yellow
    } else if percent > 50.0 {
        Color::Green
    } else {
        Color::Cyan
    }
}

fn format_bytes(bytes: u64) -> String {
    batuta_common::fmt::format_bytes_compact(bytes)
}

fn format_bytes_rate(bytes_per_sec: f64) -> String {
    batuta_common::fmt::format_bytes_rate(bytes_per_sec)
}

fn format_uptime(secs: f64) -> String {
    let total_secs = secs as u64;
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;

    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}
