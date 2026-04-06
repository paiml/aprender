//! `ProcessTable` widget for system process monitoring.
//!
//! Displays running processes with CPU/Memory usage in a ttop/btop style.
//! Reference: ttop/btop process displays.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::time::Duration;

/// Process state (from /proc/[pid]/stat)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessState {
    /// Running (R)
    Running,
    /// Sleeping (S)
    #[default]
    Sleeping,
    /// Disk sleep/waiting (D)
    DiskWait,
    /// Zombie (Z)
    Zombie,
    /// Stopped (T)
    Stopped,
    /// Idle (I)
    Idle,
}

impl ProcessState {
    /// Get the single-character representation
    #[must_use]
    pub fn char(&self) -> char {
        match self {
            Self::Running => 'R',
            Self::Sleeping => 'S',
            Self::DiskWait => 'D',
            Self::Zombie => 'Z',
            Self::Stopped => 'T',
            Self::Idle => 'I',
        }
    }

    /// Get the color for this state
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Running => Color::new(0.3, 0.9, 0.3, 1.0), // Green
            Self::Sleeping => Color::new(0.5, 0.5, 0.5, 1.0), // Gray
            Self::DiskWait => Color::new(1.0, 0.7, 0.2, 1.0), // Orange
            Self::Zombie => Color::new(1.0, 0.3, 0.3, 1.0),  // Red
            Self::Stopped => Color::new(0.9, 0.9, 0.3, 1.0), // Yellow
            Self::Idle => Color::new(0.4, 0.4, 0.4, 1.0),    // Dark gray
        }
    }
}

/// A running process entry.
#[derive(Debug, Clone)]
pub struct ProcessEntry {
    /// Process ID.
    pub pid: u32,
    /// User running the process.
    pub user: String,
    /// CPU usage percentage (0-100).
    pub cpu_percent: f32,
    /// Memory usage percentage (0-100).
    pub mem_percent: f32,
    /// Command name.
    pub command: String,
    /// Full command line (optional).
    pub cmdline: Option<String>,
    /// Process state.
    pub state: ProcessState,
    /// OOM score (0-1000, higher = more likely to be killed).
    pub oom_score: Option<i32>,
    /// cgroup path (short form).
    pub cgroup: Option<String>,
    /// Nice value (-20 to +19).
    pub nice: Option<i32>,
    /// Thread count (CB-PROC-006).
    pub threads: Option<u32>,
    /// Parent process ID (CB-PROC-001 tree view).
    pub parent_pid: Option<u32>,
    /// Tree depth level for indentation (CB-PROC-001).
    pub tree_depth: usize,
    /// Whether this is the last child at its level (CB-PROC-001).
    pub is_last_child: bool,
    /// Tree prefix string (e.g., "│ └─") for display (CB-PROC-001).
    pub tree_prefix: String,
}

impl ProcessEntry {
    /// Create a new process entry.
    #[must_use]
    pub fn new(
        pid: u32,
        user: impl Into<String>,
        cpu: f32,
        mem: f32,
        command: impl Into<String>,
    ) -> Self {
        Self {
            pid,
            user: user.into(),
            cpu_percent: cpu,
            mem_percent: mem,
            command: command.into(),
            cmdline: None,
            state: ProcessState::default(),
            oom_score: None,
            cgroup: None,
            nice: None,
            threads: None,
            parent_pid: None,
            tree_depth: 0,
            is_last_child: false,
            tree_prefix: String::new(),
        }
    }

    /// Set full command line.
    #[must_use]
    pub fn with_cmdline(mut self, cmdline: impl Into<String>) -> Self {
        self.cmdline = Some(cmdline.into());
        self
    }

    /// Set process state.
    #[must_use]
    pub fn with_state(mut self, state: ProcessState) -> Self {
        self.state = state;
        self
    }

    /// Set OOM score (0-1000).
    #[must_use]
    pub fn with_oom_score(mut self, score: i32) -> Self {
        self.oom_score = Some(score);
        self
    }

    /// Set cgroup path.
    #[must_use]
    pub fn with_cgroup(mut self, cgroup: impl Into<String>) -> Self {
        self.cgroup = Some(cgroup.into());
        self
    }

    /// Set nice value.
    #[must_use]
    pub fn with_nice(mut self, nice: i32) -> Self {
        self.nice = Some(nice);
        self
    }

    /// Set thread count (CB-PROC-006).
    #[must_use]
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Set parent PID (CB-PROC-001 tree view).
    #[must_use]
    pub fn with_parent_pid(mut self, ppid: u32) -> Self {
        self.parent_pid = Some(ppid);
        self
    }

    /// Set tree display info (CB-PROC-001 tree view).
    pub fn set_tree_info(&mut self, depth: usize, is_last: bool, prefix: String) {
        self.tree_depth = depth;
        self.is_last_child = is_last;
        self.tree_prefix = prefix;
    }
}

/// Sort column for process table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessSort {
    Pid,
    User,
    Cpu,
    Memory,
    Command,
    Oom,
}

/// Process table widget with color-coded CPU/Memory bars.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProcessTable {
    /// Process entries.
    processes: Vec<ProcessEntry>,
    /// Selected row index.
    selected: usize,
    /// Scroll offset.
    scroll_offset: usize,
    /// Current sort column.
    sort_by: ProcessSort,
    /// Sort ascending.
    sort_ascending: bool,
    /// CPU gradient (low → high).
    cpu_gradient: Gradient,
    /// Memory gradient (low → high).
    mem_gradient: Gradient,
    /// Show command line instead of command name.
    show_cmdline: bool,
    /// Compact mode (fewer columns).
    compact: bool,
    /// Show OOM score column.
    show_oom: bool,
    /// Show nice value column.
    show_nice: bool,
    /// Show thread count column (CB-PROC-006).
    show_threads: bool,
    /// Tree view mode (CB-PROC-001).
    tree_view: bool,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for ProcessTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessTable {
    /// Create a new process table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            processes: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            sort_by: ProcessSort::Cpu,
            sort_ascending: false, // Default: highest CPU first
            cpu_gradient: Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]),
            mem_gradient: Gradient::from_hex(&["#9ece6a", "#e0af68", "#f7768e"]),
            show_cmdline: false,
            compact: false,
            show_oom: false,
            show_nice: false,
            show_threads: false,
            tree_view: false,
            bounds: Rect::default(),
        }
    }

    /// Set processes.
    pub fn set_processes(&mut self, processes: Vec<ProcessEntry>) {
        self.processes = processes;
        // Tree view (CB-PROC-001) takes precedence over sorting
        if self.tree_view {
            self.build_tree();
        } else {
            self.sort_processes();
        }
        // Clamp selection
        if !self.processes.is_empty() && self.selected >= self.processes.len() {
            self.selected = self.processes.len() - 1;
        }
    }

    /// Add a process.
    pub fn add_process(&mut self, process: ProcessEntry) {
        self.processes.push(process);
    }

    /// Clear all processes.
    pub fn clear(&mut self) {
        self.processes.clear();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Set CPU gradient.
    #[must_use]
    pub fn with_cpu_gradient(mut self, gradient: Gradient) -> Self {
        self.cpu_gradient = gradient;
        self
    }

    /// Set memory gradient.
    #[must_use]
    pub fn with_mem_gradient(mut self, gradient: Gradient) -> Self {
        self.mem_gradient = gradient;
        self
    }

    /// Enable compact mode.
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Show full command line.
    #[must_use]
    pub fn with_cmdline(mut self) -> Self {
        self.show_cmdline = true;
        self
    }

    /// Show OOM score column.
    #[must_use]
    pub fn with_oom(mut self) -> Self {
        self.show_oom = true;
        self
    }

    /// Show nice value column.
    #[must_use]
    pub fn with_nice_column(mut self) -> Self {
        self.show_nice = true;
        self
    }

    /// Show thread count column (CB-PROC-006).
    #[must_use]
    pub fn with_threads_column(mut self) -> Self {
        self.show_threads = true;
        self
    }

    /// Enable tree view mode (CB-PROC-001).
    #[must_use]
    pub fn with_tree_view(mut self) -> Self {
        self.tree_view = true;
        self
    }

    /// Toggle tree view mode (CB-PROC-001).
    pub fn toggle_tree_view(&mut self) {
        self.tree_view = !self.tree_view;
        if self.tree_view {
            self.build_tree();
        }
    }

    /// Check if tree view is enabled (CB-PROC-001).
    #[must_use]
    pub fn is_tree_view(&self) -> bool {
        self.tree_view
    }

    /// Set sort column.
    pub fn sort_by(&mut self, column: ProcessSort) {
        if self.sort_by == column {
            self.sort_ascending = !self.sort_ascending;
        } else {
            self.sort_by = column;
            // Default directions (CPU/Memory/OOM default to descending)
            self.sort_ascending = !matches!(
                column,
                ProcessSort::Cpu | ProcessSort::Memory | ProcessSort::Oom
            );
        }
        self.sort_processes();
    }

    /// Get current sort column.
    #[must_use]
    pub fn current_sort(&self) -> ProcessSort {
        self.sort_by
    }

    /// Get selected index.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Get selected process.
    #[must_use]
    pub fn selected_process(&self) -> Option<&ProcessEntry> {
        self.processes.get(self.selected)
    }

    /// Select a row.
    pub fn select(&mut self, row: usize) {
        if !self.processes.is_empty() {
            self.selected = row.min(self.processes.len() - 1);
            self.ensure_visible();
        }
    }

    /// Move selection up.
    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.ensure_visible();
        }
    }

    /// Move selection down.
    pub fn select_next(&mut self) {
        if !self.processes.is_empty() && self.selected < self.processes.len() - 1 {
            self.selected += 1;
            self.ensure_visible();
        }
    }

    /// Get process count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.processes.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }

    /// Build process tree structure (CB-PROC-001).
    ///
    /// Reorganizes processes into a tree by parent-child relationships.
    /// Uses ASCII art prefixes: └─ (last child), ├─ (middle child), │ (continuation).
    fn build_tree(&mut self) {
        use std::collections::HashMap;

        if self.processes.is_empty() {
            return;
        }

        // Build PID -> index map
        let pid_to_idx: HashMap<u32, usize> = self
            .processes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.pid, i))
            .collect();

        // Build PPID -> children map
        let mut children: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut roots: Vec<usize> = Vec::new();

        for (idx, proc) in self.processes.iter().enumerate() {
            if let Some(ppid) = proc.parent_pid {
                if pid_to_idx.contains_key(&ppid) {
                    children.entry(ppid).or_default().push(idx);
                } else {
                    // Parent not in list, treat as root
                    roots.push(idx);
                }
            } else {
                roots.push(idx);
            }
        }

        // Sort children by CPU descending at each level
        for children_list in children.values_mut() {
            children_list.sort_by(|&a, &b| {
                self.processes[b]
                    .cpu_percent
                    .partial_cmp(&self.processes[a].cpu_percent)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Sort roots by CPU descending
        roots.sort_by(|&a, &b| {
            self.processes[b]
                .cpu_percent
                .partial_cmp(&self.processes[a].cpu_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // DFS walk to build tree order
        let mut tree_order: Vec<(usize, usize, bool, String)> = Vec::new();

        let roots_len = roots.len();
        for (i, &root_idx) in roots.iter().enumerate() {
            let is_last = i == roots_len - 1;
            Self::build_tree_dfs(
                root_idx,
                0,
                "",
                is_last,
                &self.processes,
                &children,
                &mut tree_order,
            );
        }

        // Reorder processes and apply tree info
        let old_processes = std::mem::take(&mut self.processes);
        self.processes.reserve(tree_order.len());

        for (idx, depth, is_last, prefix) in tree_order {
            let mut proc = old_processes[idx].clone();
            proc.set_tree_info(depth, is_last, prefix);
            self.processes.push(proc);
        }
    }

    fn build_tree_dfs(
        idx: usize,
        depth: usize,
        prefix: &str,
        is_last: bool,
        processes: &[ProcessEntry],
        children: &std::collections::HashMap<u32, Vec<usize>>,
        tree_order: &mut Vec<(usize, usize, bool, String)>,
    ) {
        let proc = &processes[idx];
        let current_prefix = if depth == 0 {
            String::new()
        } else if is_last {
            format!("{prefix}└─")
        } else {
            format!("{prefix}├─")
        };

        tree_order.push((idx, depth, is_last, current_prefix));

        // Calculate next prefix for children
        let next_prefix = if depth == 0 {
            String::new()
        } else if is_last {
            format!("{prefix}  ")
        } else {
            format!("{prefix}│ ")
        };

        if let Some(child_indices) = children.get(&proc.pid) {
            let len = child_indices.len();
            for (i, &child_idx) in child_indices.iter().enumerate() {
                let child_is_last = i == len - 1;
                Self::build_tree_dfs(
                    child_idx,
                    depth + 1,
                    &next_prefix,
                    child_is_last,
                    processes,
                    children,
                    tree_order,
                );
            }
        }
    }

    fn sort_processes(&mut self) {
        let ascending = self.sort_ascending;
        match self.sort_by {
            ProcessSort::Pid => {
                self.processes.sort_by(|a, b| {
                    if ascending {
                        a.pid.cmp(&b.pid)
                    } else {
                        b.pid.cmp(&a.pid)
                    }
                });
            }
            ProcessSort::User => {
                self.processes.sort_by(|a, b| {
                    if ascending {
                        a.user.cmp(&b.user)
                    } else {
                        b.user.cmp(&a.user)
                    }
                });
            }
            ProcessSort::Cpu => {
                self.processes.sort_by(|a, b| {
                    let cmp = a
                        .cpu_percent
                        .partial_cmp(&b.cpu_percent)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if ascending {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                });
            }
            ProcessSort::Memory => {
                self.processes.sort_by(|a, b| {
                    let cmp = a
                        .mem_percent
                        .partial_cmp(&b.mem_percent)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if ascending {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                });
            }
            ProcessSort::Command => {
                self.processes.sort_by(|a, b| {
                    if ascending {
                        a.command.cmp(&b.command)
                    } else {
                        b.command.cmp(&a.command)
                    }
                });
            }
            ProcessSort::Oom => {
                self.processes.sort_by(|a, b| {
                    let a_oom = a.oom_score.unwrap_or(0);
                    let b_oom = b.oom_score.unwrap_or(0);
                    if ascending {
                        a_oom.cmp(&b_oom)
                    } else {
                        b_oom.cmp(&a_oom)
                    }
                });
            }
        }
    }

    fn ensure_visible(&mut self) {
        let visible_rows = (self.bounds.height as usize).saturating_sub(2);
        if visible_rows == 0 {
            return;
        }

        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + visible_rows {
            self.scroll_offset = self.selected - visible_rows + 1;
        }
    }

    fn truncate(s: &str, width: usize) -> String {
        if s.chars().count() <= width {
            format!("{s:width$}")
        } else if width > 1 {
            // Use proper ellipsis character "…" instead of "..."
            let chars: String = s.chars().take(width - 1).collect();
            format!("{chars}…")
        } else {
            s.chars().take(width).collect()
        }
    }

    /// Get OOM score color based on risk level.
    fn oom_color(oom: i32) -> Color {
        if oom > 500 {
            Color::new(1.0, 0.3, 0.3, 1.0) // Red - high risk
        } else if oom > 200 {
            Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - medium risk
        } else {
            Color::new(0.5, 0.8, 0.5, 1.0) // Green - low risk
        }
    }

    /// Get nice value color based on priority.
    fn nice_color(ni: i32) -> Color {
        match ni.cmp(&0) {
            Ordering::Less => Color::new(0.3, 0.9, 0.9, 1.0), // Cyan - high priority
            Ordering::Greater => Color::new(0.6, 0.6, 0.6, 1.0), // Gray - low priority
            Ordering::Equal => Color::new(0.8, 0.8, 0.8, 1.0), // White - normal
        }
    }

    /// Get thread count color based on count.
    fn threads_color(th: u32) -> Color {
        if th > 50 {
            Color::new(0.3, 0.9, 0.9, 1.0) // Cyan - many threads
        } else if th > 10 {
            Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - moderate
        } else {
            Color::new(0.8, 0.8, 0.8, 1.0) // White - normal
        }
    }

    /// Build header string for the table.
    fn build_header(&self, cols: &ColumnWidths) -> String {
        let sep = if self.compact { " " } else { " │ " };
        let mut header = String::new();
        let _ = write!(header, "{:>w$}", "PID", w = cols.pid);
        if self.compact {
            header.push(' ');
            let _ = write!(header, "{:>1}", "S");
        } else {
            header.push_str(sep);
            let _ = write!(header, "{:w$}", "USER", w = cols.user);
        }
        if self.show_oom {
            header.push_str(sep);
            let _ = write!(header, "{:>3}", "OOM");
        }
        if self.show_nice {
            header.push_str(sep);
            let _ = write!(header, "{:>3}", "NI");
        }
        if self.show_threads {
            header.push_str(sep);
            let _ = write!(header, "{:>3}", "TH");
        }
        header.push_str(sep);
        let _ = write!(
            header,
            "{:>w$}",
            if self.compact { "C%" } else { "CPU%" },
            w = cols.cpu
        );
        header.push_str(sep);
        let _ = write!(
            header,
            "{:>w$}",
            if self.compact { "M%" } else { "MEM%" },
            w = cols.mem
        );
        header.push_str(sep);
        let _ = write!(header, "{:w$}", "COMMAND", w = cols.cmd);
        header
    }

    /// Draw a single process row.
    #[allow(clippy::too_many_arguments)]
    fn draw_row(
        &self,
        canvas: &mut dyn Canvas,
        proc: &ProcessEntry,
        y: f32,
        is_selected: bool,
        cols: &ColumnWidths,
        default_style: &TextStyle,
    ) {
        let sep = if self.compact { 1.0 } else { 3.0 };
        let mut x = self.bounds.x;
        // PID
        canvas.draw_text(
            &format!("{:>w$}", proc.pid, w = cols.pid),
            Point::new(x, y),
            default_style,
        );
        x += cols.pid as f32;
        // State or User
        if self.compact {
            x += 1.0;
            canvas.draw_text(
                &proc.state.char().to_string(),
                Point::new(x, y),
                &TextStyle {
                    color: proc.state.color(),
                    ..Default::default()
                },
            );
            x += 1.0;
        } else {
            x += sep;
            canvas.draw_text(
                &Self::truncate(&proc.user, cols.user),
                Point::new(x, y),
                default_style,
            );
            x += cols.user as f32;
        }
        // OOM
        if self.show_oom {
            x += sep;
            let oom = proc.oom_score.unwrap_or(0);
            canvas.draw_text(
                &format!("{oom:>3}"),
                Point::new(x, y),
                &TextStyle {
                    color: Self::oom_color(oom),
                    ..Default::default()
                },
            );
            x += 3.0;
        }
        // Nice
        if self.show_nice {
            x += sep;
            let ni = proc.nice.unwrap_or(0);
            canvas.draw_text(
                &format!("{ni:>3}"),
                Point::new(x, y),
                &TextStyle {
                    color: Self::nice_color(ni),
                    ..Default::default()
                },
            );
            x += 3.0;
        }
        // Threads
        if self.show_threads {
            x += sep;
            let th = proc.threads.unwrap_or(1);
            canvas.draw_text(
                &format!("{th:>3}"),
                Point::new(x, y),
                &TextStyle {
                    color: Self::threads_color(th),
                    ..Default::default()
                },
            );
            x += 3.0;
        }
        // CPU
        x += sep;
        canvas.draw_text(
            &format!("{:>5.1}%", proc.cpu_percent),
            Point::new(x, y),
            &TextStyle {
                color: self.cpu_gradient.for_percent(proc.cpu_percent as f64),
                ..Default::default()
            },
        );
        x += cols.cpu as f32;
        // Mem
        x += sep;
        canvas.draw_text(
            &format!("{:>5.1}%", proc.mem_percent),
            Point::new(x, y),
            &TextStyle {
                color: self.mem_gradient.for_percent(proc.mem_percent as f64),
                ..Default::default()
            },
        );
        x += cols.mem as f32;
        // Command
        x += sep;
        self.draw_command(canvas, proc, x, y, is_selected, cols.cmd, default_style);
    }

    /// Draw command column with optional tree prefix.
    #[allow(clippy::too_many_arguments)]
    fn draw_command(
        &self,
        canvas: &mut dyn Canvas,
        proc: &ProcessEntry,
        x: f32,
        y: f32,
        is_selected: bool,
        cmd_w: usize,
        default_style: &TextStyle,
    ) {
        let cmd = if self.show_cmdline {
            proc.cmdline.as_deref().unwrap_or(&proc.command)
        } else {
            &proc.command
        };
        let cmd_style = if is_selected {
            TextStyle {
                color: Color::new(1.0, 1.0, 1.0, 1.0),
                ..Default::default()
            }
        } else {
            default_style.clone()
        };
        if self.tree_view && !proc.tree_prefix.is_empty() {
            let prefix_len = proc.tree_prefix.chars().count();
            canvas.draw_text(
                &proc.tree_prefix,
                Point::new(x, y),
                &TextStyle {
                    color: Color::new(0.4, 0.5, 0.6, 1.0),
                    ..Default::default()
                },
            );
            canvas.draw_text(
                &Self::truncate(cmd, cmd_w.saturating_sub(prefix_len)),
                Point::new(x + prefix_len as f32, y),
                &cmd_style,
            );
        } else {
            canvas.draw_text(&Self::truncate(cmd, cmd_w), Point::new(x, y), &cmd_style);
        }
    }
}

/// Column widths for process table layout.
#[allow(dead_code)]
struct ColumnWidths {
    pid: usize,
    state: usize,
    oom: usize,
    nice: usize,
    threads: usize,
    user: usize,
    cpu: usize,
    mem: usize,
    sep: usize,
    cmd: usize,
    num_seps: usize,
}

impl ColumnWidths {
    fn new(table: &ProcessTable, width: usize) -> Self {
        let pid = 7;
        let state = if table.compact { 2 } else { 0 };
        let oom = if table.show_oom { 4 } else { 0 };
        let nice = if table.show_nice { 4 } else { 0 };
        let threads = if table.show_threads { 4 } else { 0 };
        let user = if table.compact { 0 } else { 8 };
        let cpu = 6;
        let mem = 6;
        let sep = if table.compact { 1 } else { 3 };
        let extra_cols = usize::from(table.show_oom)
            + usize::from(table.show_nice)
            + usize::from(table.show_threads);
        let num_seps = if table.compact { 3 } else { 4 } + extra_cols;
        let fixed = pid + state + oom + nice + threads + user + cpu + mem + sep * num_seps;
        let cmd = width.saturating_sub(fixed);
        Self {
            pid,
            state,
            oom,
            nice,
            threads,
            user,
            cpu,
            mem,
            sep,
            cmd,
            num_seps,
        }
    }
}

impl Brick for ProcessTable {
    fn brick_name(&self) -> &'static str {
        "process_table"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let passed = if self.processes.is_empty() || self.selected < self.processes.len() {
            vec![BrickAssertion::max_latency_ms(16)]
        } else {
            vec![]
        };
        let failed = if !self.processes.is_empty() && self.selected >= self.processes.len() {
            vec![(
                BrickAssertion::max_latency_ms(16),
                format!(
                    "Selected {} >= process count {}",
                    self.selected,
                    self.processes.len()
                ),
            )]
        } else {
            vec![]
        };

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for ProcessTable {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let min_width = if self.compact { 40.0 } else { 60.0 };
        let width = constraints.max_width.max(min_width);
        let height = (self.processes.len() + 2).min(30) as f32;
        constraints.constrain(Size::new(width, height.max(3.0)))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.ensure_visible();
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 {
            return;
        }

        let cols = ColumnWidths::new(self, width);

        // Draw header
        let header_style = TextStyle {
            color: Color::new(0.0, 1.0, 1.0, 1.0),
            weight: presentar_core::FontWeight::Bold,
            ..Default::default()
        };
        canvas.draw_text(
            &self.build_header(&cols),
            Point::new(self.bounds.x, self.bounds.y),
            &header_style,
        );

        // Draw separator
        if height > 1 {
            canvas.draw_text(
                &"─".repeat(width),
                Point::new(self.bounds.x, self.bounds.y + 1.0),
                &TextStyle {
                    color: Color::new(0.3, 0.3, 0.4, 1.0),
                    ..Default::default()
                },
            );
        }

        // Draw rows
        let default_style = TextStyle {
            color: Color::new(0.8, 0.8, 0.8, 1.0),
            ..Default::default()
        };
        let visible_rows = height.saturating_sub(2);
        for (i, proc_idx) in (self.scroll_offset..self.processes.len())
            .take(visible_rows)
            .enumerate()
        {
            let proc = &self.processes[proc_idx];
            let y = self.bounds.y + 2.0 + i as f32;
            let is_selected = proc_idx == self.selected;
            if is_selected {
                canvas.fill_rect(
                    Rect::new(self.bounds.x, y, self.bounds.width, 1.0),
                    Color::new(0.2, 0.2, 0.4, 0.5),
                );
            }
            self.draw_row(canvas, proc, y, is_selected, &cols, &default_style);
        }

        // Empty state
        if self.processes.is_empty() && height > 2 {
            canvas.draw_text(
                "No processes",
                Point::new(self.bounds.x + 1.0, self.bounds.y + 2.0),
                &TextStyle {
                    color: Color::new(0.5, 0.5, 0.5, 1.0),
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown { key, .. } => {
                match key {
                    Key::Up | Key::K => self.select_prev(),
                    Key::Down | Key::J => self.select_next(),
                    Key::C => self.sort_by(ProcessSort::Cpu),
                    Key::M => self.sort_by(ProcessSort::Memory),
                    Key::P => self.sort_by(ProcessSort::Pid),
                    Key::N => self.sort_by(ProcessSort::Command),
                    Key::O => self.sort_by(ProcessSort::Oom),
                    Key::T => self.toggle_tree_view(), // CB-PROC-001
                    _ => {}
                }
                None
            }
            _ => None,
        }
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
#[path = "process_table_tests.rs"]
mod tests;
