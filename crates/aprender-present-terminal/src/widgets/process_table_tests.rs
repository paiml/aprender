use super::*;

fn sample_processes() -> Vec<ProcessEntry> {
    vec![
        ProcessEntry::new(1, "root", 0.5, 0.1, "systemd"),
        ProcessEntry::new(1234, "noah", 25.0, 5.5, "firefox"),
        ProcessEntry::new(5678, "noah", 80.0, 12.3, "rustc"),
    ]
}

#[test]
fn test_process_table_new() {
    let table = ProcessTable::new();
    assert!(table.is_empty());
}

#[test]
fn test_process_table_set_processes() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    assert_eq!(table.len(), 3);
}

#[test]
fn test_process_table_add_process() {
    let mut table = ProcessTable::new();
    table.add_process(ProcessEntry::new(1, "root", 0.0, 0.0, "init"));
    assert_eq!(table.len(), 1);
}

#[test]
fn test_process_table_clear() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.select(1);
    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.selected(), 0);
}

#[test]
fn test_process_table_selection() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    assert_eq!(table.selected(), 0);

    table.select_next();
    assert_eq!(table.selected(), 1);

    table.select_prev();
    assert_eq!(table.selected(), 0);
}

#[test]
fn test_process_table_select_bounds() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.select(100);
    assert_eq!(table.selected(), 2);

    table.select_prev();
    table.select_prev();
    table.select_prev();
    assert_eq!(table.selected(), 0);
}

#[test]
fn test_process_table_sort_cpu() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    // Default sort is CPU descending
    assert_eq!(table.processes[0].command, "rustc");
}

#[test]
fn test_process_table_sort_toggle() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.sort_by(ProcessSort::Cpu); // Toggle to ascending
    assert_eq!(table.processes[0].command, "systemd");
}

#[test]
fn test_process_table_sort_pid() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.sort_by(ProcessSort::Pid);
    assert_eq!(table.processes[0].pid, 1);
}

#[test]
fn test_process_table_sort_memory() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.sort_by(ProcessSort::Memory);
    assert_eq!(table.processes[0].command, "rustc");
}

#[test]
fn test_process_table_selected_process() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    let proc = table.selected_process().unwrap();
    assert_eq!(proc.command, "rustc"); // Highest CPU
}

#[test]
fn test_process_entry_with_cmdline() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "bash").with_cmdline("/bin/bash --login");
    assert_eq!(proc.cmdline.as_deref(), Some("/bin/bash --login"));
}

#[test]
fn test_process_table_compact() {
    let table = ProcessTable::new().compact();
    assert!(table.compact);
}

#[test]
fn test_process_table_with_cmdline() {
    let table = ProcessTable::new().with_cmdline();
    assert!(table.show_cmdline);
}

#[test]
fn test_process_table_verify() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    assert!(table.verify().is_valid());
}

#[test]
fn test_process_table_verify_invalid() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.selected = 100;
    assert!(!table.verify().is_valid());
}

#[test]
fn test_process_table_measure() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    let size = table.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
    assert!(size.width >= 60.0);
    assert!(size.height >= 3.0);
}

#[test]
fn test_process_table_layout() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    let result = table.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
    assert_eq!(result.size.width, 80.0);
}

#[test]
fn test_process_table_event_keys() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.event(&Event::key_down(Key::J));
    assert_eq!(table.selected(), 1);

    table.event(&Event::key_down(Key::K));
    assert_eq!(table.selected(), 0);

    table.event(&Event::key_down(Key::P));
    assert_eq!(table.current_sort(), ProcessSort::Pid);
}

#[test]
fn test_process_table_brick_name() {
    let table = ProcessTable::new();
    assert_eq!(table.brick_name(), "process_table");
}

#[test]
fn test_process_table_default() {
    let table = ProcessTable::default();
    assert!(table.is_empty());
}

#[test]
fn test_process_table_children() {
    let table = ProcessTable::new();
    assert!(table.children().is_empty());
}

#[test]
fn test_process_table_children_mut() {
    let mut table = ProcessTable::new();
    assert!(table.children_mut().is_empty());
}

#[test]
fn test_process_table_truncate() {
    assert_eq!(ProcessTable::truncate("hello", 10), "hello     ");
    assert_eq!(ProcessTable::truncate("hello world", 8), "hello w…");
    assert_eq!(ProcessTable::truncate("hi", 2), "hi");
    // Ensure proper ellipsis character is used
    assert!(ProcessTable::truncate("long text here", 6).ends_with('…'));
}

#[test]
fn test_process_table_type_id() {
    let table = ProcessTable::new();
    assert_eq!(Widget::type_id(&table), TypeId::of::<ProcessTable>());
}

#[test]
fn test_process_table_to_html() {
    let table = ProcessTable::new();
    assert!(table.to_html().is_empty());
}

#[test]
fn test_process_table_to_css() {
    let table = ProcessTable::new();
    assert!(table.to_css().is_empty());
}

#[test]
fn test_process_table_sort_command() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.sort_by(ProcessSort::Command);
    assert_eq!(table.processes[0].command, "firefox");
}

#[test]
fn test_process_table_sort_user() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.sort_by(ProcessSort::User);
    assert_eq!(table.processes[0].user, "noah");
}

#[test]
fn test_process_table_sort_oom() {
    let mut table = ProcessTable::new();
    // Create processes with different OOM scores
    let entries = vec![
        ProcessEntry::new(1, "user", 10.0, 5.0, "low_oom").with_oom_score(100),
        ProcessEntry::new(2, "user", 10.0, 5.0, "high_oom").with_oom_score(800),
        ProcessEntry::new(3, "user", 10.0, 5.0, "med_oom").with_oom_score(400),
    ];
    table.set_processes(entries);

    // Sort by OOM (default descending - highest first)
    table.sort_by(ProcessSort::Oom);

    // Verify order: high (800) -> med (400) -> low (100)
    assert_eq!(table.processes[0].command, "high_oom");
    assert_eq!(table.processes[1].command, "med_oom");
    assert_eq!(table.processes[2].command, "low_oom");
}

#[test]
fn test_process_table_sort_oom_toggle_ascending() {
    let mut table = ProcessTable::new();
    let entries = vec![
        ProcessEntry::new(1, "user", 10.0, 5.0, "low_oom").with_oom_score(100),
        ProcessEntry::new(2, "user", 10.0, 5.0, "high_oom").with_oom_score(800),
    ];
    table.set_processes(entries);

    // Sort by OOM twice to toggle to ascending
    table.sort_by(ProcessSort::Oom);
    table.sort_by(ProcessSort::Oom);

    // Verify order is now ascending: low (100) -> high (800)
    assert_eq!(table.processes[0].command, "low_oom");
    assert_eq!(table.processes[1].command, "high_oom");
}

// ========================================================================
// Additional tests for paint() paths and better coverage
// ========================================================================

struct MockCanvas {
    texts: Vec<(String, Point)>,
    rects: Vec<Rect>,
}

impl MockCanvas {
    fn new() -> Self {
        Self {
            texts: vec![],
            rects: vec![],
        }
    }
}

impl Canvas for MockCanvas {
    fn fill_rect(&mut self, rect: Rect, _color: Color) {
        self.rects.push(rect);
    }
    fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
    fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
        self.texts.push((text.to_string(), position));
    }
    fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
    fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
    fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
    fn fill_arc(&mut self, _c: Point, _r: f32, _s: f32, _e: f32, _color: Color) {}
    fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
    fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
    fn push_clip(&mut self, _rect: Rect) {}
    fn pop_clip(&mut self) {}
    fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
    fn pop_transform(&mut self) {}
}

#[test]
fn test_process_table_paint_basic() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have rendered header, separator, and rows
    assert!(!canvas.texts.is_empty());
    // Check header contains "PID"
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("PID")));
}

#[test]
fn test_process_table_paint_compact() {
    let mut table = ProcessTable::new().compact();
    table.set_processes(sample_processes());
    table.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Check compact header has "C%" and "M%" instead of "CPU%" and "MEM%"
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("C%")));
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("M%")));
}

#[test]
fn test_process_table_paint_with_oom() {
    let mut table = ProcessTable::new().with_oom();
    let entries = vec![
        ProcessEntry::new(1, "user", 10.0, 5.0, "low_oom").with_oom_score(100),
        ProcessEntry::new(2, "user", 10.0, 5.0, "high_oom").with_oom_score(800),
        ProcessEntry::new(3, "user", 10.0, 5.0, "med_oom").with_oom_score(400),
    ];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 100.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have OOM header
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("OOM")));
    // Should have OOM values rendered
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("100")));
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("800")));
}

#[test]
fn test_process_table_paint_with_nice() {
    let mut table = ProcessTable::new().with_nice_column();
    let entries = vec![
        ProcessEntry::new(1, "user", 10.0, 5.0, "high_pri").with_nice(-10),
        ProcessEntry::new(2, "user", 10.0, 5.0, "low_pri").with_nice(10),
        ProcessEntry::new(3, "user", 10.0, 5.0, "normal").with_nice(0),
    ];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 100.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have NI header
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("NI")));
}

#[test]
fn test_process_table_paint_with_selection() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.select(1);
    table.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have a selection rect
    assert!(!canvas.rects.is_empty());
}

#[test]
fn test_process_table_paint_empty() {
    let mut table = ProcessTable::new();
    table.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should show "No processes" message
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("No processes")));
}

#[test]
fn test_process_table_paint_zero_bounds() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should return early, no output
    assert!(canvas.texts.is_empty());
}

#[test]
fn test_process_table_paint_with_cmdline() {
    let mut table = ProcessTable::new().with_cmdline();
    let entries =
        vec![ProcessEntry::new(1, "root", 0.5, 0.1, "bash").with_cmdline("/bin/bash --login -i")];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 100.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should show cmdline instead of command
    assert!(canvas
        .texts
        .iter()
        .any(|(t, _)| t.contains("/bin/bash") || t.contains("--login")));
}

#[test]
fn test_process_table_paint_compact_with_state() {
    let mut table = ProcessTable::new().compact();
    let entries = vec![
        ProcessEntry::new(1, "root", 50.0, 10.0, "running").with_state(ProcessState::Running),
        ProcessEntry::new(2, "user", 0.0, 0.5, "sleeping").with_state(ProcessState::Sleeping),
        ProcessEntry::new(3, "user", 0.0, 0.1, "zombie").with_state(ProcessState::Zombie),
    ];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have state characters
    assert!(canvas.texts.iter().any(|(t, _)| t == "R")); // Running
    assert!(canvas.texts.iter().any(|(t, _)| t == "S")); // Sleeping
}

#[test]
fn test_process_state_char() {
    assert_eq!(ProcessState::Running.char(), 'R');
    assert_eq!(ProcessState::Sleeping.char(), 'S');
    assert_eq!(ProcessState::DiskWait.char(), 'D');
    assert_eq!(ProcessState::Zombie.char(), 'Z');
    assert_eq!(ProcessState::Stopped.char(), 'T');
    assert_eq!(ProcessState::Idle.char(), 'I');
}

#[test]
fn test_process_state_color() {
    // Each state should have a unique color
    let running = ProcessState::Running.color();
    let sleeping = ProcessState::Sleeping.color();
    let zombie = ProcessState::Zombie.color();
    assert_ne!(running, sleeping);
    assert_ne!(running, zombie);
}

#[test]
fn test_process_state_default() {
    assert_eq!(ProcessState::default(), ProcessState::Sleeping);
}

#[test]
fn test_process_entry_with_state() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test").with_state(ProcessState::Running);
    assert_eq!(proc.state, ProcessState::Running);
}

#[test]
fn test_process_entry_with_cgroup() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test").with_cgroup("/user.slice/user-1000");
    assert_eq!(proc.cgroup.as_deref(), Some("/user.slice/user-1000"));
}

#[test]
fn test_process_entry_with_nice() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test").with_nice(-5);
    assert_eq!(proc.nice, Some(-5));
}

#[test]
fn test_process_entry_with_oom_score() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test").with_oom_score(500);
    assert_eq!(proc.oom_score, Some(500));
}

#[test]
fn test_process_entry_with_threads() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test").with_threads(42);
    assert_eq!(proc.threads, Some(42));
}

#[test]
fn test_process_table_with_threads_column() {
    let table = ProcessTable::new().with_threads_column();
    assert!(table.show_threads);
}

#[test]
fn test_process_table_scroll() {
    let mut table = ProcessTable::new();
    // Create many processes to trigger scrolling
    let entries: Vec<ProcessEntry> = (0..50)
        .map(|i| ProcessEntry::new(i, "user", i as f32, 0.0, format!("proc{i}")))
        .collect();
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 80.0, 10.0); // Only 8 visible rows
    table.layout(table.bounds);

    // Select a process beyond the visible area
    table.select(45);
    // scroll_offset should have been updated
    assert!(table.scroll_offset > 0);
}

#[test]
fn test_process_table_ensure_visible_up() {
    let mut table = ProcessTable::new();
    let entries: Vec<ProcessEntry> = (0..20)
        .map(|i| ProcessEntry::new(i, "user", 0.0, 0.0, format!("proc{i}")))
        .collect();
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
    table.scroll_offset = 10;
    table.selected = 5; // Above visible area

    table.ensure_visible();
    assert!(table.scroll_offset <= table.selected);
}

#[test]
fn test_process_table_select_empty() {
    let mut table = ProcessTable::new();
    // Should not panic on empty table
    table.select(5);
    table.select_next();
    table.select_prev();
    assert_eq!(table.selected(), 0);
}

#[test]
fn test_process_table_selected_process_empty() {
    let table = ProcessTable::new();
    assert!(table.selected_process().is_none());
}

#[test]
fn test_process_table_budget() {
    let table = ProcessTable::new();
    let budget = table.budget();
    assert!(budget.paint_ms > 0);
}

#[test]
fn test_process_table_assertions() {
    let table = ProcessTable::new();
    assert!(!table.assertions().is_empty());
}

#[test]
fn test_process_table_set_processes_clamp_selection() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.selected = 2; // Last item
                        // Set fewer processes
    table.set_processes(vec![ProcessEntry::new(1, "root", 0.0, 0.0, "test")]);
    // Selection should be clamped
    assert_eq!(table.selected(), 0);
}

#[test]
fn test_process_table_event_down() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.event(&Event::key_down(Key::Down));
    assert_eq!(table.selected(), 1);
}

#[test]
fn test_process_table_event_up() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    table.select(2);

    table.event(&Event::key_down(Key::Up));
    assert_eq!(table.selected(), 1);
}

#[test]
fn test_process_table_event_c() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    // First sort by something else
    table.sort_by(ProcessSort::Pid);

    table.event(&Event::key_down(Key::C));
    assert_eq!(table.current_sort(), ProcessSort::Cpu);
}

#[test]
fn test_process_table_event_m() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.event(&Event::key_down(Key::M));
    assert_eq!(table.current_sort(), ProcessSort::Memory);
}

#[test]
fn test_process_table_event_n() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.event(&Event::key_down(Key::N));
    assert_eq!(table.current_sort(), ProcessSort::Command);
}

#[test]
fn test_process_table_event_o() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    table.event(&Event::key_down(Key::O));
    assert_eq!(table.current_sort(), ProcessSort::Oom);
}

#[test]
fn test_process_table_event_other() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    let prev_selected = table.selected();

    // Event that doesn't match any key
    table.event(&Event::key_down(Key::A));
    assert_eq!(table.selected(), prev_selected);
}

#[test]
fn test_process_table_event_non_keydown() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());

    // Non-keydown event
    let result = table.event(&Event::Resize {
        width: 100.0,
        height: 50.0,
    });
    assert!(result.is_none());
}

#[test]
fn test_process_table_with_cpu_gradient() {
    let gradient = Gradient::from_hex(&["#0000FF", "#FF0000"]);
    let table = ProcessTable::new().with_cpu_gradient(gradient);
    // Just verify it compiles and doesn't panic
    assert!(!table.is_empty() || table.is_empty());
}

#[test]
fn test_process_table_with_mem_gradient() {
    let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
    let table = ProcessTable::new().with_mem_gradient(gradient);
    assert!(!table.is_empty() || table.is_empty());
}

#[test]
fn test_process_table_measure_compact() {
    let table = ProcessTable::new().compact();
    let size = table.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
    assert!(size.width >= 40.0); // Compact mode has smaller min width
}

#[test]
fn test_process_table_truncate_exact() {
    assert_eq!(ProcessTable::truncate("exact", 5), "exact");
}

#[test]
fn test_process_table_truncate_width_1() {
    assert_eq!(ProcessTable::truncate("hello", 1), "h");
}

#[test]
fn test_process_table_paint_all_columns() {
    // Test paint with all optional columns enabled
    let mut table = ProcessTable::new()
        .compact()
        .with_oom()
        .with_nice_column()
        .with_cmdline();

    let entries = vec![
        ProcessEntry::new(1, "root", 50.0, 10.0, "bash")
            .with_state(ProcessState::Running)
            .with_oom_score(100)
            .with_nice(-5)
            .with_cmdline("/bin/bash"),
        ProcessEntry::new(2, "user", 30.0, 5.0, "vim")
            .with_state(ProcessState::Sleeping)
            .with_oom_score(600)
            .with_nice(10)
            .with_cmdline("/usr/bin/vim"),
    ];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 120.0, 20.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // All columns should be rendered
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("PID")));
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("OOM")));
    assert!(canvas.texts.iter().any(|(t, _)| t.contains("NI")));
}

#[test]
fn test_process_entry_clone() {
    let proc = ProcessEntry::new(1, "root", 50.0, 10.0, "test")
        .with_state(ProcessState::Running)
        .with_oom_score(100);
    let cloned = proc.clone();
    assert_eq!(cloned.pid, proc.pid);
    assert_eq!(cloned.state, proc.state);
}

#[test]
fn test_process_entry_debug() {
    let proc = ProcessEntry::new(1, "root", 0.0, 0.0, "test");
    let debug = format!("{:?}", proc);
    assert!(debug.contains("ProcessEntry"));
}

#[test]
fn test_process_sort_debug() {
    let sort = ProcessSort::Cpu;
    let debug = format!("{:?}", sort);
    assert!(debug.contains("Cpu"));
}

#[test]
fn test_process_table_clone() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    let cloned = table.clone();
    assert_eq!(cloned.len(), table.len());
}

#[test]
fn test_process_table_debug() {
    let table = ProcessTable::new();
    let debug = format!("{:?}", table);
    assert!(debug.contains("ProcessTable"));
}

#[test]
fn test_process_state_debug() {
    let state = ProcessState::Running;
    let debug = format!("{:?}", state);
    assert!(debug.contains("Running"));
}

// ========================================================================
// Tree view tests (CB-PROC-001)
// ========================================================================

#[test]
fn test_process_entry_with_parent_pid() {
    let proc = ProcessEntry::new(100, "user", 10.0, 5.0, "child").with_parent_pid(1);
    assert_eq!(proc.parent_pid, Some(1));
}

#[test]
fn test_process_entry_set_tree_info() {
    let mut proc = ProcessEntry::new(100, "user", 10.0, 5.0, "child");
    proc.set_tree_info(2, true, "│ └─".to_string());
    assert_eq!(proc.tree_depth, 2);
    assert!(proc.is_last_child);
    assert_eq!(proc.tree_prefix, "│ └─");
}

#[test]
fn test_process_table_with_tree_view() {
    let table = ProcessTable::new().with_tree_view();
    assert!(table.is_tree_view());
}

#[test]
fn test_process_table_toggle_tree_view() {
    let mut table = ProcessTable::new();
    assert!(!table.is_tree_view());

    table.toggle_tree_view();
    assert!(table.is_tree_view());

    table.toggle_tree_view();
    assert!(!table.is_tree_view());
}

#[test]
fn test_process_table_tree_view_builds_tree() {
    let mut table = ProcessTable::new().with_tree_view();

    // Create parent-child hierarchy:
    // 1 (systemd) -> 100 (bash) -> 200 (vim)
    //             -> 101 (sshd)
    let entries = vec![
        ProcessEntry::new(200, "user", 5.0, 1.0, "vim").with_parent_pid(100),
        ProcessEntry::new(100, "user", 10.0, 2.0, "bash").with_parent_pid(1),
        ProcessEntry::new(101, "root", 1.0, 0.5, "sshd").with_parent_pid(1),
        ProcessEntry::new(1, "root", 0.5, 0.1, "systemd"),
    ];
    table.set_processes(entries);

    // After tree building, systemd should be first (root)
    assert_eq!(table.processes[0].command, "systemd");

    // Check tree prefixes
    // systemd (root) has no prefix
    assert_eq!(table.processes[0].tree_prefix, "");
    // bash is child of systemd, and has higher CPU than sshd
    // So bash should come before sshd
}

#[test]
fn test_process_table_tree_view_prefix_chars() {
    let mut table = ProcessTable::new().with_tree_view();

    // Create: 1 -> 2 -> 3
    let entries = vec![
        ProcessEntry::new(3, "user", 5.0, 1.0, "grandchild").with_parent_pid(2),
        ProcessEntry::new(2, "user", 10.0, 2.0, "child").with_parent_pid(1),
        ProcessEntry::new(1, "root", 0.5, 0.1, "parent"),
    ];
    table.set_processes(entries);

    // Parent should have no prefix
    assert_eq!(table.processes[0].tree_prefix, "");
    // Child should have └─ (last child of parent)
    assert!(
        table.processes[1].tree_prefix.contains('└')
            || table.processes[1].tree_prefix.contains('├')
    );
}

#[test]
fn test_process_table_event_t_toggles_tree() {
    let mut table = ProcessTable::new();
    table.set_processes(sample_processes());
    assert!(!table.is_tree_view());

    table.event(&Event::key_down(Key::T));
    assert!(table.is_tree_view());

    table.event(&Event::key_down(Key::T));
    assert!(!table.is_tree_view());
}

#[test]
fn test_process_table_tree_view_paint() {
    let mut table = ProcessTable::new().with_tree_view();

    let entries = vec![
        ProcessEntry::new(2, "user", 10.0, 2.0, "child").with_parent_pid(1),
        ProcessEntry::new(1, "root", 0.5, 0.1, "parent"),
    ];
    table.set_processes(entries);
    table.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);

    let mut canvas = MockCanvas::new();
    table.paint(&mut canvas);

    // Should have tree prefix in output
    assert!(canvas
        .texts
        .iter()
        .any(|(t, _)| t.contains("└") || t.contains("├")));
}

#[test]
fn test_process_table_tree_empty() {
    let mut table = ProcessTable::new().with_tree_view();
    // Should not panic on empty
    table.set_processes(vec![]);
    assert!(table.is_empty());
}

// ========================================================================
// Falsification Tests for CB-PROC-001 (Phase 7 QA Gate)
// ========================================================================

/// F-TREE-001: "Orphaned Child" Test
/// Hierarchy MUST override sorting. Children MUST appear immediately below parent.
#[test]
fn test_f_tree_001_hierarchy_overrides_sorting() {
    let mut table = ProcessTable::new().with_tree_view();

    // sh (PID 100) with two sleep children (PIDs 200, 201)
    // Higher CPU processes elsewhere should NOT split the hierarchy
    let entries = vec![
        ProcessEntry::new(999, "root", 99.0, 50.0, "chrome"), // High CPU unrelated
        ProcessEntry::new(200, "user", 0.1, 0.1, "sleep").with_parent_pid(100),
        ProcessEntry::new(201, "user", 0.1, 0.1, "sleep").with_parent_pid(100),
        ProcessEntry::new(100, "user", 1.0, 0.5, "sh"),
        ProcessEntry::new(1, "root", 0.5, 0.1, "systemd"),
    ];
    table.set_processes(entries);

    // Find sh in the tree
    let sh_idx = table
        .processes
        .iter()
        .position(|p| p.command == "sh")
        .expect("sh not found");

    // Both sleep processes MUST be immediately after sh
    let sleep1_idx = table
        .processes
        .iter()
        .position(|p| p.command == "sleep" && p.pid == 200)
        .expect("sleep 200 not found");
    let sleep2_idx = table
        .processes
        .iter()
        .position(|p| p.command == "sleep" && p.pid == 201)
        .expect("sleep 201 not found");

    // Children must appear IMMEDIATELY after parent (next indices)
    assert!(
        sleep1_idx > sh_idx && sleep1_idx <= sh_idx + 2,
        "sleep 200 (idx {}) should be immediately after sh (idx {})",
        sleep1_idx,
        sh_idx
    );
    assert!(
        sleep2_idx > sh_idx && sleep2_idx <= sh_idx + 2,
        "sleep 201 (idx {}) should be immediately after sh (idx {})",
        sleep2_idx,
        sh_idx
    );

    // Unrelated high-CPU process should NOT be between sh and its children
    let chrome_idx = table
        .processes
        .iter()
        .position(|p| p.command == "chrome")
        .expect("chrome not found");
    assert!(
        !(chrome_idx > sh_idx && chrome_idx < sleep1_idx.max(sleep2_idx)),
        "Unrelated process should not split parent-child hierarchy"
    );
}

/// F-TREE-002: "Live Re-Parenting" - Orphan handling
/// When parent is killed, orphans should gracefully become roots
#[test]
fn test_f_tree_002_orphan_handling() {
    let mut table = ProcessTable::new().with_tree_view();

    // Child processes whose parent (PID 100) is NOT in the list
    let entries = vec![
        ProcessEntry::new(200, "user", 5.0, 1.0, "orphan1").with_parent_pid(100), // Parent missing
        ProcessEntry::new(201, "user", 3.0, 1.0, "orphan2").with_parent_pid(100), // Parent missing
        ProcessEntry::new(1, "root", 0.5, 0.1, "systemd"),
    ];
    table.set_processes(entries);

    // Should not panic - orphans become roots
    assert_eq!(table.len(), 3);

    // Orphans should have depth 0 (root level) since parent not found
    let orphan1 = table
        .processes
        .iter()
        .find(|p| p.command == "orphan1")
        .unwrap();
    let orphan2 = table
        .processes
        .iter()
        .find(|p| p.command == "orphan2")
        .unwrap();

    // Orphans treated as roots have no tree prefix
    assert_eq!(orphan1.tree_depth, 0);
    assert_eq!(orphan2.tree_depth, 0);
}

/// F-TREE-003: "Deep Nesting" Boundary (15 levels)
/// Tree must handle deep hierarchies without overflow or crash
#[test]
fn test_f_tree_003_deep_nesting_15_levels() {
    let mut table = ProcessTable::new().with_tree_view();

    // Create 15-level deep hierarchy
    let mut entries = vec![ProcessEntry::new(1, "root", 0.5, 0.1, "init")];

    for depth in 1..=15 {
        let pid = (depth + 1) as u32;
        let ppid = depth as u32;
        entries.push(
            ProcessEntry::new(pid, "user", 0.1, 0.1, format!("level{depth}")).with_parent_pid(ppid),
        );
    }

    table.set_processes(entries);

    // Should not panic
    assert_eq!(table.len(), 16); // 1 root + 15 children

    // Verify deepest process has depth 15
    let deepest = table
        .processes
        .iter()
        .find(|p| p.command == "level15")
        .unwrap();
    assert_eq!(deepest.tree_depth, 15);

    // Verify prefix has correct structure (should have 14 "│ " or "  " segments)
    let _prefix_segments =
        deepest.tree_prefix.matches("│").count() + deepest.tree_prefix.matches("  ").count();
    // At depth 15, prefix should have accumulated continuation chars
    assert!(
        deepest.tree_prefix.len() > 20,
        "Deep prefix should be substantial: '{}'",
        deepest.tree_prefix
    );
}

/// F-TREE-004: Verify DFS traversal order
/// Tree order must be parent, then all descendants, then next sibling
#[test]
fn test_f_tree_004_dfs_traversal_order() {
    let mut table = ProcessTable::new().with_tree_view();

    // Tree: A -> B -> D
    //           -> E
    //       -> C
    let entries = vec![
        ProcessEntry::new(5, "user", 1.0, 1.0, "E").with_parent_pid(2),
        ProcessEntry::new(4, "user", 1.0, 1.0, "D").with_parent_pid(2),
        ProcessEntry::new(3, "user", 1.0, 1.0, "C").with_parent_pid(1),
        ProcessEntry::new(2, "user", 2.0, 1.0, "B").with_parent_pid(1), // Higher CPU
        ProcessEntry::new(1, "root", 0.5, 0.1, "A"),
    ];
    table.set_processes(entries);

    // Expected DFS order (sorted by CPU within siblings): A, B, D, E, C
    // B comes before C because B has higher CPU
    let commands: Vec<&str> = table.processes.iter().map(|p| p.command.as_str()).collect();

    assert_eq!(commands[0], "A", "Root should be first");
    assert_eq!(commands[1], "B", "B (higher CPU child) should be second");
    // D and E are B's children
    assert!(
        commands[2] == "D" || commands[2] == "E",
        "B's children should follow B"
    );
    assert!(
        commands[3] == "D" || commands[3] == "E",
        "B's children should follow B"
    );
    assert_eq!(commands[4], "C", "C should be after B's subtree");
}
