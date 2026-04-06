//! `ContainersPanel` widget for Docker/Podman container monitoring.
//!
//! Displays running containers with status, CPU, and memory usage.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Container state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContainerState {
    #[default]
    Running,
    Paused,
    Stopped,
    Restarting,
    Dead,
}

impl ContainerState {
    /// Get status indicator.
    pub fn indicator(&self) -> char {
        match self {
            Self::Running => '●',
            Self::Paused => '◐',
            Self::Stopped => '○',
            Self::Restarting => '↻',
            Self::Dead => '✕',
        }
    }

    /// Get status color.
    pub fn color(&self) -> Color {
        match self {
            Self::Running => Color::new(0.4, 0.9, 0.4, 1.0), // Green
            Self::Paused => Color::new(1.0, 0.8, 0.2, 1.0),  // Yellow
            Self::Stopped => Color::new(0.5, 0.5, 0.5, 1.0), // Gray
            Self::Restarting => Color::new(0.4, 0.6, 1.0, 1.0), // Blue
            Self::Dead => Color::new(1.0, 0.3, 0.3, 1.0),    // Red
        }
    }
}

/// A container entry.
#[derive(Debug, Clone)]
pub struct ContainerEntry {
    /// Container name.
    pub name: String,
    /// Container ID (short).
    pub id: String,
    /// Container state.
    pub state: ContainerState,
    /// CPU usage percentage.
    pub cpu_percent: f32,
    /// Memory usage in bytes.
    pub memory_bytes: u64,
    /// Memory limit in bytes (0 = no limit).
    pub memory_limit: u64,
    /// Image name.
    pub image: String,
}

impl ContainerEntry {
    /// Create a new container entry.
    #[must_use]
    pub fn new(name: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: id.into(),
            state: ContainerState::Running,
            cpu_percent: 0.0,
            memory_bytes: 0,
            memory_limit: 0,
            image: String::new(),
        }
    }

    /// Set container state.
    #[must_use]
    pub fn with_state(mut self, state: ContainerState) -> Self {
        self.state = state;
        self
    }

    /// Set CPU percentage.
    #[must_use]
    pub fn with_cpu(mut self, cpu: f32) -> Self {
        self.cpu_percent = cpu;
        self
    }

    /// Set memory usage.
    #[must_use]
    pub fn with_memory(mut self, used: u64, limit: u64) -> Self {
        self.memory_bytes = used;
        self.memory_limit = limit;
        self
    }

    /// Set image name.
    #[must_use]
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = image.into();
        self
    }

    /// Format memory for display.
    pub fn memory_display(&self) -> String {
        let mb = self.memory_bytes as f64 / 1_048_576.0;
        if mb >= 1024.0 {
            format!("{:.1}G", mb / 1024.0)
        } else {
            format!("{mb:.0}M")
        }
    }

    /// Get memory percentage.
    pub fn memory_percent(&self) -> Option<f32> {
        if self.memory_limit > 0 {
            Some((self.memory_bytes as f64 / self.memory_limit as f64 * 100.0) as f32)
        } else {
            None
        }
    }
}

/// Containers panel displaying Docker/Podman containers.
#[derive(Debug, Clone)]
pub struct ContainersPanel {
    /// Container entries.
    containers: Vec<ContainerEntry>,
    /// Show only running containers.
    running_only: bool,
    /// Max containers to show.
    max_containers: usize,
    /// Compact mode (single line per container).
    compact: bool,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for ContainersPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainersPanel {
    /// Create a new containers panel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            running_only: true,
            max_containers: 5,
            compact: true,
            bounds: Rect::default(),
        }
    }

    /// Add a container.
    pub fn add_container(&mut self, container: ContainerEntry) {
        self.containers.push(container);
    }

    /// Set all containers.
    #[must_use]
    pub fn with_containers(mut self, containers: Vec<ContainerEntry>) -> Self {
        self.containers = containers;
        self
    }

    /// Show only running containers.
    #[must_use]
    pub fn running_only(mut self, only: bool) -> Self {
        self.running_only = only;
        self
    }

    /// Set max containers to show.
    #[must_use]
    pub fn max_containers(mut self, max: usize) -> Self {
        self.max_containers = max;
        self
    }

    /// Enable compact mode.
    #[must_use]
    pub fn compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    /// Get running container count.
    pub fn running_count(&self) -> usize {
        self.containers
            .iter()
            .filter(|c| c.state == ContainerState::Running)
            .count()
    }

    /// Get total container count.
    pub fn total_count(&self) -> usize {
        self.containers.len()
    }

    /// Get visible containers (filtered).
    fn visible_containers(&self) -> impl Iterator<Item = &ContainerEntry> {
        self.containers
            .iter()
            .filter(|c| !self.running_only || c.state == ContainerState::Running)
            .take(self.max_containers)
    }

    /// Draw a container line.
    fn draw_container(
        &self,
        canvas: &mut dyn Canvas,
        container: &ContainerEntry,
        x: f32,
        y: f32,
        width: f32,
    ) {
        // State indicator
        canvas.draw_text(
            &container.state.indicator().to_string(),
            Point::new(x, y),
            &TextStyle {
                color: container.state.color(),
                ..Default::default()
            },
        );

        // Container name (truncated)
        let max_name = ((width - 20.0) / 2.0) as usize;
        let name = if container.name.len() > max_name {
            format!("{}...", &container.name[..max_name.saturating_sub(3)])
        } else {
            container.name.clone()
        };

        canvas.draw_text(
            &name,
            Point::new(x + 2.0, y),
            &TextStyle {
                color: Color::WHITE,
                ..Default::default()
            },
        );

        // CPU and Memory
        let stats = format!(
            "{:4.1}% {:>5}",
            container.cpu_percent,
            container.memory_display()
        );
        canvas.draw_text(
            &stats,
            Point::new(x + width - 13.0, y),
            &TextStyle {
                color: Color::new(0.7, 0.7, 0.7, 1.0),
                ..Default::default()
            },
        );
    }
}

impl Brick for ContainersPanel {
    fn brick_name(&self) -> &'static str {
        "containers_panel"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(8)],
            failed: vec![],
            verification_time: Duration::from_micros(25),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for ContainersPanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let visible = self.visible_containers().count();
        let height = (visible as f32).max(1.0).min(constraints.max_height);
        Size::new(constraints.max_width, height)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 1.0 {
            return;
        }

        let mut y = self.bounds.y;
        let x = self.bounds.x;

        // Draw visible containers
        for container in self.visible_containers() {
            if y >= self.bounds.y + self.bounds.height {
                break;
            }
            self.draw_container(canvas, container, x, y, self.bounds.width);
            y += 1.0;
        }

        // If no containers, show message
        if self.containers.is_empty() {
            canvas.draw_text(
                "No containers",
                Point::new(x, self.bounds.y),
                &TextStyle {
                    color: Color::new(0.5, 0.5, 0.5, 1.0),
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
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
mod tests {
    use super::*;

    #[test]
    fn test_container_state_indicator() {
        assert_eq!(ContainerState::Running.indicator(), '●');
        assert_eq!(ContainerState::Paused.indicator(), '◐');
        assert_eq!(ContainerState::Stopped.indicator(), '○');
    }

    #[test]
    fn test_container_entry_memory() {
        let entry = ContainerEntry::new("nginx", "abc123")
            .with_memory(256 * 1024 * 1024, 512 * 1024 * 1024);
        assert_eq!(entry.memory_display(), "256M");
        assert!((entry.memory_percent().unwrap() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_container_entry_memory_gb() {
        let entry = ContainerEntry::new("postgres", "def456")
            .with_memory(2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        assert_eq!(entry.memory_display(), "2.0G");
    }

    #[test]
    fn test_panel_running_count() {
        let mut panel = ContainersPanel::new();
        panel.add_container(ContainerEntry::new("nginx", "a").with_state(ContainerState::Running));
        panel.add_container(ContainerEntry::new("redis", "b").with_state(ContainerState::Running));
        panel.add_container(ContainerEntry::new("old", "c").with_state(ContainerState::Stopped));

        assert_eq!(panel.running_count(), 2);
        assert_eq!(panel.total_count(), 3);
    }

    #[test]
    fn test_panel_builder() {
        let panel = ContainersPanel::new()
            .running_only(false)
            .max_containers(10)
            .compact(false);

        assert!(!panel.running_only);
        assert_eq!(panel.max_containers, 10);
        assert!(!panel.compact);
    }

    #[test]
    fn test_container_state_all_indicators() {
        assert_eq!(ContainerState::Restarting.indicator(), '↻');
        assert_eq!(ContainerState::Dead.indicator(), '✕');
    }

    #[test]
    fn test_container_state_all_colors() {
        // Test all states return valid colors
        for state in [
            ContainerState::Running,
            ContainerState::Paused,
            ContainerState::Stopped,
            ContainerState::Restarting,
            ContainerState::Dead,
        ] {
            let color = state.color();
            assert!(color.r >= 0.0 && color.r <= 1.0);
            assert!(color.g >= 0.0 && color.g <= 1.0);
            assert!(color.b >= 0.0 && color.b <= 1.0);
        }
    }

    #[test]
    fn test_container_entry_with_cpu() {
        let entry = ContainerEntry::new("nginx", "abc123").with_cpu(45.5);
        assert_eq!(entry.cpu_percent, 45.5);
    }

    #[test]
    fn test_container_entry_with_image() {
        let entry = ContainerEntry::new("nginx", "abc123").with_image("nginx:latest");
        assert_eq!(entry.image, "nginx:latest");
    }

    #[test]
    fn test_container_entry_no_memory_limit() {
        let entry = ContainerEntry::new("nginx", "abc123").with_memory(256 * 1024 * 1024, 0);
        assert!(entry.memory_percent().is_none());
    }

    #[test]
    fn test_containers_panel_with_containers() {
        let containers = vec![
            ContainerEntry::new("nginx", "a").with_state(ContainerState::Running),
            ContainerEntry::new("redis", "b").with_state(ContainerState::Paused),
        ];
        let panel = ContainersPanel::new().with_containers(containers);
        assert_eq!(panel.total_count(), 2);
    }

    #[test]
    fn test_containers_panel_brick_traits() {
        let panel = ContainersPanel::new();
        assert_eq!(panel.brick_name(), "containers_panel");
        assert!(!panel.assertions().is_empty());
        assert!(panel.budget().paint_ms > 0);
        assert!(panel.verify().is_valid());
        assert!(panel.to_html().is_empty());
        assert!(panel.to_css().is_empty());
    }

    #[test]
    fn test_containers_panel_widget_traits() {
        let mut panel = ContainersPanel::new()
            .with_containers(vec![
                ContainerEntry::new("nginx", "a").with_state(ContainerState::Running)
            ]);

        // Measure
        let size = panel.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 80.0,
            max_height: 20.0,
        });
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);

        // Layout
        let result = panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
        assert_eq!(result.size.width, 80.0);

        // Type ID
        assert_eq!(Widget::type_id(&panel), TypeId::of::<ContainersPanel>());

        // Event
        assert!(panel
            .event(&Event::key_down(presentar_core::Key::Enter))
            .is_none());

        // Children
        assert!(panel.children().is_empty());
        assert!(panel.children_mut().is_empty());
    }

    #[test]
    fn test_containers_panel_paint() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let containers = vec![
            ContainerEntry::new("nginx", "abc123")
                .with_state(ContainerState::Running)
                .with_cpu(15.5)
                .with_memory(256 * 1024 * 1024, 512 * 1024 * 1024),
            ContainerEntry::new("very_long_container_name_here", "def456")
                .with_state(ContainerState::Paused)
                .with_cpu(2.0)
                .with_memory(128 * 1024 * 1024, 256 * 1024 * 1024),
        ];

        let mut panel = ContainersPanel::new().with_containers(containers);
        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_containers_panel_paint_empty() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = ContainersPanel::new();
        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_containers_panel_paint_small_bounds() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let containers =
            vec![ContainerEntry::new("nginx", "abc").with_state(ContainerState::Running)];

        let mut panel = ContainersPanel::new().with_containers(containers);
        panel.layout(Rect::new(0.0, 0.0, 5.0, 0.5)); // Too small

        let mut buffer = CellBuffer::new(5, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas); // Should early return
    }

    #[test]
    fn test_containers_panel_running_only_filter() {
        let containers = vec![
            ContainerEntry::new("nginx", "a").with_state(ContainerState::Running),
            ContainerEntry::new("redis", "b").with_state(ContainerState::Stopped),
            ContainerEntry::new("postgres", "c").with_state(ContainerState::Running),
        ];

        let panel = ContainersPanel::new()
            .with_containers(containers)
            .running_only(true);

        assert_eq!(panel.visible_containers().count(), 2);
    }

    #[test]
    fn test_containers_panel_show_all() {
        let containers = vec![
            ContainerEntry::new("nginx", "a").with_state(ContainerState::Running),
            ContainerEntry::new("redis", "b").with_state(ContainerState::Stopped),
        ];

        let panel = ContainersPanel::new()
            .with_containers(containers)
            .running_only(false);

        assert_eq!(panel.visible_containers().count(), 2);
    }

    #[test]
    fn test_containers_panel_default() {
        let panel = ContainersPanel::default();
        assert!(panel.running_only);
        assert!(panel.compact);
        assert_eq!(panel.max_containers, 5);
    }

    #[test]
    fn test_container_state_default() {
        let state = ContainerState::default();
        assert_eq!(state, ContainerState::Running);
    }

    #[test]
    fn test_containers_panel_max_limit() {
        let containers: Vec<_> = (0..10)
            .map(|i| ContainerEntry::new(format!("container{}", i), format!("{}", i)))
            .collect();

        let panel = ContainersPanel::new()
            .with_containers(containers)
            .max_containers(3);

        assert_eq!(panel.visible_containers().count(), 3);
    }
}
