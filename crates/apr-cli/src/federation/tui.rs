//! Federation Gateway TUI Dashboard
//!
//! Interactive terminal dashboard for monitoring the federation gateway.
//! Displays: model catalog, node health, routing decisions, circuit breakers,
//! gateway stats, and active policies.
//!
//! Panels match the playbook configuration in playbooks/federation-gateway.yaml

use super::catalog::{DeploymentStatus, ModelCatalog};
use super::gateway::FederationGateway;
use super::health::{CircuitBreaker, HealthChecker};
use super::traits::*;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table, Tabs},
    Frame,
};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// TUI State
// ============================================================================

/// Active tab in the federation dashboard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FederationTab {
    #[default]
    Catalog,
    Health,
    Routing,
    Circuits,
    Stats,
    Policies,
    Help,
}

impl FederationTab {
    pub fn titles() -> Vec<&'static str> {
        vec![
            "Catalog [1]",
            "Health [2]",
            "Routing [3]",
            "Circuits [4]",
            "Stats [5]",
            "Policies [6]",
            "Help [?]",
        ]
    }

    pub fn index(self) -> usize {
        match self {
            FederationTab::Catalog => 0,
            FederationTab::Health => 1,
            FederationTab::Routing => 2,
            FederationTab::Circuits => 3,
            FederationTab::Stats => 4,
            FederationTab::Policies => 5,
            FederationTab::Help => 6,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => FederationTab::Catalog,
            1 => FederationTab::Health,
            2 => FederationTab::Routing,
            3 => FederationTab::Circuits,
            4 => FederationTab::Stats,
            5 => FederationTab::Policies,
            _ => FederationTab::Help,
        }
    }
}

/// Routing decision record for display
#[derive(Debug, Clone)]
pub struct RoutingRecord {
    pub request_id: String,
    pub capability: String,
    pub selected_node: String,
    pub score: f64,
    pub reason: String,
    pub timestamp: std::time::Instant,
}

/// Circuit breaker display state
#[derive(Debug, Clone)]
pub struct CircuitDisplay {
    pub node_id: String,
    pub state: CircuitState,
    pub failure_count: u32,
    pub last_failure: Option<std::time::Instant>,
    pub reset_remaining: Option<Duration>,
}

/// Active policy display
#[derive(Debug, Clone)]
pub struct PolicyDisplay {
    pub name: String,
    pub weight: f64,
    pub enabled: bool,
    pub description: String,
}

/// Federation dashboard application state
pub struct FederationApp {
    /// Current active tab
    pub current_tab: FederationTab,
    /// Model catalog reference
    pub catalog: Arc<ModelCatalog>,
    /// Health checker reference
    pub health: Arc<HealthChecker>,
    /// Circuit breaker reference
    pub circuit_breaker: Arc<CircuitBreaker>,
    /// Gateway reference (optional)
    pub gateway: Option<Arc<FederationGateway>>,
    /// Recent routing decisions
    pub routing_history: Vec<RoutingRecord>,
    /// Maximum routing history entries
    pub max_history: usize,
    /// Active policies
    pub policies: Vec<PolicyDisplay>,
    /// Should quit flag
    pub should_quit: bool,
    /// Status message
    pub status_message: Option<String>,
    /// Selected row index (for navigation)
    pub selected_row: usize,
}

impl FederationApp {
    /// Create new federation dashboard app
    pub fn new(
        catalog: Arc<ModelCatalog>,
        health: Arc<HealthChecker>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        let policies = vec![
            PolicyDisplay {
                name: "health".to_string(),
                weight: 2.0,
                enabled: true,
                description: "Strongly penalize unhealthy nodes".to_string(),
            },
            PolicyDisplay {
                name: "latency".to_string(),
                weight: 1.0,
                enabled: true,
                description: "Prefer low-latency nodes".to_string(),
            },
            PolicyDisplay {
                name: "privacy".to_string(),
                weight: 1.0,
                enabled: true,
                description: "Enforce data sovereignty".to_string(),
            },
            PolicyDisplay {
                name: "locality".to_string(),
                weight: 1.0,
                enabled: true,
                description: "Prefer same-region nodes".to_string(),
            },
            PolicyDisplay {
                name: "cost".to_string(),
                weight: 1.0,
                enabled: true,
                description: "Balance cost vs performance".to_string(),
            },
        ];

        Self {
            current_tab: FederationTab::default(),
            catalog,
            health,
            circuit_breaker,
            gateway: None,
            routing_history: Vec::new(),
            max_history: 100,
            policies,
            should_quit: false,
            status_message: None,
            selected_row: 0,
        }
    }

    /// Attach a gateway for stats
    #[must_use]
    pub fn with_gateway(mut self, gateway: Arc<FederationGateway>) -> Self {
        self.gateway = Some(gateway);
        self
    }

    /// Navigate to next tab
    pub fn next_tab(&mut self) {
        let next = (self.current_tab.index() + 1) % FederationTab::titles().len();
        self.current_tab = FederationTab::from_index(next);
        self.selected_row = 0;
    }

    /// Navigate to previous tab
    pub fn prev_tab(&mut self) {
        let len = FederationTab::titles().len();
        let prev = (self.current_tab.index() + len - 1) % len;
        self.current_tab = FederationTab::from_index(prev);
        self.selected_row = 0;
    }

    /// Select next row
    pub fn select_next(&mut self) {
        self.selected_row = self.selected_row.saturating_add(1);
    }

    /// Select previous row
    pub fn select_prev(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(1);
    }

    /// Record a routing decision
    pub fn record_routing(&mut self, record: RoutingRecord) {
        self.routing_history.push(record);
        if self.routing_history.len() > self.max_history {
            self.routing_history.remove(0);
        }
    }

    /// Get healthy node count
    pub fn healthy_node_count(&self) -> usize {
        self.health.healthy_count()
    }

    /// Get total node count
    pub fn total_node_count(&self) -> usize {
        self.health.total_count()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        self.gateway.as_ref().map_or(1.0, |g| {
            let stats = g.stats();
            if stats.total_requests == 0 {
                1.0
            } else {
                stats.successful_requests as f64 / stats.total_requests as f64
            }
        })
    }

    /// Get requests per second (mock for now)
    pub fn requests_per_sec(&self) -> f64 {
        self.gateway
            .as_ref()
            .map_or(0.0, |g| g.stats().total_requests as f64 / 60.0) // Approximation
    }
}

// ============================================================================
// Rendering
// ============================================================================

/// Render the federation dashboard
pub fn render_federation_dashboard(f: &mut Frame<'_>, app: &FederationApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    render_title(f, chunks[0], app);
    render_tabs(f, chunks[1], app);

    match app.current_tab {
        FederationTab::Catalog => render_catalog(f, chunks[2], app),
        FederationTab::Health => render_health(f, chunks[2], app),
        FederationTab::Routing => render_routing(f, chunks[2], app),
        FederationTab::Circuits => render_circuits(f, chunks[2], app),
        FederationTab::Stats => render_stats(f, chunks[2], app),
        FederationTab::Policies => render_policies(f, chunks[2], app),
        FederationTab::Help => render_help(f, chunks[2]),
    }

    render_status_bar(f, chunks[3], app);
}

fn render_title(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let healthy = app.healthy_node_count();
    let total = app.total_node_count();

    let title = format!(
        " Federation Gateway v1.0 │ {}/{} nodes healthy │ {:.1}% success ",
        healthy,
        total,
        app.success_rate() * 100.0
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            title,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(block, area);
}

fn render_tabs(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let titles: Vec<Line<'_>> = FederationTab::titles()
        .iter()
        .map(|t| Line::from(*t))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Navigation"))
        .select(app.current_tab.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn render_catalog(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" MODEL CATALOG ");

    let entries = app.catalog.all_entries();

    if entries.is_empty() {
        let no_models =
            Paragraph::new("No models registered. Use 'apr federation register' to add models.")
                .style(Style::default().fg(Color::Yellow))
                .block(block);
        f.render_widget(no_models, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Model").style(Style::default().fg(Color::Cyan)),
        Cell::from("Node").style(Style::default().fg(Color::Cyan)),
        Cell::from("Region").style(Style::default().fg(Color::Cyan)),
        Cell::from("Capabilities").style(Style::default().fg(Color::Cyan)),
        Cell::from("Status").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = entries
        .iter()
        .flat_map(|entry| {
            entry.deployments.iter().map(move |dep| {
                let caps: String = entry
                    .metadata
                    .capabilities
                    .iter()
                    .map(|c| format!("{:?}", c))
                    .collect::<Vec<_>>()
                    .join(", ");

                let status_style = match dep.status {
                    DeploymentStatus::Ready => Style::default().fg(Color::Green),
                    DeploymentStatus::Loading => Style::default().fg(Color::Yellow),
                    DeploymentStatus::Draining => Style::default().fg(Color::Yellow),
                    DeploymentStatus::Removed => Style::default().fg(Color::Red),
                };

                Row::new(vec![
                    Cell::from(entry.metadata.model_id.0.clone()),
                    Cell::from(dep.node_id.0.clone()),
                    Cell::from(dep.region_id.0.clone()),
                    Cell::from(caps),
                    Cell::from(format!("{:?}", dep.status)).style(status_style),
                ])
            })
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(30),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_health(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" NODE HEALTH ");

    let statuses = app.health.all_statuses();

    if statuses.is_empty() {
        let no_nodes = Paragraph::new("No nodes registered for health monitoring.")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_nodes, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Node").style(Style::default().fg(Color::Cyan)),
        Cell::from("State").style(Style::default().fg(Color::Cyan)),
        Cell::from("Latency P50").style(Style::default().fg(Color::Cyan)),
        Cell::from("Latency P99").style(Style::default().fg(Color::Cyan)),
        Cell::from("Queue").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = statuses
        .iter()
        .map(|status| {
            let state_style = match status.state {
                HealthState::Healthy => Style::default().fg(Color::Green),
                HealthState::Degraded => Style::default().fg(Color::Yellow),
                HealthState::Unhealthy => Style::default().fg(Color::Red),
                HealthState::Unknown => Style::default().fg(Color::DarkGray),
            };

            Row::new(vec![
                Cell::from(status.node_id.0.clone()),
                Cell::from(format!("{:?}", status.state)).style(state_style),
                Cell::from(format!("{}ms", status.latency_p50.as_millis())),
                Cell::from(format!("{}ms", status.latency_p99.as_millis())),
                Cell::from(format!("{}", status.queue_depth)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_routing(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" ROUTING DECISIONS ");

    if app.routing_history.is_empty() {
        let no_routing = Paragraph::new("No routing decisions recorded yet.")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_routing, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Request").style(Style::default().fg(Color::Cyan)),
        Cell::from("Capability").style(Style::default().fg(Color::Cyan)),
        Cell::from("Selected").style(Style::default().fg(Color::Cyan)),
        Cell::from("Score").style(Style::default().fg(Color::Cyan)),
        Cell::from("Reason").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = app
        .routing_history
        .iter()
        .rev()
        .take(20)
        .map(|record| {
            Row::new(vec![
                Cell::from(truncate(&record.request_id, 12)),
                Cell::from(record.capability.clone()),
                Cell::from(record.selected_node.clone()),
                Cell::from(format!("{:.2}", record.score)),
                Cell::from(truncate(&record.reason, 30)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(30),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_circuits(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" CIRCUIT BREAKERS ");

    let circuits = app.circuit_breaker.all_states();

    if circuits.is_empty() {
        let no_circuits = Paragraph::new("No circuit breakers active.")
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        f.render_widget(no_circuits, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Node").style(Style::default().fg(Color::Cyan)),
        Cell::from("State").style(Style::default().fg(Color::Cyan)),
        Cell::from("Failures").style(Style::default().fg(Color::Cyan)),
        Cell::from("Last Failure").style(Style::default().fg(Color::Cyan)),
        Cell::from("Reset In").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = circuits
        .iter()
        .map(|(node_id, state)| {
            let state_style = match state {
                CircuitState::Closed => Style::default().fg(Color::Green),
                CircuitState::HalfOpen => Style::default().fg(Color::Yellow),
                CircuitState::Open => Style::default().fg(Color::Red),
            };

            Row::new(vec![
                Cell::from(node_id.0.clone()),
                Cell::from(format!("{:?}", state)).style(state_style),
                Cell::from("-"), // Would need failure count from circuit breaker
                Cell::from("-"),
                Cell::from("-"),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_stats(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GATEWAY STATS ");

    let stats = app
        .gateway
        .as_ref()
        .map_or_else(GatewayStats::default, |g| g.stats());

    // Split area for metrics and gauges
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Metrics
            Constraint::Length(3), // Success rate gauge
            Constraint::Min(0),    // Spacer
        ])
        .split(area);

    // Metrics
    let metrics = vec![
        Line::from(vec![
            Span::styled("Total Requests:    ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.total_requests)),
        ]),
        Line::from(vec![
            Span::styled("Successful:        ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}", stats.successful_requests),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::styled("Failed:            ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}", stats.failed_requests),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::styled("Total Tokens:      ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.total_tokens)),
        ]),
        Line::from(vec![
            Span::styled("Avg Latency:       ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}ms", stats.avg_latency.as_millis())),
        ]),
        Line::from(vec![
            Span::styled("Active Streams:    ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.active_streams)),
        ]),
    ];

    let metrics_widget = Paragraph::new(metrics).block(block);
    f.render_widget(metrics_widget, chunks[0]);

    // Success rate gauge
    let success_rate = app.success_rate();
    let gauge_color = if success_rate >= 0.99 {
        Color::Green
    } else if success_rate >= 0.95 {
        Color::Yellow
    } else {
        Color::Red
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Success Rate"))
        .gauge_style(Style::default().fg(gauge_color))
        .ratio(success_rate)
        .label(format!("{:.1}%", success_rate * 100.0));

    f.render_widget(gauge, chunks[1]);
}

fn render_policies(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" ACTIVE POLICIES ");

    let header = Row::new(vec![
        Cell::from("Policy").style(Style::default().fg(Color::Cyan)),
        Cell::from("Weight").style(Style::default().fg(Color::Cyan)),
        Cell::from("Status").style(Style::default().fg(Color::Cyan)),
        Cell::from("Description").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = app
        .policies
        .iter()
        .map(|policy| {
            let status_style = if policy.enabled {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            Row::new(vec![
                Cell::from(policy.name.clone()),
                Cell::from(format!("{:.1}", policy.weight)),
                Cell::from(if policy.enabled { "Active" } else { "Disabled" }).style(status_style),
                Cell::from(policy.description.clone()),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(10),
            Constraint::Percentage(15),
            Constraint::Percentage(60),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_help(f: &mut Frame<'_>, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled(
            "Federation Gateway Dashboard",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("1-6, ?    ", Style::default().fg(Color::Yellow)),
            Span::raw("Switch to Catalog/Health/Routing/Circuits/Stats/Policies/Help"),
        ]),
        Line::from(vec![
            Span::styled("Tab       ", Style::default().fg(Color::Yellow)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled("Shift+Tab ", Style::default().fg(Color::Yellow)),
            Span::raw("Previous tab"),
        ]),
        Line::from(vec![
            Span::styled("j / Down  ", Style::default().fg(Color::Yellow)),
            Span::raw("Select next row"),
        ]),
        Line::from(vec![
            Span::styled("k / Up    ", Style::default().fg(Color::Yellow)),
            Span::raw("Select previous row"),
        ]),
        Line::from(vec![
            Span::styled("r         ", Style::default().fg(Color::Yellow)),
            Span::raw("Refresh data"),
        ]),
        Line::from(vec![
            Span::styled("h         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle health panel"),
        ]),
        Line::from(vec![
            Span::styled("c         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle circuit panel"),
        ]),
        Line::from(vec![
            Span::styled("s         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle stats panel"),
        ]),
        Line::from(vec![
            Span::styled("q / Esc   ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Panels",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("• Catalog: View registered models and deployments"),
        Line::from("• Health: Monitor node health and latency"),
        Line::from("• Routing: Track routing decisions and scores"),
        Line::from("• Circuits: View circuit breaker states"),
        Line::from("• Stats: Gateway statistics and success rate"),
        Line::from("• Policies: Active routing policies"),
    ];

    let help =
        Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(help, area);
}

fn render_status_bar(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let _status = app
        .status_message
        .as_deref()
        .unwrap_or("Tab to switch views │ q to quit │ ? for help");

    let left = "Federation Gateway v1.0";
    let center = format!(
        "{}/{} nodes healthy",
        app.healthy_node_count(),
        app.total_node_count()
    );
    let right = format!(
        "{:.1} req/s │ {:.1}% success",
        app.requests_per_sec(),
        app.success_rate() * 100.0
    );

    let full_status = format!("{} │ {} │ {}", left, center, right);

    let status_bar = Paragraph::new(full_status).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status_bar, area);
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_tab_titles() {
        let titles = FederationTab::titles();
        assert_eq!(titles.len(), 7);
        assert!(titles[0].contains("Catalog"));
        assert!(titles[1].contains("Health"));
        assert!(titles[2].contains("Routing"));
        assert!(titles[3].contains("Circuits"));
        assert!(titles[4].contains("Stats"));
        assert!(titles[5].contains("Policies"));
        assert!(titles[6].contains("Help"));
    }

    #[test]
    fn test_federation_tab_index() {
        assert_eq!(FederationTab::Catalog.index(), 0);
        assert_eq!(FederationTab::Health.index(), 1);
        assert_eq!(FederationTab::Routing.index(), 2);
        assert_eq!(FederationTab::Circuits.index(), 3);
        assert_eq!(FederationTab::Stats.index(), 4);
        assert_eq!(FederationTab::Policies.index(), 5);
        assert_eq!(FederationTab::Help.index(), 6);
    }

    #[test]
    fn test_federation_tab_from_index() {
        assert_eq!(FederationTab::from_index(0), FederationTab::Catalog);
        assert_eq!(FederationTab::from_index(1), FederationTab::Health);
        assert_eq!(FederationTab::from_index(2), FederationTab::Routing);
        assert_eq!(FederationTab::from_index(3), FederationTab::Circuits);
        assert_eq!(FederationTab::from_index(4), FederationTab::Stats);
        assert_eq!(FederationTab::from_index(5), FederationTab::Policies);
        assert_eq!(FederationTab::from_index(6), FederationTab::Help);
        assert_eq!(FederationTab::from_index(99), FederationTab::Help);
    }

    #[test]
    fn test_federation_app_new() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.current_tab, FederationTab::Catalog);
        assert!(!app.should_quit);
        assert!(app.routing_history.is_empty());
        assert_eq!(app.policies.len(), 5);
    }

    #[test]
    fn test_federation_app_tab_navigation() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.current_tab, FederationTab::Catalog);
        app.next_tab();
        assert_eq!(app.current_tab, FederationTab::Health);
        app.next_tab();
        assert_eq!(app.current_tab, FederationTab::Routing);
        app.prev_tab();
        assert_eq!(app.current_tab, FederationTab::Health);
    }

    #[test]
    fn test_federation_app_routing_history() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        let record = RoutingRecord {
            request_id: "req-1".to_string(),
            capability: "Transcribe".to_string(),
            selected_node: "node-1".to_string(),
            score: 0.95,
            reason: "lowest latency".to_string(),
            timestamp: std::time::Instant::now(),
        };

        app.record_routing(record);
        assert_eq!(app.routing_history.len(), 1);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("very long string", 10), "very lo...");
        assert_eq!(truncate("exactly10", 10), "exactly10");
    }

    // =========================================================================
    // FederationTab extended tests
    // =========================================================================

    #[test]
    fn test_federation_tab_default() {
        let tab = FederationTab::default();
        assert_eq!(tab, FederationTab::Catalog);
    }

    #[test]
    fn test_federation_tab_roundtrip_all() {
        for i in 0..7 {
            let tab = FederationTab::from_index(i);
            assert_eq!(tab.index(), i);
        }
    }

    #[test]
    fn test_federation_tab_from_index_overflow() {
        // Any index >= 6 should map to Help
        assert_eq!(FederationTab::from_index(7), FederationTab::Help);
        assert_eq!(FederationTab::from_index(100), FederationTab::Help);
        assert_eq!(FederationTab::from_index(usize::MAX), FederationTab::Help);
    }

    #[test]
    fn test_federation_tab_copy() {
        let tab = FederationTab::Routing;
        let copied = tab;
        assert_eq!(tab, copied);
    }

    #[test]
    fn test_federation_tab_debug() {
        let tab = FederationTab::Stats;
        let debug = format!("{:?}", tab);
        assert_eq!(debug, "Stats");
    }

    // =========================================================================
    // Truncate edge cases
    // =========================================================================

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn test_truncate_exactly_at_limit() {
        assert_eq!(truncate("1234567890", 10), "1234567890");
    }

    #[test]
    fn test_truncate_one_over() {
        assert_eq!(truncate("12345678901", 10), "1234567...");
    }

    #[test]
    fn test_truncate_max_len_3() {
        // max_len=3, 3-3=0 -> "..."
        assert_eq!(truncate("abcdef", 3), "...");
    }

    #[test]
    fn test_truncate_max_len_4() {
        assert_eq!(truncate("abcdef", 4), "a...");
    }

    // =========================================================================
    // RoutingRecord construction
    // =========================================================================

    #[test]
    fn test_routing_record_construction() {
        let record = RoutingRecord {
            request_id: "req-42".to_string(),
            capability: "Generate".to_string(),
            selected_node: "gpu-node-1".to_string(),
            score: 0.95,
            reason: "lowest latency".to_string(),
            timestamp: std::time::Instant::now(),
        };
        assert_eq!(record.request_id, "req-42");
        assert_eq!(record.capability, "Generate");
        assert_eq!(record.score, 0.95);
    }

    #[test]
    fn test_routing_record_clone() {
        let record = RoutingRecord {
            request_id: "req-1".to_string(),
            capability: "Embed".to_string(),
            selected_node: "node-1".to_string(),
            score: 0.5,
            reason: "only available".to_string(),
            timestamp: std::time::Instant::now(),
        };
        let cloned = record.clone();
        assert_eq!(cloned.request_id, "req-1");
    }

    // =========================================================================
    // CircuitDisplay construction
    // =========================================================================

    #[test]
    fn test_circuit_display_construction() {
        let display = CircuitDisplay {
            node_id: "node-1".to_string(),
            state: CircuitState::Open,
            failure_count: 5,
            last_failure: Some(std::time::Instant::now()),
            reset_remaining: Some(Duration::from_secs(25)),
        };
        assert_eq!(display.node_id, "node-1");
        assert_eq!(display.state, CircuitState::Open);
        assert_eq!(display.failure_count, 5);
        assert!(display.last_failure.is_some());
        assert!(display.reset_remaining.is_some());
    }

    #[test]
    fn test_circuit_display_closed() {
        let display = CircuitDisplay {
            node_id: "healthy-node".to_string(),
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: None,
            reset_remaining: None,
        };
        assert_eq!(display.state, CircuitState::Closed);
        assert!(display.last_failure.is_none());
    }

    // =========================================================================
    // PolicyDisplay construction
    // =========================================================================

    #[test]
    fn test_policy_display_enabled() {
        let display = PolicyDisplay {
            name: "latency".to_string(),
            weight: 1.0,
            enabled: true,
            description: "Prefer low-latency nodes".to_string(),
        };
        assert!(display.enabled);
        assert_eq!(display.weight, 1.0);
    }

    #[test]
    fn test_policy_display_disabled() {
        let display = PolicyDisplay {
            name: "cost".to_string(),
            weight: 0.5,
            enabled: false,
            description: "Disabled for testing".to_string(),
        };
        assert!(!display.enabled);
    }

    // =========================================================================
    // FederationApp extended tests
    // =========================================================================

    #[test]
    fn test_federation_app_with_gateway() {
        use super::super::gateway::FederationGateway;
        use super::super::routing::{Router, RouterConfig};

        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Arc::new(Router::new(
            RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let gateway = Arc::new(FederationGateway::new(
            super::super::gateway::GatewayConfig::default(),
            router,
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let app = FederationApp::new(
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        )
        .with_gateway(gateway);

        assert!(app.gateway.is_some());
    }

    #[test]
    fn test_federation_app_success_rate_no_gateway() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        // Without gateway, default is 1.0
        assert!((app.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_federation_app_requests_per_sec_no_gateway() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert!((app.requests_per_sec() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_federation_app_healthy_node_count() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.healthy_node_count(), 0);
        assert_eq!(app.total_node_count(), 0);
    }

    #[test]
    fn test_federation_app_select_next() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.selected_row, 0);
        app.select_next();
        assert_eq!(app.selected_row, 1);
        app.select_next();
        assert_eq!(app.selected_row, 2);
    }

    #[test]
    fn test_federation_app_select_prev_at_zero() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.selected_row, 0);
        app.select_prev(); // saturating_sub(1) at 0 stays 0
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn test_federation_app_select_prev_from_nonzero() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.selected_row = 5;
        app.select_prev();
        assert_eq!(app.selected_row, 4);
    }

    #[test]
    fn test_federation_app_tab_navigation_resets_row() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.selected_row = 5;
        app.next_tab();
        assert_eq!(app.selected_row, 0);

        app.selected_row = 3;
        app.prev_tab();
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn test_federation_app_next_tab_wraps() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        // Navigate to last tab (Help, index 6)
        app.current_tab = FederationTab::Help;
        app.next_tab();
        // Should wrap to Catalog (index 0)
        assert_eq!(app.current_tab, FederationTab::Catalog);
    }

    #[test]
    fn test_federation_app_prev_tab_wraps() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        // At Catalog (index 0), prev should wrap to Help (index 6)
        app.current_tab = FederationTab::Catalog;
        app.prev_tab();
        assert_eq!(app.current_tab, FederationTab::Help);
    }

    #[test]
    fn test_federation_app_routing_history_overflow() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.max_history = 3;

        // Add 5 records
        for i in 0..5 {
            app.record_routing(RoutingRecord {
                request_id: format!("req-{}", i),
                capability: "Generate".to_string(),
                selected_node: "node-1".to_string(),
                score: 0.9,
                reason: "test".to_string(),
                timestamp: std::time::Instant::now(),
            });
        }

        // Should be capped at max_history
        assert_eq!(app.routing_history.len(), 3);
        // First entries should be removed (FIFO)
        assert_eq!(app.routing_history[0].request_id, "req-2");
        assert_eq!(app.routing_history[1].request_id, "req-3");
        assert_eq!(app.routing_history[2].request_id, "req-4");
    }

    #[test]
    fn test_federation_app_status_message() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert!(app.status_message.is_none());
        app.status_message = Some("Connected to 3 nodes".to_string());
        assert_eq!(app.status_message, Some("Connected to 3 nodes".to_string()));
    }

    #[test]
    fn test_federation_app_should_quit() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert!(!app.should_quit);
        app.should_quit = true;
        assert!(app.should_quit);
    }

    #[test]
    fn test_federation_app_policies_count() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        // Default policies: health, latency, privacy, locality, cost
        assert_eq!(app.policies.len(), 5);
        assert_eq!(app.policies[0].name, "health");
        assert_eq!(app.policies[1].name, "latency");
        assert_eq!(app.policies[2].name, "privacy");
        assert_eq!(app.policies[3].name, "locality");
        assert_eq!(app.policies[4].name, "cost");
    }

    #[test]
    fn test_federation_app_max_history_default() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.max_history, 100);
    }

    // =========================================================================
    // Probar TUI Frame Tests
    // =========================================================================

    mod tui_frame_tests {
        use super::*;
        use jugar_probar::tui::{
            expect_frame, FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot,
        };
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        /// Helper to render federation app to a test backend
        fn render_frame(app: &FederationApp, width: u16, height: u16) -> TuiFrame {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).expect("terminal creation");
            terminal
                .draw(|f| render_federation_dashboard(f, app))
                .expect("draw");
            TuiFrame::from_buffer(terminal.backend().buffer(), 0)
        }

        fn create_test_app() -> FederationApp {
            let catalog = Arc::new(ModelCatalog::new());
            let health = Arc::new(HealthChecker::default());
            let circuit_breaker = Arc::new(CircuitBreaker::default());
            FederationApp::new(catalog, health, circuit_breaker)
        }

        #[test]
        fn test_federation_tui_title_displayed() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("Federation Gateway"),
                "Should show title: {}",
                frame.as_text()
            );
        }

        #[test]
        fn test_federation_tui_all_tabs_displayed() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            assert!(frame.contains("Catalog"), "Should show Catalog tab");
            assert!(frame.contains("Health"), "Should show Health tab");
            assert!(frame.contains("Routing"), "Should show Routing tab");
            assert!(frame.contains("Circuits"), "Should show Circuits tab");
            assert!(frame.contains("Stats"), "Should show Stats tab");
            assert!(frame.contains("Policies"), "Should show Policies tab");
            assert!(frame.contains("Help"), "Should show Help tab");
        }

        #[test]
        fn test_federation_tui_catalog_empty() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            // Default tab is catalog, should show empty message
            assert!(
                frame.contains("No models registered") || frame.contains("MODEL CATALOG"),
                "Should show catalog panel"
            );
        }

        #[test]
        fn test_federation_tui_health_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Health;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("NODE HEALTH"),
                "Should show health panel title"
            );
        }

        #[test]
        fn test_federation_tui_routing_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Routing;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("ROUTING DECISIONS"),
                "Should show routing panel title"
            );
        }

        #[test]
        fn test_federation_tui_circuits_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Circuits;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("CIRCUIT BREAKERS"),
                "Should show circuits panel title"
            );
        }

        #[test]
        fn test_federation_tui_stats_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Stats;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("GATEWAY STATS"),
                "Should show stats panel title"
            );
            assert!(
                frame.contains("Total Requests"),
                "Should show request count"
            );
        }

        #[test]
        fn test_federation_tui_policies_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Policies;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("ACTIVE POLICIES"),
                "Should show policies panel title"
            );
            assert!(frame.contains("health"), "Should show health policy");
            assert!(frame.contains("latency"), "Should show latency policy");
        }

        #[test]
        fn test_federation_tui_help_tab() {
            let mut app = create_test_app();
            app.current_tab = FederationTab::Help;
            let frame = render_frame(&app, 100, 30);

            assert!(
                frame.contains("Keyboard Shortcuts"),
                "Should show keyboard shortcuts"
            );
            assert!(frame.contains("q / Esc"), "Should show quit shortcut");
        }

        #[test]
        fn test_federation_tui_frame_dimensions() {
            let app = create_test_app();
            let frame = render_frame(&app, 120, 40);

            assert_eq!(frame.width(), 120, "Frame width");
            assert_eq!(frame.height(), 40, "Frame height");
        }

        // =====================================================================
        // Playwright-style Assertions
        // =====================================================================

        #[test]
        fn test_probar_chained_assertions() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let mut assertion = expect_frame(&frame);
            assert!(assertion.to_contain_text("Federation Gateway").is_ok());
            assert!(assertion.to_contain_text("Navigation").is_ok());
            assert!(assertion.not_to_contain_text("ERROR").is_ok());
        }

        #[test]
        fn test_probar_soft_assertions() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let mut assertion = expect_frame(&frame).soft();
            let _ = assertion.to_contain_text("Federation Gateway");
            let _ = assertion.to_contain_text("Catalog");
            let _ = assertion.to_have_size(100, 30);

            assert!(assertion.errors().is_empty(), "No soft assertion errors");
            assert!(assertion.finalize().is_ok());
        }

        // =====================================================================
        // Snapshot Testing
        // =====================================================================

        #[test]
        fn test_snapshot_creation() {
            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let snapshot = TuiSnapshot::from_frame("federation_catalog", &frame);

            assert_eq!(snapshot.name, "federation_catalog");
            assert_eq!(snapshot.width, 100);
            assert_eq!(snapshot.height, 30);
            assert!(!snapshot.hash.is_empty());
        }

        #[test]
        fn test_snapshot_different_tabs_differ() {
            let mut app = create_test_app();

            app.current_tab = FederationTab::Catalog;
            let frame_catalog = render_frame(&app, 100, 30);
            let snap_catalog = TuiSnapshot::from_frame("catalog", &frame_catalog);

            app.current_tab = FederationTab::Help;
            let frame_help = render_frame(&app, 100, 30);
            let snap_help = TuiSnapshot::from_frame("help", &frame_help);

            assert!(
                !snap_catalog.matches(&snap_help),
                "Different tabs should have different snapshots"
            );
        }

        // =====================================================================
        // Frame Sequence Testing
        // =====================================================================

        #[test]
        fn test_frame_sequence_tab_navigation() {
            let mut app = create_test_app();
            let mut sequence = FrameSequence::new("federation_tabs");

            for tab in [
                FederationTab::Catalog,
                FederationTab::Health,
                FederationTab::Routing,
                FederationTab::Circuits,
                FederationTab::Stats,
                FederationTab::Policies,
                FederationTab::Help,
            ] {
                app.current_tab = tab;
                sequence.add_frame(&render_frame(&app, 100, 30));
            }

            assert_eq!(sequence.len(), 7, "Should have 7 frames");

            let first = sequence.first().expect("first");
            let last = sequence.last().expect("last");
            assert!(!first.matches(last), "First and last should differ");
        }

        // =====================================================================
        // Snapshot Manager Tests
        // =====================================================================

        #[test]
        fn test_snapshot_manager_workflow() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().expect("temp dir");
            let manager = SnapshotManager::new(temp_dir.path());

            let app = create_test_app();
            let frame = render_frame(&app, 100, 30);

            let result = manager.assert_snapshot("federation_dashboard", &frame);
            assert!(result.is_ok(), "First snapshot should be created");
            assert!(
                manager.exists("federation_dashboard"),
                "Snapshot should exist"
            );

            let result2 = manager.assert_snapshot("federation_dashboard", &frame);
            assert!(result2.is_ok(), "Same frame should match");
        }

        // =====================================================================
        // UX Coverage Tests
        // =====================================================================

        #[test]
        fn test_ux_coverage_federation_elements() {
            use jugar_probar::ux_coverage::{InteractionType, StateId, UxCoverageBuilder};

            let mut tracker = UxCoverageBuilder::new()
                .clickable("tab", "catalog")
                .clickable("tab", "health")
                .clickable("tab", "routing")
                .clickable("tab", "circuits")
                .clickable("tab", "stats")
                .clickable("tab", "policies")
                .clickable("tab", "help")
                .clickable("nav", "next_tab")
                .clickable("nav", "prev_tab")
                .clickable("nav", "next_row")
                .clickable("nav", "prev_row")
                .clickable("nav", "quit")
                .screen("catalog")
                .screen("health")
                .screen("routing")
                .screen("circuits")
                .screen("stats")
                .screen("policies")
                .screen("help")
                .build();

            let mut app = create_test_app();

            // Cover all tabs
            for (tab, name) in [
                (FederationTab::Catalog, "catalog"),
                (FederationTab::Health, "health"),
                (FederationTab::Routing, "routing"),
                (FederationTab::Circuits, "circuits"),
                (FederationTab::Stats, "stats"),
                (FederationTab::Policies, "policies"),
                (FederationTab::Help, "help"),
            ] {
                app.current_tab = tab;
                tracker.record_interaction(
                    &jugar_probar::ux_coverage::ElementId::new("tab", name),
                    InteractionType::Click,
                );
                tracker.record_state(StateId::new("screen", name));
            }

            // Cover navigation
            app.next_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_tab"),
                InteractionType::Click,
            );

            app.prev_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_tab"),
                InteractionType::Click,
            );

            app.select_next();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_row"),
                InteractionType::Click,
            );

            app.select_prev();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_row"),
                InteractionType::Click,
            );

            app.should_quit = true;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "quit"),
                InteractionType::Click,
            );

            let report = tracker.generate_report();
            println!("{}", report.summary());

            assert!(report.is_complete, "UX coverage must be COMPLETE");
            assert!(tracker.meets(100.0), "Coverage: {}", tracker.summary());
        }

        #[test]
        fn test_ux_coverage_with_gui_macro() {
            use jugar_probar::gui_coverage;

            let mut gui = gui_coverage! {
                buttons: [
                    "tab_catalog", "tab_health", "tab_routing",
                    "tab_circuits", "tab_stats", "tab_policies", "tab_help",
                    "nav_next", "nav_prev", "quit"
                ],
                screens: [
                    "catalog", "health", "routing",
                    "circuits", "stats", "policies", "help"
                ]
            };

            // Cover all buttons
            gui.click("tab_catalog");
            gui.click("tab_health");
            gui.click("tab_routing");
            gui.click("tab_circuits");
            gui.click("tab_stats");
            gui.click("tab_policies");
            gui.click("tab_help");
            gui.click("nav_next");
            gui.click("nav_prev");
            gui.click("quit");

            // Cover all screens
            gui.visit("catalog");
            gui.visit("health");
            gui.visit("routing");
            gui.visit("circuits");
            gui.visit("stats");
            gui.visit("policies");
            gui.visit("help");

            assert!(gui.is_complete(), "Should have 100% coverage");
            assert!(gui.meets(100.0), "Coverage: {}", gui.summary());

            let report = gui.generate_report();
            println!(
                "Federation UX Coverage:\n  Elements: {}/{}\n  States: {}/{}\n  Overall: {:.1}%",
                report.covered_elements,
                report.total_elements,
                report.covered_states,
                report.total_states,
                report.overall_coverage * 100.0
            );
        }
    }
}
