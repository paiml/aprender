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

include!("rendering.rs");
include!("tui_part_03.rs");
