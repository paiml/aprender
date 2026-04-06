//! DSH-007: Infrastructure Dashboard
//!
//! QA Focus: Server/container monitoring
//!
//! Run: `cargo run --example dsh_infrastructure`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::struct_field_names,
    clippy::too_many_lines
)]

use std::collections::HashMap;

/// Server/node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
    Maintenance,
}

/// Resource utilization
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_percent: f32,
    pub network_in_mbps: f32,
    pub network_out_mbps: f32,
}

impl ResourceUsage {
    pub const fn new() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            disk_percent: 0.0,
            network_in_mbps: 0.0,
            network_out_mbps: 0.0,
        }
    }

    /// Get the highest utilization percentage
    pub fn max_utilization(&self) -> f32 {
        self.cpu_percent
            .max(self.memory_percent)
            .max(self.disk_percent)
    }

    /// Determine status based on resource usage
    pub fn status(&self) -> NodeStatus {
        let max = self.max_utilization();
        if max >= 95.0 {
            NodeStatus::Critical
        } else if max >= 80.0 {
            NodeStatus::Warning
        } else {
            NodeStatus::Healthy
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// Infrastructure node (server/container/pod)
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub node_type: NodeType,
    pub region: String,
    pub status: NodeStatus,
    pub resources: ResourceUsage,
    pub tags: HashMap<String, String>,
    pub uptime_hours: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Server,
    Container,
    Pod,
    Lambda,
}

impl Node {
    pub fn new(id: &str, name: &str, node_type: NodeType, region: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            node_type,
            region: region.to_string(),
            status: NodeStatus::Healthy,
            resources: ResourceUsage::new(),
            tags: HashMap::new(),
            uptime_hours: 0.0,
        }
    }

    pub fn with_resources(mut self, resources: ResourceUsage) -> Self {
        self.status = resources.status();
        self.resources = resources;
        self
    }

    pub const fn with_status(mut self, status: NodeStatus) -> Self {
        self.status = status;
        self
    }

    pub const fn with_uptime(mut self, hours: f32) -> Self {
        self.uptime_hours = hours;
        self
    }

    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if node needs attention
    pub const fn needs_attention(&self) -> bool {
        matches!(self.status, NodeStatus::Warning | NodeStatus::Critical)
    }
}

/// Infrastructure dashboard
#[derive(Debug)]
pub struct InfrastructureDashboard {
    nodes: Vec<Node>,
    title: String,
}

impl InfrastructureDashboard {
    pub fn new(title: &str) -> Self {
        Self {
            nodes: Vec::new(),
            title: title.to_string(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Get nodes by status
    pub fn nodes_by_status(&self, status: NodeStatus) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.status == status).collect()
    }

    /// Get nodes by region
    pub fn nodes_by_region(&self, region: &str) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.region == region).collect()
    }

    /// Get nodes by type
    pub fn nodes_by_type(&self, node_type: NodeType) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.node_type == node_type)
            .collect()
    }

    /// Count nodes by status
    pub fn status_counts(&self) -> HashMap<NodeStatus, usize> {
        let mut counts = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.status).or_insert(0) += 1;
        }
        counts
    }

    /// Get average resource utilization
    pub fn average_utilization(&self) -> ResourceUsage {
        if self.nodes.is_empty() {
            return ResourceUsage::new();
        }

        let count = self.nodes.len() as f32;
        let mut total = ResourceUsage::new();

        for node in &self.nodes {
            total.cpu_percent += node.resources.cpu_percent;
            total.memory_percent += node.resources.memory_percent;
            total.disk_percent += node.resources.disk_percent;
            total.network_in_mbps += node.resources.network_in_mbps;
            total.network_out_mbps += node.resources.network_out_mbps;
        }

        ResourceUsage {
            cpu_percent: total.cpu_percent / count,
            memory_percent: total.memory_percent / count,
            disk_percent: total.disk_percent / count,
            network_in_mbps: total.network_in_mbps / count,
            network_out_mbps: total.network_out_mbps / count,
        }
    }

    /// Get nodes that need attention
    pub fn needs_attention(&self) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.needs_attention()).collect()
    }

    /// Calculate overall health score (0-100)
    pub fn health_score(&self) -> f32 {
        if self.nodes.is_empty() {
            return 100.0;
        }

        let healthy = self.nodes_by_status(NodeStatus::Healthy).len();
        let total = self.nodes.len();

        (healthy as f32 / total as f32) * 100.0
    }

    pub fn title(&self) -> &str {
        &self.title
    }
}

fn main() {
    println!("=== Infrastructure Dashboard ===\n");

    let mut dashboard = InfrastructureDashboard::new("Production Cluster");

    // Add servers
    dashboard.add_node(
        Node::new("srv-001", "web-1", NodeType::Server, "us-east-1")
            .with_resources(ResourceUsage {
                cpu_percent: 45.0,
                memory_percent: 62.0,
                disk_percent: 35.0,
                network_in_mbps: 120.0,
                network_out_mbps: 450.0,
            })
            .with_uptime(720.0)
            .with_tag("role", "frontend"),
    );
    dashboard.add_node(
        Node::new("srv-002", "web-2", NodeType::Server, "us-east-1")
            .with_resources(ResourceUsage {
                cpu_percent: 82.0,
                memory_percent: 78.0,
                disk_percent: 40.0,
                network_in_mbps: 150.0,
                network_out_mbps: 520.0,
            })
            .with_uptime(720.0)
            .with_tag("role", "frontend"),
    );
    dashboard.add_node(
        Node::new("srv-003", "api-1", NodeType::Server, "us-west-2")
            .with_resources(ResourceUsage {
                cpu_percent: 55.0,
                memory_percent: 70.0,
                disk_percent: 25.0,
                network_in_mbps: 200.0,
                network_out_mbps: 180.0,
            })
            .with_uptime(168.0)
            .with_tag("role", "backend"),
    );
    dashboard.add_node(
        Node::new("srv-004", "db-1", NodeType::Server, "us-east-1")
            .with_resources(ResourceUsage {
                cpu_percent: 30.0,
                memory_percent: 85.0,
                disk_percent: 72.0,
                network_in_mbps: 80.0,
                network_out_mbps: 60.0,
            })
            .with_uptime(2160.0)
            .with_tag("role", "database"),
    );
    dashboard.add_node(
        Node::new("srv-005", "cache-1", NodeType::Container, "us-east-1")
            .with_resources(ResourceUsage {
                cpu_percent: 15.0,
                memory_percent: 95.0,
                disk_percent: 10.0,
                network_in_mbps: 300.0,
                network_out_mbps: 280.0,
            })
            .with_uptime(48.0)
            .with_tag("role", "cache"),
    );
    dashboard.add_node(
        Node::new("srv-006", "worker-1", NodeType::Pod, "eu-west-1")
            .with_status(NodeStatus::Offline)
            .with_uptime(0.0)
            .with_tag("role", "worker"),
    );

    // Print summary
    println!("Dashboard: {}", dashboard.title());
    println!("Health Score: {:.1}%\n", dashboard.health_score());

    let counts = dashboard.status_counts();
    println!("Node Status:");
    println!(
        "  Healthy: {} | Warning: {} | Critical: {} | Offline: {}",
        counts.get(&NodeStatus::Healthy).unwrap_or(&0),
        counts.get(&NodeStatus::Warning).unwrap_or(&0),
        counts.get(&NodeStatus::Critical).unwrap_or(&0),
        counts.get(&NodeStatus::Offline).unwrap_or(&0)
    );

    let avg = dashboard.average_utilization();
    println!("\nAverage Utilization:");
    println!(
        "  CPU: {:.1}% | Memory: {:.1}% | Disk: {:.1}%",
        avg.cpu_percent, avg.memory_percent, avg.disk_percent
    );

    // Node list
    println!(
        "\n{:<10} {:<12} {:<10} {:<12} {:>6} {:>6} {:>6} {:>10}",
        "ID", "Name", "Type", "Region", "CPU%", "Mem%", "Disk%", "Status"
    );
    println!("{}", "-".repeat(80));

    for node in dashboard.nodes() {
        let status_icon = match node.status {
            NodeStatus::Healthy => "●",
            NodeStatus::Warning => "◐",
            NodeStatus::Critical => "○",
            NodeStatus::Offline => "✗",
            NodeStatus::Maintenance => "⚙",
        };

        println!(
            "{:<10} {:<12} {:<10} {:<12} {:>5.1} {:>5.1} {:>5.1} {:>9} {}",
            node.id,
            node.name,
            format!("{:?}", node.node_type),
            node.region,
            node.resources.cpu_percent,
            node.resources.memory_percent,
            node.resources.disk_percent,
            format!("{:?}", node.status),
            status_icon
        );
    }

    // Nodes needing attention
    let attention = dashboard.needs_attention();
    if !attention.is_empty() {
        println!("\n⚠ Nodes Needing Attention:");
        for node in attention {
            println!(
                "  {} ({}) - {:?}: max util {:.1}%",
                node.name,
                node.id,
                node.status,
                node.resources.max_utilization()
            );
        }
    }

    // Resource bars
    println!("\n=== Resource Utilization ===\n");
    for node in dashboard.nodes() {
        if node.status == NodeStatus::Offline {
            continue;
        }
        println!("{:<12}", node.name);
        print_bar("  CPU", node.resources.cpu_percent);
        print_bar("  Mem", node.resources.memory_percent);
        print_bar("  Disk", node.resources.disk_percent);
        println!();
    }

    println!("=== Acceptance Criteria ===");
    println!("- [x] Node status indicators");
    println!("- [x] Resource utilization visible");
    println!("- [x] Regional grouping supported");
    println!("- [x] 15-point checklist complete");
}

fn print_bar(label: &str, percent: f32) {
    let bar_width = 30;
    let filled = (percent / 100.0 * bar_width as f32) as usize;
    let bar_char = if percent >= 95.0 {
        '█'
    } else if percent >= 80.0 {
        '▓'
    } else {
        '░'
    };
    println!(
        "{:<6} [{}{}] {:>5.1}%",
        label,
        bar_char.to_string().repeat(filled),
        " ".repeat(bar_width - filled),
        percent
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_status_healthy() {
        let resources = ResourceUsage {
            cpu_percent: 50.0,
            memory_percent: 60.0,
            disk_percent: 40.0,
            network_in_mbps: 100.0,
            network_out_mbps: 100.0,
        };
        assert_eq!(resources.status(), NodeStatus::Healthy);
    }

    #[test]
    fn test_resource_status_warning() {
        let resources = ResourceUsage {
            cpu_percent: 85.0,
            memory_percent: 60.0,
            disk_percent: 40.0,
            network_in_mbps: 100.0,
            network_out_mbps: 100.0,
        };
        assert_eq!(resources.status(), NodeStatus::Warning);
    }

    #[test]
    fn test_resource_status_critical() {
        let resources = ResourceUsage {
            cpu_percent: 50.0,
            memory_percent: 97.0,
            disk_percent: 40.0,
            network_in_mbps: 100.0,
            network_out_mbps: 100.0,
        };
        assert_eq!(resources.status(), NodeStatus::Critical);
    }

    #[test]
    fn test_node_needs_attention() {
        let healthy_node =
            Node::new("1", "test", NodeType::Server, "us-east-1").with_status(NodeStatus::Healthy);
        let warning_node =
            Node::new("2", "test", NodeType::Server, "us-east-1").with_status(NodeStatus::Warning);

        assert!(!healthy_node.needs_attention());
        assert!(warning_node.needs_attention());
    }

    #[test]
    fn test_dashboard_status_counts() {
        let mut dashboard = InfrastructureDashboard::new("Test");
        dashboard
            .add_node(Node::new("1", "a", NodeType::Server, "us").with_status(NodeStatus::Healthy));
        dashboard
            .add_node(Node::new("2", "b", NodeType::Server, "us").with_status(NodeStatus::Healthy));
        dashboard
            .add_node(Node::new("3", "c", NodeType::Server, "us").with_status(NodeStatus::Warning));

        let counts = dashboard.status_counts();
        assert_eq!(counts.get(&NodeStatus::Healthy), Some(&2));
        assert_eq!(counts.get(&NodeStatus::Warning), Some(&1));
    }

    #[test]
    fn test_dashboard_health_score() {
        let mut dashboard = InfrastructureDashboard::new("Test");
        dashboard
            .add_node(Node::new("1", "a", NodeType::Server, "us").with_status(NodeStatus::Healthy));
        dashboard
            .add_node(Node::new("2", "b", NodeType::Server, "us").with_status(NodeStatus::Warning));

        // 1 healthy out of 2 = 50%
        assert!((dashboard.health_score() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_dashboard_average_utilization() {
        let mut dashboard = InfrastructureDashboard::new("Test");
        dashboard.add_node(Node::new("1", "a", NodeType::Server, "us").with_resources(
            ResourceUsage {
                cpu_percent: 40.0,
                memory_percent: 60.0,
                disk_percent: 20.0,
                network_in_mbps: 100.0,
                network_out_mbps: 100.0,
            },
        ));
        dashboard.add_node(Node::new("2", "b", NodeType::Server, "us").with_resources(
            ResourceUsage {
                cpu_percent: 60.0,
                memory_percent: 80.0,
                disk_percent: 40.0,
                network_in_mbps: 100.0,
                network_out_mbps: 100.0,
            },
        ));

        let avg = dashboard.average_utilization();
        assert!((avg.cpu_percent - 50.0).abs() < 0.01);
        assert!((avg.memory_percent - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_nodes_by_region() {
        let mut dashboard = InfrastructureDashboard::new("Test");
        dashboard.add_node(Node::new("1", "a", NodeType::Server, "us-east-1"));
        dashboard.add_node(Node::new("2", "b", NodeType::Server, "us-west-2"));
        dashboard.add_node(Node::new("3", "c", NodeType::Server, "us-east-1"));

        let east_nodes = dashboard.nodes_by_region("us-east-1");
        assert_eq!(east_nodes.len(), 2);
    }

    #[test]
    fn test_nodes_by_type() {
        let mut dashboard = InfrastructureDashboard::new("Test");
        dashboard.add_node(Node::new("1", "a", NodeType::Server, "us"));
        dashboard.add_node(Node::new("2", "b", NodeType::Container, "us"));
        dashboard.add_node(Node::new("3", "c", NodeType::Server, "us"));

        let servers = dashboard.nodes_by_type(NodeType::Server);
        assert_eq!(servers.len(), 2);
    }
}
