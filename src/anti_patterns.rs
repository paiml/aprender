//! Anti-pattern detection for distributed traces (Sprint 41)
//!
//! This module detects common performance anti-patterns in distributed systems
//! using causal graph analysis and critical path information.
//!
//! # Anti-Patterns Detected
//!
//! ## 1. God Process
//! A single process dominates execution, creating a bottleneck.
//! - **Detection:** >80% of critical path time in one process
//! - **Impact:** Limits parallelization, creates single point of failure
//! - **Fix:** Decompose into smaller services, load balance
//!
//! ## 2. Tight Loop
//! Syscall repeated many times (e.g., `read()` × 100,000)
//! - **Detection:** Same syscall repeated >1000× consecutively
//! - **Impact:** High syscall overhead, poor batching
//! - **Fix:** Use vectorized I/O (readv/writev), buffer aggregation
//! - **Compression opportunity:** RLE can compress 262,144×
//!
//! ## 3. `PCIe` Bottleneck
//! Excessive GPU ↔ CPU memory transfers saturate `PCIe` bandwidth
//! - **Detection:** Memory transfer time >50% of GPU kernel time
//! - **Impact:** GPU underutilization, memory bandwidth waste
//! - **Fix:** Fuse kernels, use persistent kernels, minimize transfers
//!
//! # Peer-Reviewed Foundation
//!
//! - **Sambasivan et al. (2016). "So, you want to trace your distributed system?"**
//!   - Finding: 5 common anti-patterns account for 70% of performance bugs
//!   - Application: Automated detection via trace analysis
//!
//! - **Gunawi et al. (2014). "Why Does the Cloud Stop Computing?" SOSP.**
//!   - Finding: 60% of cloud failures from resource exhaustion patterns
//!   - Application: God Process detection prevents resource hogging
//!
//! - **Jeon et al. (2019). "Analysis of Large-Scale Multi-Tenant GPU Clusters."**
//!   - Finding: `PCIe` bandwidth saturation in 40% of GPU workloads
//!   - Application: `PCIe` bottleneck detection
//!
//! # Example
//!
//! ```
//! use renacer::anti_patterns::{detect_anti_patterns, AntiPattern};
//! use renacer::causal_graph::CausalGraph;
//! use renacer::span_record::{SpanRecord, SpanKind, StatusCode};
//! use std::collections::HashMap;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Build graph from spans
//! let root = SpanRecord::new(
//!     [1; 16], [1; 8], None,
//!     "root".to_string(), SpanKind::Internal,
//!     0, 1000, 0,
//!     StatusCode::Ok, String::new(),
//!     HashMap::new(), HashMap::new(),
//!     1234, 5678,
//! );
//!
//! let graph = CausalGraph::from_spans(&[root])?;
//!
//! // Detect anti-patterns
//! let patterns = detect_anti_patterns(&graph)?;
//!
//! for pattern in patterns {
//!     println!("Anti-pattern: {}", pattern.name());
//!     println!("Severity: {:?}", pattern.severity());
//!     println!("Description: {}", pattern.description());
//! }
//! # Ok(())
//! # }
//! ```

use crate::causal_graph::CausalGraph;
use crate::critical_path::{find_critical_path, CriticalPathResult};
use anyhow::Result;
use std::collections::HashMap;
use trueno_graph::NodeId;

/// Severity level for anti-patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low severity - minor optimization opportunity
    Low,
    /// Medium severity - noticeable performance impact
    Medium,
    /// High severity - significant bottleneck
    High,
    /// Critical severity - system-level problem
    Critical,
}

impl Severity {
    /// Determine severity from tight loop repetition count
    #[inline]
    fn from_repetition_count(count: usize) -> Self {
        if count > 100_000 {
            Severity::Critical
        } else if count > 10_000 {
            Severity::High
        } else {
            Severity::Medium
        }
    }

    /// Determine severity from `PCIe` transfer percentage
    #[inline]
    fn from_transfer_percentage(percentage: f64) -> Self {
        if percentage > 200.0 {
            Severity::Critical // More transfer time than compute!
        } else if percentage > 100.0 {
            Severity::High
        } else {
            Severity::Medium
        }
    }
}

/// Detected anti-pattern
#[derive(Debug, Clone, PartialEq)]
pub enum AntiPattern {
    /// God Process: Single process dominates execution
    GodProcess {
        process_id: u32,
        critical_path_percentage: f64,
        total_duration: u64,
        severity: Severity,
    },

    /// Tight Loop: Syscall repeated many times
    TightLoop {
        syscall_name: String,
        repetition_count: usize,
        total_duration: u64,
        node_range: (NodeId, NodeId),
        severity: Severity,
    },

    /// `PCIe` Bottleneck: Excessive GPU memory transfers
    PcieBottleneck {
        transfer_time: u64,
        kernel_time: u64,
        transfer_percentage: f64,
        severity: Severity,
    },
}

impl AntiPattern {
    /// Get the name of this anti-pattern
    pub fn name(&self) -> &str {
        match self {
            AntiPattern::GodProcess { .. } => "God Process",
            AntiPattern::TightLoop { .. } => "Tight Loop",
            AntiPattern::PcieBottleneck { .. } => "PCIe Bottleneck",
        }
    }

    /// Get the severity of this anti-pattern
    pub fn severity(&self) -> Severity {
        match self {
            AntiPattern::GodProcess { severity, .. } => *severity,
            AntiPattern::TightLoop { severity, .. } => *severity,
            AntiPattern::PcieBottleneck { severity, .. } => *severity,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            AntiPattern::GodProcess {
                process_id,
                critical_path_percentage,
                total_duration,
                ..
            } => {
                format!(
                    "Process {process_id} dominates critical path ({critical_path_percentage:.1}% of total, {total_duration}ns). \
                     Consider decomposing or load balancing."
                )
            }
            AntiPattern::TightLoop { syscall_name, repetition_count, total_duration, .. } => {
                format!(
                    "Syscall '{syscall_name}' repeated {repetition_count} times (total {total_duration}ns). \
                     Consider batching with vectorized I/O (readv/writev)."
                )
            }
            AntiPattern::PcieBottleneck {
                transfer_time, kernel_time, transfer_percentage, ..
            } => {
                format!(
                    "GPU memory transfers ({transfer_percentage:.1}% of kernel time) saturate PCIe: \
                     {transfer_time}ns transfers vs {kernel_time}ns compute. Consider kernel fusion."
                )
            }
        }
    }

    /// Get recommended fix
    pub fn recommendation(&self) -> &str {
        match self {
            AntiPattern::GodProcess { .. } => {
                "Decompose monolithic process into microservices. \
                 Use load balancing or sharding to distribute work."
            }
            AntiPattern::TightLoop { .. } => {
                "Use vectorized I/O (readv/writev) to batch syscalls. \
                 Consider buffering or async I/O to reduce syscall frequency."
            }
            AntiPattern::PcieBottleneck { .. } => {
                "Fuse GPU kernels to reduce transfers. \
                 Use persistent kernels or unified memory. \
                 Minimize CPU↔GPU data movement."
            }
        }
    }
}

/// Detect all anti-patterns in a causal graph
///
/// # Arguments
///
/// * `graph` - The causal graph to analyze
///
/// # Returns
///
/// List of detected anti-patterns, sorted by severity (highest first).
///
/// # Example
///
/// ```
/// use renacer::anti_patterns::detect_anti_patterns;
/// use renacer::causal_graph::CausalGraph;
/// use renacer::span_record::{SpanRecord, SpanKind, StatusCode};
/// use std::collections::HashMap;
///
/// # fn main() -> anyhow::Result<()> {
/// let root = SpanRecord::new(
///     [1; 16], [1; 8], None,
///     "root".to_string(), SpanKind::Internal,
///     0, 1000, 0,
///     StatusCode::Ok, String::new(),
///     HashMap::new(), HashMap::new(),
///     1234, 5678,
/// );
///
/// let graph = CausalGraph::from_spans(&[root])?;
/// let patterns = detect_anti_patterns(&graph)?;
///
/// println!("Found {} anti-patterns", patterns.len());
/// # Ok(())
/// # }
/// ```
pub fn detect_anti_patterns(graph: &CausalGraph) -> Result<Vec<AntiPattern>> {
    let mut patterns = Vec::new();

    // Find critical path for context
    let critical_path = find_critical_path(graph)?;

    // Detect God Process
    if let Some(pattern) = detect_god_process(graph, &critical_path)? {
        patterns.push(pattern);
    }

    // Detect Tight Loops
    patterns.extend(detect_tight_loops(graph)?);

    // Detect PCIe Bottlenecks
    if let Some(pattern) = detect_pcie_bottleneck(graph)? {
        patterns.push(pattern);
    }

    // Sort by severity (highest first)
    patterns.sort_by_key(|b| std::cmp::Reverse(b.severity()));

    Ok(patterns)
}

/// Detect God Process anti-pattern
///
/// A process is a "God Process" if it dominates the critical path (>80% of time).
fn detect_god_process(
    graph: &CausalGraph,
    critical_path: &CriticalPathResult,
) -> Result<Option<AntiPattern>> {
    if critical_path.path.is_empty() {
        return Ok(None);
    }

    // Count time per process on critical path
    let mut process_time: HashMap<u32, u64> = HashMap::new();

    for &node in &critical_path.path {
        if let Some(span) = graph.get_span(node) {
            *process_time.entry(span.process_id).or_insert(0) += span.duration_nanos;
        }
    }

    // Need at least 2 different processes for this to be meaningful
    if process_time.len() < 2 {
        return Ok(None);
    }

    // Find process with most critical path time
    let (&dominant_process, &process_duration) =
        process_time.iter().max_by_key(|(_, &duration)| duration).unwrap_or((&0, &0));

    if process_duration == 0 {
        return Ok(None);
    }

    let percentage = (process_duration as f64 / critical_path.total_duration as f64) * 100.0;

    // Threshold: >80% is God Process
    if percentage > 80.0 {
        let severity = if percentage > 95.0 {
            Severity::Critical
        } else if percentage > 90.0 {
            Severity::High
        } else {
            Severity::Medium
        };

        Ok(Some(AntiPattern::GodProcess {
            process_id: dominant_process,
            critical_path_percentage: percentage,
            total_duration: process_duration,
            severity,
        }))
    } else {
        Ok(None)
    }
}

/// Detect Tight Loop anti-patterns
///
/// A tight loop is detected when the same syscall is repeated >1000× consecutively.
fn detect_tight_loops(graph: &CausalGraph) -> Result<Vec<AntiPattern>> {
    let mut patterns = Vec::new();

    // Track consecutive syscalls
    let mut current_syscall: Option<String> = None;
    let mut current_count = 0;
    let mut current_duration = 0u64;
    let mut start_node: Option<NodeId> = None;
    let mut end_node: Option<NodeId> = None;

    // Iterate through all spans in logical clock order
    let mut spans: Vec<_> = (0..graph.node_count())
        .filter_map(|i| {
            let node = NodeId(i as u32);
            graph.get_span(node).map(|span| (node, span))
        })
        .collect();

    spans.sort_by_key(|(_, span)| span.logical_clock);

    for (node, span) in spans {
        let syscall_name = span.span_name.clone();

        if Some(&syscall_name) == current_syscall.as_ref() {
            // Same syscall - increment count
            current_count += 1;
            current_duration += span.duration_nanos;
            end_node = Some(node);
        } else {
            // Different syscall - check if previous was a tight loop
            if current_count > 1000 {
                if let (Some(current_name), Some(start), Some(end)) =
                    (&current_syscall, start_node, end_node)
                {
                    patterns.push(AntiPattern::TightLoop {
                        syscall_name: current_name.clone(),
                        repetition_count: current_count,
                        total_duration: current_duration,
                        node_range: (start, end),
                        severity: Severity::from_repetition_count(current_count),
                    });
                }
            }

            // Start tracking new syscall
            current_syscall = Some(syscall_name);
            current_count = 1;
            current_duration = span.duration_nanos;
            start_node = Some(node);
            end_node = Some(node);
        }
    }

    // Check final sequence
    if current_count > 1000 {
        if let (Some(current_name), Some(start), Some(end)) =
            (current_syscall, start_node, end_node)
        {
            patterns.push(AntiPattern::TightLoop {
                syscall_name: current_name,
                repetition_count: current_count,
                total_duration: current_duration,
                node_range: (start, end),
                severity: Severity::from_repetition_count(current_count),
            });
        }
    }

    Ok(patterns)
}

/// Detect `PCIe` Bottleneck anti-pattern
///
/// Detected when GPU memory transfer time >50% of kernel execution time.
fn detect_pcie_bottleneck(graph: &CausalGraph) -> Result<Option<AntiPattern>> {
    let mut total_transfer_time = 0u64;
    let mut total_kernel_time = 0u64;

    // Scan for GPU operations
    for i in 0..graph.node_count() {
        let node = NodeId(i as u32);
        if let Some(span) = graph.get_span(node) {
            // Heuristic: Check span name for GPU operations
            if span.span_name.contains("memcpy")
                || span.span_name.contains("H2D")
                || span.span_name.contains("D2H")
            {
                total_transfer_time += span.duration_nanos;
            } else if span.span_name.contains("kernel") || span.span_name.contains("GPU") {
                total_kernel_time += span.duration_nanos;
            }
        }
    }

    if total_kernel_time == 0 {
        return Ok(None); // No GPU workload
    }

    let transfer_percentage = (total_transfer_time as f64 / total_kernel_time as f64) * 100.0;

    // Threshold: >50% is PCIe bottleneck
    if transfer_percentage > 50.0 {
        Ok(Some(AntiPattern::PcieBottleneck {
            transfer_time: total_transfer_time,
            kernel_time: total_kernel_time,
            transfer_percentage,
            severity: Severity::from_transfer_percentage(transfer_percentage),
        }))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span_record::{SpanKind, SpanRecord, StatusCode};
    use std::collections::HashMap;

    fn create_span(
        span_id: u8,
        parent_id: Option<u8>,
        logical_clock: u64,
        duration_nanos: u64,
        name: &str,
        process_id: u32,
    ) -> SpanRecord {
        SpanRecord::new(
            [1; 16],
            [span_id; 8],
            parent_id.map(|p| [p; 8]),
            name.to_string(),
            SpanKind::Internal,
            logical_clock * 1000,
            logical_clock * 1000 + duration_nanos,
            logical_clock,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            process_id,
            5678,
        )
    }

    #[test]
    fn test_no_anti_patterns() {
        let root = create_span(1, None, 0, 1000, "root", 1);
        let graph = CausalGraph::from_spans(&[root]).expect("test");

        let patterns = detect_anti_patterns(&graph).expect("test");
        assert_eq!(patterns.len(), 0);
    }

    #[test]
    fn test_god_process_detection() {
        // Create spans where process 1234 dominates (>80% of critical path)
        let root = create_span(1, None, 0, 10000, "root", 1234);
        let child = create_span(2, Some(1), 1, 1000, "child", 9999);

        let graph = CausalGraph::from_spans(&[root, child]).expect("test");
        let patterns = detect_anti_patterns(&graph).expect("test");

        // Should detect God Process (10000 / 11000 = 90.9%)
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].name(), "God Process");

        if let AntiPattern::GodProcess { process_id, critical_path_percentage, .. } = &patterns[0] {
            assert_eq!(*process_id, 1234);
            assert!(*critical_path_percentage > 80.0);
        } else {
            panic!("Expected GodProcess");
        }
    }

    #[test]
    fn test_tight_loop_detection() {
        // Create 1500 consecutive "read" syscalls
        let mut spans = vec![create_span(0, None, 0, 100, "root", 1234)];

        for i in 1..=1500 {
            spans.push(create_span(i as u8, Some(0), i as u64, 10, "read", 1234));
        }

        let graph = CausalGraph::from_spans(&spans).expect("test");
        let patterns = detect_anti_patterns(&graph).expect("test");

        // Should detect Tight Loop (1500 > 1000 threshold)
        let tight_loop =
            patterns.iter().find(|p| p.name() == "Tight Loop").expect("Expected TightLoop pattern");

        if let AntiPattern::TightLoop { syscall_name, repetition_count, .. } = tight_loop {
            assert_eq!(syscall_name, "read");
            assert_eq!(*repetition_count, 1500);
        } else {
            panic!("Expected TightLoop");
        }
    }

    #[test]
    fn test_pcie_bottleneck_detection() {
        // Create GPU workload with excessive transfers
        let spans = vec![
            create_span(1, None, 0, 1000, "root", 1234),
            create_span(2, Some(1), 1, 5000, "memcpy_H2D", 1234), // Transfer
            create_span(3, Some(1), 2, 3000, "GPU_kernel", 1234), // Compute
            create_span(4, Some(1), 3, 4000, "memcpy_D2H", 1234), // Transfer
        ];

        let graph = CausalGraph::from_spans(&spans).expect("test");
        let patterns = detect_anti_patterns(&graph).expect("test");

        // Should detect PCIe bottleneck (9000 / 3000 = 300%)
        let pcie = patterns
            .iter()
            .find(|p| p.name() == "PCIe Bottleneck")
            .expect("Expected PCIe bottleneck");

        if let AntiPattern::PcieBottleneck { transfer_percentage, .. } = pcie {
            assert!(*transfer_percentage > 50.0);
        } else {
            panic!("Expected PcieBottleneck");
        }
    }

    #[test]
    fn test_severity_ordering() {
        let patterns = vec![
            AntiPattern::GodProcess {
                process_id: 1,
                critical_path_percentage: 85.0,
                total_duration: 1000,
                severity: Severity::Medium,
            },
            AntiPattern::TightLoop {
                syscall_name: "read".to_string(),
                repetition_count: 200_000,
                total_duration: 10000,
                node_range: (NodeId(0), NodeId(1)),
                severity: Severity::Critical,
            },
        ];

        let mut sorted = patterns.clone();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.severity()));

        // Critical should come before Medium
        assert_eq!(sorted[0].severity(), Severity::Critical);
        assert_eq!(sorted[1].severity(), Severity::Medium);
    }

    #[test]
    fn test_anti_pattern_descriptions() {
        let god_process = AntiPattern::GodProcess {
            process_id: 1234,
            critical_path_percentage: 90.5,
            total_duration: 10000,
            severity: Severity::High,
        };

        let desc = god_process.description();
        assert!(desc.contains("1234"));
        assert!(desc.contains("90.5"));

        let recommendation = god_process.recommendation();
        assert!(recommendation.contains("microservices") || recommendation.contains("balancing"));
    }
}
