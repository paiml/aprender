//! Correlation analysis types: events, samples, results, and recommendations.

/// System event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// CPU interrupt
    Interrupt,
    /// Disk I/O
    DiskIo,
    /// Network activity
    Network,
    /// Context switch
    ContextSwitch,
    /// Page fault
    PageFault,
    /// Other process CPU usage
    ProcessCpu,
}

impl EventType {
    /// Get event type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Interrupt => "interrupt",
            Self::DiskIo => "disk_io",
            Self::Network => "network",
            Self::ContextSwitch => "context_switch",
            Self::PageFault => "page_fault",
            Self::ProcessCpu => "process_cpu",
        }
    }
}

/// System event sample
#[derive(Debug, Clone)]
pub struct EventSample {
    /// Event type
    pub event_type: EventType,
    /// Timestamp (seconds since start)
    pub timestamp: f64,
    /// Event count or value
    pub value: f64,
}

impl EventSample {
    /// Create new sample
    pub fn new(event_type: EventType, timestamp: f64, value: f64) -> Self {
        Self {
            event_type,
            timestamp,
            value,
        }
    }
}

/// Performance sample with CV
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Timestamp (seconds since start)
    pub timestamp: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Latency (microseconds)
    pub latency_us: f64,
}

impl PerformanceSample {
    /// Create new sample
    pub fn new(timestamp: f64, cv_percent: f64, latency_us: f64) -> Self {
        Self {
            timestamp,
            cv_percent,
            latency_us,
        }
    }

    /// Check if this is a CV spike (>15%)
    pub fn is_spike(&self, threshold: f64) -> bool {
        self.cv_percent > threshold
    }
}

/// Correlation result between performance and events
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Event type
    pub event_type: EventType,
    /// Pearson correlation coefficient (-1 to 1)
    pub pearson_r: f64,
    /// Number of paired samples
    pub sample_count: usize,
    /// Is correlation significant?
    pub is_significant: bool,
    /// Lag (seconds) at max correlation
    pub optimal_lag: f64,
}

impl CorrelationResult {
    /// Check if there's positive correlation
    pub fn has_correlation(&self) -> bool {
        self.pearson_r.abs() > 0.3 && self.is_significant
    }

    /// Get correlation strength description
    pub fn strength(&self) -> &'static str {
        let r = self.pearson_r.abs();
        if r < 0.1 {
            "negligible"
        } else if r < 0.3 {
            "weak"
        } else if r < 0.5 {
            "moderate"
        } else if r < 0.7 {
            "strong"
        } else {
            "very strong"
        }
    }
}

/// Interference detection result
#[derive(Debug, Clone)]
pub struct InterferenceResult {
    /// Primary interference source
    pub primary_source: EventType,
    /// Correlation strength
    pub correlation: f64,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Secondary sources
    pub secondary_sources: Vec<(EventType, f64)>,
}

impl InterferenceResult {
    /// Get interference category
    pub fn category(&self) -> InterferenceCategory {
        if self.confidence > 0.8 && self.correlation > 0.5 {
            InterferenceCategory::High
        } else if self.confidence > 0.5 || self.correlation > 0.3 {
            InterferenceCategory::Moderate
        } else {
            InterferenceCategory::Low
        }
    }
}

/// Interference severity category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceCategory {
    /// Low interference
    Low,
    /// Moderate interference
    Moderate,
    /// High interference
    High,
}

impl InterferenceCategory {
    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
        }
    }
}

/// Isolation recommendation
#[derive(Debug, Clone)]
pub struct IsolationRecommendation {
    /// Recommended action
    pub action: IsolationAction,
    /// Reason for recommendation
    pub reason: String,
    /// Expected improvement (%)
    pub expected_improvement: f64,
    /// Confidence (0-1)
    pub confidence: f64,
}

/// Isolation action types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationAction {
    /// Pin to specific CPU cores
    CpuPin,
    /// Use memory isolation (cgroups)
    MemoryIsolation,
    /// Disable hyperthreading on core
    DisableHyperthread,
    /// Use dedicated NUMA node
    NumaIsolation,
    /// Reduce network polling
    NetworkIsolation,
    /// Use realtime priority
    RealtimePriority,
    /// No action needed
    None,
}

impl IsolationAction {
    /// Get action name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuPin => "cpu_pin",
            Self::MemoryIsolation => "memory_isolation",
            Self::DisableHyperthread => "disable_hyperthread",
            Self::NumaIsolation => "numa_isolation",
            Self::NetworkIsolation => "network_isolation",
            Self::RealtimePriority => "realtime_priority",
            Self::None => "none",
        }
    }

    /// Get action description
    pub fn description(&self) -> &'static str {
        match self {
            Self::CpuPin => "Pin benchmark to specific CPU cores using taskset/numactl",
            Self::MemoryIsolation => "Use cgroups to limit memory access from other processes",
            Self::DisableHyperthread => "Disable hyperthreading on benchmark cores",
            Self::NumaIsolation => "Run benchmark on dedicated NUMA node",
            Self::NetworkIsolation => "Reduce network polling frequency during benchmark",
            Self::RealtimePriority => "Use SCHED_FIFO realtime priority",
            Self::None => "No isolation needed",
        }
    }
}

/// System state snapshot
#[derive(Debug, Clone, Default)]
pub struct SystemSnapshot {
    /// Timestamp
    pub timestamp: f64,
    /// IRQ counts by type
    pub irq_counts: Vec<(String, u64)>,
    /// Disk I/O bytes/sec
    pub disk_io_bytes_per_sec: f64,
    /// Network packets/sec
    pub network_packets_per_sec: f64,
    /// Context switches/sec
    pub context_switches_per_sec: f64,
    /// Top CPU consumers (name, %)
    pub top_processes: Vec<(String, f64)>,
    /// Load average (1, 5, 15 min)
    pub load_average: (f64, f64, f64),
}

impl SystemSnapshot {
    /// Create empty snapshot
    pub fn new(timestamp: f64) -> Self {
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Add IRQ count
    pub fn with_irq(mut self, name: &str, count: u64) -> Self {
        self.irq_counts.push((name.to_string(), count));
        self
    }

    /// Set disk I/O
    pub fn with_disk_io(mut self, bytes_per_sec: f64) -> Self {
        self.disk_io_bytes_per_sec = bytes_per_sec;
        self
    }

    /// Set network activity
    pub fn with_network(mut self, packets_per_sec: f64) -> Self {
        self.network_packets_per_sec = packets_per_sec;
        self
    }

    /// Set context switches
    pub fn with_context_switches(mut self, per_sec: f64) -> Self {
        self.context_switches_per_sec = per_sec;
        self
    }

    /// Add top process
    pub fn with_process(mut self, name: &str, cpu_percent: f64) -> Self {
        self.top_processes.push((name.to_string(), cpu_percent));
        self
    }

    /// Set load average
    pub fn with_load_average(mut self, load1: f64, load5: f64, load15: f64) -> Self {
        self.load_average = (load1, load5, load15);
        self
    }

    /// Get total IRQ count
    pub fn total_irqs(&self) -> u64 {
        self.irq_counts.iter().map(|(_, c)| c).sum()
    }
}
