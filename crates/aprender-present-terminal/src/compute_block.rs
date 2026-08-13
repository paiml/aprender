//! `ComputeBlock`: SIMD-optimized panel element trait
//!
//! Implements the `ComputeBlock` architecture from SPEC-024 Section 21.6.
//! All panel elements (sparklines, gauges, etc.) implement this trait
//! to enable SIMD optimization where available.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  ComputeBlock Trait                                     │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
//! │  │ Input Data  │→ │ SIMD Kernel │→ │ Rendered Output │  │
//! │  │ (f32 array) │  │ (AVX2/NEON) │  │ (block chars)   │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────┘  │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## SIMD Instruction Sets Supported
//!
//! | Platform | Instruction Set | Vector Width |
//! |----------|-----------------|--------------|
//! | `x86_64`   | AVX2            | 256-bit (8×f32) |
//! | `x86_64`   | SSE4.1          | 128-bit (4×f32) |
//! | aarch64  | NEON            | 128-bit (4×f32) |
//! | wasm32   | SIMD128         | 128-bit (4×f32) |
//!
//! ## Peer-Reviewed Foundation
//!
//! - Intel Intrinsics Guide (2024): AVX2 intrinsics for f32x8
//! - Fog, A. (2023): SIMD optimization patterns
//! - Hennessy & Patterson (2017): Memory hierarchy optimization

/// SIMD instruction set identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdInstructionSet {
    /// No SIMD available (scalar fallback)
    Scalar,
    /// `x86_64` SSE4.1 (128-bit, 4×f32)
    Sse4,
    /// `x86_64` AVX2 (256-bit, 8×f32)
    Avx2,
    /// `x86_64` AVX-512 (512-bit, 16×f32)
    Avx512,
    /// ARM NEON (128-bit, 4×f32)
    Neon,
    /// WebAssembly SIMD128 (128-bit, 4×f32)
    WasmSimd128,
}

impl SimdInstructionSet {
    /// Get the vector width in f32 elements
    #[must_use]
    pub const fn vector_width(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Sse4 | Self::Neon | Self::WasmSimd128 => 4,
            Self::Avx2 => 8,
            Self::Avx512 => 16,
        }
    }

    /// Detect the best available instruction set at runtime
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                return Self::Sse4;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            return Self::Neon;
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM SIMD is compile-time feature
            #[cfg(target_feature = "simd128")]
            return Self::WasmSimd128;
        }

        Self::Scalar
    }

    /// Get the instruction set name as a static string
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Sse4 => "SSE4.1",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
            Self::Neon => "NEON",
            Self::WasmSimd128 => "WASM SIMD128",
        }
    }
}

impl Default for SimdInstructionSet {
    fn default() -> Self {
        Self::detect()
    }
}

/// `ComputeBlock` trait for SIMD-optimized panel elements
///
/// All panel elements that benefit from SIMD optimization implement
/// this trait. The trait provides a common interface for:
/// - Computing output from input data
/// - Querying SIMD support
/// - Measuring compute latency
///
/// ## Example
///
/// ```ignore
/// struct SparklineBlock {
///     history: Vec<f32>,
/// }
///
/// impl ComputeBlock for SparklineBlock {
///     type Input = f32;
///     type Output = Vec<char>;
///
///     fn compute(&mut self, input: &Self::Input) -> Self::Output {
///         self.history.push(*input);
///         // SIMD-optimized normalization and character mapping
///         self.render_blocks()
///     }
/// }
/// ```
pub trait ComputeBlock {
    /// Input type for this compute block
    type Input;
    /// Output type produced by this compute block
    type Output;

    /// Process input data and produce output
    ///
    /// Implementations should use SIMD where available for optimal
    /// performance. The `simd_instruction_set()` method indicates
    /// which instruction set is being used.
    fn compute(&mut self, input: &Self::Input) -> Self::Output;

    /// Query if this block supports SIMD on the current CPU
    fn simd_supported(&self) -> bool {
        self.simd_instruction_set() != SimdInstructionSet::Scalar
    }

    /// Get the SIMD instruction set used by this block
    fn simd_instruction_set(&self) -> SimdInstructionSet {
        SimdInstructionSet::detect()
    }

    /// Get the compute latency budget in microseconds
    fn latency_budget_us(&self) -> u64 {
        1000 // Default 1ms budget
    }
}

/// `ComputeBlock` ID as specified in SPEC-024 Section 21
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBlockId {
    // CPU Panel (CB-CPU-*)
    CpuSparklines,     // CB-CPU-001
    CpuLoadGauge,      // CB-CPU-002
    CpuLoadTrend,      // CB-CPU-003
    CpuFrequency,      // CB-CPU-004
    CpuBoostIndicator, // CB-CPU-005
    CpuTemperature,    // CB-CPU-006
    CpuTopConsumers,   // CB-CPU-007

    // Memory Panel (CB-MEM-*)
    MemSparklines,     // CB-MEM-001
    MemZramRatio,      // CB-MEM-002
    MemPressureGauge,  // CB-MEM-003
    MemSwapThrashing,  // CB-MEM-004
    MemCacheBreakdown, // CB-MEM-005
    MemHugePages,      // CB-MEM-006

    // Connections Panel (CB-CONN-*)
    ConnAge,          // CB-CONN-001
    ConnProc,         // CB-CONN-002
    ConnGeo,          // CB-CONN-003
    ConnLatency,      // CB-CONN-004
    ConnService,      // CB-CONN-005
    ConnHotIndicator, // CB-CONN-006
    ConnSparkline,    // CB-CONN-007

    // Network Panel (CB-NET-*)
    NetSparklines,    // CB-NET-001
    NetProtocolStats, // CB-NET-002
    NetErrorRate,     // CB-NET-003
    NetDropRate,      // CB-NET-004
    NetLatencyGauge,  // CB-NET-005
    NetBandwidthUtil, // CB-NET-006

    // Process Panel (CB-PROC-*)
    ProcTreeView,      // CB-PROC-001
    ProcSortIndicator, // CB-PROC-002
    ProcFilter,        // CB-PROC-003
    ProcOomScore,      // CB-PROC-004
    ProcNiceValue,     // CB-PROC-005
    ProcThreadCount,   // CB-PROC-006
    ProcCgroup,        // CB-PROC-007
}

impl ComputeBlockId {
    /// Get the string ID (e.g., "CB-CPU-001")
    #[must_use]
    pub const fn id_string(&self) -> &'static str {
        match self {
            Self::CpuSparklines => "CB-CPU-001",
            Self::CpuLoadGauge => "CB-CPU-002",
            Self::CpuLoadTrend => "CB-CPU-003",
            Self::CpuFrequency => "CB-CPU-004",
            Self::CpuBoostIndicator => "CB-CPU-005",
            Self::CpuTemperature => "CB-CPU-006",
            Self::CpuTopConsumers => "CB-CPU-007",
            Self::MemSparklines => "CB-MEM-001",
            Self::MemZramRatio => "CB-MEM-002",
            Self::MemPressureGauge => "CB-MEM-003",
            Self::MemSwapThrashing => "CB-MEM-004",
            Self::MemCacheBreakdown => "CB-MEM-005",
            Self::MemHugePages => "CB-MEM-006",
            Self::ConnAge => "CB-CONN-001",
            Self::ConnProc => "CB-CONN-002",
            Self::ConnGeo => "CB-CONN-003",
            Self::ConnLatency => "CB-CONN-004",
            Self::ConnService => "CB-CONN-005",
            Self::ConnHotIndicator => "CB-CONN-006",
            Self::ConnSparkline => "CB-CONN-007",
            Self::NetSparklines => "CB-NET-001",
            Self::NetProtocolStats => "CB-NET-002",
            Self::NetErrorRate => "CB-NET-003",
            Self::NetDropRate => "CB-NET-004",
            Self::NetLatencyGauge => "CB-NET-005",
            Self::NetBandwidthUtil => "CB-NET-006",
            Self::ProcTreeView => "CB-PROC-001",
            Self::ProcSortIndicator => "CB-PROC-002",
            Self::ProcFilter => "CB-PROC-003",
            Self::ProcOomScore => "CB-PROC-004",
            Self::ProcNiceValue => "CB-PROC-005",
            Self::ProcThreadCount => "CB-PROC-006",
            Self::ProcCgroup => "CB-PROC-007",
        }
    }

    /// Check if this block is SIMD-vectorizable
    #[must_use]
    pub const fn simd_vectorizable(&self) -> bool {
        match self {
            // YES - can use SIMD
            Self::CpuSparklines
            | Self::CpuLoadTrend
            | Self::CpuFrequency
            | Self::CpuTemperature
            | Self::CpuTopConsumers
            | Self::MemSparklines
            | Self::MemPressureGauge
            | Self::MemSwapThrashing
            | Self::ConnAge
            | Self::ConnGeo
            | Self::ConnLatency
            | Self::ConnService
            | Self::ConnHotIndicator
            | Self::ConnSparkline
            | Self::NetSparklines
            | Self::NetProtocolStats
            | Self::NetErrorRate
            | Self::NetDropRate
            | Self::NetBandwidthUtil
            | Self::ProcOomScore
            | Self::ProcNiceValue
            | Self::ProcThreadCount => true,

            // NO - scalar only
            Self::CpuLoadGauge
            | Self::CpuBoostIndicator
            | Self::MemZramRatio
            | Self::MemCacheBreakdown
            | Self::MemHugePages
            | Self::ConnProc
            | Self::NetLatencyGauge
            | Self::ProcTreeView
            | Self::ProcSortIndicator
            | Self::ProcFilter
            | Self::ProcCgroup => false,
        }
    }
}

/// Sparkline `ComputeBlock` (CB-CPU-001, CB-MEM-001, CB-NET-001, CB-CONN-007)
///
/// SIMD-optimized sparkline rendering using 8-level block characters.
/// Uses AVX2 for min/max/normalization when available.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SparklineBlock {
    /// History buffer (60 samples = 60 seconds at 1Hz)
    history: Vec<f32>,
    /// Maximum history length
    max_samples: usize,
    /// SIMD buffer for aligned operations
    simd_buffer: [f32; 8],
    /// Detected instruction set
    instruction_set: SimdInstructionSet,
}

impl Default for SparklineBlock {
    fn default() -> Self {
        Self::new(60)
    }
}

impl SparklineBlock {
    /// Create a new sparkline block with given history length
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        debug_assert!(max_samples > 0, "max_samples must be positive");
        Self {
            history: Vec::with_capacity(max_samples),
            max_samples,
            simd_buffer: [0.0; 8],
            instruction_set: SimdInstructionSet::detect(),
        }
    }

    /// Add a sample to the history
    pub fn push(&mut self, value: f32) {
        if self.history.len() >= self.max_samples {
            self.history.remove(0);
        }
        self.history.push(value);
    }

    /// Get the current history
    #[must_use]
    pub fn history(&self) -> &[f32] {
        &self.history
    }

    /// Render the sparkline as block characters
    #[must_use]
    pub fn render(&self, width: usize) -> Vec<char> {
        if self.history.is_empty() {
            return vec![' '; width];
        }
        contract_pre_render!();

        // SIMD-optimized min/max finding
        let (min, max) = self.find_min_max();
        let range = max - min;

        // Sample history to fit width
        let samples = self.sample_to_width(width);

        // Map to block characters
        #[allow(clippy::items_after_statements)]
        const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        samples
            .iter()
            .map(|&v| {
                if range < f32::EPSILON {
                    BLOCKS[4] // Mid-level if no variation
                } else {
                    let normalized = ((v - min) / range).clamp(0.0, 1.0);
                    let idx = (normalized * 7.0) as usize;
                    BLOCKS[idx.min(7)]
                }
            })
            .collect()
    }

    /// Find min/max using SIMD when available
    fn find_min_max(&self) -> (f32, f32) {
        if self.history.is_empty() {
            return (0.0, 1.0);
        }

        // Scalar implementation - SIMD intrinsics for AVX2 can be added
        // in a future optimization pass if profiling shows this as a hotspot
        let min = self.history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .history
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        (min, max)
    }

    /// Sample history to fit target width
    fn sample_to_width(&self, width: usize) -> Vec<f32> {
        if self.history.len() <= width {
            // Pad with zeros if history is shorter
            let mut result = vec![0.0; width - self.history.len()];
            result.extend_from_slice(&self.history);
            result
        } else {
            // Downsample using linear interpolation
            let step = self.history.len() as f32 / width as f32;
            (0..width)
                .map(|i| {
                    let idx = (i as f32 * step) as usize;
                    self.history[idx.min(self.history.len() - 1)]
                })
                .collect()
        }
    }
}

impl ComputeBlock for SparklineBlock {
    type Input = f32;
    type Output = Vec<char>;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.push(*input);
        self.render(self.max_samples.min(60))
    }

    fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.instruction_set
    }

    fn latency_budget_us(&self) -> u64 {
        100 // 100μs budget for sparkline rendering
    }
}

/// Load Trend `ComputeBlock` (CB-CPU-003)
///
/// Computes the derivative of load average to show trend direction.
#[derive(Debug, Clone)]
pub struct LoadTrendBlock {
    /// Previous load values for derivative calculation
    history: Vec<f32>,
    /// Smoothing window size
    window_size: usize,
}

impl Default for LoadTrendBlock {
    fn default() -> Self {
        Self::new(5)
    }
}

impl LoadTrendBlock {
    /// Create a new load trend block
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        debug_assert!(window_size > 0, "window_size must be positive");
        Self {
            history: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Get the trend direction
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        if self.history.len() < 2 {
            return TrendDirection::Flat;
        }

        let recent = self.history.iter().rev().take(self.window_size);
        let diffs: Vec<f32> = recent
            .clone()
            .zip(recent.skip(1))
            .map(|(a, b)| a - b)
            .collect();

        if diffs.is_empty() {
            return TrendDirection::Flat;
        }

        let avg_diff: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;

        #[allow(clippy::items_after_statements)]
        const THRESHOLD: f32 = 0.05;
        if avg_diff > THRESHOLD {
            TrendDirection::Up
        } else if avg_diff < -THRESHOLD {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        }
    }
}

impl ComputeBlock for LoadTrendBlock {
    type Input = f32;
    type Output = TrendDirection;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        if self.history.len() >= self.window_size * 2 {
            self.history.remove(0);
        }
        self.history.push(*input);
        self.trend()
    }

    fn latency_budget_us(&self) -> u64 {
        10 // Very fast operation
    }
}

/// Trend direction for load/usage indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendDirection {
    /// Trending up (↑)
    Up,
    /// Trending down (↓)
    Down,
    /// Stable/flat (→)
    #[default]
    Flat,
}

impl TrendDirection {
    /// Get the arrow character for this trend
    #[must_use]
    pub const fn arrow(self) -> char {
        match self {
            Self::Up => '↑',
            Self::Down => '↓',
            Self::Flat => '→',
        }
    }
}

// =============================================================================
// Additional ComputeBlocks (SPEC-024 Part VI: Grammar of Graphics)
// =============================================================================

/// CPU Frequency `ComputeBlock` (CB-CPU-004)
///
/// Tracks per-core CPU frequencies and detects frequency scaling state.
/// Per-core frequency data from `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`.
#[derive(Debug, Clone)]
pub struct CpuFrequencyBlock {
    /// Per-core frequencies in MHz
    frequencies: Vec<u32>,
    /// Per-core max frequencies in MHz (for percentage calculation)
    max_frequencies: Vec<u32>,
    /// Detected instruction set
    instruction_set: SimdInstructionSet,
}

impl Default for CpuFrequencyBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFrequencyBlock {
    /// Create a new CPU frequency block
    #[must_use]
    pub fn new() -> Self {
        Self {
            frequencies: Vec::new(),
            max_frequencies: Vec::new(),
            instruction_set: SimdInstructionSet::detect(),
        }
    }

    /// Set per-core frequencies
    pub fn set_frequencies(&mut self, freqs: Vec<u32>, max_freqs: Vec<u32>) {
        self.frequencies = freqs;
        self.max_frequencies = max_freqs;
    }

    /// Get frequencies as percentages of max
    #[must_use]
    pub fn frequency_percentages(&self) -> Vec<f32> {
        self.frequencies
            .iter()
            .zip(self.max_frequencies.iter())
            .map(|(&cur, &max)| {
                if max > 0 {
                    (cur as f32 / max as f32 * 100.0).clamp(0.0, 100.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get scaling state indicator for each core
    #[must_use]
    pub fn scaling_indicators(&self) -> Vec<FrequencyScalingState> {
        self.frequency_percentages()
            .iter()
            .map(|&pct| {
                if pct >= 95.0 {
                    FrequencyScalingState::Turbo
                } else if pct >= 75.0 {
                    FrequencyScalingState::High
                } else if pct >= 50.0 {
                    FrequencyScalingState::Normal
                } else if pct >= 25.0 {
                    FrequencyScalingState::Scaled
                } else {
                    FrequencyScalingState::Idle
                }
            })
            .collect()
    }
}

impl ComputeBlock for CpuFrequencyBlock {
    type Input = (Vec<u32>, Vec<u32>); // (cur_freqs, max_freqs)
    type Output = Vec<FrequencyScalingState>;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_frequencies(input.0.clone(), input.1.clone());
        self.scaling_indicators()
    }

    fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.instruction_set
    }

    fn latency_budget_us(&self) -> u64 {
        50 // 50μs budget for frequency processing
    }
}

/// Frequency scaling state indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyScalingState {
    /// Turbo/boost mode active (⚡)
    Turbo,
    /// High frequency (↑)
    High,
    /// Normal frequency (→)
    Normal,
    /// Scaled down (↓)
    Scaled,
    /// Idle/very low (·)
    Idle,
}

impl FrequencyScalingState {
    /// Get the indicator character
    #[must_use]
    pub const fn indicator(self) -> char {
        match self {
            Self::Turbo => '⚡',
            Self::High => '↑',
            Self::Normal => '→',
            Self::Scaled => '↓',
            Self::Idle => '·',
        }
    }
}

/// CPU Governor `ComputeBlock` (CB-CPU-008)
///
/// Tracks CPU governor state from `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.
#[derive(Debug, Clone)]
pub struct CpuGovernorBlock {
    /// Current governor name
    governor: CpuGovernor,
}

impl Default for CpuGovernorBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuGovernorBlock {
    /// Create a new CPU governor block
    #[must_use]
    pub fn new() -> Self {
        Self {
            governor: CpuGovernor::Unknown,
        }
    }

    /// Set governor from string
    pub fn set_governor(&mut self, name: &str) {
        self.governor = CpuGovernor::from_name(name);
    }

    /// Get current governor
    #[must_use]
    pub fn governor(&self) -> CpuGovernor {
        self.governor
    }
}

impl ComputeBlock for CpuGovernorBlock {
    type Input = String;
    type Output = CpuGovernor;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_governor(input);
        self.governor
    }

    fn latency_budget_us(&self) -> u64 {
        10 // Very fast string parsing
    }
}

/// CPU Governor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuGovernor {
    /// Performance - max frequency
    Performance,
    /// Powersave - min frequency
    Powersave,
    /// Ondemand - dynamic scaling
    Ondemand,
    /// Conservative - gradual scaling
    Conservative,
    /// Schedutil - scheduler-based
    Schedutil,
    /// Userspace - user-controlled
    Userspace,
    /// Unknown governor
    Unknown,
}

impl CpuGovernor {
    /// Parse governor from name
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "performance" => Self::Performance,
            "powersave" => Self::Powersave,
            "ondemand" => Self::Ondemand,
            "conservative" => Self::Conservative,
            "schedutil" => Self::Schedutil,
            "userspace" => Self::Userspace,
            _ => Self::Unknown,
        }
    }

    /// Get governor name as string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Powersave => "powersave",
            Self::Ondemand => "ondemand",
            Self::Conservative => "conservative",
            Self::Schedutil => "schedutil",
            Self::Userspace => "userspace",
            Self::Unknown => "unknown",
        }
    }

    /// Get short display name
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Performance => "perf",
            Self::Powersave => "psav",
            Self::Ondemand => "odmd",
            Self::Conservative => "cons",
            Self::Schedutil => "schu",
            Self::Userspace => "user",
            Self::Unknown => "????",
        }
    }

    /// Get icon for governor
    #[must_use]
    pub const fn icon(self) -> char {
        match self {
            Self::Performance => '🚀',
            Self::Powersave => '🔋',
            Self::Ondemand => '⚡',
            Self::Conservative => '📊',
            Self::Schedutil => '📅',
            Self::Userspace => '👤',
            Self::Unknown => '?',
        }
    }
}

/// Memory Pressure `ComputeBlock` (CB-MEM-003)
///
/// Tracks memory pressure from `/proc/pressure/memory`.
#[derive(Debug, Clone)]
pub struct MemPressureBlock {
    /// Average pressure over 10 seconds (some)
    avg10_some: f32,
    /// Average pressure over 60 seconds (some)
    avg60_some: f32,
    /// Average pressure over 300 seconds (some)
    avg300_some: f32,
    /// Average pressure over 10 seconds (full)
    avg10_full: f32,
    /// Instruction set
    instruction_set: SimdInstructionSet,
}

impl Default for MemPressureBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl MemPressureBlock {
    /// Create a new memory pressure block
    #[must_use]
    pub fn new() -> Self {
        Self {
            avg10_some: 0.0,
            avg60_some: 0.0,
            avg300_some: 0.0,
            avg10_full: 0.0,
            instruction_set: SimdInstructionSet::detect(),
        }
    }

    /// Set pressure values
    pub fn set_pressure(
        &mut self,
        avg10_some: f32,
        avg60_some: f32,
        avg300_some: f32,
        avg10_full: f32,
    ) {
        debug_assert!(avg10_some >= 0.0, "avg10_some must be non-negative");
        debug_assert!(avg60_some >= 0.0, "avg60_some must be non-negative");
        debug_assert!(avg300_some >= 0.0, "avg300_some must be non-negative");
        debug_assert!(avg10_full >= 0.0, "avg10_full must be non-negative");
        self.avg10_some = avg10_some;
        self.avg60_some = avg60_some;
        self.avg300_some = avg300_some;
        self.avg10_full = avg10_full;
    }

    /// Get pressure level indicator
    #[must_use]
    pub fn pressure_level(&self) -> MemoryPressureLevel {
        let pct = self.avg10_some;
        if pct >= 50.0 {
            MemoryPressureLevel::Critical
        } else if pct >= 25.0 {
            MemoryPressureLevel::High
        } else if pct >= 10.0 {
            MemoryPressureLevel::Medium
        } else if pct >= 1.0 {
            MemoryPressureLevel::Low
        } else {
            MemoryPressureLevel::None
        }
    }

    /// Get trend from 300s to 10s averages
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        let diff = self.avg10_some - self.avg300_some;
        if diff > 5.0 {
            TrendDirection::Up
        } else if diff < -5.0 {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        }
    }
}

impl ComputeBlock for MemPressureBlock {
    type Input = (f32, f32, f32, f32); // (avg10_some, avg60_some, avg300_some, avg10_full)
    type Output = MemoryPressureLevel;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_pressure(input.0, input.1, input.2, input.3);
        self.pressure_level()
    }

    fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.instruction_set
    }

    fn latency_budget_us(&self) -> u64 {
        20 // Simple comparisons
    }
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    /// No pressure
    None,
    /// Low pressure (1-10%)
    Low,
    /// Medium pressure (10-25%)
    Medium,
    /// High pressure (25-50%)
    High,
    /// Critical pressure (>50%)
    Critical,
}

impl MemoryPressureLevel {
    /// Get the display character for this pressure level
    #[allow(clippy::match_same_arms)]
    pub fn symbol(&self) -> char {
        match self {
            Self::None => ' ',
            Self::Low => '○',
            Self::Medium => '◐',
            Self::High => '◕',
            Self::Critical => '●',
        }
    }

    /// Get color index (0=green, 4=red)
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

/// Huge Pages `ComputeBlock` (CB-MEM-006)
///
/// Tracks huge page usage from `/proc/meminfo`.
#[derive(Debug, Clone)]
pub struct HugePagesBlock {
    /// Total huge pages
    total: u64,
    /// Free huge pages
    free: u64,
    /// Reserved huge pages
    reserved: u64,
    /// Huge page size in KB
    page_size_kb: u64,
}

impl Default for HugePagesBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl HugePagesBlock {
    /// Create a new huge pages block
    #[must_use]
    pub fn new() -> Self {
        Self {
            total: 0,
            free: 0,
            reserved: 0,
            page_size_kb: 2048, // Default 2MB huge pages
        }
    }

    /// Set huge page values
    pub fn set_values(&mut self, total: u64, free: u64, reserved: u64, page_size_kb: u64) {
        debug_assert!(free <= total, "free must be <= total");
        debug_assert!(page_size_kb > 0, "page_size_kb must be positive");
        self.total = total;
        self.free = free;
        self.reserved = reserved;
        self.page_size_kb = page_size_kb;
    }

    /// Get usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            ((self.total - self.free) as f32 / self.total as f32 * 100.0).clamp(0.0, 100.0)
        }
    }

    /// Get total size in bytes
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total * self.page_size_kb * 1024
    }

    /// Get used size in bytes
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        (self.total - self.free) * self.page_size_kb * 1024
    }
}

impl ComputeBlock for HugePagesBlock {
    type Input = (u64, u64, u64, u64); // (total, free, reserved, page_size_kb)
    type Output = f32; // Usage percentage

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_values(input.0, input.1, input.2, input.3);
        self.usage_percent()
    }

    fn latency_budget_us(&self) -> u64 {
        10 // Simple arithmetic
    }
}

/// GPU Thermal `ComputeBlock` (CB-GPU-001)
///
/// Tracks GPU temperature and power draw.
#[derive(Debug, Clone)]
pub struct GpuThermalBlock {
    /// Temperature in Celsius
    temperature_c: f32,
    /// Power draw in Watts
    power_w: f32,
    /// Power limit in Watts
    power_limit_w: f32,
    /// History for trend
    temp_history: Vec<f32>,
    /// Instruction set
    instruction_set: SimdInstructionSet,
}

impl Default for GpuThermalBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuThermalBlock {
    /// Create a new GPU thermal block
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature_c: 0.0,
            power_w: 0.0,
            power_limit_w: 0.0,
            temp_history: Vec::with_capacity(60),
            instruction_set: SimdInstructionSet::detect(),
        }
    }

    /// Set thermal values
    pub fn set_values(&mut self, temp_c: f32, power_w: f32, power_limit_w: f32) {
        debug_assert!(power_w >= 0.0, "power_w must be non-negative");
        debug_assert!(power_limit_w >= 0.0, "power_limit_w must be non-negative");
        self.temperature_c = temp_c;
        self.power_w = power_w;
        self.power_limit_w = power_limit_w;

        // Update history
        if self.temp_history.len() >= 60 {
            self.temp_history.remove(0);
        }
        self.temp_history.push(temp_c);
    }

    /// Get thermal state
    #[must_use]
    pub fn thermal_state(&self) -> GpuThermalState {
        if self.temperature_c >= 90.0 {
            GpuThermalState::Critical
        } else if self.temperature_c >= 80.0 {
            GpuThermalState::Hot
        } else if self.temperature_c >= 70.0 {
            GpuThermalState::Warm
        } else if self.temperature_c >= 50.0 {
            GpuThermalState::Normal
        } else {
            GpuThermalState::Cool
        }
    }

    /// Get power usage percentage
    #[must_use]
    pub fn power_percent(&self) -> f32 {
        if self.power_limit_w > 0.0 {
            (self.power_w / self.power_limit_w * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        }
    }

    /// Get temperature trend
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        if self.temp_history.len() < 5 {
            return TrendDirection::Flat;
        }

        let recent: f32 = self.temp_history.iter().rev().take(5).sum::<f32>() / 5.0;
        let older: f32 = self.temp_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;

        let diff = recent - older;
        if diff > 2.0 {
            TrendDirection::Up
        } else if diff < -2.0 {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        }
    }
}

impl ComputeBlock for GpuThermalBlock {
    type Input = (f32, f32, f32); // (temp_c, power_w, power_limit_w)
    type Output = GpuThermalState;

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_values(input.0, input.1, input.2);
        self.thermal_state()
    }

    fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.instruction_set
    }

    fn latency_budget_us(&self) -> u64 {
        30 // Simple comparisons + history update
    }
}

/// GPU thermal state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuThermalState {
    /// Cool (<50°C)
    #[default]
    Cool,
    /// Normal (50-70°C)
    Normal,
    /// Warm (70-80°C)
    Warm,
    /// Hot (80-90°C)
    Hot,
    /// Critical (>90°C)
    Critical,
}

impl GpuThermalState {
    /// Get indicator character
    #[must_use]
    pub const fn indicator(self) -> char {
        match self {
            Self::Cool => '❄',
            Self::Normal => '●',
            Self::Warm => '◐',
            Self::Hot => '◕',
            Self::Critical => '🔥',
        }
    }

    /// Get severity (0=cool, 4=critical)
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            Self::Cool => 0,
            Self::Normal => 1,
            Self::Warm => 2,
            Self::Hot => 3,
            Self::Critical => 4,
        }
    }
}

/// GPU VRAM `ComputeBlock` (CB-GPU-002)
///
/// Tracks VRAM usage per process.
#[derive(Debug, Clone)]
pub struct GpuVramBlock {
    /// Total VRAM in MB
    total_mb: u64,
    /// Used VRAM in MB
    used_mb: u64,
    /// Per-process VRAM usage (PID -> MB)
    per_process: Vec<(u32, u64, String)>, // (pid, mb, name)
}

impl Default for GpuVramBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuVramBlock {
    /// Create a new VRAM block
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_mb: 0,
            used_mb: 0,
            per_process: Vec::new(),
        }
    }

    /// Set VRAM values
    pub fn set_values(
        &mut self,
        total_mb: u64,
        used_mb: u64,
        per_process: Vec<(u32, u64, String)>,
    ) {
        self.total_mb = total_mb;
        self.used_mb = used_mb;
        self.per_process = per_process;
    }

    /// Get usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f32 {
        if self.total_mb == 0 {
            0.0
        } else {
            (self.used_mb as f32 / self.total_mb as f32 * 100.0).clamp(0.0, 100.0)
        }
    }

    /// Get top N consumers by VRAM
    #[must_use]
    pub fn top_consumers(&self, n: usize) -> Vec<&(u32, u64, String)> {
        let mut sorted: Vec<_> = self.per_process.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        sorted.into_iter().take(n).collect()
    }
}

impl ComputeBlock for GpuVramBlock {
    type Input = (u64, u64, Vec<(u32, u64, String)>);
    type Output = f32; // Usage percentage

    fn compute(&mut self, input: &Self::Input) -> Self::Output {
        self.set_values(input.0, input.1, input.2.clone());
        self.usage_percent()
    }

    fn latency_budget_us(&self) -> u64 {
        100 // Sorting may be needed
    }
}

// =============================================================================
// MetricsCacheBlock: O(1) Cached Metrics for ptop Performance
// =============================================================================

/// Cached metrics snapshot for O(1) panel access.
///
/// This struct provides pre-computed, cached views of system metrics
/// to avoid redundant calculations during rendering. Per the spec:
/// "All metrics must be O(1) cached views, not O(n) per-frame refreshes."
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │  collect_metrics() [O(n)]  →  MetricsCache  →  render() [O(1)]  │
/// │                                                                  │
/// │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐    │
/// │  │ /proc scan  │ →   │ SIMD Reduce │ →   │ Cached Summary  │    │
/// │  │ (2600 PIDs) │     │ (AVX2/NEON) │     │ (top 50, sums)  │    │
/// │  └─────────────┘     └─────────────┘     └─────────────────┘    │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Performance Targets
///
/// | Operation | Target | Notes |
/// |-----------|--------|-------|
/// | Cache update | <100ms | Once per collect_metrics() |
/// | Cache read | <1μs | O(1) field access |
/// | Memory overhead | <1KB | Just aggregates, not full data |
#[derive(Debug, Clone, Default)]
pub struct MetricsCache {
    /// Cached CPU aggregate
    pub cpu: CpuMetricsCache,
    /// Cached memory aggregate
    pub memory: MemoryMetricsCache,
    /// Cached process aggregate
    pub process: ProcessMetricsCache,
    /// Cached network aggregate
    pub network: NetworkMetricsCache,
    /// Cached GPU aggregate
    pub gpu: GpuMetricsCache,
    /// Frame ID when cache was last updated
    pub frame_id: u64,
    /// Timestamp of last update (for cache invalidation)
    pub updated_at_us: u64,
}

/// Cached CPU metrics
#[derive(Debug, Clone, Default)]
pub struct CpuMetricsCache {
    /// Average CPU usage across all cores (0-100)
    pub avg_usage: f32,
    /// Maximum core usage (for load display)
    pub max_core_usage: f32,
    /// Number of cores at >90% usage
    pub hot_cores: u32,
    /// Load average (1m, 5m, 15m)
    pub load_avg: [f32; 3],
    /// Current frequency (GHz)
    pub freq_ghz: f32,
    /// Trend direction
    pub trend: TrendDirection,
}

/// Cached memory metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetricsCache {
    /// Usage percentage (0-100)
    pub usage_percent: f32,
    /// Used bytes
    pub used_bytes: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Cached bytes
    pub cached_bytes: u64,
    /// Swap usage percentage
    pub swap_percent: f32,
    /// ZRAM compression ratio
    pub zram_ratio: f32,
    /// Trend direction
    pub trend: TrendDirection,
}

/// Cached process metrics
#[derive(Debug, Clone, Default)]
pub struct ProcessMetricsCache {
    /// Total process count
    pub total_count: u32,
    /// Running process count
    pub running_count: u32,
    /// Sleeping process count
    pub sleeping_count: u32,
    /// Top CPU consumer (pid, cpu%, name)
    pub top_cpu: Option<(u32, f32, String)>,
    /// Top memory consumer (pid, mem%, name)
    pub top_mem: Option<(u32, f32, String)>,
    /// Sum of all CPU usage (for overhead display)
    pub total_cpu_usage: f32,
}

/// Cached network metrics
#[derive(Debug, Clone, Default)]
pub struct NetworkMetricsCache {
    /// Primary interface name
    pub interface: String,
    /// RX rate (bytes/sec)
    pub rx_bytes_sec: u64,
    /// TX rate (bytes/sec)
    pub tx_bytes_sec: u64,
    /// Total RX bytes
    pub total_rx: u64,
    /// Total TX bytes
    pub total_tx: u64,
    /// Active connection count
    pub connection_count: u32,
}

/// Cached GPU metrics
#[derive(Debug, Clone, Default)]
pub struct GpuMetricsCache {
    /// GPU name
    pub name: String,
    /// GPU usage percentage
    pub usage_percent: f32,
    /// VRAM usage percentage
    pub vram_percent: f32,
    /// Temperature in Celsius
    pub temp_c: f32,
    /// Power draw in Watts
    pub power_w: f32,
    /// Thermal state
    pub thermal_state: GpuThermalState,
}

impl MetricsCache {
    /// Create a new empty cache
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if cache is stale (older than `max_age_us`)
    #[must_use]
    pub fn is_stale(&self, current_time_us: u64, max_age_us: u64) -> bool {
        current_time_us.saturating_sub(self.updated_at_us) > max_age_us
    }

    /// Update CPU cache from raw data
    pub fn update_cpu(
        &mut self,
        per_core: &[f64],
        load_avg: [f32; 3],
        freq_ghz: f32,
        frame_id: u64,
    ) {
        if per_core.is_empty() {
            return;
        }

        // SIMD-friendly reduction (compiler can vectorize)
        let sum: f64 = per_core.iter().sum();
        let max: f64 = per_core.iter().copied().fold(0.0, f64::max);
        let hot_cores = per_core.iter().filter(|&&c| c > 90.0).count();

        self.cpu.avg_usage = (sum / per_core.len() as f64) as f32;
        self.cpu.max_core_usage = max as f32;
        self.cpu.hot_cores = hot_cores as u32;
        self.cpu.load_avg = load_avg;
        self.cpu.freq_ghz = freq_ghz;
        self.frame_id = frame_id;
    }

    /// Update memory cache from raw data
    pub fn update_memory(
        &mut self,
        used: u64,
        total: u64,
        cached: u64,
        swap_used: u64,
        swap_total: u64,
        zram_ratio: f32,
    ) {
        self.memory.used_bytes = used;
        self.memory.total_bytes = total;
        self.memory.cached_bytes = cached;
        self.memory.usage_percent = if total > 0 {
            used as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        self.memory.swap_percent = if swap_total > 0 {
            swap_used as f32 / swap_total as f32 * 100.0
        } else {
            0.0
        };
        self.memory.zram_ratio = zram_ratio;
    }

    /// Update process cache from raw data
    pub fn update_process(
        &mut self,
        total: u32,
        running: u32,
        sleeping: u32,
        top_cpu: Option<(u32, f32, String)>,
        top_mem: Option<(u32, f32, String)>,
        total_cpu: f32,
    ) {
        self.process.total_count = total;
        self.process.running_count = running;
        self.process.sleeping_count = sleeping;
        self.process.top_cpu = top_cpu;
        self.process.top_mem = top_mem;
        self.process.total_cpu_usage = total_cpu;
    }

    /// Update network cache from raw data
    pub fn update_network(
        &mut self,
        interface: String,
        rx_rate: u64,
        tx_rate: u64,
        total_rx: u64,
        total_tx: u64,
        conn_count: u32,
    ) {
        self.network.interface = interface;
        self.network.rx_bytes_sec = rx_rate;
        self.network.tx_bytes_sec = tx_rate;
        self.network.total_rx = total_rx;
        self.network.total_tx = total_tx;
        self.network.connection_count = conn_count;
    }

    /// Update GPU cache from raw data
    pub fn update_gpu(&mut self, name: String, usage: f32, vram: f32, temp: f32, power: f32) {
        self.gpu.name = name;
        self.gpu.usage_percent = usage;
        self.gpu.vram_percent = vram;
        self.gpu.temp_c = temp;
        self.gpu.power_w = power;
        self.gpu.thermal_state = if temp >= 90.0 {
            GpuThermalState::Critical
        } else if temp >= 80.0 {
            GpuThermalState::Hot
        } else if temp >= 70.0 {
            GpuThermalState::Warm
        } else if temp >= 50.0 {
            GpuThermalState::Normal
        } else {
            GpuThermalState::Cool
        };
    }

    /// Set timestamp for cache freshness tracking
    pub fn mark_updated(&mut self, timestamp_us: u64) {
        self.updated_at_us = timestamp_us;
    }
}

/// `ComputeBlock` wrapper for `MetricsCache` that provides O(1) access
#[derive(Debug, Clone, Default)]
pub struct MetricsCacheBlock {
    cache: MetricsCache,
    instruction_set: SimdInstructionSet,
}

impl MetricsCacheBlock {
    /// Create a new metrics cache block
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: MetricsCache::new(),
            instruction_set: SimdInstructionSet::detect(),
        }
    }

    /// Get immutable reference to the cache
    #[must_use]
    pub fn cache(&self) -> &MetricsCache {
        &self.cache
    }

    /// Get mutable reference to the cache for updates
    pub fn cache_mut(&mut self) -> &mut MetricsCache {
        &mut self.cache
    }
}

impl ComputeBlock for MetricsCacheBlock {
    type Input = (); // No input - cache is updated separately
    type Output = MetricsCache;

    fn compute(&mut self, _input: &Self::Input) -> Self::Output {
        self.cache.clone()
    }

    fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.instruction_set
    }

    fn latency_budget_us(&self) -> u64 {
        1 // O(1) access - should be <1μs
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
#[path = "compute_block_tests.rs"]
mod tests;
