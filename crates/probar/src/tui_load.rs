//! TUI Load Testing Module
//!
//! Framework-agnostic load testing for terminal user interfaces.
//! Works with any TUI framework (presentar, ratatui, crossterm, etc.).
//!
//! ## Key Features
//!
//! - **Framework Agnostic**: Uses callbacks/closures, not tied to any TUI library
//! - **Data Generation**: Generate realistic data volumes (1000+ items)
//! - **Frame Timing**: Measure render time per frame
//! - **Hang Detection**: Timeout-based hang detection with detailed diagnostics
//! - **Filter/Search Testing**: Test filter performance with large datasets
//!
//! ## Toyota Way Application
//!
//! - **Jidoka**: Automatic hang detection stops tests before infinite loops
//! - **Muda**: Eliminate wasted time from slow filter implementations
//! - **Genchi Genbutsu**: Measure actual render times, not theoretical
//!
//! ## Example
//!
//! ```ignore
//! use jugar_probar::tui_load::{TuiLoadTest, DataGenerator, TuiFrameMetrics};
//!
//! // Create load test with 5000 synthetic processes
//! let mut load_test = TuiLoadTest::new()
//!     .with_item_count(5000)
//!     .with_frame_budget_ms(16)  // 60 FPS target
//!     .with_timeout_ms(1000);     // 1 second hang detection
//!
//! // Run test with your render function
//! let result = load_test.run(|items, filter| {
//!     // Your TUI render logic here
//!     // Returns frame time in microseconds
//!     render_process_list(items, filter)
//! });
//!
//! assert!(result.is_ok(), "TUI should not hang");
//! assert!(result.unwrap().p95_frame_ms < 16.0, "Should maintain 60 FPS");
//! ```

use std::time::{Duration, Instant};

/// Result type for TUI load tests
pub type TuiLoadResult<T> = Result<T, TuiLoadError>;

/// Errors that can occur during TUI load testing
#[derive(Debug, Clone, PartialEq)]
pub enum TuiLoadError {
    /// Frame render exceeded timeout (likely hang)
    FrameTimeout {
        /// Frame number that timed out
        frame: usize,
        /// Timeout in milliseconds
        timeout_ms: u64,
        /// Filter text being used (if any)
        filter: String,
        /// Number of items being rendered
        item_count: usize,
    },
    /// Frame budget exceeded (too slow, but not a hang)
    BudgetExceeded {
        /// Frame number
        frame: usize,
        /// Actual frame time in milliseconds
        actual_ms: f64,
        /// Budget in milliseconds
        budget_ms: f64,
    },
    /// Data generation failed
    DataGenerationFailed {
        /// Error message
        message: String,
    },
}

impl std::fmt::Display for TuiLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameTimeout {
                frame,
                timeout_ms,
                filter,
                item_count,
            } => {
                write!(
                    f,
                    "Frame {} timed out after {}ms (filter='{}', items={})",
                    frame, timeout_ms, filter, item_count
                )
            }
            Self::BudgetExceeded {
                frame,
                actual_ms,
                budget_ms,
            } => {
                write!(
                    f,
                    "Frame {} exceeded budget: {:.2}ms > {:.2}ms",
                    frame, actual_ms, budget_ms
                )
            }
            Self::DataGenerationFailed { message } => {
                write!(f, "Data generation failed: {}", message)
            }
        }
    }
}

impl std::error::Error for TuiLoadError {}

/// Metrics collected during TUI load testing
#[derive(Debug, Clone, Default)]
pub struct TuiFrameMetrics {
    /// Total frames rendered
    pub frame_count: usize,
    /// Total render time in microseconds
    pub total_time_us: u64,
    /// Minimum frame time in microseconds
    pub min_frame_us: u64,
    /// Maximum frame time in microseconds
    pub max_frame_us: u64,
    /// Frame times for percentile calculation
    pub frame_times_us: Vec<u64>,
}

impl TuiFrameMetrics {
    /// Create new empty metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_frame_us: u64::MAX,
            ..Default::default()
        }
    }

    /// Record a frame time
    pub fn record(&mut self, frame_time_us: u64) {
        self.frame_count += 1;
        self.total_time_us += frame_time_us;
        self.min_frame_us = self.min_frame_us.min(frame_time_us);
        self.max_frame_us = self.max_frame_us.max(frame_time_us);
        self.frame_times_us.push(frame_time_us);
    }

    /// Get average frame time in milliseconds
    #[must_use]
    pub fn avg_frame_ms(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        (self.total_time_us as f64 / self.frame_count as f64) / 1000.0
    }

    /// Get minimum frame time in milliseconds
    #[must_use]
    pub fn min_frame_ms(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.min_frame_us as f64 / 1000.0
    }

    /// Get maximum frame time in milliseconds
    #[must_use]
    pub fn max_frame_ms(&self) -> f64 {
        self.max_frame_us as f64 / 1000.0
    }

    /// Get p50 (median) frame time in milliseconds
    #[must_use]
    pub fn p50_frame_ms(&self) -> f64 {
        self.percentile(50)
    }

    /// Get p95 frame time in milliseconds
    #[must_use]
    pub fn p95_frame_ms(&self) -> f64 {
        self.percentile(95)
    }

    /// Get p99 frame time in milliseconds
    #[must_use]
    pub fn p99_frame_ms(&self) -> f64 {
        self.percentile(99)
    }

    /// Get percentile frame time in milliseconds
    #[must_use]
    pub fn percentile(&self, p: u8) -> f64 {
        if self.frame_times_us.is_empty() {
            return 0.0;
        }
        let mut sorted = self.frame_times_us.clone();
        sorted.sort_unstable();
        let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted[idx.min(sorted.len() - 1)] as f64 / 1000.0
    }

    /// Check if frame times meet target FPS
    #[must_use]
    pub fn meets_fps(&self, target_fps: u32) -> bool {
        let budget_ms = 1000.0 / target_fps as f64;
        self.p95_frame_ms() <= budget_ms
    }
}

/// A synthetic item for load testing (framework-agnostic)
#[derive(Debug, Clone)]
pub struct SyntheticItem {
    /// Unique identifier
    pub id: u32,
    /// Name (searchable)
    pub name: String,
    /// Description/command line (searchable, can be long)
    pub description: String,
    /// Numeric value 1 (e.g., CPU %)
    pub value1: f32,
    /// Numeric value 2 (e.g., memory %)
    pub value2: f32,
    /// State string
    pub state: String,
    /// Owner/user
    pub owner: String,
    /// Additional count (e.g., threads)
    pub count: u32,
}

impl SyntheticItem {
    /// Check if item matches filter (case-insensitive)
    #[must_use]
    pub fn matches_filter(&self, filter: &str) -> bool {
        if filter.is_empty() {
            return true;
        }
        let filter_lower = filter.to_lowercase();
        self.name.to_lowercase().contains(&filter_lower)
            || self.description.to_lowercase().contains(&filter_lower)
    }

    /// Optimized filter matching with pre-lowercased filter
    #[must_use]
    pub fn matches_filter_precomputed(&self, filter_lower: &str) -> bool {
        if filter_lower.is_empty() {
            return true;
        }
        self.name.to_lowercase().contains(filter_lower)
            || self.description.to_lowercase().contains(filter_lower)
    }
}

/// Data generator for synthetic load test data
#[derive(Debug, Clone)]
pub struct DataGenerator {
    /// Random seed for reproducibility
    seed: u64,
    /// Number of items to generate
    item_count: usize,
    /// Average description length
    avg_description_len: usize,
}

impl DataGenerator {
    /// Create a new data generator
    #[must_use]
    pub fn new(item_count: usize) -> Self {
        Self {
            seed: 42,
            item_count,
            avg_description_len: 100,
        }
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set average description length
    #[must_use]
    pub fn with_description_len(mut self, len: usize) -> Self {
        self.avg_description_len = len;
        self
    }

    /// Generate synthetic items
    #[must_use]
    pub fn generate(&self) -> Vec<SyntheticItem> {
        let mut items = Vec::with_capacity(self.item_count);
        let mut rng_state = self.seed;

        let names = [
            "systemd",
            "kworker",
            "chrome",
            "firefox",
            "code",
            "rust-analyzer",
            "node",
            "python",
            "java",
            "postgres",
            "nginx",
            "docker",
            "containerd",
            "ssh",
            "bash",
            "zsh",
            "fish",
            "vim",
            "nvim",
            "emacs",
            "tmux",
            "htop",
            "top",
            "ps",
            "grep",
            "find",
            "cargo",
            "rustc",
            "gcc",
            "clang",
            "llvm",
            "git",
            "make",
            "cmake",
            "webpack",
            "vite",
        ];

        let states = ["R", "S", "D", "Z", "T", "I"];
        let users = ["root", "noah", "www-data", "postgres", "nobody", "daemon"];

        for i in 0..self.item_count {
            // Simple LCG PRNG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r1 = rng_state;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r2 = rng_state;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r3 = rng_state;

            let name_idx = (r1 as usize) % names.len();
            let state_idx = (r2 as usize) % states.len();
            let user_idx = (r3 as usize) % users.len();

            // Generate a realistic command line
            let base_name = names[name_idx];
            let pid = 1000 + i as u32;
            let description = self.generate_cmdline(base_name, r1);

            items.push(SyntheticItem {
                id: pid,
                name: format!("{}-{}", base_name, i % 100),
                description,
                value1: ((r1 % 10000) as f32) / 100.0, // CPU 0-100%
                value2: ((r2 % 10000) as f32) / 100.0, // Mem 0-100%
                state: states[state_idx].to_string(),
                owner: users[user_idx].to_string(),
                count: ((r3 % 64) + 1) as u32, // threads 1-64
            });
        }

        items
    }

    fn generate_cmdline(&self, base_name: &str, seed: u64) -> String {
        let args = [
            "--config",
            "/etc/config.yaml",
            "--port",
            "8080",
            "--workers",
            "4",
            "--log-level",
            "debug",
            "--data-dir",
            "/var/lib/data",
            "--cache-size",
            "1024",
            "--timeout",
            "30",
            "--max-connections",
            "1000",
            "--enable-metrics",
            "--prometheus-port",
            "9090",
        ];

        let mut cmdline = format!("/usr/bin/{}", base_name);
        let arg_count = ((seed % 6) + 2) as usize;

        for i in 0..arg_count {
            let arg_idx = ((seed.wrapping_add(i as u64 * 7)) % (args.len() as u64)) as usize;
            cmdline.push(' ');
            cmdline.push_str(args[arg_idx]);
        }

        // Pad to approximate target length
        while cmdline.len() < self.avg_description_len {
            cmdline.push_str(" --extra-arg");
        }

        cmdline
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Configuration for TUI load testing
#[derive(Debug, Clone)]
pub struct TuiLoadConfig {
    /// Number of items to generate
    pub item_count: usize,
    /// Frame budget in milliseconds (e.g., 16ms for 60 FPS)
    pub frame_budget_ms: f64,
    /// Timeout for hang detection in milliseconds
    pub timeout_ms: u64,
    /// Number of frames to render per filter
    pub frames_per_filter: usize,
    /// Filter strings to test
    pub filters: Vec<String>,
    /// Fail on budget exceeded (vs just warn)
    pub strict_budget: bool,
}

impl Default for TuiLoadConfig {
    fn default() -> Self {
        Self {
            item_count: 1000,
            frame_budget_ms: 16.67, // 60 FPS
            timeout_ms: 1000,       // 1 second
            frames_per_filter: 10,
            filters: vec![
                String::new(),
                "a".to_string(),
                "sys".to_string(),
                "chrome".to_string(),
                "nonexistent_filter_that_matches_nothing".to_string(),
            ],
            strict_budget: false,
        }
    }
}

/// TUI Load Test Runner
///
/// Framework-agnostic load testing for TUI applications.
#[derive(Debug)]
pub struct TuiLoadTest {
    config: TuiLoadConfig,
    data: Vec<SyntheticItem>,
}

impl TuiLoadTest {
    /// Create a new TUI load test with default configuration
    #[must_use]
    pub fn new() -> Self {
        let config = TuiLoadConfig::default();
        let data = DataGenerator::new(config.item_count).generate();
        Self { config, data }
    }

    /// Create with specific item count
    #[must_use]
    pub fn with_item_count(mut self, count: usize) -> Self {
        self.config.item_count = count;
        self.data = DataGenerator::new(count).generate();
        self
    }

    /// Set frame budget in milliseconds
    #[must_use]
    pub fn with_frame_budget_ms(mut self, budget_ms: f64) -> Self {
        self.config.frame_budget_ms = budget_ms;
        self
    }

    /// Set timeout for hang detection
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = timeout_ms;
        self
    }

    /// Set filters to test
    #[must_use]
    pub fn with_filters(mut self, filters: Vec<String>) -> Self {
        self.config.filters = filters;
        self
    }

    /// Set number of frames per filter
    #[must_use]
    pub fn with_frames_per_filter(mut self, count: usize) -> Self {
        self.config.frames_per_filter = count;
        self
    }

    /// Enable strict budget enforcement
    #[must_use]
    pub fn with_strict_budget(mut self, strict: bool) -> Self {
        self.config.strict_budget = strict;
        self
    }

    /// Get the generated test data
    #[must_use]
    pub fn data(&self) -> &[SyntheticItem] {
        &self.data
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &TuiLoadConfig {
        &self.config
    }

    /// Run load test with a render callback
    ///
    /// The callback receives:
    /// - `items`: Slice of synthetic items to render
    /// - `filter`: Current filter string
    ///
    /// The callback should perform the actual TUI rendering and return
    /// the frame time in microseconds, or None if it wants the test
    /// harness to measure time.
    ///
    /// # Errors
    ///
    /// Returns error if a frame times out (hang detected) or exceeds
    /// budget in strict mode.
    pub fn run<F>(&self, mut render: F) -> TuiLoadResult<TuiFrameMetrics>
    where
        F: FnMut(&[SyntheticItem], &str) -> Option<u64>,
    {
        let mut metrics = TuiFrameMetrics::new();
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let mut frame_num = 0;

        for filter in &self.config.filters {
            for _ in 0..self.config.frames_per_filter {
                let start = Instant::now();

                // Call render function
                let frame_time_us = if let Some(reported_time) = render(&self.data, filter) {
                    reported_time
                } else {
                    // Measure time ourselves
                    start.elapsed().as_micros() as u64
                };

                let elapsed = start.elapsed();

                // Check for timeout (hang detection)
                if elapsed > timeout {
                    return Err(TuiLoadError::FrameTimeout {
                        frame: frame_num,
                        timeout_ms: self.config.timeout_ms,
                        filter: filter.clone(),
                        item_count: self.data.len(),
                    });
                }

                // Check budget
                let frame_ms = frame_time_us as f64 / 1000.0;
                if self.config.strict_budget && frame_ms > self.config.frame_budget_ms {
                    return Err(TuiLoadError::BudgetExceeded {
                        frame: frame_num,
                        actual_ms: frame_ms,
                        budget_ms: self.config.frame_budget_ms,
                    });
                }

                metrics.record(frame_time_us);
                frame_num += 1;
            }
        }

        Ok(metrics)
    }

    /// Run filter-specific performance test
    ///
    /// Tests filtering with increasing filter lengths to detect
    /// O(n²) or worse complexity in filter implementations.
    ///
    /// # Errors
    ///
    /// Returns error if hang detected or budget exceeded.
    pub fn run_filter_stress<F>(
        &self,
        mut filter_fn: F,
    ) -> TuiLoadResult<Vec<(String, TuiFrameMetrics)>>
    where
        F: FnMut(&[SyntheticItem], &str) -> Vec<SyntheticItem>,
    {
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let mut results = Vec::new();

        // Test filters of increasing length/complexity
        let stress_filters = [
            "",
            "a",
            "ab",
            "abc",
            "sys",
            "syst",
            "syste",
            "system",
            "systemd",
            "chrome",
            "rust-analyzer",
            "this_filter_will_match_nothing_at_all",
        ];

        for filter in stress_filters {
            let mut metrics = TuiFrameMetrics::new();

            for frame in 0..self.config.frames_per_filter {
                let start = Instant::now();

                // Call filter function
                let _filtered = filter_fn(&self.data, filter);

                let elapsed = start.elapsed();

                // Check for timeout
                if elapsed > timeout {
                    return Err(TuiLoadError::FrameTimeout {
                        frame,
                        timeout_ms: self.config.timeout_ms,
                        filter: filter.to_string(),
                        item_count: self.data.len(),
                    });
                }

                metrics.record(elapsed.as_micros() as u64);
            }

            results.push((filter.to_string(), metrics));
        }

        Ok(results)
    }
}

impl Default for TuiLoadTest {
    fn default() -> Self {
        Self::new()
    }
}

/// Assertion helpers for TUI load test results
#[derive(Debug, Clone, Copy, Default)]
pub struct TuiLoadAssertion;

impl TuiLoadAssertion {
    /// Assert that p95 frame time meets target FPS
    pub fn assert_meets_fps(metrics: &TuiFrameMetrics, target_fps: u32) {
        let budget_ms = 1000.0 / target_fps as f64;
        assert!(
            metrics.p95_frame_ms() <= budget_ms,
            "p95 frame time {:.2}ms exceeds {:.2}ms budget for {} FPS",
            metrics.p95_frame_ms(),
            budget_ms,
            target_fps
        );
    }

    /// Assert that no frame exceeded timeout
    pub fn assert_no_hang(result: &TuiLoadResult<TuiFrameMetrics>) {
        assert!(
            result.is_ok(),
            "TUI hang detected: {:?}",
            result.as_ref().err()
        );
    }

    /// Assert filter performance doesn't degrade with item count
    pub fn assert_filter_scales_linearly(
        results: &[(String, TuiFrameMetrics)],
        max_degradation_factor: f64,
    ) {
        if results.len() < 2 {
            return;
        }

        let baseline = results[0].1.avg_frame_ms();
        if baseline == 0.0 {
            return;
        }

        for (filter, metrics) in results.iter().skip(1) {
            let factor = metrics.avg_frame_ms() / baseline;
            assert!(
                factor <= max_degradation_factor,
                "Filter '{}' degraded by {:.1}x (max allowed: {:.1}x)",
                filter,
                factor,
                max_degradation_factor
            );
        }
    }
}

/// Integration load test that measures real application frame times.
///
/// Unlike synthetic load tests, this tests the ACTUAL application with
/// real collectors, real system calls, and real data. It catches issues
/// like blocking I/O, slow system calls, and expensive operations.
///
/// ## Example
///
/// ```ignore
/// use jugar_probar::tui_load::IntegrationLoadTest;
///
/// // Test that real app renders frames under 100ms
/// let test = IntegrationLoadTest::new()
///     .with_frame_budget_ms(100.0)
///     .with_timeout_ms(5000)
///     .with_frame_count(10);
///
/// let result = test.run(|| {
///     // Your real app initialization and render
///     let mut app = App::new();
///     app.collect_metrics();
///     // Simulate frame render...
/// });
///
/// assert!(result.is_ok(), "Real app should not hang");
/// ```
#[derive(Debug, Clone)]
pub struct IntegrationLoadTest {
    /// Frame budget in milliseconds
    frame_budget_ms: f64,
    /// Timeout for hang detection
    timeout_ms: u64,
    /// Number of frames to test
    frame_count: usize,
    /// Component-level timing thresholds (name -> max_ms)
    component_budgets: std::collections::HashMap<String, f64>,
}

impl IntegrationLoadTest {
    /// Create new integration load test
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_budget_ms: 100.0, // 10 FPS minimum
            timeout_ms: 5000,       // 5 second hang detection
            frame_count: 5,
            component_budgets: std::collections::HashMap::new(),
        }
    }

    /// Set frame budget
    #[must_use]
    pub fn with_frame_budget_ms(mut self, budget: f64) -> Self {
        self.frame_budget_ms = budget;
        self
    }

    /// Set timeout for hang detection
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Set number of frames to test
    #[must_use]
    pub fn with_frame_count(mut self, count: usize) -> Self {
        self.frame_count = count;
        self
    }

    /// Add component budget (e.g., "container_analyzer" -> 100ms max)
    #[must_use]
    pub fn with_component_budget(mut self, name: &str, max_ms: f64) -> Self {
        self.component_budgets.insert(name.to_string(), max_ms);
        self
    }

    /// Run integration test with a closure that performs one frame
    ///
    /// The closure should:
    /// 1. Initialize app (first call only)
    /// 2. Collect metrics
    /// 3. Render frame
    ///
    /// Returns timing for each frame.
    pub fn run<F>(&self, mut frame_fn: F) -> TuiLoadResult<TuiFrameMetrics>
    where
        F: FnMut() -> ComponentTimings,
    {
        let mut metrics = TuiFrameMetrics::new();
        let timeout = Duration::from_millis(self.timeout_ms);

        for frame in 0..self.frame_count {
            let start = Instant::now();

            let timings = frame_fn();

            let elapsed = start.elapsed();

            // Check for hang
            if elapsed > timeout {
                return Err(TuiLoadError::FrameTimeout {
                    frame,
                    timeout_ms: self.timeout_ms,
                    filter: format!("frame {}", frame),
                    item_count: 0,
                });
            }

            // Check component budgets
            for (name, &max_ms) in &self.component_budgets {
                if let Some(&actual_ms) = timings.0.get(name) {
                    if actual_ms > max_ms {
                        return Err(TuiLoadError::BudgetExceeded {
                            frame,
                            actual_ms,
                            budget_ms: max_ms,
                        });
                    }
                }
            }

            metrics.record(elapsed.as_micros() as u64);
        }

        Ok(metrics)
    }
}

impl Default for IntegrationLoadTest {
    fn default() -> Self {
        Self::new()
    }
}

/// Component timing results from a frame render
#[derive(Debug, Clone, Default)]
pub struct ComponentTimings(pub std::collections::HashMap<String, f64>);

impl ComponentTimings {
    /// Create empty timings
    #[must_use]
    pub fn new() -> Self {
        Self(std::collections::HashMap::new())
    }

    /// Record a component timing
    pub fn record(&mut self, name: &str, duration_ms: f64) {
        self.0.insert(name.to_string(), duration_ms);
    }

    /// Get timing for a component
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generator_creates_items() {
        let gen = DataGenerator::new(100);
        let items = gen.generate();
        assert_eq!(items.len(), 100);
    }

    #[test]
    fn test_data_generator_deterministic() {
        let gen1 = DataGenerator::new(50).with_seed(12345);
        let gen2 = DataGenerator::new(50).with_seed(12345);
        let items1 = gen1.generate();
        let items2 = gen2.generate();

        for (a, b) in items1.iter().zip(items2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.name, b.name);
        }
    }

    #[test]
    fn test_synthetic_item_filter_empty() {
        let item = SyntheticItem {
            id: 1,
            name: "test".to_string(),
            description: "desc".to_string(),
            value1: 0.0,
            value2: 0.0,
            state: "R".to_string(),
            owner: "root".to_string(),
            count: 1,
        };
        assert!(item.matches_filter(""));
    }

    #[test]
    fn test_synthetic_item_filter_name() {
        let item = SyntheticItem {
            id: 1,
            name: "systemd".to_string(),
            description: "init system".to_string(),
            value1: 0.0,
            value2: 0.0,
            state: "S".to_string(),
            owner: "root".to_string(),
            count: 1,
        };
        assert!(item.matches_filter("sys"));
        assert!(item.matches_filter("SYS")); // case insensitive
        assert!(!item.matches_filter("chrome"));
    }

    #[test]
    fn test_synthetic_item_filter_description() {
        let item = SyntheticItem {
            id: 1,
            name: "init".to_string(),
            description: "/usr/lib/systemd/systemd".to_string(),
            value1: 0.0,
            value2: 0.0,
            state: "S".to_string(),
            owner: "root".to_string(),
            count: 1,
        };
        assert!(item.matches_filter("systemd"));
    }

    #[test]
    fn test_frame_metrics_percentiles() {
        let mut metrics = TuiFrameMetrics::new();
        for i in 1..=100 {
            metrics.record(i * 1000); // 1-100ms
        }

        assert_eq!(metrics.frame_count, 100);
        assert!((metrics.p50_frame_ms() - 50.0).abs() < 2.0);
        assert!(metrics.p95_frame_ms() >= 95.0);
    }

    #[test]
    fn test_frame_metrics_meets_fps() {
        let mut metrics = TuiFrameMetrics::new();
        // All frames at 10ms = 100 FPS
        for _ in 0..100 {
            metrics.record(10_000); // 10ms in microseconds
        }

        assert!(metrics.meets_fps(60)); // Should meet 60 FPS
        assert!(metrics.meets_fps(100)); // Should meet 100 FPS exactly
        assert!(!metrics.meets_fps(120)); // Should NOT meet 120 FPS
    }

    #[test]
    fn test_tui_load_test_no_hang() {
        let test = TuiLoadTest::new()
            .with_item_count(100)
            .with_timeout_ms(1000);

        let result = test.run(|_items, _filter| {
            // Fast render - just return immediately
            Some(100) // 0.1ms
        });

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.frame_count > 0);
    }

    #[test]
    fn test_tui_load_test_detects_hang() {
        let test = TuiLoadTest::new()
            .with_item_count(10)
            .with_timeout_ms(50) // Very short timeout
            .with_frames_per_filter(1);

        let result = test.run(|_items, _filter| {
            // Simulate hang
            std::thread::sleep(Duration::from_millis(100));
            None
        });

        assert!(result.is_err());
        match result {
            Err(TuiLoadError::FrameTimeout { .. }) => {}
            _ => panic!("Expected FrameTimeout error"),
        }
    }

    #[test]
    fn test_tui_load_test_large_dataset() {
        let test = TuiLoadTest::new()
            .with_item_count(5000)
            .with_timeout_ms(5000)
            .with_frames_per_filter(3);

        // Simulate realistic filtering
        let result = test.run(|items, filter| {
            let filter_lower = filter.to_lowercase();
            let _filtered: Vec<_> = items
                .iter()
                .filter(|item| item.matches_filter_precomputed(&filter_lower))
                .collect();
            None // Let harness measure time
        });

        assert!(result.is_ok(), "Should handle 5000 items without hang");
        let metrics = result.unwrap();

        // Should complete reasonably fast (p95 under 100ms)
        assert!(
            metrics.p95_frame_ms() < 100.0,
            "p95 = {:.2}ms, should be < 100ms",
            metrics.p95_frame_ms()
        );
    }

    #[test]
    fn test_filter_stress_test() {
        let test = TuiLoadTest::new()
            .with_item_count(1000)
            .with_timeout_ms(2000)
            .with_frames_per_filter(5);

        let result = test.run_filter_stress(|items, filter| {
            let filter_lower = filter.to_lowercase();
            items
                .iter()
                .filter(|item| item.matches_filter_precomputed(&filter_lower))
                .cloned()
                .collect()
        });

        assert!(result.is_ok());
        let results = result.unwrap();

        // Check that all filters were tested
        assert!(!results.is_empty());

        // Check that performance doesn't degrade too much
        TuiLoadAssertion::assert_filter_scales_linearly(&results, 5.0);
    }

    #[test]
    fn test_tui_load_error_display() {
        let err = TuiLoadError::FrameTimeout {
            frame: 5,
            timeout_ms: 1000,
            filter: "test".to_string(),
            item_count: 5000,
        };
        let msg = err.to_string();
        assert!(msg.contains('5'));
        assert!(msg.contains("1000"));
        assert!(msg.contains("test"));
        assert!(msg.contains("5000"));
    }

    #[test]
    fn test_data_generator_with_long_descriptions() {
        let gen = DataGenerator::new(10).with_description_len(200);
        let items = gen.generate();

        // All descriptions should be at least close to target length
        for item in &items {
            assert!(
                item.description.len() >= 100,
                "Description too short: {}",
                item.description.len()
            );
        }
    }

    #[test]
    fn test_data_generator_default() {
        let gen = DataGenerator::default();
        let items = gen.generate();
        assert_eq!(items.len(), 1000);
    }

    #[test]
    fn test_integration_load_test_run_success() {
        let test = IntegrationLoadTest::new()
            .with_frame_budget_ms(500.0)
            .with_timeout_ms(2000)
            .with_frame_count(3);

        let mut call_count = 0;
        let result = test.run(|| {
            call_count += 1;
            let mut timings = ComponentTimings::new();
            timings.record("render", 1.0);
            timings
        });

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.frame_count, 3);
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_integration_load_test_default() {
        let test = IntegrationLoadTest::default();
        let result = test.run(|| ComponentTimings::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_integration_load_test_budget_exceeded() {
        let test = IntegrationLoadTest::new()
            .with_frame_count(2)
            .with_timeout_ms(5000)
            .with_component_budget("slow_component", 0.001); // impossibly tight

        let result = test.run(|| {
            let mut timings = ComponentTimings::new();
            timings.record("slow_component", 10.0); // 10ms, way over 0.001ms
            timings
        });

        assert!(result.is_err());
        match result {
            Err(TuiLoadError::BudgetExceeded {
                actual_ms,
                budget_ms,
                ..
            }) => {
                assert!((actual_ms - 10.0).abs() < f64::EPSILON);
                assert!((budget_ms - 0.001).abs() < f64::EPSILON);
            }
            _ => panic!("Expected BudgetExceeded error"),
        }
    }

    #[test]
    fn test_integration_load_test_component_within_budget() {
        let test = IntegrationLoadTest::new()
            .with_frame_count(2)
            .with_timeout_ms(5000)
            .with_component_budget("fast", 100.0);

        let result = test.run(|| {
            let mut timings = ComponentTimings::new();
            timings.record("fast", 1.0);
            timings
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_component_timings() {
        let mut t = ComponentTimings::new();
        assert!(t.get("render").is_none());

        t.record("render", 5.5);
        assert!((t.get("render").unwrap() - 5.5).abs() < f64::EPSILON);

        t.record("layout", 2.0);
        assert!((t.get("layout").unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tui_load_config_builder() {
        let config = TuiLoadConfig {
            item_count: 500,
            frame_budget_ms: 32.0,
            timeout_ms: 3000,
            frames_per_filter: 10,
            filters: vec!["sys".to_string(), "usr".to_string()],
            strict_budget: true,
        };
        assert_eq!(config.item_count, 500);
        assert!((config.frame_budget_ms - 32.0).abs() < f64::EPSILON);
        assert_eq!(config.timeout_ms, 3000);
        assert_eq!(config.frames_per_filter, 10);
        assert_eq!(config.filters.len(), 2);
        assert!(config.strict_budget);
    }

    #[test]
    fn test_tui_load_error_budget_exceeded_display() {
        let err = TuiLoadError::BudgetExceeded {
            frame: 3,
            actual_ms: 150.0,
            budget_ms: 16.6,
        };
        let msg = err.to_string();
        assert!(msg.contains("150"));
        assert!(msg.contains("16.6"));
    }

    #[test]
    fn test_frame_metrics_empty() {
        let metrics = TuiFrameMetrics::new();
        assert_eq!(metrics.frame_count, 0);
        assert_eq!(metrics.p50_frame_ms(), 0.0);
        assert_eq!(metrics.p95_frame_ms(), 0.0);
    }
}
