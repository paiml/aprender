
impl ErrorPattern {
    /// Create new error pattern
    #[must_use]
    pub fn new(id: impl Into<String>, keywords: Vec<String>, fix: FixAction) -> Self {
        Self {
            id: id.into(),
            keywords,
            fix,
            applications: 0,
            successes: 0,
            source: PatternSource::Manual,
        }
    }

    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        if self.applications == 0 {
            0.0
        } else {
            self.successes as f32 / self.applications as f32
        }
    }

    /// Check if pattern should be retired
    ///
    /// Per spec: "Patterns with <30% success rate after 5 applications
    /// are automatically retired."
    #[must_use]
    pub fn should_retire(&self) -> bool {
        self.applications >= 5 && self.success_rate() < 0.30
    }

    /// Record application of this pattern
    pub fn record_application(&mut self, success: bool) {
        self.applications += 1;
        if success {
            self.successes += 1;
        }
    }

    /// Check if error message matches this pattern
    #[must_use]
    pub fn matches(&self, error_message: &str) -> bool {
        let lower = error_message.to_lowercase();
        self.keywords
            .iter()
            .any(|k| lower.contains(&k.to_lowercase()))
    }

    /// Calculate match confidence (0.0 - 1.0)
    #[must_use]
    pub fn match_confidence(&self, error_message: &str) -> f32 {
        let lower = error_message.to_lowercase();
        let matches = self
            .keywords
            .iter()
            .filter(|k| lower.contains(&k.to_lowercase()))
            .count();
        (matches as f32 / self.keywords.len() as f32).min(1.0)
    }
}

/// Error pattern library with hybrid retrieval
#[derive(Debug, Clone, Default)]
pub struct ErrorPatternLibrary {
    patterns: Vec<ErrorPattern>,
    /// Hit rate: matches / queries
    queries: usize,
    matches: usize,
}

impl ErrorPatternLibrary {
    /// Create new pattern library
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bootstrap with common conversion error patterns
    #[must_use]
    pub fn bootstrap() -> Self {
        let mut lib = Self::new();

        // Column-major ghost pattern
        lib.add_pattern(ErrorPattern {
            id: "COL_MAJOR_GHOST".into(),
            keywords: vec![
                "dimension".into(),
                "mismatch".into(),
                "transpos".into(),
                "layout".into(),
                "column".into(),
                "row".into(),
            ],
            fix: FixAction::SwapDimensions,
            applications: 0,
            successes: 0,
            source: PatternSource::Bootstrap,
        });

        // Quantization artifact pattern
        lib.add_pattern(ErrorPattern {
            id: "QUANT_ARTIFACT".into(),
            keywords: vec![
                "quantiz".into(),
                "precision".into(),
                "overflow".into(),
                "underflow".into(),
                "block".into(),
            ],
            fix: FixAction::Requantize { block_size: 32 },
            applications: 0,
            successes: 0,
            source: PatternSource::Bootstrap,
        });

        // Checksum failure pattern
        lib.add_pattern(ErrorPattern {
            id: "CHECKSUM_FAIL".into(),
            keywords: vec![
                "checksum".into(),
                "crc".into(),
                "integrity".into(),
                "corrupt".into(),
            ],
            fix: FixAction::RecomputeChecksum,
            applications: 0,
            successes: 0,
            source: PatternSource::Bootstrap,
        });

        // Alignment pattern
        lib.add_pattern(ErrorPattern {
            id: "ALIGNMENT_ERR".into(),
            keywords: vec![
                "align".into(),
                "padding".into(),
                "offset".into(),
                "boundary".into(),
            ],
            fix: FixAction::PadAlignment { alignment: 64 },
            applications: 0,
            successes: 0,
            source: PatternSource::Bootstrap,
        });

        lib
    }

    /// Add pattern to library
    pub fn add_pattern(&mut self, pattern: ErrorPattern) {
        self.patterns.push(pattern);
    }

    /// Find best matching pattern for error message
    #[must_use]
    pub fn find_match(&mut self, error_message: &str) -> Option<&ErrorPattern> {
        self.queries += 1;

        let best = self
            .patterns
            .iter()
            .filter(|p| p.matches(error_message))
            .max_by(|a, b| {
                let conf_a = a.match_confidence(error_message) * a.success_rate();
                let conf_b = b.match_confidence(error_message) * b.success_rate();
                conf_a
                    .partial_cmp(&conf_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if best.is_some() {
            self.matches += 1;
        }

        best
    }

    /// Get hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        if self.queries == 0 {
            0.0
        } else {
            self.matches as f32 / self.queries as f32
        }
    }

    /// Retire low-performing patterns
    pub fn retire_failing_patterns(&mut self) {
        self.patterns.retain(|p| !p.should_retire());
    }

    /// Record pattern application result
    pub fn record_result(&mut self, pattern_id: &str, success: bool) {
        if let Some(pattern) = self.patterns.iter_mut().find(|p| p.id == pattern_id) {
            pattern.record_application(success);
        }
    }
}

// ============================================================================
// Hansei Reflection System (Toyota Way)
// ============================================================================

/// Trend direction for conversion quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Quality is improving
    Improving,
    /// Quality is degrading
    Degrading,
    /// Quality is stable
    Stable,
    /// Quality is oscillating
    Oscillating,
}

/// Conversion category for Pareto analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionCategory {
    /// GGUF to APR
    GgufToApr,
    /// APR to GGUF
    AprToGguf,
    /// SafeTensors to APR
    SafeTensorsToApr,
    /// APR to SafeTensors
    AprToSafeTensors,
    /// GGUF to SafeTensors
    GgufToSafeTensors,
    /// SafeTensors to GGUF
    SafeTensorsToGguf,
}

impl fmt::Display for ConversionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GgufToApr => write!(f, "GGUF→APR"),
            Self::AprToGguf => write!(f, "APR→GGUF"),
            Self::SafeTensorsToApr => write!(f, "SafeTensors→APR"),
            Self::AprToSafeTensors => write!(f, "APR→SafeTensors"),
            Self::GgufToSafeTensors => write!(f, "GGUF→SafeTensors"),
            Self::SafeTensorsToGguf => write!(f, "SafeTensors→GGUF"),
        }
    }
}

/// Issue severity for Hansei report
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Conversion issue identified in Hansei analysis
#[derive(Debug, Clone)]
pub struct ConversionIssue {
    /// Issue severity
    pub severity: Severity,
    /// Category where issue occurred
    pub category: ConversionCategory,
    /// Description of issue
    pub description: String,
    /// Suggested remediation
    pub remediation: String,
    /// Tarantula suspiciousness score
    pub suspiciousness: f32,
}

/// Category summary with Tarantula suspiciousness
#[derive(Debug, Clone)]
pub struct CategorySummary {
    /// Category
    pub category: ConversionCategory,
    /// Total attempts
    pub attempts: usize,
    /// Successful conversions
    pub successes: usize,
    /// Success rate
    pub success_rate: f32,
    /// Tarantula suspiciousness score
    pub suspiciousness: f32,
    /// Trend
    pub trend: Trend,
    /// Failure share (% of total failures)
    pub failure_share: f32,
}

/// Hansei (reflection) report for conversion batch
///
/// Implements Toyota Way principle of systematic reflection.
#[derive(Debug, Clone)]
pub struct HanseiReport {
    /// Total conversions attempted
    pub total_attempts: usize,
    /// Successful conversions
    pub successes: usize,
    /// Overall success rate
    pub success_rate: f32,
    /// Category-wise breakdown
    pub category_summaries: HashMap<ConversionCategory, CategorySummary>,
    /// Pareto categories (20% causing 80% of failures)
    pub pareto_categories: Vec<ConversionCategory>,
    /// Overall trend
    pub trend: Trend,
    /// Actionable issues sorted by priority
    pub issues: Vec<ConversionIssue>,
    /// Wilson confidence interval on success rate
    pub confidence_interval: WilsonScore,
}

impl HanseiReport {
    /// Create Hansei report from conversion results
    #[must_use]
    pub fn from_results(results: &[(ConversionCategory, bool)]) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let total_attempts = results.len();
        let successes = results.iter().filter(|(_, success)| *success).count();
        let success_rate = successes as f32 / total_attempts as f32;

        // Category-wise breakdown
        let mut cat_stats: HashMap<ConversionCategory, (usize, usize)> = HashMap::new();
        for (cat, success) in results {
            let entry = cat_stats.entry(*cat).or_insert((0, 0));
            entry.0 += 1; // attempts
            if *success {
                entry.1 += 1; // successes
            }
        }

        let total_failures = total_attempts - successes;
        let mut category_summaries = HashMap::new();
        for (cat, (attempts, cat_successes)) in &cat_stats {
            let cat_failures = attempts - cat_successes;
            let failure_share = if total_failures > 0 {
                cat_failures as f32 / total_failures as f32
            } else {
                0.0
            };

            category_summaries.insert(
                *cat,
                CategorySummary {
                    category: *cat,
                    attempts: *attempts,
                    successes: *cat_successes,
                    success_rate: *cat_successes as f32 / *attempts as f32,
                    suspiciousness: 0.0,  // Computed separately with Tarantula
                    trend: Trend::Stable, // Would need historical data
                    failure_share,
                },
            );
        }

        // Pareto analysis
        let pareto_categories = compute_pareto(&cat_stats, total_failures);

        // Confidence interval
        let confidence_interval = WilsonScore::calculate(successes, total_attempts, 0.95);

        Self {
            total_attempts,
            successes,
            success_rate,
            category_summaries,
            pareto_categories,
            trend: Trend::Stable,
            issues: Vec::new(),
            confidence_interval,
        }
    }

    /// Create empty report
    #[must_use]
    pub fn empty() -> Self {
        Self {
            total_attempts: 0,
            successes: 0,
            success_rate: 0.0,
            category_summaries: HashMap::new(),
            pareto_categories: Vec::new(),
            trend: Trend::Stable,
            issues: Vec::new(),
            confidence_interval: WilsonScore::calculate(0, 0, 0.95),
        }
    }

    /// Get Andon level for overall success rate
    #[must_use]
    pub fn andon_level(&self, target: f32) -> AndonLevel {
        self.confidence_interval.andon_level(target)
    }
}

/// Compute Pareto categories (20% causing 80% of failures)
fn compute_pareto(
    cat_stats: &HashMap<ConversionCategory, (usize, usize)>,
    total_failures: usize,
) -> Vec<ConversionCategory> {
    if total_failures == 0 {
        return Vec::new();
    }

    let mut failures: Vec<_> = cat_stats
        .iter()
        .map(|(cat, (attempts, successes))| (*cat, attempts - successes))
        .collect();

    failures.sort_by(|a, b| b.1.cmp(&a.1));

    let threshold = (total_failures as f32 * 0.80) as usize;
    let mut cumulative = 0;
    let mut pareto = Vec::new();

    for (cat, count) in failures {
        pareto.push(cat);
        cumulative += count;
        if cumulative >= threshold {
            break;
        }
    }

    pareto
}
