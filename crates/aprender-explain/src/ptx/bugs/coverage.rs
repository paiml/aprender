//! PTX coverage tracking for test completeness.

/// PTX coverage tracker for test completeness
pub struct PtxCoverageTracker {
    features: Vec<PtxFeature>,
}

/// A PTX feature that can be tracked for coverage
#[derive(Debug, Clone)]
pub struct PtxFeature {
    /// Feature name
    pub name: String,
    /// Whether this feature has been covered
    pub covered: bool,
    /// How many times this feature was seen
    pub hit_count: usize,
}

/// Coverage report summary
#[derive(Debug, Clone)]
pub struct PtxCoverageReport {
    /// Total features tracked
    pub total_features: usize,
    /// Features that were covered
    pub covered_features: usize,
    /// Coverage percentage (0.0 - 1.0)
    pub coverage: f64,
    /// Details per feature
    pub features: Vec<PtxFeature>,
}

impl PtxCoverageTracker {
    /// Create a new coverage tracker builder
    #[must_use]
    pub fn builder() -> PtxCoverageTrackerBuilder {
        PtxCoverageTrackerBuilder {
            features: Vec::new(),
        }
    }

    /// Analyze PTX code and update coverage
    pub fn analyze(&mut self, ptx: &str) {
        for feature in &mut self.features {
            let covered = match feature.name.as_str() {
                "barrier_sync" => ptx.contains("bar.sync"),
                "shared_memory" => {
                    ptx.contains(".shared")
                        || ptx.contains("st.shared")
                        || ptx.contains("ld.shared")
                }
                "global_memory" => ptx.contains("ld.global") || ptx.contains("st.global"),
                "register_allocation" => ptx.contains(".reg"),
                "loop_patterns" => {
                    ptx.contains("bra") && (ptx.contains("_loop") || ptx.contains("loop_"))
                }
                "control_flow" => {
                    ptx.contains("@%p") || ptx.contains("bra") || ptx.contains("setp")
                }
                "local_memory" => ptx.contains(".local"),
                "entry_point" => ptx.contains(".entry"),
                "predicates" => ptx.contains(".pred") || ptx.contains("@%p"),
                "fma_ops" => ptx.contains("fma.") || ptx.contains("mad."),
                _ => false,
            };

            if covered {
                feature.covered = true;
                feature.hit_count += 1;
            }
        }
    }

    /// Generate coverage report
    #[must_use]
    pub fn generate_report(&self) -> PtxCoverageReport {
        let total = self.features.len();
        let covered = self.features.iter().filter(|f| f.covered).count();
        let coverage = if total > 0 {
            covered as f64 / total as f64
        } else {
            1.0
        };

        PtxCoverageReport {
            total_features: total,
            covered_features: covered,
            coverage,
            features: self.features.clone(),
        }
    }
}

impl Default for PtxCoverageTracker {
    fn default() -> Self {
        PtxCoverageTrackerBuilder::new()
            .feature("barrier_sync")
            .feature("shared_memory")
            .feature("global_memory")
            .feature("register_allocation")
            .feature("loop_patterns")
            .feature("control_flow")
            .build()
    }
}

/// Builder for `PtxCoverageTracker`
#[derive(Debug)]
pub struct PtxCoverageTrackerBuilder {
    features: Vec<PtxFeature>,
}

impl PtxCoverageTrackerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a feature to track
    #[must_use]
    pub fn feature(mut self, name: &str) -> Self {
        self.features.push(PtxFeature {
            name: name.to_string(),
            covered: false,
            hit_count: 0,
        });
        self
    }

    /// Build the coverage tracker
    #[must_use]
    pub fn build(self) -> PtxCoverageTracker {
        PtxCoverageTracker {
            features: self.features,
        }
    }
}

impl Default for PtxCoverageTrackerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
