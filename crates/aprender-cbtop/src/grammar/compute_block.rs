//! ComputeBlock orchestrator (GGPlot equivalent in Grammar of Graphics).

use std::collections::HashMap;
use std::time::Duration;

use super::composition::CompositionMode;
use super::context::ExecutionContext;
use super::error::{GrammarError, GrammarResult};
use super::policy::ExecutionPolicy;
use super::resources::ResourceMapping;
use super::strategy::{ExecutionStrategy, StrategyLayer};
use super::transform::DataTransform;
use super::workload::WorkloadSpec;

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution time
    pub duration: Duration,
    /// GFLOP/s achieved
    pub gflops: f64,
    /// Memory bandwidth achieved (GB/s)
    pub bandwidth_gbps: f64,
    /// Strategy that was used
    pub strategy_used: String,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Validated ComputeBlock ready for execution
#[derive(Debug, Clone)]
pub struct BuiltComputeBlock {
    pub(crate) inner: ComputeBlock,
}

impl BuiltComputeBlock {
    /// Execute the compute block
    pub fn execute(&self) -> GrammarResult<ExecutionResult> {
        let start = std::time::Instant::now();

        // Select strategy (in order of priority)
        let mut strategies = self.inner.strategies.clone();
        strategies.sort_by(|a, b| b.priority.cmp(&a.priority));

        let strategy_used = if let Some(layer) = strategies.first() {
            format!("{:?}", layer.strategy)
        } else {
            "Sequential".to_string()
        };

        // Simulate execution (real implementation would dispatch to backends)
        let duration = start.elapsed();
        let flops = self
            .inner
            .workload
            .as_ref()
            .map(|w| w.flop_count())
            .unwrap_or(0);
        let gflops = if duration.as_secs_f64() > 0.0 {
            flops as f64 / duration.as_secs_f64() / 1e9
        } else {
            0.0
        };

        Ok(ExecutionResult {
            duration,
            gflops,
            bandwidth_gbps: 0.0,
            strategy_used,
            metrics: HashMap::new(),
        })
    }

    /// Get the workload spec
    pub fn workload(&self) -> Option<&WorkloadSpec> {
        self.inner.workload.as_ref()
    }
}

/// ComputeBlock - the main orchestrator (analogous to GGPlot)
#[derive(Debug, Clone, Default)]
pub struct ComputeBlock {
    /// Workload specification
    pub(crate) workload: Option<WorkloadSpec>,
    /// Resource mapping
    pub(crate) resources: ResourceMapping,
    /// Strategy layers (multiple, with priority)
    pub(crate) strategies: Vec<StrategyLayer>,
    /// Data transform
    pub(crate) transform: DataTransform,
    /// Execution context
    pub(crate) context: ExecutionContext,
    /// Composition mode
    pub(crate) composition: CompositionMode,
    /// Execution policy
    pub(crate) policy: ExecutionPolicy,
    /// Facet parameters for parameter sweep
    pub(crate) facet_params: Option<(String, Vec<f64>)>,
}

impl ComputeBlock {
    /// Create a new ComputeBlock builder
    pub fn builder() -> ComputeBlockBuilder {
        ComputeBlockBuilder::new()
    }

    /// Validate the ComputeBlock configuration
    fn validate(&self) -> GrammarResult<()> {
        // F701: Builder rejects incomplete spec
        if self.workload.is_none() {
            return Err(GrammarError::MissingWorkload);
        }

        // F711: Scale domain validation
        // (handled in ResourceScale)

        Ok(())
    }

    /// Build and validate the ComputeBlock
    pub fn build(self) -> GrammarResult<BuiltComputeBlock> {
        self.validate()?;
        Ok(BuiltComputeBlock { inner: self })
    }
}

/// Builder for ComputeBlock (fluent API)
#[derive(Debug, Clone, Default)]
pub struct ComputeBlockBuilder {
    inner: ComputeBlock,
}

impl ComputeBlockBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            inner: ComputeBlock {
                transform: DataTransform::Identity,
                context: ExecutionContext::Cpu {
                    affinity: None,
                    numa_node: None,
                },
                composition: CompositionMode::None,
                policy: ExecutionPolicy::default(),
                ..Default::default()
            },
        }
    }

    /// Set workload specification
    pub fn workload(mut self, workload: WorkloadSpec) -> Self {
        self.inner.workload = Some(workload);
        self
    }

    /// Set resource mapping
    pub fn resources(mut self, resources: ResourceMapping) -> Self {
        self.inner.resources = resources;
        self
    }

    /// Add a strategy layer
    pub fn strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.inner.strategies.push(StrategyLayer::new(strategy));
        self
    }

    /// Add a strategy layer with priority
    pub fn strategy_with_priority(mut self, strategy: ExecutionStrategy, priority: i32) -> Self {
        self.inner
            .strategies
            .push(StrategyLayer::new(strategy).priority(priority));
        self
    }

    /// Set data transform
    pub fn transform(mut self, transform: DataTransform) -> Self {
        self.inner.transform = transform;
        self
    }

    /// Set execution context
    pub fn context(mut self, context: ExecutionContext) -> Self {
        self.inner.context = context;
        self
    }

    /// Set composition mode
    pub fn composition(mut self, composition: CompositionMode) -> Self {
        self.inner.composition = composition;
        self
    }

    /// Set execution policy
    pub fn policy(mut self, policy: ExecutionPolicy) -> Self {
        self.inner.policy = policy;
        self
    }

    /// Set facet parameters for parameter sweep
    pub fn facet_by(mut self, param: impl Into<String>, values: Vec<f64>) -> Self {
        self.inner.facet_params = Some((param.into(), values));
        self
    }

    /// Build and validate
    pub fn build(self) -> GrammarResult<BuiltComputeBlock> {
        self.inner.build()
    }
}
