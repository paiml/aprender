//! Backend Selection and Privacy Tiers
//!
//! Implements Toyota Way "Poka-Yoke" (Mistake Proofing) with privacy gates.
//!
//! ## Privacy Tiers
//!
//! - `Sovereign` - Local only, blocks all external API calls
//! - `Private` - Dedicated/VPC endpoints only
//! - `Standard` - Public APIs acceptable

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// SERVE-BKD-001: Backend Types
// ============================================================================

/// Supported serving backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServingBackend {
    // Local backends
    Realizar,
    Ollama,
    LlamaCpp,
    Llamafile,
    Candle,
    Vllm,
    Tgi,
    LocalAI,

    // Remote backends
    HuggingFace,
    Together,
    Replicate,
    Anyscale,
    Modal,
    Fireworks,
    Groq,
    OpenAI,
    Anthropic,
    AzureOpenAI,
    AwsBedrock,
    GoogleVertex,

    // Serverless backends
    AwsLambda,
    CloudflareWorkers,
}

impl ServingBackend {
    /// Check if this is a local backend (no network calls)
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(
            self,
            Self::Realizar
                | Self::Ollama
                | Self::LlamaCpp
                | Self::Llamafile
                | Self::Candle
                | Self::Vllm
                | Self::Tgi
                | Self::LocalAI
        )
    }

    /// Check if this is a remote/cloud backend
    #[must_use]
    pub const fn is_remote(&self) -> bool {
        !self.is_local()
    }

    /// Get the API endpoint host for remote backends
    #[must_use]
    pub const fn api_host(&self) -> Option<&'static str> {
        match self {
            Self::HuggingFace => Some("api-inference.huggingface.co"),
            Self::Together => Some("api.together.xyz"),
            Self::Replicate => Some("api.replicate.com"),
            Self::Anyscale => Some("api.anyscale.com"),
            Self::Modal => Some("api.modal.com"),
            Self::Fireworks => Some("api.fireworks.ai"),
            Self::Groq => Some("api.groq.com"),
            Self::OpenAI => Some("api.openai.com"),
            Self::Anthropic => Some("api.anthropic.com"),
            Self::AzureOpenAI => Some("openai.azure.com"),
            Self::AwsBedrock => Some("bedrock-runtime.amazonaws.com"),
            Self::GoogleVertex => Some("aiplatform.googleapis.com"),
            Self::AwsLambda => Some("lambda.amazonaws.com"),
            Self::CloudflareWorkers => Some("workers.cloudflare.com"),
            _ => None,
        }
    }

    /// Check if this is a serverless backend
    #[must_use]
    pub const fn is_serverless(&self) -> bool {
        matches!(self, Self::AwsLambda | Self::CloudflareWorkers | Self::Modal)
    }
}

// ============================================================================
// SERVE-BKD-002: Privacy Tiers
// ============================================================================

/// Privacy tier for backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PrivacyTier {
    /// Local only - blocks ALL external API calls (Poka-Yoke)
    Sovereign,
    /// Dedicated/VPC endpoints only
    Private,
    /// Public APIs acceptable
    #[default]
    Standard,
}

impl PrivacyTier {
    /// Check if a backend is allowed under this privacy tier
    #[must_use]
    pub fn allows(&self, backend: ServingBackend) -> bool {
        match self {
            Self::Sovereign => backend.is_local(),
            Self::Private => {
                backend.is_local()
                    || matches!(
                        backend,
                        ServingBackend::AzureOpenAI
                            | ServingBackend::AwsBedrock
                            | ServingBackend::GoogleVertex
                            | ServingBackend::AwsLambda // Your own Lambda functions
                    )
            }
            Self::Standard => true,
        }
    }

    /// Get all blocked API hosts for this tier (for network egress locking)
    #[must_use]
    pub fn blocked_hosts(&self) -> Vec<&'static str> {
        match self {
            Self::Sovereign => {
                // Block ALL remote API hosts
                vec![
                    "api-inference.huggingface.co",
                    "api.together.xyz",
                    "api.replicate.com",
                    "api.anyscale.com",
                    "api.modal.com",
                    "api.fireworks.ai",
                    "api.groq.com",
                    "api.openai.com",
                    "api.anthropic.com",
                    "openai.azure.com",
                    "bedrock-runtime.amazonaws.com",
                    "aiplatform.googleapis.com",
                    "lambda.amazonaws.com",
                    "workers.cloudflare.com",
                ]
            }
            Self::Private => {
                // Block public APIs but allow enterprise/owned endpoints
                vec![
                    "api-inference.huggingface.co",
                    "api.together.xyz",
                    "api.replicate.com",
                    "api.anyscale.com",
                    "api.modal.com",
                    "api.fireworks.ai",
                    "api.groq.com",
                    "api.openai.com",
                    "api.anthropic.com",
                    "workers.cloudflare.com", // Cloudflare not in your control
                ]
            }
            Self::Standard => vec![],
        }
    }
}

// ============================================================================
// SERVE-BKD-003: Latency and Throughput Tiers
// ============================================================================

/// Latency requirement tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LatencyTier {
    /// <100ms - requires local GPU or Groq
    RealTime,
    /// <1s - local or fast remote
    #[default]
    Interactive,
    /// >1s acceptable - any backend
    Batch,
}

/// Throughput requirement tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThroughputTier {
    /// Low volume (<10 req/s)
    #[default]
    Low,
    /// Medium volume (10-100 req/s)
    Medium,
    /// High volume (>100 req/s)
    High,
}

/// Cost sensitivity tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CostTier {
    /// Minimize cost at all costs
    Frugal,
    /// Balance cost and performance
    #[default]
    Balanced,
    /// Performance over cost
    Premium,
}

// ============================================================================
// SERVE-BKD-004: Backend Selector
// ============================================================================

/// Backend selection configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendSelector {
    pub privacy: PrivacyTier,
    pub latency: LatencyTier,
    pub throughput: ThroughputTier,
    pub cost: CostTier,
    /// Explicitly disabled backends
    pub disabled: HashSet<ServingBackend>,
    /// Preferred backends (tried first)
    pub preferred: Vec<ServingBackend>,
}

impl BackendSelector {
    /// Create a new backend selector with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set privacy tier
    #[must_use]
    pub fn with_privacy(mut self, tier: PrivacyTier) -> Self {
        self.privacy = tier;
        self
    }

    /// Set latency tier
    #[must_use]
    pub fn with_latency(mut self, tier: LatencyTier) -> Self {
        self.latency = tier;
        self
    }

    /// Set throughput tier
    #[must_use]
    pub fn with_throughput(mut self, tier: ThroughputTier) -> Self {
        self.throughput = tier;
        self
    }

    /// Set cost tier
    #[must_use]
    pub fn with_cost(mut self, tier: CostTier) -> Self {
        self.cost = tier;
        self
    }

    /// Disable a specific backend
    #[must_use]
    pub fn disable(mut self, backend: ServingBackend) -> Self {
        self.disabled.insert(backend);
        self
    }

    /// Add a preferred backend
    #[must_use]
    pub fn prefer(mut self, backend: ServingBackend) -> Self {
        self.preferred.push(backend);
        self
    }

    /// Recommend backends based on requirements
    #[must_use]
    pub fn recommend(&self) -> Vec<ServingBackend> {
        let mut candidates: Vec<ServingBackend> = Vec::new();

        // Start with preferred backends
        for backend in &self.preferred {
            if self.is_valid(*backend) {
                candidates.push(*backend);
            }
        }

        // Add tier-appropriate backends
        let tier_backends = self.get_tier_backends();
        for backend in tier_backends {
            if self.is_valid(backend) && !candidates.contains(&backend) {
                candidates.push(backend);
            }
        }

        candidates
    }

    /// Check if a backend is valid for current configuration
    #[must_use]
    pub fn is_valid(&self, backend: ServingBackend) -> bool {
        !self.disabled.contains(&backend) && self.privacy.allows(backend)
    }

    /// Validate a request against privacy tier (Poka-Yoke gate)
    ///
    /// Returns `Err` if the backend would violate privacy constraints.
    pub fn validate(&self, backend: ServingBackend) -> Result<(), PrivacyViolation> {
        if self.disabled.contains(&backend) {
            return Err(PrivacyViolation::BackendDisabled(backend));
        }

        if !self.privacy.allows(backend) {
            return Err(PrivacyViolation::TierViolation { backend, tier: self.privacy });
        }

        Ok(())
    }

    fn get_tier_backends(&self) -> Vec<ServingBackend> {
        match (self.latency, self.privacy) {
            // Real-time + Sovereign: only local with GPU
            (LatencyTier::RealTime, PrivacyTier::Sovereign) => {
                vec![ServingBackend::Realizar, ServingBackend::LlamaCpp]
            }
            // Real-time + any: Groq is fastest, then local
            (LatencyTier::RealTime, _) => {
                vec![ServingBackend::Groq, ServingBackend::Realizar, ServingBackend::Fireworks]
            }
            // Interactive + Sovereign: local options
            (LatencyTier::Interactive, PrivacyTier::Sovereign) => {
                vec![ServingBackend::Realizar, ServingBackend::Ollama, ServingBackend::LlamaCpp]
            }
            // Interactive + Private: local + enterprise + Lambda
            (LatencyTier::Interactive, PrivacyTier::Private) => {
                vec![
                    ServingBackend::Realizar,
                    ServingBackend::Ollama,
                    ServingBackend::AzureOpenAI,
                    ServingBackend::AwsBedrock,
                    ServingBackend::AwsLambda,
                ]
            }
            // Interactive + Standard: mix of fast options
            (LatencyTier::Interactive, PrivacyTier::Standard) => {
                vec![
                    ServingBackend::Realizar,
                    ServingBackend::Groq,
                    ServingBackend::Together,
                    ServingBackend::Fireworks,
                ]
            }
            // Batch + Sovereign: local only
            (LatencyTier::Batch, PrivacyTier::Sovereign) => {
                vec![ServingBackend::Realizar, ServingBackend::Ollama]
            }
            // Batch + Private: Lambda is excellent for batch (pay per use)
            (LatencyTier::Batch, PrivacyTier::Private) => {
                vec![
                    ServingBackend::AwsLambda,
                    ServingBackend::Realizar,
                    ServingBackend::AwsBedrock,
                ]
            }
            // Batch + Standard: prioritize cost
            (LatencyTier::Batch, PrivacyTier::Standard) => {
                vec![
                    ServingBackend::AwsLambda,
                    ServingBackend::Together,
                    ServingBackend::HuggingFace,
                    ServingBackend::Replicate,
                ]
            }
        }
    }
}

/// Privacy violation error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrivacyViolation {
    /// Backend is explicitly disabled
    BackendDisabled(ServingBackend),
    /// Backend violates privacy tier
    TierViolation { backend: ServingBackend, tier: PrivacyTier },
}

impl std::fmt::Display for PrivacyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendDisabled(b) => write!(f, "Backend {:?} is disabled", b),
            Self::TierViolation { backend, tier } => {
                write!(f, "Backend {:?} violates {:?} privacy tier", backend, tier)
            }
        }
    }
}

impl std::error::Error for PrivacyViolation {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // SERVE-BKD-001: Backend Type Tests
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_001_local_backends() {
        assert!(ServingBackend::Realizar.is_local());
        assert!(ServingBackend::Ollama.is_local());
        assert!(ServingBackend::LlamaCpp.is_local());
        assert!(ServingBackend::Llamafile.is_local());
        assert!(ServingBackend::Candle.is_local());
        assert!(ServingBackend::Vllm.is_local());
        assert!(ServingBackend::Tgi.is_local());
        assert!(ServingBackend::LocalAI.is_local());
    }

    #[test]
    fn test_SERVE_BKD_001_remote_backends() {
        assert!(ServingBackend::OpenAI.is_remote());
        assert!(ServingBackend::Anthropic.is_remote());
        assert!(ServingBackend::Together.is_remote());
        assert!(ServingBackend::Groq.is_remote());
        assert!(ServingBackend::HuggingFace.is_remote());
    }

    #[test]
    fn test_SERVE_BKD_001_api_hosts() {
        assert_eq!(ServingBackend::OpenAI.api_host(), Some("api.openai.com"));
        assert_eq!(ServingBackend::Anthropic.api_host(), Some("api.anthropic.com"));
        assert_eq!(ServingBackend::Realizar.api_host(), None);
    }

    // ========================================================================
    // SERVE-BKD-002: Privacy Tier Tests
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_002_sovereign_blocks_remote() {
        let tier = PrivacyTier::Sovereign;
        assert!(tier.allows(ServingBackend::Realizar));
        assert!(tier.allows(ServingBackend::Ollama));
        assert!(!tier.allows(ServingBackend::OpenAI));
        assert!(!tier.allows(ServingBackend::Anthropic));
        assert!(!tier.allows(ServingBackend::AzureOpenAI));
    }

    #[test]
    fn test_SERVE_BKD_002_private_allows_enterprise() {
        let tier = PrivacyTier::Private;
        assert!(tier.allows(ServingBackend::Realizar));
        assert!(tier.allows(ServingBackend::AzureOpenAI));
        assert!(tier.allows(ServingBackend::AwsBedrock));
        assert!(tier.allows(ServingBackend::GoogleVertex));
        assert!(!tier.allows(ServingBackend::OpenAI));
        assert!(!tier.allows(ServingBackend::Together));
    }

    #[test]
    fn test_SERVE_BKD_002_standard_allows_all() {
        let tier = PrivacyTier::Standard;
        assert!(tier.allows(ServingBackend::Realizar));
        assert!(tier.allows(ServingBackend::OpenAI));
        assert!(tier.allows(ServingBackend::Together));
    }

    #[test]
    fn test_SERVE_BKD_002_sovereign_blocked_hosts() {
        let hosts = PrivacyTier::Sovereign.blocked_hosts();
        assert!(hosts.contains(&"api.openai.com"));
        assert!(hosts.contains(&"api.anthropic.com"));
        assert!(hosts.contains(&"api.together.xyz"));
        assert!(hosts.contains(&"lambda.amazonaws.com"));
        assert!(hosts.contains(&"workers.cloudflare.com"));
        assert_eq!(hosts.len(), 14);
    }

    #[test]
    fn test_SERVE_BKD_002_standard_no_blocked_hosts() {
        let hosts = PrivacyTier::Standard.blocked_hosts();
        assert!(hosts.is_empty());
    }

    // ========================================================================
    // SERVE-BKD-003: Backend Selector Tests
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_003_default_selector() {
        let selector = BackendSelector::new();
        assert_eq!(selector.privacy, PrivacyTier::Standard);
        assert_eq!(selector.latency, LatencyTier::Interactive);
    }

    #[test]
    fn test_SERVE_BKD_003_sovereign_recommend() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);
        let backends = selector.recommend();
        // All recommended backends should be local
        for backend in &backends {
            assert!(backend.is_local(), "{:?} should be local", backend);
        }
    }

    #[test]
    fn test_SERVE_BKD_003_realtime_recommend() {
        let selector = BackendSelector::new().with_latency(LatencyTier::RealTime);
        let backends = selector.recommend();
        // Should include Groq (fastest)
        assert!(backends.contains(&ServingBackend::Groq));
    }

    #[test]
    fn test_SERVE_BKD_003_disabled_backend() {
        let selector = BackendSelector::new().disable(ServingBackend::OpenAI);
        assert!(!selector.is_valid(ServingBackend::OpenAI));
        assert!(selector.is_valid(ServingBackend::Anthropic));
    }

    #[test]
    fn test_SERVE_BKD_003_preferred_backend() {
        let selector = BackendSelector::new().prefer(ServingBackend::Anthropic);
        let backends = selector.recommend();
        assert_eq!(backends[0], ServingBackend::Anthropic);
    }

    // ========================================================================
    // SERVE-BKD-004: Validation Tests (Poka-Yoke)
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_004_validate_sovereign_blocks_openai() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);
        let result = selector.validate(ServingBackend::OpenAI);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrivacyViolation::TierViolation { .. }));
    }

    #[test]
    fn test_SERVE_BKD_004_validate_sovereign_allows_local() {
        let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);
        assert!(selector.validate(ServingBackend::Realizar).is_ok());
        assert!(selector.validate(ServingBackend::Ollama).is_ok());
    }

    #[test]
    fn test_SERVE_BKD_004_validate_disabled() {
        let selector = BackendSelector::new().disable(ServingBackend::Together);
        let result = selector.validate(ServingBackend::Together);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PrivacyViolation::BackendDisabled(_)));
    }

    #[test]
    fn test_SERVE_BKD_004_privacy_violation_display() {
        let err = PrivacyViolation::TierViolation {
            backend: ServingBackend::OpenAI,
            tier: PrivacyTier::Sovereign,
        };
        assert!(err.to_string().contains("OpenAI"));
        assert!(err.to_string().contains("Sovereign"));
    }

    // ========================================================================
    // SERVE-BKD-005: Builder Pattern Tests
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_005_builder_chain() {
        let selector = BackendSelector::new()
            .with_privacy(PrivacyTier::Private)
            .with_latency(LatencyTier::RealTime)
            .with_throughput(ThroughputTier::High)
            .with_cost(CostTier::Premium)
            .prefer(ServingBackend::AzureOpenAI)
            .disable(ServingBackend::AwsBedrock);

        assert_eq!(selector.privacy, PrivacyTier::Private);
        assert_eq!(selector.latency, LatencyTier::RealTime);
        assert_eq!(selector.throughput, ThroughputTier::High);
        assert_eq!(selector.cost, CostTier::Premium);
        assert!(selector.preferred.contains(&ServingBackend::AzureOpenAI));
        assert!(selector.disabled.contains(&ServingBackend::AwsBedrock));
    }

    // ========================================================================
    // SERVE-BKD-006: Edge Cases
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_006_empty_recommend() {
        // If all tier-recommended backends are disabled, recommend returns empty
        let selector = BackendSelector::new()
            .with_privacy(PrivacyTier::Sovereign)
            .with_latency(LatencyTier::Interactive)
            .disable(ServingBackend::Realizar)
            .disable(ServingBackend::Ollama)
            .disable(ServingBackend::LlamaCpp);
        let backends = selector.recommend();
        // With Interactive + Sovereign tier, only these 3 are recommended
        // When all are disabled, recommend returns empty
        assert!(backends.is_empty());
    }

    #[test]
    fn test_SERVE_BKD_006_batch_tier_prefers_cheap() {
        let selector = BackendSelector::new()
            .with_latency(LatencyTier::Batch)
            .with_privacy(PrivacyTier::Standard);
        let backends = selector.recommend();
        // Should include Lambda (excellent for batch)
        assert!(backends.contains(&ServingBackend::AwsLambda));
    }

    // ========================================================================
    // SERVE-BKD-007: Lambda & Serverless Tests
    // ========================================================================

    #[test]
    fn test_SERVE_BKD_007_lambda_is_serverless() {
        assert!(ServingBackend::AwsLambda.is_serverless());
        assert!(ServingBackend::CloudflareWorkers.is_serverless());
        assert!(ServingBackend::Modal.is_serverless());
        assert!(!ServingBackend::Realizar.is_serverless());
        assert!(!ServingBackend::OpenAI.is_serverless());
    }

    #[test]
    fn test_SERVE_BKD_007_lambda_api_host() {
        assert_eq!(ServingBackend::AwsLambda.api_host(), Some("lambda.amazonaws.com"));
        assert_eq!(ServingBackend::CloudflareWorkers.api_host(), Some("workers.cloudflare.com"));
    }

    #[test]
    fn test_SERVE_BKD_007_lambda_privacy_tier() {
        // Lambda should be allowed in Private tier (it's your own account)
        assert!(PrivacyTier::Private.allows(ServingBackend::AwsLambda));
        assert!(PrivacyTier::Standard.allows(ServingBackend::AwsLambda));
        // But not Sovereign (no network calls)
        assert!(!PrivacyTier::Sovereign.allows(ServingBackend::AwsLambda));
    }

    #[test]
    fn test_SERVE_BKD_007_batch_private_includes_lambda() {
        let selector = BackendSelector::new()
            .with_latency(LatencyTier::Batch)
            .with_privacy(PrivacyTier::Private);
        let backends = selector.recommend();
        // Lambda should be first for batch + private (pay per use)
        assert!(backends.contains(&ServingBackend::AwsLambda));
    }

    #[test]
    fn test_SERVE_BKD_007_interactive_private_includes_lambda() {
        let selector = BackendSelector::new()
            .with_latency(LatencyTier::Interactive)
            .with_privacy(PrivacyTier::Private);
        let backends = selector.recommend();
        // Lambda should be available for interactive private
        assert!(backends.contains(&ServingBackend::AwsLambda));
    }

    #[test]
    fn test_SERVE_BKD_007_lambda_is_remote() {
        assert!(ServingBackend::AwsLambda.is_remote());
        assert!(ServingBackend::CloudflareWorkers.is_remote());
    }

    #[test]
    fn test_SERVE_BKD_007_cloudflare_blocked_in_private() {
        // Cloudflare Workers not allowed in Private (not your infrastructure)
        assert!(!PrivacyTier::Private.allows(ServingBackend::CloudflareWorkers));
    }

    #[test]
    fn test_SERVE_BKD_007_private_blocked_hosts_includes_cloudflare() {
        let hosts = PrivacyTier::Private.blocked_hosts();
        assert!(hosts.contains(&"workers.cloudflare.com"));
        // But Lambda is NOT blocked
        assert!(!hosts.contains(&"lambda.amazonaws.com"));
    }
}
