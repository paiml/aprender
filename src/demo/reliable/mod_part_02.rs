#[allow(clippy::wildcard_imports)]
use super::*;
use std::time::{Duration, Instant};

impl PerformanceMetrics {
    /// Calculate tokens per second
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_time.as_secs_f64() > 0.0 {
            self.tokens_generated as f64 / self.generation_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Format metrics for display
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "Performance:\n  Load time: {:.2}s\n  Time to first token: {:.0}ms\n  Tokens/sec: {:.1}\n  Peak memory: {:.1} MB\n  Backend: {}",
            self.load_time.as_secs_f64(),
            self.time_to_first_token.as_millis(),
            self.tokens_per_second(),
            self.peak_memory as f64 / 1_000_000.0,
            self.backend
        )
    }

    /// Format as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"load_time_ms":{},"ttft_ms":{},"tokens_per_sec":{:.1},"peak_memory_mb":{:.1},"backend":"{}","tokens_generated":{}}}"#,
            self.load_time.as_millis(),
            self.time_to_first_token.as_millis(),
            self.tokens_per_second(),
            self.peak_memory as f64 / 1_000_000.0,
            self.backend,
            self.tokens_generated
        )
    }
}

/// Performance timer for tracking operations
#[derive(Debug)]
pub struct PerfTimer {
    start: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfTimer {
    /// Start a new timer
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Record a checkpoint
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints
            .push((name.to_string(), self.start.elapsed()));
    }

    /// Get elapsed time since start
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get time since last checkpoint (or start)
    #[must_use]
    pub fn since_last(&self) -> Duration {
        let last = self.checkpoints.last().map_or(Duration::ZERO, |(_, d)| *d);
        self.start.elapsed().saturating_sub(last)
    }

    /// Print all checkpoints (verbose mode)
    pub fn print_verbose(&self) {
        eprintln!("Timing breakdown:");
        let mut prev = Duration::ZERO;
        for (name, total) in &self.checkpoints {
            let delta = total.saturating_sub(prev);
            eprintln!(
                "  {}: {:.0}ms (total: {:.0}ms)",
                name,
                delta.as_millis(),
                total.as_millis()
            );
            prev = *total;
        }
    }
}

/// Detect the compute backend
#[must_use]
pub fn detect_backend() -> String {
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512".to_string();
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2".to_string();
        }
        if is_x86_feature_detected!("avx") {
            return "AVX".to_string();
        }
        if is_x86_feature_detected!("sse4.2") {
            return "SSE4.2".to_string();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON".to_string();
    }
    "Scalar".to_string()
}

// ============================================================================
// RDB-06: Model Provenance & Licensing
// ============================================================================

/// Model provenance information
#[derive(Debug, Clone)]
pub struct ModelProvenance {
    /// Model name
    pub name: String,
    /// Version or revision
    pub version: String,
    /// License (e.g., "Apache-2.0", "MIT", "Llama Community")
    pub license: String,
    /// Link to model card or paper
    pub source_url: Option<String>,
    /// Authors or organization
    pub authors: Option<String>,
}

impl ModelProvenance {
    /// Create new provenance info
    #[must_use]
    pub fn new(name: &str, version: &str, license: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            license: license.to_string(),
            source_url: None,
            authors: None,
        }
    }

    /// Add source URL
    #[must_use]
    pub fn with_source(mut self, url: &str) -> Self {
        self.source_url = Some(url.to_string());
        self
    }

    /// Add authors
    #[must_use]
    pub fn with_authors(mut self, authors: &str) -> Self {
        self.authors = Some(authors.to_string());
        self
    }

    /// Format for display at startup
    #[must_use]
    pub fn format(&self) -> String {
        use std::fmt::Write;
        let mut output = format!(
            "Model: {} ({})\nLicense: {}",
            self.name, self.version, self.license
        );
        if let Some(authors) = &self.authors {
            let _ = write!(output, "\nAuthors: {}", authors);
        }
        if let Some(url) = &self.source_url {
            let _ = write!(output, "\nSource: {}", url);
        }
        output
    }

    /// Format as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut json = format!(
            r#"{{"name":"{}","version":"{}","license":"{}""#,
            self.name, self.version, self.license
        );
        if let Some(authors) = &self.authors {
            let _ = write!(json, r#","authors":"{}""#, authors);
        }
        if let Some(url) = &self.source_url {
            let _ = write!(json, r#","source":"{}""#, url);
        }
        json.push('}');
        json
    }
}

/// Common model provenances
pub mod models {
    use super::ModelProvenance;

    /// TinyLlama 1.1B Chat
    #[must_use]
    pub fn tinyllama_chat() -> ModelProvenance {
        ModelProvenance::new("TinyLlama-1.1B-Chat", "v1.0", "Apache-2.0")
            .with_source("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            .with_authors("TinyLlama Team")
    }

    /// Qwen2 0.5B Instruct
    #[must_use]
    pub fn qwen2_0_5b() -> ModelProvenance {
        ModelProvenance::new("Qwen2-0.5B-Instruct", "v1.0", "Apache-2.0")
            .with_source("https://huggingface.co/Qwen/Qwen2-0.5B-Instruct")
            .with_authors("Alibaba Cloud")
    }

    /// Mistral 7B Instruct
    #[must_use]
    pub fn mistral_7b() -> ModelProvenance {
        ModelProvenance::new("Mistral-7B-Instruct", "v0.2", "Apache-2.0")
            .with_source("https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
            .with_authors("Mistral AI")
    }

    /// Phi-2
    #[must_use]
    pub fn phi2() -> ModelProvenance {
        ModelProvenance::new("phi-2", "v1.0", "MIT")
            .with_source("https://huggingface.co/microsoft/phi-2")
            .with_authors("Microsoft Research")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
