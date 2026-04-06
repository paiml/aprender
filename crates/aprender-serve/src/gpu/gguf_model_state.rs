
impl ResourceMonitor {
    /// Create new resource monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory_bytes: std::sync::atomic::AtomicU64::new(0),
            gpu_utilization: std::sync::Mutex::new(0.0),
            queue_depth: std::sync::atomic::AtomicUsize::new(0),
            latencies: std::sync::Mutex::new(Vec::new()),
            last_latency_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes: u64) {
        self.memory_bytes
            .store(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record GPU utilization
    pub fn record_gpu_utilization(&self, utilization: f64) {
        *self.gpu_utilization.lock().expect("mutex poisoned") = utilization;
    }

    /// Record queue depth
    pub fn record_queue_depth(&self, depth: usize) {
        self.queue_depth
            .store(depth, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record latency
    pub fn record_latency(&self, duration: Duration) {
        let ms = duration.as_millis() as u64;
        self.last_latency_ms
            .store(ms, std::sync::atomic::Ordering::SeqCst);
        self.latencies.lock().expect("mutex poisoned").push(ms);
    }

    /// Get current metrics
    #[must_use]
    pub fn current_metrics(&self) -> ResourceMetrics {
        ResourceMetrics {
            memory_bytes: self.memory_bytes.load(std::sync::atomic::Ordering::SeqCst),
            gpu_utilization: *self.gpu_utilization.lock().expect("mutex poisoned"),
            queue_depth: self.queue_depth.load(std::sync::atomic::Ordering::SeqCst),
            last_latency_ms: self
                .last_latency_ms
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Get latency statistics
    #[must_use]
    pub fn latency_stats(&self) -> LatencyStats {
        let latencies = self.latencies.lock().expect("mutex poisoned");
        if latencies.is_empty() {
            return LatencyStats {
                min_ms: 0,
                max_ms: 0,
                avg_ms: 0,
            };
        }

        let min_ms = *latencies.iter().min().unwrap_or(&0);
        let max_ms = *latencies.iter().max().unwrap_or(&0);
        let sum: u64 = latencies.iter().sum();
        let avg_ms = sum / latencies.len() as u64;

        LatencyStats {
            min_ms,
            max_ms,
            avg_ms,
        }
    }

    /// Get snapshot for reporting
    #[must_use]
    pub fn snapshot(&self) -> ResourceSnapshot {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ResourceSnapshot {
            timestamp,
            memory_bytes: self.memory_bytes.load(std::sync::atomic::Ordering::SeqCst),
            gpu_utilization: *self.gpu_utilization.lock().expect("mutex poisoned"),
            queue_depth: self.queue_depth.load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// M33: GGUF HTTP Serving Integration (IMP-082, IMP-083)
// Per spec v2.15.0: Wire GpuModel to HTTP server
// ============================================================================

/// State for holding a loaded GGUF model in HTTP server context (IMP-082)
///
/// This struct wraps a GpuModel and provides thread-safe access for
/// the HTTP server to perform inference requests.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gpu::GgufModelState;
///
/// let state = GgufModelState::new();
/// assert!(!state.is_loaded());
///
/// // Load model
/// let state = load_gguf_to_gpu(vocab_size, hidden_dim, num_layers)?;
/// assert!(state.is_loaded());
/// ```
pub struct GgufModelState {
    /// Loaded GPU model (None if not loaded)
    model: Option<GpuModel>,
    /// Model name/path
    model_name: Option<String>,
    /// Whether model is ready for inference
    ready: bool,
}

impl std::fmt::Debug for GgufModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModelState")
            .field("model_name", &self.model_name)
            .field("ready", &self.ready)
            .field("is_loaded", &self.model.is_some())
            .finish()
    }
}

impl GgufModelState {
    /// Create empty state (no model loaded)
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            model_name: None,
            ready: false,
        }
    }

    /// Create state with a loaded model
    #[must_use]
    pub fn with_model(model: GpuModel, name: String) -> Self {
        Self {
            model: Some(model),
            model_name: Some(name),
            ready: true,
        }
    }

    /// Check if a model is loaded
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Check if model is ready for inference
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready && self.model.is_some()
    }

    /// Get model name
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// Get vocab size (0 if no model loaded)
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.model.as_ref().map_or(0, |m| m.config().vocab_size)
    }

    /// Get reference to the model (for inference)
    #[must_use]
    pub fn model(&self) -> Option<&GpuModel> {
        self.model.as_ref()
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> Option<&mut GpuModel> {
        self.model.as_mut()
    }
}

impl Default for GgufModelState {
    fn default() -> Self {
        Self::new()
    }
}

/// Load GGUF model to GPU (IMP-083)
///
/// Creates a minimal GPU model from configuration parameters.
/// This is the pipeline entry point for serving GGUF models via HTTP.
///
/// # Arguments
///
/// * `vocab_size` - Vocabulary size
/// * `hidden_dim` - Hidden dimension
/// * `num_layers` - Number of transformer layers
///
/// # Returns
///
/// * `Ok(GgufModelState)` - State with loaded model ready for inference
/// * `Err(RealizarError)` - If model creation fails
///
/// # Errors
///
/// Returns error if GPU initialization fails or model creation fails.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gpu::load_gguf_to_gpu;
///
/// let state = load_gguf_to_gpu(32000, 4096, 32)?;
/// assert!(state.is_ready());
/// ```
pub fn load_gguf_to_gpu(
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> Result<GgufModelState> {
    // Create GPU model config
    let num_heads = hidden_dim / 64; // Standard head dim of 64
    let config = GpuModelConfig {
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads: num_heads, // Standard MHA (no GQA)
        num_layers,
        intermediate_dim: hidden_dim * 4, // Standard FFN expansion
        eps: 1e-5,
        rope_theta: 10000.0, // Standard RoPE base frequency
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    // Create GPU model
    let model = GpuModel::new(config)?;

    // Wrap in state
    let model_name = format!("test_{}x{}x{}", vocab_size, hidden_dim, num_layers);
    Ok(GgufModelState::with_model(model, model_name))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod planner_tests;
