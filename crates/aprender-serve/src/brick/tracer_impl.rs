
impl BrickTracer {
    /// Create a new tracer
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            position: 0,
            verbose: false,
            index: HashMap::new(),
        }
    }

    /// Create a tracer with verbose mode (stores full tensors)
    pub fn verbose() -> Self {
        Self {
            events: Vec::new(),
            position: 0,
            verbose: true,
            index: HashMap::new(),
        }
    }

    /// Set the current position being traced
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Log a tensor at the current computation step
    pub fn log(&mut self, name: &str, tensor: &[f32]) {
        let event = TraceEvent::new(name, tensor, self.position, self.verbose);
        let idx = self.events.len();
        self.index.insert(name.to_string(), idx);
        self.events.push(event);
    }

    /// Log a tensor with explicit position
    pub fn log_at(&mut self, name: &str, tensor: &[f32], position: usize) {
        let event = TraceEvent::new(name, tensor, position, self.verbose);
        let idx = self.events.len();
        self.index.insert(name.to_string(), idx);
        self.events.push(event);
    }

    /// Get all trace events
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Get event by name
    pub fn get(&self, name: &str) -> Option<&TraceEvent> {
        self.index.get(name).map(|&idx| &self.events[idx])
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.index.clear();
        self.position = 0;
    }

    /// Compare two tracers (CPU vs GPU)
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU tracer
    /// * `gpu` - GPU tracer
    /// * `tolerance` - Relative tolerance for L2 norm comparison (e.g., 0.01 = 1%)
    ///
    /// # Returns
    ///
    /// Comparison result with all divergences
    pub fn compare(cpu: &Self, gpu: &Self, tolerance: f32) -> TraceComparison {
        let mut diffs = Vec::new();

        // Compare events in order
        for cpu_event in &cpu.events {
            if let Some(gpu_event) = gpu.get(&cpu_event.name) {
                if !cpu_event.approx_eq(gpu_event, tolerance) {
                    diffs.push(TraceDiff {
                        name: cpu_event.name.clone(),
                        position: cpu_event.position,
                        cpu_l2: cpu_event.l2_norm,
                        gpu_l2: gpu_event.l2_norm,
                        relative_diff: cpu_event.relative_diff(gpu_event),
                        cpu_head: cpu_event.head,
                        gpu_head: gpu_event.head,
                    });
                }
            }
            // Skip if GPU doesn't have this event (may have different instrumentation)
        }

        TraceComparison { diffs, tolerance }
    }

    /// Print all events to stderr for debugging
    pub fn dump(&self) {
        eprintln!("=== BRICK TRACE ({} events) ===", self.events.len());
        for event in &self.events {
            eprintln!("  {event}");
        }
    }

    /// Print summary statistics
    pub fn summary(&self) {
        eprintln!("=== TRACE SUMMARY ===");
        eprintln!("Events: {}", self.events.len());
        if let Some(first) = self.events.first() {
            eprintln!("First: {}", first.name);
        }
        if let Some(last) = self.events.last() {
            eprintln!("Last: {}", last.name);
        }

        // Find largest L2 norm
        if let Some(max_event) = self.events.iter().max_by(|a, b| {
            a.l2_norm
                .partial_cmp(&b.l2_norm)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            eprintln!("Max L2: {} = {:.6}", max_event.name, max_event.l2_norm);
        }
    }
}

// Global tracers for CPU and GPU paths (thread-local to avoid contention).
// Used by trace_cpu! and trace_gpu! macros when "trace" feature is enabled.
#[cfg(feature = "trace")]
thread_local! {
    /// Thread-local tracer for CPU inference path
    pub static CPU_TRACER: std::cell::RefCell<BrickTracer> = std::cell::RefCell::new(BrickTracer::new());
    /// Thread-local tracer for GPU inference path
    pub static GPU_TRACER: std::cell::RefCell<BrickTracer> = std::cell::RefCell::new(BrickTracer::new());
}

/// Log a tensor to the CPU tracer for parity debugging.
///
/// Only active when the "trace" feature is enabled. When disabled, this macro
/// compiles to nothing (zero overhead).
///
/// # Arguments
///
/// * `$name` - Name of the computation step (e.g., "layer0_rope_q")
/// * `$tensor` - Slice of f32 values to log
/// * `$pos` - (Optional) Explicit position index
///
/// # Example
///
/// ```rust,ignore
/// trace_cpu!("embedding", &embedding_output);
/// trace_cpu!("layer0_attn", &attn_out, position);
/// ```
#[macro_export]
#[cfg(feature = "trace")]
macro_rules! trace_cpu {
    ($name:expr, $tensor:expr) => {
        $crate::brick::tracer::CPU_TRACER.with(|t| {
            t.borrow_mut().log($name, $tensor);
        });
    };
    ($name:expr, $tensor:expr, $pos:expr) => {
        $crate::brick::tracer::CPU_TRACER.with(|t| {
            t.borrow_mut().log_at($name, $tensor, $pos);
        });
    };
}

/// No-op version of trace_cpu when "trace" feature is disabled.
#[macro_export]
#[cfg(not(feature = "trace"))]
macro_rules! trace_cpu {
    ($name:expr, $tensor:expr) => {};
    ($name:expr, $tensor:expr, $pos:expr) => {};
}

/// Log a tensor to the GPU tracer for parity debugging.
///
/// Only active when the "trace" feature is enabled. When disabled, this macro
/// compiles to nothing (zero overhead).
///
/// # Arguments
///
/// * `$name` - Name of the computation step (e.g., "layer0_rope_q")
/// * `$tensor` - Slice of f32 values to log (must be downloaded from GPU first)
/// * `$pos` - (Optional) Explicit position index
///
/// # Example
///
/// ```rust,ignore
/// // After D2H copy from GPU buffer
/// trace_gpu!("embedding", &embedding_output);
/// trace_gpu!("layer0_attn", &attn_out, position);
/// ```
#[macro_export]
#[cfg(feature = "trace")]
macro_rules! trace_gpu {
    ($name:expr, $tensor:expr) => {
        $crate::brick::tracer::GPU_TRACER.with(|t| {
            t.borrow_mut().log($name, $tensor);
        });
    };
    ($name:expr, $tensor:expr, $pos:expr) => {
        $crate::brick::tracer::GPU_TRACER.with(|t| {
            t.borrow_mut().log_at($name, $tensor, $pos);
        });
    };
}

/// No-op version of trace_gpu when "trace" feature is disabled.
#[macro_export]
#[cfg(not(feature = "trace"))]
macro_rules! trace_gpu {
    ($name:expr, $tensor:expr) => {};
    ($name:expr, $tensor:expr, $pos:expr) => {};
}
