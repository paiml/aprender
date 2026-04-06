//! OpenTelemetry OTLP Exporter — exports syscall traces as spans via OTLP protocol.
#[cfg(feature = "otlp")]
use anyhow::Result;
#[cfg(feature = "otlp")]
use opentelemetry::{
    trace::{
        Span, SpanContext, SpanKind, Status, TraceContextExt, TraceFlags, TraceState, Tracer,
        TracerProvider as _,
    },
    KeyValue,
};
#[cfg(feature = "otlp")]
use opentelemetry_otlp::WithExportConfig;
#[cfg(feature = "otlp")]
use opentelemetry_sdk::{
    trace::{BatchSpanProcessor, SdkTracerProvider as TracerProvider},
    Resource,
};

#[allow(unused_imports)]
use crate::trace_context::TraceContext;

use crate::metrics::Registry;

// Data types extracted to otlp_types.rs (500-line limit compliance)
pub use crate::otlp_types::*;

/// OTLP exporter for syscall traces
#[cfg(feature = "otlp")]
pub struct OtlpExporter {
    _runtime: tokio::runtime::Runtime, // Tokio runtime for async OTLP operations
    _provider: TracerProvider,
    tracer: opentelemetry_sdk::trace::Tracer,
    root_span: Option<opentelemetry_sdk::trace::Span>,
    remote_parent_context: Option<opentelemetry::Context>,
}

#[cfg(feature = "otlp")]
impl OtlpExporter {
    /// Create a new OTLP exporter (Sprint 33: with optional trace context)
    pub fn new(config: OtlpConfig, trace_context: Option<TraceContext>) -> Result<Self> {
        // Create a Tokio runtime for OTLP async operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("Failed to create Tokio runtime: {e}"))?;

        let (provider, tracer) = runtime.block_on(async {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(&config.endpoint)
                .build()?;

            // Default 30s export timeout causes CI hangs. Set 5s if unset.
            if std::env::var("OTEL_BSP_EXPORT_TIMEOUT").is_err() {
                std::env::set_var("OTEL_BSP_EXPORT_TIMEOUT", "5000");
            }
            let span_processor = BatchSpanProcessor::builder(exporter).build();

            eprintln!(
                "[renacer: OTLP batch config - size: {}, delay: {}ms, queue: {}]",
                config.batch_size, config.batch_delay_ms, config.queue_size
            );

            #[cfg_attr(not(feature = "gpu-tracing"), allow(unused_mut))]
            let mut resource_attrs = vec![
                KeyValue::new("compute.library", "trueno"),
                KeyValue::new("compute.library.version", "0.4.0"),
                KeyValue::new("compute.tracing.abstraction", "block_level"),
            ];

            #[cfg(feature = "gpu-tracing")]
            {
                resource_attrs.push(KeyValue::new("gpu.library", "wgpu"));
                resource_attrs.push(KeyValue::new("gpu.tracing.abstraction", "kernel_level"));
            }

            let resource = Resource::builder()
                .with_service_name(config.service_name.clone())
                .with_attributes(resource_attrs)
                .build();

            // Create tracer provider
            let provider = TracerProvider::builder()
                .with_span_processor(span_processor)
                .with_resource(resource)
                .build();

            let tracer = provider.tracer("renacer");

            Ok::<_, anyhow::Error>((provider, tracer))
        })?;

        let remote_parent_context = trace_context.map(|ctx| {
            let span_context = SpanContext::new(
                ctx.otel_trace_id(),
                ctx.otel_parent_id(),
                TraceFlags::new(ctx.trace_flags),
                true, // is_remote = true (context from external system)
                TraceState::default(),
            );

            opentelemetry::Context::current().with_remote_span_context(span_context)
        });

        Ok(OtlpExporter {
            _runtime: runtime,
            _provider: provider,
            tracer,
            root_span: None,
            remote_parent_context,
        })
    }

    /// Start a root span for the traced process (Sprint 33: with optional parent context)
    pub fn start_root_span(&mut self, program: &str, pid: i32) {
        let span_builder = self
            .tracer
            .span_builder(format!("process: {program}"))
            .with_kind(SpanKind::Server)
            .with_attributes(vec![
                KeyValue::new("process.command", program.to_string()),
                KeyValue::new("process.pid", i64::from(pid)),
            ]);

        let span = if let Some(ref parent_ctx) = self.remote_parent_context {
            span_builder.start_with_context(&self.tracer, parent_ctx)
        } else {
            span_builder.start(&self.tracer)
        };

        self.root_span = Some(span);
    }

    /// Record a syscall as a span
    pub fn record_syscall(
        &self,
        name: &str,
        duration_us: Option<u64>,
        result: i64,
        source_file: Option<&str>,
        source_line: Option<u32>,
    ) {
        let mut span = self
            .tracer
            .span_builder(format!("syscall: {name}"))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new("syscall.name", name.to_string()),
                KeyValue::new("syscall.result", result),
            ])
            .start(&self.tracer);

        // Add duration if available
        if let Some(duration) = duration_us {
            span.set_attribute(KeyValue::new("syscall.duration_us", duration as i64));
        }

        // Add source location if available
        if let Some(file) = source_file {
            span.set_attribute(KeyValue::new("code.filepath", file.to_string()));
        }
        if let Some(line) = source_line {
            span.set_attribute(KeyValue::new("code.lineno", i64::from(line)));
        }

        // Mark as error if syscall failed
        if result < 0 {
            span.set_status(Status::Error {
                description: format!("syscall failed with code: {result}").into(),
            });
        }

        span.end();
    }

    /// Record a transpiler decision as a span event (Sprint 31)
    pub fn record_decision(
        &mut self,
        category: &str,
        name: &str,
        result: Option<&str>,
        timestamp_us: u64,
    ) {
        if let Some(ref mut span) = self.root_span {
            // Create attributes for the decision event
            let mut attributes = vec![
                KeyValue::new("decision.category", category.to_string()),
                KeyValue::new("decision.name", name.to_string()),
                KeyValue::new("decision.timestamp_us", timestamp_us as i64),
            ];

            // Add result if available
            if let Some(res) = result {
                attributes.push(KeyValue::new("decision.result", res.to_string()));
            }

            // Add event to the root span
            span.add_event(format!("decision: {category}::{name}"), attributes);
        }
    }

    /// Record a compute block (multiple Trueno operations) as a span (Sprint 32)
    ///
    /// This exports a block of statistical computations (e.g., `calculate_statistics`)
    /// as a single span. Following Toyota Way principles, we trace at the block level
    /// rather than individual SIMD operations to avoid overhead (Muda) and false
    /// observability (Genchi Genbutsu).
    ///
    /// # Arguments
    ///
    /// * `block` - Metadata about the compute block (operation, duration, elements)
    ///
    /// # Adaptive Sampling
    ///
    /// This method should only be called if duration >= threshold (default 100μs).
    /// The caller (`trace_compute_block`! macro) handles sampling decisions.
    pub fn record_compute_block(&self, block: ComputeBlock) {
        let mut span = self
            .tracer
            .span_builder(format!("compute_block: {}", block.operation))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                // Only dynamic attributes on span (Toyota Way: no attribute explosion)
                KeyValue::new("compute.operation", block.operation.to_string()),
                KeyValue::new("compute.duration_us", block.duration_us as i64),
                KeyValue::new("compute.elements", block.elements as i64),
                KeyValue::new("compute.is_slow", block.is_slow),
            ])
            .start(&self.tracer);

        span.set_status(Status::Ok);
        span.end();
    }

    /// Record a GPU kernel execution as a span (Sprint 37)
    ///
    /// Exports GPU kernel timing captured via wgpu-profiler timestamp queries.
    /// Follows Sprint 32's adaptive sampling pattern (only trace if duration > threshold).
    ///
    /// # Arguments
    ///
    /// * `kernel` - Metadata about the GPU kernel execution
    ///
    /// # Adaptive Sampling
    ///
    /// This method should only be called if duration >= threshold (default 100μs).
    /// The caller (`GpuProfilerWrapper`) handles sampling decisions.
    pub fn record_gpu_kernel(&self, kernel: GpuKernel) {
        let mut span_attrs = vec![
            // Only dynamic attributes on span (Toyota Way: no attribute explosion)
            KeyValue::new("gpu.backend", kernel.backend.to_string()),
            KeyValue::new("gpu.kernel", kernel.kernel.clone()),
            KeyValue::new("gpu.duration_us", kernel.duration_us as i64),
            KeyValue::new("gpu.is_slow", kernel.is_slow),
        ];

        // Optional attributes
        if let Some(ref wg_size) = kernel.workgroup_size {
            span_attrs.push(KeyValue::new("gpu.workgroup_size", wg_size.clone()));
        }
        if let Some(elements) = kernel.elements {
            span_attrs.push(KeyValue::new("gpu.elements", elements as i64));
        }

        let mut span = self
            .tracer
            .span_builder(format!("gpu_kernel: {}", kernel.kernel))
            .with_kind(SpanKind::Internal)
            .with_attributes(span_attrs)
            .start(&self.tracer);

        span.set_status(Status::Ok);
        span.end();
    }

    /// Record a GPU memory transfer as a span (Sprint 39 - Phase 4)
    ///
    /// Exports GPU memory transfer timing (CPU↔GPU) captured via wall-clock measurement.
    /// Follows Sprint 37's adaptive sampling pattern (only trace if duration > threshold).
    ///
    /// # Arguments
    ///
    /// * `transfer` - Metadata about the GPU memory transfer
    ///
    /// # Adaptive Sampling
    ///
    /// This method should only be called if duration >= threshold (default 100μs).
    /// The caller (transfer tracking wrapper) handles sampling decisions.
    pub fn record_gpu_transfer(&self, transfer: GpuMemoryTransfer) {
        let mut span_attrs = vec![
            // Only dynamic attributes on span (Toyota Way: no attribute explosion)
            KeyValue::new("gpu_transfer.direction", transfer.direction.as_str().to_string()),
            KeyValue::new("gpu_transfer.bytes", transfer.bytes as i64),
            KeyValue::new("gpu_transfer.duration_us", transfer.duration_us as i64),
            KeyValue::new("gpu_transfer.bandwidth_mbps", transfer.bandwidth_mbps),
            KeyValue::new("gpu_transfer.is_slow", transfer.is_slow),
        ];

        // Optional buffer usage
        if let Some(ref usage) = transfer.buffer_usage {
            span_attrs.push(KeyValue::new("gpu_transfer.buffer_usage", usage.clone()));
        }

        let mut span = self
            .tracer
            .span_builder(format!("gpu_transfer: {}", transfer.label))
            .with_kind(SpanKind::Internal)
            .with_attributes(span_attrs)
            .start(&self.tracer);
        span.set_status(Status::Ok);
        span.end();
    }
    /// Finish the root span
    pub fn end_root_span(&mut self, exit_code: i32) {
        if let Some(mut span) = self.root_span.take() {
            span.set_attribute(KeyValue::new("process.exit_code", i64::from(exit_code)));
            if exit_code != 0 {
                span.set_status(Status::Error {
                    description: format!("process exited with code: {exit_code}").into(),
                });
            }
            span.end();
        }
    }
    /// Export a complete `UnifiedTrace` to OTLP (Sprint 40 - Section 7.1)
    ///
    /// - `ProcessSpan` as root span
    /// - `SyscallSpans` as children of process span
    /// - GPU kernels, GPU memory transfers, SIMD blocks, transpiler decisions
    ///
    ///
    /// # Arguments
    ///
    /// * `trace` - The unified trace to export
    ///
    /// # Returns
    ///
    /// Returns Ok(()) on success, or error if OTLP export fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let trace = UnifiedTrace::new(...);
    /// let mut exporter = OtlpExporter::new(config, None)?;
    /// exporter.export_unified_trace(&trace)?;
    /// ```
    pub fn export_unified_trace(
        &mut self,
        trace: &crate::unified_trace::UnifiedTrace,
    ) -> Result<()> {
        contract_pre_error_handling!();
        self.start_root_span(&trace.process_span.name, trace.process_span.pid);

        for syscall in &trace.syscall_spans {
            self.record_syscall(
                &syscall.name,
                Some(syscall.duration_nanos / 1000), // Convert nanos to micros
                syscall.return_value,
                None, // Future: Add source file tracking to SyscallSpan
                None, // Future: Add source line tracking to SyscallSpan
            );
        }

        for gpu_kernel in &trace.gpu_spans {
            self.record_gpu_kernel(gpu_kernel.clone());
        }

        for gpu_transfer in &trace.gpu_memory_transfers {
            self.record_gpu_transfer(gpu_transfer.clone());
        }
        for simd_block in &trace.simd_spans {
            self.record_compute_block(simd_block.clone());
        }

        for decision in &trace.transpiler_spans {
            self.record_decision(
                &decision.category,
                &decision.name,
                decision.result.as_ref().and_then(|v| v.as_str()),
                decision.timestamp_us,
            );
        }

        // 7. End the process span
        if let Some(exit_code) = trace.process_span.exit_code {
            self.end_root_span(exit_code);
        }

        Ok(())
    }

    /// Record metrics from a snapshot (Sprint 56)
    ///
    /// Exports metrics as span events on the root span. This is a simple
    /// integration that doesn't require the full OTLP metrics pipeline.
    /// For production use, consider using opentelemetry-metrics.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - Metrics snapshot to export
    pub fn record_metrics(&mut self, snapshot: &MetricsSnapshot) {
        if let Some(ref mut span) = self.root_span {
            // Export counters as events
            for counter in &snapshot.counters {
                let mut attrs = vec![
                    KeyValue::new("metric.type", "counter"),
                    KeyValue::new("metric.name", counter.name.clone()),
                    KeyValue::new("metric.value", counter.value as i64),
                ];
                for (k, v) in &counter.labels {
                    attrs.push(KeyValue::new(format!("metric.label.{k}"), v.clone()));
                }
                span.add_event("metric", attrs);
            }

            // Export gauges as events
            for gauge in &snapshot.gauges {
                let mut attrs = vec![
                    KeyValue::new("metric.type", "gauge"),
                    KeyValue::new("metric.name", gauge.name.clone()),
                    KeyValue::new("metric.value", gauge.value),
                ];
                for (k, v) in &gauge.labels {
                    attrs.push(KeyValue::new(format!("metric.label.{k}"), v.clone()));
                }
                span.add_event("metric", attrs);
            }

            // Export histograms as events
            for histogram in &snapshot.histograms {
                let mut attrs = vec![
                    KeyValue::new("metric.type", "histogram"),
                    KeyValue::new("metric.name", histogram.name.clone()),
                    KeyValue::new("metric.count", histogram.count as i64),
                    KeyValue::new("metric.sum", histogram.sum),
                ];
                for (k, v) in &histogram.labels {
                    attrs.push(KeyValue::new(format!("metric.label.{k}"), v.clone()));
                }
                // Add bucket info as JSON
                let buckets_json: String = histogram
                    .buckets
                    .iter()
                    .map(|(le, count)| format!("{{\"le\":{le},\"count\":{count}}}"))
                    .collect::<Vec<_>>()
                    .join(",");
                attrs.push(KeyValue::new("metric.buckets", format!("[{buckets_json}]")));
                span.add_event("metric", attrs);
            }
        }
    }

    /// Export metrics from a registry (Sprint 56)
    ///
    /// Convenience method that creates a snapshot and exports it.
    pub fn export_metrics(&mut self, registry: &std::sync::Arc<Registry>) {
        let snapshot = MetricsSnapshot::from_registry(registry);
        self.record_metrics(&snapshot);
    }

    /// Shutdown the exporter and flush remaining spans
    pub fn shutdown(&mut self) {
        // Span processor automatically flushes on drop
        // But we explicitly end the root span if it exists
        if self.root_span.is_some() {
            self.end_root_span(0);
        }
    }
}

#[cfg(feature = "otlp")]
impl Drop for OtlpExporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(not(feature = "otlp"))]
pub struct OtlpExporter;
#[cfg(not(feature = "otlp"))]
impl OtlpExporter {
    pub fn new(
        _: crate::otlp_types::OtlpConfig,
        _: Option<crate::trace_context::TraceContext>,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("OTLP not compiled in")
    }
    pub fn start_root_span(&mut self, _: &str, _: i32) {}
    pub fn record_syscall(&self, _: &str, _: Option<u64>, _: i64, _: Option<&str>, _: Option<u32>) {
    }
    pub fn record_decision(&mut self, _: &str, _: &str, _: Option<&str>, _: u64) {}
    pub fn record_metrics(&mut self, _: &MetricsSnapshot) {}
    pub fn export_metrics(&mut self, _: &std::sync::Arc<crate::metrics::Registry>) {}
    pub fn record_compute_block(&self, _: ComputeBlock) {}
    pub fn end_root_span(&mut self, _: i32) {}
    pub fn shutdown(&mut self) {}
}

#[cfg(test)]
#[path = "otlp_exporter_tests.rs"]
mod tests;
