
impl InferenceTracer {

    /// Record an execution failure (F-JID-01: Jidoka error handling)
    ///
    /// Emits an `ExecutionFailed` event per AWS Step Functions parity (F-AWS-05).
    /// Use this when the inference cannot proceed due to missing config,
    /// invalid model format, or other fatal errors.
    ///
    /// # Arguments
    /// * `error` - High-level error category (e.g., "Initialization Failure")
    /// * `cause` - Specific cause of failure (e.g., "Missing config.json")
    pub fn record_execution_failed(&mut self, error: &str, cause: &str) {
        if !self.config.enabled {
            return;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::ExecutionFailed,
            previous_event_id: None,
            step: TraceStep::Tokenize, // Use first step as placeholder
            iteration: 0,
            layer: None,
            input_shape: vec![],
            output_shape: vec![],
            stats: TensorStats::default(),
            duration_us: 0,
            error: Some(TraceError::ExecutionFailed {
                cause: error.to_string(),
            }),
            cause: Some(cause.to_string()),
            details: TraceDetails::default(),
        };
        self.events.push(event);
        self.error_count += 1;
    }

    /// Format trace output as text
    #[must_use]
    pub fn format_text(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("=== APR Inference Trace ===\n");
        if !self.model_info.name.is_empty() {
            output.push_str(&format!(
                "Model: {} ({} layers, hidden={})\n",
                self.model_info.name, self.model_info.num_layers, self.model_info.hidden_dim
            ));
        }
        output.push('\n');

        // Group events by step type for cleaner output
        let mut current_step = None;
        let mut layer_count = 0;

        for event in &self.events {
            // Step header
            if current_step != Some(event.step) {
                if current_step == Some(TraceStep::TransformerBlock) && layer_count > 0 {
                    output.push_str(&format!("  ... ({} layers total)\n", layer_count));
                }
                current_step = Some(event.step);
                layer_count = 0;

                output.push_str(&format!(
                    "[{}/8] {}\n",
                    event.step.step_number(),
                    event.step.name()
                ));
            }

            // Step content
            match event.step {
                TraceStep::Tokenize => {
                    if let Some(ref text) = event.details.input_text {
                        let display_text = if text.len() > 50 {
                            format!("{}...", text.get(..50).expect("text length checked > 50"))
                        } else {
                            text.clone()
                        };
                        output.push_str(&format!("  Input:  {:?}\n", display_text));
                    }
                    if let Some(ref tokens) = event.details.output_tokens {
                        let display_tokens: Vec<_> = tokens.iter().take(10).collect();
                        if tokens.len() > 10 {
                            output.push_str(&format!(
                                "  Output: {:?}...  ({} tokens)\n",
                                display_tokens,
                                tokens.len()
                            ));
                        } else {
                            output.push_str(&format!(
                                "  Output: {:?}  ({} tokens)\n",
                                display_tokens,
                                tokens.len()
                            ));
                        }
                    }
                },
                TraceStep::Embed => {
                    output.push_str(&format!(
                        "  Input:  [{} token IDs]\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    output.push_str(&format!("  Output: {:?} float32\n", event.output_shape));
                    output.push_str(&format!(
                        "  Range:  min={:.2}, max={:.2}, mean={:.3}\n",
                        event.stats.min, event.stats.max, event.stats.mean
                    ));
                },
                TraceStep::TransformerBlock => {
                    layer_count += 1;
                    if layer_count <= 3 || self.config.verbose {
                        output.push_str(&format!(
                            "  Layer {:2}: attn {} ffn {}  {:?} range=[{:.1}, {:.1}]\n",
                            event.layer.unwrap_or(0),
                            if event.error.is_none() { "OK" } else { "ERR" },
                            if event.error.is_none() { "OK" } else { "ERR" },
                            event.output_shape,
                            event.stats.min,
                            event.stats.max
                        ));
                    }
                },
                TraceStep::LmHead => {
                    output.push_str(&format!(
                        "  Input:  [{}] (last token hidden state)\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    output.push_str(&format!(
                        "  Output: [{}] logits\n",
                        event.output_shape.first().unwrap_or(&0)
                    ));
                    if let Some(ref top_k) = event.details.top_k_logits {
                        output.push_str("  Top 5:  ");
                        for (i, (tok, logit)) in top_k.iter().take(5).enumerate() {
                            if i > 0 {
                                output.push_str(", ");
                            }
                            output.push_str(&format!("{}={:.2}", tok, logit));
                        }
                        output.push('\n');
                    }
                },
                TraceStep::Sample => {
                    output.push_str(&format!(
                        "  Logits:  [{}] -> scaled -> filtered\n",
                        event.input_shape.first().unwrap_or(&0)
                    ));
                    if let Some(ref probs) = event.details.top_k_probs {
                        output.push_str("  Probs:   ");
                        for (i, (tok, prob)) in probs.iter().take(5).enumerate() {
                            if i > 0 {
                                output.push_str(", ");
                            }
                            output.push_str(&format!("{}={:.2}", tok, prob));
                        }
                        output.push('\n');
                    }
                    if let Some(token) = event.details.sampled_token {
                        output.push_str(&format!("  Sampled: token_id={}\n", token));
                    }
                },
                TraceStep::Decode => {
                    if let Some(token) = event.details.sampled_token {
                        output.push_str(&format!("  Token ID:  {}\n", token));
                    }
                    if let Some(ref text) = event.details.decoded_text {
                        output.push_str(&format!("  Decoded:   {:?}\n", text));
                    }
                },
                TraceStep::KernelLaunch => {
                    if let Some(ref name) = event.details.kernel_name {
                        output.push_str(&format!("  Kernel:   {}\n", name));
                    }
                    if let Some([gx, gy, gz]) = event.details.grid_dims {
                        output.push_str(&format!("  Grid:     ({}, {}, {})\n", gx, gy, gz));
                    }
                    if let Some([bx, by, bz]) = event.details.block_dims {
                        output.push_str(&format!("  Block:    ({}, {}, {})\n", bx, by, bz));
                    }
                    if let Some(smem) = event.details.shared_mem_bytes {
                        output.push_str(&format!("  SharedMem: {} bytes\n", smem));
                    }
                    if let Some(ref strategy) = event.details.dispatch_strategy {
                        output.push_str(&format!("  Dispatch: {}\n", strategy));
                    }
                    if let Some(layer) = event.details.kernel_layer {
                        output.push_str(&format!("  Layer:    {}\n", layer));
                    }
                    output.push_str(&format!("  Duration: {}us\n", event.duration_us));
                },
                TraceStep::BrickProfile => {
                    if let Some(ref categories) = event.details.brick_categories {
                        output.push_str("  Categories:\n");
                        for (name, ns) in categories {
                            output.push_str(&format!(
                                "    {:<12} {:.3}ms\n",
                                name,
                                *ns as f64 / 1_000_000.0
                            ));
                        }
                    }
                    if let Some(ref timings) = event.details.brick_timings {
                        output.push_str("  Bricks:\n");
                        for (name, total_ns, count) in timings {
                            output.push_str(&format!(
                                "    {:<20} {:.3}ms (x{})\n",
                                name,
                                *total_ns as f64 / 1_000_000.0,
                                count
                            ));
                        }
                    }
                    output.push_str(&format!("  Duration: {}us\n", event.duration_us));
                },
                _ => {},
            }

            // Error output (Jidoka)
            if let Some(ref err) = event.error {
                output.push_str(&format!("  ERROR: {}\n", err));
                output.push_str(&format!("  Hint: {}\n", get_error_hint(err)));
            } else {
                output.push_str("  OK\n");
            }
            output.push('\n');
        }

        // Summary
        if self.error_count > 0 {
            output.push_str(&format!(
                "\n=== TRACE SUMMARY: {} errors, {} warnings ===\n",
                self.error_count, self.warning_count
            ));
        } else {
            output.push_str("\n=== TRACE COMPLETE: No errors ===\n");
        }

        output
    }

    /// Format trace as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"version\": \"1.0\",\n");
        json.push_str(&format!(
            "  \"timestamp\": \"{}\",\n",
            chrono::Utc::now().to_rfc3339()
        ));

        // Model info
        json.push_str("  \"model\": {\n");
        json.push_str(&format!("    \"name\": {:?},\n", self.model_info.name));
        json.push_str(&format!(
            "    \"num_layers\": {},\n",
            self.model_info.num_layers
        ));
        json.push_str(&format!(
            "    \"hidden_dim\": {},\n",
            self.model_info.hidden_dim
        ));
        json.push_str(&format!(
            "    \"vocab_size\": {},\n",
            self.model_info.vocab_size
        ));
        json.push_str(&format!(
            "    \"num_heads\": {}\n",
            self.model_info.num_heads
        ));
        json.push_str("  },\n");

        // Events
        json.push_str("  \"events\": [\n");
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("    {\n");
            // AWS Step Functions parity fields (F-AWS-01, F-AWS-02)
            json.push_str(&format!("      \"id\": {},\n", event.id));
            json.push_str(&format!("      \"timestamp\": {:?},\n", event.timestamp));
            json.push_str(&format!("      \"type\": {:?},\n", event.event_type.name()));
            json.push_str(&format!(
                "      \"previous_event_id\": {},\n",
                event
                    .previous_event_id
                    .map_or("null".to_string(), |id| id.to_string())
            ));
            // State details
            json.push_str(&format!("      \"step\": {:?},\n", event.step.name()));
            json.push_str(&format!("      \"iteration\": {},\n", event.iteration));
            json.push_str(&format!(
                "      \"layer\": {},\n",
                event.layer.map_or("null".to_string(), |l| l.to_string())
            ));
            json.push_str(&format!(
                "      \"input_shape\": {:?},\n",
                event.input_shape
            ));
            json.push_str(&format!(
                "      \"output_shape\": {:?},\n",
                event.output_shape
            ));
            json.push_str(&format!("      \"duration_us\": {},\n", event.duration_us));
            json.push_str("      \"stats\": {\n");
            json.push_str(&format!(
                "        \"min\": {},\n",
                format_json_float(event.stats.min)
            ));
            json.push_str(&format!(
                "        \"max\": {},\n",
                format_json_float(event.stats.max)
            ));
            json.push_str(&format!(
                "        \"mean\": {},\n",
                format_json_float(event.stats.mean)
            ));
            json.push_str(&format!(
                "        \"std\": {},\n",
                format_json_float(event.stats.std)
            ));
            json.push_str(&format!("        \"has_nan\": {},\n", event.stats.has_nan));
            json.push_str(&format!("        \"has_inf\": {}\n", event.stats.has_inf));
            json.push_str("      },\n");
            json.push_str(&format!(
                "      \"error\": {},\n",
                event
                    .error
                    .as_ref()
                    .map_or("null".to_string(), |e| format!("{:?}", e.to_string()))
            ));
            // F-AWS-05: cause field required for ExecutionFailed events
            json.push_str(&format!(
                "      \"cause\": {}\n",
                event
                    .cause
                    .as_ref()
                    .map_or("null".to_string(), |c| format!("{:?}", c))
            ));
            json.push_str("    }");
        }
        json.push_str("\n  ],\n");

        // Summary
        json.push_str(&format!("  \"error_count\": {},\n", self.error_count));
        json.push_str(&format!("  \"warning_count\": {}\n", self.warning_count));
        json.push_str("}\n");

        json
    }

    /// Write trace to configured output
    pub fn write_output(&self) -> std::io::Result<()> {
        let output = if self.config.output.is_some() {
            self.to_json()
        } else {
            self.format_text()
        };

        if let Some(ref path) = self.config.output {
            std::fs::write(path, output)?;
        } else {
            eprint!("{}", output);
        }

        Ok(())
    }
}

include!("mod_disabled_inference_tracer.rs");
