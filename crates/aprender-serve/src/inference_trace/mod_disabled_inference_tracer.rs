impl InferenceTracer {
    /// Create a new tracer with config
    #[must_use]
    pub fn new(config: TraceConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            model_info: ModelInfo::default(),
            step_start: None,
            error_count: 0,
            warning_count: 0,
            next_event_id: 1, // AWS Step Functions IDs start at 1
            last_entered_id: None,
        }
    }

    /// Create a disabled tracer (no-op)
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(TraceConfig::default())
    }

    /// Set model info
    pub fn set_model_info(&mut self, info: ModelInfo) {
        self.model_info = info;
    }

    /// Check if tracing is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if verbose tracing is enabled (requires D2H sync for stats)
    #[must_use]
    pub fn is_verbose(&self) -> bool {
        self.config.enabled && self.config.verbose
    }

    /// Get next event ID and increment (AWS Step Functions: monotonically increasing)
    fn next_id(&mut self) -> u64 {
        let id = self.next_event_id;
        self.next_event_id += 1;
        id
    }

    /// Generate ISO 8601 timestamp
    fn timestamp() -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }

    /// Start timing a step and emit TaskStateEntered event (AWS Step Functions F-AWS-01)
    pub fn start_step(&mut self, step: TraceStep) {
        if self.config.should_trace(step) {
            self.step_start = Some(Instant::now());

            // Emit TaskStateEntered event (F-AWS-01: Entry/Exit pairing)
            let entry_id = self.next_id();
            let event = TraceEvent {
                id: entry_id,
                timestamp: Self::timestamp(),
                event_type: AwsEventType::TaskStateEntered,
                previous_event_id: None, // Entry events have no predecessor
                step,
                iteration: 0,
                layer: None,
                input_shape: vec![],
                output_shape: vec![],
                stats: TensorStats::default(),
                duration_us: 0,
                error: None,
                cause: None,
                details: TraceDetails::default(),
            };
            self.events.push(event);
            // Store entry ID for the corresponding Exit event (F-AWS-02)
            self.last_entered_id = Some(entry_id);
        }
    }

    /// Trace encode step (tokenization)
    pub fn trace_encode(&mut self, input_text: &str, output_tokens: &[u32], vocab_size: usize) {
        if !self.config.should_trace(TraceStep::Tokenize) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Check for OOV tokens (Jidoka)
        let mut error = None;
        for &token_id in output_tokens {
            if token_id as usize >= vocab_size {
                error = Some(TraceError::VocabOverflow {
                    token_id,
                    vocab_size,
                });
                self.error_count += 1;
                break;
            }
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Tokenize,
            iteration: 0,
            layer: None,
            input_shape: vec![input_text.len()],
            output_shape: vec![output_tokens.len()],
            stats: TensorStats::default(),
            duration_us: duration,
            error,
            cause: None,
            details: TraceDetails {
                input_text: Some(input_text.to_string()),
                output_tokens: Some(output_tokens.to_vec()),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace embed step
    pub fn trace_embed(
        &mut self,
        token_count: usize,
        hidden_dim: usize,
        embeddings: Option<&[f32]>,
    ) {
        if !self.config.should_trace(TraceStep::Embed) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = embeddings.map(TensorStats::from_slice).unwrap_or_default();

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected { layer: None });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected { layer: None });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Embed,
            iteration: 0,
            layer: None,
            input_shape: vec![token_count],
            output_shape: vec![token_count, hidden_dim],
            stats,
            duration_us: duration,
            error,
            cause: None,
            details: TraceDetails::default(),
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace transformer layer
    pub fn trace_layer(
        &mut self,
        layer_idx: usize,
        iteration: usize,
        hidden_state: Option<&[f32]>,
        seq_len: usize,
        hidden_dim: usize,
    ) {
        if !self.config.should_trace(TraceStep::TransformerBlock) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = hidden_state
            .map(TensorStats::from_slice)
            .unwrap_or_default();

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected {
                layer: Some(layer_idx),
            });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected {
                layer: Some(layer_idx),
            });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::TransformerBlock,
            iteration,
            layer: Some(layer_idx),
            input_shape: vec![seq_len, hidden_dim],
            output_shape: vec![seq_len, hidden_dim],
            stats,
            duration_us: duration,
            error,
            cause: None,
            details: TraceDetails::default(),
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace LM head projection
    pub fn trace_lm_head(&mut self, iteration: usize, logits: &[f32], vocab_size: usize) {
        if !self.config.should_trace(TraceStep::LmHead) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);
        let stats = TensorStats::from_slice(logits);

        // Get top-5 logits
        let top_k = get_top_k_indices(logits, 5);

        let mut error = None;
        if stats.has_nan {
            error = Some(TraceError::NaNDetected { layer: None });
            self.error_count += 1;
        } else if stats.has_inf {
            error = Some(TraceError::InfDetected { layer: None });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::LmHead,
            iteration,
            layer: None,
            input_shape: vec![self.model_info.hidden_dim],
            output_shape: vec![vocab_size],
            stats,
            duration_us: duration,
            error,
            cause: None,
            details: TraceDetails {
                top_k_logits: Some(top_k),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace sampling step
    pub fn trace_sample(
        &mut self,
        iteration: usize,
        logits: &[f32],
        sampled_token: u32,
        temperature: f32,
        top_k: usize,
    ) {
        if !self.config.should_trace(TraceStep::Sample) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Compute softmax probabilities for top-k display
        let top_k_logits = get_top_k_indices(logits, top_k.min(10));
        let top_k_probs = compute_top_k_probs(logits, &top_k_logits);

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Sample,
            iteration,
            layer: None,
            input_shape: vec![logits.len()],
            output_shape: vec![1],
            stats: TensorStats::from_slice(logits),
            duration_us: duration,
            error: None,
            cause: None,
            details: TraceDetails {
                top_k_logits: Some(top_k_logits),
                top_k_probs: Some(top_k_probs),
                sampled_token: Some(sampled_token),
                temperature: Some(temperature),
                top_k: Some(top_k),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace decode step
    pub fn trace_decode(
        &mut self,
        iteration: usize,
        token_id: u32,
        decoded_text: &str,
        vocab_size: usize,
    ) {
        if !self.config.should_trace(TraceStep::Decode) {
            return;
        }

        let duration = self
            .step_start
            .map_or(0, |s| s.elapsed().as_micros() as u64);

        // Check for garbage output (APR-TOK-001 Jidoka)
        let mut error = None;
        if token_id as usize >= vocab_size {
            error = Some(TraceError::VocabOverflow {
                token_id,
                vocab_size,
            });
            self.error_count += 1;
        } else if is_garbage_output(decoded_text) {
            error = Some(TraceError::GarbageOutput {
                sample: decoded_text.chars().take(20).collect(),
            });
            self.error_count += 1;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: self.last_entered_id.take(),
            step: TraceStep::Decode,
            iteration,
            layer: None,
            input_shape: vec![1],
            output_shape: vec![decoded_text.len()],
            stats: TensorStats::default(),
            duration_us: duration,
            error,
            cause: None,
            details: TraceDetails {
                sampled_token: Some(token_id),
                decoded_text: Some(decoded_text.to_string()),
                ..Default::default()
            },
        };

        self.events.push(event);
        self.step_start = None;
    }

    /// Trace GPU kernel launch (GH-219: PTX-level tracing)
    ///
    /// Records a CUDA kernel launch event with grid/block configuration,
    /// shared memory usage, and dispatch strategy. This connects the
    /// logical layer trace to the physical PTX execution.
    pub fn trace_kernel_launch(
        &mut self,
        kernel_name: &str,
        layer_idx: Option<usize>,
        grid_dims: [u32; 3],
        block_dims: [u32; 3],
        shared_mem_bytes: u32,
        dispatch_strategy: Option<&str>,
        duration_us: u64,
    ) {
        if !self.config.should_trace(TraceStep::KernelLaunch) {
            return;
        }

        let event = TraceEvent {
            id: self.next_id(),
            timestamp: Self::timestamp(),
            event_type: AwsEventType::TaskStateExited,
            previous_event_id: None, // Kernel launches are leaf events
            step: TraceStep::KernelLaunch,
            iteration: 0,
            layer: layer_idx,
            input_shape: vec![],
            output_shape: vec![],
            stats: TensorStats::default(),
            duration_us,
            error: None,
            cause: None,
            details: TraceDetails {
                kernel_name: Some(kernel_name.to_string()),
                grid_dims: Some(grid_dims),
                block_dims: Some(block_dims),
                shared_mem_bytes: Some(shared_mem_bytes),
                kernel_layer: layer_idx,
                dispatch_strategy: dispatch_strategy.map(String::from),
                ..Default::default()
            },
        };

        self.events.push(event);
    }

    /// Get all collected events
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Get error count
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.error_count
    }
}
