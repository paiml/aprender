
impl AuditSink for JsonFileAuditSink {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| AuditError::IoError(e.to_string()))?;

        for record in records {
            let json = serde_json::to_string(record)
                .map_err(|e| AuditError::SerializationError(e.to_string()))?;
            writeln!(file, "{}", json).map_err(|e| AuditError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

/// Audit error types
#[derive(Debug, Clone)]
pub enum AuditError {
    /// IO error writing audit records
    IoError(String),
    /// Serialization error
    SerializationError(String),
    /// Record not found
    NotFound(String),
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "Audit IO error: {msg}"),
            Self::SerializationError(msg) => write!(f, "Audit serialization error: {msg}"),
            Self::NotFound(id) => write!(f, "Audit record not found: {id}"),
        }
    }
}

impl std::error::Error for AuditError {}

/// High-performance audit logger with batching
pub struct AuditLogger {
    /// Output sink (file, database, event stream)
    sink: Box<dyn AuditSink>,
    /// Batch buffer for efficiency
    buffer: Mutex<VecDeque<AuditRecord>>,
    /// Buffer size threshold for auto-flush
    buffer_threshold: usize,
    /// Total records logged (for metrics)
    total_logged: AtomicU64,
    /// Model hash for provenance
    model_hash: String,
    /// Load timestamp
    load_timestamp: DateTime<Utc>,
}

impl AuditLogger {
    /// Create a new audit logger with default settings
    pub fn new(sink: Box<dyn AuditSink>) -> Self {
        Self {
            sink,
            buffer: Mutex::new(VecDeque::new()),
            buffer_threshold: 1000,
            total_logged: AtomicU64::new(0),
            model_hash: String::new(),
            load_timestamp: Utc::now(),
        }
    }

    /// Create an audit logger with in-memory sink (for testing)
    pub fn in_memory() -> (Self, std::sync::Arc<InMemoryAuditSink>) {
        let sink = std::sync::Arc::new(InMemoryAuditSink::new());
        let logger = Self::new(Box::new(InMemorySinkWrapper(sink.clone())));
        (logger, sink)
    }

    /// Set the model hash for provenance tracking
    pub fn with_model_hash(mut self, hash: impl Into<String>) -> Self {
        self.model_hash = hash.into();
        self
    }

    /// Set the load timestamp
    pub fn with_load_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.load_timestamp = timestamp;
        self
    }

    /// Set buffer threshold for auto-flush
    pub fn with_buffer_threshold(mut self, threshold: usize) -> Self {
        self.buffer_threshold = threshold;
        self
    }

    /// Log a request (returns the request ID)
    pub fn log_request(&self, model_type: &str, input_dims: &[usize]) -> Uuid {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, &self.model_hash, model_type)
            .with_input_dims(input_dims.to_vec());

        let mut buffer = self.buffer.lock().expect("test");
        buffer.push_back(record);

        request_id
    }

    /// Complete a request with response data
    pub fn log_response(
        &self,
        request_id: Uuid,
        prediction: serde_json::Value,
        latency: Duration,
        confidence: Option<f32>,
    ) {
        let mut buffer = self.buffer.lock().expect("test");

        // Find and update the record
        if let Some(record) = buffer
            .iter_mut()
            .find(|r| r.request_id == request_id.to_string())
        {
            record.prediction = prediction;
            record.latency_ms = latency.as_secs_f64() * 1000.0;
            record.confidence = confidence;
        }

        // Auto-flush if buffer exceeds threshold
        if buffer.len() >= self.buffer_threshold {
            if let Err(e) = self.flush_buffer_locked(&mut buffer) {
                eprintln!("audit: auto-flush failed: {e}");
            }
        }
    }

    /// Manually flush the buffer
    pub fn flush(&self) -> Result<(), AuditError> {
        let mut buffer = self.buffer.lock().expect("test");
        self.flush_buffer_locked(&mut buffer)
    }

    /// Internal flush with lock already held
    fn flush_buffer_locked(&self, buffer: &mut VecDeque<AuditRecord>) -> Result<(), AuditError> {
        if buffer.is_empty() {
            return Ok(());
        }

        let records: Vec<AuditRecord> = buffer.drain(..).collect();
        let count = records.len() as u64;

        self.sink.write_batch(&records)?;
        self.total_logged.fetch_add(count, Ordering::Relaxed);

        Ok(())
    }

    /// Get total records logged
    pub fn total_logged(&self) -> u64 {
        self.total_logged.load(Ordering::Relaxed)
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().expect("test").len()
    }

    /// Get model hash
    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    /// Get load timestamp
    pub fn load_timestamp(&self) -> DateTime<Utc> {
        self.load_timestamp
    }
}

/// Wrapper to make Arc<InMemoryAuditSink> implement AuditSink
struct InMemorySinkWrapper(std::sync::Arc<InMemoryAuditSink>);

impl AuditSink for InMemorySinkWrapper {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        self.0.write_batch(records)
    }

    fn flush(&self) -> Result<(), AuditError> {
        self.0.flush()
    }
}
