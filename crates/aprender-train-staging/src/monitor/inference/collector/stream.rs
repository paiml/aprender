//! StreamCollector - Write-through for persistent logging

use super::super::path::DecisionPath;
use super::super::trace::DecisionTrace;
use super::traits::TraceCollector;
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Trace format for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamFormat {
    /// Binary format (compact, fast)
    Binary,
    /// JSON format (human-readable)
    Json,
    /// JSON Lines (one JSON object per line)
    JsonLines,
}

/// Stream collector for persistent logging
///
/// Target: <1µs per trace
///
/// # Features
/// - Write-through to any `Write` impl
/// - Supports binary and JSON formats
/// - Buffered writes for efficiency
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{StreamCollector, LinearPath, StreamFormat};
/// use std::fs::File;
///
/// let file = File::create("traces.jsonl")?;
/// let mut collector = StreamCollector::<LinearPath, _>::new(file, StreamFormat::JsonLines);
/// collector.record(trace);
/// collector.flush()?;
/// ```
pub struct StreamCollector<P: DecisionPath, W: Write + Send> {
    writer: W,
    format: StreamFormat,
    buffer: Vec<DecisionTrace<P>>,
    flush_threshold: usize,
    count: usize,
}

impl<P: DecisionPath + Serialize, W: Write + Send + Sync> StreamCollector<P, W> {
    /// Create a new stream collector
    pub fn new(writer: W, format: StreamFormat) -> Self {
        Self { writer, format, buffer: Vec::with_capacity(100), flush_threshold: 100, count: 0 }
    }

    /// Set the flush threshold (number of traces before auto-flush)
    pub fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.flush_threshold = threshold;
        self
    }

    /// Get reference to the underlying writer
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Get mutable reference to the underlying writer
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Write a single trace
    fn write_trace(&mut self, trace: &DecisionTrace<P>) -> std::io::Result<()> {
        match self.format {
            StreamFormat::Binary => {
                let bytes = trace.to_bytes();
                // Write length prefix
                self.writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
                self.writer.write_all(&bytes)?;
            }
            StreamFormat::Json => {
                serde_json::to_writer(&mut self.writer, trace)?;
            }
            StreamFormat::JsonLines => {
                serde_json::to_writer(&mut self.writer, trace)?;
                self.writer.write_all(b"\n")?;
            }
        }
        Ok(())
    }
}

impl<P: DecisionPath + Serialize, W: Write + Send + Sync> TraceCollector<P>
    for StreamCollector<P, W>
{
    fn record(&mut self, trace: DecisionTrace<P>) {
        self.buffer.push(trace);
        self.count += 1;

        if self.buffer.len() >= self.flush_threshold {
            let _ = self.flush();
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let traces: Vec<_> = self.buffer.drain(..).collect();
        for trace in traces {
            self.write_trace(&trace)?;
        }
        self.writer.flush()
    }

    fn len(&self) -> usize {
        self.count
    }
}
