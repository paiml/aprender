//! Golden trace manager for storing and comparing multiple trace versions.

use std::collections::HashMap;
use std::path::Path;

use super::trace::{GoldenComparator, GoldenTrace, TraceComparison};
use super::types::{GoldenTraceError, GoldenTraceResult, TraceMetrics};

/// Golden trace manager for storing multiple versions
#[derive(Debug, Clone)]
pub struct GoldenTraceManager {
    /// Storage directory
    storage_dir: std::path::PathBuf,
    /// Cached traces
    cache: HashMap<String, GoldenTrace>,
}

impl GoldenTraceManager {
    /// Create new manager
    pub fn new(storage_dir: std::path::PathBuf) -> Self {
        Self { storage_dir, cache: HashMap::new() }
    }

    /// Ensure storage directory exists
    pub fn ensure_directory(&self) -> GoldenTraceResult<()> {
        if !self.storage_dir.exists() {
            std::fs::create_dir_all(&self.storage_dir)
                .map_err(|e| GoldenTraceError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Capture current metrics as golden trace
    pub fn capture_golden(&mut self, name: &str, metrics: TraceMetrics) -> GoldenTraceResult<()> {
        self.ensure_directory()?;

        let trace = GoldenTrace::new(name, metrics);
        let path = self.storage_dir.join(format!("{}.toml", name));
        trace.save(&path)?;

        self.cache.insert(name.to_string(), trace);
        Ok(())
    }

    /// Load golden trace by name
    pub fn load_golden(&mut self, name: &str) -> GoldenTraceResult<GoldenTrace> {
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }

        let path = self.storage_dir.join(format!("{}.toml", name));
        let trace = GoldenTrace::load(&path)?;
        self.cache.insert(name.to_string(), trace.clone());
        Ok(trace)
    }

    /// Compare current metrics to golden
    pub fn compare_to_golden(
        &mut self,
        name: &str,
        current: &TraceMetrics,
    ) -> GoldenTraceResult<TraceComparison> {
        let golden = self.load_golden(name)?;
        let comparator = GoldenComparator::new();
        comparator.compare(current, &golden)
    }

    /// Detect regression against golden
    pub fn detect_regression(
        &mut self,
        name: &str,
        current: &TraceMetrics,
        threshold_percent: f64,
    ) -> GoldenTraceResult<bool> {
        let golden = self.load_golden(name)?;
        let comparator = GoldenComparator::new().with_threshold(threshold_percent);
        let comparison = comparator.compare(current, &golden)?;
        Ok(comparison.is_regression)
    }

    /// List all golden traces
    pub fn list_goldens(&self) -> GoldenTraceResult<Vec<String>> {
        if !self.storage_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&self.storage_dir)
            .map_err(|e| GoldenTraceError::IoError(e.to_string()))?;

        let mut names = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "toml" || ext == "json") {
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        names.push(name.to_string());
                    }
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Export golden trace to path
    pub fn export_trace(&self, name: &str, export_path: &Path) -> GoldenTraceResult<()> {
        let source_path = self.storage_dir.join(format!("{}.toml", name));
        let trace = GoldenTrace::load(&source_path)?;
        trace.save(export_path)
    }

    /// Delete golden trace
    pub fn delete_golden(&mut self, name: &str) -> GoldenTraceResult<()> {
        let path = self.storage_dir.join(format!("{}.toml", name));
        if !path.exists() {
            return Err(GoldenTraceError::NoBaseline);
        }

        std::fs::remove_file(&path).map_err(|e| GoldenTraceError::IoError(e.to_string()))?;
        self.cache.remove(name);
        Ok(())
    }

    /// Check if golden exists
    pub fn golden_exists(&self, name: &str) -> bool {
        self.cache.contains_key(name)
            || self.storage_dir.join(format!("{}.toml", name)).exists()
            || self.storage_dir.join(format!("{}.json", name)).exists()
    }
}
