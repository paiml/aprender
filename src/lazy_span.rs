/// Sprint 36: Lazy Span Creation
///
/// Defers expensive span building operations until the span is actually exported.
/// This reduces overhead when:
/// - OTLP export is disabled
/// - Sampling drops the span
/// - Span is never finished
use std::borrow::Cow;

/// Lazy span builder that defers work until commit
pub struct LazySpan {
    name: Option<Cow<'static, str>>,
    attributes: Vec<(Cow<'static, str>, String)>,
    timestamp_nanos: u64,
    duration_nanos: u64,
    status_code: i32,
    committed: bool,
}

impl LazySpan {
    /// Create a new lazy span (minimal allocation)
    pub fn new() -> Self {
        LazySpan {
            name: None,
            attributes: Vec::new(),
            timestamp_nanos: 0,
            duration_nanos: 0,
            status_code: 0,
            committed: false,
        }
    }

    /// Set span name (zero-copy for static strings)
    pub fn with_name_static(mut self, name: &'static str) -> Self {
        self.name = Some(Cow::Borrowed(name));
        self
    }

    /// Set span name (owned string)
    pub fn with_name_owned(mut self, name: String) -> Self {
        self.name = Some(Cow::Owned(name));
        self
    }

    /// Add attribute (zero-copy key)
    pub fn with_attribute_static(mut self, key: &'static str, value: String) -> Self {
        self.attributes.push((Cow::Borrowed(key), value));
        self
    }

    /// Add attribute (owned key)
    pub fn with_attribute_owned(mut self, key: String, value: String) -> Self {
        self.attributes.push((Cow::Owned(key), value));
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp_nanos: u64) -> Self {
        self.timestamp_nanos = timestamp_nanos;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration_nanos: u64) -> Self {
        self.duration_nanos = duration_nanos;
        self
    }

    /// Set status code
    pub fn with_status(mut self, status_code: i32) -> Self {
        self.status_code = status_code;
        self
    }

    /// Check if span has been committed
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Commit the span (mark as ready for export)
    ///
    /// This is when the actual work happens. Before this, the span is just
    /// a lightweight builder collecting parameters.
    pub fn commit(mut self) -> CommittedSpan {
        self.committed = true;
        CommittedSpan {
            name: self.name.unwrap_or(Cow::Borrowed("")),
            attributes: self.attributes,
            timestamp_nanos: self.timestamp_nanos,
            duration_nanos: self.duration_nanos,
            status_code: self.status_code,
        }
    }

    /// Drop without committing (zero-cost when span not exported)
    pub fn cancel(self) {
        // Just drop - no export happens
    }
}

impl Default for LazySpan {
    fn default() -> Self {
        Self::new()
    }
}

/// Committed span ready for export
pub struct CommittedSpan {
    pub name: Cow<'static, str>,
    pub attributes: Vec<(Cow<'static, str>, String)>,
    pub timestamp_nanos: u64,
    pub duration_nanos: u64,
    pub status_code: i32,
}

/// Convenience macro for creating lazy spans with zero-copy
#[macro_export]
macro_rules! lazy_span {
    ($name:expr) => {
        LazySpan::new().with_name_static($name)
    };
    ($name:expr, $($key:expr => $value:expr),* $(,)?) => {
        {
            let mut span = LazySpan::new().with_name_static($name);
            $(
                span = span.with_attribute_static($key, $value);
            )*
            span
        }
    };
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(LazySpan: Send, Sync);
static_assertions::assert_impl_all!(CommittedSpan: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_span_minimal() {
        let span = LazySpan::new();
        assert!(!span.is_committed());
        assert_eq!(span.name, None);
        assert_eq!(span.attributes.len(), 0);
    }

    #[test]
    fn test_lazy_span_builder() {
        let span = LazySpan::new()
            .with_name_static("syscall:open")
            .with_attribute_static("syscall.name", "open".to_string())
            .with_attribute_static("syscall.result", "3".to_string())
            .with_timestamp(1234567890)
            .with_duration(1000);

        assert!(!span.is_committed());
        assert_eq!(span.name, Some(Cow::Borrowed("syscall:open")));
        assert_eq!(span.attributes.len(), 2);
    }

    #[test]
    fn test_lazy_span_commit() {
        let span = LazySpan::new().with_name_static("test").with_timestamp(100).with_duration(50);

        let committed = span.commit();
        assert_eq!(committed.name.as_ref(), "test");
        assert_eq!(committed.timestamp_nanos, 100);
        assert_eq!(committed.duration_nanos, 50);
    }

    #[test]
    fn test_lazy_span_cancel() {
        let span = LazySpan::new()
            .with_name_static("cancelled")
            .with_attribute_static("test", "value".to_string());

        // Just drop it - no work done
        span.cancel();
        // Test passes if no panic
    }

    #[test]
    fn test_lazy_span_zero_copy() {
        let span = LazySpan::new()
            .with_name_static("syscall:open")
            .with_attribute_static("syscall.name", "open".to_string());

        if let Some(Cow::Borrowed(_)) = span.name {
            // Zero-copy for static name
        } else {
            panic!("Expected borrowed name");
        }

        assert!(matches!(span.attributes[0].0, Cow::Borrowed(_)));
    }

    #[test]
    fn test_lazy_span_owned() {
        let dynamic_name = format!("dynamic_{}", 42);
        let span = LazySpan::new()
            .with_name_owned(dynamic_name.clone())
            .with_attribute_owned("key".to_string(), "value".to_string());

        if let Some(Cow::Owned(_)) = span.name {
            // Owned for dynamic name
        } else {
            panic!("Expected owned name");
        }

        assert!(matches!(span.attributes[0].0, Cow::Owned(_)));
    }

    #[test]
    fn test_span_not_committed_by_default() {
        let span = LazySpan::new().with_name_static("test");
        assert!(!span.is_committed());
    }

    #[test]
    fn test_multiple_attributes() {
        let span = LazySpan::new()
            .with_name_static("test")
            .with_attribute_static("key1", "value1".to_string())
            .with_attribute_static("key2", "value2".to_string())
            .with_attribute_static("key3", "value3".to_string());

        assert_eq!(span.attributes.len(), 3);
        assert_eq!(span.attributes[0].1, "value1");
        assert_eq!(span.attributes[1].1, "value2");
        assert_eq!(span.attributes[2].1, "value3");
    }
}
