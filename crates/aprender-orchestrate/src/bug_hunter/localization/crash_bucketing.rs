//! Semantic crash bucketing (BH-20).
//!
//! Groups crashes by root cause pattern (e.g., index out of bounds,
//! null pointer dereference) for deduplication and triage.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::bug_hunter::types::{CrashBucketingMode, Finding, FindingSeverity, HuntMode};

/// Crash bucket for semantic grouping (BH-20).
#[derive(Debug, Clone)]
pub struct CrashBucket {
    /// Root cause pattern identifier
    pub pattern: String,
    /// Description of the root cause
    pub description: String,
    /// Crashes in this bucket
    pub crashes: Vec<CrashInfo>,
    /// Representative crash
    pub representative: Option<CrashInfo>,
}

/// Information about a single crash.
#[derive(Debug, Clone)]
pub struct CrashInfo {
    pub id: String,
    pub file: PathBuf,
    pub line: usize,
    pub message: String,
    pub stack_trace: Vec<StackFrame>,
}

/// A stack frame in a crash trace.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function: String,
    pub file: Option<PathBuf>,
    pub line: Option<usize>,
}

/// Root cause patterns for crash bucketing (BH-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootCausePattern {
    IndexOutOfBounds,
    NullPointerDeref,
    IntegerOverflow,
    DivisionByZero,
    StackOverflow,
    HeapOverflow,
    UseAfterFree,
    DoubleFree,
    UnwrapOnNone,
    AssertionFailed,
    Unknown,
}

impl std::fmt::Display for RootCausePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds => write!(f, "index_out_of_bounds"),
            Self::NullPointerDeref => write!(f, "null_pointer_deref"),
            Self::IntegerOverflow => write!(f, "integer_overflow"),
            Self::DivisionByZero => write!(f, "division_by_zero"),
            Self::StackOverflow => write!(f, "stack_overflow"),
            Self::HeapOverflow => write!(f, "heap_overflow"),
            Self::UseAfterFree => write!(f, "use_after_free"),
            Self::DoubleFree => write!(f, "double_free"),
            Self::UnwrapOnNone => write!(f, "unwrap_on_none"),
            Self::AssertionFailed => write!(f, "assertion_failed"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Semantic crash bucketer (BH-20).
pub struct CrashBucketer {
    pub mode: CrashBucketingMode,
    pub buckets: HashMap<String, CrashBucket>,
}

impl CrashBucketer {
    pub fn new(mode: CrashBucketingMode) -> Self {
        Self { mode, buckets: HashMap::new() }
    }

    /// Detect root cause pattern from crash message.
    pub fn detect_pattern(message: &str) -> RootCausePattern {
        let msg_lower = message.to_lowercase();
        detect_pattern_from_lower(&msg_lower)
    }
}

/// Pattern detection rules (ordered by specificity).
const PATTERN_RULES: &[(&[&str], RootCausePattern)] = &[
    (&["index out of bounds"], RootCausePattern::IndexOutOfBounds),
    (&["indexoutofbounds"], RootCausePattern::IndexOutOfBounds),
    (&["null"], RootCausePattern::NullPointerDeref),
    (&["nullptr"], RootCausePattern::NullPointerDeref),
    (&["division by zero"], RootCausePattern::DivisionByZero),
    (&["divide by zero"], RootCausePattern::DivisionByZero),
    (&["use after free"], RootCausePattern::UseAfterFree),
    (&["double free"], RootCausePattern::DoubleFree),
    (&["called `option::unwrap()`"], RootCausePattern::UnwrapOnNone),
];

/// Multi-keyword rules (all keywords must match).
const MULTI_KEYWORD_RULES: &[(&[&str], RootCausePattern)] = &[
    (&["overflow", "integer"], RootCausePattern::IntegerOverflow),
    (&["overflow", "stack"], RootCausePattern::StackOverflow),
    (&["unwrap", "none"], RootCausePattern::UnwrapOnNone),
];

fn detect_pattern_from_lower(msg: &str) -> RootCausePattern {
    // Multi-keyword rules first (more specific)
    for (keywords, pattern) in MULTI_KEYWORD_RULES {
        if keywords.iter().all(|kw| msg.contains(kw)) {
            return *pattern;
        }
    }
    // Single-keyword rules
    for (keywords, pattern) in PATTERN_RULES {
        if keywords.iter().any(|kw| msg.contains(kw)) {
            return *pattern;
        }
    }
    // Fallback checks
    if msg.contains("overflow") {
        return RootCausePattern::HeapOverflow;
    }
    if msg.contains("assertion") || msg.contains("assert") {
        return RootCausePattern::AssertionFailed;
    }
    RootCausePattern::Unknown
}

impl CrashBucketer {
    /// Add a crash to the appropriate bucket.
    pub fn add_crash(&mut self, crash: CrashInfo) {
        let bucket_key = match self.mode {
            CrashBucketingMode::None => {
                // Each crash gets its own bucket
                crash.id.clone()
            }
            CrashBucketingMode::StackTrace => {
                // Bucket by top 3 stack frames
                let frames: Vec<String> =
                    crash.stack_trace.iter().take(3).map(|f| f.function.clone()).collect();
                frames.join("::")
            }
            CrashBucketingMode::Semantic => {
                // Bucket by root cause pattern
                let pattern = Self::detect_pattern(&crash.message);
                format!("{}:{}", pattern, crash.file.display())
            }
        };

        let bucket = self.buckets.entry(bucket_key.clone()).or_insert_with(|| {
            let pattern = Self::detect_pattern(&crash.message);
            CrashBucket {
                pattern: pattern.to_string(),
                description: format!("{} in {}", pattern, crash.file.display()),
                crashes: Vec::new(),
                representative: None,
            }
        });

        // First crash becomes representative
        if bucket.representative.is_none() {
            bucket.representative = Some(crash.clone());
        }

        bucket.crashes.push(crash);
    }

    /// Get deduplicated findings from bucketed crashes.
    pub fn to_findings(&self) -> Vec<Finding> {
        self.buckets
            .values()
            .filter_map(|bucket| {
                bucket.representative.as_ref().map(|rep| {
                    Finding::new(
                        format!("BH-CRASH-{}", bucket.pattern.to_uppercase()),
                        &rep.file,
                        rep.line,
                        &bucket.description,
                    )
                    .with_description(format!(
                        "{} occurrence(s) of {} pattern",
                        bucket.crashes.len(),
                        bucket.pattern
                    ))
                    .with_severity(FindingSeverity::High)
                    .with_suspiciousness(0.8)
                    .with_discovered_by(HuntMode::Hunt)
                })
            })
            .collect()
    }

    /// Get deduplication statistics.
    pub fn stats(&self) -> (usize, usize) {
        let total_crashes: usize = self.buckets.values().map(|b| b.crashes.len()).sum();
        let unique_buckets = self.buckets.len();
        (total_crashes, unique_buckets)
    }
}
