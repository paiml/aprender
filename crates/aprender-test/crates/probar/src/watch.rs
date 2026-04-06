//! Watch Mode with Hot Reload (Feature 6)
//!
//! Automatic test re-execution on file changes.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Jidoka**: Immediate feedback on test failures
//! - **Kaizen**: Continuous improvement through rapid iteration
//! - **Muda**: Only re-run affected tests (smart filtering)

use crate::result::{ProbarError, ProbarResult};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

/// Configuration for watch mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    /// Patterns to watch (glob patterns)
    pub patterns: Vec<String>,
    /// Patterns to ignore
    pub ignore_patterns: Vec<String>,
    /// Debounce duration in milliseconds
    pub debounce_ms: u64,
    /// Whether to clear screen before re-run
    pub clear_screen: bool,
    /// Whether to run on initial start
    pub run_on_start: bool,
    /// Directories to watch
    pub watch_dirs: Vec<PathBuf>,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            patterns: vec!["**/*.rs".to_string(), "**/*.toml".to_string()],
            ignore_patterns: vec![
                "**/target/**".to_string(),
                "**/.git/**".to_string(),
                "**/node_modules/**".to_string(),
            ],
            debounce_ms: 300,
            clear_screen: true,
            run_on_start: true,
            watch_dirs: vec![PathBuf::from(".")],
        }
    }
}

impl WatchConfig {
    /// Create a new watch config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pattern to watch
    #[must_use]
    pub fn with_pattern(mut self, pattern: &str) -> Self {
        self.patterns.push(pattern.to_string());
        self
    }

    /// Add a pattern to ignore
    #[must_use]
    pub fn with_ignore(mut self, pattern: &str) -> Self {
        self.ignore_patterns.push(pattern.to_string());
        self
    }

    /// Set debounce duration
    #[must_use]
    pub const fn with_debounce(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Set clear screen behavior
    #[must_use]
    pub const fn with_clear_screen(mut self, clear: bool) -> Self {
        self.clear_screen = clear;
        self
    }

    /// Add a directory to watch
    #[must_use]
    pub fn with_watch_dir(mut self, dir: &Path) -> Self {
        self.watch_dirs.push(dir.to_path_buf());
        self
    }

    /// Check if a path matches watch patterns
    #[must_use]
    pub fn matches_pattern(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check ignore patterns first
        for pattern in &self.ignore_patterns {
            if Self::glob_matches(pattern, &path_str) {
                return false;
            }
        }

        // Check watch patterns
        for pattern in &self.patterns {
            if Self::glob_matches(pattern, &path_str) {
                return true;
            }
        }

        false
    }

    /// Simple glob matching (supports **, *, ?)
    fn glob_matches(pattern: &str, path: &str) -> bool {
        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        Self::glob_match_parts(&pattern_parts, &path_parts)
    }

    fn glob_match_parts(pattern_parts: &[&str], path_parts: &[&str]) -> bool {
        if pattern_parts.is_empty() {
            return path_parts.is_empty();
        }

        let first_pattern = pattern_parts[0];

        if first_pattern == "**" {
            // ** matches zero or more path segments
            let rest_pattern = &pattern_parts[1..];
            if rest_pattern.is_empty() {
                return true;
            }

            for i in 0..=path_parts.len() {
                if Self::glob_match_parts(rest_pattern, &path_parts[i..]) {
                    return true;
                }
            }
            return false;
        }

        if path_parts.is_empty() {
            return false;
        }

        let first_path = path_parts[0];

        // Match current segment
        if Self::glob_match_segment(first_pattern, first_path) {
            Self::glob_match_parts(&pattern_parts[1..], &path_parts[1..])
        } else {
            false
        }
    }

    fn glob_match_segment(pattern: &str, segment: &str) -> bool {
        // Handle * and ? in segment matching
        let mut pattern_chars = pattern.chars().peekable();
        let mut segment_chars = segment.chars();

        while let Some(p) = pattern_chars.next() {
            match p {
                '*' => {
                    // * matches any sequence of characters
                    if pattern_chars.peek().is_none() {
                        return true;
                    }
                    // Try matching remaining pattern at each position
                    let remaining: String = pattern_chars.collect();
                    let remaining_segment: String = segment_chars.collect();
                    for i in 0..=remaining_segment.len() {
                        if Self::glob_match_segment(&remaining, &remaining_segment[i..]) {
                            return true;
                        }
                    }
                    return false;
                }
                '?' => {
                    // ? matches any single character
                    if segment_chars.next().is_none() {
                        return false;
                    }
                }
                c => {
                    if segment_chars.next() != Some(c) {
                        return false;
                    }
                }
            }
        }

        segment_chars.next().is_none()
    }
}

/// A file change event
#[derive(Debug, Clone)]
pub struct FileChange {
    /// The changed file path
    pub path: PathBuf,
    /// Type of change
    pub kind: FileChangeKind,
    /// Timestamp of the change
    pub timestamp: Instant,
}

/// Kind of file change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileChangeKind {
    /// File was created
    Created,
    /// File was modified
    Modified,
    /// File was deleted
    Deleted,
    /// File was renamed
    Renamed,
    /// Unknown change type
    Other,
}

impl From<EventKind> for FileChangeKind {
    fn from(kind: EventKind) -> Self {
        match kind {
            EventKind::Create(_) => Self::Created,
            EventKind::Modify(_) => Self::Modified,
            EventKind::Remove(_) => Self::Deleted,
            EventKind::Other => Self::Other,
            EventKind::Any | EventKind::Access(_) => Self::Other,
        }
    }
}

/// Watch mode handler trait
pub trait WatchHandler: Send + Sync {
    /// Called when files change
    fn on_change(&self, changes: &[FileChange]) -> ProbarResult<()>;

    /// Called when watch starts
    fn on_start(&self) -> ProbarResult<()> {
        Ok(())
    }

    /// Called when watch stops
    fn on_stop(&self) -> ProbarResult<()> {
        Ok(())
    }
}

/// Simple closure-based watch handler
pub struct FnWatchHandler<F>
where
    F: Fn(&[FileChange]) -> ProbarResult<()> + Send + Sync,
{
    handler: F,
}

impl<F> std::fmt::Debug for FnWatchHandler<F>
where
    F: Fn(&[FileChange]) -> ProbarResult<()> + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnWatchHandler").finish_non_exhaustive()
    }
}

impl<F> FnWatchHandler<F>
where
    F: Fn(&[FileChange]) -> ProbarResult<()> + Send + Sync,
{
    /// Create a new function-based handler
    #[must_use]
    pub fn new(handler: F) -> Self {
        Self { handler }
    }
}

impl<F> WatchHandler for FnWatchHandler<F>
where
    F: Fn(&[FileChange]) -> ProbarResult<()> + Send + Sync,
{
    fn on_change(&self, changes: &[FileChange]) -> ProbarResult<()> {
        (self.handler)(changes)
    }
}

/// File system watcher for watch mode
pub struct FileWatcher {
    config: WatchConfig,
    watcher: Option<RecommendedWatcher>,
    receiver: Option<Receiver<Result<Event, notify::Error>>>,
    last_trigger: Option<Instant>,
    pending_changes: Vec<FileChange>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(config: WatchConfig) -> ProbarResult<Self> {
        Ok(Self {
            config,
            watcher: None,
            receiver: None,
            last_trigger: None,
            pending_changes: Vec::new(),
        })
    }

    /// Start watching
    pub fn start(&mut self) -> ProbarResult<()> {
        let (tx, rx): (
            Sender<Result<Event, notify::Error>>,
            Receiver<Result<Event, notify::Error>>,
        ) = channel();

        let watcher_config = Config::default().with_poll_interval(Duration::from_millis(100));

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                // Ignore send errors (receiver may have dropped)
                let _ = tx.send(res);
            },
            watcher_config,
        )
        .map_err(|e| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create watcher: {e}"),
            ))
        })?;

        // Watch configured directories
        for dir in &self.config.watch_dirs {
            if dir.exists() {
                watcher.watch(dir, RecursiveMode::Recursive).map_err(|e| {
                    ProbarError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to watch directory {:?}: {e}", dir),
                    ))
                })?;
            }
        }

        self.watcher = Some(watcher);
        self.receiver = Some(rx);
        Ok(())
    }

    /// Stop watching
    pub fn stop(&mut self) {
        self.watcher = None;
        self.receiver = None;
    }

    /// Check for changes (non-blocking)
    pub fn check_changes(&mut self) -> Option<Vec<FileChange>> {
        let receiver = self.receiver.as_ref()?;
        let now = Instant::now();

        // Collect all pending events
        while let Ok(result) = receiver.try_recv() {
            if let Ok(event) = result {
                for path in event.paths {
                    if self.config.matches_pattern(&path) {
                        self.pending_changes.push(FileChange {
                            path,
                            kind: event.kind.into(),
                            timestamp: now,
                        });
                    }
                }
            }
        }

        // Check if we should trigger (debounce)
        if self.pending_changes.is_empty() {
            return None;
        }

        let should_trigger = match self.last_trigger {
            Some(last) => now.duration_since(last).as_millis() >= self.config.debounce_ms as u128,
            None => true,
        };

        if should_trigger {
            self.last_trigger = Some(now);
            let changes = std::mem::take(&mut self.pending_changes);

            // Deduplicate by path
            let unique_paths: HashSet<_> = changes.iter().map(|c| c.path.clone()).collect();
            let deduped: Vec<FileChange> = unique_paths
                .into_iter()
                .filter_map(|path| changes.iter().find(|c| c.path == path).cloned())
                .collect();

            if deduped.is_empty() {
                None
            } else {
                Some(deduped)
            }
        } else {
            None
        }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &WatchConfig {
        &self.config
    }

    /// Check if watcher is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.watcher.is_some()
    }
}

impl std::fmt::Debug for FileWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWatcher")
            .field("config", &self.config)
            .field("is_running", &self.is_running())
            .field("pending_changes", &self.pending_changes.len())
            .finish()
    }
}

/// Watch session state
#[derive(Debug, Clone, Default)]
pub struct WatchStats {
    /// Number of times tests were triggered
    pub trigger_count: u64,
    /// Number of file changes detected
    pub change_count: u64,
    /// Total run time
    pub total_runtime: Duration,
    /// Time of last trigger
    pub last_trigger: Option<Instant>,
}

impl WatchStats {
    /// Create new stats
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a trigger
    pub fn record_trigger(&mut self, change_count: usize) {
        self.trigger_count += 1;
        self.change_count += change_count as u64;
        self.last_trigger = Some(Instant::now());
    }
}

/// Builder for creating watch mode configurations
#[derive(Debug)]
pub struct WatchBuilder {
    config: WatchConfig,
}

impl Default for WatchBuilder {
    fn default() -> Self {
        Self {
            config: WatchConfig {
                patterns: Vec::new(),
                ignore_patterns: Vec::new(),
                debounce_ms: 300,
                clear_screen: true,
                run_on_start: true,
                watch_dirs: vec![std::path::PathBuf::from(".")],
            },
        }
    }
}

impl WatchBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Watch Rust files
    #[must_use]
    pub fn rust_files(mut self) -> Self {
        self.config.patterns.push("**/*.rs".to_string());
        self
    }

    /// Watch TOML files
    #[must_use]
    pub fn toml_files(mut self) -> Self {
        self.config.patterns.push("**/*.toml".to_string());
        self
    }

    /// Watch test files only
    #[must_use]
    pub fn test_files(mut self) -> Self {
        self.config.patterns.push("**/tests/**/*.rs".to_string());
        self.config.patterns.push("**/*_test.rs".to_string());
        self.config.patterns.push("**/test_*.rs".to_string());
        self
    }

    /// Watch source directory
    #[must_use]
    pub fn src_dir(mut self) -> Self {
        self.config.watch_dirs.push(PathBuf::from("src"));
        self
    }

    /// Ignore target directory
    #[must_use]
    pub fn ignore_target(mut self) -> Self {
        self.config.ignore_patterns.push("**/target/**".to_string());
        self
    }

    /// Set debounce duration
    #[must_use]
    pub const fn debounce(mut self, ms: u64) -> Self {
        self.config.debounce_ms = ms;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> WatchConfig {
        self.config
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod watch_config_tests {
        use super::*;

        #[test]
        fn test_default() {
            let config = WatchConfig::default();
            assert!(!config.patterns.is_empty());
            assert!(!config.ignore_patterns.is_empty());
            assert_eq!(config.debounce_ms, 300);
        }

        #[test]
        fn test_with_pattern() {
            let config = WatchConfig::new().with_pattern("**/*.js");
            assert!(config.patterns.contains(&"**/*.js".to_string()));
        }

        #[test]
        fn test_with_ignore() {
            let config = WatchConfig::new().with_ignore("**/dist/**");
            assert!(config.ignore_patterns.contains(&"**/dist/**".to_string()));
        }

        #[test]
        fn test_with_debounce() {
            let config = WatchConfig::new().with_debounce(500);
            assert_eq!(config.debounce_ms, 500);
        }

        #[test]
        fn test_with_clear_screen() {
            let config = WatchConfig::new().with_clear_screen(false);
            assert!(!config.clear_screen);
            let config = WatchConfig::new().with_clear_screen(true);
            assert!(config.clear_screen);
        }

        #[test]
        fn test_with_watch_dir() {
            let config = WatchConfig::new().with_watch_dir(Path::new("src"));
            assert!(config.watch_dirs.contains(&PathBuf::from("src")));
        }

        #[test]
        fn test_matches_pattern_rust_file() {
            let config = WatchConfig::default();
            assert!(config.matches_pattern(Path::new("src/main.rs")));
            assert!(config.matches_pattern(Path::new("tests/test.rs")));
        }

        #[test]
        fn test_matches_pattern_toml_file() {
            let config = WatchConfig::default();
            assert!(config.matches_pattern(Path::new("Cargo.toml")));
        }

        #[test]
        fn test_ignores_target() {
            let config = WatchConfig::default();
            assert!(!config.matches_pattern(Path::new("target/debug/main.rs")));
        }

        #[test]
        fn test_ignores_git() {
            let config = WatchConfig::default();
            assert!(!config.matches_pattern(Path::new(".git/config")));
        }
    }

    mod glob_matching_tests {
        use super::*;

        #[test]
        fn test_exact_match() {
            assert!(WatchConfig::glob_match_segment("main.rs", "main.rs"));
            assert!(!WatchConfig::glob_match_segment("main.rs", "test.rs"));
        }

        #[test]
        fn test_star_wildcard() {
            assert!(WatchConfig::glob_match_segment("*.rs", "main.rs"));
            assert!(WatchConfig::glob_match_segment("*.rs", "test.rs"));
            assert!(!WatchConfig::glob_match_segment("*.rs", "main.js"));
        }

        #[test]
        fn test_question_wildcard() {
            assert!(WatchConfig::glob_match_segment("?.rs", "a.rs"));
            assert!(!WatchConfig::glob_match_segment("?.rs", "ab.rs"));
        }

        #[test]
        fn test_double_star() {
            assert!(WatchConfig::glob_matches("**/*.rs", "src/main.rs"));
            assert!(WatchConfig::glob_matches("**/*.rs", "src/lib/mod.rs"));
            assert!(WatchConfig::glob_matches(
                "**/target/**",
                "crates/probar/target/debug/build"
            ));
        }

        #[test]
        fn test_prefix_pattern() {
            assert!(WatchConfig::glob_matches("src/**/*.rs", "src/lib.rs"));
            assert!(WatchConfig::glob_matches(
                "src/**/*.rs",
                "src/modules/test.rs"
            ));
            assert!(!WatchConfig::glob_matches("src/**/*.rs", "tests/test.rs"));
        }
    }

    mod file_change_tests {
        use super::*;

        #[test]
        fn test_file_change_kind_from_event() {
            assert_eq!(
                FileChangeKind::from(EventKind::Create(notify::event::CreateKind::File)),
                FileChangeKind::Created
            );
            assert_eq!(
                FileChangeKind::from(EventKind::Modify(notify::event::ModifyKind::Data(
                    notify::event::DataChange::Content
                ))),
                FileChangeKind::Modified
            );
            assert_eq!(
                FileChangeKind::from(EventKind::Remove(notify::event::RemoveKind::File)),
                FileChangeKind::Deleted
            );
        }
    }

    mod file_watcher_tests {
        use super::*;

        #[test]
        fn test_new() {
            let config = WatchConfig::default();
            let watcher = FileWatcher::new(config);
            assert!(watcher.is_ok());
        }

        #[test]
        fn test_is_running_before_start() {
            let config = WatchConfig::default();
            let watcher = FileWatcher::new(config).unwrap();
            assert!(!watcher.is_running());
        }

        #[test]
        fn test_check_changes_before_start() {
            let config = WatchConfig::default();
            let mut watcher = FileWatcher::new(config).unwrap();
            assert!(watcher.check_changes().is_none());
        }

        #[test]
        fn test_start_and_stop() {
            let config = WatchConfig::new().with_watch_dir(Path::new("."));
            let mut watcher = FileWatcher::new(config).unwrap();
            assert!(!watcher.is_running());

            watcher.start().unwrap();
            assert!(watcher.is_running());

            watcher.stop();
            assert!(!watcher.is_running());
        }

        #[test]
        fn test_config_accessor() {
            let config = WatchConfig::new().with_debounce(500);
            let watcher = FileWatcher::new(config).unwrap();
            assert_eq!(watcher.config().debounce_ms, 500);
        }

        #[test]
        fn test_debug() {
            let config = WatchConfig::default();
            let watcher = FileWatcher::new(config).unwrap();
            let debug_str = format!("{:?}", watcher);
            assert!(debug_str.contains("FileWatcher"));
            assert!(debug_str.contains("is_running"));
        }

        #[test]
        fn test_start_stop_multiple_times() {
            let config = WatchConfig::new().with_watch_dir(Path::new("."));
            let mut watcher = FileWatcher::new(config).unwrap();

            watcher.start().unwrap();
            assert!(watcher.is_running());
            watcher.stop();
            assert!(!watcher.is_running());

            // Start again
            watcher.start().unwrap();
            assert!(watcher.is_running());
            watcher.stop();
            assert!(!watcher.is_running());
        }

        #[test]
        fn test_check_changes_after_start_no_events() {
            let config = WatchConfig::new().with_watch_dir(Path::new("."));
            let mut watcher = FileWatcher::new(config).unwrap();
            watcher.start().unwrap();

            // No changes should be detected immediately
            let changes = watcher.check_changes();
            assert!(changes.is_none());

            watcher.stop();
        }
    }

    mod watch_stats_tests {
        use super::*;

        #[test]
        fn test_new() {
            let stats = WatchStats::new();
            assert_eq!(stats.trigger_count, 0);
            assert_eq!(stats.change_count, 0);
        }

        #[test]
        fn test_record_trigger() {
            let mut stats = WatchStats::new();
            stats.record_trigger(3);

            assert_eq!(stats.trigger_count, 1);
            assert_eq!(stats.change_count, 3);
            assert!(stats.last_trigger.is_some());
        }

        #[test]
        fn test_multiple_triggers() {
            let mut stats = WatchStats::new();
            stats.record_trigger(2);
            stats.record_trigger(5);

            assert_eq!(stats.trigger_count, 2);
            assert_eq!(stats.change_count, 7);
        }
    }

    mod watch_builder_tests {
        use super::*;

        #[test]
        fn test_new() {
            let builder = WatchBuilder::new();
            let config = builder.build();
            assert!(config.patterns.is_empty());
        }

        #[test]
        fn test_rust_files() {
            let config = WatchBuilder::new().rust_files().build();
            assert!(config.patterns.contains(&"**/*.rs".to_string()));
        }

        #[test]
        fn test_toml_files() {
            let config = WatchBuilder::new().toml_files().build();
            assert!(config.patterns.contains(&"**/*.toml".to_string()));
        }

        #[test]
        fn test_test_files() {
            let config = WatchBuilder::new().test_files().build();
            assert!(config.patterns.contains(&"**/tests/**/*.rs".to_string()));
            assert!(config.patterns.contains(&"**/*_test.rs".to_string()));
        }

        #[test]
        fn test_ignore_target() {
            let config = WatchBuilder::new().ignore_target().build();
            assert!(config.ignore_patterns.contains(&"**/target/**".to_string()));
        }

        #[test]
        fn test_src_dir() {
            let config = WatchBuilder::new().src_dir().build();
            assert!(config.watch_dirs.contains(&PathBuf::from("src")));
        }

        #[test]
        fn test_debounce() {
            let config = WatchBuilder::new().debounce(500).build();
            assert_eq!(config.debounce_ms, 500);
        }

        #[test]
        fn test_chained_builder() {
            let config = WatchBuilder::new()
                .rust_files()
                .toml_files()
                .ignore_target()
                .debounce(200)
                .build();

            assert!(config.patterns.contains(&"**/*.rs".to_string()));
            assert!(config.patterns.contains(&"**/*.toml".to_string()));
            assert!(config.ignore_patterns.contains(&"**/target/**".to_string()));
            assert_eq!(config.debounce_ms, 200);
        }
    }

    mod fn_watch_handler_tests {
        use super::*;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        #[test]
        fn test_on_change() {
            let counter = Arc::new(AtomicU32::new(0));
            let counter_clone = Arc::clone(&counter);

            let handler = FnWatchHandler::new(move |_changes| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(())
            });

            let changes = vec![FileChange {
                path: PathBuf::from("test.rs"),
                kind: FileChangeKind::Modified,
                timestamp: Instant::now(),
            }];

            handler.on_change(&changes).unwrap();
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn test_debug() {
            let handler = FnWatchHandler::new(|_changes| Ok(()));
            let debug_str = format!("{:?}", handler);
            assert!(debug_str.contains("FnWatchHandler"));
        }
    }

    mod file_change_kind_tests {
        use super::*;

        #[test]
        fn test_other_kind() {
            let kind = FileChangeKind::from(EventKind::Other);
            assert_eq!(kind, FileChangeKind::Other);
        }

        #[test]
        fn test_access_kind() {
            let kind = FileChangeKind::from(EventKind::Access(notify::event::AccessKind::Read));
            assert_eq!(kind, FileChangeKind::Other);
        }
    }

    mod file_change_additional_tests {
        use super::*;

        #[test]
        fn test_debug() {
            let change = FileChange {
                path: PathBuf::from("test.rs"),
                kind: FileChangeKind::Modified,
                timestamp: Instant::now(),
            };
            let debug_str = format!("{:?}", change);
            assert!(debug_str.contains("test.rs"));
            assert!(debug_str.contains("Modified"));
        }

        #[test]
        fn test_clone() {
            let change = FileChange {
                path: PathBuf::from("test.rs"),
                kind: FileChangeKind::Created,
                timestamp: Instant::now(),
            };
            let cloned = change.clone();
            assert_eq!(change.path, cloned.path);
            assert_eq!(change.kind, cloned.kind);
        }
    }

    mod watch_handler_default_tests {
        use super::*;

        struct TestHandler;

        impl WatchHandler for TestHandler {
            fn on_change(&self, _changes: &[FileChange]) -> ProbarResult<()> {
                Ok(())
            }
        }

        #[test]
        fn test_on_start_default() {
            let handler = TestHandler;
            assert!(handler.on_start().is_ok());
        }

        #[test]
        fn test_on_stop_default() {
            let handler = TestHandler;
            assert!(handler.on_stop().is_ok());
        }
    }

    mod additional_coverage_tests {
        use super::*;

        // FileChangeKind additional coverage
        #[test]
        fn test_file_change_kind_any() {
            let kind = FileChangeKind::from(EventKind::Any);
            assert_eq!(kind, FileChangeKind::Other);
        }

        #[test]
        fn test_file_change_kind_renamed_variant() {
            // Test the Renamed variant exists and can be used
            let kind = FileChangeKind::Renamed;
            assert_eq!(kind, FileChangeKind::Renamed);
        }

        #[test]
        fn test_file_change_kind_hash() {
            // Test Hash trait for FileChangeKind
            use std::collections::HashSet;
            let mut set = HashSet::new();
            set.insert(FileChangeKind::Created);
            set.insert(FileChangeKind::Modified);
            set.insert(FileChangeKind::Deleted);
            set.insert(FileChangeKind::Renamed);
            set.insert(FileChangeKind::Other);
            assert_eq!(set.len(), 5);
        }

        #[test]
        fn test_file_change_kind_copy() {
            let kind = FileChangeKind::Modified;
            let copied = kind;
            assert_eq!(kind, copied);
        }

        // WatchConfig serialization tests
        #[test]
        fn test_watch_config_serialize() {
            let config = WatchConfig::default();
            let json = serde_json::to_string(&config).unwrap();
            assert!(json.contains("patterns"));
            assert!(json.contains("debounce_ms"));
        }

        #[test]
        fn test_watch_config_deserialize() {
            let json = r#"{
                "patterns": ["**/*.rs"],
                "ignore_patterns": ["**/target/**"],
                "debounce_ms": 500,
                "clear_screen": false,
                "run_on_start": false,
                "watch_dirs": ["."]
            }"#;
            let config: WatchConfig = serde_json::from_str(json).unwrap();
            assert_eq!(config.debounce_ms, 500);
            assert!(!config.clear_screen);
            assert!(!config.run_on_start);
        }

        #[test]
        fn test_watch_config_clone() {
            let config = WatchConfig::default();
            let cloned = config.clone();
            assert_eq!(config.debounce_ms, cloned.debounce_ms);
            assert_eq!(config.patterns.len(), cloned.patterns.len());
        }

        #[test]
        fn test_watch_config_debug() {
            let config = WatchConfig::default();
            let debug_str = format!("{:?}", config);
            assert!(debug_str.contains("WatchConfig"));
            assert!(debug_str.contains("patterns"));
        }

        // Glob matching edge cases
        #[test]
        fn test_glob_matches_empty_pattern_parts() {
            // Test with path having multiple slashes creating empty parts
            assert!(WatchConfig::glob_matches("**/*.rs", "//src//main.rs"));
        }

        #[test]
        fn test_glob_match_segment_star_at_end() {
            // Star at end of pattern matches any suffix
            assert!(WatchConfig::glob_match_segment("test*", "testing"));
            assert!(WatchConfig::glob_match_segment("test*", "test"));
            assert!(WatchConfig::glob_match_segment("test*", "test123"));
        }

        #[test]
        fn test_glob_match_segment_star_in_middle() {
            // Star in middle of pattern
            assert!(WatchConfig::glob_match_segment("te*st", "test"));
            assert!(WatchConfig::glob_match_segment("te*st", "teaast"));
            assert!(!WatchConfig::glob_match_segment("te*st", "testing"));
        }

        #[test]
        fn test_glob_match_segment_multiple_stars() {
            // Multiple stars in pattern
            assert!(WatchConfig::glob_match_segment("*.*", "test.rs"));
            assert!(WatchConfig::glob_match_segment("*_*", "test_file"));
        }

        #[test]
        fn test_glob_match_segment_question_at_end() {
            // Question mark at end
            assert!(WatchConfig::glob_match_segment("test?", "testX"));
            assert!(!WatchConfig::glob_match_segment("test?", "test"));
            assert!(!WatchConfig::glob_match_segment("test?", "testXY"));
        }

        #[test]
        fn test_glob_match_segment_question_in_middle() {
            // Question mark in middle
            assert!(WatchConfig::glob_match_segment("te?t", "test"));
            assert!(WatchConfig::glob_match_segment("te?t", "teat"));
            assert!(!WatchConfig::glob_match_segment("te?t", "tet"));
        }

        #[test]
        fn test_glob_match_segment_empty_pattern() {
            assert!(WatchConfig::glob_match_segment("", ""));
            assert!(!WatchConfig::glob_match_segment("", "x"));
        }

        #[test]
        fn test_glob_match_segment_empty_segment() {
            assert!(WatchConfig::glob_match_segment("", ""));
            assert!(!WatchConfig::glob_match_segment("a", ""));
        }

        #[test]
        fn test_glob_match_parts_empty_both() {
            let empty: Vec<&str> = vec![];
            assert!(WatchConfig::glob_match_parts(&empty, &empty));
        }

        #[test]
        fn test_glob_match_parts_empty_pattern_non_empty_path() {
            let empty: Vec<&str> = vec![];
            let path = vec!["src"];
            assert!(!WatchConfig::glob_match_parts(&empty, &path));
        }

        #[test]
        fn test_glob_match_parts_double_star_at_end() {
            // Double star at end matches any remaining path
            let pattern = vec!["src", "**"];
            let path = vec!["src", "lib", "mod.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path));
        }

        #[test]
        fn test_glob_match_parts_double_star_matches_zero_segments() {
            // Double star can match zero path segments
            let pattern = vec!["**", "*.rs"];
            let path = vec!["main.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path));
        }

        #[test]
        fn test_glob_match_parts_double_star_no_match() {
            // Double star followed by pattern that doesn't match anywhere
            let pattern = vec!["**", "*.xyz"];
            let path = vec!["src", "main.rs"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        #[test]
        fn test_glob_matches_node_modules() {
            let config = WatchConfig::default();
            assert!(!config.matches_pattern(Path::new("node_modules/package/index.js")));
        }

        #[test]
        fn test_matches_pattern_no_watch_patterns() {
            // Test with empty patterns - should not match anything
            let config = WatchConfig {
                patterns: vec![],
                ignore_patterns: vec![],
                debounce_ms: 300,
                clear_screen: true,
                run_on_start: true,
                watch_dirs: vec![],
            };
            assert!(!config.matches_pattern(Path::new("src/main.rs")));
        }

        #[test]
        fn test_matches_pattern_not_matching_any() {
            // File type that doesn't match any pattern
            let config = WatchConfig::default();
            assert!(!config.matches_pattern(Path::new("src/main.xyz")));
        }

        // WatchStats additional coverage
        #[test]
        fn test_watch_stats_default() {
            let stats = WatchStats::default();
            assert_eq!(stats.trigger_count, 0);
            assert_eq!(stats.change_count, 0);
            assert_eq!(stats.total_runtime, Duration::default());
            assert!(stats.last_trigger.is_none());
        }

        #[test]
        fn test_watch_stats_clone() {
            let mut stats = WatchStats::new();
            stats.record_trigger(5);
            let cloned = stats.clone();
            assert_eq!(stats.trigger_count, cloned.trigger_count);
            assert_eq!(stats.change_count, cloned.change_count);
        }

        #[test]
        fn test_watch_stats_debug() {
            let stats = WatchStats::new();
            let debug_str = format!("{:?}", stats);
            assert!(debug_str.contains("WatchStats"));
            assert!(debug_str.contains("trigger_count"));
        }

        // WatchBuilder additional coverage
        #[test]
        fn test_watch_builder_default() {
            let builder = WatchBuilder::default();
            let config = builder.build();
            assert!(config.patterns.is_empty());
            assert!(config.ignore_patterns.is_empty());
            assert_eq!(config.debounce_ms, 300);
            assert!(config.clear_screen);
            assert!(config.run_on_start);
        }

        #[test]
        fn test_watch_builder_debug() {
            let builder = WatchBuilder::new();
            let debug_str = format!("{:?}", builder);
            assert!(debug_str.contains("WatchBuilder"));
        }

        // FileChange additional tests
        #[test]
        fn test_file_change_all_kinds() {
            let kinds = [
                FileChangeKind::Created,
                FileChangeKind::Modified,
                FileChangeKind::Deleted,
                FileChangeKind::Renamed,
                FileChangeKind::Other,
            ];
            for kind in kinds {
                let change = FileChange {
                    path: PathBuf::from("test.rs"),
                    kind,
                    timestamp: Instant::now(),
                };
                let _ = format!("{:?}", change);
            }
        }

        // FnWatchHandler additional tests
        #[test]
        fn test_fn_watch_handler_with_error() {
            let handler = FnWatchHandler::new(|_changes| {
                Err(ProbarError::AssertionFailed {
                    message: "test error".to_string(),
                })
            });
            let changes = vec![FileChange {
                path: PathBuf::from("test.rs"),
                kind: FileChangeKind::Modified,
                timestamp: Instant::now(),
            }];
            assert!(handler.on_change(&changes).is_err());
        }

        #[test]
        fn test_fn_watch_handler_access_changes() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;

            let count = Arc::new(AtomicUsize::new(0));
            let count_clone = Arc::clone(&count);

            let handler = FnWatchHandler::new(move |changes| {
                count_clone.store(changes.len(), Ordering::SeqCst);
                Ok(())
            });

            let changes = vec![
                FileChange {
                    path: PathBuf::from("a.rs"),
                    kind: FileChangeKind::Created,
                    timestamp: Instant::now(),
                },
                FileChange {
                    path: PathBuf::from("b.rs"),
                    kind: FileChangeKind::Modified,
                    timestamp: Instant::now(),
                },
            ];

            handler.on_change(&changes).unwrap();
            assert_eq!(count.load(Ordering::SeqCst), 2);
        }

        // FileWatcher with non-existent directory
        #[test]
        fn test_file_watcher_with_nonexistent_dir() {
            let config =
                WatchConfig::new().with_watch_dir(Path::new("/nonexistent/directory/12345"));
            let mut watcher = FileWatcher::new(config).unwrap();
            // Should not error because we skip non-existent directories
            let result = watcher.start();
            assert!(result.is_ok());
            watcher.stop();
        }

        // Test watch config with multiple patterns
        #[test]
        fn test_watch_config_multiple_patterns_chained() {
            let config = WatchConfig::new()
                .with_pattern("**/*.rs")
                .with_pattern("**/*.toml")
                .with_pattern("**/*.md")
                .with_ignore("**/target/**")
                .with_ignore("**/.git/**");

            assert!(config.patterns.len() >= 3);
            assert!(config.ignore_patterns.len() >= 2);
        }

        // More glob edge cases
        #[test]
        fn test_glob_matches_deep_nesting() {
            assert!(WatchConfig::glob_matches(
                "**/*.rs",
                "a/b/c/d/e/f/g/h/i/j/test.rs"
            ));
        }

        #[test]
        fn test_glob_matches_single_segment() {
            assert!(WatchConfig::glob_matches("*.rs", "main.rs"));
            assert!(!WatchConfig::glob_matches("*.rs", "src/main.rs"));
        }

        #[test]
        fn test_glob_match_segment_star_no_match() {
            // Star pattern that can't match
            assert!(!WatchConfig::glob_match_segment("*.rs", "main.js"));
        }

        #[test]
        fn test_glob_match_segment_literal_mismatch() {
            assert!(!WatchConfig::glob_match_segment("abc", "abd"));
            assert!(!WatchConfig::glob_match_segment("abc", "ab"));
        }

        #[test]
        fn test_glob_match_segment_pattern_longer_than_segment() {
            assert!(!WatchConfig::glob_match_segment("abcdef", "abc"));
        }

        // Test WatchConfig::new explicitly
        #[test]
        fn test_watch_config_new() {
            let config = WatchConfig::new();
            assert!(!config.patterns.is_empty());
            assert!(config.run_on_start);
        }

        // Test glob_match_parts when pattern doesn't match path segment
        #[test]
        fn test_glob_match_parts_segment_mismatch() {
            let pattern = vec!["src", "lib.rs"];
            let path = vec!["src", "main.rs"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test glob_match_parts with single non-matching segment
        #[test]
        fn test_glob_match_parts_single_mismatch() {
            let pattern = vec!["foo"];
            let path = vec!["bar"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test pattern non-empty but path is empty (coverage for line 149-151)
        #[test]
        fn test_glob_match_parts_pattern_longer_than_path() {
            let pattern = vec!["src", "lib.rs"];
            let path = vec!["src"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test non-** pattern with empty path
        #[test]
        fn test_glob_match_parts_non_doublestar_empty_path() {
            let pattern = vec!["*.rs"];
            let empty: Vec<&str> = vec![];
            assert!(!WatchConfig::glob_match_parts(&pattern, &empty));
        }

        // Test double star matching multiple segments then failing
        #[test]
        fn test_glob_match_parts_double_star_exhaustive_search() {
            // ** tries all positions but none match
            let pattern = vec!["**", "specific.txt"];
            let path = vec!["a", "b", "c", "other.txt"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test segment with star that needs to try multiple positions
        #[test]
        fn test_glob_match_segment_star_backtrack() {
            // Pattern "a*b*c" against "aXXbYYc"
            assert!(WatchConfig::glob_match_segment("a*b*c", "aXXbYYc"));
            assert!(WatchConfig::glob_match_segment("a*b*c", "abc"));
            assert!(!WatchConfig::glob_match_segment("a*b*c", "aXXbYY"));
        }

        // Test question mark when segment is shorter than pattern needs
        #[test]
        fn test_glob_match_segment_question_exhausts_segment() {
            assert!(!WatchConfig::glob_match_segment("a??", "ab"));
        }

        // Test literal char mismatch at specific position
        #[test]
        fn test_glob_match_segment_literal_char_mismatch() {
            assert!(!WatchConfig::glob_match_segment("test", "Test"));
            assert!(!WatchConfig::glob_match_segment("abc", "axc"));
        }

        // Test pattern ends but segment has more chars
        #[test]
        fn test_glob_match_segment_pattern_shorter() {
            assert!(!WatchConfig::glob_match_segment("ab", "abc"));
        }

        // Test FileWatcher internal state - pending changes
        #[test]
        fn test_file_watcher_pending_changes_init() {
            let config = WatchConfig::new();
            let watcher = FileWatcher::new(config).unwrap();
            // Initial state has no pending changes
            assert!(!watcher.is_running());
            assert_eq!(watcher.config().debounce_ms, 300);
        }

        // Test FileWatcher debug with running state
        #[test]
        fn test_file_watcher_debug_running() {
            let config = WatchConfig::new().with_watch_dir(Path::new("."));
            let mut watcher = FileWatcher::new(config).unwrap();
            watcher.start().unwrap();
            let debug_str = format!("{:?}", watcher);
            assert!(debug_str.contains("true")); // is_running is true
            watcher.stop();
        }

        // Test WatchStats with total_runtime modification
        #[test]
        fn test_watch_stats_total_runtime() {
            let mut stats = WatchStats::new();
            stats.total_runtime = Duration::from_secs(10);
            assert_eq!(stats.total_runtime, Duration::from_secs(10));
        }

        // Test ignore pattern matching more thoroughly
        #[test]
        fn test_matches_pattern_ignore_takes_precedence() {
            let config = WatchConfig {
                patterns: vec!["**/*.rs".to_string()],
                ignore_patterns: vec!["**/test/**".to_string()],
                debounce_ms: 300,
                clear_screen: true,
                run_on_start: true,
                watch_dirs: vec![],
            };
            // Should be ignored even though it matches *.rs
            assert!(!config.matches_pattern(Path::new("test/main.rs")));
            // Should match since not in ignored path
            assert!(config.matches_pattern(Path::new("src/main.rs")));
        }

        // Test glob_matches with exact file name (no directory)
        #[test]
        fn test_glob_matches_root_file() {
            assert!(WatchConfig::glob_matches("*.toml", "Cargo.toml"));
            assert!(!WatchConfig::glob_matches("*.toml", "src/Cargo.toml"));
        }

        // Test the ** matching exactly at different depths
        #[test]
        fn test_double_star_various_depths() {
            // ** at start matches 0 segments
            let pattern = vec!["**", "src", "main.rs"];
            let path = vec!["src", "main.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path));

            // ** at start matches 1 segment
            let path2 = vec!["project", "src", "main.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path2));

            // ** at start matches many segments
            let path3 = vec!["a", "b", "c", "src", "main.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path3));
        }

        // WatchConfig run_on_start field test
        #[test]
        fn test_watch_config_run_on_start_default_true() {
            let config = WatchConfig::default();
            assert!(config.run_on_start);
        }

        // Additional glob edge cases for remaining coverage
        #[test]
        fn test_glob_match_segment_star_with_remaining_pattern() {
            // * in middle followed by more pattern
            assert!(WatchConfig::glob_match_segment("foo*bar", "fooXbar"));
            assert!(WatchConfig::glob_match_segment("foo*bar", "foobar"));
            assert!(WatchConfig::glob_match_segment("foo*bar", "fooXXXbar"));
            assert!(!WatchConfig::glob_match_segment("foo*bar", "fooXba"));
        }

        #[test]
        fn test_glob_match_segment_star_matches_empty() {
            // Star can match zero characters
            assert!(WatchConfig::glob_match_segment("a*b", "ab"));
        }

        #[test]
        fn test_glob_match_segment_consecutive_stars() {
            // Multiple consecutive stars (edge case)
            assert!(WatchConfig::glob_match_segment("**", "anything"));
            assert!(WatchConfig::glob_match_segment("a**b", "aXXXb"));
        }

        #[test]
        fn test_glob_match_segment_star_followed_by_literal() {
            // Star followed by literal that appears multiple times
            assert!(WatchConfig::glob_match_segment("*a", "aaa"));
            assert!(WatchConfig::glob_match_segment("*a", "XXXa"));
            assert!(!WatchConfig::glob_match_segment("*a", "XXXb"));
        }

        #[test]
        fn test_glob_matches_leading_slash() {
            // Path with leading slash
            assert!(WatchConfig::glob_matches("**/*.rs", "/src/main.rs"));
        }

        #[test]
        fn test_glob_match_parts_double_star_only() {
            // Just ** matches everything
            let pattern = vec!["**"];
            let path = vec!["a", "b", "c"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path));

            let empty: Vec<&str> = vec![];
            assert!(WatchConfig::glob_match_parts(&pattern, &empty));
        }

        #[test]
        fn test_watch_config_watch_dirs_default() {
            let config = WatchConfig::default();
            assert!(!config.watch_dirs.is_empty());
            assert!(config.watch_dirs.contains(&PathBuf::from(".")));
        }

        // Test FileChange kind variants debug
        #[test]
        fn test_file_change_kind_debug() {
            let kinds = [
                (FileChangeKind::Created, "Created"),
                (FileChangeKind::Modified, "Modified"),
                (FileChangeKind::Deleted, "Deleted"),
                (FileChangeKind::Renamed, "Renamed"),
                (FileChangeKind::Other, "Other"),
            ];
            for (kind, expected) in kinds {
                let debug_str = format!("{:?}", kind);
                assert!(debug_str.contains(expected));
            }
        }

        // Test EventKind conversion comprehensively
        #[test]
        fn test_file_change_kind_from_create_any() {
            let kind = FileChangeKind::from(EventKind::Create(notify::event::CreateKind::Any));
            assert_eq!(kind, FileChangeKind::Created);
        }

        #[test]
        fn test_file_change_kind_from_create_folder() {
            let kind = FileChangeKind::from(EventKind::Create(notify::event::CreateKind::Folder));
            assert_eq!(kind, FileChangeKind::Created);
        }

        #[test]
        fn test_file_change_kind_from_modify_any() {
            let kind = FileChangeKind::from(EventKind::Modify(notify::event::ModifyKind::Any));
            assert_eq!(kind, FileChangeKind::Modified);
        }

        #[test]
        fn test_file_change_kind_from_modify_name() {
            let kind = FileChangeKind::from(EventKind::Modify(notify::event::ModifyKind::Name(
                notify::event::RenameMode::Any,
            )));
            assert_eq!(kind, FileChangeKind::Modified);
        }

        #[test]
        fn test_file_change_kind_from_modify_metadata() {
            let kind = FileChangeKind::from(EventKind::Modify(
                notify::event::ModifyKind::Metadata(notify::event::MetadataKind::Any),
            ));
            assert_eq!(kind, FileChangeKind::Modified);
        }

        #[test]
        fn test_file_change_kind_from_remove_any() {
            let kind = FileChangeKind::from(EventKind::Remove(notify::event::RemoveKind::Any));
            assert_eq!(kind, FileChangeKind::Deleted);
        }

        #[test]
        fn test_file_change_kind_from_remove_folder() {
            let kind = FileChangeKind::from(EventKind::Remove(notify::event::RemoveKind::Folder));
            assert_eq!(kind, FileChangeKind::Deleted);
        }

        #[test]
        fn test_file_change_kind_from_access_close() {
            let kind = FileChangeKind::from(EventKind::Access(notify::event::AccessKind::Close(
                notify::event::AccessMode::Any,
            )));
            assert_eq!(kind, FileChangeKind::Other);
        }

        // Test WatchBuilder with all methods chained
        #[test]
        fn test_watch_builder_all_options() {
            let config = WatchBuilder::new()
                .rust_files()
                .toml_files()
                .test_files()
                .src_dir()
                .ignore_target()
                .debounce(100)
                .build();

            assert!(config.patterns.contains(&"**/*.rs".to_string()));
            assert!(config.patterns.contains(&"**/*.toml".to_string()));
            assert!(config.patterns.contains(&"**/tests/**/*.rs".to_string()));
            assert!(config.patterns.contains(&"**/*_test.rs".to_string()));
            assert!(config.patterns.contains(&"**/test_*.rs".to_string()));
            assert!(config.watch_dirs.contains(&PathBuf::from("src")));
            assert!(config.ignore_patterns.contains(&"**/target/**".to_string()));
            assert_eq!(config.debounce_ms, 100);
        }

        // Test matches_pattern with various path formats
        #[test]
        fn test_matches_pattern_various_paths() {
            let config = WatchConfig::default();

            // Absolute-like paths (Unix style)
            assert!(config.matches_pattern(Path::new("/home/user/project/src/main.rs")));

            // Windows-like paths (if running on Windows this would match differently)
            // For now just test that it handles them gracefully
            let _ = config.matches_pattern(Path::new("C:\\Users\\test\\main.rs"));
        }

        // Test empty string pattern matching
        #[test]
        fn test_glob_matches_empty_strings() {
            // Empty pattern splits to [""], path splits to [] after filtering empty strings
            // glob_match_parts([""], []) -> first_pattern = "", path_parts is empty
            // Since "" != "**" and path_parts is empty, returns false

            // Empty pattern against non-empty path should not match
            assert!(!WatchConfig::glob_matches("", "src"));

            // Empty pattern against empty path
            // This won't match because pattern_parts [""] is not empty but path_parts [] is
            assert!(!WatchConfig::glob_matches("", ""));
        }

        // Test FileChange timestamp field
        #[test]
        fn test_file_change_timestamp() {
            let before = Instant::now();
            let change = FileChange {
                path: PathBuf::from("test.rs"),
                kind: FileChangeKind::Modified,
                timestamp: Instant::now(),
            };
            let after = Instant::now();

            assert!(change.timestamp >= before);
            assert!(change.timestamp <= after);
        }

        // Test FileWatcher with empty watch_dirs
        #[test]
        fn test_file_watcher_empty_watch_dirs() {
            let config = WatchConfig {
                patterns: vec!["**/*.rs".to_string()],
                ignore_patterns: vec![],
                debounce_ms: 300,
                clear_screen: true,
                run_on_start: true,
                watch_dirs: vec![],
            };
            let mut watcher = FileWatcher::new(config).unwrap();
            // Should succeed with no directories to watch
            assert!(watcher.start().is_ok());
            assert!(watcher.is_running());
            watcher.stop();
        }

        // Test WatchConfig fields with custom values
        #[test]
        fn test_watch_config_all_custom_values() {
            let config = WatchConfig {
                patterns: vec!["custom".to_string()],
                ignore_patterns: vec!["ignore".to_string()],
                debounce_ms: 1000,
                clear_screen: false,
                run_on_start: false,
                watch_dirs: vec![PathBuf::from("/tmp")],
            };

            assert_eq!(config.patterns, vec!["custom".to_string()]);
            assert_eq!(config.ignore_patterns, vec!["ignore".to_string()]);
            assert_eq!(config.debounce_ms, 1000);
            assert!(!config.clear_screen);
            assert!(!config.run_on_start);
            assert_eq!(config.watch_dirs, vec![PathBuf::from("/tmp")]);
        }

        // Test WatchStats fields
        #[test]
        fn test_watch_stats_all_fields() {
            let mut stats = WatchStats {
                trigger_count: 10,
                change_count: 25,
                total_runtime: Duration::from_secs(60),
                last_trigger: Some(Instant::now()),
            };

            assert_eq!(stats.trigger_count, 10);
            assert_eq!(stats.change_count, 25);
            assert_eq!(stats.total_runtime.as_secs(), 60);
            assert!(stats.last_trigger.is_some());

            // Test record_trigger updates
            stats.record_trigger(5);
            assert_eq!(stats.trigger_count, 11);
            assert_eq!(stats.change_count, 30);
        }

        // Test glob_match_parts with segment that doesn't match
        #[test]
        fn test_glob_match_parts_first_segment_fail() {
            let pattern = vec!["foo", "bar"];
            let path = vec!["baz", "bar"];
            assert!(!WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test FileChange path field
        #[test]
        fn test_file_change_path_field() {
            let change = FileChange {
                path: PathBuf::from("/home/user/test.rs"),
                kind: FileChangeKind::Created,
                timestamp: Instant::now(),
            };
            assert_eq!(change.path, PathBuf::from("/home/user/test.rs"));
        }

        // Test FnWatchHandler with empty changes
        #[test]
        fn test_fn_watch_handler_empty_changes() {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            let called = Arc::new(AtomicBool::new(false));
            let called_clone = Arc::clone(&called);

            let handler = FnWatchHandler::new(move |changes| {
                called_clone.store(true, Ordering::SeqCst);
                assert!(changes.is_empty());
                Ok(())
            });

            let empty_changes: Vec<FileChange> = vec![];
            handler.on_change(&empty_changes).unwrap();
            assert!(called.load(Ordering::SeqCst));
        }

        // Test WatchHandler trait default implementations explicitly
        #[test]
        fn test_watch_handler_trait_defaults() {
            struct MinimalHandler;
            impl WatchHandler for MinimalHandler {
                fn on_change(&self, _: &[FileChange]) -> ProbarResult<()> {
                    Ok(())
                }
            }

            let handler = MinimalHandler;

            // These use the default implementations
            assert!(handler.on_start().is_ok());
            assert!(handler.on_stop().is_ok());
            assert!(handler.on_change(&[]).is_ok());
        }

        // Test matches_pattern when pattern matches but ignore also matches
        #[test]
        fn test_matches_pattern_ignore_vs_pattern_priority() {
            let config = WatchConfig {
                patterns: vec!["**/*.rs".to_string()],
                ignore_patterns: vec!["**/*.rs".to_string()], // Same pattern in ignore
                debounce_ms: 300,
                clear_screen: true,
                run_on_start: true,
                watch_dirs: vec![],
            };
            // Ignore patterns are checked first, so this should be ignored
            assert!(!config.matches_pattern(Path::new("src/main.rs")));
        }

        // Test glob_match_segment with only question marks
        #[test]
        fn test_glob_match_segment_only_questions() {
            assert!(WatchConfig::glob_match_segment("???", "abc"));
            assert!(!WatchConfig::glob_match_segment("???", "ab"));
            assert!(!WatchConfig::glob_match_segment("???", "abcd"));
        }

        // Test glob_match_segment with only stars
        #[test]
        fn test_glob_match_segment_only_stars() {
            assert!(WatchConfig::glob_match_segment("*", ""));
            assert!(WatchConfig::glob_match_segment("*", "anything"));
            assert!(WatchConfig::glob_match_segment("***", "test"));
        }

        // Test glob_match_parts matching exactly
        #[test]
        fn test_glob_match_parts_exact_match() {
            let pattern = vec!["src", "lib", "mod.rs"];
            let path = vec!["src", "lib", "mod.rs"];
            assert!(WatchConfig::glob_match_parts(&pattern, &path));
        }

        // Test config accessor returns reference
        #[test]
        fn test_file_watcher_config_reference() {
            let original_debounce = 500;
            let config = WatchConfig::new().with_debounce(original_debounce);
            let watcher = FileWatcher::new(config).unwrap();
            let config_ref = watcher.config();
            assert_eq!(config_ref.debounce_ms, original_debounce);
        }

        // Test WatchBuilder default impl
        #[test]
        fn test_watch_builder_default_impl() {
            let builder1 = WatchBuilder::default();
            let builder2 = WatchBuilder::new();
            // Both should produce configs with same defaults
            assert_eq!(builder1.build().debounce_ms, builder2.build().debounce_ms);
        }
    }
}
