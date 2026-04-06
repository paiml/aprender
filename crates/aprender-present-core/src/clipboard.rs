#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Clipboard API for copy, cut, and paste operations.
//!
//! This module provides:
//! - Async clipboard read/write
//! - Multiple data format support (text, HTML, image, custom)
//! - Clipboard change detection
//! - Cross-platform abstraction

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Clipboard data format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClipboardFormat {
    /// Plain text.
    Text,
    /// HTML content.
    Html,
    /// Rich text format.
    Rtf,
    /// Image data (PNG format).
    ImagePng,
    /// Image data (JPEG format).
    ImageJpeg,
    /// File list (paths).
    Files,
    /// Custom format with MIME type.
    Custom(String),
}

impl ClipboardFormat {
    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Text => "text/plain",
            Self::Html => "text/html",
            Self::Rtf => "text/rtf",
            Self::ImagePng => "image/png",
            Self::ImageJpeg => "image/jpeg",
            Self::Files => "application/x-file-list",
            Self::Custom(mime) => mime,
        }
    }

    /// Create from MIME type.
    pub fn from_mime(mime: &str) -> Self {
        match mime {
            "text/plain" => Self::Text,
            "text/html" => Self::Html,
            "text/rtf" => Self::Rtf,
            "image/png" => Self::ImagePng,
            "image/jpeg" => Self::ImageJpeg,
            "application/x-file-list" => Self::Files,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Check if this is a text format.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text | Self::Html | Self::Rtf)
    }

    /// Check if this is an image format.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::ImagePng | Self::ImageJpeg)
    }
}

/// Data stored in the clipboard.
#[derive(Debug, Clone, Default)]
pub struct ClipboardData {
    /// Data in various formats.
    formats: HashMap<ClipboardFormat, Vec<u8>>,
}

impl ClipboardData {
    /// Create empty clipboard data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create clipboard data with plain text.
    pub fn text(content: &str) -> Self {
        let mut data = Self::new();
        data.set_text(content);
        data
    }

    /// Create clipboard data with HTML.
    pub fn html(content: &str) -> Self {
        let mut data = Self::new();
        data.set_html(content);
        data
    }

    /// Set text content.
    pub fn set_text(&mut self, content: &str) {
        self.formats
            .insert(ClipboardFormat::Text, content.as_bytes().to_vec());
    }

    /// Get text content.
    pub fn get_text(&self) -> Option<String> {
        self.formats
            .get(&ClipboardFormat::Text)
            .and_then(|bytes| String::from_utf8(bytes.clone()).ok())
    }

    /// Set HTML content.
    pub fn set_html(&mut self, content: &str) {
        self.formats
            .insert(ClipboardFormat::Html, content.as_bytes().to_vec());
    }

    /// Get HTML content.
    pub fn get_html(&self) -> Option<String> {
        self.formats
            .get(&ClipboardFormat::Html)
            .and_then(|bytes| String::from_utf8(bytes.clone()).ok())
    }

    /// Set data for a specific format.
    pub fn set(&mut self, format: ClipboardFormat, data: Vec<u8>) {
        self.formats.insert(format, data);
    }

    /// Get data for a specific format.
    pub fn get(&self, format: &ClipboardFormat) -> Option<&[u8]> {
        self.formats.get(format).map(std::vec::Vec::as_slice)
    }

    /// Check if a format is available.
    pub fn has_format(&self, format: &ClipboardFormat) -> bool {
        self.formats.contains_key(format)
    }

    /// Get all available formats.
    pub fn formats(&self) -> impl Iterator<Item = &ClipboardFormat> {
        self.formats.keys()
    }

    /// Check if clipboard data is empty.
    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.formats.clear();
    }

    /// Get the number of formats.
    pub fn format_count(&self) -> usize {
        self.formats.len()
    }
}

/// Result of a clipboard operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipboardResult {
    /// Operation succeeded.
    Success,
    /// Clipboard is not available.
    Unavailable,
    /// Permission denied.
    PermissionDenied,
    /// Format not supported.
    UnsupportedFormat,
    /// Other error.
    Error(String),
}

impl ClipboardResult {
    /// Check if operation was successful.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Check if operation failed.
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

/// Clipboard operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardOperation {
    /// Copy operation.
    Copy,
    /// Cut operation.
    Cut,
    /// Paste operation.
    Paste,
}

/// Event triggered by clipboard changes.
#[derive(Debug, Clone)]
pub struct ClipboardEvent {
    /// Operation that occurred.
    pub operation: ClipboardOperation,
    /// Available formats.
    pub formats: Vec<ClipboardFormat>,
    /// Timestamp (monotonic counter).
    pub timestamp: u64,
}

impl ClipboardEvent {
    /// Create a new clipboard event.
    pub fn new(
        operation: ClipboardOperation,
        formats: Vec<ClipboardFormat>,
        timestamp: u64,
    ) -> Self {
        Self {
            operation,
            formats,
            timestamp,
        }
    }
}

/// Callback for clipboard changes.
pub type ClipboardCallback = Box<dyn Fn(&ClipboardEvent) + Send + Sync>;

/// Clipboard manager for handling copy/cut/paste operations.
pub struct Clipboard {
    /// Current clipboard content.
    data: Arc<RwLock<ClipboardData>>,
    /// Change listeners.
    listeners: Vec<ClipboardCallback>,
    /// Event counter.
    counter: u64,
    /// Whether clipboard is available.
    available: bool,
}

impl Clipboard {
    /// Create a new clipboard.
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(ClipboardData::new())),
            listeners: Vec::new(),
            counter: 0,
            available: true,
        }
    }

    /// Create an unavailable clipboard (for testing).
    pub fn unavailable() -> Self {
        Self {
            data: Arc::new(RwLock::new(ClipboardData::new())),
            listeners: Vec::new(),
            counter: 0,
            available: false,
        }
    }

    /// Check if clipboard is available.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Write data to the clipboard.
    pub fn write(&mut self, data: ClipboardData) -> ClipboardResult {
        if !self.available {
            return ClipboardResult::Unavailable;
        }

        let formats: Vec<ClipboardFormat> = data.formats().cloned().collect();

        if let Ok(mut clipboard) = self.data.write() {
            *clipboard = data;
        } else {
            return ClipboardResult::Error("Lock error".to_string());
        }

        self.counter += 1;
        self.notify(ClipboardOperation::Copy, formats);

        ClipboardResult::Success
    }

    /// Write text to the clipboard.
    pub fn write_text(&mut self, text: &str) -> ClipboardResult {
        self.write(ClipboardData::text(text))
    }

    /// Write HTML to the clipboard.
    pub fn write_html(&mut self, html: &str) -> ClipboardResult {
        self.write(ClipboardData::html(html))
    }

    /// Read all data from the clipboard.
    pub fn read(&self) -> Result<ClipboardData, ClipboardResult> {
        if !self.available {
            return Err(ClipboardResult::Unavailable);
        }

        self.data
            .read()
            .map(|data| data.clone())
            .map_err(|_| ClipboardResult::Error("Lock error".to_string()))
    }

    /// Read text from the clipboard.
    pub fn read_text(&self) -> Result<Option<String>, ClipboardResult> {
        if !self.available {
            return Err(ClipboardResult::Unavailable);
        }

        self.data
            .read()
            .map(|data| data.get_text())
            .map_err(|_| ClipboardResult::Error("Lock error".to_string()))
    }

    /// Read HTML from the clipboard.
    pub fn read_html(&self) -> Result<Option<String>, ClipboardResult> {
        if !self.available {
            return Err(ClipboardResult::Unavailable);
        }

        self.data
            .read()
            .map(|data| data.get_html())
            .map_err(|_| ClipboardResult::Error("Lock error".to_string()))
    }

    /// Check if clipboard has a specific format.
    pub fn has_format(&self, format: &ClipboardFormat) -> bool {
        self.data
            .read()
            .map(|data| data.has_format(format))
            .unwrap_or(false)
    }

    /// Get available formats.
    pub fn available_formats(&self) -> Vec<ClipboardFormat> {
        self.data
            .read()
            .map(|data| data.formats().cloned().collect())
            .unwrap_or_default()
    }

    /// Clear the clipboard.
    pub fn clear(&mut self) -> ClipboardResult {
        if !self.available {
            return ClipboardResult::Unavailable;
        }

        if let Ok(mut data) = self.data.write() {
            data.clear();
            ClipboardResult::Success
        } else {
            ClipboardResult::Error("Lock error".to_string())
        }
    }

    /// Add a listener for clipboard changes.
    pub fn on_change(&mut self, callback: ClipboardCallback) {
        self.listeners.push(callback);
    }

    /// Get listener count.
    pub fn listener_count(&self) -> usize {
        self.listeners.len()
    }

    /// Notify listeners of a change.
    fn notify(&self, operation: ClipboardOperation, formats: Vec<ClipboardFormat>) {
        let event = ClipboardEvent::new(operation, formats, self.counter);
        for listener in &self.listeners {
            listener(&event);
        }
    }

    /// Simulate a cut operation (copies and signals cut).
    pub fn cut(&mut self, data: ClipboardData) -> ClipboardResult {
        if !self.available {
            return ClipboardResult::Unavailable;
        }

        let formats: Vec<ClipboardFormat> = data.formats().cloned().collect();

        if let Ok(mut clipboard) = self.data.write() {
            *clipboard = data;
        } else {
            return ClipboardResult::Error("Lock error".to_string());
        }

        self.counter += 1;
        self.notify(ClipboardOperation::Cut, formats);

        ClipboardResult::Success
    }

    /// Signal that a paste occurred.
    pub fn signal_paste(&mut self) {
        let formats = self.available_formats();
        self.counter += 1;
        self.notify(ClipboardOperation::Paste, formats);
    }
}

impl Default for Clipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Clipboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Clipboard")
            .field("available", &self.available)
            .field("counter", &self.counter)
            .field("listener_count", &self.listeners.len())
            .finish()
    }
}

/// Clipboard history for undo support.
#[derive(Debug, Default)]
pub struct ClipboardHistory {
    /// History entries.
    entries: Vec<ClipboardData>,
    /// Maximum history size.
    max_size: usize,
    /// Current index in history.
    current: usize,
}

impl ClipboardHistory {
    /// Create a new clipboard history.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            current: 0,
        }
    }

    /// Add an entry to the history.
    pub fn push(&mut self, data: ClipboardData) {
        // Trim history if we're not at the end
        if self.current < self.entries.len() {
            self.entries.truncate(self.current);
        }

        self.entries.push(data);

        // Trim to max size
        while self.entries.len() > self.max_size {
            self.entries.remove(0);
        }

        self.current = self.entries.len();
    }

    /// Get the current entry.
    pub fn current(&self) -> Option<&ClipboardData> {
        if self.current > 0 && self.current <= self.entries.len() {
            self.entries.get(self.current - 1)
        } else {
            None
        }
    }

    /// Go to previous entry.
    pub fn previous(&mut self) -> Option<&ClipboardData> {
        if self.current > 1 {
            self.current -= 1;
            self.entries.get(self.current - 1)
        } else {
            None
        }
    }

    /// Go to next entry.
    pub fn next(&mut self) -> Option<&ClipboardData> {
        if self.current < self.entries.len() {
            self.current += 1;
            self.entries.get(self.current - 1)
        } else {
            None
        }
    }

    /// Get entry at index.
    pub fn get(&self, index: usize) -> Option<&ClipboardData> {
        self.entries.get(index)
    }

    /// Get history length.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current = 0;
    }

    /// Get current index (1-based).
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Check if can go back.
    pub fn can_go_back(&self) -> bool {
        self.current > 1
    }

    /// Check if can go forward.
    pub fn can_go_forward(&self) -> bool {
        self.current < self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ClipboardFormat tests
    #[test]
    fn test_clipboard_format_mime_type() {
        assert_eq!(ClipboardFormat::Text.mime_type(), "text/plain");
        assert_eq!(ClipboardFormat::Html.mime_type(), "text/html");
        assert_eq!(ClipboardFormat::ImagePng.mime_type(), "image/png");
        assert_eq!(
            ClipboardFormat::Custom("application/json".to_string()).mime_type(),
            "application/json"
        );
    }

    #[test]
    fn test_clipboard_format_from_mime() {
        assert_eq!(
            ClipboardFormat::from_mime("text/plain"),
            ClipboardFormat::Text
        );
        assert_eq!(
            ClipboardFormat::from_mime("text/html"),
            ClipboardFormat::Html
        );
        assert_eq!(
            ClipboardFormat::from_mime("image/png"),
            ClipboardFormat::ImagePng
        );
        assert_eq!(
            ClipboardFormat::from_mime("application/json"),
            ClipboardFormat::Custom("application/json".to_string())
        );
    }

    #[test]
    fn test_clipboard_format_is_text() {
        assert!(ClipboardFormat::Text.is_text());
        assert!(ClipboardFormat::Html.is_text());
        assert!(ClipboardFormat::Rtf.is_text());
        assert!(!ClipboardFormat::ImagePng.is_text());
    }

    #[test]
    fn test_clipboard_format_is_image() {
        assert!(ClipboardFormat::ImagePng.is_image());
        assert!(ClipboardFormat::ImageJpeg.is_image());
        assert!(!ClipboardFormat::Text.is_image());
    }

    // ClipboardData tests
    #[test]
    fn test_clipboard_data_new() {
        let data = ClipboardData::new();
        assert!(data.is_empty());
        assert_eq!(data.format_count(), 0);
    }

    #[test]
    fn test_clipboard_data_text() {
        let data = ClipboardData::text("Hello");
        assert!(!data.is_empty());
        assert!(data.has_format(&ClipboardFormat::Text));
        assert_eq!(data.get_text(), Some("Hello".to_string()));
    }

    #[test]
    fn test_clipboard_data_html() {
        let data = ClipboardData::html("<b>Bold</b>");
        assert!(data.has_format(&ClipboardFormat::Html));
        assert_eq!(data.get_html(), Some("<b>Bold</b>".to_string()));
    }

    #[test]
    fn test_clipboard_data_set_get() {
        let mut data = ClipboardData::new();
        data.set(ClipboardFormat::Text, b"test".to_vec());

        assert!(data.has_format(&ClipboardFormat::Text));
        assert_eq!(data.get(&ClipboardFormat::Text), Some(b"test".as_slice()));
    }

    #[test]
    fn test_clipboard_data_clear() {
        let mut data = ClipboardData::text("test");
        assert!(!data.is_empty());

        data.clear();
        assert!(data.is_empty());
    }

    #[test]
    fn test_clipboard_data_formats() {
        let mut data = ClipboardData::new();
        data.set_text("text");
        data.set_html("<p>html</p>");

        assert_eq!(data.formats().count(), 2);
    }

    // ClipboardResult tests
    #[test]
    fn test_clipboard_result_is_success() {
        assert!(ClipboardResult::Success.is_success());
        assert!(!ClipboardResult::Unavailable.is_success());
        assert!(!ClipboardResult::PermissionDenied.is_success());
    }

    #[test]
    fn test_clipboard_result_is_error() {
        assert!(!ClipboardResult::Success.is_error());
        assert!(ClipboardResult::Unavailable.is_error());
        assert!(ClipboardResult::Error("test".to_string()).is_error());
    }

    // ClipboardEvent tests
    #[test]
    fn test_clipboard_event_new() {
        let event = ClipboardEvent::new(ClipboardOperation::Copy, vec![ClipboardFormat::Text], 42);
        assert_eq!(event.operation, ClipboardOperation::Copy);
        assert_eq!(event.formats.len(), 1);
        assert_eq!(event.timestamp, 42);
    }

    // Clipboard tests
    #[test]
    fn test_clipboard_new() {
        let clipboard = Clipboard::new();
        assert!(clipboard.is_available());
        assert_eq!(clipboard.listener_count(), 0);
    }

    #[test]
    fn test_clipboard_unavailable() {
        let mut clipboard = Clipboard::unavailable();
        assert!(!clipboard.is_available());

        let result = clipboard.write_text("test");
        assert_eq!(result, ClipboardResult::Unavailable);

        let result = clipboard.read();
        assert!(result.is_err());
    }

    #[test]
    fn test_clipboard_write_text() {
        let mut clipboard = Clipboard::new();
        let result = clipboard.write_text("Hello");

        assert!(result.is_success());
        assert!(clipboard.has_format(&ClipboardFormat::Text));
    }

    #[test]
    fn test_clipboard_read_text() {
        let mut clipboard = Clipboard::new();
        clipboard.write_text("Hello");

        let text = clipboard.read_text().unwrap();
        assert_eq!(text, Some("Hello".to_string()));
    }

    #[test]
    fn test_clipboard_write_html() {
        let mut clipboard = Clipboard::new();
        let result = clipboard.write_html("<b>Bold</b>");

        assert!(result.is_success());
        assert!(clipboard.has_format(&ClipboardFormat::Html));
    }

    #[test]
    fn test_clipboard_read_html() {
        let mut clipboard = Clipboard::new();
        clipboard.write_html("<p>Test</p>");

        let html = clipboard.read_html().unwrap();
        assert_eq!(html, Some("<p>Test</p>".to_string()));
    }

    #[test]
    fn test_clipboard_read() {
        let mut clipboard = Clipboard::new();
        clipboard.write_text("test");

        let data = clipboard.read().unwrap();
        assert_eq!(data.get_text(), Some("test".to_string()));
    }

    #[test]
    fn test_clipboard_clear() {
        let mut clipboard = Clipboard::new();
        clipboard.write_text("test");
        assert!(clipboard.has_format(&ClipboardFormat::Text));

        let result = clipboard.clear();
        assert!(result.is_success());
        assert!(!clipboard.has_format(&ClipboardFormat::Text));
    }

    #[test]
    fn test_clipboard_available_formats() {
        let mut clipboard = Clipboard::new();

        let mut data = ClipboardData::new();
        data.set_text("text");
        data.set_html("html");
        clipboard.write(data);

        let formats = clipboard.available_formats();
        assert_eq!(formats.len(), 2);
    }

    #[test]
    fn test_clipboard_on_change() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut clipboard = Clipboard::new();
        clipboard.on_change(Box::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        clipboard.write_text("test");
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        clipboard.write_text("test2");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_clipboard_cut() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut clipboard = Clipboard::new();
        clipboard.on_change(Box::new(move |event| {
            if event.operation == ClipboardOperation::Cut {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        }));

        clipboard.cut(ClipboardData::text("cut text"));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(clipboard.read_text().unwrap(), Some("cut text".to_string()));
    }

    #[test]
    fn test_clipboard_signal_paste() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut clipboard = Clipboard::new();
        clipboard.write_text("test");

        clipboard.on_change(Box::new(move |event| {
            if event.operation == ClipboardOperation::Paste {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        }));

        clipboard.signal_paste();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ClipboardHistory tests
    #[test]
    fn test_history_new() {
        let history = ClipboardHistory::new(10);
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_history_push() {
        let mut history = ClipboardHistory::new(10);
        history.push(ClipboardData::text("first"));
        history.push(ClipboardData::text("second"));

        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_history_current() {
        let mut history = ClipboardHistory::new(10);
        assert!(history.current().is_none());

        history.push(ClipboardData::text("first"));
        assert_eq!(
            history.current().unwrap().get_text(),
            Some("first".to_string())
        );

        history.push(ClipboardData::text("second"));
        assert_eq!(
            history.current().unwrap().get_text(),
            Some("second".to_string())
        );
    }

    #[test]
    fn test_history_previous_next() {
        let mut history = ClipboardHistory::new(10);
        history.push(ClipboardData::text("first"));
        history.push(ClipboardData::text("second"));
        history.push(ClipboardData::text("third"));

        // At "third"
        assert_eq!(
            history.current().unwrap().get_text(),
            Some("third".to_string())
        );

        // Go back to "second"
        let prev = history.previous();
        assert_eq!(prev.unwrap().get_text(), Some("second".to_string()));

        // Go back to "first"
        let prev = history.previous();
        assert_eq!(prev.unwrap().get_text(), Some("first".to_string()));

        // Can't go back further
        assert!(history.previous().is_none());

        // Go forward to "second"
        let next = history.next();
        assert_eq!(next.unwrap().get_text(), Some("second".to_string()));

        // Go forward to "third"
        let next = history.next();
        assert_eq!(next.unwrap().get_text(), Some("third".to_string()));

        // Can't go forward further
        assert!(history.next().is_none());
    }

    #[test]
    fn test_history_max_size() {
        let mut history = ClipboardHistory::new(3);

        history.push(ClipboardData::text("1"));
        history.push(ClipboardData::text("2"));
        history.push(ClipboardData::text("3"));
        history.push(ClipboardData::text("4"));

        assert_eq!(history.len(), 3);
        assert_eq!(history.get(0).unwrap().get_text(), Some("2".to_string()));
    }

    #[test]
    fn test_history_clear() {
        let mut history = ClipboardHistory::new(10);
        history.push(ClipboardData::text("test"));

        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.current_index(), 0);
    }

    #[test]
    fn test_history_can_navigate() {
        let mut history = ClipboardHistory::new(10);
        assert!(!history.can_go_back());
        assert!(!history.can_go_forward());

        history.push(ClipboardData::text("first"));
        assert!(!history.can_go_back());
        assert!(!history.can_go_forward());

        history.push(ClipboardData::text("second"));
        assert!(history.can_go_back());
        assert!(!history.can_go_forward());

        history.previous();
        assert!(!history.can_go_back());
        assert!(history.can_go_forward());
    }

    #[test]
    fn test_history_get() {
        let mut history = ClipboardHistory::new(10);
        history.push(ClipboardData::text("first"));
        history.push(ClipboardData::text("second"));

        assert_eq!(
            history.get(0).unwrap().get_text(),
            Some("first".to_string())
        );
        assert_eq!(
            history.get(1).unwrap().get_text(),
            Some("second".to_string())
        );
        assert!(history.get(2).is_none());
    }

    #[test]
    fn test_history_truncate_on_push() {
        let mut history = ClipboardHistory::new(10);
        history.push(ClipboardData::text("first"));
        history.push(ClipboardData::text("second"));
        history.push(ClipboardData::text("third"));

        // Go back two steps
        history.previous();
        history.previous();
        assert_eq!(history.current_index(), 1);

        // Push new entry - should truncate "second" and "third"
        history.push(ClipboardData::text("new"));
        assert_eq!(history.len(), 2);
        assert_eq!(
            history.get(0).unwrap().get_text(),
            Some("first".to_string())
        );
        assert_eq!(history.get(1).unwrap().get_text(), Some("new".to_string()));
    }
}
