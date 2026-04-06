//! Calculation history with 100% test coverage
//!
//! Probar: Visual feedback (Visualization) - Track and display calculation history

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single entry in the calculation history
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The expression that was evaluated
    pub expression: String,
    /// The result of the calculation
    pub result: f64,
    /// Timestamp of when the calculation was performed (Unix epoch millis)
    pub timestamp: u64,
}

impl HistoryEntry {
    /// Creates a new history entry
    #[must_use]
    pub fn new(expression: String, result: f64) -> Self {
        Self {
            expression,
            result,
            timestamp: Self::current_timestamp(),
        }
    }

    /// Creates a history entry with a specific timestamp (for testing)
    #[must_use]
    pub fn with_timestamp(expression: String, result: f64, timestamp: u64) -> Self {
        Self {
            expression,
            result,
            timestamp,
        }
    }

    /// Returns the current timestamp in milliseconds
    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Returns a formatted display string
    #[must_use]
    pub fn display(&self) -> String {
        format!("{} = {}", self.expression, self.result)
    }
}

/// Calculator history manager
///
/// Tracks past calculations for display and recall.
/// Implements a bounded queue to prevent unbounded memory growth.
#[derive(Debug, Clone)]
pub struct History {
    /// The history entries
    entries: VecDeque<HistoryEntry>,
    /// Maximum number of entries to keep
    max_entries: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    /// Default maximum history size
    pub const DEFAULT_MAX_ENTRIES: usize = 100;

    /// Creates a new history with default capacity
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: Self::DEFAULT_MAX_ENTRIES,
        }
    }

    /// Creates a history with custom maximum size
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Adds an entry to the history
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Adds a calculation result to the history
    pub fn record(&mut self, expression: &str, result: f64) {
        let entry = HistoryEntry::new(expression.to_string(), result);
        self.push(entry);
    }

    /// Returns the number of entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the history is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the maximum number of entries
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Clears all history entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns an iterator over the entries (oldest first)
    pub fn iter(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter()
    }

    /// Returns an iterator over the entries (newest first)
    pub fn iter_rev(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter().rev()
    }

    /// Returns the most recent entry
    #[must_use]
    pub fn last(&self) -> Option<&HistoryEntry> {
        self.entries.back()
    }

    /// Returns the oldest entry
    #[must_use]
    pub fn first(&self) -> Option<&HistoryEntry> {
        self.entries.front()
    }

    /// Returns the entry at the given index (0 = oldest)
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&HistoryEntry> {
        self.entries.get(index)
    }

    /// Returns the last n entries (newest first)
    #[must_use]
    pub fn last_n(&self, n: usize) -> Vec<&HistoryEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Serializes the history to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.entries.iter().collect::<Vec<_>>())
    }

    /// Deserializes history from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let entries: Vec<HistoryEntry> = serde_json::from_str(json)?;
        let mut history = Self::new();
        for entry in entries {
            history.push(entry);
        }
        Ok(history)
    }

    /// Exports history to a formatted string
    #[must_use]
    pub fn export_formatted(&self) -> String {
        self.entries
            .iter()
            .map(HistoryEntry::display)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // ===== HistoryEntry tests =====

    #[test]
    fn test_history_entry_new() {
        let entry = HistoryEntry::new("2 + 2".into(), 4.0);
        assert_eq!(entry.expression, "2 + 2");
        assert_eq!(entry.result, 4.0);
        assert!(entry.timestamp > 0);
    }

    #[test]
    fn test_history_entry_with_timestamp() {
        let entry = HistoryEntry::with_timestamp("3 * 3".into(), 9.0, 1234567890);
        assert_eq!(entry.expression, "3 * 3");
        assert_eq!(entry.result, 9.0);
        assert_eq!(entry.timestamp, 1234567890);
    }

    #[test]
    fn test_history_entry_display() {
        let entry = HistoryEntry::new("5 + 3".into(), 8.0);
        assert_eq!(entry.display(), "5 + 3 = 8");
    }

    #[test]
    fn test_history_entry_clone() {
        let entry = HistoryEntry::new("1 + 1".into(), 2.0);
        let cloned = entry.clone();
        assert_eq!(entry, cloned);
    }

    #[test]
    fn test_history_entry_serialize() {
        let entry = HistoryEntry::with_timestamp("2 ^ 3".into(), 8.0, 1000);
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"expression\":\"2 ^ 3\""));
        assert!(json.contains("\"result\":8.0"));
    }

    #[test]
    fn test_history_entry_deserialize() {
        let json = r#"{"expression":"10 / 2","result":5.0,"timestamp":2000}"#;
        let entry: HistoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.expression, "10 / 2");
        assert_eq!(entry.result, 5.0);
        assert_eq!(entry.timestamp, 2000);
    }

    // ===== History tests =====

    #[test]
    fn test_history_new() {
        let history = History::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.max_entries(), History::DEFAULT_MAX_ENTRIES);
    }

    #[test]
    fn test_history_default() {
        let history = History::default();
        assert!(history.is_empty());
    }

    #[test]
    fn test_history_with_capacity() {
        let history = History::with_capacity(50);
        assert_eq!(history.max_entries(), 50);
    }

    #[test]
    fn test_history_push() {
        let mut history = History::new();
        let entry = HistoryEntry::new("1 + 1".into(), 2.0);
        history.push(entry);
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_history_record() {
        let mut history = History::new();
        history.record("3 + 4", 7.0);
        assert_eq!(history.len(), 1);
        assert_eq!(history.last().unwrap().expression, "3 + 4");
        assert_eq!(history.last().unwrap().result, 7.0);
    }

    #[test]
    fn test_history_max_entries_enforcement() {
        let mut history = History::with_capacity(3);
        history.record("1", 1.0);
        history.record("2", 2.0);
        history.record("3", 3.0);
        history.record("4", 4.0);

        assert_eq!(history.len(), 3);
        // First entry (1) should be removed
        assert_eq!(history.first().unwrap().result, 2.0);
        assert_eq!(history.last().unwrap().result, 4.0);
    }

    #[test]
    fn test_history_clear() {
        let mut history = History::new();
        history.record("1", 1.0);
        history.record("2", 2.0);
        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn test_history_iter() {
        let mut history = History::new();
        history.record("a", 1.0);
        history.record("b", 2.0);
        history.record("c", 3.0);

        let results: Vec<f64> = history.iter().map(|e| e.result).collect();
        assert_eq!(results, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_history_iter_rev() {
        let mut history = History::new();
        history.record("a", 1.0);
        history.record("b", 2.0);
        history.record("c", 3.0);

        let results: Vec<f64> = history.iter_rev().map(|e| e.result).collect();
        assert_eq!(results, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_history_last() {
        let mut history = History::new();
        assert!(history.last().is_none());

        history.record("x", 10.0);
        assert_eq!(history.last().unwrap().result, 10.0);

        history.record("y", 20.0);
        assert_eq!(history.last().unwrap().result, 20.0);
    }

    #[test]
    fn test_history_first() {
        let mut history = History::new();
        assert!(history.first().is_none());

        history.record("x", 10.0);
        assert_eq!(history.first().unwrap().result, 10.0);

        history.record("y", 20.0);
        assert_eq!(history.first().unwrap().result, 10.0);
    }

    #[test]
    fn test_history_get() {
        let mut history = History::new();
        history.record("a", 1.0);
        history.record("b", 2.0);
        history.record("c", 3.0);

        assert_eq!(history.get(0).unwrap().result, 1.0);
        assert_eq!(history.get(1).unwrap().result, 2.0);
        assert_eq!(history.get(2).unwrap().result, 3.0);
        assert!(history.get(3).is_none());
    }

    #[test]
    fn test_history_last_n() {
        let mut history = History::new();
        history.record("a", 1.0);
        history.record("b", 2.0);
        history.record("c", 3.0);
        history.record("d", 4.0);

        let last_2: Vec<f64> = history.last_n(2).iter().map(|e| e.result).collect();
        assert_eq!(last_2, vec![4.0, 3.0]);

        let last_10 = history.last_n(10);
        assert_eq!(last_10.len(), 4);
    }

    #[test]
    fn test_history_to_json() {
        let mut history = History::new();
        history.push(HistoryEntry::with_timestamp("1+1".into(), 2.0, 1000));
        history.push(HistoryEntry::with_timestamp("2+2".into(), 4.0, 2000));

        let json = history.to_json().unwrap();
        assert!(json.contains("1+1"));
        assert!(json.contains("2+2"));
    }

    #[test]
    fn test_history_from_json() {
        let json = r#"[
            {"expression":"a","result":1.0,"timestamp":1000},
            {"expression":"b","result":2.0,"timestamp":2000}
        ]"#;

        let history = History::from_json(json).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history.first().unwrap().expression, "a");
        assert_eq!(history.last().unwrap().expression, "b");
    }

    #[test]
    fn test_history_from_json_invalid() {
        let result = History::from_json("invalid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_history_export_formatted() {
        let mut history = History::new();
        history.push(HistoryEntry::with_timestamp("1+1".into(), 2.0, 1000));
        history.push(HistoryEntry::with_timestamp("2*3".into(), 6.0, 2000));

        let formatted = history.export_formatted();
        assert_eq!(formatted, "1+1 = 2\n2*3 = 6");
    }

    #[test]
    fn test_history_export_formatted_empty() {
        let history = History::new();
        let formatted = history.export_formatted();
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_history_clone() {
        let mut history = History::new();
        history.record("test", 42.0);

        let cloned = history.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.last().unwrap().result, 42.0);
    }

    #[test]
    fn test_history_round_trip_json() {
        let mut original = History::new();
        original.push(HistoryEntry::with_timestamp("x".into(), 10.0, 100));
        original.push(HistoryEntry::with_timestamp("y".into(), 20.0, 200));

        let json = original.to_json().unwrap();
        let restored = History::from_json(&json).unwrap();

        assert_eq!(original.len(), restored.len());
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert_eq!(orig, rest);
        }
    }
}
