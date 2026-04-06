//! CRDT (Conflict-free Replicated Data Types) for federated metrics.

use std::collections::{HashMap, HashSet};

/// G-Counter CRDT for monotonic counters
#[derive(Debug, Clone, Default)]
pub struct GCounter {
    /// Per-host counts
    counts: HashMap<String, u64>,
}

impl GCounter {
    /// Create a new G-Counter
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment counter for a host
    pub fn increment(&mut self, host_id: &str, amount: u64) {
        *self.counts.entry(host_id.to_string()).or_insert(0) += amount;
    }

    /// Get total count across all hosts
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another G-Counter (take max per host)
    pub fn merge(&mut self, other: &GCounter) {
        for (host, count) in &other.counts {
            let entry = self.counts.entry(host.clone()).or_insert(0);
            *entry = (*entry).max(*count);
        }
    }

    /// Get count for a specific host
    pub fn host_count(&self, host_id: &str) -> u64 {
        self.counts.get(host_id).copied().unwrap_or(0)
    }
}

/// LWW-Register CRDT for last-writer-wins values
#[derive(Debug, Clone)]
pub struct LwwRegister<T: Clone> {
    /// Current value
    value: T,
    /// Timestamp of last write
    timestamp: u64,
    /// Host that performed last write
    writer: String,
}

impl<T: Clone + Default> Default for LwwRegister<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            timestamp: 0,
            writer: String::new(),
        }
    }
}

impl<T: Clone> LwwRegister<T> {
    /// Create a new register with initial value
    pub fn new(value: T, timestamp: u64, writer: impl Into<String>) -> Self {
        Self {
            value,
            timestamp,
            writer: writer.into(),
        }
    }

    /// Update value if timestamp is newer
    pub fn update(&mut self, value: T, timestamp: u64, writer: impl Into<String>) {
        if timestamp > self.timestamp {
            self.value = value;
            self.timestamp = timestamp;
            self.writer = writer.into();
        }
    }

    /// Get current value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Merge with another register (keep newer)
    pub fn merge(&mut self, other: &LwwRegister<T>) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.writer = other.writer.clone();
        }
    }
}

/// OR-Set CRDT for add/remove sets
#[derive(Debug, Clone)]
pub struct OrSet<T: Clone + Eq + std::hash::Hash> {
    /// Elements with their unique tags
    elements: HashMap<T, HashSet<String>>,
    /// Tombstones for removed elements
    tombstones: HashMap<T, HashSet<String>>,
}

impl<T: Clone + Eq + std::hash::Hash> Default for OrSet<T> {
    fn default() -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashMap::new(),
        }
    }
}

impl<T: Clone + Eq + std::hash::Hash> OrSet<T> {
    /// Create a new OR-Set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an element with a unique tag
    pub fn add(&mut self, element: T, tag: String) {
        self.elements.entry(element).or_default().insert(tag);
    }

    /// Remove an element (tombstone all tags)
    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.get(element) {
            let tombstone_entry = self.tombstones.entry(element.clone()).or_default();
            for tag in tags {
                tombstone_entry.insert(tag.clone());
            }
        }
    }

    /// Check if element is in set
    pub fn contains(&self, element: &T) -> bool {
        if let Some(tags) = self.elements.get(element) {
            let tombstones = self.tombstones.get(element);
            tags.iter()
                .any(|tag| tombstones.map_or(true, |ts| !ts.contains(tag)))
        } else {
            false
        }
    }

    /// Get all active elements
    pub fn elements(&self) -> Vec<&T> {
        self.elements.keys().filter(|e| self.contains(e)).collect()
    }

    /// Merge with another OR-Set
    pub fn merge(&mut self, other: &OrSet<T>) {
        // Merge elements
        for (elem, tags) in &other.elements {
            let entry = self.elements.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
        // Merge tombstones
        for (elem, tags) in &other.tombstones {
            let entry = self.tombstones.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
    }
}
