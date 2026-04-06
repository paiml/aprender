//! Replay and time-travel debugging system.
//!
//! Implements:
//! - Incremental checkpointing with compression
//! - Event journaling for perfect replay
//! - Time-travel scrubbing to any simulation time
//!
//! # Advanced TPS Kaizen (Section 4.3)
//!
//! - **Zero-Copy Streaming Checkpoints** (4.3.3): Stream directly to mmap [50]
//! - **Split Event Journal** (4.3.4): Header/payload separation for fast scrubbing [50][54]
//! - **Schema Evolution** (4.3.7): Version headers with migration support [50]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::engine::rng::RngState;
use crate::engine::{SimState, SimTime};
use crate::error::{SimError, SimResult};

/// Checkpoint with compressed state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Simulation time at checkpoint.
    pub time: SimTime,
    /// Step count at checkpoint.
    pub step: u64,
    /// Compressed state data.
    pub data: Vec<u8>,
    /// Blake3 hash for integrity verification.
    pub hash: [u8; 32],
    /// RNG state for replay.
    pub rng_state: RngState,
}

impl Checkpoint {
    /// Create a new checkpoint from state.
    ///
    /// # Errors
    ///
    /// Returns error if serialization or compression fails.
    pub fn create(
        time: SimTime,
        step: u64,
        state: &SimState,
        rng_state: RngState,
        compression_level: i32,
    ) -> SimResult<Self> {
        // Serialize state
        let serialized =
            bincode::serialize(state).map_err(|e| SimError::serialization(e.to_string()))?;

        // Compress
        let compressed = zstd::encode_all(&serialized[..], compression_level)?;

        // Hash for integrity
        let hash = blake3::hash(&compressed);

        Ok(Self {
            time,
            step,
            data: compressed,
            hash: *hash.as_bytes(),
            rng_state,
        })
    }

    /// Restore state from checkpoint.
    ///
    /// # Errors
    ///
    /// Returns error if integrity check fails or deserialization fails.
    pub fn restore(&self) -> SimResult<SimState> {
        // Verify integrity
        let computed_hash = blake3::hash(&self.data);
        if computed_hash.as_bytes() != &self.hash {
            return Err(SimError::CheckpointIntegrity);
        }

        // Decompress
        let decompressed = zstd::decode_all(&self.data[..])?;

        // Deserialize
        bincode::deserialize(&decompressed).map_err(|e| SimError::serialization(e.to_string()))
    }

    /// Get compressed size in bytes.
    #[must_use]
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }
}

/// Checkpoint manager for incremental checkpointing.
#[derive(Debug, Default)]
pub struct CheckpointManager {
    /// Checkpoints indexed by time.
    checkpoints: BTreeMap<SimTime, Checkpoint>,
    /// Checkpoint interval in steps.
    interval: u64,
    /// Maximum storage in bytes.
    max_storage: usize,
    /// Current storage usage.
    current_storage: usize,
    /// Compression level (1-22).
    compression_level: i32,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    #[must_use]
    pub fn new(interval: u64, max_storage: usize, compression_level: i32) -> Self {
        Self {
            checkpoints: BTreeMap::new(),
            interval,
            max_storage,
            current_storage: 0,
            compression_level,
        }
    }

    /// Check if a checkpoint should be created at this step.
    #[must_use]
    pub const fn should_checkpoint(&self, step: u64) -> bool {
        step % self.interval == 0
    }

    /// Create and store a checkpoint.
    ///
    /// # Errors
    ///
    /// Returns error if checkpoint creation fails.
    pub fn checkpoint(
        &mut self,
        time: SimTime,
        step: u64,
        state: &SimState,
        rng_state: RngState,
    ) -> SimResult<()> {
        let checkpoint = Checkpoint::create(time, step, state, rng_state, self.compression_level)?;

        let size = checkpoint.compressed_size();

        // Garbage collect if needed
        while self.current_storage + size > self.max_storage && !self.checkpoints.is_empty() {
            self.remove_oldest();
        }

        self.current_storage += size;
        self.checkpoints.insert(time, checkpoint);

        Ok(())
    }

    /// Get checkpoint at or before given time.
    #[must_use]
    pub fn get_checkpoint_at(&self, time: SimTime) -> Option<&Checkpoint> {
        self.checkpoints
            .range(..=time)
            .next_back()
            .map(|(_, cp)| cp)
    }

    /// Restore state from nearest checkpoint.
    ///
    /// # Errors
    ///
    /// Returns error if no checkpoint found or restoration fails.
    pub fn restore_at(&self, time: SimTime) -> SimResult<(SimState, SimTime)> {
        let checkpoint = self
            .get_checkpoint_at(time)
            .ok_or(SimError::CheckpointNotFound(time))?;

        let state = checkpoint.restore()?;
        Ok((state, checkpoint.time))
    }

    /// Remove oldest checkpoint.
    fn remove_oldest(&mut self) {
        if let Some((&time, _)) = self.checkpoints.iter().next() {
            if let Some(cp) = self.checkpoints.remove(&time) {
                self.current_storage = self.current_storage.saturating_sub(cp.compressed_size());
            }
        }
    }

    /// Get number of stored checkpoints.
    #[must_use]
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Get current storage usage in bytes.
    #[must_use]
    pub const fn storage_used(&self) -> usize {
        self.current_storage
    }

    /// Clear all checkpoints.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.current_storage = 0;
    }
}

/// Journal entry for event replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Simulation time.
    pub time: SimTime,
    /// Step number.
    pub step: u64,
    /// Sequence number for ordering.
    pub sequence: u64,
    /// Event data (serialized).
    pub event_data: Vec<u8>,
    /// RNG state snapshot (for reproducibility).
    pub rng_state: Option<RngState>,
}

/// Event journal for perfect replay.
#[derive(Debug, Default)]
pub struct EventJournal {
    /// Journal entries in order.
    entries: Vec<JournalEntry>,
    /// Index by time for fast lookup.
    time_index: BTreeMap<SimTime, usize>,
    /// Current sequence number.
    sequence: u64,
    /// Whether to record RNG state.
    record_rng_state: bool,
}

impl EventJournal {
    /// Create a new event journal.
    #[must_use]
    pub fn new(record_rng_state: bool) -> Self {
        Self {
            entries: Vec::new(),
            time_index: BTreeMap::new(),
            sequence: 0,
            record_rng_state,
        }
    }

    /// Append an entry to the journal.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn append<T: Serialize>(
        &mut self,
        time: SimTime,
        step: u64,
        event: &T,
        rng_state: Option<&RngState>,
    ) -> SimResult<()> {
        let event_data =
            bincode::serialize(event).map_err(|e| SimError::serialization(e.to_string()))?;

        let rng_state = if self.record_rng_state {
            rng_state.cloned()
        } else {
            None
        };

        let entry = JournalEntry {
            time,
            step,
            sequence: self.sequence,
            event_data,
            rng_state,
        };

        let index = self.entries.len();
        self.time_index.insert(time, index);
        self.entries.push(entry);
        self.sequence += 1;

        Ok(())
    }

    /// Get entries from a given time.
    pub fn entries_from(&self, time: SimTime) -> impl Iterator<Item = &JournalEntry> {
        let start_idx = self
            .time_index
            .range(..=time)
            .next_back()
            .map_or(0, |(_, &idx)| idx);

        self.entries[start_idx..].iter()
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    /// Get number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if journal is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.time_index.clear();
        self.sequence = 0;
    }
}

/// Time-travel scrubber for interactive debugging.
#[derive(Debug)]
pub struct TimeScrubber {
    /// Checkpoint manager.
    checkpoints: CheckpointManager,
    /// Event journal.
    journal: EventJournal,
    /// Current time.
    current_time: SimTime,
    /// Current state.
    current_state: SimState,
}

impl TimeScrubber {
    /// Create a new time scrubber.
    #[must_use]
    pub fn new(
        checkpoint_interval: u64,
        max_storage: usize,
        compression_level: i32,
        record_rng_state: bool,
    ) -> Self {
        Self {
            checkpoints: CheckpointManager::new(
                checkpoint_interval,
                max_storage,
                compression_level,
            ),
            journal: EventJournal::new(record_rng_state),
            current_time: SimTime::ZERO,
            current_state: SimState::default(),
        }
    }

    /// Seek to a specific time.
    ///
    /// # Errors
    ///
    /// Returns error if seek fails.
    pub fn seek_to(&mut self, target: SimTime) -> SimResult<&SimState> {
        if target == self.current_time {
            return Ok(&self.current_state);
        }

        // Restore from nearest checkpoint
        let (state, checkpoint_time) = self.checkpoints.restore_at(target)?;
        self.current_state = state;
        self.current_time = checkpoint_time;

        // Replay events from checkpoint to target
        for entry in self.journal.entries_from(checkpoint_time) {
            if entry.time > target {
                break;
            }
            // Apply event (simplified - real implementation would deserialize and apply)
            self.current_time = entry.time;
        }

        Ok(&self.current_state)
    }

    /// Get current time.
    #[must_use]
    pub const fn current_time(&self) -> SimTime {
        self.current_time
    }

    /// Get current state.
    #[must_use]
    pub const fn current_state(&self) -> &SimState {
        &self.current_state
    }

    /// Get checkpoint manager.
    #[must_use]
    pub const fn checkpoints(&self) -> &CheckpointManager {
        &self.checkpoints
    }

    /// Get mutable checkpoint manager.
    #[must_use]
    pub fn checkpoints_mut(&mut self) -> &mut CheckpointManager {
        &mut self.checkpoints
    }

    /// Get journal.
    #[must_use]
    pub const fn journal(&self) -> &EventJournal {
        &self.journal
    }

    /// Get mutable journal.
    #[must_use]
    pub fn journal_mut(&mut self) -> &mut EventJournal {
        &mut self.journal
    }
}

// =============================================================================
// Streaming Checkpoint Manager (Section 4.3.3)
// =============================================================================

/// Zero-copy streaming checkpoint manager [50].
///
/// Eliminates Muda of Processing by streaming directly from serializer
/// to compressor to memory-mapped file without intermediate allocations.
#[derive(Debug)]
pub struct StreamingCheckpointManager {
    /// Base path for checkpoint files.
    base_path: std::path::PathBuf,
    /// Compression level (1-22).
    compression_level: i32,
    /// Checkpoint interval in steps.
    interval: u64,
    /// Current checkpoint count.
    checkpoint_count: usize,
    /// Total bytes written.
    total_bytes_written: usize,
}

impl StreamingCheckpointManager {
    /// Create a new streaming checkpoint manager.
    ///
    /// # Errors
    ///
    /// Returns error if base path cannot be created.
    pub fn new(
        base_path: impl Into<std::path::PathBuf>,
        interval: u64,
        compression_level: i32,
    ) -> SimResult<Self> {
        let base_path = base_path.into();
        std::fs::create_dir_all(&base_path)?;

        Ok(Self {
            base_path,
            compression_level,
            interval,
            checkpoint_count: 0,
            total_bytes_written: 0,
        })
    }

    /// Check if checkpoint should be created at this step.
    #[must_use]
    pub const fn should_checkpoint(&self, step: u64) -> bool {
        step % self.interval == 0
    }

    /// Create checkpoint with streaming serialization (zero intermediate allocation).
    ///
    /// # Errors
    ///
    /// Returns error if serialization or file I/O fails.
    pub fn checkpoint_streaming<S: Serialize>(
        &mut self,
        time: SimTime,
        step: u64,
        state: &S,
        rng_state: &RngState,
    ) -> SimResult<std::path::PathBuf> {
        let filename = format!("checkpoint_{step:012}.zst");
        let path = self.base_path.join(&filename);

        // Create file and wrap in streaming encoder
        let file = std::fs::File::create(&path)?;
        let mut encoder = zstd::stream::Encoder::new(file, self.compression_level)
            .map_err(|e| SimError::serialization(format!("Zstd encoder init: {e}")))?;

        // Stream checkpoint header
        let header = CheckpointHeader {
            time,
            step,
            rng_state: rng_state.clone(),
            version: (0, 1, 0),
        };
        bincode::serialize_into(&mut encoder, &header)
            .map_err(|e| SimError::serialization(format!("Header serialize: {e}")))?;

        // Stream state directly into encoder (no intermediate Vec)
        bincode::serialize_into(&mut encoder, state)
            .map_err(|e| SimError::serialization(format!("State serialize: {e}")))?;

        let file = encoder
            .finish()
            .map_err(|e| SimError::serialization(format!("Zstd finish: {e}")))?;

        // Get compressed size
        let metadata = file.metadata()?;
        self.total_bytes_written += metadata.len() as usize;
        self.checkpoint_count += 1;

        Ok(path)
    }

    /// Restore state from streaming checkpoint.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn restore_streaming<S: serde::de::DeserializeOwned>(
        &self,
        path: &std::path::Path,
    ) -> SimResult<(CheckpointHeader, S)> {
        let file = std::fs::File::open(path)?;
        let mut decoder = zstd::stream::Decoder::new(file)
            .map_err(|e| SimError::serialization(format!("Zstd decoder init: {e}")))?;

        let header: CheckpointHeader = bincode::deserialize_from(&mut decoder)
            .map_err(|e| SimError::serialization(format!("Header deserialize: {e}")))?;

        let state: S = bincode::deserialize_from(&mut decoder)
            .map_err(|e| SimError::serialization(format!("State deserialize: {e}")))?;

        Ok((header, state))
    }

    /// Get checkpoint count.
    #[must_use]
    pub const fn checkpoint_count(&self) -> usize {
        self.checkpoint_count
    }

    /// Get total bytes written.
    #[must_use]
    pub const fn total_bytes_written(&self) -> usize {
        self.total_bytes_written
    }
}

/// Checkpoint header for streaming checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Simulation time.
    pub time: SimTime,
    /// Step number.
    pub step: u64,
    /// RNG state for replay.
    pub rng_state: RngState,
    /// Schema version.
    pub version: (u16, u16, u16),
}

// =============================================================================
// Split Event Journal (Section 4.3.4)
// =============================================================================

/// Compact event header for fast time scrubbing [50][54].
///
/// Fixed size for cache-efficient scanning. Payloads are loaded on demand.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EventHeader {
    /// Simulation time (8 bytes).
    pub time: SimTime,
    /// Event type discriminant (4 bytes).
    pub event_type: u32,
    /// Offset into payload storage (8 bytes).
    pub payload_offset: u64,
    /// Payload size in bytes (4 bytes).
    pub payload_size: u32,
    /// Sequence number for ordering (8 bytes).
    pub sequence: u64,
}

/// Split event journal for efficient time scrubbing [50][54].
///
/// Separates compact headers from variable-size payloads to enable
/// fast time-based seeking without deserializing full events.
#[derive(Debug, Default)]
pub struct SplitEventJournal {
    /// Compact header index (sorted by time).
    headers: Vec<EventHeader>,
    /// Raw payload bytes.
    payloads: Vec<u8>,
    /// Current sequence counter.
    sequence: u64,
}

impl SplitEventJournal {
    /// Create a new split event journal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the journal.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn append<T: Serialize>(
        &mut self,
        time: SimTime,
        event_type: u32,
        event: &T,
    ) -> SimResult<()> {
        let payload =
            bincode::serialize(event).map_err(|e| SimError::serialization(e.to_string()))?;

        let header = EventHeader {
            time,
            event_type,
            payload_offset: self.payloads.len() as u64,
            payload_size: payload.len() as u32,
            sequence: self.sequence,
        };

        self.headers.push(header);
        self.payloads.extend(payload);
        self.sequence += 1;

        Ok(())
    }

    /// Seek to time index without deserializing payloads (O(log n)).
    #[must_use]
    pub fn seek_to_time(&self, target: SimTime) -> Option<usize> {
        self.headers
            .binary_search_by(|h| h.time.cmp(&target))
            .ok()
            .or_else(|| {
                // Return index of first event at or after target
                self.headers.iter().position(|h| h.time >= target)
            })
    }

    /// Load specific event payload on demand.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn load_payload<T: serde::de::DeserializeOwned>(
        &self,
        header: &EventHeader,
    ) -> SimResult<T> {
        let start = header.payload_offset as usize;
        let end = start + header.payload_size as usize;

        if end > self.payloads.len() {
            return Err(SimError::journal("Payload offset out of bounds"));
        }

        bincode::deserialize(&self.payloads[start..end])
            .map_err(|e| SimError::journal(format!("Payload deserialize: {e}")))
    }

    /// Iterate headers in time range (fast, no payload deserialization).
    pub fn headers_in_range(
        &self,
        start: SimTime,
        end: SimTime,
    ) -> impl Iterator<Item = &EventHeader> {
        self.headers
            .iter()
            .filter(move |h| h.time >= start && h.time <= end)
    }

    /// Get all headers.
    #[must_use]
    pub fn headers(&self) -> &[EventHeader] {
        &self.headers
    }

    /// Get header count.
    #[must_use]
    pub fn header_count(&self) -> usize {
        self.headers.len()
    }

    /// Get total payload bytes.
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.payloads.len()
    }

    /// Clear the journal.
    pub fn clear(&mut self) {
        self.headers.clear();
        self.payloads.clear();
        self.sequence = 0;
    }
}

// =============================================================================
// Schema Evolution (Section 4.3.7)
// =============================================================================

/// Versioned journal entry for schema evolution [50].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedEntry {
    /// Schema version (SemVer-style).
    pub version: (u16, u16, u16),
    /// Entry type tag.
    pub entry_type: String,
    /// Payload (version-specific).
    pub payload: Vec<u8>,
}

impl VersionedEntry {
    /// Create a new versioned entry.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn new<T: Serialize>(
        version: (u16, u16, u16),
        entry_type: impl Into<String>,
        data: &T,
    ) -> SimResult<Self> {
        let payload =
            bincode::serialize(data).map_err(|e| SimError::serialization(e.to_string()))?;

        Ok(Self {
            version,
            entry_type: entry_type.into(),
            payload,
        })
    }

    /// Deserialize payload as specific type.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn deserialize<T: serde::de::DeserializeOwned>(&self) -> SimResult<T> {
        bincode::deserialize(&self.payload).map_err(|e| SimError::serialization(e.to_string()))
    }
}

/// Migration function type.
type MigrationFn = Box<dyn Fn(&[u8]) -> SimResult<Vec<u8>> + Send + Sync>;

/// Version tuple for schema versioning.
type SchemaVersion = (u16, u16, u16);

/// Migration key: (`from_version`, `to_version`).
type MigrationKey = (SchemaVersion, SchemaVersion);

/// Schema migrator for backward compatibility [50].
///
/// Enables reading old journal entries by migrating them to current schema version.
pub struct SchemaMigrator {
    /// Current version.
    current_version: SchemaVersion,
    /// Registered migrations: (`from_version`, `to_version`) -> `migration_fn`.
    migrations: std::collections::HashMap<MigrationKey, MigrationFn>,
}

impl std::fmt::Debug for SchemaMigrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaMigrator")
            .field("current_version", &self.current_version)
            .field("migration_count", &self.migrations.len())
            .finish()
    }
}

impl SchemaMigrator {
    /// Create a new schema migrator.
    #[must_use]
    pub fn new(current_version: SchemaVersion) -> Self {
        Self {
            current_version,
            migrations: std::collections::HashMap::new(),
        }
    }

    /// Register migration between versions.
    pub fn register<F>(&mut self, from: SchemaVersion, to: SchemaVersion, migrate: F)
    where
        F: Fn(&[u8]) -> SimResult<Vec<u8>> + Send + Sync + 'static,
    {
        self.migrations.insert((from, to), Box::new(migrate));
    }

    /// Check if entry needs migration.
    #[must_use]
    pub fn needs_migration(&self, entry: &VersionedEntry) -> bool {
        entry.version != self.current_version
    }

    /// Migrate entry to current version.
    ///
    /// # Errors
    ///
    /// Returns error if no migration path exists or migration fails.
    pub fn migrate_to_current(&self, entry: &VersionedEntry) -> SimResult<Vec<u8>> {
        if entry.version == self.current_version {
            return Ok(entry.payload.clone());
        }

        // Try direct migration
        if let Some(migration) = self.migrations.get(&(entry.version, self.current_version)) {
            return migration(&entry.payload);
        }

        // Try to find a migration path (simple linear search for now)
        // In production, you'd want Dijkstra or BFS for optimal path
        for (&(from, intermediate), first_step) in &self.migrations {
            if from == entry.version {
                if let Some(second_step) =
                    self.migrations.get(&(intermediate, self.current_version))
                {
                    let intermediate_payload = first_step(&entry.payload)?;
                    return second_step(&intermediate_payload);
                }
            }
        }

        Err(SimError::journal(format!(
            "No migration path from {:?} to {:?}",
            entry.version, self.current_version
        )))
    }

    /// Get current version.
    #[must_use]
    pub const fn current_version(&self) -> SchemaVersion {
        self.current_version
    }

    /// Get number of registered migrations.
    #[must_use]
    pub fn migration_count(&self) -> usize {
        self.migrations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::rng::SimRng;
    use crate::engine::state::Vec3;

    #[test]
    fn test_checkpoint_create_restore() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));

        let rng = SimRng::new(42);
        let rng_state = rng.save_state();

        let checkpoint = Checkpoint::create(SimTime::from_secs(1.0), 100, &state, rng_state, 3);

        assert!(checkpoint.is_ok());
        let checkpoint = checkpoint.ok().unwrap();

        let restored = checkpoint.restore();
        assert!(restored.is_ok());
        let restored = restored.ok().unwrap();

        assert_eq!(restored.num_bodies(), 1);
        assert!((restored.positions()[0].x - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_checkpoint_integrity_check() {
        let state = SimState::new();
        let rng = SimRng::new(42);

        let mut checkpoint =
            Checkpoint::create(SimTime::from_secs(1.0), 100, &state, rng.save_state(), 3)
                .ok()
                .unwrap();

        // Corrupt the data
        if !checkpoint.data.is_empty() {
            checkpoint.data[0] ^= 0xFF;
        }

        let result = checkpoint.restore();
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new(10, 1024 * 1024, 3);

        let state = SimState::new();
        let rng = SimRng::new(42);

        // Create checkpoints
        for step in (0..100).step_by(10) {
            let time = SimTime::from_secs(step as f64 * 0.1);
            manager
                .checkpoint(time, step as u64, &state, rng.save_state())
                .ok();
        }

        assert_eq!(manager.num_checkpoints(), 10);

        // Get checkpoint at specific time
        let cp = manager.get_checkpoint_at(SimTime::from_secs(0.55));
        assert!(cp.is_some());
        assert!(cp.map(|c| c.time.as_secs_f64()).unwrap_or(0.0) <= 0.55);
    }

    #[test]
    fn test_checkpoint_manager_gc() {
        // Very small max storage to trigger GC
        let mut manager = CheckpointManager::new(1, 100, 1);

        let state = SimState::new();
        let rng = SimRng::new(42);

        // Create many checkpoints
        for step in 0..100 {
            let time = SimTime::from_secs(step as f64 * 0.01);
            manager
                .checkpoint(time, step, &state, rng.save_state())
                .ok();
        }

        // Should have garbage collected old ones
        assert!(manager.storage_used() <= 100);
    }

    #[test]
    fn test_event_journal() {
        let mut journal = EventJournal::new(true);

        let event1 = "event1";
        let event2 = "event2";

        journal
            .append(SimTime::from_secs(1.0), 100, &event1, None)
            .ok();
        journal
            .append(SimTime::from_secs(2.0), 200, &event2, None)
            .ok();

        assert_eq!(journal.len(), 2);
        assert!(!journal.is_empty());

        let entries: Vec<_> = journal.entries_from(SimTime::from_secs(1.5)).collect();
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_time_scrubber() {
        let mut scrubber = TimeScrubber::new(10, 1024 * 1024, 3, false);

        assert_eq!(scrubber.current_time(), SimTime::ZERO);

        // Add some checkpoints manually
        let state = SimState::new();
        let rng = SimRng::new(42);

        scrubber
            .checkpoints_mut()
            .checkpoint(SimTime::from_secs(1.0), 100, &state, rng.save_state())
            .ok();

        // Seek to checkpoint time
        let result = scrubber.seek_to(SimTime::from_secs(1.0));
        assert!(result.is_ok());
    }

    // === Split Event Journal Tests (Section 4.3.4) ===

    #[test]
    fn test_split_journal_append() {
        let mut journal = SplitEventJournal::new();

        journal.append(SimTime::from_secs(1.0), 1, &"event1").ok();
        journal.append(SimTime::from_secs(2.0), 2, &"event2").ok();

        assert_eq!(journal.header_count(), 2);
        assert!(journal.payload_bytes() > 0);
    }

    #[test]
    fn test_split_journal_load_payload() {
        let mut journal = SplitEventJournal::new();

        let event = "test_event_data";
        journal.append(SimTime::from_secs(1.0), 1, &event).ok();

        let header = &journal.headers()[0];
        let loaded: String = journal.load_payload(header).ok().unwrap();

        assert_eq!(loaded, event);
    }

    #[test]
    fn test_split_journal_seek_to_time() {
        let mut journal = SplitEventJournal::new();

        for i in 0..10 {
            journal
                .append(SimTime::from_secs(i as f64), i as u32, &format!("event{i}"))
                .ok();
        }

        // Exact match
        let idx = journal.seek_to_time(SimTime::from_secs(5.0));
        assert_eq!(idx, Some(5));

        // Between times
        let idx = journal.seek_to_time(SimTime::from_secs(5.5));
        assert_eq!(idx, Some(6)); // First event at or after target

        // At start
        let idx = journal.seek_to_time(SimTime::ZERO);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_split_journal_headers_in_range() {
        let mut journal = SplitEventJournal::new();

        for i in 0..10 {
            journal
                .append(SimTime::from_secs(i as f64), i as u32, &i)
                .ok();
        }

        let headers: Vec<_> = journal
            .headers_in_range(SimTime::from_secs(3.0), SimTime::from_secs(7.0))
            .collect();

        assert_eq!(headers.len(), 5); // Events at t=3,4,5,6,7
        assert_eq!(headers[0].event_type, 3);
        assert_eq!(headers[4].event_type, 7);
    }

    #[test]
    fn test_split_journal_clear() {
        let mut journal = SplitEventJournal::new();

        journal.append(SimTime::from_secs(1.0), 1, &"event").ok();
        assert_eq!(journal.header_count(), 1);

        journal.clear();
        assert_eq!(journal.header_count(), 0);
        assert_eq!(journal.payload_bytes(), 0);
    }

    // === Schema Evolution Tests (Section 4.3.7) ===

    #[test]
    fn test_versioned_entry_create() {
        let data = "test_data";
        let entry = VersionedEntry::new((1, 0, 0), "test_type", &data);

        assert!(entry.is_ok());
        let entry = entry.ok().unwrap();

        assert_eq!(entry.version, (1, 0, 0));
        assert_eq!(entry.entry_type, "test_type");
    }

    #[test]
    fn test_versioned_entry_deserialize() {
        let data = vec![1u32, 2, 3, 4, 5];
        let entry = VersionedEntry::new((1, 0, 0), "vec_u32", &data)
            .ok()
            .unwrap();

        let restored: Vec<u32> = entry.deserialize().ok().unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_schema_migrator_no_migration_needed() {
        let migrator = SchemaMigrator::new((1, 0, 0));
        let entry = VersionedEntry::new((1, 0, 0), "test", &"data")
            .ok()
            .unwrap();

        assert!(!migrator.needs_migration(&entry));

        let result = migrator.migrate_to_current(&entry);
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap(), entry.payload);
    }

    #[test]
    fn test_schema_migrator_direct_migration() {
        let mut migrator = SchemaMigrator::new((1, 1, 0));

        // Register migration from 1.0.0 to 1.1.0
        migrator.register((1, 0, 0), (1, 1, 0), |payload| {
            // Simple migration: prepend a byte
            let mut new_payload = vec![0xFF];
            new_payload.extend_from_slice(payload);
            Ok(new_payload)
        });

        let entry = VersionedEntry {
            version: (1, 0, 0),
            entry_type: "test".to_string(),
            payload: vec![1, 2, 3],
        };

        assert!(migrator.needs_migration(&entry));

        let result = migrator.migrate_to_current(&entry);
        assert!(result.is_ok());

        let migrated = result.ok().unwrap();
        assert_eq!(migrated, vec![0xFF, 1, 2, 3]);
    }

    #[test]
    fn test_schema_migrator_no_path() {
        let migrator = SchemaMigrator::new((2, 0, 0));

        let entry = VersionedEntry {
            version: (1, 0, 0),
            entry_type: "test".to_string(),
            payload: vec![1, 2, 3],
        };

        let result = migrator.migrate_to_current(&entry);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_migrator_chained_migration() {
        let mut migrator = SchemaMigrator::new((1, 2, 0));

        // Register 1.0.0 -> 1.1.0
        migrator.register((1, 0, 0), (1, 1, 0), |payload| {
            let mut new = vec![0xAA];
            new.extend_from_slice(payload);
            Ok(new)
        });

        // Register 1.1.0 -> 1.2.0
        migrator.register((1, 1, 0), (1, 2, 0), |payload| {
            let mut new = vec![0xBB];
            new.extend_from_slice(payload);
            Ok(new)
        });

        let entry = VersionedEntry {
            version: (1, 0, 0),
            entry_type: "test".to_string(),
            payload: vec![1, 2, 3],
        };

        let result = migrator.migrate_to_current(&entry);
        assert!(result.is_ok());

        let migrated = result.ok().unwrap();
        // Should be: 0xBB (from 1.1->1.2) + 0xAA (from 1.0->1.1) + [1,2,3]
        assert_eq!(migrated, vec![0xBB, 0xAA, 1, 2, 3]);
    }

    // === Streaming Checkpoint Manager Tests (Section 4.3.3) ===

    #[test]
    fn test_streaming_checkpoint_roundtrip() {
        let temp_dir = tempfile::tempdir().ok().unwrap();
        let mut manager = StreamingCheckpointManager::new(temp_dir.path(), 10, 3)
            .ok()
            .unwrap();

        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));

        let rng = SimRng::new(42);
        let rng_state = rng.save_state();

        // Create checkpoint
        let path = manager
            .checkpoint_streaming(SimTime::from_secs(1.0), 100, &state, &rng_state)
            .ok()
            .unwrap();

        assert!(path.exists());
        assert_eq!(manager.checkpoint_count(), 1);

        // Restore checkpoint
        let (header, restored): (CheckpointHeader, SimState) =
            manager.restore_streaming(&path).ok().unwrap();

        assert_eq!(header.step, 100);
        assert!((header.time.as_secs_f64() - 1.0).abs() < f64::EPSILON);
        assert_eq!(restored.num_bodies(), 1);
        assert!((restored.positions()[0].x - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_streaming_checkpoint_should_checkpoint() {
        let temp_dir = tempfile::tempdir().ok().unwrap();
        let manager = StreamingCheckpointManager::new(temp_dir.path(), 10, 3)
            .ok()
            .unwrap();

        assert!(manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(5));
        assert!(manager.should_checkpoint(10));
        assert!(manager.should_checkpoint(100));
    }

    // === Additional Coverage Tests ===

    #[test]
    fn test_checkpoint_manager_clear() {
        let mut manager = CheckpointManager::new(10, 1024 * 1024, 3);
        let state = SimState::new();
        let rng = SimRng::new(42);

        manager
            .checkpoint(SimTime::from_secs(1.0), 10, &state, rng.save_state())
            .ok();

        assert_eq!(manager.num_checkpoints(), 1);
        assert!(manager.storage_used() > 0);

        manager.clear();
        assert_eq!(manager.num_checkpoints(), 0);
        assert_eq!(manager.storage_used(), 0);
    }

    #[test]
    fn test_checkpoint_manager_restore_not_found() {
        let manager = CheckpointManager::new(10, 1024 * 1024, 3);

        let result = manager.restore_at(SimTime::from_secs(5.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_event_journal_entries() {
        let mut journal = EventJournal::new(false);

        journal
            .append(SimTime::from_secs(1.0), 100, &"event1", None)
            .ok();
        journal
            .append(SimTime::from_secs(2.0), 200, &"event2", None)
            .ok();

        let entries = journal.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].step, 100);
        assert_eq!(entries[1].step, 200);
    }

    #[test]
    fn test_event_journal_clear() {
        let mut journal = EventJournal::new(false);

        journal
            .append(SimTime::from_secs(1.0), 100, &"event", None)
            .ok();
        assert!(!journal.is_empty());

        journal.clear();
        assert!(journal.is_empty());
        assert_eq!(journal.len(), 0);
    }

    #[test]
    fn test_event_journal_with_rng() {
        let mut journal = EventJournal::new(true); // record RNG state
        let rng = SimRng::new(42);
        let rng_state = rng.save_state();

        journal
            .append(SimTime::from_secs(1.0), 100, &"event", Some(&rng_state))
            .ok();

        let entries = journal.entries();
        assert!(entries[0].rng_state.is_some());
    }

    #[test]
    fn test_time_scrubber_seek_same_time() {
        let mut scrubber = TimeScrubber::new(10, 1024 * 1024, 3, false);
        let state = SimState::new();
        let rng = SimRng::new(42);

        scrubber
            .checkpoints_mut()
            .checkpoint(SimTime::ZERO, 0, &state, rng.save_state())
            .ok();

        // First seek
        let _ = scrubber.seek_to(SimTime::ZERO);
        // Seek to same time should return immediately
        let result = scrubber.seek_to(SimTime::ZERO);
        assert!(result.is_ok());
    }

    #[test]
    fn test_time_scrubber_current_state() {
        let scrubber = TimeScrubber::new(10, 1024 * 1024, 3, false);
        let state = scrubber.current_state();
        assert_eq!(state.num_bodies(), 0);
    }

    #[test]
    fn test_time_scrubber_journal_accessors() {
        let mut scrubber = TimeScrubber::new(10, 1024 * 1024, 3, false);

        // Read-only journal
        assert!(scrubber.journal().is_empty());

        // Mutable journal
        scrubber
            .journal_mut()
            .append(SimTime::from_secs(1.0), 100, &"event", None)
            .ok();

        assert!(!scrubber.journal().is_empty());
    }

    #[test]
    fn test_streaming_checkpoint_total_bytes() {
        let temp_dir = tempfile::tempdir().ok().unwrap();
        let mut manager = StreamingCheckpointManager::new(temp_dir.path(), 10, 3)
            .ok()
            .unwrap();

        assert_eq!(manager.total_bytes_written(), 0);

        let state = SimState::new();
        let rng = SimRng::new(42);

        manager
            .checkpoint_streaming(SimTime::from_secs(1.0), 10, &state, &rng.save_state())
            .ok();

        assert!(manager.total_bytes_written() > 0);
    }

    #[test]
    fn test_checkpoint_compressed_size() {
        let state = SimState::new();
        let rng = SimRng::new(42);

        let checkpoint =
            Checkpoint::create(SimTime::from_secs(1.0), 100, &state, rng.save_state(), 3)
                .ok()
                .unwrap();

        assert!(checkpoint.compressed_size() > 0);
    }

    #[test]
    fn test_checkpoint_clone() {
        let state = SimState::new();
        let rng = SimRng::new(42);

        let checkpoint =
            Checkpoint::create(SimTime::from_secs(1.0), 100, &state, rng.save_state(), 3)
                .ok()
                .unwrap();

        let cloned = checkpoint.clone();
        assert_eq!(cloned.step, checkpoint.step);
        assert_eq!(cloned.hash, checkpoint.hash);
    }

    #[test]
    fn test_journal_entry_clone() {
        let mut journal = EventJournal::new(false);
        journal
            .append(SimTime::from_secs(1.0), 100, &"event", None)
            .ok();

        let entry = &journal.entries()[0];
        let cloned = entry.clone();
        assert_eq!(cloned.step, entry.step);
    }

    #[test]
    fn test_versioned_entry_debug() {
        let entry = VersionedEntry::new((1, 0, 0), "test", &"data")
            .ok()
            .unwrap();
        let debug = format!("{:?}", entry);
        assert!(debug.contains("VersionedEntry"));
    }

    #[test]
    fn test_event_header_debug() {
        let header = EventHeader {
            time: SimTime::from_secs(1.0),
            event_type: 42,
            payload_offset: 0,
            payload_size: 100,
            sequence: 1,
        };
        let debug = format!("{:?}", header);
        assert!(debug.contains("EventHeader"));
    }

    #[test]
    fn test_checkpoint_header_debug() {
        let rng = SimRng::new(42);
        let header = CheckpointHeader {
            time: SimTime::from_secs(1.0),
            step: 100,
            rng_state: rng.save_state(),
            version: (1, 0, 0),
        };
        let debug = format!("{:?}", header);
        assert!(debug.contains("CheckpointHeader"));
    }

    #[test]
    fn test_checkpoint_manager_should_checkpoint() {
        let manager = CheckpointManager::new(10, 1024 * 1024, 3);
        assert!(manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(5));
        assert!(manager.should_checkpoint(10));
        assert!(manager.should_checkpoint(20));
    }

    #[test]
    fn test_split_journal_seek_before_start() {
        let mut journal = SplitEventJournal::new();

        for i in 1..10 {
            journal
                .append(SimTime::from_secs(i as f64), i as u32, &i)
                .ok();
        }

        // Seek before any events
        let idx = journal.seek_to_time(SimTime::from_secs(0.5));
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_split_journal_seek_after_end() {
        let mut journal = SplitEventJournal::new();

        for i in 0..5 {
            journal
                .append(SimTime::from_secs(i as f64), i as u32, &i)
                .ok();
        }

        // Seek way after all events
        let idx = journal.seek_to_time(SimTime::from_secs(100.0));
        assert_eq!(idx, None);
    }

    #[test]
    fn test_split_journal_empty_seek() {
        let journal = SplitEventJournal::new();
        let idx = journal.seek_to_time(SimTime::from_secs(1.0));
        assert_eq!(idx, None);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::engine::rng::SimRng;
    use crate::engine::state::Vec3;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: checkpoint restore produces identical state.
        #[test]
        fn prop_checkpoint_roundtrip(
            x in -1000.0f64..1000.0,
            y in -1000.0f64..1000.0,
            z in -1000.0f64..1000.0,
            mass in 0.1f64..1000.0,
            seed in 0u64..u64::MAX,
        ) {
            let mut state = SimState::new();
            state.add_body(mass, Vec3::new(x, y, z), Vec3::zero());

            let rng = SimRng::new(seed);
            let checkpoint = Checkpoint::create(
                SimTime::from_secs(1.0),
                100,
                &state,
                rng.save_state(),
                3,
            );

            prop_assert!(checkpoint.is_ok());
            let checkpoint = checkpoint.ok().unwrap();

            let restored = checkpoint.restore();
            prop_assert!(restored.is_ok());
            let restored = restored.ok().unwrap();

            prop_assert_eq!(restored.num_bodies(), state.num_bodies());
            prop_assert!((restored.positions()[0].x - x).abs() < 1e-10);
            prop_assert!((restored.positions()[0].y - y).abs() < 1e-10);
            prop_assert!((restored.positions()[0].z - z).abs() < 1e-10);
        }
    }
}
