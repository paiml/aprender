//! StateBridge - Game State Inspection Bridge
//!
//! Per spec Section 3.3: Bridge between browser and game state with zero-copy views.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │  StateBridge                                                        │
//! │  ──────────────────                                                │
//! │                                                                    │
//! │  ┌─────────────────────┐     ┌─────────────────────────┐          │
//! │  │  Zero-Copy Path     │     │  Serialized Path        │          │
//! │  │  (WasmRuntime)      │     │  (BrowserController)    │          │
//! │  │                     │     │                         │          │
//! │  │  • MemoryView       │     │  • bincode RPC          │          │
//! │  │  • Direct access    │     │  • Snapshot cache       │          │
//! │  │  • < 100ns reads    │     │  • Delta encoding       │          │
//! │  └─────────────────────┘     └─────────────────────────┘          │
//! │                                                                    │
//! │  Toyota Principle: Muda elimination via zero-copy where possible  │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Toyota Principles Applied
//!
//! - **Muda (Waste Elimination)**: Zero-copy memory views avoid serialization
//! - **Poka-Yoke (Error Proofing)**: Type-safe entity queries

use crate::result::ProbarResult;
use crate::runtime::{EntityId, MemoryView, StateDelta};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Game state snapshot with delta encoding
///
/// Per spec: Delta encoding achieves 94% overhead reduction (Lavoie \[9\])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStateSnapshot {
    /// Frame number when snapshot was taken
    pub frame: u64,
    /// Game state data
    pub state: GameStateData,
    /// Perceptual hash for visual comparison (more robust than SHA-256)
    pub visual_phash: u64,
    /// Cryptographic hash for determinism verification
    pub state_hash: u64,
}

impl GameStateSnapshot {
    /// Create a new snapshot
    #[must_use]
    pub fn new(frame: u64, state: GameStateData) -> Self {
        let state_hash = state.compute_hash();
        Self {
            frame,
            state,
            visual_phash: 0,
            state_hash,
        }
    }

    /// Set the perceptual hash
    #[must_use]
    pub const fn with_phash(mut self, phash: u64) -> Self {
        self.visual_phash = phash;
        self
    }
}

/// Game state data container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GameStateData {
    /// Entity positions (entity_id -> (x, y))
    pub positions: HashMap<u32, (f32, f32)>,
    /// Entity velocities (entity_id -> (vx, vy))
    pub velocities: HashMap<u32, (f32, f32)>,
    /// Scores
    pub scores: HashMap<String, i32>,
    /// Game flags
    pub flags: HashMap<String, bool>,
    /// Custom state values
    pub custom: HashMap<String, serde_json::Value>,
}

impl GameStateData {
    /// Create empty state data
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add entity position
    pub fn add_position(&mut self, entity_id: u32, x: f32, y: f32) {
        self.positions.insert(entity_id, (x, y));
    }

    /// Add entity velocity
    pub fn add_velocity(&mut self, entity_id: u32, vx: f32, vy: f32) {
        self.velocities.insert(entity_id, (vx, vy));
    }

    /// Set a score
    pub fn set_score(&mut self, name: impl Into<String>, value: i32) {
        self.scores.insert(name.into(), value);
    }

    /// Set a flag
    pub fn set_flag(&mut self, name: impl Into<String>, value: bool) {
        self.flags.insert(name.into(), value);
    }

    /// Get entity position
    #[must_use]
    pub fn get_position(&self, entity_id: u32) -> Option<(f32, f32)> {
        self.positions.get(&entity_id).copied()
    }

    /// Get entity velocity
    #[must_use]
    pub fn get_velocity(&self, entity_id: u32) -> Option<(f32, f32)> {
        self.velocities.get(&entity_id).copied()
    }

    /// Get a score
    #[must_use]
    pub fn get_score(&self, name: &str) -> Option<i32> {
        self.scores.get(name).copied()
    }

    /// Get a flag
    #[must_use]
    pub fn get_flag(&self, name: &str) -> Option<bool> {
        self.flags.get(name).copied()
    }

    /// Compute hash of the state
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash positions in sorted order for determinism
        let mut positions: Vec<_> = self.positions.iter().collect();
        positions.sort_by_key(|(k, _)| *k);
        for (k, (x, y)) in positions {
            k.hash(&mut hasher);
            x.to_bits().hash(&mut hasher);
            y.to_bits().hash(&mut hasher);
        }

        // Hash velocities
        let mut velocities: Vec<_> = self.velocities.iter().collect();
        velocities.sort_by_key(|(k, _)| *k);
        for (k, (vx, vy)) in velocities {
            k.hash(&mut hasher);
            vx.to_bits().hash(&mut hasher);
            vy.to_bits().hash(&mut hasher);
        }

        // Hash scores
        let mut scores: Vec<_> = self.scores.iter().collect();
        scores.sort_by_key(|(k, _)| k.as_str());
        for (k, v) in scores {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        // Hash flags
        let mut flags: Vec<_> = self.flags.iter().collect();
        flags.sort_by_key(|(k, _)| k.as_str());
        for (k, v) in flags {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Entity snapshot for inspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// Entity ID
    pub id: EntityId,
    /// Entity name (for debugging)
    pub name: String,
    /// Position (x, y)
    pub position: Option<(f32, f32)>,
    /// Velocity (vx, vy)
    pub velocity: Option<(f32, f32)>,
    /// Is entity active
    pub active: bool,
    /// Custom components as JSON
    pub components: HashMap<String, serde_json::Value>,
}

impl EntitySnapshot {
    /// Create a new entity snapshot
    #[must_use]
    pub fn new(id: EntityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            position: None,
            velocity: None,
            active: true,
            components: HashMap::new(),
        }
    }

    /// Set position
    #[must_use]
    pub const fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = Some((x, y));
        self
    }

    /// Set velocity
    #[must_use]
    pub const fn with_velocity(mut self, vx: f32, vy: f32) -> Self {
        self.velocity = Some((vx, vy));
        self
    }

    /// Add a component
    pub fn add_component(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.components.insert(name.into(), value);
    }
}

/// Visual comparison result using perceptual hash
///
/// Per Shamir \[19\]: pHash more robust than SHA-256 for game frame comparison
#[derive(Debug, Clone)]
pub struct VisualDiff {
    /// Perceptual similarity (0.0 to 1.0)
    pub perceptual_similarity: f64,
    /// Pixel-level difference count
    pub pixel_diff_count: u64,
    /// Highlighted regions that differ
    pub diff_regions: Vec<DiffRegion>,
    /// Expected image data (if available)
    pub expected: Option<Vec<u8>>,
    /// Actual image data
    pub actual: Vec<u8>,
    /// Diff overlay image
    pub highlighted: Vec<u8>,
}

impl VisualDiff {
    /// Create a new visual diff
    #[must_use]
    pub fn new(similarity: f64, actual: Vec<u8>) -> Self {
        Self {
            perceptual_similarity: similarity,
            pixel_diff_count: 0,
            diff_regions: Vec::new(),
            expected: None,
            actual,
            highlighted: Vec::new(),
        }
    }

    /// Check if images match within threshold
    #[must_use]
    pub fn matches(&self, threshold: f64) -> bool {
        self.perceptual_similarity >= threshold
    }

    /// Check if images are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.perceptual_similarity >= 1.0 - f64::EPSILON
    }
}

/// Region where images differ
#[derive(Debug, Clone)]
pub struct DiffRegion {
    /// X position
    pub x: u32,
    /// Y position
    pub y: u32,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
    /// Difference intensity (0.0 to 1.0)
    pub intensity: f64,
}

/// Bridge connection type
#[derive(Debug, Clone)]
pub enum BridgeConnection {
    /// Direct memory access (WasmRuntime)
    Direct,
    /// RPC via browser (BrowserController)
    Rpc {
        /// Session ID
        session_id: String,
    },
}

/// LRU cache for snapshot storage
#[derive(Debug)]
pub struct SnapshotCache {
    /// Maximum cache size
    max_size: usize,
    /// Cached snapshots (frame -> snapshot)
    cache: HashMap<u64, GameStateSnapshot>,
    /// Access order for LRU eviction
    access_order: Vec<u64>,
}

impl SnapshotCache {
    /// Create new cache with given size
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    /// Get snapshot from cache
    pub fn get(&mut self, frame: u64) -> Option<&GameStateSnapshot> {
        if self.cache.contains_key(&frame) {
            // Update access order
            self.access_order.retain(|&f| f != frame);
            self.access_order.push(frame);
            self.cache.get(&frame)
        } else {
            None
        }
    }

    /// Insert snapshot into cache
    pub fn insert(&mut self, frame: u64, snapshot: GameStateSnapshot) {
        // Evict if at capacity
        while self.cache.len() >= self.max_size {
            if let Some(oldest) = self.access_order.first().copied() {
                self.cache.remove(&oldest);
                self.access_order.remove(0);
            } else {
                break;
            }
        }

        self.cache.insert(frame, snapshot);
        self.access_order.push(frame);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get cache size
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// State bridge for game state inspection
///
/// Provides unified access to game state whether using WasmRuntime (zero-copy)
/// or BrowserController (serialized RPC).
#[derive(Debug)]
pub struct StateBridge {
    /// Connection type
    connection: BridgeConnection,
    /// Memory view for zero-copy access (used in direct mode)
    memory_view: Option<MemoryView>,
    /// Snapshot cache for RPC mode
    snapshot_cache: SnapshotCache,
    /// Delta history for replay
    delta_history: Vec<StateDelta>,
}

impl StateBridge {
    /// Create bridge with direct memory access (for WasmRuntime)
    #[must_use]
    pub fn direct(memory_view: MemoryView) -> Self {
        Self {
            connection: BridgeConnection::Direct,
            memory_view: Some(memory_view),
            snapshot_cache: SnapshotCache::new(100),
            delta_history: Vec::new(),
        }
    }

    /// Create bridge with RPC connection (for BrowserController)
    #[must_use]
    pub fn rpc(session_id: impl Into<String>) -> Self {
        Self {
            connection: BridgeConnection::Rpc {
                session_id: session_id.into(),
            },
            memory_view: None,
            snapshot_cache: SnapshotCache::new(100),
            delta_history: Vec::new(),
        }
    }

    /// Check if using direct memory access
    #[must_use]
    pub const fn is_direct(&self) -> bool {
        matches!(self.connection, BridgeConnection::Direct)
    }

    /// Get the memory view if using direct mode
    #[must_use]
    pub const fn memory_view(&self) -> Option<&MemoryView> {
        self.memory_view.as_ref()
    }

    /// Query entity by ID
    ///
    /// # Errors
    ///
    /// Returns error if entity not found
    pub fn query_entity(&self, entity_id: EntityId) -> ProbarResult<EntitySnapshot> {
        match &self.connection {
            BridgeConnection::Direct => {
                // In a real implementation, this would read from MemoryView
                // For now, return a mock entity
                Ok(EntitySnapshot::new(
                    entity_id,
                    format!("entity_{}", entity_id.raw()),
                ))
            }
            BridgeConnection::Rpc { session_id } => {
                // In a real implementation, this would make an RPC call
                let _ = session_id;
                Ok(EntitySnapshot::new(
                    entity_id,
                    format!("entity_{}", entity_id.raw()),
                ))
            }
        }
    }

    /// Get current game state snapshot
    ///
    /// # Errors
    ///
    /// Returns error if state cannot be captured
    pub fn snapshot(&mut self, frame: u64) -> ProbarResult<GameStateSnapshot> {
        // Check cache first
        if let Some(cached) = self.snapshot_cache.get(frame) {
            return Ok(cached.clone());
        }

        // Create new snapshot
        let state = GameStateData::new();
        let snapshot = GameStateSnapshot::new(frame, state);

        // Cache it
        self.snapshot_cache.insert(frame, snapshot.clone());

        Ok(snapshot)
    }

    /// Record a delta from current state
    pub fn record_delta(&mut self, delta: StateDelta) {
        self.delta_history.push(delta);
    }

    /// Get delta history
    #[must_use]
    pub fn deltas(&self) -> &[StateDelta] {
        &self.delta_history
    }

    /// Clear delta history
    pub fn clear_deltas(&mut self) {
        self.delta_history.clear();
    }

    /// Compute perceptual hash for image
    ///
    /// Per Shamir \[19\]: pHash is more robust than pixel comparison
    #[must_use]
    pub fn compute_phash(image_data: &[u8]) -> u64 {
        // Simplified pHash implementation
        // In production, use proper DCT-based algorithm

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Sample every Nth byte for speed
        let sample_rate = (image_data.len() / 64).max(1);
        for (i, &byte) in image_data.iter().enumerate() {
            if i % sample_rate == 0 {
                byte.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Compare two images using perceptual hash
    #[must_use]
    pub fn visual_compare(expected: &[u8], actual: &[u8]) -> VisualDiff {
        let phash_expected = Self::compute_phash(expected);
        let phash_actual = Self::compute_phash(actual);

        // Hamming distance between hashes
        let hamming_distance = (phash_expected ^ phash_actual).count_ones();
        let similarity = 1.0 - (f64::from(hamming_distance) / 64.0);

        VisualDiff {
            perceptual_similarity: similarity,
            pixel_diff_count: 0, // Would need actual pixel comparison
            diff_regions: Vec::new(),
            expected: Some(expected.to_vec()),
            actual: actual.to_vec(),
            highlighted: Vec::new(),
        }
    }
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec Section 6.1
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod game_state_data_tests {
        use super::*;

        #[test]
        fn test_new_state_data() {
            let state = GameStateData::new();
            assert!(state.positions.is_empty());
            assert!(state.velocities.is_empty());
            assert!(state.scores.is_empty());
        }

        #[test]
        fn test_add_position() {
            let mut state = GameStateData::new();
            state.add_position(1, 100.0, 200.0);
            assert_eq!(state.get_position(1), Some((100.0, 200.0)));
            assert_eq!(state.get_position(2), None);
        }

        #[test]
        fn test_add_velocity() {
            let mut state = GameStateData::new();
            state.add_velocity(1, 5.0, -3.0);
            assert_eq!(state.get_velocity(1), Some((5.0, -3.0)));
        }

        #[test]
        fn test_set_score() {
            let mut state = GameStateData::new();
            state.set_score("player1", 100);
            state.set_score("player2", 50);
            assert_eq!(state.get_score("player1"), Some(100));
            assert_eq!(state.get_score("player2"), Some(50));
            assert_eq!(state.get_score("player3"), None);
        }

        #[test]
        fn test_set_flag() {
            let mut state = GameStateData::new();
            state.set_flag("game_over", false);
            state.set_flag("paused", true);
            assert_eq!(state.get_flag("game_over"), Some(false));
            assert_eq!(state.get_flag("paused"), Some(true));
        }

        #[test]
        fn test_compute_hash_deterministic() {
            let mut state1 = GameStateData::new();
            state1.add_position(1, 100.0, 200.0);
            state1.set_score("player1", 50);

            let mut state2 = GameStateData::new();
            state2.set_score("player1", 50);
            state2.add_position(1, 100.0, 200.0);

            // Hash should be same regardless of insertion order
            assert_eq!(state1.compute_hash(), state2.compute_hash());
        }

        #[test]
        fn test_compute_hash_different() {
            let mut state1 = GameStateData::new();
            state1.add_position(1, 100.0, 200.0);

            let mut state2 = GameStateData::new();
            state2.add_position(1, 100.0, 201.0);

            assert_ne!(state1.compute_hash(), state2.compute_hash());
        }
    }

    mod entity_snapshot_tests {
        use super::*;

        #[test]
        fn test_new_entity_snapshot() {
            let entity = EntitySnapshot::new(EntityId::new(42), "player");
            assert_eq!(entity.id.raw(), 42);
            assert_eq!(entity.name, "player");
            assert!(entity.active);
        }

        #[test]
        fn test_with_position() {
            let entity = EntitySnapshot::new(EntityId::new(1), "ball").with_position(100.0, 200.0);
            assert_eq!(entity.position, Some((100.0, 200.0)));
        }

        #[test]
        fn test_with_velocity() {
            let entity = EntitySnapshot::new(EntityId::new(1), "ball").with_velocity(5.0, -3.0);
            assert_eq!(entity.velocity, Some((5.0, -3.0)));
        }

        #[test]
        fn test_add_component() {
            let mut entity = EntitySnapshot::new(EntityId::new(1), "player");
            entity.add_component("health", serde_json::json!(100));
            assert_eq!(
                entity.components.get("health"),
                Some(&serde_json::json!(100))
            );
        }
    }

    mod game_state_snapshot_tests {
        use super::*;

        #[test]
        fn test_new_snapshot() {
            let state = GameStateData::new();
            let snapshot = GameStateSnapshot::new(100, state);
            assert_eq!(snapshot.frame, 100);
            assert_ne!(snapshot.state_hash, 0);
        }

        #[test]
        fn test_with_phash() {
            let state = GameStateData::new();
            let snapshot = GameStateSnapshot::new(0, state).with_phash(12345);
            assert_eq!(snapshot.visual_phash, 12345);
        }
    }

    mod visual_diff_tests {
        use super::*;

        #[test]
        fn test_new_visual_diff() {
            let diff = VisualDiff::new(0.95, vec![1, 2, 3]);
            assert!((diff.perceptual_similarity - 0.95).abs() < f64::EPSILON);
        }

        #[test]
        fn test_matches_threshold() {
            let diff = VisualDiff::new(0.95, vec![]);
            assert!(diff.matches(0.90));
            assert!(diff.matches(0.95));
            assert!(!diff.matches(0.99));
        }

        #[test]
        fn test_is_identical() {
            let identical = VisualDiff::new(1.0, vec![]);
            assert!(identical.is_identical());

            let different = VisualDiff::new(0.99, vec![]);
            assert!(!different.is_identical());
        }
    }

    mod snapshot_cache_tests {
        use super::*;

        #[test]
        fn test_new_cache() {
            let cache = SnapshotCache::new(10);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_insert_and_get() {
            let mut cache = SnapshotCache::new(10);
            let snapshot = GameStateSnapshot::new(1, GameStateData::new());
            cache.insert(1, snapshot);

            assert!(!cache.is_empty());
            assert_eq!(cache.len(), 1);
            assert!(cache.get(1).is_some());
            assert!(cache.get(2).is_none());
        }

        #[test]
        fn test_lru_eviction() {
            let mut cache = SnapshotCache::new(2);

            cache.insert(1, GameStateSnapshot::new(1, GameStateData::new()));
            cache.insert(2, GameStateSnapshot::new(2, GameStateData::new()));

            // Access frame 1 to make it most recently used
            let _ = cache.get(1);

            // Insert frame 3, should evict frame 2 (least recently used)
            cache.insert(3, GameStateSnapshot::new(3, GameStateData::new()));

            assert!(cache.get(1).is_some()); // Still there
            assert!(cache.get(3).is_some()); // Added
                                             // Frame 2 was evicted
        }

        #[test]
        fn test_clear() {
            let mut cache = SnapshotCache::new(10);
            cache.insert(1, GameStateSnapshot::new(1, GameStateData::new()));
            cache.insert(2, GameStateSnapshot::new(2, GameStateData::new()));
            cache.clear();
            assert!(cache.is_empty());
        }
    }

    mod state_bridge_tests {
        use super::*;

        #[test]
        fn test_direct_bridge() {
            let view = MemoryView::new(1024);
            let bridge = StateBridge::direct(view);
            assert!(bridge.is_direct());
        }

        #[test]
        fn test_rpc_bridge() {
            let bridge = StateBridge::rpc("session-123");
            assert!(!bridge.is_direct());
        }

        #[test]
        fn test_query_entity() {
            let view = MemoryView::new(1024);
            let bridge = StateBridge::direct(view);
            let entity = bridge.query_entity(EntityId::new(42)).unwrap();
            assert_eq!(entity.id.raw(), 42);
        }

        #[test]
        fn test_snapshot_caching() {
            let view = MemoryView::new(1024);
            let mut bridge = StateBridge::direct(view);

            let snap1 = bridge.snapshot(100).unwrap();
            let snap2 = bridge.snapshot(100).unwrap();

            // Same frame should return same hash
            assert_eq!(snap1.state_hash, snap2.state_hash);
        }

        #[test]
        fn test_record_delta() {
            let view = MemoryView::new(1024);
            let mut bridge = StateBridge::direct(view);

            let delta = StateDelta::empty(0);
            bridge.record_delta(delta);

            assert_eq!(bridge.deltas().len(), 1);
        }

        #[test]
        fn test_clear_deltas() {
            let view = MemoryView::new(1024);
            let mut bridge = StateBridge::direct(view);

            bridge.record_delta(StateDelta::empty(0));
            bridge.record_delta(StateDelta::empty(1));
            bridge.clear_deltas();

            assert!(bridge.deltas().is_empty());
        }

        #[test]
        fn test_compute_phash() {
            let data1 = vec![1, 2, 3, 4, 5];
            let data2 = vec![1, 2, 3, 4, 5];
            let data3 = vec![5, 4, 3, 2, 1];

            let hash1 = StateBridge::compute_phash(&data1);
            let hash2 = StateBridge::compute_phash(&data2);
            let hash3 = StateBridge::compute_phash(&data3);

            assert_eq!(hash1, hash2);
            assert_ne!(hash1, hash3);
        }

        #[test]
        fn test_visual_compare() {
            let expected = vec![1, 2, 3, 4, 5];
            let actual = vec![1, 2, 3, 4, 5];

            let diff = StateBridge::visual_compare(&expected, &actual);
            assert!(diff.perceptual_similarity > 0.99);
            assert!(diff.is_identical());
        }

        #[test]
        fn test_visual_compare_different() {
            let expected = vec![1, 2, 3, 4, 5, 6, 7, 8];
            let actual = vec![8, 7, 6, 5, 4, 3, 2, 1];

            let diff = StateBridge::visual_compare(&expected, &actual);
            // Should be different
            assert!(diff.perceptual_similarity < 1.0);
        }
    }
}
