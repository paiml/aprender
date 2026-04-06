//! LRU cache for GPU graph tiles
//!
//! Implements Least Recently Used eviction policy for managing GPU memory.
//! Based on standard LRU cache algorithms with GPU buffer lifecycle management.

use std::collections::{HashMap, VecDeque};
use wgpu::Buffer;

/// Graph tile identifier (tile index)
pub type TileId = usize;

/// LRU cache for GPU buffers
///
/// Manages a fixed-capacity cache of GPU buffers with LRU eviction policy.
/// When capacity is reached, least recently used tile is evicted.
pub struct LruTileCache {
    /// Maximum number of tiles to cache
    capacity: usize,

    /// Map from tile ID to GPU buffer
    buffers: HashMap<TileId, Buffer>,

    /// Access order (front = most recent, back = least recent)
    access_order: VecDeque<TileId>,
}

impl LruTileCache {
    /// Create new LRU cache with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), buffers: HashMap::new(), access_order: VecDeque::new() }
    }

    /// Get buffer from cache (marks as recently used)
    pub fn get(&mut self, tile_id: TileId) -> Option<&Buffer> {
        if self.buffers.contains_key(&tile_id) {
            // Move to front (most recently used)
            self.access_order.retain(|&id| id != tile_id);
            self.access_order.push_front(tile_id);
            self.buffers.get(&tile_id)
        } else {
            None
        }
    }

    /// Insert buffer into cache (may evict LRU tile)
    ///
    /// Returns evicted tile ID if eviction occurred
    pub fn insert(&mut self, tile_id: TileId, buffer: Buffer) -> Option<TileId> {
        let mut evicted = None;

        // If already exists, just update access order
        if self.buffers.contains_key(&tile_id) {
            self.access_order.retain(|&id| id != tile_id);
            self.access_order.push_front(tile_id);
            self.buffers.insert(tile_id, buffer);
            return None;
        }

        // Evict LRU if at capacity
        if self.buffers.len() >= self.capacity {
            if let Some(lru_id) = self.access_order.pop_back() {
                self.buffers.remove(&lru_id);
                evicted = Some(lru_id);
            }
        }

        // Insert new buffer
        self.buffers.insert(tile_id, buffer);
        self.access_order.push_front(tile_id);

        evicted
    }

    /// Check if tile is in cache
    #[must_use]
    pub fn contains(&self, tile_id: TileId) -> bool {
        self.buffers.contains_key(&tile_id)
    }

    /// Get current cache size
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.access_order.clear();
    }

    /// Get cache hit rate statistics
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuDevice;

    #[tokio::test]
    async fn test_lru_cache_basic() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_lru_cache_basic: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let mut cache = LruTileCache::new(3);

        // Create test buffers
        let buf1 = device.create_buffer("tile_1", 1024, wgpu::BufferUsages::STORAGE).unwrap();
        let buf2 = device.create_buffer("tile_2", 1024, wgpu::BufferUsages::STORAGE).unwrap();
        let buf3 = device.create_buffer("tile_3", 1024, wgpu::BufferUsages::STORAGE).unwrap();
        let buf4 = device.create_buffer("tile_4", 1024, wgpu::BufferUsages::STORAGE).unwrap();

        // Insert 3 buffers
        assert_eq!(cache.insert(1, buf1), None);
        assert_eq!(cache.insert(2, buf2), None);
        assert_eq!(cache.insert(3, buf3), None);
        assert_eq!(cache.len(), 3);

        // Access tile 1 (makes it most recent)
        assert!(cache.get(1).is_some());

        // Insert 4th buffer should evict LRU (tile 2)
        assert_eq!(cache.insert(4, buf4), Some(2));
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(1));
        assert!(!cache.contains(2)); // Evicted
        assert!(cache.contains(3));
        assert!(cache.contains(4));
    }

    #[test]
    fn test_lru_cache_capacity() {
        let cache = LruTileCache::new(5);
        assert_eq!(cache.capacity(), 5);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn test_lru_cache_reinsertion() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_lru_cache_reinsertion: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let mut cache = LruTileCache::new(2);

        let buf1 = device.create_buffer("tile_1", 1024, wgpu::BufferUsages::STORAGE).unwrap();
        let buf1_new =
            device.create_buffer("tile_1_new", 1024, wgpu::BufferUsages::STORAGE).unwrap();

        // Insert tile 1
        assert_eq!(cache.insert(1, buf1), None);
        assert_eq!(cache.len(), 1);

        // Re-insert tile 1 (should not evict)
        assert_eq!(cache.insert(1, buf1_new), None);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(1));
    }

    #[tokio::test]
    async fn test_lru_cache_clear() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_lru_cache_clear: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let mut cache = LruTileCache::new(3);

        let buf1 = device.create_buffer("tile_1", 1024, wgpu::BufferUsages::STORAGE).unwrap();
        let buf2 = device.create_buffer("tile_2", 1024, wgpu::BufferUsages::STORAGE).unwrap();

        cache.insert(1, buf1);
        cache.insert(2, buf2);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(!cache.contains(1));
        assert!(!cache.contains(2));
    }

    #[tokio::test]
    async fn test_lru_cache_get_nonexistent() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_lru_cache_get_nonexistent: GPU not available");
            return;
        }

        let mut cache = LruTileCache::new(3);
        assert!(cache.get(999).is_none());
    }

    #[test]
    fn test_lru_cache_zero_capacity() {
        let cache = LruTileCache::new(0);
        assert_eq!(cache.capacity(), 1); // Minimum capacity is 1
    }
}
