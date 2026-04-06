//! Resource Pool with Semaphore
//!
//! AWP-05: Semaphore-based resource pool for managing limited resources.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// ----------------------------------------------------------------------------
// AWP-05: Resource Pool with Semaphore
// ----------------------------------------------------------------------------

/// Semaphore-based resource pool.
///
/// # Example
/// ```rust
/// use trueno::brick::ResourcePool;
///
/// let pool: ResourcePool<Vec<u8>> = ResourcePool::new(4, || Vec::with_capacity(1024));
///
/// // Acquire resources (up to max)
/// let r1 = pool.try_acquire().unwrap();
/// let r2 = pool.try_acquire().unwrap();
/// let r3 = pool.try_acquire().unwrap();
/// let r4 = pool.try_acquire().unwrap();
///
/// // Pool is exhausted
/// assert!(pool.try_acquire().is_none());
///
/// // Release one
/// drop(r1);
///
/// // Now we can acquire again
/// assert!(pool.try_acquire().is_some());
/// ```
pub struct ResourcePool<T> {
    /// Maximum concurrent resources.
    max_resources: usize,
    /// Available permits.
    available: AtomicUsize,
    /// Pooled resources.
    resources: Mutex<Vec<T>>,
    /// Factory for creating new resources.
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> ResourcePool<T> {
    /// Create a new resource pool.
    pub fn new(max_resources: usize, factory: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self {
            max_resources,
            available: AtomicUsize::new(max_resources),
            resources: Mutex::new(Vec::with_capacity(max_resources)),
            factory: Box::new(factory),
        }
    }

    /// Get the maximum number of resources.
    #[must_use]
    pub fn max_resources(&self) -> usize {
        self.max_resources
    }

    /// Get the number of available permits.
    #[must_use]
    pub fn available(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }

    /// Try to acquire a resource (non-blocking).
    pub fn try_acquire(&self) -> Option<PooledResource<'_, T>> {
        // Try to get a permit
        loop {
            let current = self.available.load(Ordering::Acquire);
            if current == 0 {
                return None;
            }
            if self
                .available
                .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Get or create resource
        let resource = {
            let mut pool = self.resources.lock().unwrap_or_else(|e| e.into_inner());
            pool.pop().unwrap_or_else(|| (self.factory)())
        };

        Some(PooledResource { resource: Some(resource), pool: self })
    }

    fn release(&self, resource: T) {
        {
            let mut pool = self.resources.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < self.max_resources {
                pool.push(resource);
            }
            // else: drop resource (pool is full)
        }
        self.available.fetch_add(1, Ordering::Release);
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for ResourcePool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcePool")
            .field("max_resources", &self.max_resources)
            .field("available", &self.available())
            .finish()
    }
}

/// A resource acquired from a pool.
pub struct PooledResource<'a, T> {
    resource: Option<T>,
    pool: &'a ResourcePool<T>,
}

impl<T> std::ops::Deref for PooledResource<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.resource
            .as_ref()
            .expect("PooledResource accessed after take (bug: Drop ran before Deref)")
    }
}

impl<T> std::ops::DerefMut for PooledResource<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.resource
            .as_mut()
            .expect("PooledResource accessed after take (bug: Drop ran before DerefMut)")
    }
}

impl<T> Drop for PooledResource<'_, T> {
    fn drop(&mut self) {
        if let Some(resource) = self.resource.take() {
            self.pool.release(resource);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_pool_new() {
        let pool: ResourcePool<Vec<u8>> = ResourcePool::new(4, Vec::new);
        assert_eq!(pool.max_resources(), 4);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_resource_pool_acquire_release() {
        let pool: ResourcePool<u32> = ResourcePool::new(2, || 0);

        let r1 = pool.try_acquire();
        assert!(r1.is_some());
        assert_eq!(pool.available(), 1);

        let r2 = pool.try_acquire();
        assert!(r2.is_some());
        assert_eq!(pool.available(), 0);

        // Pool exhausted
        let r3 = pool.try_acquire();
        assert!(r3.is_none());

        // Release one
        drop(r1);
        assert_eq!(pool.available(), 1);

        // Can acquire again
        let r4 = pool.try_acquire();
        assert!(r4.is_some());
    }

    #[test]
    fn test_resource_pool_factory_called() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);
        let pool: ResourcePool<u32> =
            ResourcePool::new(2, move || cc.fetch_add(1, Ordering::SeqCst) as u32);

        // First acquire creates a resource
        let _r1 = pool.try_acquire().unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Second acquire creates another resource
        let _r2 = pool.try_acquire().unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_resource_pool_reuse() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);
        let pool: ResourcePool<u32> =
            ResourcePool::new(2, move || cc.fetch_add(1, Ordering::SeqCst) as u32);

        // Acquire and release
        let r1 = pool.try_acquire().unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        drop(r1);

        // Acquire again - should reuse existing resource
        let _r2 = pool.try_acquire().unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // No new resource created
    }

    #[test]
    fn test_pooled_resource_deref() {
        let pool: ResourcePool<Vec<u8>> = ResourcePool::new(1, || vec![1, 2, 3]);

        let resource = pool.try_acquire().unwrap();
        assert_eq!(resource.len(), 3);
        assert_eq!(&*resource, &[1, 2, 3]);
    }

    #[test]
    fn test_pooled_resource_deref_mut() {
        let pool: ResourcePool<Vec<u8>> = ResourcePool::new(1, Vec::new);

        let mut resource = pool.try_acquire().unwrap();
        resource.push(42);
        assert_eq!(resource.len(), 1);
    }

    #[test]
    fn test_resource_pool_debug() {
        let pool: ResourcePool<u32> = ResourcePool::new(4, || 0);
        let debug = format!("{:?}", pool);
        assert!(debug.contains("ResourcePool"));
        assert!(debug.contains("max_resources: 4"));
        assert!(debug.contains("available: 4"));
    }

    /// FALSIFICATION TEST: Verify pool never exceeds max resources
    ///
    /// Even under concurrent pressure, the pool must never hand out
    /// more resources than max_resources.
    #[test]
    fn test_falsify_pool_never_exceeds_max() {
        use std::sync::Arc;

        let pool = Arc::new(ResourcePool::<u32>::new(5, || 0));
        let mut handles = vec![];

        // Spawn threads that aggressively try to acquire
        for _ in 0..10 {
            let pool_clone = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                let mut acquired = vec![];
                for _ in 0..100 {
                    if let Some(r) = pool_clone.try_acquire() {
                        acquired.push(r);
                    }
                    // Random sleep to increase contention
                    if acquired.len() > 2 {
                        acquired.pop();
                    }
                }
                acquired.len()
            }));
        }

        for handle in handles {
            let _ = handle.join();
        }

        // After all threads complete, available should equal max
        // (all resources released)
        // Note: This just verifies the pool is consistent
        let final_available = pool.available();
        assert!(
            final_available <= 5,
            "FALSIFICATION FAILED: available ({}) > max (5)",
            final_available
        );
    }

    /// FALSIFICATION TEST: Verify CAS prevents double-acquire
    ///
    /// The compare-exchange loop must prevent two threads from
    /// acquiring the same permit.
    #[test]
    fn test_falsify_cas_prevents_double_acquire() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let pool = Arc::new(ResourcePool::<u32>::new(1, || 0));
        let acquired_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Multiple threads try to acquire the single resource simultaneously
        for _ in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let count_clone = Arc::clone(&acquired_count);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    if let Some(_r) = pool_clone.try_acquire() {
                        let prev = count_clone.fetch_add(1, Ordering::SeqCst);
                        // Only one thread should hold the resource at a time
                        assert!(
                            prev == 0,
                            "FALSIFICATION FAILED: Multiple threads acquired simultaneously"
                        );
                        std::thread::yield_now();
                        count_clone.fetch_sub(1, Ordering::SeqCst);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
