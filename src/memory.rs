//! Memory management for hot expert pinning
//!
//! Per Dean & Barroso (2013) "The Tail at Scale", page faults cause
//! tail latency. This module provides mlock support for hot experts.

/// Configuration for memory pinning
#[derive(Debug, Clone, Default)]
pub struct MlockConfig {
    /// Whether to attempt memory locking
    pub enabled: bool,
    /// Maximum bytes to lock (0 = unlimited)
    pub max_locked_bytes: usize,
}

/// Result of mlock operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlockResult {
    /// Successfully locked memory
    Locked,
    /// Mlock disabled in config
    Disabled,
    /// Failed to lock (insufficient privileges)
    InsufficientPrivileges,
    /// Failed to lock (resource limit)
    ResourceLimit,
    /// Platform not supported
    Unsupported,
}

/// Memory region that can be pinned
pub struct PinnedRegion {
    /// Pointer to locked memory (for tracking)
    ptr: *const u8,
    /// Size of locked region
    len: usize,
    /// Whether actually locked
    locked: bool,
}

// Safety: PinnedRegion only holds a pointer for tracking, doesn't access it
unsafe impl Send for PinnedRegion {}
unsafe impl Sync for PinnedRegion {}

impl PinnedRegion {
    /// Create a pinned region (attempts mlock)
    ///
    /// # Safety
    ///
    /// Caller must ensure ptr points to valid memory for len bytes.
    #[must_use]
    // SAFETY: Caller must satisfy the documented preconditions
    pub unsafe fn new(ptr: *const u8, len: usize, config: &MlockConfig) -> (Self, MlockResult) {
        if !config.enabled {
            return (
                Self {
                    ptr,
                    len,
                    locked: false,
                },
                MlockResult::Disabled,
            );
        }

        if config.max_locked_bytes > 0 && len > config.max_locked_bytes {
            return (
                Self {
                    ptr,
                    len,
                    locked: false,
                },
                MlockResult::ResourceLimit,
            );
        }

        let result = Self::mlock_impl(ptr, len);
        let locked = result == MlockResult::Locked;
        (Self { ptr, len, locked }, result)
    }

    /// Check if memory is locked
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Get size of region
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if region is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(target_family = "unix")]
    fn mlock_impl(ptr: *const u8, len: usize) -> MlockResult {
        // Safety: ptr and len validated by caller
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let result = unsafe { libc::mlock(ptr.cast(), len) };
        if result == 0 {
            MlockResult::Locked
        } else {
            // Check errno for specific error
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if errno == libc::EPERM {
                MlockResult::InsufficientPrivileges
            } else {
                MlockResult::ResourceLimit
            }
        }
    }

    #[cfg(not(target_family = "unix"))]
    fn mlock_impl(_ptr: *const u8, _len: usize) -> MlockResult {
        MlockResult::Unsupported
    }
}

impl Drop for PinnedRegion {
    fn drop(&mut self) {
        if self.locked {
            #[cfg(target_family = "unix")]
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                libc::munlock(self.ptr.cast(), self.len);
            }
        }
    }
}

/// Expert tier classification for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTier {
    /// Hot tier - pinned in memory, never swapped
    Hot,
    /// Warm tier - in memory but may be swapped
    Warm,
    /// Cold tier - on disk, loaded on demand
    Cold,
}

impl ExpertTier {
    /// Determine tier based on access frequency
    #[must_use]
    pub fn from_access_count(count: usize, hot_threshold: usize, warm_threshold: usize) -> Self {
        if count >= hot_threshold {
            Self::Hot
        } else if count >= warm_threshold {
            Self::Warm
        } else {
            Self::Cold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlock_disabled_and_resource_limit() {
        // Disabled config
        let config_off = MlockConfig {
            enabled: false,
            max_locked_bytes: 0,
        };
        let data = vec![0u8; 1024];
        // SAFETY: data is valid
        let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config_off) };
        assert_eq!(result, MlockResult::Disabled);
        assert!(!region.is_locked());

        // Resource limit exceeded
        let config_limit = MlockConfig {
            enabled: true,
            max_locked_bytes: 100,
        };
        // SAFETY: data is valid
        let (r2, res2) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config_limit) };
        assert_eq!(res2, MlockResult::ResourceLimit);
        assert!(!r2.is_locked());
    }

    #[test]
    fn test_expert_tier_all_cases() {
        // Basic classification
        assert_eq!(
            ExpertTier::from_access_count(1000, 100, 10),
            ExpertTier::Hot
        );
        assert_eq!(ExpertTier::from_access_count(50, 100, 10), ExpertTier::Warm);
        assert_eq!(ExpertTier::from_access_count(5, 100, 10), ExpertTier::Cold);
        // Boundaries
        assert_eq!(ExpertTier::from_access_count(100, 100, 10), ExpertTier::Hot);
        assert_eq!(ExpertTier::from_access_count(99, 100, 10), ExpertTier::Warm);
        assert_eq!(ExpertTier::from_access_count(10, 100, 10), ExpertTier::Warm);
        assert_eq!(ExpertTier::from_access_count(9, 100, 10), ExpertTier::Cold);
        // Edge: zero thresholds
        assert_eq!(ExpertTier::from_access_count(0, 0, 0), ExpertTier::Hot);
        // Edge: same thresholds
        assert_eq!(ExpertTier::from_access_count(10, 10, 10), ExpertTier::Hot);
        assert_eq!(ExpertTier::from_access_count(9, 10, 10), ExpertTier::Cold);
        // Edge: max values
        assert_eq!(
            ExpertTier::from_access_count(usize::MAX, usize::MAX, usize::MAX),
            ExpertTier::Hot
        );
    }

    #[test]
    fn test_pinned_region_len_empty() {
        let config = MlockConfig::default();
        let data = vec![0u8; 1024];
        // SAFETY: data is valid
        let (region, _) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_eq!(region.len(), 1024);
        assert!(!region.is_empty());
        let empty: Vec<u8> = vec![];
        // SAFETY: empty slice is valid
        let (er, _) = unsafe { PinnedRegion::new(empty.as_ptr(), empty.len(), &config) };
        assert!(er.is_empty());
    }

    #[test]
    fn test_mlock_config_default_and_mutation() {
        let config = MlockConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_locked_bytes, 0);
        let mut m = MlockConfig::default();
        m.enabled = true;
        m.max_locked_bytes = 4096;
        assert!(m.enabled);
        assert_eq!(m.max_locked_bytes, 4096);
    }

    #[test]
    fn test_mlock_result_equality_and_inequality() {
        assert_eq!(MlockResult::Locked, MlockResult::Locked);
        assert_ne!(MlockResult::Locked, MlockResult::Disabled);
        let variants = [
            MlockResult::Locked,
            MlockResult::Disabled,
            MlockResult::InsufficientPrivileges,
            MlockResult::ResourceLimit,
            MlockResult::Unsupported,
        ];
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                assert_eq!(i == j, v1 == v2);
            }
        }
    }

    #[test]
    fn test_mlock_enabled_actual() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 0,
        };
        let data: [u8; 64] = [0u8; 64];
        // SAFETY: data is valid
        let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert!(matches!(
            result,
            MlockResult::Locked
                | MlockResult::InsufficientPrivileges
                | MlockResult::ResourceLimit
                | MlockResult::Unsupported
        ));
        assert_eq!(region.len(), 64);
    }

    #[test]
    fn test_debug_and_clone_traits() {
        // MlockConfig
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 1024,
        };
        assert!(format!("{:?}", config).contains("MlockConfig"));
        let c2 = config.clone();
        assert_eq!(config.enabled, c2.enabled);
        // MlockResult
        for r in [
            MlockResult::Locked,
            MlockResult::Disabled,
            MlockResult::InsufficientPrivileges,
            MlockResult::ResourceLimit,
            MlockResult::Unsupported,
        ] {
            assert!(!format!("{:?}", r).is_empty());
            assert_eq!(r.clone(), r);
        }
        // ExpertTier
        for t in [ExpertTier::Hot, ExpertTier::Warm, ExpertTier::Cold] {
            assert!(!format!("{:?}", t).is_empty());
            let copied = t;
            assert_eq!(t, copied);
        }
    }

    #[test]
    fn test_mlock_boundary_conditions() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 1024,
        };
        // At limit
        let data = vec![0u8; 1024];
        // SAFETY: data is valid
        let (_, r1) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
        assert_ne!(r1, MlockResult::Disabled);
        // Over limit
        let over = vec![0u8; 1025];
        // SAFETY: data is valid
        let (r2, res) = unsafe { PinnedRegion::new(over.as_ptr(), over.len(), &config) };
        assert_eq!(res, MlockResult::ResourceLimit);
        assert!(!r2.is_locked());
        // Large allocation
        let config2 = MlockConfig {
            enabled: true,
            max_locked_bytes: 0,
        };
        let large = vec![0u8; 64 * 1024];
        // SAFETY: data is valid
        let (r3, _) = unsafe { PinnedRegion::new(large.as_ptr(), large.len(), &config2) };
        assert_eq!(r3.len(), 64 * 1024);
    }

    #[test]
    fn test_multiple_regions_and_drop() {
        let config = MlockConfig {
            enabled: false,
            max_locked_bytes: 0,
        };
        let d1 = vec![1u8; 512];
        let d2 = vec![2u8; 256];
        // SAFETY: data is valid
        let (r1, res1) = unsafe { PinnedRegion::new(d1.as_ptr(), d1.len(), &config) };
        let (r2, res2) = unsafe { PinnedRegion::new(d2.as_ptr(), d2.len(), &config) };
        assert_eq!(res1, MlockResult::Disabled);
        assert_eq!(res2, MlockResult::Disabled);
        assert_eq!(r1.len(), 512);
        assert_eq!(r2.len(), 256);
        // Drop behavior
        let data = vec![42u8; 256];
        {
            // SAFETY: data is valid
            let (region, _) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
            assert!(!region.is_locked());
        }
        assert_eq!(data[0], 42);
    }

    #[test]
    fn test_drop_with_mlock_attempt() {
        let config = MlockConfig {
            enabled: true,
            max_locked_bytes: 0,
        };
        let data = vec![99u8; 128];
        {
            // SAFETY: data is valid
            let (region, result) = unsafe { PinnedRegion::new(data.as_ptr(), data.len(), &config) };
            assert_eq!(region.is_locked(), result == MlockResult::Locked);
        }
        assert_eq!(data[0], 99);
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<PinnedRegion>();
        assert_sync::<PinnedRegion>();
        fn assert_both<T: Send + Sync>() {}
        assert_both::<MlockConfig>();
        assert_both::<ExpertTier>();
    }

    #[test]
    fn test_config_various_values() {
        for (en, mb) in [(true, 0), (false, 1), (true, usize::MAX)] {
            let c = MlockConfig {
                enabled: en,
                max_locked_bytes: mb,
            };
            let c2 = c.clone();
            assert_eq!(c.enabled, c2.enabled);
            assert_eq!(c.max_locked_bytes, c2.max_locked_bytes);
        }
    }
}
