//! Memory Management Primitives
//!
//! Cache line alignment, direct I/O buffers, memory advice, and prefetch utilities.

use crate::error::TruenoError;

// ----------------------------------------------------------------------------
// LCP-06: Cache Line Padding
// ----------------------------------------------------------------------------

/// Cache line size (64 bytes on most modern CPUs).
pub const CACHE_LINE_SIZE: usize = 64;

/// Number of f32 values per cache line.
pub const CACHE_LINE_SIZE_F32: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>();

/// Cache-line aligned wrapper to prevent false sharing.
///
/// # Example
/// ```rust
/// use trueno::brick::CacheAligned;
/// use std::sync::atomic::AtomicU64;
///
/// let aligned: CacheAligned<AtomicU64> = CacheAligned::new(AtomicU64::new(0));
/// assert_eq!(std::mem::align_of_val(&aligned), 64);
/// ```
#[repr(align(64))]
#[derive(Debug)]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get a reference to the inner value.
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Consume the wrapper and return the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T: Clone> Clone for CacheAligned<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// ----------------------------------------------------------------------------
// LCP-02: Direct I/O Alignment
// ----------------------------------------------------------------------------

/// Memory alignment for direct I/O (4KB page aligned).
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Check if a pointer is aligned for direct I/O.
#[must_use]
pub fn is_direct_io_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(DIRECT_IO_ALIGNMENT)
}

/// Aligned buffer for direct I/O operations.
#[cfg(not(target_arch = "wasm32"))]
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

#[cfg(not(target_arch = "wasm32"))]
impl AlignedBuffer {
    /// Allocate a new aligned buffer.
    ///
    /// # Errors
    /// Returns an error if allocation fails.
    pub fn new(size: usize) -> Result<Self, TruenoError> {
        use std::alloc::{alloc_zeroed, Layout};

        let layout = Layout::from_size_align(size, DIRECT_IO_ALIGNMENT)
            .map_err(|e| TruenoError::InvalidInput(format!("invalid alignment: {e}")))?;

        // SAFETY: layout is valid, pointer was allocated with matching layout
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(TruenoError::InvalidInput("allocation failed".into()));
        }

        Ok(Self { ptr, len: size, layout })
    }

    /// Get the buffer as a slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null and points to `self.len` bytes allocated in `new()`;
        // the allocation lives for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null and points to `self.len` bytes allocated in `new()`;
        // `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get the mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the buffer length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: self.ptr was allocated with std::alloc::alloc_zeroed using self.layout
        // in AlignedBuffer::new(); dealloc uses the matching layout.
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
// SAFETY: type invariants ensure trait contract is upheld
unsafe impl Send for AlignedBuffer {}

#[cfg(not(target_arch = "wasm32"))]
// SAFETY: type invariants ensure trait contract is upheld
unsafe impl Sync for AlignedBuffer {}

// ----------------------------------------------------------------------------
// LCP-03: Memory Advice (madvise patterns)
// ----------------------------------------------------------------------------

/// Memory advice for mmap regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// Sequential access (enable readahead)
    Sequential,
    /// Random access (disable readahead)
    Random,
    /// Will need soon (prefetch)
    WillNeed,
    /// Don't need (can be paged out)
    DontNeed,
}

// Linux madvise constants (from linux/mman.h)
#[cfg(target_os = "linux")]
const MADV_SEQUENTIAL: i32 = 2;
#[cfg(target_os = "linux")]
const MADV_RANDOM: i32 = 1;
#[cfg(target_os = "linux")]
const MADV_WILLNEED: i32 = 3;
#[cfg(target_os = "linux")]
const MADV_DONTNEED: i32 = 4;

/// Apply memory advice to a region (Linux only).
///
/// # Safety
/// The pointer must be valid and the length must not exceed the mapped region.
#[cfg(target_os = "linux")]
// SAFETY: Caller ensures pointer is valid and length does not exceed the mapped region
pub unsafe fn madvise_region(
    addr: *mut u8,
    len: usize,
    advice: MemoryAdvice,
) -> std::io::Result<()> {
    unsafe {
        // madvise syscall number is 28 on x86_64
        #[cfg(target_arch = "x86_64")]
        const SYS_MADVISE: i64 = 28;
        #[cfg(target_arch = "aarch64")]
        const SYS_MADVISE: i64 = 233;

        let advice_flag: i32 = match advice {
            MemoryAdvice::Sequential => MADV_SEQUENTIAL,
            MemoryAdvice::Random => MADV_RANDOM,
            MemoryAdvice::WillNeed => MADV_WILLNEED,
            MemoryAdvice::DontNeed => MADV_DONTNEED,
        };

        let ret: i64;
        #[cfg(target_arch = "x86_64")]
        {
            core::arch::asm!(
                "syscall",
                inout("rax") SYS_MADVISE => ret,
                in("rdi") addr as usize,
                in("rsi") len,
                in("rdx") advice_flag as i64,
                out("rcx") _,
                out("r11") _,
                options(nostack)
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            core::arch::asm!(
                "svc 0",
                inout("x8") SYS_MADVISE => _,
                inout("x0") addr as usize => ret,
                in("x1") len,
                in("x2") advice_flag as i64,
                options(nostack)
            );
        }

        if ret < 0 {
            return Err(std::io::Error::from_raw_os_error(-ret as i32));
        }

        Ok(())
    }
}

/// Stub for non-Linux platforms.
#[cfg(not(target_os = "linux"))]
// SAFETY: No-op stub, no actual unsafe operations performed
pub unsafe fn madvise_region(
    _addr: *mut u8,
    _len: usize,
    _advice: MemoryAdvice,
) -> std::io::Result<()> {
    Ok(()) // No-op on non-Linux
}

/// Apply dual-level prefetch strategy (WILLNEED + RANDOM).
///
/// This is the llama.cpp pattern for model loading:
/// 1. MADV_WILLNEED: Tell kernel to prefetch the data
/// 2. MADV_RANDOM: Disable readahead (model access is random)
///
/// # Safety
/// The pointer must be valid and the length must not exceed the mapped region.
#[cfg(target_os = "linux")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn prefetch_for_inference(addr: *mut u8, len: usize) -> std::io::Result<()> {
    unsafe {
        // First: tell kernel we'll need this data
        madvise_region(addr, len, MemoryAdvice::WillNeed)?;
        // Second: hint random access pattern (disables readahead waste)
        madvise_region(addr, len, MemoryAdvice::Random)?;
        Ok(())
    }
}

/// Stub for non-Linux platforms.
#[cfg(not(target_os = "linux"))]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn prefetch_for_inference(_addr: *mut u8, _len: usize) -> std::io::Result<()> {
    Ok(()) // No-op on non-Linux
}

// ----------------------------------------------------------------------------
// LCP-11: Prefetch with Locality Hints
// ----------------------------------------------------------------------------

/// Prefetch locality hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLocality {
    /// No temporal locality (use once, don't pollute cache)
    None = 0,
    /// Low temporal locality (use a few times)
    Low = 1,
    /// Moderate temporal locality
    Moderate = 2,
    /// High temporal locality (keep in all cache levels)
    High = 3,
}

/// Prefetch data into cache.
///
/// # Safety
/// The pointer must be valid for reading.
#[inline]
#[cfg(target_arch = "x86_64")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn prefetch_ptr<T>(ptr: *const T, locality: PrefetchLocality) {
    unsafe {
        use core::arch::x86_64::*;
        match locality {
            PrefetchLocality::None => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
            PrefetchLocality::Low => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            PrefetchLocality::Moderate => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            PrefetchLocality::High => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
        }
    }
}

/// Prefetch data into cache (ARM64).
#[inline]
#[cfg(target_arch = "aarch64")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn prefetch_ptr<T>(ptr: *const T, _locality: PrefetchLocality) {
    // ARM prefetch (PRFM instruction) - locality hints are limited
    core::arch::asm!(
        "prfm pldl1keep, [{ptr}]",
        ptr = in(reg) ptr,
        options(nostack, preserves_flags)
    );
}

/// Fallback for other architectures.
#[inline]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn prefetch_ptr<T>(_ptr: *const T, _locality: PrefetchLocality) {
    // No-op on unsupported architectures
}

/// Prefetch a slice of data.
///
/// Prefetches each cache line in the slice.
#[inline]
pub fn prefetch_slice<T>(slice: &[T], locality: PrefetchLocality) {
    let ptr = slice.as_ptr() as *const u8;
    let len = std::mem::size_of_val(slice);

    for offset in (0..len).step_by(CACHE_LINE_SIZE) {
        // SAFETY: ptr.add(offset) is bounded by the slice length; prefetch is
        // a hint and does not dereference memory.
        unsafe {
            prefetch_ptr(ptr.add(offset), locality);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_alignment() {
        let aligned: CacheAligned<u64> = CacheAligned::new(42);
        assert_eq!(std::mem::align_of_val(&aligned), 64);
    }

    #[test]
    fn test_cache_aligned_value() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(*aligned.get(), 42);
    }

    #[test]
    fn test_cache_aligned_get_mut() {
        let mut aligned = CacheAligned::new(42u64);
        *aligned.get_mut() = 100;
        assert_eq!(*aligned.get(), 100);
    }

    #[test]
    fn test_cache_aligned_into_inner() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(aligned.into_inner(), 42);
    }

    #[test]
    fn test_cache_aligned_default() {
        let aligned: CacheAligned<u64> = CacheAligned::default();
        assert_eq!(*aligned.get(), 0);
    }

    #[test]
    fn test_cache_aligned_clone() {
        let aligned = CacheAligned::new(42u64);
        let cloned = aligned.clone();
        assert_eq!(*cloned.get(), 42);
    }

    #[test]
    fn test_cache_line_size_f32() {
        assert_eq!(CACHE_LINE_SIZE_F32, 16); // 64 / 4 = 16
    }

    #[test]
    fn test_direct_io_alignment() {
        assert_eq!(DIRECT_IO_ALIGNMENT, 4096);
    }

    #[test]
    fn test_is_direct_io_aligned() {
        let aligned_addr: usize = 4096 * 10;
        let unaligned_addr: usize = 4096 * 10 + 1;

        assert!(is_direct_io_aligned(aligned_addr as *const u8));
        assert!(!is_direct_io_aligned(unaligned_addr as *const u8));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::new(4096).unwrap();
        assert_eq!(buffer.len(), 4096);
        assert!(!buffer.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_aligned_buffer_zeroed() {
        let buffer = AlignedBuffer::new(1024).unwrap();
        let slice = buffer.as_slice();
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_aligned_buffer_write() {
        let mut buffer = AlignedBuffer::new(1024).unwrap();
        buffer.as_mut_slice()[0] = 42;
        assert_eq!(buffer.as_slice()[0], 42);
    }

    #[test]
    fn test_memory_advice_eq() {
        assert_eq!(MemoryAdvice::Sequential, MemoryAdvice::Sequential);
        assert_ne!(MemoryAdvice::Sequential, MemoryAdvice::Random);
    }

    #[test]
    fn test_prefetch_locality_values() {
        assert_eq!(PrefetchLocality::None as u8, 0);
        assert_eq!(PrefetchLocality::Low as u8, 1);
        assert_eq!(PrefetchLocality::Moderate as u8, 2);
        assert_eq!(PrefetchLocality::High as u8, 3);
    }

    #[test]
    fn test_prefetch_slice_empty() {
        let empty: &[f32] = &[];
        prefetch_slice(empty, PrefetchLocality::High);
        // Should not panic
    }

    #[test]
    fn test_prefetch_slice_small() {
        let data = [1.0f32; 8];
        prefetch_slice(&data, PrefetchLocality::High);
        // Should not panic
    }

    #[test]
    fn test_madvise_region_stub() {
        // On non-Linux, this is a no-op
        // SAFETY: preconditions verified by caller
        unsafe {
            let mut data = [0u8; 4096];
            let _result = madvise_region(data.as_mut_ptr(), data.len(), MemoryAdvice::WillNeed);
            #[cfg(not(target_os = "linux"))]
            assert!(_result.is_ok());
        }
    }

    #[test]
    fn test_prefetch_for_inference_stub() {
        // SAFETY: preconditions verified by caller
        unsafe {
            let mut data = [0u8; 4096];
            let _result = prefetch_for_inference(data.as_mut_ptr(), data.len());
            #[cfg(not(target_os = "linux"))]
            assert!(_result.is_ok());
        }
    }
}
