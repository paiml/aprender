//! Raw FFI bindings for CUPTI.
//!
//! This module provides low-level bindings to the NVIDIA CUPTI library.
//! These are not meant to be used directly - use the safe wrappers in
//! the parent modules instead.
//!
//! # Safety
//!
//! All functions in this module are unsafe and require:
//! - Valid pointers
//! - Proper CUDA context initialization
//! - Correct parameter sizes and alignment

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use libc::{c_char, c_int, c_uint, c_void, size_t};

/// CUPTI result codes.
pub type CUptiResult = c_int;

pub const CUPTI_SUCCESS: CUptiResult = 0;
pub const CUPTI_ERROR_INVALID_PARAMETER: CUptiResult = 1;
pub const CUPTI_ERROR_OUT_OF_MEMORY: CUptiResult = 2;
pub const CUPTI_ERROR_NOT_INITIALIZED: CUptiResult = 3;
pub const CUPTI_ERROR_INVALID_CONTEXT: CUptiResult = 4;
pub const CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID: CUptiResult = 5;
pub const CUPTI_ERROR_INVALID_EVENT_ID: CUptiResult = 6;
pub const CUPTI_ERROR_INVALID_METRIC_ID: CUptiResult = 7;
pub const CUPTI_ERROR_HARDWARE: CUptiResult = 8;
pub const CUPTI_ERROR_UNKNOWN: CUptiResult = 999;

/// Activity kinds.
pub type CUpti_ActivityKind = c_uint;

pub const CUPTI_ACTIVITY_KIND_KERNEL: CUpti_ActivityKind = 1;
pub const CUPTI_ACTIVITY_KIND_MEMCPY: CUpti_ActivityKind = 2;
pub const CUPTI_ACTIVITY_KIND_MEMSET: CUpti_ActivityKind = 3;
pub const CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: CUpti_ActivityKind = 4;
pub const CUPTI_ACTIVITY_KIND_CONTEXT: CUpti_ActivityKind = 5;
pub const CUPTI_ACTIVITY_KIND_STREAM: CUpti_ActivityKind = 6;
pub const CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: CUpti_ActivityKind = 7;

/// Callback domains.
pub type CUpti_CallbackDomain = c_uint;

pub const CUPTI_CB_DOMAIN_RUNTIME_API: CUpti_CallbackDomain = 1;
pub const CUPTI_CB_DOMAIN_DRIVER_API: CUpti_CallbackDomain = 2;
pub const CUPTI_CB_DOMAIN_RESOURCE: CUpti_CallbackDomain = 3;
pub const CUPTI_CB_DOMAIN_SYNCHRONIZE: CUpti_CallbackDomain = 4;
pub const CUPTI_CB_DOMAIN_NVTX: CUpti_CallbackDomain = 5;

/// Subscriber handle.
pub type CUpti_SubscriberHandle = *mut c_void;

/// Callback function type.
pub type CUpti_CallbackFunc = Option<
    unsafe extern "C" fn(
        userdata: *mut c_void,
        domain: CUpti_CallbackDomain,
        cbid: c_uint,
        cbdata: *const c_void,
    ),
>;

/// Activity buffer request callback.
pub type CUpti_BufferRequestFunc =
    Option<unsafe extern "C" fn(buffer: *mut *mut u8, size: *mut size_t, maxNumRecords: *mut size_t)>;

/// Activity buffer completed callback.
pub type CUpti_BufferCompletedFunc = Option<
    unsafe extern "C" fn(
        context: *mut c_void,
        streamId: c_uint,
        buffer: *mut u8,
        size: size_t,
        validSize: size_t,
    ),
>;

/// Activity record header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUpti_Activity {
    pub kind: CUpti_ActivityKind,
}

/// Kernel activity record.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUpti_ActivityKernel {
    pub kind: CUpti_ActivityKind,
    pub cacheConfigRequested: c_uint,
    pub cacheConfigExecuted: c_uint,
    pub registersPerThread: c_uint,
    pub staticSharedMemory: c_int,
    pub dynamicSharedMemory: c_int,
    pub gridX: c_int,
    pub gridY: c_int,
    pub gridZ: c_int,
    pub blockX: c_int,
    pub blockY: c_int,
    pub blockZ: c_int,
    pub start: u64,
    pub end: u64,
    pub deviceId: c_uint,
    pub contextId: c_uint,
    pub streamId: c_uint,
    pub correlationId: c_uint,
    pub pad: c_uint,
    pub name: *const c_char,
}

/// Memcpy activity record.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUpti_ActivityMemcpy {
    pub kind: CUpti_ActivityKind,
    pub copyKind: c_uint,
    pub srcKind: c_uint,
    pub dstKind: c_uint,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub deviceId: c_uint,
    pub contextId: c_uint,
    pub streamId: c_uint,
    pub correlationId: c_uint,
    pub srcDeviceId: c_uint,
    pub dstDeviceId: c_uint,
}

/// Synchronization activity record.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUpti_ActivitySynchronization {
    pub kind: CUpti_ActivityKind,
    pub syncKind: c_uint,
    pub start: u64,
    pub end: u64,
    pub deviceId: c_uint,
    pub contextId: c_uint,
    pub streamId: c_uint,
    pub correlationId: c_uint,
}

// CUPTI function declarations - these would be linked dynamically
#[cfg(feature = "bindgen")]
extern "C" {
    /// Initialize CUPTI.
    pub fn cuptiGetVersion(version: *mut c_uint) -> CUptiResult;

    /// Activity API.
    pub fn cuptiActivityEnable(kind: CUpti_ActivityKind) -> CUptiResult;
    pub fn cuptiActivityDisable(kind: CUpti_ActivityKind) -> CUptiResult;
    pub fn cuptiActivityFlushAll(flag: c_uint) -> CUptiResult;
    pub fn cuptiActivityRegisterCallbacks(
        funcBufferRequested: CUpti_BufferRequestFunc,
        funcBufferCompleted: CUpti_BufferCompletedFunc,
    ) -> CUptiResult;
    pub fn cuptiActivityGetNextRecord(
        buffer: *mut u8,
        validBufferSizeBytes: size_t,
        record: *mut *mut CUpti_Activity,
    ) -> CUptiResult;

    /// Callback API.
    pub fn cuptiSubscribe(
        subscriber: *mut CUpti_SubscriberHandle,
        callback: CUpti_CallbackFunc,
        userdata: *mut c_void,
    ) -> CUptiResult;
    pub fn cuptiUnsubscribe(subscriber: CUpti_SubscriberHandle) -> CUptiResult;
    pub fn cuptiEnableDomain(
        enable: c_uint,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
    ) -> CUptiResult;
    pub fn cuptiEnableCallback(
        enable: c_uint,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
        cbid: c_uint,
    ) -> CUptiResult;

    /// Metric API.
    pub fn cuptiDeviceGetNumMetrics(device: c_uint, numMetrics: *mut c_uint) -> CUptiResult;
    pub fn cuptiMetricGetValue(
        device: c_uint,
        metric: c_uint,
        eventIdArraySizeBytes: size_t,
        eventIdArray: *const c_uint,
        eventValueArraySizeBytes: size_t,
        eventValueArray: *const u64,
        durationNs: u64,
        metricValue: *mut u64,
    ) -> CUptiResult;
}

/// Dynamic library loader for CUPTI.
pub struct CuptiLibrary {
    #[allow(dead_code)]
    handle: *mut c_void,
}

impl CuptiLibrary {
    /// Attempt to load CUPTI library.
    ///
    /// # Safety
    ///
    /// This function loads a dynamic library which may have side effects.
    pub unsafe fn load() -> Option<Self> {
        // Try to find libcupti.so
        #[cfg(target_os = "linux")]
        {
            let paths = [
                "/usr/local/cuda/lib64/libcupti.so\0",
                "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so\0",
                "/usr/lib/x86_64-linux-gnu/libcupti.so\0",
                "/opt/cuda/lib64/libcupti.so\0",
                "libcupti.so\0",
            ];

            for path in &paths {
                let handle = libc::dlopen(path.as_ptr() as *const c_char, libc::RTLD_NOW);
                if !handle.is_null() {
                    return Some(Self { handle });
                }
            }
        }

        None
    }

    /// Get a function pointer from the library.
    ///
    /// # Safety
    ///
    /// The caller must ensure the symbol exists and has the correct type.
    #[allow(dead_code)]
    pub unsafe fn get_fn(&self, name: &str) -> Option<*mut c_void> {
        let c_name = std::ffi::CString::new(name).ok()?;
        #[cfg(target_os = "linux")]
        {
            let ptr = libc::dlsym(self.handle, c_name.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = c_name;
            None
        }
    }
}

impl Drop for CuptiLibrary {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe {
            if !self.handle.is_null() {
                libc::dlclose(self.handle);
            }
        }
    }
}

// Safety: The library handle is thread-safe once loaded
unsafe impl Send for CuptiLibrary {}
unsafe impl Sync for CuptiLibrary {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_codes() {
        assert_eq!(CUPTI_SUCCESS, 0);
        assert_eq!(CUPTI_ERROR_INVALID_PARAMETER, 1);
    }

    #[test]
    fn test_activity_kinds() {
        assert_eq!(CUPTI_ACTIVITY_KIND_KERNEL, 1);
        assert_eq!(CUPTI_ACTIVITY_KIND_MEMCPY, 2);
    }

    #[test]
    fn test_struct_sizes() {
        // Ensure structs have reasonable sizes (sanity check)
        assert!(std::mem::size_of::<CUpti_Activity>() >= 4);
        assert!(std::mem::size_of::<CUpti_ActivityKernel>() > std::mem::size_of::<CUpti_Activity>());
    }
}
