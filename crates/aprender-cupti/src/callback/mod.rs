//! Callback API for real-time CUDA event notifications.
//!
//! The callback API provides synchronous notifications of CUDA runtime and
//! driver API calls, enabling real-time monitoring and debugging.

use crate::error::CuptiResult;
use std::ffi::c_void;

/// Domain for callback subscriptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallbackDomain {
    /// CUDA Runtime API callbacks
    RuntimeApi,
    /// CUDA Driver API callbacks
    DriverApi,
    /// Resource tracking callbacks
    Resource,
    /// Synchronization callbacks
    Synchronize,
    /// NVTX (NVIDIA Tools Extension) callbacks
    Nvtx,
}

impl CallbackDomain {
    /// Get CUPTI domain ID.
    pub fn cupti_id(&self) -> u32 {
        match self {
            CallbackDomain::RuntimeApi => 1,
            CallbackDomain::DriverApi => 2,
            CallbackDomain::Resource => 3,
            CallbackDomain::Synchronize => 4,
            CallbackDomain::Nvtx => 5,
        }
    }
}

/// Callback identifier for specific API calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallbackId {
    // Runtime API
    /// cudaMalloc
    CudaMalloc,
    /// cudaFree
    CudaFree,
    /// cudaMemcpy
    CudaMemcpy,
    /// cudaMemcpyAsync
    CudaMemcpyAsync,
    /// cudaLaunchKernel
    CudaLaunchKernel,
    /// cudaDeviceSynchronize
    CudaDeviceSynchronize,
    /// cudaStreamSynchronize
    CudaStreamSynchronize,

    // Driver API
    /// cuMemAlloc
    CuMemAlloc,
    /// cuMemFree
    CuMemFree,
    /// cuLaunchKernel
    CuLaunchKernel,
    /// cuCtxSynchronize
    CuCtxSynchronize,

    // Resource
    /// Context creation
    ContextCreated,
    /// Context destruction
    ContextDestroyed,
    /// Stream creation
    StreamCreated,
    /// Stream destruction
    StreamDestroyed,
    /// Module load
    ModuleLoaded,
    /// Module unload
    ModuleUnloaded,

    /// Custom/other callback
    Other(u32),
}

impl CallbackId {
    /// Get CUPTI callback ID.
    pub fn cupti_id(&self) -> u32 {
        match self {
            CallbackId::CudaMalloc => 1,
            CallbackId::CudaFree => 2,
            CallbackId::CudaMemcpy => 3,
            CallbackId::CudaMemcpyAsync => 4,
            CallbackId::CudaLaunchKernel => 5,
            CallbackId::CudaDeviceSynchronize => 6,
            CallbackId::CudaStreamSynchronize => 7,
            CallbackId::CuMemAlloc => 100,
            CallbackId::CuMemFree => 101,
            CallbackId::CuLaunchKernel => 102,
            CallbackId::CuCtxSynchronize => 103,
            CallbackId::ContextCreated => 200,
            CallbackId::ContextDestroyed => 201,
            CallbackId::StreamCreated => 202,
            CallbackId::StreamDestroyed => 203,
            CallbackId::ModuleLoaded => 204,
            CallbackId::ModuleUnloaded => 205,
            CallbackId::Other(id) => *id,
        }
    }
}

/// When the callback is invoked relative to the API call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackSite {
    /// Before API call executes
    Enter,
    /// After API call completes
    Exit,
}

/// Data passed to a callback function.
#[derive(Debug)]
pub struct CallbackData {
    /// Domain of the callback
    pub domain: CallbackDomain,
    /// Callback ID
    pub callback_id: CallbackId,
    /// Entry or exit point
    pub site: CallbackSite,
    /// Correlation ID to match enter/exit
    pub correlation_id: u64,
    /// Context handle (if applicable)
    pub context: Option<u64>,
    /// Function name (if available)
    pub function_name: Option<String>,
    /// Return value (for exit callbacks)
    pub return_value: Option<i32>,
}

/// Callback function type.
pub type CallbackFn = Box<dyn Fn(&CallbackData) + Send + Sync>;

/// Subscriber for callback events.
pub struct CallbackSubscriber {
    /// Unique subscriber ID
    id: u64,
    /// Registered callbacks by domain
    callbacks: Vec<(CallbackDomain, Option<CallbackId>, CallbackFn)>,
    /// Whether the subscriber is active
    active: bool,
}

impl CallbackSubscriber {
    /// Create a new callback subscriber.
    pub fn new() -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            callbacks: Vec::new(),
            active: false,
        }
    }

    /// Get subscriber ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Subscribe to all callbacks in a domain.
    pub fn subscribe_domain<F>(&mut self, domain: CallbackDomain, callback: F) -> CuptiResult<()>
    where
        F: Fn(&CallbackData) + Send + Sync + 'static,
    {
        self.callbacks.push((domain, None, Box::new(callback)));
        Ok(())
    }

    /// Subscribe to a specific callback.
    pub fn subscribe<F>(
        &mut self,
        domain: CallbackDomain,
        callback_id: CallbackId,
        callback: F,
    ) -> CuptiResult<()>
    where
        F: Fn(&CallbackData) + Send + Sync + 'static,
    {
        self.callbacks
            .push((domain, Some(callback_id), Box::new(callback)));
        Ok(())
    }

    /// Enable the subscriber (start receiving callbacks).
    pub fn enable(&mut self) -> CuptiResult<()> {
        self.active = true;
        // In real implementation: cuptiEnableCallback for each subscription
        Ok(())
    }

    /// Disable the subscriber (stop receiving callbacks).
    pub fn disable(&mut self) -> CuptiResult<()> {
        self.active = false;
        // In real implementation: cuptiDisableCallback
        Ok(())
    }

    /// Check if subscriber is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Internal: dispatch callback to registered handlers.
    #[doc(hidden)]
    pub fn dispatch(&self, data: &CallbackData) {
        if !self.active {
            return;
        }

        for (domain, callback_id, handler) in &self.callbacks {
            if *domain == data.domain {
                match callback_id {
                    None => handler(data),
                    Some(id) if *id == data.callback_id => handler(data),
                    _ => {}
                }
            }
        }
    }
}

impl Default for CallbackSubscriber {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating callback subscriptions.
#[derive(Default)]
pub struct CallbackBuilder {
    subscriptions: Vec<(CallbackDomain, Option<CallbackId>)>,
}

impl CallbackBuilder {
    /// Create a new callback builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to kernel launches.
    #[must_use]
    pub fn on_kernel_launch(mut self) -> Self {
        self.subscriptions
            .push((CallbackDomain::RuntimeApi, Some(CallbackId::CudaLaunchKernel)));
        self.subscriptions
            .push((CallbackDomain::DriverApi, Some(CallbackId::CuLaunchKernel)));
        self
    }

    /// Subscribe to memory operations.
    #[must_use]
    pub fn on_memory_ops(mut self) -> Self {
        self.subscriptions
            .push((CallbackDomain::RuntimeApi, Some(CallbackId::CudaMalloc)));
        self.subscriptions
            .push((CallbackDomain::RuntimeApi, Some(CallbackId::CudaFree)));
        self.subscriptions
            .push((CallbackDomain::RuntimeApi, Some(CallbackId::CudaMemcpy)));
        self
    }

    /// Subscribe to synchronization events.
    #[must_use]
    pub fn on_synchronization(mut self) -> Self {
        self.subscriptions
            .push((CallbackDomain::Synchronize, None));
        self
    }

    /// Subscribe to all runtime API calls.
    #[must_use]
    pub fn on_runtime_api(mut self) -> Self {
        self.subscriptions.push((CallbackDomain::RuntimeApi, None));
        self
    }

    /// Subscribe to all driver API calls.
    #[must_use]
    pub fn on_driver_api(mut self) -> Self {
        self.subscriptions.push((CallbackDomain::DriverApi, None));
        self
    }

    /// Build the subscriber with the given handler.
    pub fn build<F>(self, handler: F) -> CuptiResult<CallbackSubscriber>
    where
        F: Fn(&CallbackData) + Send + Sync + Clone + 'static,
    {
        let mut subscriber = CallbackSubscriber::new();
        for (domain, callback_id) in self.subscriptions {
            let h = handler.clone();
            match callback_id {
                Some(id) => subscriber.subscribe(domain, id, h)?,
                None => subscriber.subscribe_domain(domain, h)?,
            }
        }
        Ok(subscriber)
    }
}

/// Raw callback function pointer type (for FFI).
pub type RawCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    domain: u32,
    callback_id: u32,
    callback_data: *const c_void,
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_callback_domain_id() {
        assert_eq!(CallbackDomain::RuntimeApi.cupti_id(), 1);
        assert_eq!(CallbackDomain::DriverApi.cupti_id(), 2);
    }

    #[test]
    fn test_callback_subscriber() {
        let mut subscriber = CallbackSubscriber::new();
        assert!(!subscriber.is_active());

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        subscriber
            .subscribe_domain(CallbackDomain::RuntimeApi, move |_data| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();

        subscriber.enable().unwrap();
        assert!(subscriber.is_active());

        // Simulate callback
        let data = CallbackData {
            domain: CallbackDomain::RuntimeApi,
            callback_id: CallbackId::CudaMalloc,
            site: CallbackSite::Enter,
            correlation_id: 1,
            context: None,
            function_name: Some("cudaMalloc".to_string()),
            return_value: None,
        };

        subscriber.dispatch(&data);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_builder() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let subscriber = CallbackBuilder::new()
            .on_kernel_launch()
            .on_memory_ops()
            .build(move |_data| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();

        // Has subscriptions but not active yet
        assert!(!subscriber.is_active());
    }

    #[test]
    fn test_callback_filtering() {
        let mut subscriber = CallbackSubscriber::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        // Only subscribe to CudaMalloc
        subscriber
            .subscribe(
                CallbackDomain::RuntimeApi,
                CallbackId::CudaMalloc,
                move |_data| {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                },
            )
            .unwrap();

        subscriber.enable().unwrap();

        // CudaMalloc should trigger
        let malloc_data = CallbackData {
            domain: CallbackDomain::RuntimeApi,
            callback_id: CallbackId::CudaMalloc,
            site: CallbackSite::Enter,
            correlation_id: 1,
            context: None,
            function_name: None,
            return_value: None,
        };
        subscriber.dispatch(&malloc_data);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // CudaFree should NOT trigger
        let free_data = CallbackData {
            domain: CallbackDomain::RuntimeApi,
            callback_id: CallbackId::CudaFree,
            site: CallbackSite::Enter,
            correlation_id: 2,
            context: None,
            function_name: None,
            return_value: None,
        };
        subscriber.dispatch(&free_data);
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Still 1
    }
}
