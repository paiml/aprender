//! Activity tracing for CUDA operations.
//!
//! CUPTI activity API enables asynchronous collection of GPU activity data
//! including kernel execution, memory copies, and synchronization events.

use std::time::Duration;

/// Types of CUDA activity that can be traced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivityKind {
    /// CUDA kernel execution
    Kernel,
    /// Memory copy operations (H2D, D2H, D2D)
    MemoryCopy,
    /// Memory set operations
    MemorySet,
    /// Synchronization events
    Synchronization,
    /// CUDA stream activity
    Stream,
    /// CUDA context activity
    Context,
    /// Unified memory activity
    UnifiedMemory,
    /// PC sampling activity (requires pc-sampling feature)
    PcSampling,
}

/// A recorded activity from CUPTI.
#[derive(Debug, Clone)]
pub enum ActivityRecord {
    /// Kernel execution record
    Kernel(KernelRecord),
    /// Memory copy record
    Memcpy(MemcpyRecord),
    /// Synchronization record
    Sync(SyncRecord),
    /// Generic activity record
    Generic(GenericRecord),
}

impl ActivityRecord {
    /// Get the activity kind.
    pub fn kind(&self) -> ActivityKind {
        match self {
            ActivityRecord::Kernel(_) => ActivityKind::Kernel,
            ActivityRecord::Memcpy(_) => ActivityKind::MemoryCopy,
            ActivityRecord::Sync(_) => ActivityKind::Synchronization,
            ActivityRecord::Generic(r) => r.kind,
        }
    }

    /// Get the start timestamp in nanoseconds.
    pub fn start_ns(&self) -> u64 {
        match self {
            ActivityRecord::Kernel(r) => r.start_ns,
            ActivityRecord::Memcpy(r) => r.start_ns,
            ActivityRecord::Sync(r) => r.start_ns,
            ActivityRecord::Generic(r) => r.start_ns,
        }
    }

    /// Get the end timestamp in nanoseconds.
    pub fn end_ns(&self) -> u64 {
        match self {
            ActivityRecord::Kernel(r) => r.end_ns,
            ActivityRecord::Memcpy(r) => r.end_ns,
            ActivityRecord::Sync(r) => r.end_ns,
            ActivityRecord::Generic(r) => r.end_ns,
        }
    }

    /// Get the duration.
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.end_ns() - self.start_ns())
    }
}

/// Record of a CUDA kernel execution.
#[derive(Debug, Clone)]
pub struct KernelRecord {
    /// Kernel function name
    pub name: String,
    /// Start timestamp (ns since CUDA context creation)
    pub start_ns: u64,
    /// End timestamp (ns since CUDA context creation)
    pub end_ns: u64,
    /// Device ID where kernel executed
    pub device_id: u32,
    /// Context ID
    pub context_id: u32,
    /// Stream ID
    pub stream_id: u32,
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Static shared memory per block (bytes)
    pub static_shared_mem: u32,
    /// Dynamic shared memory per block (bytes)
    pub dynamic_shared_mem: u32,
    /// Registers per thread
    pub registers_per_thread: u32,
    /// Theoretical occupancy (0.0-1.0)
    pub occupancy: f32,
}

impl KernelRecord {
    /// Get the duration in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }

    /// Get the duration.
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.duration_ns())
    }

    /// Get total number of threads launched.
    pub fn total_threads(&self) -> u64 {
        let grid = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block = self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid * block
    }

    /// Get total shared memory per block.
    pub fn total_shared_mem(&self) -> u32 {
        self.static_shared_mem + self.dynamic_shared_mem
    }
}

/// Direction of memory copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    /// Host to Device
    HostToDevice,
    /// Device to Host
    DeviceToHost,
    /// Device to Device
    DeviceToDevice,
    /// Host to Host (via CUDA)
    HostToHost,
    /// Peer to Peer (multi-GPU)
    PeerToPeer,
}

/// Record of a CUDA memory copy operation.
#[derive(Debug, Clone)]
pub struct MemcpyRecord {
    /// Copy direction
    pub kind: MemcpyKind,
    /// Start timestamp (ns)
    pub start_ns: u64,
    /// End timestamp (ns)
    pub end_ns: u64,
    /// Device ID
    pub device_id: u32,
    /// Context ID
    pub context_id: u32,
    /// Stream ID
    pub stream_id: u32,
    /// Bytes copied
    pub bytes: u64,
    /// Source device (for peer-to-peer)
    pub src_device: Option<u32>,
    /// Destination device (for peer-to-peer)
    pub dst_device: Option<u32>,
}

impl MemcpyRecord {
    /// Get the duration in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }

    /// Get the duration.
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.duration_ns())
    }

    /// Get throughput in GB/s.
    pub fn throughput_gbps(&self) -> f64 {
        let duration_s = self.duration_ns() as f64 / 1e9;
        if duration_s > 0.0 {
            (self.bytes as f64 / 1e9) / duration_s
        } else {
            0.0
        }
    }
}

/// Record of a CUDA synchronization event.
#[derive(Debug, Clone)]
pub struct SyncRecord {
    /// Synchronization type
    pub kind: SyncKind,
    /// Start timestamp (ns)
    pub start_ns: u64,
    /// End timestamp (ns)
    pub end_ns: u64,
    /// Device ID
    pub device_id: u32,
    /// Context ID
    pub context_id: u32,
    /// Stream ID (if stream sync)
    pub stream_id: Option<u32>,
}

/// Type of synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncKind {
    /// cudaDeviceSynchronize
    Device,
    /// cudaStreamSynchronize
    Stream,
    /// cudaEventSynchronize
    Event,
    /// cudaStreamWaitEvent
    StreamWait,
}

impl SyncRecord {
    /// Get the duration in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }

    /// Get the duration.
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.duration_ns())
    }
}

/// Generic activity record for less common activities.
#[derive(Debug, Clone)]
pub struct GenericRecord {
    /// Activity kind
    pub kind: ActivityKind,
    /// Start timestamp (ns)
    pub start_ns: u64,
    /// End timestamp (ns)
    pub end_ns: u64,
    /// Device ID
    pub device_id: u32,
    /// Context ID
    pub context_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_record() {
        let record = KernelRecord {
            name: "test_kernel".to_string(),
            start_ns: 1000,
            end_ns: 2000,
            device_id: 0,
            context_id: 1,
            stream_id: 0,
            grid_dim: (128, 1, 1),
            block_dim: (256, 1, 1),
            static_shared_mem: 1024,
            dynamic_shared_mem: 512,
            registers_per_thread: 32,
            occupancy: 0.75,
        };

        assert_eq!(record.duration_ns(), 1000);
        assert_eq!(record.total_threads(), 128 * 256);
        assert_eq!(record.total_shared_mem(), 1536);
    }

    #[test]
    fn test_memcpy_throughput() {
        let record = MemcpyRecord {
            kind: MemcpyKind::HostToDevice,
            start_ns: 0,
            end_ns: 1_000_000, // 1ms
            device_id: 0,
            context_id: 1,
            stream_id: 0,
            bytes: 1_000_000_000, // 1GB
            src_device: None,
            dst_device: None,
        };

        // 1GB in 1ms = 1000 GB/s
        assert!((record.throughput_gbps() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_activity_record_kind() {
        let kernel = ActivityRecord::Kernel(KernelRecord {
            name: "test".to_string(),
            start_ns: 0,
            end_ns: 100,
            device_id: 0,
            context_id: 0,
            stream_id: 0,
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            static_shared_mem: 0,
            dynamic_shared_mem: 0,
            registers_per_thread: 0,
            occupancy: 0.0,
        });

        assert_eq!(kernel.kind(), ActivityKind::Kernel);
        assert_eq!(kernel.duration(), Duration::from_nanos(100));
    }
}
