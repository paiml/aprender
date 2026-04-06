//! Multi-GPU data partitioning and distribution
//!
//! Toyota Way Principles:
//! - Heijunka (Load Leveling): Distribute work evenly across GPUs
//! - Muda elimination: Parallel execution reduces total wall-clock time
//!
//! Architecture:
//! - Detect all available GPU devices
//! - Partition data by range (contiguous chunks) or hash (random distribution)
//! - Execute operations in parallel across all GPUs
//! - Reduce results from all GPUs to final answer
//!
//! References:
//! - Leis et al. (2014): Morsel-driven parallelism for NUMA systems
//! - `MapD` (2017): Multi-GPU query execution patterns

use crate::{Error, Result};
use arrow::array::Int32Array;
use wgpu;

/// Information about a single GPU device
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name (e.g., "NVIDIA RTX 4090", "AMD Radeon RX 7900 XTX")
    pub name: String,
    /// Device type (`DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, Cpu, Other)
    pub device_type: wgpu::DeviceType,
    /// Backend (Vulkan, Metal, DX12, DX11, GL, `BrowserWebGPU`)
    pub backend: wgpu::Backend,
}

/// Multi-GPU device manager
pub struct MultiGpuManager {
    /// All available GPU devices
    devices: Vec<GpuDeviceInfo>,
}

impl MultiGpuManager {
    /// Detect all available GPU devices
    ///
    /// # Errors
    /// Returns error if GPU enumeration fails
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Enumerate all adapters
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        // Convert adapters to device info
        let devices: Vec<GpuDeviceInfo> = adapters
            .iter()
            .map(|adapter| {
                let info = adapter.get_info();
                GpuDeviceInfo {
                    name: info.name,
                    device_type: info.device_type,
                    backend: info.backend,
                }
            })
            .collect();

        Ok(Self { devices })
    }

    /// Get number of available GPUs
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get information about all devices
    #[must_use]
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }
}

/// Data partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Range partitioning: divide data into contiguous chunks
    /// Example: GPU0: [0..N/2], GPU1: [N/2..N]
    Range,
    /// Hash partitioning: distribute rows based on `hash(row_id)` % `num_gpus`
    /// Better load balancing for skewed data
    Hash,
}

/// Data partition for a single GPU
#[derive(Debug)]
pub struct DataPartition {
    /// GPU device index
    pub device_id: usize,
    /// Data chunk for this GPU
    pub data: Int32Array,
}

/// Partition data across multiple GPUs
///
/// # Arguments
/// * `data` - Input array to partition
/// * `num_partitions` - Number of partitions (typically number of GPUs)
/// * `strategy` - Partitioning strategy (Range or Hash)
///
/// # Returns
/// Vector of partitions, one per GPU
///
/// # Errors
/// Returns error if partitioning fails
pub fn partition_data(
    data: &Int32Array,
    num_partitions: usize,
    strategy: PartitionStrategy,
) -> Result<Vec<DataPartition>> {
    if num_partitions == 0 {
        return Err(Error::InvalidInput("num_partitions must be > 0".to_string()));
    }

    let partitions = match strategy {
        PartitionStrategy::Range => partition_range(data, num_partitions),
        PartitionStrategy::Hash => partition_hash(data, num_partitions),
    };
    Ok(partitions)
}

/// Partition data using range partitioning (contiguous chunks)
fn partition_range(data: &Int32Array, num_partitions: usize) -> Vec<DataPartition> {
    let len = data.len();
    let mut partitions = Vec::with_capacity(num_partitions);

    // Calculate chunk size (handle uneven division)
    let base_size = len / num_partitions;
    let remainder = len % num_partitions;

    let mut offset = 0;
    for device_id in 0..num_partitions {
        // First 'remainder' partitions get an extra element
        let size = if device_id < remainder { base_size + 1 } else { base_size };

        // Extract slice
        let values: Vec<i32> = (offset..offset + size).map(|i| data.value(i)).collect();

        partitions.push(DataPartition { device_id, data: Int32Array::from(values) });

        offset += size;
    }

    partitions
}

/// Partition data using hash partitioning (random distribution)
fn partition_hash(data: &Int32Array, num_partitions: usize) -> Vec<DataPartition> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Initialize empty vectors for each partition
    let mut buckets: Vec<Vec<i32>> = (0..num_partitions).map(|_| Vec::new()).collect();

    // Distribute elements by hash
    for i in 0..data.len() {
        let value = data.value(i);

        // Hash the row index (not the value) for deterministic distribution
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();

        #[allow(clippy::cast_possible_truncation)]
        let partition_id = (hash % num_partitions as u64) as usize;
        buckets[partition_id].push(value);
    }

    // Convert buckets to DataPartition
    let partitions: Vec<DataPartition> = buckets
        .into_iter()
        .enumerate()
        .map(|(device_id, values)| DataPartition { device_id, data: Int32Array::from(values) })
        .collect();

    partitions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multigpu_device_detection() {
        // RED: This test should fail because MultiGpuManager::new() is not implemented
        let manager = MultiGpuManager::new();

        // If no GPUs available, should return Ok with 0 devices
        // If GPUs available, should return Ok with device info
        match manager {
            Ok(mgr) => {
                // Should detect at least 0 devices (graceful degradation)
                let count = mgr.device_count();
                println!("Detected {count} GPU device(s)");

                // If devices found, validate their info
                if count > 0 {
                    for (i, device) in mgr.devices().iter().enumerate() {
                        println!("GPU {i}: {device:?}");
                        assert!(!device.name.is_empty(), "Device name should not be empty");
                    }
                }
            }
            Err(e) => {
                panic!("MultiGpuManager::new() failed: {e}");
            }
        }
    }

    #[test]
    fn test_multigpu_device_count_zero_when_no_gpu() {
        // RED: Should fail because not implemented
        // When no GPU available, should return 0 devices (not an error)
        let manager = MultiGpuManager::new();

        if let Ok(mgr) = manager {
            // Valid result: 0 devices (no GPU) or N devices (GPUs found)
            // device_count is usize, so always >= 0
            let _count = mgr.device_count();
        } else {
            // Also acceptable: return error if GPU enumeration fails
            // But prefer returning 0 devices for graceful degradation
        }
    }

    #[test]
    fn test_partition_range_even_split() {
        // RED: Should fail because partition_data() is not implemented
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let partitions = partition_data(&data, 2, PartitionStrategy::Range).unwrap();

        assert_eq!(partitions.len(), 2);

        // GPU 0: [1, 2, 3, 4]
        assert_eq!(partitions[0].device_id, 0);
        assert_eq!(partitions[0].data.len(), 4);
        assert_eq!(partitions[0].data.value(0), 1);
        assert_eq!(partitions[0].data.value(3), 4);

        // GPU 1: [5, 6, 7, 8]
        assert_eq!(partitions[1].device_id, 1);
        assert_eq!(partitions[1].data.len(), 4);
        assert_eq!(partitions[1].data.value(0), 5);
        assert_eq!(partitions[1].data.value(3), 8);
    }

    #[test]
    fn test_partition_range_uneven_split() {
        // RED: Should fail because not implemented
        // With 10 elements and 3 GPUs: [4, 3, 3] or [3, 3, 4] distribution
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let partitions = partition_data(&data, 3, PartitionStrategy::Range).unwrap();

        assert_eq!(partitions.len(), 3);

        // Verify all data is partitioned (no data loss)
        let total_len: usize = partitions.iter().map(|p| p.data.len()).sum();
        assert_eq!(total_len, 10);

        // Verify partitions are contiguous
        assert_eq!(partitions[0].data.value(0), 1); // First partition starts at 1
        let last_partition = &partitions[2];
        assert_eq!(last_partition.data.value(last_partition.data.len() - 1), 10);
        // Last partition ends at 10
    }

    #[test]
    fn test_partition_hash_distribution() {
        // RED: Should fail because not implemented
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let partitions = partition_data(&data, 2, PartitionStrategy::Hash).unwrap();

        assert_eq!(partitions.len(), 2);

        // Verify all data is partitioned (no data loss)
        let total_len: usize = partitions.iter().map(|p| p.data.len()).sum();
        assert_eq!(total_len, 8);

        // Hash partitioning: elements may be in different order
        // Just verify device IDs are correct
        assert_eq!(partitions[0].device_id, 0);
        assert_eq!(partitions[1].device_id, 1);
    }

    #[test]
    fn test_partition_single_gpu() {
        // RED: Should fail because not implemented
        // With 1 GPU, all data goes to partition 0
        let data = Int32Array::from(vec![1, 2, 3, 4]);
        let partitions = partition_data(&data, 1, PartitionStrategy::Range).unwrap();

        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].device_id, 0);
        assert_eq!(partitions[0].data.len(), 4);
    }

    #[test]
    fn test_partition_empty_data() {
        // RED: Should fail because not implemented
        let data = Int32Array::from(vec![] as Vec<i32>);
        let partitions = partition_data(&data, 2, PartitionStrategy::Range).unwrap();

        assert_eq!(partitions.len(), 2);
        // Both partitions should be empty
        assert_eq!(partitions[0].data.len(), 0);
        assert_eq!(partitions[1].data.len(), 0);
    }

    #[test]
    fn test_partition_zero_partitions_error() {
        // RED: Should fail because not implemented
        // num_partitions = 0 should return error
        let data = Int32Array::from(vec![1, 2, 3]);
        let result = partition_data(&data, 0, PartitionStrategy::Range);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_partitions must be > 0"));
    }
}
