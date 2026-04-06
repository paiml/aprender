//! Sanitizer types: buffer registry, error types, and source locations.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::super::sys::CUdeviceptr;

// ============================================================================
// Address Registry - Tracks all GPU allocations
// ============================================================================

/// Information about a registered GPU buffer
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Human-readable name
    pub name: String,
    /// Device pointer (start address)
    pub ptr: CUdeviceptr,
    /// Size in bytes
    pub size: usize,
    /// Element type name (e.g., "f32", "u8")
    pub type_name: String,
    /// Element size in bytes
    pub element_size: usize,
}

impl BufferInfo {
    /// Check if an address falls within this buffer
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.ptr + self.size as u64
    }

    /// Get the byte offset of an address within this buffer
    pub fn offset_of(&self, addr: u64) -> Option<usize> {
        if self.contains(addr) {
            Some((addr - self.ptr) as usize)
        } else {
            None
        }
    }

    /// Get the element index of an address within this buffer
    pub fn element_index_of(&self, addr: u64) -> Option<usize> {
        self.offset_of(addr).map(|off| off / self.element_size)
    }
}

/// Global registry of GPU buffer allocations
///
/// Thread-safe singleton that tracks all active GPU allocations.
/// Used by the sanitizer to translate raw addresses into meaningful names.
pub struct AddressRegistry {
    pub(super) buffers: HashMap<CUdeviceptr, BufferInfo>,
}

impl AddressRegistry {
    pub(super) fn new() -> Self {
        Self { buffers: HashMap::new() }
    }

    /// Get the global address registry
    pub fn global() -> &'static Mutex<AddressRegistry> {
        static REGISTRY: OnceLock<Mutex<AddressRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(AddressRegistry::new()))
    }

    /// Register a buffer with the registry
    pub fn register(
        &mut self,
        name: impl Into<String>,
        ptr: CUdeviceptr,
        size: usize,
        type_name: impl Into<String>,
        element_size: usize,
    ) {
        let info =
            BufferInfo { name: name.into(), ptr, size, type_name: type_name.into(), element_size };
        self.buffers.insert(ptr, info);
    }

    /// Unregister a buffer when freed
    pub fn unregister(&mut self, ptr: CUdeviceptr) {
        self.buffers.remove(&ptr);
    }

    /// Look up buffer information by an address (may be within buffer)
    pub fn lookup(&self, addr: u64) -> Option<&BufferInfo> {
        // Check if address matches a buffer start
        if let Some(info) = self.buffers.get(&addr) {
            return Some(info);
        }

        // Check if address falls within any buffer
        for info in self.buffers.values() {
            if info.contains(addr) {
                return Some(info);
            }
        }

        None
    }

    /// Format an address with semantic context
    pub fn format_address(&self, addr: u64) -> String {
        if let Some(info) = self.lookup(addr) {
            if let Some(offset) = info.offset_of(addr) {
                let elem_idx = offset / info.element_size;
                let byte_in_elem = offset % info.element_size;
                if byte_in_elem == 0 {
                    format!("{}[{}] (0x{:X} + {} bytes)", info.name, elem_idx, info.ptr, offset)
                } else {
                    format!(
                        "{}[{}]+{} (0x{:X} + {} bytes)",
                        info.name, elem_idx, byte_in_elem, info.ptr, offset
                    )
                }
            } else {
                format!("{} @ 0x{:X}", info.name, addr)
            }
        } else {
            format!("0x{:X} (unknown buffer)", addr)
        }
    }
}

// ============================================================================
// Sanitizer Error Types
// ============================================================================

/// Type of memory violation detected by sanitizer
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryViolationType {
    /// Invalid read from global memory
    InvalidGlobalRead {
        /// Size of the invalid read in bytes
        size: usize,
    },
    /// Invalid write to global memory
    InvalidGlobalWrite {
        /// Size of the invalid write in bytes
        size: usize,
    },
    /// Invalid read from shared memory
    InvalidSharedRead {
        /// Size of the invalid read in bytes
        size: usize,
    },
    /// Invalid write to shared memory
    InvalidSharedWrite {
        /// Size of the invalid write in bytes
        size: usize,
    },
    /// Misaligned access
    MisalignedAccess {
        /// Address that was misaligned
        addr: u64,
    },
    /// Race condition detected
    RaceCondition,
    /// Other/unknown error
    Other(String),
}

/// A memory violation detected by the sanitizer
#[derive(Debug, Clone)]
pub struct MemoryViolation {
    /// Type of violation
    pub violation_type: MemoryViolationType,
    /// Kernel name
    pub kernel_name: String,
    /// SASS offset within kernel
    pub sass_offset: u64,
    /// Thread coordinates (x, y, z)
    pub thread: (u32, u32, u32),
    /// Block coordinates (x, y, z)
    pub block: (u32, u32, u32),
    /// Address that caused the violation
    pub address: u64,
    /// Raw error message from sanitizer
    pub raw_message: String,
}

impl MemoryViolation {
    /// Format with semantic address information
    pub fn format_with_registry(&self, registry: &AddressRegistry) -> String {
        let addr_info = registry.format_address(self.address);
        let violation_desc = match &self.violation_type {
            MemoryViolationType::InvalidGlobalRead { size } => {
                format!("Invalid global read of {} bytes", size)
            }
            MemoryViolationType::InvalidGlobalWrite { size } => {
                format!("Invalid global write of {} bytes", size)
            }
            MemoryViolationType::InvalidSharedRead { size } => {
                format!("Invalid shared read of {} bytes", size)
            }
            MemoryViolationType::InvalidSharedWrite { size } => {
                format!("Invalid shared write of {} bytes", size)
            }
            MemoryViolationType::MisalignedAccess { addr } => {
                format!("Misaligned access at 0x{:X}", addr)
            }
            MemoryViolationType::RaceCondition => "Race condition detected".to_string(),
            MemoryViolationType::Other(msg) => msg.clone(),
        };

        format!(
            "🛑 MEMORY VIOLATION\n\
             ├─ Kernel: {} @ SASS offset 0x{:X}\n\
             ├─ Thread: ({}, {}, {}) in Block ({}, {}, {})\n\
             ├─ Error: {}\n\
             └─ Address: {}",
            self.kernel_name,
            self.sass_offset,
            self.thread.0,
            self.thread.1,
            self.thread.2,
            self.block.0,
            self.block.1,
            self.block.2,
            violation_desc,
            addr_info
        )
    }
}

// ============================================================================
// PTX Source Location
// ============================================================================

/// Maps PTX line numbers to source locations
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Source file path
    pub file: String,
    /// Line number (1-based)
    pub line: u32,
    /// Column (if available)
    pub column: Option<u32>,
    /// Function/label name
    pub function: Option<String>,
}
