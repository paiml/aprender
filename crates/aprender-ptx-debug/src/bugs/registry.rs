//! Bug class registry

use super::Severity;
use crate::parser::SourceLocation;

/// Bug class identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BugClass {
    /// Generic address corruption (cvta.shared creates 64-bit generic addr)
    GenericAddressCorruption,
    /// Shared memory u64 addressing (should use 32-bit offset)
    SharedMemU64Addressing,
    /// Missing direct shared addressing
    MissingDirectShared,
    /// Missing barrier synchronization
    MissingBarrierSync,
    /// Register type invariant violation
    RegisterTypeInvariant,
    /// Unaligned memory access
    UnalignedMemoryAccess,
    /// Data dependent store (ld.shared value used in store)
    DataDependentStore,
    /// Computed address from loaded value
    ComputedAddrFromLoaded,
    /// Sequential code sensitivity (adding one instruction causes crash)
    SequentialCodeSensitivity,
    /// cvta.shared inside loop
    LoopCvtaShared,
    /// Incompatible address space
    IncompatibleAddressSpace,
}

impl std::fmt::Display for BugClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BugClass::GenericAddressCorruption => write!(f, "GenericAddressCorruption"),
            BugClass::SharedMemU64Addressing => write!(f, "SharedMemU64Addressing"),
            BugClass::MissingDirectShared => write!(f, "MissingDirectShared"),
            BugClass::MissingBarrierSync => write!(f, "MissingBarrierSync"),
            BugClass::RegisterTypeInvariant => write!(f, "RegisterTypeInvariant"),
            BugClass::UnalignedMemoryAccess => write!(f, "UnalignedMemoryAccess"),
            BugClass::DataDependentStore => write!(f, "DataDependentStore"),
            BugClass::ComputedAddrFromLoaded => write!(f, "ComputedAddrFromLoaded"),
            BugClass::SequentialCodeSensitivity => write!(f, "SequentialCodeSensitivity"),
            BugClass::LoopCvtaShared => write!(f, "LoopCvtaShared"),
            BugClass::IncompatibleAddressSpace => write!(f, "IncompatibleAddressSpace"),
        }
    }
}

impl BugClass {
    /// Get the severity of this bug class
    pub fn severity(&self) -> Severity {
        match self {
            BugClass::GenericAddressCorruption => Severity::Critical,
            BugClass::SharedMemU64Addressing => Severity::High,
            BugClass::MissingDirectShared => Severity::High,
            BugClass::MissingBarrierSync => Severity::High,
            BugClass::RegisterTypeInvariant => Severity::Medium,
            BugClass::UnalignedMemoryAccess => Severity::High,
            BugClass::DataDependentStore => Severity::Critical,
            BugClass::ComputedAddrFromLoaded => Severity::Critical,
            BugClass::SequentialCodeSensitivity => Severity::High,
            BugClass::LoopCvtaShared => Severity::High,
            BugClass::IncompatibleAddressSpace => Severity::Medium,
        }
    }

    /// Get a description of this bug class
    pub fn description(&self) -> &'static str {
        match self {
            BugClass::GenericAddressCorruption => {
                "cvta.shared creates 64-bit generic address that SASS clobbers"
            }
            BugClass::SharedMemU64Addressing => {
                "Using u64 for shared memory addresses (should use 32-bit offset)"
            }
            BugClass::MissingDirectShared => "Using generic ld/st instead of ld.shared/st.shared",
            BugClass::MissingBarrierSync => {
                "Missing bar.sync between shared memory writes and reads"
            }
            BugClass::RegisterTypeInvariant => {
                "Wrong register type for operation (e.g., f32 vs u32)"
            }
            BugClass::UnalignedMemoryAccess => "Non-aligned global/shared memory access",
            BugClass::DataDependentStore => "Store using value derived from ld.shared crashes",
            BugClass::ComputedAddrFromLoaded => {
                "Address computed from ld.shared value causes store crash"
            }
            BugClass::SequentialCodeSensitivity => {
                "Adding one instruction causes crash (ptxas JIT bug)"
            }
            BugClass::LoopCvtaShared => "cvta.shared inside loop causes register pressure issues",
            BugClass::IncompatibleAddressSpace => "Mismatched address space qualifiers",
        }
    }

    /// Get mitigation advice for this bug class
    pub fn mitigation(&self) -> &'static str {
        match self {
            BugClass::GenericAddressCorruption => {
                "Use direct shared memory addressing with 32-bit offsets"
            }
            BugClass::SharedMemU64Addressing => "Use 32-bit offset for shared memory addressing",
            BugClass::MissingDirectShared => "Use ld.shared/st.shared with 32-bit offset instead",
            BugClass::MissingBarrierSync => "Add bar.sync between shared memory write and read",
            BugClass::RegisterTypeInvariant => {
                "Ensure register type matches instruction type modifier"
            }
            BugClass::UnalignedMemoryAccess => "Ensure memory addresses are aligned to access size",
            BugClass::DataDependentStore => "Use constant value or pre-computed address",
            BugClass::ComputedAddrFromLoaded => {
                "Use constant-only address computation, try membar.cta (partial), or Kernel Fission"
            }
            BugClass::SequentialCodeSensitivity => {
                "Split kernel into simpler kernels (Kernel Fission)"
            }
            BugClass::LoopCvtaShared => "Move cvta.shared outside loop",
            BugClass::IncompatibleAddressSpace => {
                "Use explicit address space qualifiers consistently"
            }
        }
    }
}

/// Pattern for bug detection
#[derive(Debug, Clone)]
pub enum BugPattern {
    /// Generic shared access pattern
    GenericSharedAccess,
    /// Missing barrier pattern
    MissingBarrier {
        /// Write location
        write_loc: SourceLocation,
        /// Read location
        read_loc: SourceLocation,
    },
    /// Loaded value in store pattern
    LoadedValueStore {
        /// Load location
        load_loc: SourceLocation,
        /// Store location
        store_loc: SourceLocation,
    },
    /// Computed address from loaded value pattern
    ComputedAddrFromLoaded {
        /// Load location
        load_loc: SourceLocation,
        /// Computation location
        compute_loc: SourceLocation,
    },
    /// Other pattern
    Other(String),
}

/// Detected bug instance
#[derive(Debug, Clone)]
pub struct Bug {
    /// Bug class
    pub class: BugClass,
    /// Source location
    pub location: SourceLocation,
    /// Bug pattern
    pub pattern: BugPattern,
    /// Additional message
    pub message: String,
}

/// Bug registry - tracks all known bug patterns
#[derive(Debug, Default)]
pub struct BugRegistry {
    /// Detected bugs
    bugs: Vec<Bug>,
}

impl BugRegistry {
    /// Create a new bug registry
    pub fn new() -> Self {
        Self { bugs: Vec::new() }
    }

    /// Add a bug to the registry
    pub fn add(&mut self, bug: Bug) {
        self.bugs.push(bug);
    }

    /// Get all detected bugs
    pub fn bugs(&self) -> &[Bug] {
        &self.bugs
    }

    /// Get critical bugs only
    pub fn critical_bugs(&self) -> Vec<&Bug> {
        self.bugs.iter().filter(|b| b.class.severity() == Severity::Critical).collect()
    }

    /// Check if any critical bugs were detected
    pub fn has_critical_bugs(&self) -> bool {
        self.bugs.iter().any(|b| b.class.severity() == Severity::Critical)
    }

    /// Clear all bugs
    pub fn clear(&mut self) {
        self.bugs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug_class_severity() {
        assert_eq!(BugClass::GenericAddressCorruption.severity(), Severity::Critical);
        assert_eq!(BugClass::DataDependentStore.severity(), Severity::Critical);
        assert_eq!(BugClass::ComputedAddrFromLoaded.severity(), Severity::Critical);
        assert_eq!(BugClass::MissingBarrierSync.severity(), Severity::High);
    }

    #[test]
    fn test_bug_registry() {
        let mut registry = BugRegistry::new();
        assert!(registry.bugs().is_empty());

        registry.add(Bug {
            class: BugClass::DataDependentStore,
            location: SourceLocation::default(),
            pattern: BugPattern::LoadedValueStore {
                load_loc: SourceLocation::default(),
                store_loc: SourceLocation::default(),
            },
            message: "Test bug".into(),
        });

        assert_eq!(registry.bugs().len(), 1);
        assert!(registry.has_critical_bugs());
    }
}
