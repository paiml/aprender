//! Falsification claim checklist.
//!
//! Defines 50 static falsification claims across all edge-case frameworks.
//! Each claim represents a property that tests should attempt to falsify.

use serde::{Deserialize, Serialize};

/// A framework within trueno-cuda-edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    /// Null pointer fuzzing.
    NullFuzzer,
    /// Shared memory boundary probing.
    ShmemProber,
    /// Context lifecycle chaos testing.
    LifecycleChaos,
    /// Quantization parity oracle.
    QuantOracle,
    /// PTX mutation testing.
    PtxPoison,
    /// Worker supervision.
    Supervisor,
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullFuzzer => write!(f, "null_fuzzer"),
            Self::ShmemProber => write!(f, "shmem_prober"),
            Self::LifecycleChaos => write!(f, "lifecycle_chaos"),
            Self::QuantOracle => write!(f, "quant_oracle"),
            Self::PtxPoison => write!(f, "ptx_poison"),
            Self::Supervisor => write!(f, "supervisor"),
        }
    }
}

/// A falsification claim to be tested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationClaim {
    /// Unique identifier.
    pub id: &'static str,
    /// Framework this claim belongs to.
    pub framework: Framework,
    /// Human-readable description of the property.
    pub description: &'static str,
    /// Priority (1 = highest).
    pub priority: u8,
}

/// Status of a falsification claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimStatus {
    /// Not yet tested.
    Pending,
    /// Testing in progress.
    InProgress,
    /// Property held under all tests (not falsified).
    Verified,
    /// Property was falsified (bug found).
    Violated,
    /// Testing was skipped (e.g., no GPU available).
    Skipped,
}

/// All 50 static falsification claims.
#[must_use]
pub fn all_claims() -> Vec<FalsificationClaim> {
    let mut claims = Vec::with_capacity(50);
    claims.extend(null_fuzzer_claims());
    claims.extend(shmem_prober_claims());
    claims.extend(lifecycle_chaos_claims());
    claims.extend(quant_oracle_claims());
    claims.extend(ptx_poison_claims());
    claims.extend(supervisor_claims());
    claims
}

fn null_fuzzer_claims() -> [FalsificationClaim; 10] {
    [
        FalsificationClaim {
            id: "NF-001",
            framework: Framework::NullFuzzer,
            description: "NonNullDevicePtr rejects address 0",
            priority: 1,
        },
        FalsificationClaim {
            id: "NF-002",
            framework: Framework::NullFuzzer,
            description: "Periodic injection fires at exact intervals",
            priority: 1,
        },
        FalsificationClaim {
            id: "NF-003",
            framework: Framework::NullFuzzer,
            description: "Size threshold injection triggers above threshold",
            priority: 2,
        },
        FalsificationClaim {
            id: "NF-004",
            framework: Framework::NullFuzzer,
            description: "Probabilistic injection is deterministic per index",
            priority: 2,
        },
        FalsificationClaim {
            id: "NF-005",
            framework: Framework::NullFuzzer,
            description: "Targeted injection injects at specified indices",
            priority: 2,
        },
        FalsificationClaim {
            id: "NF-006",
            framework: Framework::NullFuzzer,
            description: "Propagation tracker records full call chains",
            priority: 1,
        },
        FalsificationClaim {
            id: "NF-007",
            framework: Framework::NullFuzzer,
            description: "Null injection is caught by error handlers",
            priority: 1,
        },
        FalsificationClaim {
            id: "NF-008",
            framework: Framework::NullFuzzer,
            description: "Fuzzer report catch rate is in [0, 1]",
            priority: 3,
        },
        FalsificationClaim {
            id: "NF-009",
            framework: Framework::NullFuzzer,
            description: "Zero interval never injects",
            priority: 2,
        },
        FalsificationClaim {
            id: "NF-010",
            framework: Framework::NullFuzzer,
            description: "Call index monotonically increases",
            priority: 3,
        },
    ]
}

fn shmem_prober_claims() -> [FalsificationClaim; 10] {
    [
        FalsificationClaim {
            id: "SP-001",
            framework: Framework::ShmemProber,
            description: "Shared memory limit matches compute capability",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-002",
            framework: Framework::ShmemProber,
            description: "Allocation at limit succeeds",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-003",
            framework: Framework::ShmemProber,
            description: "Allocation above limit fails",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-004",
            framework: Framework::ShmemProber,
            description: "Sentinel values detect underflow writes",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-005",
            framework: Framework::ShmemProber,
            description: "Sentinel values detect overflow writes",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-006",
            framework: Framework::ShmemProber,
            description: "Full bank conflict gives 32x serialization",
            priority: 1,
        },
        FalsificationClaim {
            id: "SP-007",
            framework: Framework::ShmemProber,
            description: "Stride-2 access gives 2x serialization",
            priority: 2,
        },
        FalsificationClaim {
            id: "SP-008",
            framework: Framework::ShmemProber,
            description: "Padded access avoids conflicts",
            priority: 2,
        },
        FalsificationClaim {
            id: "SP-009",
            framework: Framework::ShmemProber,
            description: "Bank index cycles every 32 words",
            priority: 2,
        },
        FalsificationClaim {
            id: "SP-010",
            framework: Framework::ShmemProber,
            description: "Compute capability display is sm_XY format",
            priority: 3,
        },
    ]
}

fn lifecycle_chaos_claims() -> [FalsificationClaim; 8] {
    [
        FalsificationClaim {
            id: "LC-001",
            framework: Framework::LifecycleChaos,
            description: "All 8 chaos scenarios are enumerated",
            priority: 1,
        },
        FalsificationClaim {
            id: "LC-002",
            framework: Framework::LifecycleChaos,
            description: "Destruction orderings are valid permutations",
            priority: 1,
        },
        FalsificationClaim {
            id: "LC-003",
            framework: Framework::LifecycleChaos,
            description: "Leak detector respects 1MB tolerance",
            priority: 1,
        },
        FalsificationClaim {
            id: "LC-004",
            framework: Framework::LifecycleChaos,
            description: "Context leaks are detected",
            priority: 1,
        },
        FalsificationClaim {
            id: "LC-005",
            framework: Framework::LifecycleChaos,
            description: "N contexts produce N! orderings",
            priority: 2,
        },
        FalsificationClaim {
            id: "LC-006",
            framework: Framework::LifecycleChaos,
            description: "Reverse ordering is LIFO",
            priority: 2,
        },
        FalsificationClaim {
            id: "LC-007",
            framework: Framework::LifecycleChaos,
            description: "Memory decrease is not a leak",
            priority: 2,
        },
        FalsificationClaim {
            id: "LC-008",
            framework: Framework::LifecycleChaos,
            description: "Default config includes all scenarios",
            priority: 3,
        },
    ]
}

fn quant_oracle_claims() -> [FalsificationClaim; 8] {
    [
        FalsificationClaim {
            id: "QO-001",
            framework: Framework::QuantOracle,
            description: "Q4K tolerance is 0.05",
            priority: 1,
        },
        FalsificationClaim {
            id: "QO-002",
            framework: Framework::QuantOracle,
            description: "Parity check detects differences above tolerance",
            priority: 1,
        },
        FalsificationClaim {
            id: "QO-003",
            framework: Framework::QuantOracle,
            description: "NaN vs NaN is not a violation",
            priority: 1,
        },
        FalsificationClaim {
            id: "QO-004",
            framework: Framework::QuantOracle,
            description: "Boundary generator includes universal values",
            priority: 1,
        },
        FalsificationClaim {
            id: "QO-005",
            framework: Framework::QuantOracle,
            description: "Format boundaries match level count",
            priority: 2,
        },
        FalsificationClaim {
            id: "QO-006",
            framework: Framework::QuantOracle,
            description: "Roundtrip is idempotent for zero",
            priority: 2,
        },
        FalsificationClaim {
            id: "QO-007",
            framework: Framework::QuantOracle,
            description: "Tolerance is positive for all formats",
            priority: 2,
        },
        FalsificationClaim {
            id: "QO-008",
            framework: Framework::QuantOracle,
            description: "Identical values always pass parity",
            priority: 3,
        },
    ]
}

fn ptx_poison_claims() -> [FalsificationClaim; 8] {
    [
        FalsificationClaim {
            id: "PP-001",
            framework: Framework::PtxPoison,
            description: "8 mutation operators are defined",
            priority: 1,
        },
        FalsificationClaim {
            id: "PP-002",
            framework: Framework::PtxPoison,
            description: "FlipAddSub replaces add with sub",
            priority: 1,
        },
        FalsificationClaim {
            id: "PP-003",
            framework: Framework::PtxPoison,
            description: "PTX verifier rejects empty source",
            priority: 1,
        },
        FalsificationClaim {
            id: "PP-004",
            framework: Framework::PtxPoison,
            description: "PTX verifier requires .version directive",
            priority: 1,
        },
        FalsificationClaim {
            id: "PP-005",
            framework: Framework::PtxPoison,
            description: "Mutation score excludes compile errors",
            priority: 1,
        },
        FalsificationClaim {
            id: "PP-006",
            framework: Framework::PtxPoison,
            description: "VerifiedPtx cannot be constructed externally",
            priority: 2,
        },
        FalsificationClaim {
            id: "PP-007",
            framework: Framework::PtxPoison,
            description: "Timeout counts as killed",
            priority: 2,
        },
        FalsificationClaim {
            id: "PP-008",
            framework: Framework::PtxPoison,
            description: "Mutation not found returns None",
            priority: 3,
        },
    ]
}

fn supervisor_claims() -> [FalsificationClaim; 6] {
    [
        FalsificationClaim {
            id: "SV-001",
            framework: Framework::Supervisor,
            description: "OneForOne restarts only crashed worker",
            priority: 1,
        },
        FalsificationClaim {
            id: "SV-002",
            framework: Framework::Supervisor,
            description: "OneForAll restarts all workers",
            priority: 1,
        },
        FalsificationClaim {
            id: "SV-003",
            framework: Framework::Supervisor,
            description: "RestForOne restarts crashed and later workers",
            priority: 1,
        },
        FalsificationClaim {
            id: "SV-004",
            framework: Framework::Supervisor,
            description: "Exhausted budget escalates",
            priority: 1,
        },
        FalsificationClaim {
            id: "SV-005",
            framework: Framework::Supervisor,
            description: "Heartbeat threshold triggers restart",
            priority: 2,
        },
        FalsificationClaim {
            id: "SV-006",
            framework: Framework::Supervisor,
            description: "Thermal shutdown at threshold",
            priority: 2,
        },
    ]
}

/// Returns claims for a specific framework.
#[must_use]
pub fn claims_for_framework(framework: Framework) -> Vec<FalsificationClaim> {
    all_claims()
        .into_iter()
        .filter(|c| c.framework == framework)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_claims_has_50_entries() {
        assert_eq!(all_claims().len(), 50);
    }

    #[test]
    fn claim_ids_are_unique() {
        let claims = all_claims();
        let mut ids: Vec<_> = claims.iter().map(|c| c.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), claims.len());
    }

    #[test]
    fn null_fuzzer_has_10_claims() {
        let claims = claims_for_framework(Framework::NullFuzzer);
        assert_eq!(claims.len(), 10);
    }

    #[test]
    fn shmem_prober_has_10_claims() {
        let claims = claims_for_framework(Framework::ShmemProber);
        assert_eq!(claims.len(), 10);
    }

    #[test]
    fn lifecycle_chaos_has_8_claims() {
        let claims = claims_for_framework(Framework::LifecycleChaos);
        assert_eq!(claims.len(), 8);
    }

    #[test]
    fn quant_oracle_has_8_claims() {
        let claims = claims_for_framework(Framework::QuantOracle);
        assert_eq!(claims.len(), 8);
    }

    #[test]
    fn ptx_poison_has_8_claims() {
        let claims = claims_for_framework(Framework::PtxPoison);
        assert_eq!(claims.len(), 8);
    }

    #[test]
    fn supervisor_has_6_claims() {
        let claims = claims_for_framework(Framework::Supervisor);
        assert_eq!(claims.len(), 6);
    }
}
