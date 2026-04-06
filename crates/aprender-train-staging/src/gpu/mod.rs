//! GPU resource management and multi-node training infrastructure.
//!
//! Implements the GPU sharing spec across three phases:
//!
//! ## Phase 1: VRAM Guard + Sequential Queue
//! - **Ledger**: flock-based VRAM reservation with lease expiry (GPU-SHARE-001)
//! - **Guard**: Pre-allocation VRAM check + post-init tracking (GPU-SHARE-002)
//! - **Wait**: Polling wait queue with timeout (GPU-SHARE-003)
//! - **Profiler**: Brick-phase profiler for GPU sharing ops (GPU-SHARE-005)
//! - **MPS**: Experimental CUDA MPS support (opt-in, §1.5)
//!
//! ## Phase 2: Multi-Adapter Single-Process
//! See [`crate::finetune::multi_adapter_pipeline`].
//!
//! ## Phase 3: Multi-Node via Forjar
//! - **Cluster**: Cluster YAML config schema + validation (§3.2)
//! - **Placement**: Greedy job placement with FLOPS scoring (§3.3)
//! - **Coordinator**: Checkpoint polling + leaderboard (§3.4)

pub mod cluster;
pub mod coordinator;
pub mod error;
pub mod guard;
pub mod ledger;
pub mod mps;
pub mod placement;
pub mod profiler;
pub mod wait;
