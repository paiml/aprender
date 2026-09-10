//! Launch budget validation — the enforcement `ptx-codegen-safety-v1` has always declared.
//!
//! `contracts/trueno/ptx-codegen-safety-v1.yaml` equation `register_budget` states
//!
//! ```text
//! forall kernel K:
//!   reg_count(K)  <= max_regs_per_thread(sm)
//!   shared_mem(K) <= max_shared_per_block(sm)
//! postcondition: cuOccupancyMaxActiveBlocksPerMultiprocessor > 0
//! ```
//!
//! and nothing enforced it: the generated `contract_register_budget!` macro is invoked
//! nowhere in the tree, and its postcondition names an unbound identifier that would not
//! compile if it ever were. This module is that missing enforcement.
//!
//! **Arch-agnostic by construction.** The contract's own `domain` stops at sm_90, and so
//! does the hand-written arch table in `driver/sys/mod.rs` (`CU_TARGET_COMPUTE_90` is the
//! last constant). A per-SM limit table is a thing that goes stale every GPU generation —
//! this one has, twice. So the limits here are **queried from the device**
//! (`cuDeviceGetAttribute`) rather than looked up, and adding a new architecture requires
//! no change to this file.
//!
//! The policy — [`validate_launch`] — is pure and needs no GPU, so its case table runs in
//! the required check. Only [`DeviceLimits::query`] and [`KernelAttributes::query`] need
//! CUDA.

/// Per-kernel resource usage, as reported by the JIT for a *compiled* kernel.
///
/// These are properties of the cubin the driver actually produced — not of the PTX we
/// emitted — which is why they can only be read back after a module load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelAttributes {
    /// Largest block size this kernel can be launched with (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`).
    pub max_threads_per_block: u32,
    /// Registers used per thread (`CU_FUNC_ATTRIBUTE_NUM_REGS`).
    pub num_regs: u32,
    /// Statically declared shared memory, bytes (`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`).
    pub static_shared_bytes: u32,
    /// Per-thread local memory, bytes (`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`). Non-zero means
    /// the kernel spilled registers to local memory — legal, but a performance cliff.
    pub local_bytes: u32,
}

/// Device-side limits, queried rather than tabulated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    /// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`.
    pub max_threads_per_block: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`.
    pub max_shared_per_block: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`.
    pub max_regs_per_block: u32,
    /// `CU_DEVICE_ATTRIBUTE_WARP_SIZE`.
    pub warp_size: u32,
}

/// A way a launch violates the kernel's or the device's budget.
///
/// Every variant names both sides of the comparison: a violation you cannot act on is a
/// log line, not a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LaunchBudgetViolation {
    /// Block size exceeds what the compiled kernel supports (usually register pressure).
    #[error("block size {requested} exceeds the kernel's max_threads_per_block {kernel_max} (register pressure); launch would fail with CUDA_ERROR_INVALID_VALUE")]
    BlockExceedsKernelMax {
        /// Threads per block the launch site asked for.
        requested: u32,
        /// What the compiled kernel supports.
        kernel_max: u32,
    },
    /// Block size exceeds the device maximum.
    #[error("block size {requested} exceeds the device's max_threads_per_block {device_max}")]
    BlockExceedsDeviceMax {
        /// Threads per block the launch site asked for.
        requested: u32,
        /// What the device supports.
        device_max: u32,
    },
    /// Static + dynamic shared memory exceeds the per-block limit.
    #[error("shared memory {static_bytes}+{dynamic_bytes}={total} bytes exceeds the device's max_shared_per_block {device_max}")]
    SharedMemoryExceeded {
        /// Statically declared bytes.
        static_bytes: u32,
        /// Dynamically requested bytes.
        dynamic_bytes: u32,
        /// Their sum.
        total: u32,
        /// Device per-block limit.
        device_max: u32,
    },
    /// Registers for the whole block exceed the per-block register file.
    #[error("register budget {num_regs}regs x {block_size}threads = {total} exceeds the device's max_regs_per_block {device_max}")]
    RegisterBudgetExceeded {
        /// Registers per thread.
        num_regs: u32,
        /// Threads per block.
        block_size: u32,
        /// Their product.
        total: u32,
        /// Device per-block limit.
        device_max: u32,
    },
    /// A zero-sized launch. Always a bug at the call site, never a legal request.
    #[error("block size is zero")]
    ZeroBlockSize,
}

/// Validate one launch configuration against a compiled kernel and its device.
///
/// This is the `register_budget` equation, executable. It is **pure** — no CUDA, no global
/// state — so the case table below runs anywhere, including the CPU-only required check.
///
/// # Errors
///
/// Returns the first [`LaunchBudgetViolation`] found. Checks are ordered cheapest-first and
/// most-specific-first, so `BlockExceedsKernelMax` (the actionable one, naming the kernel's
/// own ceiling) is reported ahead of the device-wide limit.
pub fn validate_launch(
    attrs: &KernelAttributes,
    limits: &DeviceLimits,
    block_size: u32,
    dynamic_shared_bytes: u32,
) -> Result<(), LaunchBudgetViolation> {
    if block_size == 0 {
        return Err(LaunchBudgetViolation::ZeroBlockSize);
    }
    if block_size > attrs.max_threads_per_block {
        return Err(LaunchBudgetViolation::BlockExceedsKernelMax {
            requested: block_size,
            kernel_max: attrs.max_threads_per_block,
        });
    }
    if block_size > limits.max_threads_per_block {
        return Err(LaunchBudgetViolation::BlockExceedsDeviceMax {
            requested: block_size,
            device_max: limits.max_threads_per_block,
        });
    }
    let total_shared = attrs
        .static_shared_bytes
        .saturating_add(dynamic_shared_bytes);
    if total_shared > limits.max_shared_per_block {
        return Err(LaunchBudgetViolation::SharedMemoryExceeded {
            static_bytes: attrs.static_shared_bytes,
            dynamic_bytes: dynamic_shared_bytes,
            total: total_shared,
            device_max: limits.max_shared_per_block,
        });
    }
    let total_regs = attrs.num_regs.saturating_mul(block_size);
    if total_regs > limits.max_regs_per_block {
        return Err(LaunchBudgetViolation::RegisterBudgetExceeded {
            num_regs: attrs.num_regs,
            block_size,
            total: total_regs,
            device_max: limits.max_regs_per_block,
        });
    }
    Ok(())
}

/// Whether a kernel spilled registers to local memory.
///
/// Not a violation — a spilled kernel is correct, just slow — so this is reported
/// separately from [`validate_launch`] rather than failing it. FALSIFY-PTX-003's
/// "register spilling causes performance cliff" half.
#[must_use]
pub fn spills_to_local(attrs: &KernelAttributes) -> bool {
    attrs.local_bytes > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel that comfortably fits: the shape `probe.ptx` reported on GB10.
    fn ok_kernel() -> KernelAttributes {
        KernelAttributes {
            max_threads_per_block: 1024,
            num_regs: 10,
            static_shared_bytes: 0,
            local_bytes: 0,
        }
    }

    /// Limits as an sm_89/sm_121-class device reports them.
    fn limits() -> DeviceLimits {
        DeviceLimits {
            max_threads_per_block: 1024,
            max_shared_per_block: 49152,
            max_regs_per_block: 65536,
            warp_size: 32,
        }
    }

    #[test]
    fn accepts_a_legal_launch() {
        assert!(validate_launch(&ok_kernel(), &limits(), 256, 0).is_ok());
    }

    #[test]
    fn accepts_the_kernel_max_exactly() {
        // Boundary: block_size == max_threads_per_block is legal, not off-by-one.
        assert!(validate_launch(&ok_kernel(), &limits(), 1024, 0).is_ok());
    }

    #[test]
    fn rejects_zero_block_size() {
        assert_eq!(
            validate_launch(&ok_kernel(), &limits(), 0, 0),
            Err(LaunchBudgetViolation::ZeroBlockSize)
        );
    }

    #[test]
    fn rejects_block_above_kernel_max() {
        // The real GH-613 shape: a register-heavy kernel caps below the device max, and a
        // hardcoded block=256 launch site walks straight into CUDA_ERROR_INVALID_VALUE.
        let heavy = KernelAttributes {
            max_threads_per_block: 128,
            num_regs: 200,
            static_shared_bytes: 0,
            local_bytes: 0,
        };
        assert_eq!(
            validate_launch(&heavy, &limits(), 256, 0),
            Err(LaunchBudgetViolation::BlockExceedsKernelMax {
                requested: 256,
                kernel_max: 128,
            })
        );
    }

    #[test]
    fn kernel_max_is_reported_before_device_max() {
        // Both are violated; the kernel's own ceiling is the actionable one.
        let heavy = KernelAttributes {
            max_threads_per_block: 128,
            ..ok_kernel()
        };
        let tight = DeviceLimits {
            max_threads_per_block: 512,
            ..limits()
        };
        assert!(matches!(
            validate_launch(&heavy, &tight, 2048, 0),
            Err(LaunchBudgetViolation::BlockExceedsKernelMax { .. })
        ));
    }

    #[test]
    fn rejects_block_above_device_max() {
        let permissive = KernelAttributes {
            max_threads_per_block: 4096,
            ..ok_kernel()
        };
        assert_eq!(
            validate_launch(&permissive, &limits(), 2048, 0),
            Err(LaunchBudgetViolation::BlockExceedsDeviceMax {
                requested: 2048,
                device_max: 1024,
            })
        );
    }

    #[test]
    fn rejects_shared_memory_overflow_counting_both_halves() {
        // static alone fits and dynamic alone fits; only the SUM violates.
        let k = KernelAttributes {
            static_shared_bytes: 32768,
            ..ok_kernel()
        };
        assert_eq!(
            validate_launch(&k, &limits(), 256, 32768),
            Err(LaunchBudgetViolation::SharedMemoryExceeded {
                static_bytes: 32768,
                dynamic_bytes: 32768,
                total: 65536,
                device_max: 49152,
            })
        );
    }

    #[test]
    fn accepts_shared_memory_exactly_at_the_limit() {
        let k = KernelAttributes {
            static_shared_bytes: 49152,
            ..ok_kernel()
        };
        assert!(validate_launch(&k, &limits(), 256, 0).is_ok());
    }

    #[test]
    fn rejects_register_budget_overflow() {
        // 255 regs x 1024 threads = 261_120 > 65_536.
        let k = KernelAttributes {
            max_threads_per_block: 1024,
            num_regs: 255,
            ..ok_kernel()
        };
        assert_eq!(
            validate_launch(&k, &limits(), 1024, 0),
            Err(LaunchBudgetViolation::RegisterBudgetExceeded {
                num_regs: 255,
                block_size: 1024,
                total: 261_120,
                device_max: 65536,
            })
        );
    }

    #[test]
    fn register_product_saturates_instead_of_overflowing() {
        // u32 multiply of two large values must not wrap into a FALSE PASS.
        let k = KernelAttributes {
            max_threads_per_block: u32::MAX,
            num_regs: u32::MAX,
            ..ok_kernel()
        };
        let wide = DeviceLimits {
            max_threads_per_block: u32::MAX,
            ..limits()
        };
        assert!(matches!(
            validate_launch(&k, &wide, u32::MAX, 0),
            Err(LaunchBudgetViolation::RegisterBudgetExceeded { .. })
        ));
    }

    #[test]
    fn shared_memory_sum_saturates_instead_of_overflowing() {
        let k = KernelAttributes {
            static_shared_bytes: u32::MAX,
            ..ok_kernel()
        };
        assert!(matches!(
            validate_launch(&k, &limits(), 256, u32::MAX),
            Err(LaunchBudgetViolation::SharedMemoryExceeded { .. })
        ));
    }

    #[test]
    fn spill_detection_is_separate_from_validity() {
        let spilled = KernelAttributes {
            local_bytes: 128,
            ..ok_kernel()
        };
        // Spilling is a performance cliff, not an illegal launch.
        assert!(validate_launch(&spilled, &limits(), 256, 0).is_ok());
        assert!(spills_to_local(&spilled));
        assert!(!spills_to_local(&ok_kernel()));
    }

    #[test]
    fn violations_name_both_sides_of_the_comparison() {
        // A violation you cannot act on is a log line, not a gate.
        let heavy = KernelAttributes {
            max_threads_per_block: 128,
            ..ok_kernel()
        };
        let Err(v) = validate_launch(&heavy, &limits(), 256, 0) else {
            panic!("expected a violation");
        };
        let msg = v.to_string();
        assert!(msg.contains("256"), "message must name the request: {msg}");
        assert!(msg.contains("128"), "message must name the limit: {msg}");
    }
}
