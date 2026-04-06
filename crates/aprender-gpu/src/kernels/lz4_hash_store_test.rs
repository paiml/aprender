//! EXTREME TDD: Minimal kernel to test hash table store in shared memory
//!
//! Popperian Falsification Test:
//! - H0 (null): Store to computed smem address works
//! - H1 (alt): Store crashes due to address computation bug
//!
//! This kernel does ONLY:
//! 1. Compute hash_idx from a constant (deterministic)
//! 2. Compute hash_entry_addr = smem_base + PAGE_SIZE + (hash_idx * 4)
//! 3. Store a constant to hash_entry_addr
//! 4. Write success marker to global memory

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

pub const PAGE_SIZE: u32 = 4096;
pub const HASH_TABLE_SIZE: u32 = 8192; // 2048 entries * 4 bytes
pub const SMEM_SIZE: usize = (PAGE_SIZE + HASH_TABLE_SIZE) as usize; // 12288 bytes

/// Minimal test kernel for hash table store
pub struct HashStoreTestKernel;

impl HashStoreTestKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HashStoreTestKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for HashStoreTestKernel {
    fn name(&self) -> &str {
        "hash_store_test"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new(self.name())
            .param(PtxType::U64, "output")
            .shared_memory(SMEM_SIZE)
            .build(|ctx| {
                let output_ptr = ctx.load_param_u64("output");

                // Get lane ID = threadIdx.x % 32
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let mask_31 = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(thread_id, mask_31);

                // Only lane 0 does the test
                let zero = ctx.mov_u32_imm(0);
                let is_leader = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_leader, "L_done");

                // Get shared memory base (generic address)
                let smem_base = ctx.shared_base_addr();

                // === TEST 1: Store to fixed offset (known working baseline) ===
                let fixed_off = ctx.mov_u64_imm(PAGE_SIZE as u64);
                let fixed_addr = ctx.add_u64(smem_base, fixed_off);
                let test_val_1 = ctx.mov_u32_imm(0xDEAD_0001);
                ctx.st_generic_u32(fixed_addr, test_val_1);

                // === TEST 2: Store to computed offset via mul + cvt ===
                // hash_idx = 100 (arbitrary valid index, deterministic)
                let hash_idx = ctx.mov_u32_imm(100);
                // hash_entry_off = hash_idx * 4 = 400
                let hash_entry_off = ctx.mul_u32(hash_idx, 4);
                // Convert to u64 for address arithmetic
                let hash_entry_off_64 = ctx.cvt_u64_u32(hash_entry_off);
                // hash_table_base = smem_base + PAGE_SIZE
                let page_size_64 = ctx.mov_u64_imm(PAGE_SIZE as u64);
                let hash_table_base = ctx.add_u64(smem_base, page_size_64);
                // hash_entry_addr = hash_table_base + hash_entry_off_64
                let hash_entry_addr = ctx.add_u64(hash_table_base, hash_entry_off_64);
                // Store test value - THIS IS THE HYPOTHESIS TEST
                let test_val_2 = ctx.mov_u32_imm(0xDEAD_0002);
                ctx.st_generic_u32(hash_entry_addr, test_val_2);

                // === TEST 3: Verify by loading back ===
                let loaded = ctx.ld_generic_u32(hash_entry_addr);

                // Write loaded value to global memory as proof
                ctx.st_global_u32(output_ptr, loaded);

                ctx.label("L_done");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TDD RED: PTX generation test (runs without GPU)
    #[test]
    fn test_hash_store_kernel_generates_valid_ptx() {
        let kernel = HashStoreTestKernel::new();
        let ptx = kernel.emit_ptx();

        // PTX must contain entry point
        assert!(
            ptx.contains(".entry hash_store_test"),
            "Missing entry point"
        );

        // PTX must contain shared memory declaration
        assert!(ptx.contains(".shared"), "Missing shared memory");

        // PTX must contain generic address conversion
        // cvta.shared converts shared → generic (not cvta.to.shared which is generic → shared)
        assert!(
            ptx.contains("cvta.shared"),
            "Missing cvta.shared for shared→generic conversion"
        );

        // PTX must contain store operations
        assert!(ptx.contains("st.u32"), "Missing st.u32 instructions");

        // PTX must contain mul.lo for offset computation
        assert!(ptx.contains("mul.lo"), "Missing mul.lo for offset");

        // PTX must contain cvt for u32->u64 conversion
        assert!(ptx.contains("cvt.u64.u32"), "Missing cvt.u64.u32");

        println!("=== HashStoreTestKernel PTX (relevant lines) ===");
        for (i, line) in ptx.lines().enumerate() {
            if line.contains("st.u32")
                || line.contains("mul.lo")
                || line.contains("add.u64")
                || line.contains("cvt.u64")
                || line.contains("cvta.to.shared")
            {
                println!("{:4}: {}", i, line);
            }
        }
    }

    /// TDD RED: Shared memory size is sufficient for hash table
    #[test]
    fn test_shared_mem_size_sufficient() {
        // Hash table at offset PAGE_SIZE, max entry at index 2047
        // Max address = PAGE_SIZE + 2047*4 = 4096 + 8188 = 12284
        // Shared mem size = 12288, so max valid byte is 12287
        // Entry at index 2047: bytes 12284-12287 (4 bytes)
        assert!(SMEM_SIZE >= (PAGE_SIZE + HASH_TABLE_SIZE) as usize);

        let max_entry_offset: u32 = 2047 * 4; // 8188
        let max_byte_accessed = PAGE_SIZE + max_entry_offset + 3; // 12287
        assert!(
            (max_byte_accessed as usize) < SMEM_SIZE,
            "Max byte {} >= smem size {}",
            max_byte_accessed,
            SMEM_SIZE
        );
    }

    /// TDD RED: Verify offset computation doesn't overflow
    #[test]
    fn test_offset_computation_no_overflow() {
        // hash_idx is 11-bit (0-2047 from hash >> 21)
        // hash_entry_off = hash_idx * 4, max = 2047 * 4 = 8188
        // This fits in u32 easily (max u32 = 4B)
        let max_hash_idx: u32 = 2047;
        let max_entry_off = max_hash_idx.checked_mul(4).expect("should not overflow");
        assert_eq!(max_entry_off, 8188);

        // Total offset from smem_base: PAGE_SIZE + max_entry_off = 4096 + 8188 = 12284
        let total_offset = PAGE_SIZE
            .checked_add(max_entry_off)
            .expect("should not overflow");
        assert_eq!(total_offset, 12284);
    }
}
