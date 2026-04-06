//! FKR-011: Direct .shared addressing test
//!
//! Tests using ld.shared.u32/st.shared.u32 with 32-bit offsets
//! instead of generic addressing with cvta.shared-derived addresses.
//! This approach bypasses the generic address space mechanism that
//! may be causing SASS register clobbering issues.

#[cfg(feature = "cuda")]
mod fkr_011_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{
        PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxMemory, PtxModule, PtxReg, PtxType,
    };

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Test direct .shared addressing with 32-bit offsets
    #[test]
    fn fkr_011a_direct_shared_addressing() {
        if !cuda_available() {
            eprintln!("FKR-011a SKIPPED: No CUDA device available");
            return;
        }
        let kernel = PtxKernel::new("direct_shared_test")
            .param(PtxType::U64, "output")
            .shared_memory(4096) // 4KB shared memory
            .build(|ctx| {
                // Get thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let mask = ctx.mov_u32_imm(31);
                let lane = ctx.and_u32(tid, mask);

                // Compute 32-bit offset directly: lane * 4
                // No cvta.shared needed!
                let offset = ctx.mul_u32(lane, 4);

                // Write lane to shared memory using direct .shared addressing
                ctx.st_shared_u32(offset, lane);
                ctx.bar_sync(0);

                // Read it back using direct .shared addressing
                let val = ctx.ld_shared_u32(offset);

                // Write to global output
                let out_ptr = ctx.load_param_u64("output");
                let out_off = ctx.mul_u32(lane, 4);
                let out_off_64 = ctx.cvt_u64_u32(out_off);
                let out_addr = ctx.add_u64(out_ptr, out_off_64);
                ctx.st_global_u32(out_addr, val);

                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== Direct .shared PTX ===\n{}", ptx);

        let cuda_ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&cuda_ctx).expect("CUDA stream");
        let output_buf: GpuBuffer<u32> = GpuBuffer::new(&cuda_ctx, 32).unwrap();

        let mut module = CudaModule::from_ptx(&cuda_ctx, &ptx).expect("PTX compilation");

        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "direct_shared_test", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut output = vec![0u32; 32];
        output_buf.copy_to_host(&mut output).unwrap();
        println!("Output: {:?}", output);

        // Verify each lane read back its lane ID
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, i as u32, "Lane {} should have value {}", i, i);
        }
        println!("FKR-011a: Direct .shared addressing PASSED!");
    }

    /// Test multi-warp direct .shared addressing (like LZ4 uses)
    #[test]
    fn fkr_011b_multi_warp_direct_shared() {
        if !cuda_available() {
            eprintln!("FKR-011b SKIPPED: No CUDA device available");
            return;
        }
        const WARP_SMEM_SIZE: u32 = 12544; // Same as LZ4

        let kernel = PtxKernel::new("multi_warp_shared_test")
            .param(PtxType::U64, "output")
            .shared_memory((WARP_SMEM_SIZE * 3) as usize) // 3 warps
            .build(|ctx| {
                // Get thread ID and compute warp ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let warp_id = ctx.shr_u32_imm(tid, 5);
                let mask31 = ctx.mov_u32_imm(31);
                let lane = ctx.and_u32(tid, mask31);

                // Compute 32-bit offset: warp_id * WARP_SMEM_SIZE + lane * 4
                let warp_offset = ctx.mul_u32(warp_id, WARP_SMEM_SIZE);
                let lane_offset = ctx.mul_u32(lane, 4);
                let offset = ctx.add_u32_reg(warp_offset, lane_offset);

                // Write tid to shared memory using direct .shared addressing
                ctx.st_shared_u32(offset, tid);
                ctx.bar_sync(0);

                // Read it back using direct .shared addressing
                let val = ctx.ld_shared_u32(offset);

                // Write to global output[tid]
                let out_ptr = ctx.load_param_u64("output");
                let out_off = ctx.mul_u32(tid, 4);
                let out_off_64 = ctx.cvt_u64_u32(out_off);
                let out_addr = ctx.add_u64(out_ptr, out_off_64);
                ctx.st_global_u32(out_addr, val);

                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== Multi-warp .shared PTX ===\n{}", ptx);

        let cuda_ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&cuda_ctx).expect("CUDA stream");
        let output_buf: GpuBuffer<u32> = GpuBuffer::new(&cuda_ctx, 96).unwrap();

        let mut module = CudaModule::from_ptx(&cuda_ctx, &ptx).expect("PTX compilation");

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (96, 1, 1), // 3 warps
            shared_mem: 0,
        };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "multi_warp_shared_test", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut output = vec![0u32; 96];
        output_buf.copy_to_host(&mut output).unwrap();
        println!("Output (first 32): {:?}", &output[0..32]);
        println!("Output (32-64): {:?}", &output[32..64]);
        println!("Output (64-96): {:?}", &output[64..96]);

        // Verify each thread read back its tid
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, i as u32, "Thread {} should have value {}", i, i);
        }
        println!("FKR-011b: Multi-warp direct .shared addressing PASSED!");
    }

    /// Test the specific state offset pattern used in LZ4
    #[test]
    fn fkr_011c_lz4_state_pattern() {
        if !cuda_available() {
            eprintln!("FKR-011c SKIPPED: No CUDA device available");
            return;
        }
        const WARP_SMEM_SIZE: u32 = 12544;
        const STATE_OFFSET: u32 = 4096 + 8192 + 128 + 4; // 12420

        let kernel = PtxKernel::new("lz4_state_pattern_test")
            .param(PtxType::U64, "output")
            .shared_memory((WARP_SMEM_SIZE * 3) as usize)
            .build(|ctx| {
                // Only thread 0 runs this test
                let tid = ctx.special_reg(PtxReg::TidX);
                let zero = ctx.mov_u32_imm(0);
                let is_leader = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_leader, "L_skip");

                // Compute state_base offset: warp_id * WARP_SMEM_SIZE + STATE_OFFSET
                let warp_id = ctx.shr_u32_imm(tid, 5); // 0 for thread 0
                let warp_offset = ctx.mul_u32(warp_id, WARP_SMEM_SIZE);
                let state_offset = ctx.add_u32(warp_offset, STATE_OFFSET);

                // Write test values to state using direct .shared
                let test_val1 = ctx.mov_u32_imm(0xDEADBEEF);
                ctx.st_shared_u32(state_offset, test_val1); // state[0] = in_pos

                let off4 = ctx.add_u32(state_offset, 4);
                let test_val2 = ctx.mov_u32_imm(0xCAFEBABE);
                ctx.st_shared_u32(off4, test_val2); // state[1] = out_pos

                let off8 = ctx.add_u32(state_offset, 8);
                let test_val3 = ctx.mov_u32_imm(0x12345678);
                ctx.st_shared_u32(off8, test_val3); // state[2] = anchor

                ctx.bar_sync(0);

                // Read back and verify
                let read1 = ctx.ld_shared_u32(state_offset);
                let read2 = ctx.ld_shared_u32(off4);
                let read3 = ctx.ld_shared_u32(off8);

                // Write to global output
                let out_ptr = ctx.load_param_u64("output");
                ctx.st_global_u32(out_ptr, read1);

                let off4 = ctx.mov_u64_imm(4);
                let out1 = ctx.add_u64(out_ptr, off4);
                ctx.st_global_u32(out1, read2);

                let off8 = ctx.mov_u64_imm(8);
                let out2 = ctx.add_u64(out_ptr, off8);
                ctx.st_global_u32(out2, read3);

                // Also write the computed offsets for debugging
                let off12 = ctx.mov_u64_imm(12);
                let out3 = ctx.add_u64(out_ptr, off12);
                ctx.st_global_u32(out3, state_offset);

                ctx.label("L_skip");
                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== LZ4 State Pattern PTX ===\n{}", ptx);

        let cuda_ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&cuda_ctx).expect("CUDA stream");
        let output_buf: GpuBuffer<u32> = GpuBuffer::new(&cuda_ctx, 4).unwrap();

        let mut module = CudaModule::from_ptx(&cuda_ctx, &ptx).expect("PTX compilation");

        let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "lz4_state_pattern_test", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut output = vec![0u32; 4];
        output_buf.copy_to_host(&mut output).unwrap();

        println!("=== LZ4 State Pattern Results ===");
        println!("state[0] (in_pos):   0x{:08X} (expected 0xDEADBEEF)", output[0]);
        println!("state[1] (out_pos):  0x{:08X} (expected 0xCAFEBABE)", output[1]);
        println!("state[2] (anchor):   0x{:08X} (expected 0x12345678)", output[2]);
        println!("state_offset:        {} (expected {})", output[3], STATE_OFFSET);

        assert_eq!(output[0], 0xDEADBEEF, "in_pos should be 0xDEADBEEF");
        assert_eq!(output[1], 0xCAFEBABE, "out_pos should be 0xCAFEBABE");
        assert_eq!(output[2], 0x12345678, "anchor should be 0x12345678");
        assert_eq!(output[3], STATE_OFFSET, "state_offset should be {}", STATE_OFFSET);

        println!("FKR-011c: LZ4 state pattern PASSED!");
    }
}

#[cfg(not(feature = "cuda"))]
mod fkr_011_tests {
    #[test]
    fn fkr_011_skip_no_cuda() {
        println!("FKR-011: Skipped - CUDA feature not enabled");
    }
}
