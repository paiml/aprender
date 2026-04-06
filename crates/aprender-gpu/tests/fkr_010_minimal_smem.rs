//! FKR-010: Minimal shared memory test to verify cvta.shared works

#[cfg(feature = "cuda")]
mod fkr_010_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

    fn emit_minimal_smem_kernel() -> String {
        let kernel = PtxKernel::new("minimal_smem_test")
            .param(PtxType::U64, "output")
            .shared_memory(4096) // 4KB shared memory
            .build(|ctx| {
                // Just read tid.x, compute smem address, write to global output
                let tid = ctx.special_reg(PtxReg::TidX);
                let mask = ctx.mov_u32_imm(31);
                let lane = ctx.and_u32(tid, mask);

                // Get smem base via cvta.shared
                let smem_base = ctx.shared_base_addr();

                // Compute offset = lane * 4
                let offset = ctx.mul_u32(lane, 4);
                let offset_64 = ctx.cvt_u64_u32(offset);
                let addr = ctx.add_u64(smem_base, offset_64);

                // Write lane to shared memory
                ctx.st_generic_u32(addr, lane);
                ctx.bar_sync(0);

                // Read it back
                let val = ctx.ld_generic_u32(addr);

                // Write to global output
                let out_ptr = ctx.load_param_u64("output");
                let out_addr = ctx.add_u64(out_ptr, offset_64);
                ctx.st_global_u32(out_addr, val);

                ctx.ret();
            });

        // Use PtxModule to add proper headers
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit()
    }

    #[test]
    fn fkr_010a_minimal_smem_test() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        let output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 32).unwrap();

        let ptx = emit_minimal_smem_kernel();
        println!("=== Minimal smem PTX ===\n{}", ptx);

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                panic!("PTX compilation failed: {:?}", e);
            }
        };

        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        println!("Launching minimal smem kernel...");
        unsafe {
            stream
                .launch_kernel(&mut module, "minimal_smem_test", &config, &mut args)
                .expect("Kernel launch");
        }

        match stream.synchronize() {
            Ok(()) => {
                let mut output = vec![0u32; 32];
                output_buf.copy_to_host(&mut output).unwrap();
                println!("Output: {:?}", output);

                // Verify each lane read back its lane ID
                for (i, &val) in output.iter().enumerate() {
                    assert_eq!(val, i as u32, "Lane {} should have value {}", i, i);
                }
                println!("✅ Minimal smem test PASSED!");
            }
            Err(e) => {
                panic!("🛑 Minimal smem kernel crashed: {:?}", e);
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod fkr_010_tests {
    #[test]
    fn fkr_010_skip_no_cuda() {
        println!("FKR-010: Skipped - CUDA feature not enabled");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn fkr_010b_debug_smem_base() {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

    // Check CUDA availability first
    if CudaContext::new(0).is_err() {
        eprintln!("FKR-010b SKIPPED: No CUDA device available");
        return;
    }

    // This test computes smem_base like LZ4 does and writes intermediate values to global memory
    let kernel = PtxKernel::new("debug_smem_base")
        .param(PtxType::U64, "output")
        .shared_memory(37632) // Same as LZ4: 3 warps * 12544
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);

            // Compute warp_id = tid >> 5
            let warp_id = ctx.shr_u32_imm(tid, 5);

            // Compute warp_offset = warp_id * 12544
            let warp_offset = ctx.mul_u32(warp_id, 12544);
            let warp_offset_64 = ctx.cvt_u64_u32(warp_offset);

            // Get smem base via cvta.shared
            let raw_smem = ctx.shared_base_addr();

            // smem_base = raw_smem + warp_offset
            let smem_base = ctx.add_u64(raw_smem, warp_offset_64);

            // Compute state_base = smem_base + 12420
            let state_offset = ctx.mov_u32_imm(12420);
            let state_offset_64 = ctx.cvt_u64_u32(state_offset);
            let state_base = ctx.add_u64(smem_base, state_offset_64);

            // Write debug values to global output
            // output[0] = raw_smem (lower 32 bits)
            // output[1] = warp_offset
            // output[2] = smem_base (lower 32 bits)
            // output[3] = state_base (lower 32 bits)
            let out_ptr = ctx.load_param_u64("output");

            let raw_smem_lo = ctx.cvt_u32_u64(raw_smem);
            ctx.st_global_u32(out_ptr, raw_smem_lo);

            let off4 = ctx.mov_u64_imm(4);
            let out1 = ctx.add_u64(out_ptr, off4);
            ctx.st_global_u32(out1, warp_offset);

            let off8 = ctx.mov_u64_imm(8);
            let out2 = ctx.add_u64(out_ptr, off8);
            let smem_base_lo = ctx.cvt_u32_u64(smem_base);
            ctx.st_global_u32(out2, smem_base_lo);

            let off12 = ctx.mov_u64_imm(12);
            let out3 = ctx.add_u64(out_ptr, off12);
            let state_base_lo = ctx.cvt_u32_u64(state_base);
            ctx.st_global_u32(out3, state_base_lo);

            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    let cuda_ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&cuda_ctx).expect("CUDA stream");
    let output_buf: GpuBuffer<u32> = GpuBuffer::new(&cuda_ctx, 4).unwrap();

    let mut module = CudaModule::from_ptx(&cuda_ctx, &ptx).expect("PTX compilation");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (1, 1, 1), // Single thread to avoid race condition on output
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    unsafe {
        stream
            .launch_kernel(&mut module, "debug_smem_base", &config, &mut args)
            .expect("Kernel launch");
    }

    stream.synchronize().expect("Sync");

    let mut output = vec![0u32; 4];
    output_buf.copy_to_host(&mut output).unwrap();

    println!("=== smem_base Debug ===");
    println!("raw_smem (lo32):   0x{:08X}", output[0]);
    println!("warp_offset:       {}", output[1]);
    println!("smem_base (lo32):  0x{:08X}", output[2]);
    println!("state_base (lo32): 0x{:08X}", output[3]);

    // state_base should be smem_base + 12420
    assert!(output[0] > 0, "raw_smem should not be 0");
    assert_eq!(output[1], 0, "warp 0 should have offset 0");
    assert_eq!(output[2], output[0], "smem_base should equal raw_smem for warp 0");
    let expected_state = output[0].wrapping_add(12420);
    assert_eq!(output[3], expected_state, "state_base should be smem_base + 12420");

    println!("✅ smem_base debug test PASSED!");
}
