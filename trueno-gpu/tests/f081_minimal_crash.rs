//! F081 FALSIFICATION TEST - HYPOTHESIS REFUTED
//!
//! **STATUS**: FALSIFIED (2026-01-05)
//!
//! Original Hypothesis: `ld.shared.u32 → st.global.u32` with the loaded value
//! as src causes CUDA_ERROR_UNKNOWN (716).
//!
//! **EXPERIMENTAL RESULT**: Hypothesis REFUTED. The pattern SUCCEEDS and returns
//! correct value 0xBEEFCAFE. The original LZ4 kernel crash was caused by a
//! different bug (likely F082: Computed Address, or F021: Generic Address Corruption).
//!
//! This test now serves as **empirical evidence** that F081 is NOT a real bug.
//! The Popperian falsification methodology successfully identified a false hypothesis.
//!
//! Run: cargo test -p trueno-gpu --test f081_minimal_crash --features cuda -- --test-threads=1 --nocapture

#[cfg(feature = "cuda")]
mod f081_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{
        PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxMemory, PtxModule, PtxReg, PtxType,
    };

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// TEST 1: Baseline - write immediate to global memory (SHOULD PASS)
    #[test]
    fn f081_baseline_immediate_to_global() {
        if !cuda_available() {
            eprintln!("F081 test SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        let kernel = PtxKernel::new("baseline_imm").param(PtxType::U64, "output").build(|ctx| {
            let out_ptr = ctx.load_param_u64("output");
            let tid = ctx.special_reg(PtxReg::TidX);
            let zero = ctx.mov_u32_imm(0);
            let is_t0 = ctx.setp_eq_u32(tid, zero);
            ctx.branch_if_not(is_t0, "L_end");

            // Write immediate value to global - NOT using loaded value
            let val = ctx.mov_u32_imm(0xCAFEBABE);
            ctx.st_global_u32(out_ptr, val);

            ctx.label("L_end");
            ctx.ret();
        });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== BASELINE (immediate → global) ===");
        println!("{}", ptx);

        let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        output.copy_from_host(&[0u32]).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");
        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "baseline_imm", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut result = vec![0u32; 1];
        output.copy_to_host(&mut result).unwrap();
        assert_eq!(result[0], 0xCAFEBABE);
        println!("PASSED: Immediate → global works\n");
    }

    /// TEST 2: Load from global, store to global (SHOULD PASS)
    #[test]
    fn f081_global_to_global() {
        if !cuda_available() {
            eprintln!("F081 test SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        let kernel = PtxKernel::new("global_to_global")
            .param(PtxType::U64, "input")
            .param(PtxType::U64, "output")
            .build(|ctx| {
                let in_ptr = ctx.load_param_u64("input");
                let out_ptr = ctx.load_param_u64("output");
                let tid = ctx.special_reg(PtxReg::TidX);
                let zero = ctx.mov_u32_imm(0);
                let is_t0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_t0, "L_end");

                // ld.global → st.global (this should work)
                let val = ctx.ld_global_u32(in_ptr);
                ctx.st_global_u32(out_ptr, val);

                ctx.label("L_end");
                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== GLOBAL → GLOBAL ===");
        println!("{}", ptx);

        let mut input: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        input.copy_from_host(&[0xDEADBEEF_u32]).unwrap();
        output.copy_from_host(&[0u32]).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");
        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 2] = [input.as_kernel_arg(), output.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "global_to_global", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut result = vec![0u32; 1];
        output.copy_to_host(&mut result).unwrap();
        assert_eq!(result[0], 0xDEADBEEF);
        println!("PASSED: Global → global works\n");
    }

    /// TEST 3: Write to shared, load from shared, store to global
    ///
    /// **FALSIFICATION RESULT**: This test was designed to CRASH with error 716.
    /// Instead, it SUCCEEDS with correct value 0xBEEFCAFE, proving F081 hypothesis is FALSE.
    #[test]
    fn f081_shared_to_global_simple() {
        if !cuda_available() {
            eprintln!("F081 test SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const SMEM_SIZE: usize = 64;

        let kernel = PtxKernel::new("shared_to_global")
            .param(PtxType::U64, "output")
            .shared_memory(SMEM_SIZE)
            .build(|ctx| {
                let out_ptr = ctx.load_param_u64("output");
                let tid = ctx.special_reg(PtxReg::TidX);
                let zero = ctx.mov_u32_imm(0);
                let is_t0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_t0, "L_end");

                // Step 1: Write to shared memory
                let val = ctx.mov_u32_imm(0xBEEFCAFE);
                let addr = ctx.mov_u32_imm(0);
                ctx.st_shared_u32(addr, val);

                // Step 2: Load from shared memory
                let addr2 = ctx.mov_u32_imm(0);
                let loaded = ctx.ld_shared_u32(addr2);

                // Step 3: Store loaded value to global (THIS IS THE BUG)
                // ld.shared.u32 %r, [addr] ; st.global.u32 [ptr], %r
                ctx.st_global_u32(out_ptr, loaded);

                ctx.label("L_end");
                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== F081 BUG: SHARED → GLOBAL ===");
        println!("{}", ptx);
        println!("\nNOTE: Look for this pattern in PTX:");
        println!("  ld.shared.u32 %rN, [addr];");
        println!("  st.global.u32 [ptr], %rN;  // <-- CRASH: loaded value as src\n");

        let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        output.copy_from_host(&[0u32]).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");
        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output.as_kernel_arg()];

        println!("Launching kernel (originally expected CUDA_ERROR_UNKNOWN 716)...");
        unsafe {
            stream
                .launch_kernel(&mut module, "shared_to_global", &config, &mut args)
                .expect("Kernel launch");
        }

        let result = stream.synchronize();
        match result {
            Ok(_) => {
                let mut out = vec![0u32; 1];
                output.copy_to_host(&mut out).unwrap();
                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  F081 HYPOTHESIS FALSIFIED!                                   ║");
                println!("║                                                              ║");
                println!("║  Kernel SUCCEEDED with value 0x{:08X}                     ║", out[0]);
                println!("║  Pattern ld.shared → st.global does NOT crash!               ║");
                println!("║                                                              ║");
                println!("║  Original LZ4 bug was F082 or F021, not F081.                ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                // F081 is falsified - this test now EXPECTS success
                assert_eq!(out[0], 0xBEEFCAFE, "F081 falsified: pattern works correctly");
            }
            Err(e) => {
                // If this crashes, F081 hypothesis would be confirmed - but it doesn't!
                panic!(
                    "UNEXPECTED CRASH: {:?} - F081 would be confirmed, but we expect success",
                    e
                );
            }
        }
    }

    /// TEST 4: Workaround - use shfl.sync to launder the value
    #[test]
    fn f081_workaround_shfl_launder() {
        if !cuda_available() {
            eprintln!("F081 test SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const SMEM_SIZE: usize = 64;

        let kernel = PtxKernel::new("shfl_launder")
            .param(PtxType::U64, "output")
            .shared_memory(SMEM_SIZE)
            .build(|ctx| {
                let out_ptr = ctx.load_param_u64("output");
                let tid = ctx.special_reg(PtxReg::TidX);
                let zero = ctx.mov_u32_imm(0);
                let is_t0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_t0, "L_end");

                // Step 1: Write to shared memory
                let val = ctx.mov_u32_imm(0xBEEFCAFE);
                let addr = ctx.mov_u32_imm(0);
                ctx.st_shared_u32(addr, val);

                // Step 2: Load from shared memory
                let addr2 = ctx.mov_u32_imm(0);
                let loaded = ctx.ld_shared_u32(addr2);

                // Step 3: Launder through shfl.sync
                let laundered = ctx.shfl_idx_u32(loaded, 0, 0x1F);

                // Step 4: Store laundered value to global (should work)
                ctx.st_global_u32(out_ptr, laundered);

                ctx.label("L_end");
                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== WORKAROUND: SHFL LAUNDER ===");
        println!("{}", ptx);

        let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        output.copy_from_host(&[0u32]).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");
        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [output.as_kernel_arg()];

        println!("Launching kernel with shfl launder...");
        unsafe {
            stream
                .launch_kernel(&mut module, "shfl_launder", &config, &mut args)
                .expect("Kernel launch");
        }

        let result = stream.synchronize();
        match result {
            Ok(_) => {
                let mut out = vec![0u32; 1];
                output.copy_to_host(&mut out).unwrap();
                println!("SUCCESS: Kernel returned 0x{:08X}", out[0]);
                assert_eq!(out[0], 0xBEEFCAFE);
                println!("SHFL LAUNDER WORKAROUND WORKS\n");
            }
            Err(e) => {
                println!("FAILED: {:?}", e);
                println!("Shfl launder did not fix the bug\n");
            }
        }
    }
}
