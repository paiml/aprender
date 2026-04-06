//! FKR-012: PTX Debug Buffer Infrastructure Tests
//!
//! Tests the atomic debug buffer mechanism for kernel debugging.

#[cfg(feature = "cuda")]
mod fkr_012_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{
        PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType,
    };

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// FKR-012a: Test atomic add to debug buffer
    #[test]
    fn fkr_012a_atomic_debug_buffer() {
        if !cuda_available() {
            eprintln!("FKR-012a SKIPPED: No CUDA device available");
            return;
        }
        let kernel =
            PtxKernel::new("debug_buffer_test").param(PtxType::U64, "debug_buf").build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let debug_ptr = ctx.load_param_u64("debug_buf");

                // Each thread writes a debug marker
                // marker = 0xDEAD0000 | tid
                let marker_base = ctx.mov_u32_imm(0xDEAD0000);
                let marker = ctx.or_u32(marker_base, tid);

                let _slot = ctx.emit_debug_value(debug_ptr, marker);

                ctx.ret();
            });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== Debug Buffer PTX ===\n{}", ptx);

        // Verify PTX contains atom.global.add
        assert!(ptx.contains("atom.global.add"), "PTX should contain atomic add");

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        // Allocate debug buffer: [counter, slot0, slot1, ..., slot31]
        let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
        // Initialize counter to 0
        let zeros = vec![0u32; 64];
        debug_buf.copy_from_host(&zeros).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compilation");

        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

        let mut args: [*mut c_void; 1] = [debug_buf.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "debug_buffer_test", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut output = vec![0u32; 64];
        debug_buf.copy_to_host(&mut output).unwrap();

        println!("Debug buffer counter: {}", output[0]);
        println!("Debug buffer values: {:08X?}", &output[1..33]);

        // Counter should be 32 (one write per thread)
        assert_eq!(output[0], 32, "Counter should be 32");

        // Each slot should have 0xDEADxxxx where xxxx is a thread ID
        let mut seen_tids = std::collections::HashSet::new();
        for &val in &output[1..33] {
            let marker_high = val & 0xFFFF0000;
            let tid = val & 0x0000FFFF;
            assert_eq!(marker_high, 0xDEAD0000, "Marker high bits should be 0xDEAD");
            assert!(tid < 32, "TID should be < 32");
            seen_tids.insert(tid);
        }
        assert_eq!(seen_tids.len(), 32, "Should have seen all 32 thread IDs");

        println!("FKR-012a: Atomic debug buffer PASSED!");
    }

    /// FKR-012b: Test emit_debug_marker with constant markers
    #[test]
    fn fkr_012b_debug_markers() {
        if !cuda_available() {
            eprintln!("FKR-012b SKIPPED: No CUDA device available");
            return;
        }
        const MARKER_ENTRY: u32 = 0x11111111;
        const MARKER_MIDDLE: u32 = 0x22222222;
        const MARKER_EXIT: u32 = 0x33333333;

        let kernel = PtxKernel::new("marker_test").param(PtxType::U64, "debug_buf").build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let debug_ptr = ctx.load_param_u64("debug_buf");

            // Only thread 0 writes markers
            let zero = ctx.mov_u32_imm(0);
            let is_thread0 = ctx.setp_eq_u32(tid, zero);
            ctx.branch_if_not(is_thread0, "L_skip");

            // Emit entry marker
            ctx.emit_debug_marker(debug_ptr, MARKER_ENTRY);

            // Emit middle marker
            ctx.emit_debug_marker(debug_ptr, MARKER_MIDDLE);

            // Emit exit marker
            ctx.emit_debug_marker(debug_ptr, MARKER_EXIT);

            ctx.label("L_skip");
            ctx.ret();
        });

        let ptx = PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(kernel)
            .emit();

        println!("=== Marker Test PTX ===\n{}", ptx);

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
        let zeros = vec![0u32; 64];
        debug_buf.copy_from_host(&zeros).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compilation");

        let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 4 };

        let mut args: [*mut c_void; 1] = [debug_buf.as_kernel_arg()];

        unsafe {
            stream
                .launch_kernel(&mut module, "marker_test", &config, &mut args)
                .expect("Kernel launch");
        }

        stream.synchronize().expect("Sync");

        let mut output = vec![0u32; 64];
        debug_buf.copy_to_host(&mut output).unwrap();

        println!("Debug buffer counter: {}", output[0]);
        for i in 0..output[0].min(10) as usize {
            let marker = output[i + 1];
            let name = match marker {
                0x11111111 => "ENTRY",
                0x22222222 => "MIDDLE",
                0x33333333 => "EXIT",
                _ => "UNKNOWN",
            };
            println!("  Marker {}: 0x{:08X} ({})", i, marker, name);
        }

        // Counter should be 3 (entry, middle, exit)
        assert_eq!(output[0], 3, "Should have 3 markers");
        // First marker should be ENTRY
        assert_eq!(output[1], MARKER_ENTRY, "First marker should be ENTRY");
        // Second marker should be MIDDLE
        assert_eq!(output[2], MARKER_MIDDLE, "Second marker should be MIDDLE");
        // Third marker should be EXIT
        assert_eq!(output[3], MARKER_EXIT, "Third marker should be EXIT");

        println!("FKR-012b: Debug markers PASSED!");
    }
}

#[cfg(not(feature = "cuda"))]
mod fkr_012_tests {
    #[test]
    fn fkr_012_skip_no_cuda() {
        println!("FKR-012: Skipped - CUDA feature not enabled");
    }
}
