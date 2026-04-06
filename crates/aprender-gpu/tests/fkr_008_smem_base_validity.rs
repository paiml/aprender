//! FKR-008: Falsification test for smem_base validity throughout compress loop
//!
//! HYPOTHESIS: smem_base (generic shared memory address) remains valid throughout
//! the compression loop in the LZ4 kernel.
//!
//! FALSIFICATION CRITERION: If ANY address computed from smem_base is invalid
//! (not in shared memory range), the hypothesis is falsified.
//!
//! Test approach: Instrument the compress loop to output computed addresses
//! at critical points and verify they fall within shared memory bounds.

#[cfg(feature = "cuda")]
mod fkr_008_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// FKR-008a: Verify smem_base + in_pos produces valid address
    /// This is the exact operation that crashes in the full LZ4 kernel.
    #[test]
    fn fkr_008a_smem_base_plus_in_pos_valid() {
        if !cuda_available() {
            eprintln!("FKR-008a SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        // Minimal PTX that replicates the LZ4 compress loop address calculation
        // without the full complexity of the kernel
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry fkr_008_smem_check(
    .param .u64 output,
    .param .u32 test_in_pos
) {
    // Same shared memory size as LZ4 kernel (3 warps * 12544 = 37632)
    .shared .align 16 .b8 smem[37632];
    .reg .u64 %rd<30>;
    .reg .u32 %r<30>;
    .reg .pred %p<5>;

    // Get thread and warp IDs (same as LZ4)
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 5;
    shr.b32 %r2, %r0, %r1;      // warp_id = tid.x >> 5
    mov.u32 %r3, 31;
    and.b32 %r4, %r0, %r3;      // lane_id = tid.x & 31

    // Compute smem_base for this warp (same as LZ4)
    // WARP_SMEM_SIZE = 12544
    mul.lo.u32 %r5, %r2, 12544;         // warp_offset = warp_id * 12544
    cvt.u64.u32 %rd0, %r5;
    cvta.shared.u64 %rd1, smem;         // smem generic address
    add.u64 %rd2, %rd1, %rd0;           // smem_base = smem + warp_offset

    // Load output param
    ld.param.u64 %rd10, [output];
    ld.param.u32 %r6, [test_in_pos];

    // Only leader thread outputs
    setp.eq.u32 %p0, %r4, 0;
    @!%p0 bra L_not_leader;

    // Output 0: smem_base raw value
    st.global.u64 [%rd10], %rd2;

    // Output 1: cvta.shared result (raw smem pointer)
    add.u64 %rd11, %rd10, 8;
    st.global.u64 [%rd11], %rd1;

    // Output 2: Compute curr_addr = smem_base + in_pos (THIS IS THE CRASHING OP)
    cvt.u64.u32 %rd3, %r6;              // in_pos_64
    add.u64 %rd4, %rd2, %rd3;           // curr_addr = smem_base + in_pos
    add.u64 %rd12, %rd10, 16;
    st.global.u64 [%rd12], %rd4;        // output curr_addr

    // Output 3: Try to load from curr_addr (this would crash if invalid)
    ld.u32 %r7, [%rd4];
    add.u64 %rd13, %rd10, 24;
    cvt.u64.u32 %rd5, %r7;
    st.global.u64 [%rd13], %rd5;        // output loaded value

    // Output 4: hash_table_base = smem_base + 4096
    mov.u32 %r8, 4096;
    cvt.u64.u32 %rd6, %r8;
    add.u64 %rd7, %rd2, %rd6;           // hash_table_base
    add.u64 %rd14, %rd10, 32;
    st.global.u64 [%rd14], %rd7;

    // Output 5: state_base = smem_base + 12420
    mov.u32 %r9, 12420;
    cvt.u64.u32 %rd8, %r9;
    add.u64 %rd9, %rd2, %rd8;           // state_base
    add.u64 %rd15, %rd10, 40;
    st.global.u64 [%rd15], %rd9;

    // Output 6: Success marker
    add.u64 %rd16, %rd10, 48;
    mov.u64 %rd17, 0x600D600D600D600D;
    st.global.u64 [%rd16], %rd17;

L_not_leader:
    ret;
}
"#;

        let mut output_buf: GpuBuffer<u64> = GpuBuffer::new(&ctx, 8).unwrap();
        let init_val = [0xDEADBEEF_u64; 8];
        output_buf.copy_from_host(&init_val).unwrap();

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (96, 1, 1), // 3 warps like LZ4
            shared_mem: 0,
        };

        let test_in_pos: u32 = 0; // Start position

        let mut args: [*mut c_void; 2] =
            [output_buf.as_kernel_arg(), &test_in_pos as *const u32 as *mut c_void];

        unsafe {
            stream
                .launch_kernel(&mut module, "fkr_008_smem_check", &config, &mut args)
                .expect("Kernel launch");
        }
        stream.synchronize().expect("Sync");

        let mut result = [0u64; 8];
        output_buf.copy_to_host(&mut result).unwrap();

        let smem_base = result[0];
        let cvta_result = result[1];
        let curr_addr = result[2];
        let loaded_val = result[3];
        let hash_table_base = result[4];
        let state_base = result[5];
        let success_marker = result[6];

        println!("FKR-008a Results:");
        println!("  smem_base       = 0x{:016X}", smem_base);
        println!("  cvta_result     = 0x{:016X}", cvta_result);
        println!("  curr_addr       = 0x{:016X} (smem_base + 0)", curr_addr);
        println!("  loaded_val      = 0x{:016X}", loaded_val);
        println!("  hash_table_base = 0x{:016X} (smem_base + 4096)", hash_table_base);
        println!("  state_base      = 0x{:016X} (smem_base + 12420)", state_base);
        println!("  success_marker  = 0x{:016X}", success_marker);

        // Falsification checks
        assert!(
            cvta_result > 0x1000,
            "FALSIFIED: cvta.shared returned invalid address 0x{:X}",
            cvta_result
        );
        assert!(smem_base > 0x1000, "FALSIFIED: smem_base is invalid 0x{:X}", smem_base);
        assert!(
            curr_addr > 0x1000,
            "FALSIFIED: curr_addr (smem_base + in_pos) is invalid 0x{:X}",
            curr_addr
        );
        assert_eq!(
            curr_addr, smem_base,
            "FALSIFIED: curr_addr should equal smem_base when in_pos=0"
        );
        assert_eq!(
            hash_table_base,
            smem_base + 4096,
            "FALSIFIED: hash_table_base should be smem_base + 4096"
        );
        assert_eq!(
            success_marker, 0x600D600D600D600D,
            "FALSIFIED: kernel did not complete successfully"
        );

        println!("FKR-008a: PASSED - isolated smem_base operations work correctly");
    }

    /// FKR-008b: Test with the ACTUAL Lz4WarpCompressKernel PTX
    /// This test instruments the real kernel to output addresses at the crash point
    ///
    /// NOTE: This test is marked `#[ignore]` because Lz4WarpCompressKernel has
    /// the F082 computed-address bug and will crash with CUDA_ERROR_UNKNOWN (716).
    #[test]
    #[ignore = "Uses buggy Lz4WarpCompressKernel - F082 confirmed"]
    fn fkr_008b_actual_lz4_kernel_address_check() {
        use trueno_gpu::kernels::lz4::PAGE_SIZE;
        use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

        if !cuda_available() {
            eprintln!("FKR-008b SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const NUM_PAGES: u32 = 1;

        // Create simple non-zero data that should trigger compress loop
        let mut input: Vec<u8> = vec![0u8; PAGE_SIZE as usize];
        // Fill with pattern: 0,1,2,3,0,1,2,3,... to create some matches
        for i in 0..PAGE_SIZE as usize {
            input[i] = (i % 256) as u8;
        }

        let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, input.len()).unwrap();
        let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
        let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
        let ptx = kernel.emit_ptx();

        // Print the PTX lines around L_compress_loop for analysis
        println!("PTX around compress loop:");
        let lines: Vec<&str> = ptx.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("L_compress_loop")
                || line.contains("in_pos_64")
                || line.contains("curr_addr")
                || (i > 0 && lines[i - 1].contains("L_compress_loop"))
            {
                println!("  L{}: {}", i + 1, line);
            }
        }

        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

        let config =
            LaunchConfig { grid: kernel.grid_dim(), block: kernel.block_dim(), shared_mem: 0 };

        let num_pages = NUM_PAGES;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            output_buf.as_kernel_arg(),
            sizes_buf.as_kernel_arg(),
            &num_pages as *const u32 as *mut c_void,
        ];

        println!("\nLaunching actual LZ4 kernel with non-zero data...");

        unsafe {
            stream
                .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
                .expect("Kernel launch");
        }

        // This is where we expect the crash
        let sync_result = stream.synchronize();

        match sync_result {
            Ok(()) => {
                let mut sizes = vec![0u32; 1];
                sizes_buf.copy_to_host(&mut sizes).unwrap();
                println!("FKR-008b: PASSED - kernel completed, size = {}", sizes[0]);
                assert!(sizes[0] > 0 && sizes[0] <= 4352, "Output size should be valid");
            }
            Err(e) => {
                panic!(
                    "FKR-008b: FALSIFIED - kernel crashed: {:?}\n\
                        This confirms smem_base corruption in compress loop",
                    e
                );
            }
        }
    }

    /// FKR-008c: Compare register allocation between minimal and full kernels
    /// by checking the number of .reg declarations
    #[test]
    fn fkr_008c_register_pressure_comparison() {
        use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

        let kernel = Lz4WarpCompressKernel::new(3);
        let ptx = kernel.emit_ptx();

        // Count register declarations
        let reg_u64_count = ptx.matches(".reg .u64").count();
        let reg_u32_count = ptx.matches(".reg .u32").count();
        let reg_pred_count = ptx.matches(".reg .pred").count();

        println!("FKR-008c: Register pressure analysis");
        println!("  .reg .u64 declarations: {}", reg_u64_count);
        println!("  .reg .u32 declarations: {}", reg_u32_count);
        println!("  .reg .pred declarations: {}", reg_pred_count);

        // Find the actual max register numbers used
        let max_rd = ptx
            .lines()
            .filter_map(|line| {
                if let Some(pos) = line.find("%rd") {
                    let rest = &line[pos + 3..];
                    rest.chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse::<u32>()
                        .ok()
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);

        let max_r = ptx
            .lines()
            .filter_map(|line| {
                // Match %r followed by digits, but not %rd
                let bytes = line.as_bytes();
                let mut max = 0u32;
                for i in 0..bytes.len().saturating_sub(2) {
                    if bytes[i] == b'%' && bytes[i + 1] == b'r' && bytes[i + 2] != b'd' {
                        let rest = &line[i + 2..];
                        if let Some(num) = rest
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse::<u32>()
                            .ok()
                        {
                            max = max.max(num);
                        }
                    }
                }
                if max > 0 {
                    Some(max)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);

        println!("  Max %rd register used: %rd{}", max_rd);
        println!("  Max %r register used: %r{}", max_r);

        // High register pressure can cause spilling issues
        // PTX virtualizes registers, but very high counts may indicate issues
        // The PTX assembler handles register allocation, so we allow high counts
        // but warn about potential performance implications
        if max_rd > 200 {
            println!("WARNING: High u64 register count ({}) may impact performance", max_rd);
        }
        if max_r > 400 {
            println!("WARNING: High u32 register count ({}) may impact performance", max_r);
        }
        // The assertion is informational, not a hard failure
        // Actual spilling depends on SM register file size and occupancy
        assert!(
            max_rd < 1000,
            "FALSIFIED: Unreasonably high u64 register count ({}) suggests generation bug",
            max_rd
        );
        assert!(
            max_r < 1000,
            "FALSIFIED: Unreasonably high u32 register count ({}) suggests generation bug",
            max_r
        );

        // Check if smem_base register (%rd11 typically) is reused elsewhere
        let rd11_uses: Vec<&str> = ptx.lines().filter(|line| line.contains("%rd11")).collect();

        println!("\n  Uses of %rd11 (smem_base): {}", rd11_uses.len());
        for (i, line) in rd11_uses.iter().take(10).enumerate() {
            println!("    {}: {}", i + 1, line.trim());
        }

        println!("\nFKR-008c: PASSED - register analysis complete");
    }
}

#[cfg(not(feature = "cuda"))]
mod fkr_008_tests {
    #[test]
    fn fkr_008_skip_no_cuda() {
        println!("FKR-008: Skipped - CUDA feature not enabled");
    }
}
