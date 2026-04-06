//! F082 Tests 1-2: Global computed address and explicit 64-bit load

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

/// F082-TEST-1: Global->Global computed address
///
/// If this CRASHES -> F082 is NOT shared-memory-specific
/// If this WORKS -> Confirms cross-address-space (shared->global) is key
#[test]
fn f082_test1_global_to_global_computed_addr() {
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test1(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 offset_ptr
) {
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;

    // Load params
    ld.param.u64 %rd0, [input_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [offset_ptr];

    // Load offset from GLOBAL memory (not shared)
    ld.global.u32 %r0, [%rd2];

    // Convert to 64-bit for address computation
    cvt.u64.u32 %rd3, %r0;

    // Compute target address from loaded value
    add.u64 %rd4, %rd1, %rd3;

    // Load data from input
    ld.global.u32 %r1, [%rd0];

    // Store to COMPUTED address (F082 pattern but global->global)
    st.global.u32 [%rd4], %r1;

    ret;
}"#;

    println!("F082-TEST-1: Global->Global computed address");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    // Allocate buffers
    let mut input_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();
    let mut offset_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

    // Set offset to 16 (bytes) = index 4
    offset_buf.copy_from_host(&[16u32]).unwrap();
    input_buf.copy_from_host(&[0xDEADBEEF_u32]).unwrap();

    let mut module = match CudaModule::from_ptx(&ctx, ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 3] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        offset_buf.as_kernel_arg(),
    ];

    let result = unsafe { stream.launch_kernel(&mut module, "f082_test1", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut output = vec![0u32; 256];
                output_buf.copy_to_host(&mut output).unwrap();

                if output[4] == 0xDEADBEEF {
                    println!("  PASSED - Global->Global computed address WORKS");
                    println!("  -> F082 is SHARED-MEMORY-SPECIFIC");
                } else {
                    println!("  Data mismatch - got {:08X} at index 4", output[4]);
                }
            }
            Err(e) => {
                println!("  CRASHED at sync: {}", e);
                println!("  -> F082 is NOT shared-memory-specific (REFUTES hypothesis)");
            }
        },
        Err(e) => {
            println!("  Launch failed: {}", e);
        }
    }
}

/// F082-TEST-2: Explicit 64-bit shared load (no conversion)
///
/// If this WORKS -> Type conversion (cvt.u64.u32) is the bug
/// If this CRASHES -> Conversion is not the issue
#[test]
fn f082_test2_explicit_64bit_load() {
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test2(
    .param .u64 output_ptr
) {
    .shared .align 8 .b8 smem[64];
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Get shared memory base
    cvta.shared.u64 %rd1, smem;

    // Store a 64-bit value (16) to shared memory
    mov.u64 %rd2, 16;
    st.shared.u64 [%rd1], %rd2;

    // Load as 64-bit directly (NO conversion needed)
    ld.shared.u64 %rd3, [%rd1];

    // Compute target address (no cvt.u64.u32!)
    add.u64 %rd4, %rd0, %rd3;

    // Store marker
    mov.u32 %r3, 0xCAFEBABE;
    st.global.u32 [%rd4], %r3;

L_skip:
    ret;
}"#;

    println!("F082-TEST-2: Explicit 64-bit load (no conversion)");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();
    let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();

    let mut module = match CudaModule::from_ptx(&ctx, ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 64,
    };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    let result = unsafe { stream.launch_kernel(&mut module, "f082_test2", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut output = vec![0u32; 256];
                output_buf.copy_to_host(&mut output).unwrap();

                if output[4] == 0xCAFEBABE {
                    println!("  PASSED - 64-bit load WORKS");
                    println!("  -> cvt.u64.u32 conversion IS the bug!");
                    println!("  -> FIX: Use 64-bit shared memory values for addresses");
                } else {
                    println!("  Data mismatch - got {:08X} at index 4", output[4]);
                }
            }
            Err(e) => {
                println!("  CRASHED at sync: {}", e);
                println!("  -> 64-bit load doesn't help (conversion not the issue)");
            }
        },
        Err(e) => {
            println!("  Launch failed: {}", e);
        }
    }
}
