//! F082 Test 5: membar.gl between load and store

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

/// F082-TEST-5: membar.gl between load and store
///
/// Tests if global-level memory barrier prevents the crash
#[test]
fn f082_test5_membar_gl() {
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test5(
    .param .u64 output_ptr
) {
    .shared .align 4 .b8 smem[64];
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

    // Store offset to shared
    mov.u32 %r3, 0;
    mov.u32 %r4, 16;
    st.shared.u32 [%r3], %r4;

    // CTA barrier
    membar.cta;

    // Load offset from shared
    ld.shared.u32 %r5, [%r3];

    // GLOBAL memory barrier (stronger than CTA)
    membar.gl;

    // Convert to 64-bit
    cvt.u64.u32 %rd1, %r5;

    // Another global barrier after conversion
    membar.gl;

    // Compute address
    add.u64 %rd2, %rd0, %rd1;

    // Final global barrier before store
    membar.gl;

    // Store
    mov.u32 %r6, 0xBABA6001;
    st.global.u32 [%rd2], %r6;

L_skip:
    ret;
}"#;

    println!("F082-TEST-5: membar.gl between operations");

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

    let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 64 };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    let result = unsafe { stream.launch_kernel(&mut module, "f082_test5", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut output = vec![0u32; 256];
                output_buf.copy_to_host(&mut output).unwrap();

                if output[4] == 0xBABA6001 {
                    println!("  PASSED - membar.gl WORKS");
                    println!("  -> REFUTES 'barriers don't work' claim");
                    println!("  -> FIX: Use membar.gl (not just membar.cta)");
                } else {
                    println!("  Data mismatch - got {:08X} at index 4", output[4]);
                }
            }
            Err(e) => {
                println!("  CRASHED: {}", e);
                println!("  -> membar.gl doesn't help (supports original F082 hypothesis)");
            }
        },
        Err(e) => {
            println!("  Launch failed: {}", e);
        }
    }
}
