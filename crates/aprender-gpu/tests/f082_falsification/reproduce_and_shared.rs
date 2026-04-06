//! F082 Tests 3-4: Reproduce F082 pattern and shared-to-shared computed address

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

/// F082-TEST-3: Control test - the exact F082 pattern (should CRASH)
///
/// This confirms we can reproduce F082 in a minimal kernel
#[test]
fn f082_test3_reproduce_f082() {
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test3(
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

    // Get shared memory base (32-bit offset)
    mov.u32 %r3, 0;

    // Store a 32-bit offset value to shared memory
    mov.u32 %r4, 16;
    st.shared.u32 [%r3], %r4;

    // THE F082 PATTERN:
    // 1. Load 32-bit value from shared memory
    ld.shared.u32 %r5, [%r3];

    // 2. Convert to 64-bit
    cvt.u64.u32 %rd1, %r5;

    // 3. Compute address from loaded value
    add.u64 %rd2, %rd0, %rd1;

    // 4. Store to computed address (THIS SHOULD CRASH)
    mov.u32 %r6, 0xF082DEAD;
    st.global.u32 [%rd2], %r6;

L_skip:
    ret;
}"#;

    println!("F082-TEST-3: Reproduce F082 pattern (expected to CRASH)");

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

    let result = unsafe { stream.launch_kernel(&mut module, "f082_test3", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut output = vec![0u32; 256];
                output_buf.copy_to_host(&mut output).unwrap();

                if output[4] == 0xF082DEAD {
                    println!("  UNEXPECTED: F082 pattern WORKS in minimal kernel!");
                    println!("  -> F082 may be triggered by kernel complexity, not this pattern");
                } else {
                    println!("  Data mismatch - got {:08X} at index 4", output[4]);
                }
            }
            Err(e) => {
                println!("  CRASHED as expected: {}", e);
                println!("  -> Confirmed: F082 pattern reproduces in minimal kernel");
            }
        },
        Err(e) => {
            println!("  Launch failed: {}", e);
        }
    }
}

/// F082-TEST-4: Shared->Shared computed address (no global involved)
///
/// If this CRASHES -> Bug is in shared memory address computation
/// If this WORKS -> Bug is specifically in shared->global crossing
#[test]
fn f082_test4_shared_to_shared_computed_addr() {
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test4(
    .param .u64 output_ptr
) {
    .shared .align 4 .b8 smem[1024];
    .reg .u64 %rd<20>;
    .reg .u32 %r<20>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Store offset value at smem[0]
    mov.u32 %r3, 0;
    mov.u32 %r4, 64;  // Target: smem[64]
    st.shared.u32 [%r3], %r4;

    // Store data at smem[4]
    mov.u32 %r5, 4;
    mov.u32 %r6, 0x5AAED01;
    st.shared.u32 [%r5], %r6;

    // Load offset from shared memory (32-bit)
    ld.shared.u32 %r7, [%r3];

    // Compute target address IN SHARED MEMORY (not global!)
    // Just use the loaded value directly as 32-bit offset
    // This is shared->shared, NOT shared->global

    // Load data from smem[4]
    ld.shared.u32 %r8, [%r5];

    // Store to COMPUTED shared address
    st.shared.u32 [%r7], %r8;

    // Verify by loading from target and storing to global
    ld.shared.u32 %r9, [%r7];
    st.global.u32 [%rd0], %r9;

L_skip:
    bar.sync 0;
    ret;
}"#;

    println!("F082-TEST-4: Shared->Shared computed address");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();
    let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

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
        shared_mem: 1024,
    };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    let result = unsafe { stream.launch_kernel(&mut module, "f082_test4", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut output = vec![0u32; 1];
                output_buf.copy_to_host(&mut output).unwrap();

                if output[0] == 0x5AAED01 {
                    println!("  PASSED - Shared->Shared computed address WORKS");
                    println!("  -> Bug is in shared->GLOBAL crossing specifically");
                } else {
                    println!("  Data mismatch - got {:08X}", output[0]);
                }
            }
            Err(e) => {
                println!("  CRASHED: {}", e);
                println!("  -> Bug is in shared memory address computation itself");
            }
        },
        Err(e) => {
            println!("  Launch failed: {}", e);
        }
    }
}
