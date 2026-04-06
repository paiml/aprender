//! Debug test to isolate which kernel in batched_multihead_attention fails

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{GemmKernel, Kernel, ScaleKernel, SoftmaxKernel, TransposeKernel};

/// Launch a kernel and synchronize the stream, returning Ok(()) on success.
/// Prints diagnostics and returns Err(()) on launch or sync failure.
#[cfg(feature = "cuda")]
unsafe fn launch_and_sync(
    stream: &CudaStream,
    module: &mut CudaModule,
    kernel_name: &str,
    config: &LaunchConfig,
    args: &mut Vec<*mut std::ffi::c_void>,
    label: &str,
) -> Result<(), ()> {
    match unsafe { stream.launch_kernel(module, kernel_name, config, args) } {
        Ok(_) => println!("  Kernel launched"),
        Err(e) => {
            println!("  {} launch FAILED: {:?}", label, e);
            return Err(());
        }
    }
    match stream.synchronize() {
        Ok(_) => {
            println!("  {} succeeded!", label);
            Ok(())
        }
        Err(e) => {
            println!("  {} CRASHED: {:?}", label, e);
            Err(())
        }
    }
}

/// Compile PTX into a CudaModule, printing the PTX size.
#[cfg(feature = "cuda")]
fn compile_ptx(ctx: &CudaContext, ptx: &str) -> CudaModule {
    println!("  PTX generated ({} bytes)", ptx.len());
    let module = CudaModule::from_ptx(ctx, ptx).expect("PTX compile failed");
    println!("  Module compiled");
    module
}

/// Run the TransposeKernel step and verify the result.
#[cfg(feature = "cuda")]
fn run_transpose_step(
    ctx: &CudaContext,
    stream: &CudaStream,
    input_buf: &GpuBuffer<f32>,
    input_data: &[f32],
    rows: u32,
    cols: u32,
) -> Result<GpuBuffer<f32>, ()> {
    let total = (rows * cols) as usize;
    println!(
        "Step 2: TransposeKernel [{}x{}] -> [{}x{}]...",
        rows, cols, cols, rows
    );
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, total).expect("Alloc failed");

    let transpose = TransposeKernel::new(rows, cols);
    let ptx = transpose.emit_ptx();
    let mut module = compile_ptx(ctx, &ptx);

    let threads = 256u32;
    let blocks = (total as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };
    println!(
        "  Launch config: grid=({}, 1, 1), block=({}, 1, 1)",
        blocks, threads
    );

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();
    println!("  Input ptr: 0x{:x}", input_ptr);
    println!("  Output ptr: 0x{:x}", output_ptr);

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    unsafe {
        launch_and_sync(
            stream,
            &mut module,
            transpose.name(),
            &config,
            &mut args,
            "TransposeKernel",
        )?
    };

    // Read back and verify
    println!("Step 3: Verify transpose result...");
    let mut result = vec![0.0f32; total];
    output_buf
        .copy_to_host(&mut result)
        .expect("Readback failed");

    let expected_0_0 = input_data[0];
    let expected_1_0 = input_data[1];
    println!(
        "  input[0,0]={:.1} -> output[0,0]={:.1} (expected {:.1})",
        input_data[0], result[0], expected_0_0
    );
    println!(
        "  input[0,1]={:.1} -> output[1,0]={:.1} (expected {:.1})",
        input_data[1], result[rows as usize], expected_1_0
    );

    Ok(output_buf)
}

/// Run the GemmKernel step: [m x k] @ [k x n] = [m x n].
#[cfg(feature = "cuda")]
fn run_gemm_step(
    ctx: &CudaContext,
    stream: &CudaStream,
    input_buf: &GpuBuffer<f32>,
    transposed_buf: &GpuBuffer<f32>,
    total: usize,
    m: u32,
    n: u32,
    k: u32,
) -> Result<GpuBuffer<f32>, ()> {
    println!(
        "Step 4: GemmKernel [{}x{}] @ [{}x{}] = [{}x{}]...",
        m, k, k, n, m, n
    );
    let c_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, (m * n) as usize).expect("Alloc C failed");

    let gemm = GemmKernel::naive(m, n, k);
    let ptx = gemm.emit_ptx();
    let mut module = compile_ptx(ctx, &ptx);

    let block_size = 16u32;
    let grid_x = (n + block_size - 1) / block_size;
    let grid_y = (m + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid: (grid_x, grid_y, 1),
        block: (block_size, block_size, 1),
        shared_mem: 0,
    };
    println!(
        "  Launch config: grid=({}, {}, 1), block=({}, {}, 1)",
        grid_x, grid_y, block_size, block_size
    );

    let a_ptr = input_buf.as_ptr();
    let b_ptr = transposed_buf.as_ptr();
    let c_ptr = c_buf.as_ptr();
    let m_val = m;
    let n_val = n;
    let k_val = k;

    println!("  A ptr: 0x{:x} (size {})", a_ptr, total);
    println!("  B ptr: 0x{:x} (size {})", b_ptr, total);
    println!("  C ptr: 0x{:x} (size {})", c_ptr, (m * n) as usize);
    println!("  M={}, N={}, K={}", m_val, n_val, k_val);

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(c_ptr) as *mut _,
        std::ptr::addr_of!(m_val) as *mut _,
        std::ptr::addr_of!(n_val) as *mut _,
        std::ptr::addr_of!(k_val) as *mut _,
    ];

    unsafe {
        launch_and_sync(
            stream,
            &mut module,
            gemm.name(),
            &config,
            &mut args,
            "GemmKernel",
        )?
    };
    Ok(c_buf)
}

/// Run the ScaleKernel step: scale elements by a constant.
#[cfg(feature = "cuda")]
fn run_scale_step(
    ctx: &CudaContext,
    stream: &CudaStream,
    c_buf: &GpuBuffer<f32>,
    size: usize,
) -> Result<GpuBuffer<f32>, ()> {
    println!("Step 5: Scale by 0.5...");
    let scale_out_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, size).expect("Alloc scale out");

    let scale_kernel = ScaleKernel::new(size as u32);
    let ptx = scale_kernel.emit_ptx();
    let mut module = compile_ptx(ctx, &ptx);

    let threads = 256u32;
    let blocks = ((size as u32) + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let scale_input_ptr = c_buf.as_ptr();
    let scale_output_ptr = scale_out_buf.as_ptr();
    let scale_val = 0.5f32;
    let scale_n = size as u32;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(scale_input_ptr) as *mut _,
        std::ptr::addr_of!(scale_output_ptr) as *mut _,
        std::ptr::addr_of!(scale_val) as *mut _,
        std::ptr::addr_of!(scale_n) as *mut _,
    ];

    unsafe {
        launch_and_sync(
            stream,
            &mut module,
            scale_kernel.name(),
            &config,
            &mut args,
            "Scale kernel",
        )?
    };
    Ok(scale_out_buf)
}

/// Run the SoftmaxKernel step: softmax over rows.
#[cfg(feature = "cuda")]
fn run_softmax_step(
    ctx: &CudaContext,
    stream: &CudaStream,
    scale_out_buf: &GpuBuffer<f32>,
    sm_rows: u32,
    sm_row_size: u32,
) -> Result<(), ()> {
    let sm_total = (sm_rows * sm_row_size) as usize;
    println!(
        "Step 6: Softmax [{} rows x {} cols]...",
        sm_rows, sm_row_size
    );

    let sm_out_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, sm_total).expect("Alloc softmax out");

    let sm_kernel = SoftmaxKernel::new(sm_row_size);
    let ptx = sm_kernel.emit_ptx();
    let mut module = compile_ptx(ctx, &ptx);

    let threads_per_block = (sm_row_size as usize).min(256) as u32;
    let config = LaunchConfig {
        grid: (sm_rows, 1, 1),
        block: (threads_per_block, 1, 1),
        shared_mem: (sm_row_size as usize * std::mem::size_of::<f32>()) as u32,
    };
    println!(
        "  Launch config: grid=({}, 1, 1), block=({}, 1, 1), smem={}",
        sm_rows, threads_per_block, config.shared_mem
    );

    let sm_input_ptr = scale_out_buf.as_ptr();
    let sm_output_ptr = sm_out_buf.as_ptr();
    let sm_row_size_val = sm_row_size;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(sm_input_ptr) as *mut _,
        std::ptr::addr_of!(sm_output_ptr) as *mut _,
        std::ptr::addr_of!(sm_row_size_val) as *mut _,
    ];

    unsafe {
        launch_and_sync(
            stream,
            &mut module,
            sm_kernel.name(),
            &config,
            &mut args,
            "Softmax kernel",
        )?
    };
    Ok(())
}

/// Debug test to isolate which kernel in batched_multihead_attention fails
#[test]
#[cfg(feature = "cuda")]
fn test_debug_isolate_crash() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    println!("\n=== DEBUG: Isolating crash ===\n");

    // Small test: 4x16 matrix (seq_len=4, d_model=16)
    let rows = 4u32;
    let cols = 16u32;
    let total = (rows * cols) as usize;
    let input_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();

    // Step 1: Upload to GPU
    println!("Step 1: Upload to GPU...");
    let input_buf = GpuBuffer::from_host(&ctx, &input_data).expect("Upload failed");
    println!("  Upload succeeded");

    let stream = CudaStream::new(&ctx).expect("Stream failed");

    // Step 2-3: Transpose and verify
    let output_buf = match run_transpose_step(&ctx, &stream, &input_buf, &input_data, rows, cols) {
        Ok(buf) => buf,
        Err(()) => return,
    };

    // Step 4: Gemm
    let m = 4u32;
    let n = 4u32;
    let k = 16u32;
    let c_buf = match run_gemm_step(&ctx, &stream, &input_buf, &output_buf, total, m, n, k) {
        Ok(buf) => buf,
        Err(()) => return,
    };

    // Step 5: Scale
    let scale_out_buf = match run_scale_step(&ctx, &stream, &c_buf, (m * n) as usize) {
        Ok(buf) => buf,
        Err(()) => return,
    };

    // Step 6: Softmax
    if run_softmax_step(&ctx, &stream, &scale_out_buf, m, n).is_err() {
        return;
    }

    println!("\n=== All kernels passed! ===");
}
