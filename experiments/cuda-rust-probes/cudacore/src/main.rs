// Falsifiable probe: does NVIDIA cuda-core 0.3.1 RUN on this host?
// Proves the mechanism engaged: device name, cc, PTX module load, kernel launch, result readback.
use cuda_core::{launch_kernel_on_stream, CudaContext, DeviceBuffer};
use std::ffi::c_void;

// Minimal hand-written PTX (same shape aprender's builder emits) targeting sm_89.
const PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64
.visible .entry scale2(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 p_n
) {
    .reg .pred  %p<2>;
    .reg .f32   %f<3>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<7>;
    ld.param.u64 %rd1, [p_in];
    ld.param.u64 %rd2, [p_out];
    ld.param.u32 %r2, [p_n];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra DONE;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    add.f32 %f2, %f1, %f1;
    add.s64 %rd6, %rd4, %rd5;
    st.global.f32 [%rd6], %f2;
DONE:
    ret;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    println!("device.name  = {}", ctx.device_name()?);
    let cc = ctx.compute_capability()?;
    println!("compute_cap  = sm_{}{}", cc.0, cc.1);
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();

    let module = CudaContext::load_module_from_ptx_src(&ctx, PTX)?;
    println!("ptx module   = LOADED via load_module_from_ptx_src");
    let func = module.load_function("scale2")?;
    println!("function     = RESOLVED scale2");
    // The occupancy/limits oracle aprender's hand-PTX path has no equivalent of:
    println!("  max_threads_per_block    = {}", func.max_threads_per_block()?);
    println!("  num_registers            = {}", func.num_registers()?);
    println!("  static_shared_mem_bytes  = {}", func.static_shared_memory_bytes()?);
    println!("  local_size_bytes         = {}", func.local_size_bytes()?);

    let d_in = DeviceBuffer::from_host(&stream, &host)?;
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let mut p_in = d_in.cu_deviceptr();
    let mut p_out = d_out.cu_deviceptr();
    let mut n: u32 = N as u32;
    let mut params: [*mut c_void; 3] = [
        &mut p_in as *mut _ as *mut c_void,
        &mut p_out as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    unsafe {
        launch_kernel_on_stream(&func, ((N as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, &stream, &mut params)?;
    }
    let out = d_out.to_host_vec(&stream)?;
    let bad = (0..N).filter(|&i| (out[i] - 2.0 * host[i]).abs() > 1e-5).count();
    println!("launch       = ran; mismatches = {bad}; out[7]={} (expect 14)", out[7]);
    println!("{}", if bad == 0 { "VERDICT: PASS - cuda-core drives this GPU end-to-end" } else { "VERDICT: FAIL" });
    Ok(())
}
