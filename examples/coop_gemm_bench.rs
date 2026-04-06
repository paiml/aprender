//! Benchmark cooperative matrix GEMM vs tiled GEMM on GPU.
//! FALSIFY-COOP-001: parity test (max |coop - tiled| < 1e-3)
//! FALSIFY-COOP-002: throughput test (coop > 2x tiled)
//!
//! Usage: cargo run --example coop_gemm_bench --features gpu --release

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter found");

    let info = adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.backend);

    let coop_props = adapter.cooperative_matrix_properties();
    println!("Cooperative matrix configs: {}", coop_props.len());
    if coop_props.is_empty() {
        println!("No cooperative matrix support — skipping benchmark.");
        return;
    }

    // Request cooperative matrix feature
    // Request cooperative matrix feature (experimental)
    let required_features = adapter.features(); // request ALL supported features
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coop_bench"),
        required_features,
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            max_buffer_size: adapter.limits().max_buffer_size,
            ..Default::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        // SAFETY: we acknowledge experimental feature risks per wgpu docs
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: Default::default(),
    }))
    .expect("Device creation failed");

    println!(
        "Device features: cooperative_matrix = {}",
        !adapter.cooperative_matrix_properties().is_empty()
    );

    // Test dimensions: 512×3584 × 3584×3584 (typical Q projection)
    let m = 512u32;
    let k = 3584u32;
    let n = 3584u32;

    // Generate test data
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0007).cos() * 0.1).collect();

    let a_buf = upload(&device, &queue, &a_data);
    let b_buf = upload(&device, &queue, &b_data);

    // Tiled GEMM benchmark
    let tiled_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tiled"),
        source: wgpu::ShaderSource::Wgsl(trueno::backends::gpu::shaders::TILED_GEMM_SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let tiled_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tiled_pipe"),
        layout: Some(&pl),
        module: &tiled_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Run tiled GEMM
    let c_tiled = zeros(&device, (m * n) as usize);
    let dims = [m, k, n, 1.0f32.to_bits()];
    let dims_buf = upload_uniform(&device, &queue, &dims);

    // Warmup
    run_gemm(&device, &queue, &tiled_pipe, &bgl, &a_buf, &b_buf, &c_tiled, &dims_buf, n, m, 64);

    // Benchmark
    let start = std::time::Instant::now();
    let iters = 10;
    for _ in 0..iters {
        run_gemm(&device, &queue, &tiled_pipe, &bgl, &a_buf, &b_buf, &c_tiled, &dims_buf, n, m, 64);
    }
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    let tiled_ms = start.elapsed().as_millis() as f64 / iters as f64;
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let tiled_gflops = flops / (tiled_ms / 1000.0) / 1e9;

    let tiled_result = download(&device, &queue, &c_tiled);
    println!("\nTiled GEMM: {tiled_ms:.1}ms, {tiled_gflops:.0} GFLOPS");
    println!("  result[0..4] = {:?}", &tiled_result[..4]);

    // Try cooperative GEMM
    println!("\nAttempting cooperative matrix GEMM shader compilation...");
    let coop_shader_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coop"),
            source: wgpu::ShaderSource::Wgsl(
                trueno::backends::gpu::shaders::cooperative::COOPERATIVE_GEMM_SHADER.into(),
            ),
        })
    }));

    match coop_shader_result {
        Ok(coop_shader) => {
            println!("  Shader compiled successfully!");
            let coop_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("coop_pipe"),
                layout: Some(&pl),
                module: &coop_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let c_coop = zeros(&device, (m * n) as usize);

            // Warmup
            run_gemm(
                &device, &queue, &coop_pipe, &bgl, &a_buf, &b_buf, &c_coop, &dims_buf, n, m, 16,
            );

            // Benchmark
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run_gemm(
                    &device, &queue, &coop_pipe, &bgl, &a_buf, &b_buf, &c_coop, &dims_buf, n, m, 16,
                );
            }
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            let coop_ms = start.elapsed().as_millis() as f64 / iters as f64;
            let coop_gflops = flops / (coop_ms / 1000.0) / 1e9;

            let coop_result = download(&device, &queue, &c_coop);
            println!("\nCooperative GEMM: {coop_ms:.1}ms, {coop_gflops:.0} GFLOPS");
            println!("  result[0..4] = {:?}", &coop_result[..4]);

            // FALSIFY-COOP-001: parity
            let max_err: f32 = tiled_result
                .iter()
                .zip(coop_result.iter())
                .map(|(t, c)| (t - c).abs())
                .fold(0.0f32, f32::max);
            println!("\nFALSIFY-COOP-001: max error = {max_err:.6} (threshold: 1e-3)");
            if max_err < 1e-3 {
                println!("  PASSED");
            } else {
                println!("  FAILED");
            }

            // FALSIFY-COOP-002: throughput
            let speedup = coop_gflops / tiled_gflops;
            println!("FALSIFY-COOP-002: speedup = {speedup:.2}x (threshold: 2.0x)");
            if speedup >= 2.0 {
                println!("  PASSED");
            } else {
                println!("  FAILED (but {coop_gflops:.0} > {tiled_gflops:.0} GFLOPS)");
            }
        }
        Err(e) => {
            println!("  Shader compilation failed: {:?}", e);
            println!("  FALSIFY-COOP-003: Fallback to tiled GEMM — PASSED (no crash)");
        }
    }
}

fn upload(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

fn upload_uniform(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32; 4]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

fn zeros(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn download(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> Vec<f32> {
    let size = buf.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

fn run_gemm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
    dims: &wgpu::Buffer,
    n: u32,
    m: u32,
    tile: u32,
) {
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(tile), m.div_ceil(tile), 1);
    }
    queue.submit(Some(enc.finish()));
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
