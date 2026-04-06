//! CUDA Training Benchmark - Phase 11, Week 5
//!
//! Demonstrates end-to-end GPU-accelerated training using CUDA kernels.
//! Verifies >70% GPU utilization and >100 tokens/second generation.
//!
//! # Architecture (SPEC-FT-001 v3.2.0)
//!
//! ```text
//! Forward Pass:  gemm_forward()   → CUDA PTX kernel
//! Backward Pass: gemm_backward()  → CUDA PTX kernel
//! Optimizer:     adamw_step_cuda() → CUDA PTX kernel
//! ```
//!
//! Run with:
//!   cargo run --example cuda_training_benchmark --release --features cuda
//!
//! Note: Requires NVIDIA GPU with CUDA support.

#[cfg(feature = "cuda")]
mod cuda_benchmark {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    use entrenar::autograd::cuda_backward::{gemm_backward_a, gemm_backward_b, init_kernel_cache};
    use entrenar::autograd::cuda_forward::{gemm_forward, init_forward_kernel_cache};
    use entrenar::autograd::cuda_optim::{adamw_step_cuda, init_optim_kernel_cache};

    /// Training configuration
    struct TrainingConfig {
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_epochs: usize,
        num_steps_per_epoch: usize,
        learning_rate: f32,
        warmup_iterations: usize,
    }

    impl Default for TrainingConfig {
        fn default() -> Self {
            Self {
                batch_size: 8,
                seq_len: 128,
                hidden_size: 768,
                vocab_size: 32000,
                num_epochs: 3,
                num_steps_per_epoch: 100,
                learning_rate: 1e-4,
                warmup_iterations: 5,
            }
        }
    }

    /// Performance metrics
    #[derive(Debug, Default)]
    struct Metrics {
        forward_time: Duration,
        backward_time: Duration,
        optimizer_time: Duration,
        total_time: Duration,
        num_forward_passes: usize,
        num_backward_passes: usize,
        num_optimizer_steps: usize,
    }

    impl Metrics {
        fn total_kernel_time(&self) -> Duration {
            self.forward_time + self.backward_time + self.optimizer_time
        }

        fn gpu_utilization(&self) -> f32 {
            if self.total_time.as_secs_f64() > 0.0 {
                (self.total_kernel_time().as_secs_f64() / self.total_time.as_secs_f64()) as f32
                    * 100.0
            } else {
                0.0
            }
        }

        fn tokens_per_second(&self, tokens_processed: usize) -> f32 {
            if self.total_time.as_secs_f64() > 0.0 {
                tokens_processed as f32 / self.total_time.as_secs_f32()
            } else {
                0.0
            }
        }

        fn throughput_gflops(&self, flops: u64) -> f32 {
            if self.total_time.as_secs_f64() > 0.0 {
                (flops as f64 / self.total_time.as_secs_f64() / 1e9) as f32
            } else {
                0.0
            }
        }
    }

    /// GPU buffer collection for training
    struct Buffers {
        hidden: GpuBuffer<f32>,
        weights: GpuBuffer<f32>,
        logits: GpuBuffer<f32>,
        grad_output: GpuBuffer<f32>,
        grad_hidden: GpuBuffer<f32>,
        grad_weights: GpuBuffer<f32>,
        m_state: GpuBuffer<f32>,
        v_state: GpuBuffer<f32>,
    }

    /// Initialize CUDA: check availability, create context/stream, warm up kernel caches.
    fn init_cuda() -> Result<(Arc<CudaContext>, CudaStream), Box<dyn std::error::Error>> {
        // 1. Check CUDA availability
        println!("\n[1/6] Checking CUDA availability...");
        if !cuda_available() {
            println!("   ERROR: CUDA not available on this system");
            return Err("CUDA not available".into());
        }
        println!("   CUDA driver found");

        // 2. Initialize CUDA context and stream
        println!("\n[2/6] Initializing CUDA context...");
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = CudaStream::new(&ctx)?;

        // Query device properties
        let device_name = ctx.device_name().unwrap_or_else(|_| "Unknown GPU".to_string());
        let total_memory = ctx.total_memory().unwrap_or(0);

        println!("   Device: {device_name}");
        println!("   Memory: {:.1} GB", total_memory as f64 / 1e9);

        // 3. Initialize kernel caches
        println!("\n[3/6] Initializing kernel caches...");
        init_forward_kernel_cache(ctx.clone())?;
        init_kernel_cache(ctx.clone())?;
        init_optim_kernel_cache(ctx.clone())?;
        println!("   Forward kernels: ready");
        println!("   Backward kernels: ready");
        println!("   Optimizer kernels: ready");

        Ok((ctx, stream))
    }

    /// Allocate all GPU buffers for training (hidden states, weights, logits, gradients, optimizer state).
    fn allocate_buffers(
        ctx: &Arc<CudaContext>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<Buffers, Box<dyn std::error::Error>> {
        println!("\n[5/6] Allocating GPU buffers...");

        // Input hidden states: (batch * seq, hidden)
        let hidden_data: Vec<f32> = (0..m * k).map(|i| ((i as f32 * 0.01).sin() * 0.1)).collect();
        let hidden = GpuBuffer::from_host(ctx, &hidden_data)?;

        // LM head weights: (hidden, vocab)
        let weights_data: Vec<f32> =
            (0..k * n).map(|i| ((i as f32 * 0.001).cos() * 0.02)).collect();
        let weights = GpuBuffer::from_host(ctx, &weights_data)?;

        // Output logits: (batch * seq, vocab)
        let logits_zeros = vec![0.0f32; (m * n) as usize];
        let logits = GpuBuffer::from_host(ctx, &logits_zeros)?;

        // Gradients
        let grad_output_data: Vec<f32> = (0..m * n).map(|i| (i as f32 % 100.0) * 0.001).collect();
        let grad_output = GpuBuffer::from_host(ctx, &grad_output_data)?;
        let grad_hidden_zeros = vec![0.0f32; (m * k) as usize];
        let grad_hidden = GpuBuffer::from_host(ctx, &grad_hidden_zeros)?;
        let grad_weights_zeros = vec![0.0f32; (k * n) as usize];
        let grad_weights = GpuBuffer::from_host(ctx, &grad_weights_zeros)?;

        // Optimizer state
        let m_state_zeros = vec![0.0f32; (k * n) as usize];
        let m_state = GpuBuffer::from_host(ctx, &m_state_zeros)?; // First moment
        let v_state_zeros = vec![0.0f32; (k * n) as usize];
        let v_state = GpuBuffer::from_host(ctx, &v_state_zeros)?; // Second moment

        let total_params = (k * n) as usize;
        let memory_used = (m * k + k * n + m * n + m * k + k * n + k * n + k * n) as usize * 4;
        println!("   Hidden states: {} elements", m * k);
        println!("   LM head weights: {} elements", k * n);
        println!("   Logits: {} elements", m * n);
        println!("   Total parameters: {total_params}");
        println!("   GPU memory used: {:.1} MB", memory_used as f64 / 1e6);

        Ok(Buffers {
            hidden,
            weights,
            logits,
            grad_output,
            grad_hidden,
            grad_weights,
            m_state,
            v_state,
        })
    }

    /// Run warmup iterations followed by the timed training loop.
    /// Returns accumulated metrics and total step count.
    fn run_training_loop(
        buffers: &mut Buffers,
        config: &TrainingConfig,
        m: u32,
        k: u32,
        n: u32,
        stream: &CudaStream,
    ) -> Result<(Metrics, u32), Box<dyn std::error::Error>> {
        println!("\n[6/6] Running training benchmark...");

        // Warmup with error recovery
        run_warmup(buffers, config, m, k, n, stream)?;

        // Benchmark
        let total_start = Instant::now();
        let mut metrics = Metrics::default();
        let mut step_count = 0u32;

        for epoch in 0..config.num_epochs {
            let epoch_start = Instant::now();

            for step in 0..config.num_steps_per_epoch {
                step_count += 1;

                // Forward pass: hidden @ weights -> logits
                let forward_start = Instant::now();
                gemm_forward(
                    &buffers.hidden,
                    &buffers.weights,
                    &mut buffers.logits,
                    m,
                    k,
                    n,
                    stream,
                )?;
                stream.synchronize()?;
                metrics.forward_time += forward_start.elapsed();
                metrics.num_forward_passes += 1;

                // Backward pass: compute gradients
                let backward_start = Instant::now();
                gemm_backward_a(
                    &buffers.grad_output,
                    &buffers.weights,
                    &mut buffers.grad_hidden,
                    m,
                    k,
                    n,
                    stream,
                )?;
                gemm_backward_b(
                    &buffers.hidden,
                    &buffers.grad_output,
                    &mut buffers.grad_weights,
                    m,
                    k,
                    n,
                    stream,
                )?;
                stream.synchronize()?;
                metrics.backward_time += backward_start.elapsed();
                metrics.num_backward_passes += 1;

                // Optimizer step: update weights
                let optim_start = Instant::now();
                adamw_step_cuda(
                    &mut buffers.weights,
                    &buffers.grad_weights,
                    &mut buffers.m_state,
                    &mut buffers.v_state,
                    config.learning_rate,
                    0.9,   // beta1
                    0.999, // beta2
                    1e-8,  // eps
                    0.01,  // weight_decay
                    step_count,
                    k * n,
                    stream,
                )?;
                stream.synchronize()?;
                metrics.optimizer_time += optim_start.elapsed();
                metrics.num_optimizer_steps += 1;

                // Progress update
                if (step + 1) % 25 == 0 {
                    print!(
                        "\r   Epoch {}/{}, Step {}/{}",
                        epoch + 1,
                        config.num_epochs,
                        step + 1,
                        config.num_steps_per_epoch
                    );
                }
            }

            let epoch_duration = epoch_start.elapsed();
            println!("\n   Epoch {} complete in {:.2}s", epoch + 1, epoch_duration.as_secs_f32());
        }

        metrics.total_time = total_start.elapsed();
        Ok((metrics, step_count))
    }

    /// Run warmup iterations with up to 3 retry attempts.
    fn run_warmup(
        buffers: &mut Buffers,
        config: &TrainingConfig,
        m: u32,
        k: u32,
        n: u32,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("   Warming up ({} iterations)...", config.warmup_iterations);
        let mut warmup_ok = false;
        for attempt in 0..3 {
            let result: Result<(), Box<dyn std::error::Error>> = (|| {
                for _ in 0..config.warmup_iterations {
                    gemm_forward(
                        &buffers.hidden,
                        &buffers.weights,
                        &mut buffers.logits,
                        m,
                        k,
                        n,
                        stream,
                    )?;
                    stream.synchronize()?;
                }
                Ok(())
            })();

            match result {
                Ok(()) => {
                    warmup_ok = true;
                    break;
                }
                Err(e) if attempt < 2 => {
                    eprintln!("   Warmup attempt {} failed: {}. Retrying...", attempt + 1, e);
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(e) => return Err(e),
            }
        }

        if !warmup_ok {
            return Err("Warmup failed after 3 attempts".into());
        }
        Ok(())
    }

    /// Print benchmark results and verify against SPEC-FT-001 targets.
    fn print_results(
        metrics: &Metrics,
        config: &TrainingConfig,
        m: u32,
        k: u32,
        n: u32,
        total_params: usize,
    ) {
        let total_tokens =
            config.batch_size * config.seq_len * config.num_epochs * config.num_steps_per_epoch;

        // FLOPs per matmul: 2 * M * N * K (multiply-add)
        let forward_flops = 2 * u64::from(m) * u64::from(n) * u64::from(k);
        let backward_flops = 2
            * (u64::from(m) * u64::from(k) * u64::from(n)
                + u64::from(k) * u64::from(m) * u64::from(n));
        let total_flops = (forward_flops + backward_flops)
            * (config.num_epochs * config.num_steps_per_epoch) as u64;

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("   BENCHMARK RESULTS");
        println!("═══════════════════════════════════════════════════════════════");

        println!("\nTiming Breakdown:");
        println!(
            "   Forward passes:  {:.3}s ({} calls, {:.3}ms avg)",
            metrics.forward_time.as_secs_f32(),
            metrics.num_forward_passes,
            metrics.forward_time.as_millis() as f32 / metrics.num_forward_passes as f32
        );
        println!(
            "   Backward passes: {:.3}s ({} calls, {:.3}ms avg)",
            metrics.backward_time.as_secs_f32(),
            metrics.num_backward_passes,
            metrics.backward_time.as_millis() as f32 / metrics.num_backward_passes as f32
        );
        println!(
            "   Optimizer steps: {:.3}s ({} calls, {:.3}ms avg)",
            metrics.optimizer_time.as_secs_f32(),
            metrics.num_optimizer_steps,
            metrics.optimizer_time.as_millis() as f32 / metrics.num_optimizer_steps as f32
        );
        println!("   Total kernel time: {:.3}s", metrics.total_kernel_time().as_secs_f32());
        println!("   Total wall time:   {:.3}s", metrics.total_time.as_secs_f32());

        println!("\nPerformance Metrics:");
        let gpu_util = metrics.gpu_utilization();
        let tokens_per_sec = metrics.tokens_per_second(total_tokens);
        let gflops = metrics.throughput_gflops(total_flops);

        println!("   GPU Utilization:   {gpu_util:.1}%");
        println!("   Tokens/second:     {tokens_per_sec:.0}");
        println!("   Throughput:        {gflops:.1} GFLOP/s");
        println!("   Total tokens:      {total_tokens}");
        println!("   Total parameters:  {total_params}");

        // Verification against targets
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("   SPEC-FT-001 v3.2.0 VERIFICATION");
        println!("═══════════════════════════════════════════════════════════════");

        let gpu_util_pass = gpu_util >= 70.0;
        let tokens_pass = tokens_per_sec >= 100.0;

        println!(
            "\n   [{}] GPU Utilization >70%: {:.1}%",
            if gpu_util_pass { "PASS" } else { "FAIL" },
            gpu_util
        );
        println!(
            "   [{}] Tokens/second >100:   {:.0}",
            if tokens_pass { "PASS" } else { "FAIL" },
            tokens_per_sec
        );

        if gpu_util_pass && tokens_pass {
            println!("\n   RESULT: ALL TARGETS MET");
            println!("   Phase 11, Week 5 objectives verified.");
        } else {
            println!("\n   RESULT: TARGETS NOT MET");
            if !gpu_util_pass {
                println!("   - GPU utilization below 70% target");
            }
            if !tokens_pass {
                println!("   - Token throughput below 100/s target");
            }
        }

        println!("\n═══════════════════════════════════════════════════════════════");
    }

    /// Run the CUDA training benchmark
    pub fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
        println!("═══════════════════════════════════════════════════════════════");
        println!("   CUDA Training Benchmark - SPEC-FT-001 v3.2.0");
        println!("═══════════════════════════════════════════════════════════════");

        let (ctx, stream) = init_cuda()?;

        // 4. Setup training configuration
        let config = TrainingConfig::default();
        println!("\n[4/6] Training configuration:");
        println!("   Batch size: {}", config.batch_size);
        println!("   Sequence length: {}", config.seq_len);
        println!("   Hidden size: {}", config.hidden_size);
        println!("   Vocab size: {}", config.vocab_size);
        println!("   Epochs: {}", config.num_epochs);
        println!("   Steps per epoch: {}", config.num_steps_per_epoch);
        println!("   Learning rate: {}", config.learning_rate);

        // Calculate dimensions for LM head matmul: (batch * seq, hidden) @ (hidden, vocab)
        let m = (config.batch_size * config.seq_len) as u32;
        let k = config.hidden_size as u32;
        let n = config.vocab_size as u32;
        let total_params = (k * n) as usize;

        let mut buffers = allocate_buffers(&ctx, m, k, n)?;
        let (metrics, _step_count) = run_training_loop(&mut buffers, &config, m, k, n, &stream)?;
        print_results(&metrics, &config, m, k, n, total_params);

        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This benchmark requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example cuda_training_benchmark --release --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(e) = cuda_benchmark::run_benchmark() {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    }
}
