//! PMAT-3725 / aprender#3725: the context rungs for the split-K decode attention.
//!
//! For each geometry (the 9B's 16/4 and the 27B's 24/4 heads, `head_dim = 256`), each
//! KV storage (f32, f16) and each rung `L` (4096 … 262,144), this runs the kernel the
//! row replaces (`DecodeAttention256Kernel`) and the split-K pair on the SAME cache and
//! records:
//!
//! - parity of split-K against the old kernel (max |Δ| relative to max |old|, cosine);
//! - both kernels against an f64 reference (the ground truth neither f32 path is);
//! - µs per call for both: host wall clock over `iters` back-to-back launches and one
//!   stream synchronize, after `warmup` launches. (`CudaEvent` here is created with
//!   timing disabled, so there is no device-event timer to use.)
//!
//! For f16 the split-K pair reads the f16 cache and the old kernel reads the SAME
//! values widened to f32, which isolates the read path from storage rounding.
//!
//! ```bash
//! flock /tmp/apr-gpu.lock choom -n 1000 -- \
//!   cargo run --release -p aprender-gpu --features cuda --example gdn_splitk_rungs -- \
//!   --out evidence/section-3725/lambda.json --host lambda --sha "$(git rev-parse --short HEAD)"
//! ```
//!
//! The 262,144 rung holds 2 GiB of f32 K+V plus the widened copy for f16: this is an
//! example, not a test, for that reason.

#[cfg(feature = "cuda")]
mod rungs {
    use std::time::Instant;

    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::gdn::{
        decode_attention_reference_f64, splitk_partial_acc_len, splitk_partial_ml_len,
        DecodeAttention256Kernel, DecodeAttentionSplitKKernel, DecodeAttentionSplitKReduceKernel,
        KvStorage, SplitKPlan,
    };
    use trueno_gpu::kernels::Kernel;

    const HEAD_DIM: usize = 256;

    struct Args {
        out: Option<String>,
        host: String,
        sha: String,
        rungs: Vec<usize>,
        warmup: usize,
        iters: usize,
        reference_max: usize,
        emit_ptx: Option<String>,
    }

    fn parse_args() -> Args {
        let mut a = Args {
            out: None,
            host: "unknown".into(),
            sha: "unknown".into(),
            rungs: vec![4096, 8192, 20_000, 60_000, 148_000, 262_144],
            warmup: 3,
            iters: 20,
            reference_max: 262_144,
            emit_ptx: None,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let v = argv.get(i + 1).cloned().unwrap_or_default();
            match argv[i].as_str() {
                "--out" => a.out = Some(v),
                "--host" => a.host = v,
                "--sha" => a.sha = v,
                "--rungs" => {
                    a.rungs = v
                        .split(',')
                        .map(|s| s.trim().parse().expect("rung"))
                        .collect()
                }
                "--warmup" => a.warmup = v.parse().expect("warmup"),
                "--iters" => a.iters = v.parse().expect("iters"),
                "--reference-max" => a.reference_max = v.parse().expect("reference-max"),
                "--emit-ptx" => a.emit_ptx = Some(v),
                other => panic!("unknown argument {other}"),
            }
            i += 2;
        }
        a
    }

    /// A seeded LCG in `[-scale, scale)`.
    struct Lcg(u32);
    impl Lcg {
        fn next_scaled(&mut self, scale: f32) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f32::from(((self.0 >> 16) & 0xFFFF) as u16) / 32768.0 - 1.0) * scale
        }
    }

    fn f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let a = x.abs();
        if a < 6.103_515_6e-5 {
            return sign | (a * 16_777_216.0).round() as u16;
        }
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = b & 0x7f_ffff;
        let mut h = ((exp as u32) << 10) | (mant >> 13);
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
            h += 1;
        }
        sign | h as u16
    }

    fn widen(b: u16) -> f32 {
        let sign = if b & 0x8000 == 0 { 1.0f32 } else { -1.0 };
        let e = i32::from((b >> 10) & 0x1f);
        let m = f32::from(b & 0x3ff);
        if e == 0 {
            sign * m * 2f32.powi(-24)
        } else {
            sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
        }
    }

    fn launch<K: Kernel>(
        stream: &CudaStream,
        module: &mut CudaModule,
        kernel: &K,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [u64],
    ) {
        let config = LaunchConfig {
            grid,
            block,
            shared_mem: 0,
        };
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: every pointer argument is a live device allocation sized to what the
        // kernel indexes at this seq_len; grid/block are the kernel's own launch shape.
        unsafe {
            stream
                .launch_kernel(module, kernel.name(), &config, &mut raw)
                .expect("launch");
        }
    }

    struct Stats {
        rel_max: f64,
        cos: f64,
    }

    fn compare(got: &[f32], want: &[f64]) -> Stats {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(want) {
            let g = f64::from(*g);
            worst = worst.max((g - w).abs());
            dot += g * w;
            ng += g * g;
            nw += w * w;
        }
        Stats {
            rel_max: worst / scale,
            cos: dot / (ng.sqrt() * nw.sqrt()),
        }
    }

    fn widen_f64(x: &[f32]) -> Vec<f64> {
        x.iter().map(|v| f64::from(*v)).collect()
    }

    /// Write the builder PTX of the split-K pair (for this device's target) so the
    /// cuda-oxide port can A/B against the exact kernels in one process.
    fn emit_ptx(dir: &str, ctx: &CudaContext) {
        let (major, minor) = ctx.compute_capability().expect("compute capability");
        let target = format!("sm_{major}{minor}");
        std::fs::create_dir_all(dir).expect("ptx dir");
        for (nh, nkv) in [(16u32, 4u32), (24, 4)] {
            for kv in [KvStorage::F32, KvStorage::F16] {
                let a = DecodeAttentionSplitKKernel::new(nh, nkv, HEAD_DIM as u32, kv);
                let path = format!("{dir}/{}_{nh}_{nkv}.{target}.ptx", a.name());
                std::fs::write(&path, a.emit_ptx_for_target(&target)).expect("write ptx");
                println!("wrote {path}");
            }
            let b = DecodeAttentionSplitKReduceKernel::new(nh, HEAD_DIM as u32);
            let path = format!("{dir}/{}_{nh}.{target}.ptx", b.name());
            std::fs::write(&path, b.emit_ptx_for_target(&target)).expect("write ptx");
            println!("wrote {path}");
        }
    }

    /// q, K and V for one geometry, `max_len` positions. Rows differ in scale so a
    /// kernel reading the wrong position or KV head lands somewhere else.
    fn fixture(nh: usize, nkv: usize, max_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let row = nkv * HEAD_DIM;
        let mut rng = Lcg(0x3725_7000 + nh as u32);
        let q: Vec<f32> = (0..nh * HEAD_DIM).map(|_| rng.next_scaled(0.3)).collect();
        let mut k = Vec::with_capacity(max_len * row);
        let mut v = Vec::with_capacity(max_len * row);
        for p in 0..max_len {
            let scale = 0.5 + 0.5 * ((p % 97) as f32 / 97.0);
            k.extend((0..row).map(|_| rng.next_scaled(scale)));
            v.extend((0..row).map(|_| rng.next_scaled(scale)));
        }
        (q, k, v)
    }

    /// Host wall µs per call of `f`: `warmup` launches, then `iters` back-to-back
    /// launches and one synchronize.
    fn time_us(stream: &CudaStream, warmup: usize, iters: usize, f: &mut dyn FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        stream.synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        stream.synchronize().expect("sync");
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    }

    /// What one rung measured.
    struct Rung {
        seq_len: usize,
        plan: SplitKPlan,
        old_us: f64,
        new_us: f64,
        vs_old: Stats,
        old_f64: Option<Stats>,
        new_f64: Option<Stats>,
    }

    fn fmt_rel(s: Option<&Stats>) -> String {
        s.map_or("null".into(), |s| format!("{:.3e}", s.rel_max))
    }

    fn print_rung(nh: usize, nkv: usize, kv: KvStorage, r: &Rung) {
        println!(
            "{:>6} {:>4} {:>7} {:>11.1} {:>11.1} {:>7.2} {:>10.3e} {:>9.7} {:>10} {:>10}",
            format!("{nh}/{nkv}"),
            kv.tag(),
            r.seq_len,
            r.old_us,
            r.new_us,
            r.old_us / r.new_us,
            r.vs_old.rel_max,
            r.vs_old.cos,
            fmt_rel(r.old_f64.as_ref()),
            fmt_rel(r.new_f64.as_ref()),
        );
    }

    fn rung_json(nh: usize, nkv: usize, kv: KvStorage, r: &Rung) -> String {
        format!(
            "    {{\"num_heads\": {nh}, \"num_kv_heads\": {nkv}, \"head_dim\": {HEAD_DIM}, \
             \"kv\": \"{}\", \"seq_len\": {}, \"chunk\": {}, \"n_splits\": {}, \
             \"old_us\": {:.2}, \"splitk_us\": {:.2}, \"speedup\": {:.3}, \
             \"splitk_vs_old_rel_max\": {:.3e}, \"splitk_vs_old_cos\": {:.9}, \
             \"old_vs_f64_rel_max\": {}, \"splitk_vs_f64_rel_max\": {}}}",
            kv.tag(),
            r.seq_len,
            r.plan.chunk,
            r.plan.n_splits,
            r.old_us,
            r.new_us,
            r.old_us / r.new_us,
            r.vs_old.rel_max,
            r.vs_old.cos,
            fmt_rel(r.old_f64.as_ref()),
            fmt_rel(r.new_f64.as_ref()),
        )
    }

    /// The device side of one geometry that every storage and rung shares.
    struct Shared {
        nh: usize,
        nkv: usize,
        old: DecodeAttention256Kernel,
        red: DecodeAttentionSplitKReduceKernel,
        old_mod: CudaModule,
        red_mod: CudaModule,
        q_buf: GpuBuffer<f32>,
        old_out: GpuBuffer<f32>,
        new_out: GpuBuffer<f32>,
        pacc: GpuBuffer<f32>,
        pml: GpuBuffer<f32>,
    }

    impl Shared {
        fn new(ctx: &CudaContext, nh: usize, nkv: usize, q: &[f32], max_len: usize) -> Self {
            let old = DecodeAttention256Kernel::new(nh as u32, nkv as u32, HEAD_DIM as u32);
            let red = DecodeAttentionSplitKReduceKernel::new(nh as u32, HEAD_DIM as u32);
            let max_splits = SplitKPlan::for_seq_len(max_len as u32).n_splits.max(256);
            Self {
                nh,
                nkv,
                old_mod: CudaModule::from_ptx(ctx, &old.emit_ptx()).expect("old module"),
                red_mod: CudaModule::from_ptx(ctx, &red.emit_ptx()).expect("reduce module"),
                old,
                red,
                q_buf: GpuBuffer::from_host(ctx, q).expect("q"),
                old_out: GpuBuffer::<f32>::new(ctx, nh * HEAD_DIM).expect("old out"),
                new_out: GpuBuffer::<f32>::new(ctx, nh * HEAD_DIM).expect("new out"),
                pacc: GpuBuffer::<f32>::new(
                    ctx,
                    splitk_partial_acc_len(nh as u32, HEAD_DIM as u32, max_splits),
                )
                .expect("pacc"),
                pml: GpuBuffer::<f32>::new(ctx, splitk_partial_ml_len(nh as u32, max_splits))
                    .expect("pml"),
            }
        }
    }

    /// One storage's caches on the device, and the values every kernel sees as f32.
    struct Cache {
        k_seen: Vec<f32>,
        v_seen: Vec<f32>,
        k32: GpuBuffer<f32>,
        v32: GpuBuffer<f32>,
        half: Option<(GpuBuffer<u16>, GpuBuffer<u16>)>,
    }

    impl Cache {
        fn new(ctx: &CudaContext, kv: KvStorage, k: &[f32], v: &[f32]) -> Self {
            let (k_seen, v_seen, half) = match kv {
                KvStorage::F32 => (k.to_vec(), v.to_vec(), None),
                KvStorage::F16 => {
                    let kb: Vec<u16> = k.iter().map(|x| f16_bits(*x)).collect();
                    let vb: Vec<u16> = v.iter().map(|x| f16_bits(*x)).collect();
                    let widened = (
                        kb.iter().map(|b| widen(*b)).collect(),
                        vb.iter().map(|b| widen(*b)).collect(),
                    );
                    let dev = (
                        GpuBuffer::from_host(ctx, &kb).expect("k16"),
                        GpuBuffer::from_host(ctx, &vb).expect("v16"),
                    );
                    (widened.0, widened.1, Some(dev))
                }
            };
            Self {
                k32: GpuBuffer::from_host(ctx, &k_seen).expect("k32"),
                v32: GpuBuffer::from_host(ctx, &v_seen).expect("v32"),
                k_seen,
                v_seen,
                half,
            }
        }

        /// The pointers the split-K kernel reads: the f16 cache if there is one.
        fn splitk_ptrs(&self) -> (u64, u64) {
            self.half
                .as_ref()
                .map_or((self.k32.as_ptr(), self.v32.as_ptr()), |(k, v)| {
                    (k.as_ptr(), v.as_ptr())
                })
        }
    }

    /// Both kernels on the same cache at one rung: timing, then parity.
    #[allow(clippy::too_many_arguments)]
    fn run_rung(
        stream: &CudaStream,
        args: &Args,
        sh: &mut Shared,
        cache: &Cache,
        a: &DecodeAttentionSplitKKernel,
        a_mod: &mut CudaModule,
        q: &[f32],
        seq_len: usize,
    ) -> Rung {
        let plan = SplitKPlan::for_seq_len(seq_len as u32);
        let (k_ptr, v_ptr) = cache.splitk_ptrs();
        let mut old_args = [
            sh.q_buf.as_ptr(),
            cache.k32.as_ptr(),
            cache.v32.as_ptr(),
            sh.old_out.as_ptr(),
            seq_len as u64,
        ];
        let mut a_args = [
            sh.q_buf.as_ptr(),
            k_ptr,
            v_ptr,
            sh.pacc.as_ptr(),
            sh.pml.as_ptr(),
            seq_len as u64,
            u64::from(plan.chunk),
            u64::from(plan.n_splits),
        ];
        let mut b_args = [
            sh.pacc.as_ptr(),
            sh.pml.as_ptr(),
            sh.new_out.as_ptr(),
            u64::from(plan.n_splits),
        ];
        let (old, red) = (sh.old, sh.red);
        let old_us = time_us(stream, args.warmup, args.iters, &mut || {
            launch(
                stream,
                &mut sh.old_mod,
                &old,
                old.grid(),
                old.block(),
                &mut old_args,
            );
        });
        let new_us = time_us(stream, args.warmup, args.iters, &mut || {
            launch(stream, a_mod, a, a.grid(plan), a.block(), &mut a_args);
            launch(
                stream,
                &mut sh.red_mod,
                &red,
                red.grid(),
                red.block(),
                &mut b_args,
            );
        });

        let mut got_old = vec![0.0f32; sh.nh * HEAD_DIM];
        let mut got_new = vec![0.0f32; sh.nh * HEAD_DIM];
        sh.old_out.copy_to_host(&mut got_old).expect("old download");
        sh.new_out.copy_to_host(&mut got_new).expect("new download");
        let reference = (seq_len <= args.reference_max).then(|| {
            decode_attention_reference_f64(
                q,
                &cache.k_seen,
                &cache.v_seen,
                sh.nh,
                sh.nkv,
                HEAD_DIM,
                seq_len,
            )
        });
        Rung {
            seq_len,
            plan,
            old_us,
            new_us,
            vs_old: compare(&got_new, &widen_f64(&got_old)),
            old_f64: reference.as_ref().map(|r| compare(&got_old, r)),
            new_f64: reference.as_ref().map(|r| compare(&got_new, r)),
        }
    }

    /// Every storage and rung at one geometry, as JSON rows.
    fn run_geometry(
        ctx: &CudaContext,
        stream: &CudaStream,
        args: &Args,
        nh: usize,
        nkv: usize,
        max_len: usize,
    ) -> Vec<String> {
        let (q, k, v) = fixture(nh, nkv, max_len);
        let mut sh = Shared::new(ctx, nh, nkv, &q, max_len);
        let mut rows = Vec::new();
        for kv in [KvStorage::F32, KvStorage::F16] {
            let cache = Cache::new(ctx, kv, &k, &v);
            let a = DecodeAttentionSplitKKernel::new(nh as u32, nkv as u32, HEAD_DIM as u32, kv);
            let mut a_mod = CudaModule::from_ptx(ctx, &a.emit_ptx()).expect("split-K module");
            for &seq_len in &args.rungs {
                let r = run_rung(stream, args, &mut sh, &cache, &a, &mut a_mod, &q, seq_len);
                print_rung(nh, nkv, kv, &r);
                rows.push(rung_json(nh, nkv, kv, &r));
            }
        }
        rows
    }

    pub fn main() {
        let args = parse_args();
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        if let Some(dir) = &args.emit_ptx {
            emit_ptx(dir, &ctx);
            return;
        }
        let stream = CudaStream::new(&ctx).expect("stream");
        let device = ctx.device_name().unwrap_or_else(|_| "?".into());
        let (cc_major, cc_minor) = ctx.compute_capability().unwrap_or((0, 0));
        let sms = ctx.multiprocessor_count().unwrap_or(0);
        let (free0, total0) = ctx.memory_info().unwrap_or((0, 0));
        let max_len = *args.rungs.iter().max().expect("at least one rung");

        println!(
            "{:>6} {:>4} {:>7} {:>11} {:>11} {:>7} {:>10} {:>9} {:>10} {:>10}",
            "heads",
            "kv",
            "L",
            "old_us",
            "splitk_us",
            "x",
            "rel_vs_old",
            "cos",
            "old_vs_f64",
            "new_vs_f64"
        );
        let mut rows = Vec::new();
        for (nh, nkv) in [(16usize, 4usize), (24, 4)] {
            rows.extend(run_geometry(&ctx, &stream, &args, nh, nkv, max_len));
        }

        let json = format!(
            "{{\n  \"schema\": \"apr-3725-splitk-rungs/v1\",\n  \"host\": \"{}\",\n  \"sha\": \"{}\",\n  \
             \"device\": \"{device}\",\n  \"compute_capability\": \"{cc_major}.{cc_minor}\",\n  \"sms\": {sms},\n  \
             \"free_mib_at_start\": {},\n  \"total_mib\": {},\n  \"timing\": \"host wall over {} back-to-back launches + one stream sync, after {} warmup\",\n  \
             \"rows\": [\n{}\n  ]\n}}\n",
            args.host,
            args.sha,
            free0 / (1 << 20),
            total0 / (1 << 20),
            args.iters,
            args.warmup,
            rows.join(",\n"),
        );
        if let Some(path) = &args.out {
            std::fs::write(path, json).expect("write receipt");
            println!("wrote {path}");
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    rungs::main();
    #[cfg(not(feature = "cuda"))]
    eprintln!("gdn_splitk_rungs needs --features cuda");
}
