//! Lower bound for a BATCHED Qwen3.5 prefill: the 13 projection shapes of Kev-0.8B's decoder blocks
//! (498M params; embedding is a lookup and Kev never runs lm_head) timed as [M x K] x [K x N] GEMMs at Kev's
//! row lengths, with the same trueno BLIS kernels the forward would use. Recurrence/attention/norm cost excluded.
use std::time::Instant;
use trueno::blis::{gemm_blis, gemm_blis_parallel};

const SHAPES: &[(usize, usize, usize)] = &[ // (count, K=in, N=out)
    (18, 1024, 2048), (6, 1024, 512), (6, 2048, 1024), (6, 1024, 4096), (18, 1024, 6144), (6, 1024, 512),
    (24, 3584, 1024), (24, 1024, 3584), (24, 1024, 3584), (18, 1024, 16), (18, 1024, 16), (18, 2048, 1024),
];

fn main() {
    let threads = rayon::current_num_threads();
    println!("| tokens M | single-thread s | GFLOP/s | parallel ({threads} thr) s | GFLOP/s | per-token ms (par) | token-at-a-time today ms |");
    println!("|---|---|---|---|---|---|---|");
    for &m in &[85usize, 300, 915] {
        let mut flops = 0f64; let (mut t1, mut tp) = (0f64, 0f64);
        for &(count, k, n) in SHAPES {
            let a = vec![0.5f32; m * k]; let b = vec![0.25f32; k * n]; let mut c = vec![0f32; m * n];
            // one warm call, then time one representative call and multiply by the layer count
            gemm_blis(m, n, k, &a, &b, &mut c, None).expect("gemm");
            let s = Instant::now(); gemm_blis(m, n, k, &a, &b, &mut c, None).expect("gemm"); t1 += s.elapsed().as_secs_f64() * count as f64;
            gemm_blis_parallel(m, n, k, &a, &b, &mut c).expect("gemm");
            let s = Instant::now(); gemm_blis_parallel(m, n, k, &a, &b, &mut c).expect("gemm"); tp += s.elapsed().as_secs_f64() * count as f64;
            flops += 2.0 * (m * k * n * count) as f64;
        }
        println!("| {m} | {t1:.3} | {:.0} | {tp:.3} | {:.0} | {:.2} | {:.0} |", flops / t1 / 1e9, flops / tp / 1e9, tp * 1e3 / m as f64, 78.9 * m as f64);
    }
}
