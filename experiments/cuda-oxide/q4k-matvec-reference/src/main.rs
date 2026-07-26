use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits: u32 = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut e: i32 = -1; let mut m = mant;
            loop { e += 1; m <<= 1; if (m & 0x400) != 0 { break; } }
            (sign << 31) | (((127 - 15 - e) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F { (sign << 31) | (0xFF << 23) | (mant << 13) }
    else { (sign << 31) | ((exp + 112) << 23) | (mant << 13) };
    f32::from_bits(bits)
}
fn extract_scale_min(scales: &[u8], j: usize) -> (f32, f32) {
    let (d, m) = if j < 4 { (scales[j] & 63, scales[j + 4] & 63) }
        else { ((scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4), (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)) };
    (d as f32, m as f32)
}
// Dequant a single element at flat row-major index idx of Q4K-packed `data`.
fn dequant_elem(data: &[u8], idx: usize) -> f32 {
    let sb = idx / 256; let local = idx % 256; let s = sb * 144;
    let d = f16_to_f32((data[s] as u16) | ((data[s+1] as u16) << 8));
    let dmin = f16_to_f32((data[s+2] as u16) | ((data[s+3] as u16) << 8));
    let g = local / 64; let w = local % 64; let bic = w % 32;
    let qb = data[s + 16 + g*32 + bic];
    let is_sub = if w < 32 { g*2 } else { g*2 + 1 };
    let nib = if w < 32 { qb & 0x0F } else { qb >> 4 };
    let (sc, mn) = extract_scale_min(&data[s+4..s+16], is_sub);
    d * sc * (nib as f32) - dmin * mn
}

#[cuda_module]
mod kernels {
    use super::*;
    // y[row] = sum_k dequant(W_q4k)[row*K + k] * x[k]   (K = x.len(), row-major Q4K weights)
    #[kernel]
    pub fn q4k_matvec(data: &[u8], x: &[f32], mut y: DisjointSlice<f32>) {
        let r = thread::index_1d();
        let row = r.get();
        if let Some(yo) = y.get_mut(r) {
            let k = x.len();
            let base = row * k;
            let mut acc = 0.0f32;
            for j in 0..k { acc += dequant_elem(data, base + j) * x[j]; }
            *yo = acc;
        }
    }
}
fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    const M: usize = 4096; const K: usize = 2048;       // K multiple of 256
    let n_sb = (M * K) / 256;
    let mut data = vec![0u8; n_sb * 144];
    for sb in 0..n_sb {
        let s = sb * 144;
        let d = half::f16::from_f32(0.0123).to_bits();
        let dmin = half::f16::from_f32(0.0061).to_bits();
        data[s]=(d&0xFF) as u8; data[s+1]=(d>>8) as u8; data[s+2]=(dmin&0xFF) as u8; data[s+3]=(dmin>>8) as u8;
        for c in 0..12 { data[s+4+c] = ((sb*7 + c*13) % 256) as u8; }
        for c in 0..128 { data[s+16+c] = ((sb*3 + c*5) % 256) as u8; }
    }
    let x_host: Vec<f32> = (0..K).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
    let d_dev = DeviceBuffer::from_host(&stream, &data).expect("upload Q4K weights to device");
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).expect("upload x vector to device");
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, M).expect("allocate y output buffer on device");
    let module = kernels::load(&ctx).expect("load");
    module.q4k_matvec(&stream, LaunchConfig::for_num_elems(M as u32), &d_dev, &x_dev, &mut y_dev).expect("launch");
    let y = y_dev.to_host_vec(&stream).expect("copy y result to host");
    // ---- timed throughput (after correctness) ----
    let iters = 3000u32;
    for _ in 0..50 { module.q4k_matvec(&stream, LaunchConfig::for_num_elems(M as u32), &d_dev, &x_dev, &mut y_dev).expect("warm"); }
    let _ = y_dev.to_host_vec(&stream).expect("sync after warm-up");
    let t0 = std::time::Instant::now();
    for _ in 0..iters { module.q4k_matvec(&stream, LaunchConfig::for_num_elems(M as u32), &d_dev, &x_dev, &mut y_dev).expect("timed"); }
    let _ = y_dev.to_host_vec(&stream).expect("sync after timed loop");
    let us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);
    println!("Q4K-MATVEC TIMING: {:.2} us/launch ({}x{} Q4K matvec) on GB10 via cuda-oxide", us, M, K);
    let mut errs = 0; let mut maxdiff = 0.0f32;
    for row in 0..M {
        let mut acc = 0.0f32;
        for j in 0..K { acc += dequant_elem(&data, row*K + j) * x_host[j]; }
        let diff = (y[row] - acc).abs();
        if diff > maxdiff { maxdiff = diff; }
        if diff > 1e-2 { errs += 1; }
    }
    if errs == 0 { println!("Q4K-MATVEC PASSED: all {} rows match reference (M={} K={}, maxdiff={:.2e})", M, M, K, maxdiff); }
    else { eprintln!("Q4K-MATVEC FAILED: {}/{} rows (maxdiff={:.2e})", errs, M, maxdiff); std::process::exit(1); }
}
