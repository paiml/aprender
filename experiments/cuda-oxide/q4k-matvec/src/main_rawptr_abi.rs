use cuda_device::{kernel, thread};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// ---------------------------------------------------------------------------
// Q4_K dequant helpers (host + device shared). Identical math to the slice-ABI
// spike at /tmp/q4kopt_spike/src/main.rs.
// ---------------------------------------------------------------------------
fn f16_to_f32(h: u16) -> f32 {
    let s=((h>>15)&1) as u32; let e=((h>>10)&0x1F) as u32; let m=(h&0x3FF) as u32;
    let b: u32 = if e==0 { if m==0 {s<<31} else { let mut ee:i32=-1; let mut mm=m; loop{ee+=1; mm<<=1; if (mm&0x400)!=0 {break;}} (s<<31)|(((127-15-ee) as u32)<<23)|((mm&0x3FF)<<13) } }
        else if e==0x1F { (s<<31)|(0xFF<<23)|(m<<13) } else { (s<<31)|((e+112)<<23)|(m<<13) };
    f32::from_bits(b)
}
fn extract_scale_min(sc:&[u8], j:usize)->(f32,f32){ let(d,m)= if j<4 {(sc[j]&63, sc[j+4]&63)} else {((sc[j+4]&0x0F)|((sc[j-4]>>6)<<4),(sc[j+4]>>4)|((sc[j]>>6)<<4))}; (d as f32, m as f32) }
fn dequant_elem(data:&[u8], idx:usize)->f32{
    let sb=idx/256; let local=idx%256; let s=sb*144;
    let d=f16_to_f32((data[s] as u16)|((data[s+1] as u16)<<8));
    let dmin=f16_to_f32((data[s+2] as u16)|((data[s+3] as u16)<<8));
    let g=local/64; let w=local%64; let bic=w%32;
    let qb=data[s+16+g*32+bic];
    let issub= if w<32 {g*2} else {g*2+1};
    let nib= if w<32 {qb&0x0F} else {qb>>4};
    let(scv,mn)=extract_scale_min(&data[s+4..s+16], issub);
    d*scv*(nib as f32)-dmin*mn
}

// Raw-pointer variant of dequant_elem: indexes a `*const u8` directly so the
// device kernel can use the C-style ABI without a fat slice pointer.
#[inline(always)]
unsafe fn extract_scale_min_ptr(sc:*const u8, j:usize)->(f32,f32){
    let g = |o:usize| unsafe { *sc.add(o) };
    let(d,m)= if j<4 {(g(j)&63, g(j+4)&63)}
        else {((g(j+4)&0x0F)|((g(j-4)>>6)<<4),(g(j+4)>>4)|((g(j)>>6)<<4))};
    (d as f32, m as f32)
}
#[inline(always)]
unsafe fn dequant_elem_ptr(data:*const u8, idx:usize)->f32{
    let sb=idx/256; let local=idx%256; let s=sb*144;
    let rd = |o:usize| unsafe { *data.add(o) };
    let d=f16_to_f32((rd(s) as u16)|((rd(s+1) as u16)<<8));
    let dmin=f16_to_f32((rd(s+2) as u16)|((rd(s+3) as u16)<<8));
    let gg=local/64; let w=local%64; let bic=w%32;
    let qb=rd(s+16+gg*32+bic);
    let issub= if w<32 {gg*2} else {gg*2+1};
    let nib= if w<32 {qb&0x0F} else {qb>>4};
    let(scv,mn)= unsafe { extract_scale_min_ptr(data.add(s+4), issub) };
    d*scv*(nib as f32)-dmin*mn
}

const T: usize = 32;   // threads per row  (TUNE_T)

#[cuda_module]
mod kernels {
    use super::*;

    // ---- Reference slice-ABI kernel (the existing spike's exact kernel) ----
    #[kernel]
    pub fn q4k_matvec_atomic(data:&[u8], x:&[f32], y:&[f32]) {
        let gid = thread::index_1d().get();
        let k = x.len();
        let m = y.len();
        let row = gid / T;
        if row >= m { return; }
        let lane = gid % T;
        let base = row * k;
        let mut acc = 0.0f32;
        let mut j = lane;
        while j < k { acc += dequant_elem(data, base + j) * x[j]; j += T; }
        let ay = unsafe { &*(y.as_ptr().add(row) as *const DeviceAtomicF32) };
        ay.fetch_add(acc, AtomicOrdering::Relaxed);
    }

    // ---- NEW: raw-pointer C-style ABI kernel ----
    // Param order: data(ptr), x(ptr), y(ptr,mut), m(u32), k(u32), t(u32)
    // Same compute: strided partial per lane + atomic-add to y[row].
    #[kernel]
    pub fn q4k_matvec(
        data: *const u8,
        x: *const f32,
        y: *mut f32,
        m: u32,
        k: u32,
        t: u32,
    ) {
        let gid = thread::index_1d().get();
        let tt = t as usize;
        let kk = k as usize;
        let mm = m as usize;
        let row = (gid as usize) / tt;
        if row >= mm { return; }
        let lane = (gid as usize) % tt;
        let base = row * kk;
        let mut acc = 0.0f32;
        let mut j = lane;
        while j < kk {
            let w = unsafe { dequant_elem_ptr(data, base + j) };
            let xv = unsafe { *x.add(j) };
            acc += w * xv;
            j += tt;
        }
        let ay = unsafe { &*(y.add(row) as *const DeviceAtomicF32) };
        ay.fetch_add(acc, AtomicOrdering::Relaxed);
    }
}

fn make_data(m: usize, k: usize) -> Vec<u8> {
    let n_sb=(m*k)/256; let mut data=vec![0u8; n_sb*144];
    for sb in 0..n_sb { let s=sb*144;
        let d=half::f16::from_f32(0.0123).to_bits(); let dmin=half::f16::from_f32(0.0061).to_bits();
        data[s]=(d&0xFF)as u8; data[s+1]=(d>>8)as u8; data[s+2]=(dmin&0xFF)as u8; data[s+3]=(dmin>>8)as u8;
        for c in 0..12 {data[s+4+c]=((sb*7+c*13)%256)as u8;}
        for c in 0..128 {data[s+16+c]=((sb*3+c*5)%256)as u8;}
    }
    data
}

fn main() {
    let m: usize = std::env::var("GXM").ok().and_then(|v|v.parse().ok()).unwrap_or(4096);
    let k: usize = std::env::var("GXK").ok().and_then(|v|v.parse().ok()).unwrap_or(2048);
    assert!(k % 256 == 0, "K must be multiple of 256");
    let ctx = CudaContext::new(0).expect("ctx"); let stream = ctx.default_stream();
    let data = make_data(m, k);
    let x_host:Vec<f32>=(0..k).map(|i|((i%11)as f32)*0.1-0.5).collect();
    let d_dev=DeviceBuffer::from_host(&stream,&data).expect("upload Q4K weight data to device");
    let x_dev=DeviceBuffer::from_host(&stream,&x_host).expect("upload x vector to device");
    let module=kernels::load(&ctx).expect("load");
    let total=(m*T) as u32;

    // ---- CPU reference ----
    let check_rows = m.min(512);
    let mut cpu_ref = vec![0.0f32; m];
    for row in 0..m { let mut acc=0.0f32; for j in 0..k {acc+=dequant_elem(&data,row*k+j)*x_host[j];} cpu_ref[row]=acc; }

    // ---- Run slice-ABI reference kernel ----
    let y_slice_dev=DeviceBuffer::<f32>::zeroed(&stream,m).expect("allocate slice-ABI y buffer");
    module.q4k_matvec_atomic(&stream,LaunchConfig::for_num_elems(total),&d_dev,&x_dev,&y_slice_dev).expect("launch slice");
    let y_slice=y_slice_dev.to_host_vec(&stream).expect("copy slice-ABI y result to host");

    // ---- Run NEW raw-ptr kernel ----
    let y_raw_dev=DeviceBuffer::<f32>::zeroed(&stream,m).expect("allocate raw-ptr y buffer");
    module.q4k_matvec(
        &stream,
        LaunchConfig::for_num_elems(total),
        d_dev.cu_deviceptr() as *const u8,
        x_dev.cu_deviceptr() as *const f32,
        y_raw_dev.cu_deviceptr() as *mut f32,
        m as u32,
        k as u32,
        T as u32,
    ).expect("launch raw");
    let y_raw=y_raw_dev.to_host_vec(&stream).expect("copy raw-ptr y result to host");

    // ---- Bit-exact: raw vs slice (must be IDENTICAL bit-for-bit) ----
    let mut bitexact_fail = 0usize;
    let mut first_diff: Option<(usize,f32,f32)> = None;
    for row in 0..m {
        if y_raw[row].to_bits() != y_slice[row].to_bits() {
            bitexact_fail += 1;
            if first_diff.is_none() { first_diff = Some((row, y_raw[row], y_slice[row])); }
        }
    }

    // ---- Correctness vs CPU ref ----
    let mut errs=0; let mut maxrel=0.0f32;
    for row in 0..check_rows {
        let acc = cpu_ref[row];
        let rel=(y_raw[row]-acc).abs()/(acc.abs()+1e-3); if rel>maxrel{maxrel=rel;} if rel>1e-2 {errs+=1;}
    }

    println!("== q4k_matvec raw-ptr ABI verification ==");
    println!("M={} K={} T={}", m, k, T);
    println!("bit-exact (raw vs slice): {} (mismatches={})", if bitexact_fail==0 {"PASS"} else {"FAIL"}, bitexact_fail);
    if let Some((r,a,b)) = first_diff { println!("  first diff row {}: raw={:.6} slice={:.6}", r, a, b); }
    println!("correctness (raw vs CPU): {} (errs={} maxrel={:.2e})", if errs==0 {"PASS"} else {"FAIL"}, errs, maxrel);
    if bitexact_fail!=0 || errs!=0 { eprintln!("VERIFICATION FAILED"); std::process::exit(1); }

    // ---- Timing: raw-ptr kernel ----
    for _ in 0..20 {
        let yz=DeviceBuffer::<f32>::zeroed(&stream,m).expect("allocate warmup y buffer");
        module.q4k_matvec(&stream,LaunchConfig::for_num_elems(total),
            d_dev.cu_deviceptr() as *const u8, x_dev.cu_deviceptr() as *const f32,
            yz.cu_deviceptr() as *mut f32, m as u32, k as u32, T as u32).expect("w");
        let _=yz.to_host_vec(&stream).expect("copy warmup y to host");
    }
    let iters=200u32; let reps=5;
    let mut times=Vec::new();
    for _ in 0..reps {
        let t0=std::time::Instant::now();
        for _ in 0..iters {
            module.q4k_matvec(&stream,LaunchConfig::for_num_elems(total),
                d_dev.cu_deviceptr() as *const u8, x_dev.cu_deviceptr() as *const f32,
                y_raw_dev.cu_deviceptr() as *mut f32, m as u32, k as u32, T as u32).expect("t");
        }
        let _=y_raw_dev.to_host_vec(&stream).expect("copy timed raw-ptr y result to host");
        times.push(t0.elapsed().as_secs_f64()*1e6/(iters as f64));
    }
    times.sort_by(|a,b|a.partial_cmp(b).expect("launch times are finite (no NaN)"));
    let med=times[times.len()/2];
    println!("OXIDE-RAWPTR M={} K={} T={} median={:.2} us/launch (reps={:?})", m, k, T, med,
        times.iter().map(|t|(t*100.0).round()/100.0).collect::<Vec<_>>());
}
