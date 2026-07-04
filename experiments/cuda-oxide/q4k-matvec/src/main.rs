use cuda_device::{kernel, thread};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

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
const T: usize = 32;   // threads per row  (TUNE_T)
#[cuda_module]
mod kernels {
    use super::*;
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
}
fn main() {
    let m: usize = std::env::var("GXM").ok().and_then(|v|v.parse().ok()).unwrap_or(4096);
    let k: usize = std::env::var("GXK").ok().and_then(|v|v.parse().ok()).unwrap_or(2048);
    assert!(k % 256 == 0, "K must be multiple of 256");
    let ctx = CudaContext::new(0).expect("ctx"); let stream = ctx.default_stream();
    let n_sb=(m*k)/256; let mut data=vec![0u8; n_sb*144];
    for sb in 0..n_sb { let s=sb*144;
        let d=half::f16::from_f32(0.0123).to_bits(); let dmin=half::f16::from_f32(0.0061).to_bits();
        data[s]=(d&0xFF)as u8; data[s+1]=(d>>8)as u8; data[s+2]=(dmin&0xFF)as u8; data[s+3]=(dmin>>8)as u8;
        for c in 0..12 {data[s+4+c]=((sb*7+c*13)%256)as u8;}
        for c in 0..128 {data[s+16+c]=((sb*3+c*5)%256)as u8;}
    }
    let x_host:Vec<f32>=(0..k).map(|i|((i%11)as f32)*0.1-0.5).collect();
    let d_dev=DeviceBuffer::from_host(&stream,&data).expect("upload Q4K weight data to device");
    let x_dev=DeviceBuffer::from_host(&stream,&x_host).expect("upload x vector to device");
    let y_dev=DeviceBuffer::<f32>::zeroed(&stream,m).expect("allocate y output buffer on device");
    let module=kernels::load(&ctx).expect("load");
    let total=(m*T) as u32;
    module.q4k_matvec_atomic(&stream,LaunchConfig::for_num_elems(total),&d_dev,&x_dev,&y_dev).expect("launch");
    let y=y_dev.to_host_vec(&stream).expect("copy y result to host for verification");
    let mut errs=0; let mut maxrel=0.0f32;
    let check_rows = m.min(512);
    for row in 0..check_rows { let mut acc=0.0f32; for j in 0..k {acc+=dequant_elem(&data,row*k+j)*x_host[j];}
        let rel=(y[row]-acc).abs()/(acc.abs()+1e-3); if rel>maxrel{maxrel=rel;} if rel>1e-2 {errs+=1;} }
    if errs!=0 { eprintln!("Q4K-MATVEC-ATOMIC FAILED: {} rows (maxrel={:.2e})", errs, maxrel); std::process::exit(1); }
    // warmup
    for _ in 0..20 { let yz=DeviceBuffer::<f32>::zeroed(&stream,m).expect("allocate warmup y buffer"); module.q4k_matvec_atomic(&stream,LaunchConfig::for_num_elems(total),&d_dev,&x_dev,&yz).expect("w"); let _=yz.to_host_vec(&stream).expect("copy warmup y to host"); }
    // timed: N bare launches, one final sync (median over reps)
    let iters=200u32; let reps=5;
    let mut times=Vec::new();
    for _ in 0..reps {
        let t0=std::time::Instant::now();
        for _ in 0..iters { module.q4k_matvec_atomic(&stream,LaunchConfig::for_num_elems(total),&d_dev,&x_dev,&y_dev).expect("t"); }
        let _=y_dev.to_host_vec(&stream).expect("copy timed y result to host");
        times.push(t0.elapsed().as_secs_f64()*1e6/(iters as f64));
    }
    times.sort_by(|a,b|a.partial_cmp(b).expect("launch times are finite (no NaN)"));
    let med=times[times.len()/2];
    println!("OXIDE M={} K={} T={} median={:.2} us/launch (reps={:?} maxrel={:.1e})", m, k, T, med, times.iter().map(|t|(t*100.0).round()/100.0).collect::<Vec<_>>(), maxrel);
}
