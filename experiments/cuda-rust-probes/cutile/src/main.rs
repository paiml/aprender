//! Fleet probe: does cutile-rs JIT a Tile kernel through CUDA Tile IR and run it on THIS GPU?
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;
    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = cuda_core::CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);
    let (z, _x, _y) = kernel::add(z, x, y).sync()?;
    let out: Vec<f32> = z.unpartition().to_host_vec().sync()?;
    let bad = out.iter().filter(|v| (**v - 2.0).abs() > 1e-5).count();
    println!("cutile: sm_{major}{minor} len={} mismatches={bad}", out.len());
    println!("{}", if bad == 0 { "VERDICT: PASS - cutile JIT-compiled and ran" } else { "VERDICT: FAIL" });
    if bad != 0 { std::process::exit(1); }
    Ok(())
}
