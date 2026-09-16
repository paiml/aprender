//! PMAT-3091 scalar kernel isolation (uncommitted example, copied to evidence scalar/). Batch mode:
//!   scalar_isolate <jobs.tsv>   each line: matmul <weight.bin> <gguf qtype> <in_dim> <out_dim> <in.f32> <out.f32>
//! runs `emulated_matvec_for(scalar = true, ..)` (the SCALAR ggml build's quantized matmul) on ONE activation vector
//! read from a dump file, and writes the output. Lets either engine's own dumped input be pushed through apr's kernel.
use realizar::quantize::ggml_vecdot_emul::emulated_matvec_for;
use std::io::Write;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn read_f32(path: &str) -> Res<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_f32(path: &str, v: &[f32]) -> Res<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}


fn fb(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// `MADD128(x, y, z)` without `__FMA__` (`vec.h`): `_mm_add_ps(_mm_mul_ps(x, y), z)`.
fn madd(x: f32, y: f32, z: f32) -> f32 {
    x * y + z
}

/// `NMADD128(x, y, z)` without `__FMA__`: `_mm_sub_ps(z, _mm_mul_ps(x, y))`.
fn nmadd(x: f32, y: f32, z: f32) -> f32 {
    z - x * y
}

/// One lane of ggml's SSE2 `ggml_v_expf` (`ggml-cpu/vec.h`, the `__SSE2__` branch), literal. Lanes are independent:
/// the movemask early return picks `MADD128(j, k, k)`, the blend picks `MADD128(k, j, k)`; IEEE mul commutes.
fn ggml_v_expf_lane(x: f32) -> f32 {
    let r = fb(0x4b400000);
    let z = madd(x, fb(0x3fb8aa3b), r);
    let n = z - r;
    let b = nmadd(n, fb(0x35bfbe8e), nmadd(n, fb(0x3f317200), x));
    let e = z.to_bits() << 23;
    let k = fb(e.wrapping_add(1.0f32.to_bits()));
    let c = n.abs() > 126.0;
    let u = b * b;
    let j = madd(
        madd(madd(fb(0x3c072010), b, fb(0x3d2b9f17)), u, madd(fb(0x3e2aaf33), b, fb(0x3efffedb))),
        u,
        fb(0x3f7ffff6) * b,
    );
    if !c {
        return madd(j, k, k);
    }
    let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = fb(g.wrapping_add(0x7f00_0000));
    let s2 = fb(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        madd(s2, j, s2) * s1
    }
}

/// `ggml_v_silu` SSE2: `x / (1 + expf_v(0 - x))`.
fn ggml_v_silu_lane(x: f32) -> f32 {
    x / (1.0 + ggml_v_expf_lane(0.0 - x))
}

fn job(t: &[&str]) -> Res<()> {
    match t {
        ["matmul", w, q, i, o, inp, out] => {
            let (in_dim, out_dim): (usize, usize) = (i.parse()?, o.parse()?);
            let weight = std::fs::read(w)?;
            let x = read_f32(inp)?;
            let mut y = vec![0.0f32; out_dim];
            emulated_matvec_for(true, q.parse()?, &weight, &x, in_dim, out_dim, &mut y)?;
            write_f32(out, &y)
        },
        // pos-0 conv (zero state: sum = 0*w0 + 0*w1 + 0*w2 + x*w3, apr causal_conv1d and ggml ssm_conv order) then
        // silu two ways: apr's `x / (1 + (-x).exp())` and ggml's SSE2 `ggml_v_silu`.
        ["convsilu0", w, q, i, o, inp, cw, out_apr, out_ggml] => {
            let (in_dim, out_dim): (usize, usize) = (i.parse()?, o.parse()?);
            let weight = std::fs::read(w)?;
            let x = read_f32(inp)?;
            let conv_w = read_f32(cw)?;
            let mut qkv = vec![0.0f32; out_dim];
            emulated_matvec_for(true, q.parse()?, &weight, &x, in_dim, out_dim, &mut qkv)?;
            let raw: Vec<f32> = qkv
                .iter()
                .enumerate()
                .map(|(c, &v)| {
                    let mut sum = 0.0f32;
                    for k in 0..3 {
                        sum += 0.0 * conv_w[c * 4 + k];
                    }
                    sum + v * conv_w[c * 4 + 3]
                })
                .collect();
            let apr: Vec<f32> = raw.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
            let ggml: Vec<f32> = raw.iter().map(|&v| ggml_v_silu_lane(v)).collect();
            write_f32(out_apr, &apr)?;
            write_f32(out_ggml, &ggml)
        },
        other => Err(format!("bad job {other:?}").into()),
    }
}

fn main() -> Res<()> {
    let path = std::env::args().nth(1).ok_or("usage: scalar_isolate <jobs.tsv>")?;
    let mut n = 0usize;
    for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.is_empty()) {
        job(&line.split('\t').collect::<Vec<_>>())?;
        n += 1;
    }
    println!("scalar_isolate: jobs={n}");
    Ok(())
}
