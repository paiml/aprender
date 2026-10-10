//! LT-1 engine probe (APR-LAYA-TRAIN-001): times one "proxy layer" of an
//! encoder forward + backward on ONE autograd engine per process.
//!
//! The proxy layer uses only operations that have a backward on BOTH engines:
//! 2-D matmul, elementwise mul, gelu (tanh form), add and sum-to-scalar.
//! Softmax, slicing, layer norm and N-D attention are deliberately absent:
//! they are not available with a backward on both engines.
//!
//! The loss sums only the TERMINAL products (S = Q·Kᵀ feeds O = S·V, so S is
//! not summed separately). `entrenar::autograd` back-propagates recursively
//! per consumer, so a tensor with two consumers would be traversed twice and
//! its grads double-counted; terminal-only keeps both engines on the same,
//! correct graph.
//!
//! Usage:
//!   laya_engine_probe --engine core|train [--hidden 1024] [--intermediate 2624]
//!     [--heads 16] [--batch 8] [--seq 128] [--config encoder/config.json]
//!     [--reps 3] [--layers 28] [--steps 0]
//!
//! Prints ONE JSON object on stdout. Exit code 1 when the gradient check fails.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    Core,
    Train,
}

impl Engine {
    fn name(self) -> &'static str {
        match self {
            Engine::Core => "core",
            Engine::Train => "train",
        }
    }
}

struct Args {
    engine: Engine,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    batch: usize,
    seq: usize,
    source: String,
    reps: usize,
    layers: usize,
    steps: usize,
}

fn parse_usize(flag: &str, v: Option<String>) -> Result<usize, String> {
    let v = v.ok_or_else(|| format!("{flag} needs a value"))?;
    v.parse::<usize>().map_err(|e| format!("{flag}: bad value {v:?}: {e}"))
}

fn json_usize(cfg: &serde_json::Value, key: &str) -> Result<usize, String> {
    cfg.get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("config: missing or non-integer {key}"))
}

/// Flags as given, before defaults and the config file are applied.
#[derive(Default)]
struct Raw {
    engine: Option<Engine>,
    hidden: Option<usize>,
    intermediate: Option<usize>,
    heads: Option<usize>,
    batch: Option<usize>,
    seq: Option<usize>,
    reps: Option<usize>,
    layers: Option<usize>,
    steps: Option<usize>,
    config: Option<String>,
}

fn parse_engine(v: Option<String>) -> Result<Engine, String> {
    match v.as_deref() {
        Some("core") => Ok(Engine::Core),
        Some("train") => Ok(Engine::Train),
        other => Err(format!("--engine must be core|train, got {other:?}")),
    }
}

/// The `Option<usize>` slot a numeric flag fills, or `None` for a non-numeric flag.
fn numeric_slot<'a>(raw: &'a mut Raw, flag: &str) -> Option<&'a mut Option<usize>> {
    match flag {
        "--hidden" => Some(&mut raw.hidden),
        "--intermediate" => Some(&mut raw.intermediate),
        "--heads" => Some(&mut raw.heads),
        "--batch" => Some(&mut raw.batch),
        "--seq" => Some(&mut raw.seq),
        "--reps" => Some(&mut raw.reps),
        "--layers" => Some(&mut raw.layers),
        "--steps" => Some(&mut raw.steps),
        _ => None,
    }
}

fn parse_raw(mut it: impl Iterator<Item = String>) -> Result<Raw, String> {
    let mut raw = Raw::default();
    while let Some(flag) = it.next() {
        if let Some(slot) = numeric_slot(&mut raw, &flag) {
            *slot = Some(parse_usize(&flag, it.next())?);
            continue;
        }
        match flag.as_str() {
            "--engine" => raw.engine = Some(parse_engine(it.next())?),
            "--config" => raw.config = Some(it.next().ok_or("--config needs a path")?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(raw)
}

/// Layer shape and where it came from: the config file wins, then flags, then defaults.
fn resolve_shape(raw: &Raw) -> Result<(usize, usize, usize, String), String> {
    if let Some(path) = &raw.config {
        let text = std::fs::read_to_string(path).map_err(|e| format!("config {path}: {e}"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("config {path}: {e}"))?;
        return Ok((
            json_usize(&cfg, "hidden_size")?,
            json_usize(&cfg, "intermediate_size")?,
            json_usize(&cfg, "num_attention_heads")?,
            format!("config:{path}"),
        ));
    }
    let any_flag = raw.hidden.is_some() || raw.intermediate.is_some() || raw.heads.is_some();
    let source = if any_flag { "flags" } else { "defaults" };
    Ok((
        raw.hidden.unwrap_or(1024),
        raw.intermediate.unwrap_or(2624),
        raw.heads.unwrap_or(16),
        source.to_string(),
    ))
}

fn parse_args() -> Result<Args, String> {
    let raw = parse_raw(std::env::args().skip(1))?;
    let engine = raw.engine.ok_or("--engine core|train is required")?;
    let (hidden, intermediate, heads, source) = resolve_shape(&raw)?;
    let a = Args {
        engine,
        hidden,
        intermediate,
        heads,
        batch: raw.batch.unwrap_or(8),
        seq: raw.seq.unwrap_or(128),
        source,
        reps: raw.reps.unwrap_or(3),
        layers: raw.layers.unwrap_or(28),
        steps: raw.steps.unwrap_or(0),
    };
    if a.heads == 0 || a.hidden % a.heads != 0 {
        return Err(format!("hidden {} not divisible by heads {}", a.hidden, a.heads));
    }
    let any_zero = [a.reps, a.batch, a.seq, a.intermediate].contains(&0);
    if any_zero {
        return Err("reps, batch, seq, intermediate must be > 0".to_string());
    }
    Ok(a)
}

/// Pre-generated random data for one proxy layer (seed 13).
struct Data {
    n: usize,
    d: usize,
    dh: usize,
    i: usize,
    seq: usize,
    x: Vec<f32>,
    wqkv: Vec<f32>,
    /// Per (batch, head) pair: Q [seq,dh], Kᵀ [dh,seq], V [seq,dh].
    qkv: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    a: Vec<f32>,
    wo: Vec<f32>,
    h: Vec<f32>,
    wi: Vec<f32>,
    g: Vec<f32>,
    g2: Vec<f32>,
    wo2: Vec<f32>,
}

fn rand_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.random_range(-0.05f32..0.05f32)).collect()
}

fn make_data(a: &Args) -> Data {
    let mut rng = StdRng::seed_from_u64(13);
    let n = a.batch * a.seq;
    let d = a.hidden;
    let dh = d / a.heads;
    let i = a.intermediate;
    let seq = a.seq;
    let x = rand_vec(&mut rng, n * d);
    let wqkv = rand_vec(&mut rng, d * 3 * d);
    let qkv = (0..a.batch * a.heads)
        .map(|_| {
            (
                rand_vec(&mut rng, seq * dh),
                rand_vec(&mut rng, dh * seq),
                rand_vec(&mut rng, seq * dh),
            )
        })
        .collect();
    Data {
        n,
        d,
        dh,
        i,
        seq,
        x,
        wqkv,
        qkv,
        a: rand_vec(&mut rng, n * d),
        wo: rand_vec(&mut rng, d * d),
        h: rand_vec(&mut rng, n * d),
        wi: rand_vec(&mut rng, d * 2 * i),
        g: rand_vec(&mut rng, n * i),
        g2: rand_vec(&mut rng, n * i),
        wo2: rand_vec(&mut rng, i * d),
    }
}

fn grad_ok(g: &[f32]) -> bool {
    !g.is_empty() && g.iter().all(|v| v.is_finite()) && g.iter().any(|v| *v != 0.0)
}

/// One rep: (forward seconds, backward seconds, grad check).
fn rep_core(dt: &Data) -> (f64, f64, bool) {
    use aprender::autograd::{clear_graph, get_grad, Tensor};
    clear_graph();
    let leaf = |v: &Vec<f32>, shape: &[usize]| Tensor::from_vec(v.clone(), shape);
    let param = |v: &Vec<f32>, shape: &[usize]| Tensor::from_vec(v.clone(), shape).requires_grad();
    let (n, d, dh, i, seq) = (dt.n, dt.d, dt.dh, dt.i, dt.seq);
    // Tensor construction (memcpy of the data) is outside the timed region.
    let x = leaf(&dt.x, &[n, d]);
    let wqkv = param(&dt.wqkv, &[d, 3 * d]);
    let heads: Vec<(Tensor, Tensor, Tensor)> = dt
        .qkv
        .iter()
        .map(|(q, k, v)| (param(q, &[seq, dh]), param(k, &[dh, seq]), param(v, &[seq, dh])))
        .collect();
    let a = param(&dt.a, &[n, d]);
    let wo = param(&dt.wo, &[d, d]);
    let h = param(&dt.h, &[n, d]);
    let wi = param(&dt.wi, &[d, 2 * i]);
    let g = param(&dt.g, &[n, i]);
    let g2 = param(&dt.g2, &[n, i]);
    let wo2 = param(&dt.wo2, &[i, d]);

    let t0 = Instant::now();
    let mut loss = x.matmul(&wqkv).sum();
    for (q, kt, v) in &heads {
        let s = q.matmul(kt);
        let o = s.matmul(v);
        loss = loss.add(&o.sum());
    }
    loss = loss.add(&a.matmul(&wo).sum());
    loss = loss.add(&h.matmul(&wi).sum());
    let m = g.gelu().mul(&g2);
    loss = loss.add(&m.matmul(&wo2).sum());
    let fwd = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    loss.backward();
    let bwd = t1.elapsed().as_secs_f64();

    let ok = [wqkv.id(), wo.id(), wi.id(), wo2.id()]
        .iter()
        .all(|id| get_grad(*id).map(|t| grad_ok(t.data())).unwrap_or(false));
    clear_graph();
    (fwd, bwd, ok)
}

fn rep_train(dt: &Data) -> (f64, f64, bool) {
    use entrenar::autograd::{add, backward, gelu, matmul, mul, sum, Tensor};
    let leaf = |v: &Vec<f32>| Tensor::from_vec(v.clone(), false);
    let param = |v: &Vec<f32>| Tensor::from_vec(v.clone(), true);
    let (n, d, dh, i, seq) = (dt.n, dt.d, dt.dh, dt.i, dt.seq);
    let x = leaf(&dt.x);
    let wqkv = param(&dt.wqkv);
    let heads: Vec<(Tensor, Tensor, Tensor)> =
        dt.qkv.iter().map(|(q, k, v)| (param(q), param(k), param(v))).collect();
    let a = param(&dt.a);
    let wo = param(&dt.wo);
    let h = param(&dt.h);
    let wi = param(&dt.wi);
    let g = param(&dt.g);
    let g2 = param(&dt.g2);
    let wo2 = param(&dt.wo2);

    let t0 = Instant::now();
    let mut loss = sum(&matmul(&x, &wqkv, n, d, 3 * d));
    for (q, kt, v) in &heads {
        let s = matmul(q, kt, seq, dh, seq);
        let o = matmul(&s, v, seq, seq, dh);
        loss = add(&loss, &sum(&o));
    }
    loss = add(&loss, &sum(&matmul(&a, &wo, n, d, d)));
    loss = add(&loss, &sum(&matmul(&h, &wi, n, d, 2 * i)));
    let m = mul(&gelu(&g), &g2);
    loss = add(&loss, &sum(&matmul(&m, &wo2, n, i, d)));
    let fwd = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    backward(&mut loss, None);
    let bwd = t1.elapsed().as_secs_f64();

    let ok = [&wqkv, &wo, &wi, &wo2]
        .iter()
        .all(|t| t.grad().and_then(|g| g.as_slice().map(grad_ok)).unwrap_or(false));
    (fwd, bwd, ok)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let m = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[m - 1] + s[m]) / 2.0
    } else {
        s[m]
    }
}

fn peak_rss_kb() -> serde_json::Value {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .map_or_else(|| serde_json::json!("unmeasured"), |kb| serde_json::json!(kb))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("laya_engine_probe: {e}");
            std::process::exit(2);
        }
    };
    let data = make_data(&args);
    let run = |dt: &Data| match args.engine {
        Engine::Core => rep_core(dt),
        Engine::Train => rep_train(dt),
    };
    // Warmup (not reported).
    let (_, _, mut grad_check) = run(&data);
    let mut fwds = Vec::with_capacity(args.reps);
    let mut bwds = Vec::with_capacity(args.reps);
    for _ in 0..args.reps {
        let (f, b, ok) = run(&data);
        fwds.push(f);
        bwds.push(b);
        grad_check &= ok;
    }
    let fwd_median = median(&fwds);
    let bwd_median = median(&bwds);
    let totals: Vec<f64> = fwds.iter().zip(&bwds).map(|(f, b)| f + b).collect();
    let projected = median(&totals) * args.layers as f64 * args.steps as f64;
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    let host = std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let sha = std::env::var("APR_PROBE_SHA").unwrap_or_else(|_| "unset".to_string());
    let out = serde_json::json!({
        "engine": args.engine.name(),
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "heads": args.heads,
            "batch": args.batch,
            "seq": args.seq,
            "source": args.source,
        },
        "reps_fwd_s": fwds,
        "reps_bwd_s": bwds,
        "fwd_median_s": fwd_median,
        "bwd_median_s": bwd_median,
        "peak_rss_kb": peak_rss_kb(),
        "threads": threads,
        "host": host,
        "sha": sha,
        "layers": args.layers,
        "steps": args.steps,
        "projected_wall_s": projected,
        "grad_check": grad_check,
    });
    println!("{out}");
    if !grad_check {
        std::process::exit(1);
    }
}
