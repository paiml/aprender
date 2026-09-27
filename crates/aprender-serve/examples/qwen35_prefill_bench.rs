//! #4376: time the batched Qwen3.5 CUDA prefill alone, llama-bench `pp` style.
//!
//! One warm-up prefill, then `reps` timed prefills of the same `n` deterministic ids
//! from a fresh state each time. Prints each rep and the median as prefill tok/s, the
//! number compared against `llama-bench -p <n> -n 0` on the same GGUF.
//!
//! Usage:
//!   cargo run --release --features cuda -p aprender-serve --example qwen35_prefill_bench -- \
//!       <model.gguf> [n_tokens=1006] [reps=3]
//!
//! Run it under the GPU lock (`flock /tmp/apr-gpu.lock …`).

use realizar::cuda::CudaExecutor;
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use realizar::gguf::Qwen35CudaModel;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: qwen35_prefill_bench <model.gguf> [n_tokens=1006] [reps=3]");
        std::process::exit(2);
    };
    let n: usize = args.get(2).map_or(1006, |s| s.parse().expect("n_tokens"));
    let reps: usize = args.get(3).map_or(3, |s| s.parse().expect("reps"));

    let mapped = MappedGGUFModel::from_path(path).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let vocab = base.config().vocab_size as u32;

    let mut s = 0x4376_0001u32;
    let ids: Vec<u32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) % vocab
        })
        .collect();

    let executor = CudaExecutor::new(0).expect("CUDA");
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, n + 1).expect("gpu model");

    let mut state = gpu.new_state().expect("state");
    gpu.prefill(&ids, &mut state, 0).expect("warm-up prefill");

    let mut ms = Vec::with_capacity(reps);
    for r in 0..reps {
        let mut state = gpu.new_state().expect("state");
        let t = Instant::now();
        let logits = gpu.prefill(&ids, &mut state, 0).expect("prefill");
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(logits.len(), vocab as usize);
        println!("rep {r}: {dt:.1} ms  {:.0} tok/s", n as f64 / dt * 1000.0);
        ms.push(dt);
    }
    ms.sort_by(f64::total_cmp);
    let med = ms[ms.len() / 2];
    println!(
        "pp{n}: median {med:.1} ms  {:.0} tok/s  (n={reps})",
        n as f64 / med * 1000.0
    );
}
