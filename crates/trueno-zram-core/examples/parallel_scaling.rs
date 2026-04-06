//! Test parallel scaling to find the bottleneck

use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};
use rayon::prelude::*;
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::time::Instant;
use trueno_zram_core::lz4::{compress, decompress_simd};

const PAGE_SIZE: usize = 4096;
const NUM_PAGES: usize = 100_000;

fn main() {
    println!("Parallel scaling test for LZ4 decompression\n");

    // Create test data
    let pages: Vec<[u8; PAGE_SIZE]> = (0..NUM_PAGES)
        .map(|i| {
            let mut page = [0u8; PAGE_SIZE];
            match i % 4 {
                0 => {}
                1 => {
                    for (j, b) in page.iter_mut().enumerate() {
                        *b = (j % 256) as u8;
                    }
                }
                _ => {
                    let mut rng = (i as u64).wrapping_mul(0x0005_DEEC_E66D);
                    for b in &mut page {
                        rng = rng.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(0xB);
                        *b = (rng >> 33) as u8;
                    }
                }
            }
            page
        })
        .collect();

    let compressed: Vec<Vec<u8>> = pages.iter().map(|p| compress(p).unwrap()).collect();

    println!("Testing with different thread counts...\n");

    for num_threads in [1, 2, 4, 8, 12, 16, 24, 32, 48] {
        // Configure rayon thread pool for this test
        let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        pool.install(|| {
            // Pre-allocate output
            let mut outputs: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; NUM_PAGES];

            // Warmup
            outputs.par_iter_mut().zip(compressed.par_iter()).take(1000).for_each(|(out, data)| {
                let _ = decompress_simd(data, out);
            });

            // Time it
            let start = Instant::now();
            outputs.par_iter_mut().zip(compressed.par_iter()).for_each(|(out, data)| {
                decompress_simd(data, out).unwrap();
            });
            let elapsed = start.elapsed();

            let bytes = NUM_PAGES * PAGE_SIZE;
            let gbps = bytes as f64 / elapsed.as_secs_f64() / 1e9;
            let scaling_efficiency = gbps / (2.5 * num_threads as f64) * 100.0;

            println!(
                "{:2} threads: {:>6.1} ms = {:>5.2} GB/s (efficiency: {:>4.1}%)",
                num_threads,
                elapsed.as_secs_f64() * 1000.0,
                gbps,
                scaling_efficiency
            );
        });
    }

    // Also test memory bandwidth with simple memcpy
    println!("\nMemory bandwidth test (parallel memcpy)...\n");

    let src: Vec<u8> = vec![0xAB; NUM_PAGES * PAGE_SIZE];
    let mut dst: Vec<u8> = vec![0; NUM_PAGES * PAGE_SIZE];

    let start = Instant::now();
    dst.par_chunks_mut(PAGE_SIZE).zip(src.par_chunks(PAGE_SIZE)).for_each(|(d, s)| {
        d.copy_from_slice(s);
    });
    let elapsed = start.elapsed();
    let gbps = (NUM_PAGES * PAGE_SIZE) as f64 / elapsed.as_secs_f64() / 1e9;
    println!("Parallel memcpy: {gbps:.2} GB/s (this is ~memory bandwidth limit)");
}
