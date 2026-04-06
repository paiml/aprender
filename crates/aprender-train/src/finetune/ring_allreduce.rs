//! Ring AllReduce over TCP for distributed gradient averaging.
//!
//! Implements the bandwidth-optimal ring AllReduce algorithm:
//! 1. **Scatter-reduce**: N-1 rounds, each worker sends 1/N chunk to right neighbor
//! 2. **All-gather**: N-1 rounds, each worker broadcasts its reduced chunk
//!
//! Total data per worker: `2 * (N-1)/N * D * 4` bytes — matches theoretical lower bound.
//!
//! # Contract
//!
//! C-RING-001: Output equals arithmetic mean of all inputs across workers.
//! C-RING-001: All workers produce identical output.
//!
//! # Two-Worker Degeneration
//!
//! For N=2, this degenerates to a simple send-and-average:
//! - Each worker sends its full vector to the other
//! - Each worker averages its local vector with the received one
//!   This is correct but not bandwidth-optimal for N=2 (ring is equivalent).

use std::io::{Read, Write};
use std::net::TcpStream;

/// A participant in the ring AllReduce.
///
/// Each worker holds a TCP connection to its right neighbor (send)
/// and left neighbor (recv) in the ring topology.
pub struct RingAllReduceWorker {
    /// This worker's rank
    rank: usize,
    /// Total number of workers
    world_size: usize,
    /// TCP stream to right neighbor (rank + 1) % N — for sending
    send_stream: TcpStream,
    /// TCP stream from left neighbor (rank - 1) % N — for receiving
    recv_stream: TcpStream,
}

impl RingAllReduceWorker {
    /// Create a new ring worker with pre-established TCP connections.
    ///
    /// # Arguments
    /// * `rank` - This worker's rank in [0, world_size)
    /// * `world_size` - Total number of workers (N >= 2)
    /// * `send_stream` - TCP connection to worker (rank + 1) % N
    /// * `recv_stream` - TCP connection from worker (rank - 1) % N
    pub fn new(
        rank: usize,
        world_size: usize,
        send_stream: TcpStream,
        recv_stream: TcpStream,
    ) -> Self {
        assert!(world_size >= 2, "ring AllReduce requires >= 2 workers");
        assert!(rank < world_size, "rank must be < world_size");
        Self { rank, world_size, send_stream, recv_stream }
    }

    /// Perform AllReduce on `data`, returning the averaged result.
    ///
    /// After this call, `data` contains `(1/N) * sum(all workers' input)`.
    ///
    /// # Contract (C-RING-001)
    ///
    /// - Output is the arithmetic mean of all inputs
    /// - All workers produce identical output
    /// - Data length must be the same on all workers
    ///
    /// # Errors
    ///
    /// Returns `Err` on TCP I/O failure.
    pub fn allreduce(&mut self, data: &mut [f32]) -> Result<(), String> {
        contract_pre_gradient_allreduce!();
        let n = self.world_size;
        let d = data.len();

        // Compute chunk boundaries: chunks[i] = (start, len)
        let chunk_size = d / n;
        let remainder = d % n;
        let chunks: Vec<(usize, usize)> = (0..n)
            .map(|i| {
                let start = i * chunk_size + i.min(remainder);
                let len = chunk_size + usize::from(i < remainder);
                (start, len)
            })
            .collect();

        // Find the maximum chunk size for buffer allocation
        let max_chunk_len = chunks.iter().map(|(_, len)| *len).max().unwrap_or(0);
        let mut send_buf = vec![0u8; max_chunk_len * 4];
        let mut recv_buf = vec![0u8; max_chunk_len * 4];

        // ── Phase 1: Scatter-reduce ──────────────────────────────────────
        // After N-1 rounds, worker w holds sum of chunk w from all workers.
        for round in 0..(n - 1) {
            // Chunk index to send (we send the chunk we just reduced)
            let send_chunk_idx = (self.rank + n - round) % n;
            let (send_start, send_len) = chunks[send_chunk_idx];

            // Chunk index to receive
            let recv_chunk_idx = (self.rank + n - round - 1) % n;
            let (recv_start, recv_len) = chunks[recv_chunk_idx];

            // Serialize chunk to send
            f32_slice_to_bytes(&data[send_start..send_start + send_len], &mut send_buf);

            // Send and receive simultaneously
            // (TCP is full-duplex, so this won't deadlock)
            self.send_stream
                .write_all(&send_buf[..send_len * 4])
                .map_err(|e| format!("ring send error (round {round}): {e}"))?;
            self.recv_stream
                .read_exact(&mut recv_buf[..recv_len * 4])
                .map_err(|e| format!("ring recv error (round {round}): {e}"))?;

            // Element-wise add received chunk into local chunk
            for i in 0..recv_len {
                let received =
                    f32::from_le_bytes(recv_buf[i * 4..(i + 1) * 4].try_into().expect("4 bytes"));
                data[recv_start + i] += received;
            }
        }

        // ── Phase 2: All-gather ──────────────────────────────────────────
        // After N-1 rounds, all workers hold all reduced chunks.
        for round in 0..(n - 1) {
            let send_chunk_idx = (self.rank + n - round + 1) % n;
            let (send_start, send_len) = chunks[send_chunk_idx];

            let recv_chunk_idx = (self.rank + n - round) % n;
            let (recv_start, recv_len) = chunks[recv_chunk_idx];

            // Serialize reduced chunk to send
            f32_slice_to_bytes(&data[send_start..send_start + send_len], &mut send_buf);

            self.send_stream
                .write_all(&send_buf[..send_len * 4])
                .map_err(|e| format!("ring allgather send error (round {round}): {e}"))?;
            self.recv_stream
                .read_exact(&mut recv_buf[..recv_len * 4])
                .map_err(|e| format!("ring allgather recv error (round {round}): {e}"))?;

            // Copy received chunk into place (no addition — just replace)
            for i in 0..recv_len {
                data[recv_start + i] =
                    f32::from_le_bytes(recv_buf[i * 4..(i + 1) * 4].try_into().expect("4 bytes"));
            }
        }

        // ── Phase 3: Divide by N to get mean ────────────────────────────
        let inv_n = 1.0 / n as f32;
        for x in data.iter_mut() {
            *x *= inv_n;
        }

        Ok(())
    }
}

/// Convert f32 slice to little-endian bytes.
fn f32_slice_to_bytes(src: &[f32], dst: &mut [u8]) {
    for (i, &val) in src.iter().enumerate() {
        dst[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Simple AllReduce for exactly 2 workers using direct exchange.
///
/// Simpler than the ring algorithm for the N=2 case: each worker sends
/// its entire vector, receives the other's, and averages.
///
/// # Contract (C-RING-001)
///
/// Output = (local + remote) / 2
pub fn allreduce_pair(
    data: &mut [f32],
    send_stream: &mut TcpStream,
    recv_stream: &mut TcpStream,
) -> Result<(), String> {
    let byte_len = data.len() * 4;
    let mut send_buf = vec![0u8; byte_len];
    let mut recv_buf = vec![0u8; byte_len];

    f32_slice_to_bytes(data, &mut send_buf);

    send_stream.write_all(&send_buf).map_err(|e| format!("pair send error: {e}"))?;
    recv_stream.read_exact(&mut recv_buf).map_err(|e| format!("pair recv error: {e}"))?;

    for i in 0..data.len() {
        let remote = f32::from_le_bytes(recv_buf[i * 4..(i + 1) * 4].try_into().expect("4 bytes"));
        data[i] = (data[i] + remote) * 0.5;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// Set up a ring of N workers connected via TCP loopback.
    /// Returns N `RingAllReduceWorker` instances.
    fn setup_ring(n: usize) -> Vec<RingAllReduceWorker> {
        // Create N listeners
        let listeners: Vec<TcpListener> =
            (0..n).map(|_| TcpListener::bind("127.0.0.1:0").expect("bind")).collect();
        let addrs: Vec<_> = listeners.iter().map(|l| l.local_addr().expect("addr")).collect();

        // Each worker connects to its right neighbor
        // Worker w sends to (w+1)%N, receives from (w-1)%N
        let mut send_streams = Vec::with_capacity(n);
        let mut recv_streams = Vec::with_capacity(n);

        // Spawn accept threads for each listener
        let accept_handles: Vec<_> = listeners
            .into_iter()
            .map(|listener| {
                thread::spawn(move || {
                    let (stream, _) = listener.accept().expect("accept");
                    stream
                })
            })
            .collect();

        // Connect: worker w connects to listener (w+1)%N
        for w in 0..n {
            let right = (w + 1) % n;
            let stream = TcpStream::connect(addrs[right]).expect("connect");
            stream.set_nodelay(true).ok();
            send_streams.push(stream);
        }

        // Collect accepted connections (each listener accepted from (w-1)%N)
        for handle in accept_handles {
            let stream = handle.join().expect("accept thread");
            stream.set_nodelay(true).ok();
            recv_streams.push(stream);
        }

        // Build workers
        let mut workers = Vec::with_capacity(n);
        for w in 0..n {
            workers.push(RingAllReduceWorker::new(
                w,
                n,
                send_streams.remove(0),
                recv_streams.remove(0),
            ));
        }
        workers
    }

    #[test]
    fn test_ring_allreduce_2_workers_identical() {
        let mut workers = setup_ring(2);

        let data0 = vec![1.0f32, 2.0, 3.0];
        let data1 = vec![1.0f32, 2.0, 3.0];

        let mut d0 = data0.clone();
        let mut w1 = workers.pop().unwrap();
        let mut d1 = data1.clone();

        let h1 = thread::spawn(move || {
            w1.allreduce(&mut d1).expect("allreduce w1");
            d1
        });

        workers[0].allreduce(&mut d0).expect("allreduce w0");
        let result1 = h1.join().expect("join w1");

        // Identical inputs → output equals input
        for (&v, &expected) in d0.iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((v - expected).abs() < 1e-6, "w0: {v} != {expected}");
        }
        for (&v, &expected) in result1.iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((v - expected).abs() < 1e-6, "w1: {v} != {expected}");
        }
    }

    #[test]
    fn test_ring_allreduce_2_workers_distinct() {
        let mut workers = setup_ring(2);

        let mut d0 = vec![2.0f32, 4.0, 6.0];
        let mut d1 = vec![8.0f32, 6.0, 4.0];
        // Expected: (2+8)/2=5, (4+6)/2=5, (6+4)/2=5

        let mut w1 = workers.pop().unwrap();

        let h1 = thread::spawn(move || {
            w1.allreduce(&mut d1).expect("allreduce w1");
            d1
        });

        workers[0].allreduce(&mut d0).expect("allreduce w0");
        let result1 = h1.join().expect("join w1");

        for &v in &d0 {
            assert!((v - 5.0).abs() < 1e-6, "w0: {v} != 5.0");
        }
        for &v in &result1 {
            assert!((v - 5.0).abs() < 1e-6, "w1: {v} != 5.0");
        }
    }

    #[test]
    fn test_ring_allreduce_3_workers() {
        let mut workers = setup_ring(3);

        let mut d0 = vec![1.0f32, 0.0, 0.0];
        let mut d1 = vec![0.0f32, 1.0, 0.0];
        let mut d2 = vec![0.0f32, 0.0, 1.0];
        // Expected: [1/3, 1/3, 1/3]

        let mut w2 = workers.pop().unwrap();
        let mut w1 = workers.pop().unwrap();

        let h2 = thread::spawn(move || {
            w2.allreduce(&mut d2).expect("allreduce w2");
            d2
        });
        let h1 = thread::spawn(move || {
            w1.allreduce(&mut d1).expect("allreduce w1");
            d1
        });

        workers[0].allreduce(&mut d0).expect("allreduce w0");
        let r1 = h1.join().expect("join w1");
        let r2 = h2.join().expect("join w2");

        let expected = 1.0 / 3.0;
        for &v in &d0 {
            assert!((v - expected).abs() < 1e-5, "w0: {v} != {expected}");
        }
        for &v in &r1 {
            assert!((v - expected).abs() < 1e-5, "w1: {v} != {expected}");
        }
        for &v in &r2 {
            assert!((v - expected).abs() < 1e-5, "w2: {v} != {expected}");
        }
    }

    #[test]
    fn test_ring_allreduce_non_divisible_length() {
        // D=7, N=3 → chunks of [3, 2, 2]
        let mut workers = setup_ring(3);

        let mut d0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut d1 = vec![7.0f32, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut d2 = vec![0.0f32; 7];

        let mut w2 = workers.pop().unwrap();
        let mut w1 = workers.pop().unwrap();

        let h2 = thread::spawn(move || {
            w2.allreduce(&mut d2).expect("allreduce");
            d2
        });
        let h1 = thread::spawn(move || {
            w1.allreduce(&mut d1).expect("allreduce");
            d1
        });
        workers[0].allreduce(&mut d0).expect("allreduce");
        let r1 = h1.join().expect("join");
        let r2 = h2.join().expect("join");

        // Expected: (d0 + d1 + d2) / 3
        let expected: Vec<f32> =
            vec![8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0];
        for (i, (&v, &e)) in d0.iter().zip(&expected).enumerate() {
            assert!((v - e).abs() < 1e-5, "w0[{i}]: {v} != {e}");
        }
        assert_eq!(d0, r1, "w0 == w1");
        assert_eq!(d0, r2, "w0 == w2");
    }

    #[test]
    fn test_ring_allreduce_large_vector() {
        let mut workers = setup_ring(2);
        let d = 100_000;
        let mut d0: Vec<f32> = (0..d).map(|i| i as f32).collect();
        let mut d1: Vec<f32> = (0..d).map(|i| (d - 1 - i) as f32).collect();
        // d0[i] + d1[i] = d-1 for all i, so average = (d-1)/2

        let mut w1 = workers.pop().unwrap();

        let h1 = thread::spawn(move || {
            w1.allreduce(&mut d1).expect("allreduce");
            d1
        });
        workers[0].allreduce(&mut d0).expect("allreduce");
        let r1 = h1.join().expect("join");

        let expected = (d as f32 - 1.0) / 2.0;
        for (i, &v) in d0.iter().enumerate() {
            assert!((v - expected).abs() < 1e-2, "w0[{i}]: {v} != {expected}");
        }
        assert_eq!(d0, r1, "results must be identical");
    }

    #[test]
    fn test_allreduce_pair() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let h = thread::spawn(move || {
            let (recv, _) = listener.accept().expect("accept");
            let send = TcpStream::connect(addr).expect("connect");
            // This won't work — need bidirectional setup
            // For the pair test, use a simpler setup
            (recv, send)
        });

        // Simplified pair test using same connection both ways
        let listener_a = TcpListener::bind("127.0.0.1:0").expect("bind");
        let listener_b = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr_a = listener_a.local_addr().expect("addr");
        let addr_b = listener_b.local_addr().expect("addr");
        drop(h);

        let ha = thread::spawn(move || {
            let send = TcpStream::connect(addr_b).expect("connect to b");
            let (recv, _) = listener_a.accept().expect("accept from b");
            (send, recv)
        });

        let send_b = TcpStream::connect(addr_a).expect("connect to a");
        let (recv_b, _) = listener_b.accept().expect("accept from a");

        let (mut send_a, mut recv_a) = ha.join().expect("join");
        let mut send_b = send_b;
        let mut recv_b = recv_b;

        let mut d_a = vec![10.0f32, 20.0, 30.0];
        let mut d_b = vec![30.0f32, 20.0, 10.0];

        let hb = thread::spawn(move || {
            allreduce_pair(&mut d_b, &mut send_b, &mut recv_b).expect("pair b");
            d_b
        });

        allreduce_pair(&mut d_a, &mut send_a, &mut recv_a).expect("pair a");
        let result_b = hb.join().expect("join");

        assert_eq!(d_a, vec![20.0, 20.0, 20.0]);
        assert_eq!(result_b, vec![20.0, 20.0, 20.0]);
    }
}
