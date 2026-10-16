//! #5279 P0: the scheduler runs over [`BatchEngine`], so any engine that
//! implements it is batched by the same admission logic. A host-only mock
//! engine shows c concurrent requests form one batch of m = c — no GPU, no
//! model file.

use super::*;
use crate::gguf::BatchedDecodeState;
use std::sync::Mutex;

/// Each decode step emits this token for every live slot.
const TOKEN: u32 = 7;
/// Tokens each request asks for.
const MAX_TOKENS: usize = 3;

/// Records the m each batch was set up with (1 for the fast path).
struct MockEngine {
    formed: Arc<Mutex<Vec<usize>>>,
}

impl BatchEngine for MockEngine {
    fn generate_single(&mut self, req: CudaBatchRequest) {
        self.formed.lock().expect("formed").push(1);
        for _ in 0..req.config.max_tokens {
            let _ = req.token_tx.try_send(Ok(TOKEN));
        }
    }

    fn batched_setup_and_prefill(
        &mut self,
        prompts: &[Vec<u32>],
        configs: &[QuantizedGenerateConfig],
        on_tokens: Vec<TokenCallback>,
        max_kv_slots: usize,
    ) -> crate::error::Result<BatchedDecodeState> {
        let m = prompts.len();
        self.formed.lock().expect("formed").push(m);
        Ok(BatchedDecodeState {
            m,
            max_kv_slots,
            max_tokens_max: configs.iter().map(|c| c.max_tokens).max().unwrap_or(0),
            gen_idx: 0,
            hidden_dim: 0,
            intermediate_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            eps: 0.0,
            prompts: prompts.to_vec(),
            configs: configs.to_vec(),
            on_tokens,
            sequences: prompts.to_vec(),
            positions: prompts.iter().map(Vec::len).collect(),
            last_tokens: prompts
                .iter()
                .map(|p| p.last().copied().unwrap_or(0))
                .collect(),
            done: vec![false; m],
            embed_buf: Vec::new(),
            pos_buf: Vec::new(),
            recorders: (0..m).map(|_| None).collect(),
            records: (0..m).map(|_| None).collect(),
        })
    }

    fn batched_decode_step(
        &mut self,
        state: &mut BatchedDecodeState,
    ) -> crate::error::Result<Vec<u32>> {
        Ok(vec![TOKEN; state.m])
    }

    fn add_slot_to_batch(
        &mut self,
        _state: &mut BatchedDecodeState,
        _prompt: Vec<u32>,
        _config: QuantizedGenerateConfig,
        _on_token: TokenCallback,
    ) -> crate::error::Result<()> {
        Err(unsupported("add_slot_to_batch"))
    }

    fn recycle_slot(
        &mut self,
        _state: &mut BatchedDecodeState,
        _slot_idx: usize,
        _prompt: Vec<u32>,
        _config: QuantizedGenerateConfig,
        _on_token: TokenCallback,
    ) -> crate::error::Result<()> {
        Err(unsupported("recycle_slot"))
    }

    fn batched_cleanup(&mut self, _state: &BatchedDecodeState) {}

    fn reset_batched_state(&mut self) {}
}

fn unsupported(operation: &str) -> crate::error::RealizarError {
    crate::error::RealizarError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: "the mock engine never joins mid-batch".to_string(),
    }
}

/// Send `c` requests at once and wait for every one to finish; what each
/// caller received.
fn band(tx: &tokio::sync::mpsc::Sender<CudaBatchRequest>, c: usize) -> Vec<Vec<u32>> {
    let receivers: Vec<_> = (0..c)
        .map(|i| {
            let (token_tx, rx) = tokio::sync::mpsc::channel(64);
            let config = QuantizedGenerateConfig {
                max_tokens: MAX_TOKENS,
                ..QuantizedGenerateConfig::default()
            };
            tx.try_send(CudaBatchRequest {
                prompt_ids: vec![1, 2, u32::try_from(i).expect("small")],
                config,
                token_tx,
                non_streaming: false,
                enqueue_time: std::time::Instant::now(),
                timing_tx: None,
                logprobs: None,
            })
            .unwrap_or_else(|e| panic!("queue request {i}: {e}"));
            rx
        })
        .collect();
    receivers
        .into_iter()
        .map(|mut rx| {
            let mut got = Vec::new();
            while let Some(t) = rx.blocking_recv() {
                got.push(t.expect("no error"));
            }
            got
        })
        .collect()
}

/// FALSIFY (#5279 P0): c concurrent requests to an engine behind the trait
/// form exactly one batch of m = c, and every caller gets its tokens.
#[test]
fn concurrent_requests_form_one_batch_of_c_over_the_trait() {
    let formed = Arc::new(Mutex::new(Vec::new()));
    let engine = Arc::new(std::sync::RwLock::new(MockEngine {
        formed: formed.clone(),
    }));
    let config = CudaBatchConfig {
        max_batch: 16,
        window_ms: 200,
    };
    let counter = crate::api::InFlightCounter::new();
    let tx = spawn_cuda_batch_scheduler(engine, config, counter);

    let ladder = [1_usize, 4, 8, 16];
    for &c in &ladder {
        let got = band(&tx, c);
        assert_eq!(got.len(), c);
        for (i, tokens) in got.iter().enumerate() {
            assert_eq!(
                tokens,
                &vec![TOKEN; MAX_TOKENS],
                "c={c}: request {i} got {tokens:?}"
            );
        }
    }
    assert_eq!(
        *formed.lock().expect("formed"),
        ladder.to_vec(),
        "each band formed one batch of m = c"
    );
}
