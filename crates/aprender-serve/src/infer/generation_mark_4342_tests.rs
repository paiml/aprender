//! #4342: the F2 guard is setup, so generation is marked only after it.
//! A file of its own: its test `ArchForward` must not count as an architecture (#4263 scan).

use super::*;
use crate::session::{ArchForward, Session};
use std::time::{Duration, Instant};

/// A forward whose `validate` stands in for the F2 guard: it takes a while,
/// as the guard's CPU probe does, and records when it finished.
struct Guarded {
    guard_done: Vec<Instant>,
}

impl ArchForward for Guarded {
    fn arch(&self) -> &'static str {
        "guarded"
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn context_length(&self) -> usize {
        64
    }
    fn batched_prefills(&self) -> usize {
        0
    }
    fn notices(&self) -> &[String] {
        &[]
    }
    fn reserve(&mut self, _positions: usize) -> Result<bool> {
        Ok(false)
    }
    fn validate(&mut self, _probe: &[u32]) -> Result<()> {
        std::thread::sleep(Duration::from_millis(20));
        self.guard_done.push(Instant::now());
        Ok(())
    }
    fn forward(&mut self, _tokens: &[u32], _start: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0.0; 8];
        logits[3] = 1.0;
        Ok(logits)
    }
}

/// The ticket's must-RED shape: generation was marked after the weights were
/// uploaded but before the F2 guard, so a qwen35 `--gpu` run's tok/s divided
/// by the guard's probe too (#4342).
#[test]
fn the_f2_guard_runs_once_before_generation_is_marked() {
    let _ = take_generation_start();
    let mut session = Session::new(Guarded { guard_done: Vec::new() });
    let config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let turn = generate_after_setup(&mut session, &[7501, 7502], &config)
        .expect("the turn runs");
    assert_eq!(turn.tokens, vec![7501, 7502, 3, 3, 3, 3]);
    let mark = take_generation_start().expect("generation was marked");
    let guard = &session.engine().guard_done;
    assert_eq!(guard.len(), 1, "the guard ran once, as setup");
    assert!(
        mark >= guard[0],
        "generation was marked before the F2 guard finished, so tok/s bills the guard"
    );
}
