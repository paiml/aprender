//! The engine's loop, judged on a scripted forward: prefix reuse, the witness,
//! stop tokens, the context cap and the scoring walk.

use super::*;

/// A forward whose logits make `next` the argmax after every position, and
/// which records each `forward` call's `(len, start)`.
struct Scripted {
    next: u32,
    context: usize,
    calls: Vec<(usize, usize)>,
    reserves: usize,
    drop_on_reserve: bool,
}

impl Scripted {
    fn new(next: u32, context: usize) -> Self {
        Self {
            next,
            context,
            calls: Vec::new(),
            reserves: 0,
            drop_on_reserve: false,
        }
    }
}

impl ArchForward for Scripted {
    fn arch(&self) -> &'static str {
        "scripted"
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn context_length(&self) -> usize {
        self.context
    }
    fn batched_prefills(&self) -> usize {
        0
    }
    fn notices(&self) -> &[String] {
        &[]
    }
    fn reserve(&mut self, _positions: usize) -> Result<bool> {
        self.reserves += 1;
        Ok(self.drop_on_reserve)
    }
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        self.calls.push((tokens.len(), start));
        let mut logits = vec![0.0; 8];
        logits[self.next as usize] = 1.0;
        Ok(logits)
    }
}

fn greedy(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    }
}

#[test]
fn generate_prefills_once_then_decodes_one_token_per_step() {
    let mut s = Session::new(Scripted::new(3, 100));
    let prompt = [7101, 7102, 7103];
    let turn = s
        .generate(&prompt, &greedy(3), &mut |_| true)
        .expect("turn");
    assert_eq!(turn.tokens, vec![7101, 7102, 7103, 3, 3, 3]);
    assert_eq!(turn.reused, 0);
    assert!(!turn.context_capped);
    // The prompt whole from 0, then each chosen token (not the last) from where it left.
    assert_eq!(s.engine().calls, vec![(3, 0), (4, 3), (5, 4)]);
    assert_eq!(s.processed_len(), 5);
}

#[test]
fn an_extending_prompt_reuses_the_state_and_a_diverging_one_resets() {
    let mut s = Session::new(Scripted::new(3, 100));
    let t1 = s
        .generate(&[7201, 7202], &greedy(1), &mut |_| true)
        .expect("t1");
    let mut p2 = t1.tokens.clone();
    p2.extend([7203, 7204]);
    let t2 = s.generate(&p2, &greedy(1), &mut |_| true).expect("t2");
    // t1 held [7201, 7202] (its one chosen token was never forwarded).
    assert_eq!(t2.reused, 2);
    let t3 = s
        .generate(&[7299, 7202], &greedy(1), &mut |_| true)
        .expect("t3");
    assert_eq!(t3.reused, 0);
    assert_eq!(s.engine().calls.last(), Some(&(2, 0)));
}

#[test]
fn a_dropped_state_is_never_reused() {
    let mut f = Scripted::new(3, 100);
    f.drop_on_reserve = true;
    let mut s = Session::new(f);
    let t1 = s
        .generate(&[7301, 7302], &greedy(1), &mut |_| true)
        .expect("t1");
    let mut p2 = t1.tokens;
    p2.push(7303);
    let t2 = s.generate(&p2, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(t2.reused, 0, "reserve said the state was dropped");
}

#[test]
fn a_stop_token_ends_the_turn_and_the_context_caps_it_by_name() {
    let mut s = Session::new(Scripted::new(3, 100));
    let mut cfg = greedy(10);
    cfg.stop_tokens = vec![3];
    let turn = s.generate(&[7401], &cfg, &mut |_| true).expect("turn");
    assert_eq!(turn.tokens, vec![7401, 3]);
    assert!(!turn.context_capped);

    let mut s = Session::new(Scripted::new(3, 4));
    let turn = s
        .generate(&[7402, 7403], &greedy(10), &mut |_| true)
        .expect("turn");
    assert_eq!(turn.tokens.len(), 4, "two positions of room");
    assert!(turn.context_capped);
    assert!(s
        .generate(&[1, 2, 3, 4], &greedy(1), &mut |_| true)
        .is_err());
}

#[test]
fn generate_leaves_exactly_one_entry() {
    let prompt = [7501, 7502, 7503, 7504];
    let mut s = Session::new(Scripted::new(3, 100));
    s.generate(&prompt, &greedy(2), &mut |_| true)
        .expect("turn");
    let entries = entries_for(&prompt);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].arch, "scripted");
    assert_eq!(entries[0].kind, EntryKind::Generate);
}

#[test]
fn score_leaves_a_score_entry_and_visits_every_position() {
    let seq = [7601, 7602, 7603];
    let mut s = Session::new(Scripted::new(3, 100));
    let mut seen = Vec::new();
    s.score(&seq, &mut |pos, logits| {
        seen.push((pos, logits.len()));
        true
    })
    .expect("score");
    assert_eq!(seen, vec![(0, 8), (1, 8), (2, 8)]);
    // Teacher-forced: each position extends the last.
    assert_eq!(s.engine().calls, vec![(1, 0), (2, 1), (3, 2)]);
    let entries = entries_for(&seq);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kind, EntryKind::Score);
    assert!(s.score(&[1; 101], &mut |_, _| true).is_err());
}

#[test]
fn digest_separates_order_and_length() {
    assert_ne!(prompt_digest(&[1, 2]), prompt_digest(&[2, 1]));
    assert_ne!(prompt_digest(&[1]), prompt_digest(&[1, 0]));
    assert_eq!(prompt_digest(&[5, 6]), prompt_digest(&[5, 6]));
}

/// Scripted with a device argmax: counts `forward_greedy` calls and answers
/// `greedy_answer` without logits.
struct DeviceGreedy {
    inner: Scripted,
    greedy_answer: u32,
    greedy_calls: usize,
}

impl ArchForward for DeviceGreedy {
    fn arch(&self) -> &'static str {
        "scripted-greedy"
    }
    fn on_gpu(&self) -> bool {
        true
    }
    fn context_length(&self) -> usize {
        self.inner.context
    }
    fn batched_prefills(&self) -> usize {
        0
    }
    fn notices(&self) -> &[String] {
        &[]
    }
    fn reserve(&mut self, positions: usize) -> Result<bool> {
        self.inner.reserve(positions)
    }
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        self.inner.forward(tokens, start)
    }
    fn forward_greedy(&mut self, tokens: &[u32], start: usize) -> Result<Option<u32>> {
        self.inner.calls.push((tokens.len(), start));
        self.greedy_calls += 1;
        Ok(Some(self.greedy_answer))
    }
}

#[test]
fn plain_greedy_takes_the_device_argmax_and_a_penalty_or_sampling_does_not() {
    let device = |answer| DeviceGreedy {
        inner: Scripted::new(3, 100),
        greedy_answer: answer,
        greedy_calls: 0,
    };
    let mut s = Session::new(device(5));
    let turn = s
        .generate(&[7701, 7702], &greedy(2), &mut |_| true)
        .expect("turn");
    assert_eq!(turn.tokens, vec![7701, 7702, 5, 5]);
    assert_eq!(s.engine().greedy_calls, 2);
    assert_eq!(
        s.engine().inner.calls,
        vec![(2, 0), (3, 2)],
        "the state still advances by extension"
    );

    let mut penalized = greedy(2);
    penalized.repeat_penalty = 1.3;
    let mut s = Session::new(device(5));
    s.generate(&[7703], &penalized, &mut |_| true)
        .expect("turn");
    assert_eq!(
        s.engine().greedy_calls,
        0,
        "a repeat penalty needs the logits"
    );
}
