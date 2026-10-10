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
    /// Checkpoint before the last position holding this token (#4214).
    marker: Option<u32>,
    /// Positions the state holds, as the forward itself tracks them.
    held: usize,
    /// Positions the saved checkpoint holds.
    saved: Option<usize>,
    restores: usize,
    /// The forward loses its checkpoint (a reallocation, a fallback).
    lose_checkpoint: bool,
    /// What `set_layer_timing` answers: whether this forward times its layers.
    times_layers: bool,
    /// Every `set_layer_timing` argument, in order.
    timing_log: Vec<bool>,
    /// What `take_layer_timings` hands back, once.
    timed: Option<Vec<LayerTiming>>,
    /// The forward call (0-based) from which every forward errors.
    fail_from: Option<usize>,
}

impl Scripted {
    fn new(next: u32, context: usize) -> Self {
        Self {
            next,
            context,
            calls: Vec::new(),
            reserves: 0,
            drop_on_reserve: false,
            marker: None,
            held: 0,
            saved: None,
            restores: 0,
            lose_checkpoint: false,
            times_layers: false,
            timing_log: Vec::new(),
            timed: None,
            fail_from: None,
        }
    }

    fn checkpointing(next: u32, context: usize, marker: u32) -> Self {
        Self {
            marker: Some(marker),
            ..Self::new(next, context)
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
    fn checkpoint_at(&self, prompt: &[u32]) -> Option<usize> {
        let marker = self.marker?;
        prompt.iter().rposition(|&t| t == marker)
    }
    fn save_checkpoint(&mut self) -> Result<()> {
        assert!(self.marker.is_some(), "saved on a forward that keeps none");
        self.saved = Some(self.held);
        Ok(())
    }
    fn restore_checkpoint(&mut self) -> Result<bool> {
        if self.lose_checkpoint {
            self.saved = None;
        }
        let Some(saved) = self.saved else {
            return Ok(false);
        };
        self.held = saved;
        self.restores += 1;
        Ok(true)
    }
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        // The session may only claim positions the state holds.
        assert!(
            start == 0 || start == self.held,
            "forward from {start}, but the state holds {}",
            self.held
        );
        if self.fail_from.is_some_and(|n| self.calls.len() >= n) {
            return Err(RealizarError::InvalidShape {
                reason: "scripted forward failure".to_string(),
            });
        }
        self.calls.push((tokens.len(), start));
        self.held = tokens.len();
        let mut logits = vec![0.0; 8];
        logits[self.next as usize] = 1.0;
        Ok(logits)
    }
    fn set_layer_timing(&mut self, on: bool) -> bool {
        self.timing_log.push(on);
        self.times_layers
    }
    fn take_layer_timings(&mut self) -> Option<Vec<LayerTiming>> {
        self.timed.take()
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

/// A tracer that records what a `step` or `layer` serve trace records.
fn step_tracer() -> crate::inference_trace::InferenceTracer {
    use crate::inference_trace::{InferenceTracer, TraceConfig, TraceStep};
    let mut config = TraceConfig::enabled();
    config.steps = [TraceStep::TransformerBlock, TraceStep::Decode]
        .into_iter()
        .collect();
    InferenceTracer::new(config)
}

/// What a traced turn filed, as `(step, iteration)` in order.
fn filed(
    tracer: &crate::inference_trace::InferenceTracer,
) -> Vec<(crate::inference_trace::TraceStep, usize)> {
    tracer
        .events()
        .iter()
        .map(|e| (e.step, e.iteration))
        .collect()
}

const TIMED: LayerTiming = LayerTiming {
    total_us: 7,
    calls: 2,
    kind: "scripted",
};

/// APR-OBS-001 OBS-09: a traced turn files one `TransformerBlock` per forward,
/// iteration 0 being the prompt's, and one `Decode` per `on_token` at the
/// iteration of the token it emits. `serve_trace` splits prefill from decode
/// on exactly this shape, so a shifted iteration would report a decode as
/// prefill and still fit under the wall clock.
#[test]
fn a_traced_turn_files_one_event_per_forward_and_per_token() {
    use crate::inference_trace::TraceStep::{Decode, TransformerBlock};
    let mut s = Session::new(Scripted {
        times_layers: true,
        timed: Some(vec![TIMED]),
        ..Scripted::new(3, 100)
    });
    let mut tracer = step_tracer();
    let turn = s
        .generate_traced(
            &[7901, 7902, 7903],
            &greedy(3),
            &mut |_| true,
            Some(TurnTrace {
                tracer: &mut tracer,
                layers: true,
            }),
        )
        .expect("turn");
    assert_eq!(turn.tokens, vec![7901, 7902, 7903, 3, 3, 3]);
    assert_eq!(
        filed(&tracer),
        [
            (TransformerBlock, 0),
            (Decode, 0),
            (TransformerBlock, 1),
            (Decode, 1),
            (TransformerBlock, 2),
            (Decode, 2),
        ]
    );
    assert_eq!(
        s.engine().timing_log,
        [true, false],
        "on for the turn, then off"
    );
    assert_eq!(s.take_layer_timings(), Some(vec![TIMED]));
    assert_eq!(s.take_layer_timings(), None, "taking it clears it");
}

/// Layer timing is asked for only by a layer trace, and turned off after the
/// turn, Ok or Err, on every forward that said it would time its layers. A
/// forward left timing would run every later decode eager, syncing after each
/// layer, with nothing in any reply to show it.
#[test]
fn layer_timing_is_on_for_the_turn_and_off_after_it_ok_or_err() {
    // (layer trace, forward times layers, first failing forward call,
    // the set_layer_timing calls the forward sees)
    let cases: [(bool, bool, Option<usize>, &[bool]); 6] = [
        (true, true, None, &[true, false]),
        (true, true, Some(0), &[true, false]), // the prompt's forward errors
        (true, true, Some(2), &[true, false]), // a decode forward errors
        (true, false, None, &[true]),          // a CPU forward: nothing to undo
        (false, true, None, &[]),              // a step trace times no layer
        (false, true, Some(0), &[]),
    ];
    for (layers, times, fail_from, want) in cases {
        let case = format!("layers={layers} times={times} fail_from={fail_from:?}");
        let mut s = Session::new(Scripted {
            times_layers: times,
            fail_from,
            ..Scripted::new(3, 100)
        });
        let mut tracer = step_tracer();
        let r = s.generate_traced(
            &[7911, 7912],
            &greedy(3),
            &mut |_| true,
            Some(TurnTrace {
                tracer: &mut tracer,
                layers,
            }),
        );
        assert_eq!(r.is_err(), fail_from.is_some(), "{case}");
        assert_eq!(s.engine().timing_log, want, "{case}");
    }
    // An untraced turn asks nothing of the forward's timing.
    let mut s = Session::new(Scripted {
        times_layers: true,
        ..Scripted::new(3, 100)
    });
    s.generate(&[7913], &greedy(2), &mut |_| true)
        .expect("turn");
    assert!(s.engine().timing_log.is_empty());
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
/// `greedy_answer`, with `read_back` as the logits when they are asked for.
struct DeviceGreedy {
    inner: Scripted,
    greedy_answer: u32,
    greedy_calls: usize,
    /// Each call's `read_logits` (#4971).
    reads: Vec<bool>,
    /// The logits a read hands back; `None` plays a backend that drops them.
    read_back: Option<Vec<f32>>,
}

impl DeviceGreedy {
    fn new(greedy_answer: u32) -> Self {
        Self {
            inner: Scripted::new(3, 100),
            greedy_answer,
            greedy_calls: 0,
            reads: Vec::new(),
            read_back: None,
        }
    }

    fn reading(greedy_answer: u32, read_back: Vec<f32>) -> Self {
        Self {
            read_back: Some(read_back),
            ..Self::new(greedy_answer)
        }
    }
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
    fn checkpoint_at(&self, prompt: &[u32]) -> Option<usize> {
        self.inner.checkpoint_at(prompt)
    }
    fn save_checkpoint(&mut self) -> Result<()> {
        self.inner.save_checkpoint()
    }
    fn restore_checkpoint(&mut self) -> Result<bool> {
        self.inner.restore_checkpoint()
    }
    fn forward_greedy(
        &mut self,
        tokens: &[u32],
        start: usize,
        read_logits: bool,
    ) -> Result<Option<GreedyStep>> {
        self.inner.calls.push((tokens.len(), start));
        self.inner.held = tokens.len();
        self.greedy_calls += 1;
        self.reads.push(read_logits);
        Ok(Some(GreedyStep {
            token: self.greedy_answer,
            logits: if read_logits {
                self.read_back.clone()
            } else {
                None
            },
        }))
    }
}

#[test]
fn plain_greedy_takes_the_device_argmax_and_a_penalty_or_sampling_does_not() {
    let device = DeviceGreedy::new;
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
// #4445: `apr bench` repeats one prompt and must time a whole prefill each time.
// An extending prompt would resume the held state; after forget_prefix it may not.
#[test]
fn forget_prefix_makes_an_extending_prompt_prefill_whole_again() {
    let mut s = Session::new(Scripted::new(3, 100));
    let t1 = s
        .generate(&[7401, 7402], &greedy(1), &mut |_| true)
        .expect("t1");
    let mut p2 = t1.tokens.clone();
    p2.extend([7403, 7404]);
    s.forget_prefix();
    assert_eq!(s.processed_len(), 0);
    let t2 = s.generate(&p2, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(t2.reused, 0, "resumed a prefix after forget_prefix");
    assert_eq!(s.engine().calls.last(), Some(&(p2.len(), 0)));
}

// #4214 / #4274: a prompt that repeats, or re-renders, an earlier turn's history
// resumes from the checkpoint the earlier turn left, not from position 0.

const MARK: u32 = 7;

#[test]
fn an_identical_prompt_again_restores_the_checkpoint_before_its_last_marker() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    let prompt = [7801, 7802, MARK, 7803];
    let t1 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    assert_eq!(t1.reused, 0);
    // Prefilled in two spans: up to the checkpoint, then the rest.
    assert_eq!(s.engine().calls, vec![(2, 0), (4, 2)]);
    let t2 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(
        t2.reused, 2,
        "the repeat prefilled only from the checkpoint"
    );
    assert_eq!(t2.tokens, t1.tokens);
    assert_eq!(s.engine().calls.last(), Some(&(4, 2)));
    assert_eq!(s.engine().restores, 1);
}

#[test]
fn a_re_rendered_history_resumes_from_the_last_turns_checkpoint() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    let t1 = s
        .generate(&[7811, MARK, 7812], &greedy(2), &mut |_| true)
        .expect("t1");
    assert_eq!(t1.tokens, vec![7811, MARK, 7812, 3, 3]);
    // Turn 2 re-renders turn 1's reply (7899, not the 3 it generated), so it
    // does not extend what the state holds.
    let p2 = [7811, MARK, 7812, 7899, MARK, 7813];
    let before = s.engine().calls.len();
    let t2 = s.generate(&p2, &greedy(2), &mut |_| true).expect("t2");
    assert_eq!(t2.reused, 1, "resumed at turn 1's checkpoint, not at 0");
    // From the checkpoint to turn 2's own (its last marker), then the rest.
    assert_eq!(s.engine().calls[before..before + 2], [(4, 1), (6, 4)]);
    // Turn 3 re-renders turn 2's reply too, and resumes at turn 2's checkpoint.
    let p3 = [7811, MARK, 7812, 7899, MARK, 7813, 7898, MARK, 7814];
    let t3 = s.generate(&p3, &greedy(1), &mut |_| true).expect("t3");
    assert_eq!(t3.reused, 4);
}

#[test]
fn a_prompt_that_diverges_before_the_checkpoint_starts_over() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    s.generate(&[7821, 7822, MARK, 7823], &greedy(1), &mut |_| true)
        .expect("t1");
    let t2 = s
        .generate(&[7829, 7822, MARK, 7823], &greedy(1), &mut |_| true)
        .expect("t2");
    assert_eq!(t2.reused, 0);
    assert_eq!(s.engine().restores, 0);
}

#[test]
fn a_checkpoint_does_not_survive_a_dropped_state() {
    let mut f = Scripted::checkpointing(3, 100, MARK);
    f.drop_on_reserve = true;
    let mut s = Session::new(f);
    let prompt = [7831, 7832, MARK, 7833];
    s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    let t2 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(t2.reused, 0, "reserve said the state was dropped");
    assert_eq!(s.engine().restores, 0);
}

#[test]
fn a_checkpoint_the_forward_lost_is_not_reused() {
    let mut f = Scripted::checkpointing(3, 100, MARK);
    f.lose_checkpoint = true;
    let mut s = Session::new(f);
    let prompt = [7841, 7842, MARK, 7843];
    s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    let t2 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(t2.reused, 0);
    assert_eq!(
        s.engine().calls.last(),
        Some(&(4, 2)),
        "re-checkpointed from 0"
    );
    assert_eq!(s.engine().calls[s.engine().calls.len() - 2], (2, 0));
}

#[test]
fn scoring_rewrites_the_state_so_the_checkpoint_is_dropped() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    let prompt = [7851, 7852, MARK, 7853];
    s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    s.score(&[7859, 7858], &mut |_, _| true).expect("score");
    let t2 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t2");
    assert_eq!(
        t2.reused, 0,
        "the positions under the checkpoint were rewritten"
    );
    assert_eq!(s.engine().restores, 0);
}

#[test]
fn a_device_argmax_from_below_the_checkpoint_drops_it() {
    // The device-argmax path rewrites positions too: a prompt it prefills from
    // under the checkpoint leaves a copy whose KV rows no longer match.
    let mut s = Session::new(DeviceGreedy {
        inner: Scripted::checkpointing(3, 100, MARK),
        greedy_answer: 5,
        greedy_calls: 0,
        reads: Vec::new(),
        read_back: None,
    });
    let prompt = [7861, 7862, MARK, 7863];
    s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    s.generate(&[7869], &greedy(1), &mut |_| true).expect("t2");
    assert!(s.engine().greedy_calls > 0, "t2 took the device argmax");
    let t3 = s.generate(&prompt, &greedy(1), &mut |_| true).expect("t3");
    assert_eq!(
        t3.reused, 0,
        "position 0 was rewritten under the checkpoint"
    );
    assert_eq!(s.engine().inner.restores, 0);
}

// #4445 x #4214: `apr bench` on a checkpointing arch (qwen35) repeats the same
// prompt. forget_prefix must drop the checkpoint too, or the repeat restores it
// and the bench times a partial prefill.
#[test]
fn forget_prefix_makes_the_same_prompt_prefill_whole_again() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    let prompt = [7871, 7872, MARK, 7873];
    s.generate(&prompt, &greedy(1), &mut |_| true).expect("t1");
    for turn in 2..=3 {
        s.forget_prefix();
        let t = s
            .generate(&prompt, &greedy(1), &mut |_| true)
            .expect("turn");
        assert_eq!(
            t.reused, 0,
            "turn {turn}: restored a checkpoint after forget_prefix"
        );
    }
    assert_eq!(s.engine().restores, 0);
}

/// A forward that implements only the required methods, so every checkpoint
/// hook is the trait default.
struct Bare;

impl ArchForward for Bare {
    fn arch(&self) -> &'static str {
        "bare"
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn context_length(&self) -> usize {
        100
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
    fn forward(&mut self, tokens: &[u32], _start: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0.0; 8];
        logits[tokens.len() % 8] = 1.0;
        Ok(logits)
    }
}

#[test]
fn the_default_forward_keeps_no_checkpoint() {
    let mut b = Bare;
    assert_eq!(b.checkpoint_at(&[1, 2, 3]), None);
    assert_eq!(b.checkpoint_at(&[]), None);
    assert!(b.save_checkpoint().is_ok());
    assert!(
        !b.restore_checkpoint().expect("default restore"),
        "the default has nothing to return to"
    );
}

/// The default times nothing and says so: a layer trace on such a forward
/// turns no timing on and leaves no per-layer rows (APR-OBS-001 OBS-09).
#[test]
fn the_default_forward_times_no_layer() {
    let mut b = Bare;
    assert!(!b.set_layer_timing(true), "the default times no layer");
    assert!(!b.set_layer_timing(false));
    assert!(b.take_layer_timings().is_none());

    let mut s = Session::new(Bare);
    let mut tracer = step_tracer();
    s.generate_traced(
        &[1, 2],
        &greedy(2),
        &mut |_| true,
        Some(TurnTrace {
            tracer: &mut tracer,
            layers: true,
        }),
    )
    .expect("a layer trace on a forward that cannot time layers still runs the turn");
    assert!(s.take_layer_timings().is_none());
}

/// A forward that names a fixed checkpoint position whatever the prompt.
struct FixedCheckpoint {
    inner: Scripted,
    at: usize,
}

impl ArchForward for FixedCheckpoint {
    fn arch(&self) -> &'static str {
        "fixed"
    }
    fn on_gpu(&self) -> bool {
        false
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
    fn checkpoint_at(&self, _prompt: &[u32]) -> Option<usize> {
        Some(self.at)
    }
    fn save_checkpoint(&mut self) -> Result<()> {
        self.inner.saved = Some(self.inner.held);
        Ok(())
    }
}

#[test]
fn a_prompt_equal_to_the_checkpoint_does_not_restore_it() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    s.generate(&[7851, MARK, 7852], &greedy(1), &mut |_| true)
        .expect("t1");
    assert_eq!(s.checkpoint, Some(vec![7851]));
    // Exactly the checkpoint: nothing beyond it to resume into.
    let reused = s.prepare_prompt(&[7851]).expect("prepare");
    assert_eq!(reused, 0);
    assert_eq!(s.engine().restores, 0);
    assert_eq!(s.checkpoint, Some(vec![7851]), "kept, not restored");
}

#[test]
fn a_checkpoint_at_the_resume_point_is_not_taken() {
    let mut s = Session::new(Scripted::checkpointing(3, 100, MARK));
    // The marker is the first token: k == start == 0.
    let reused = s.prepare_prompt(&[MARK, 7861, 7862]).expect("prepare");
    assert_eq!(reused, 0);
    assert!(
        s.engine().calls.is_empty(),
        "no prefill span for k == start"
    );
    assert_eq!(s.engine().saved, None);
    assert_eq!(s.checkpoint, None);
}

#[test]
fn a_checkpoint_at_the_prompt_end_is_not_taken() {
    let mut s = Session::new(FixedCheckpoint {
        inner: Scripted::new(3, 100),
        at: 3,
    });
    let reused = s.prepare_prompt(&[7871, 7872, 7873]).expect("prepare");
    assert_eq!(reused, 0);
    assert!(
        s.engine().inner.calls.is_empty(),
        "k == prompt.len() is no checkpoint"
    );
    assert_eq!(s.engine().inner.saved, None);
    assert_eq!(s.checkpoint, None);
}

#[test]
fn a_checkpoint_inside_the_prompt_is_taken() {
    let mut s = Session::new(FixedCheckpoint {
        inner: Scripted::new(3, 100),
        at: 2,
    });
    s.prepare_prompt(&[7881, 7882, 7883]).expect("prepare");
    assert_eq!(s.engine().inner.calls, vec![(2, 0)]);
    assert_eq!(s.checkpoint, Some(vec![7881, 7882]));
}

// #4971: logprobs come from the logits the choice read, and asking for them
// never changes the token or the path that chose it.
mod logprobs_4971 {
    use super::*;

    fn collect(
        s: &mut Session<impl ArchForward>,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        top_n: Option<usize>,
    ) -> Result<(Turn, Vec<Option<StepLogprobs>>)> {
        let mut records = Vec::new();
        let turn = s.generate_with_logprobs(prompt, config, top_n, &mut |_, r| {
            records.push(r);
            true
        })?;
        Ok((turn, records))
    }

    #[test]
    fn the_host_path_records_each_step_and_chooses_the_same_tokens() {
        let mut plain = Session::new(Scripted::new(3, 100));
        let want = plain
            .generate(&[7901, 7902], &greedy(3), &mut |_| true)
            .expect("plain");
        let mut s = Session::new(Scripted::new(3, 100));
        let (turn, records) = collect(&mut s, &[7901, 7902], &greedy(3), Some(2)).expect("turn");
        assert_eq!(turn.tokens, want.tokens);
        assert_eq!(records.len(), 3);
        for (i, r) in records.iter().enumerate() {
            let r = r.as_ref().expect("a record per step");
            assert_eq!(r.step, i);
            assert_eq!(r.chosen, 3);
            assert_eq!(r.top[0].token_id, 3, "greedy: top-1 is the choice");
            assert_eq!(r.chosen_logprob, r.top[0].logprob);
            assert_eq!(r.top.len(), 2);
            assert!(r.top2_margin().is_some_and(|m| (m - 1.0).abs() < 1e-6));
        }
        // Without top_n nothing is recorded.
        let mut s = Session::new(Scripted::new(3, 100));
        let (_, none) = collect(&mut s, &[7901, 7902], &greedy(3), None).expect("turn");
        assert!(none.iter().all(Option::is_none));
    }

    #[test]
    fn the_device_argmax_is_kept_and_its_logits_are_reported_as_read() {
        // The read-back logits rank token 2 first; the device chose 5. The
        // record must keep the device's token and report the distribution.
        let read_back = vec![0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut s = Session::new(DeviceGreedy::reading(5, read_back.clone()));
        let (turn, records) = collect(&mut s, &[7911, 7912], &greedy(2), Some(2)).expect("turn");
        assert_eq!(turn.tokens, vec![7911, 7912, 5, 5]);
        assert_eq!(s.engine().greedy_calls, 2, "the device path, every step");
        assert_eq!(s.engine().reads, vec![true, true]);
        for r in records.iter().map(|r| r.as_ref().expect("record")) {
            assert_eq!(r.chosen, 5);
            assert_eq!(r.top[0].token_id, 2);
            let want = crate::gguf::logprob_of(&read_back, 5);
            assert!((r.chosen_logprob - want).abs() < 1e-6);
        }

        // The same turn without logprobs: same tokens, same path, no read.
        let mut s = Session::new(DeviceGreedy::reading(5, read_back));
        let (plain, _) = collect(&mut s, &[7911, 7912], &greedy(2), None).expect("turn");
        assert_eq!(plain.tokens, turn.tokens);
        assert_eq!(s.engine().greedy_calls, 2);
        assert_eq!(s.engine().reads, vec![false, false]);
    }

    #[test]
    fn a_near_tie_and_a_clear_winner_read_as_their_margins() {
        let near = vec![0.0, 1.0, 1.01, -3.0, 0.0, 0.0, 0.0, 0.0];
        let mut s = Session::new(DeviceGreedy::reading(2, near));
        let (_, records) = collect(&mut s, &[7921], &greedy(1), Some(2)).expect("turn");
        let margin = records[0].as_ref().and_then(StepLogprobs::top2_margin);
        assert!(margin.is_some_and(|m| m < 0.05), "{margin:?}");

        let clear = vec![0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut s = Session::new(DeviceGreedy::reading(1, clear));
        let (_, records) = collect(&mut s, &[7922], &greedy(1), Some(2)).expect("turn");
        let margin = records[0].as_ref().and_then(StepLogprobs::top2_margin);
        assert!(margin.is_some_and(|m| m > 1.0), "{margin:?}");
    }

    #[test]
    fn a_device_path_that_drops_the_logits_fails_loudly() {
        let mut s = Session::new(DeviceGreedy::new(5));
        let err = collect(&mut s, &[7931, 7932], &greedy(2), Some(2))
            .expect_err("asked for logits, got none");
        assert!(err.to_string().contains("returned none"), "{err}");
    }

    #[test]
    fn a_penalized_step_reports_the_logits_after_the_penalty() {
        let mut config = greedy(2);
        config.repeat_penalty = 2.0;
        let mut s = Session::new(Scripted::new(3, 100));
        let (_, records) = collect(&mut s, &[7941], &config, Some(1)).expect("turn");
        let logit = |i: usize| records[i].as_ref().expect("record").top[0].logit;
        // Step 0: token 3 is not yet in the context; step 1: it is, halved.
        assert_eq!(logit(0), 1.0);
        assert_eq!(logit(1), 0.5);
    }
}
