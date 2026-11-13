//! Which `<video>` elements a frame needs, as a list of operations (aprender#6890).
//!
//! `reconcile` is the whole decision: it takes the video nodes of the previous frame and of
//! this one and returns the operations that turn the first set of DOM elements into the
//! second. It touches no browser API, so it is tested natively; the wasm32 applier in
//! `browser::video_dom` only executes the list.
//!
//! Contract: `contracts/present-video-element-v1.yaml`.

use presentar_widgets::{Video, VideoState};
use std::collections::HashSet;

/// One video widget as the DOM sees it: a key, a rectangle over the canvas and its state.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoNode {
    /// Stable key of the widget; the same key in two frames is the same element.
    pub id: String,
    /// Laid-out rectangle `[x, y, width, height]` in canvas pixels.
    pub rect: [f32; 4],
    /// Source URI.
    pub src: String,
    /// Whether the widget is playing.
    pub playing: bool,
    /// Whether the element is muted.
    pub muted: bool,
}

impl VideoNode {
    /// The node of a laid-out `Video` widget. A non-finite or negative laid-out value
    /// becomes 0, so a bad rectangle never reaches a style string.
    #[must_use]
    pub fn from_video(id: impl Into<String>, video: &Video, bounds: [f32; 4], muted: bool) -> Self {
        let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
        Self {
            id: id.into(),
            rect: [
                clean(bounds[0]),
                clean(bounds[1]),
                clean(bounds[2]).max(0.0),
                clean(bounds[3]).max(0.0),
            ],
            src: video.get_src().to_owned(),
            playing: matches!(video.state(), VideoState::Playing),
            muted,
        }
    }

    fn rect_bits(&self) -> [u32; 4] {
        self.rect.map(f32::to_bits)
    }
}

/// One DOM operation on a video element.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoOp {
    /// Add an element at `rect` with `src` and `muted`.
    Create {
        /// Element key.
        id: String,
        /// Rectangle in canvas pixels.
        rect: [f32; 4],
        /// Source URI.
        src: String,
        /// Muted flag.
        muted: bool,
    },
    /// Reposition or resize the element.
    Move {
        /// Element key.
        id: String,
        /// New rectangle.
        rect: [f32; 4],
    },
    /// Change the source.
    SetSrc {
        /// Element key.
        id: String,
        /// New source URI.
        src: String,
    },
    /// Change the muted flag.
    SetMuted {
        /// Element key.
        id: String,
        /// New muted flag.
        muted: bool,
    },
    /// Start playback.
    Play {
        /// Element key.
        id: String,
    },
    /// Pause playback.
    Pause {
        /// Element key.
        id: String,
    },
    /// Remove the element.
    Remove {
        /// Element key.
        id: String,
    },
}

/// The first node of each key, in order. A repeated key is one element, not two.
fn unique(nodes: &[VideoNode]) -> Vec<&VideoNode> {
    let mut seen = HashSet::new();
    nodes
        .iter()
        .filter(|n| seen.insert(n.id.as_str()))
        .collect()
}

/// The operations that turn the elements of `previous` into the elements of `current`.
///
/// Removes come first (previous order), then per current node in order: a new key is one
/// `Create` (plus `Play` if it is playing); a kept key gets only the operations for what
/// changed, in the order `Move`, `SetSrc`, `SetMuted`, then `Play`/`Pause`. A src change on
/// a playing node is followed by `Play` (the browser pauses on `SetSrc`). Equal sets yield
/// no operation.
#[must_use]
pub fn reconcile(previous: &[VideoNode], current: &[VideoNode]) -> Vec<VideoOp> {
    let previous = unique(previous);
    let current = unique(current);
    let kept: HashSet<&str> = current.iter().map(|n| n.id.as_str()).collect();
    let mut ops = Vec::new();
    for old in &previous {
        if !kept.contains(old.id.as_str()) {
            ops.push(VideoOp::Remove { id: old.id.clone() });
        }
    }
    for new in &current {
        let id = new.id.clone();
        let Some(old) = previous.iter().find(|o| o.id == new.id) else {
            ops.push(VideoOp::Create {
                id: id.clone(),
                rect: new.rect,
                src: new.src.clone(),
                muted: new.muted,
            });
            if new.playing {
                ops.push(VideoOp::Play { id });
            }
            continue;
        };
        if old.rect_bits() != new.rect_bits() {
            ops.push(VideoOp::Move {
                id: id.clone(),
                rect: new.rect,
            });
        }
        if old.src != new.src {
            ops.push(VideoOp::SetSrc {
                id: id.clone(),
                src: new.src.clone(),
            });
        }
        if old.muted != new.muted {
            ops.push(VideoOp::SetMuted {
                id: id.clone(),
                muted: new.muted,
            });
        }
        // Setting `src` runs the media load algorithm, which leaves the element paused. So a
        // playing node whose src changed needs a `Play` after the `SetSrc`, even though its
        // `playing` flag did not change. The play/pause op always comes last.
        let src_reset_playback = old.src != new.src && new.playing;
        if old.playing != new.playing || src_reset_playback {
            ops.push(if new.playing {
                VideoOp::Play { id }
            } else {
                VideoOp::Pause { id }
            });
        }
    }
    ops
}

/// Update the model of the DOM (`previous`) for one operation that the applier has just
/// executed. Mirrors the browser: `SetSrc` runs the media load algorithm, which leaves the
/// element paused, so the node is recorded as not playing until a `Play` follows.
fn advance(previous: &mut Vec<VideoNode>, op: &VideoOp) {
    let find = |previous: &mut Vec<VideoNode>, id: &str| previous.iter().position(|n| n.id == id);
    match op {
        VideoOp::Create {
            id,
            rect,
            src,
            muted,
        } => previous.push(VideoNode {
            id: id.clone(),
            rect: *rect,
            src: src.clone(),
            playing: false,
            muted: *muted,
        }),
        VideoOp::Remove { id } => previous.retain(|n| n.id != *id),
        VideoOp::Move { id, rect } => {
            if let Some(i) = find(previous, id) {
                previous[i].rect = *rect;
            }
        }
        VideoOp::SetSrc { id, src } => {
            if let Some(i) = find(previous, id) {
                previous[i].src = src.clone();
                previous[i].playing = false;
            }
        }
        VideoOp::SetMuted { id, muted } => {
            if let Some(i) = find(previous, id) {
                previous[i].muted = *muted;
            }
        }
        VideoOp::Play { id } => {
            if let Some(i) = find(previous, id) {
                previous[i].playing = true;
            }
        }
        VideoOp::Pause { id } => {
            if let Some(i) = find(previous, id) {
                previous[i].playing = false;
            }
        }
    }
}

/// Make the DOM match `target`, recording progress op by op.
///
/// Runs `reconcile(previous, target)` through `apply`. After each operation succeeds,
/// `previous` is advanced to describe the DOM as it now is, so when `apply` fails partway
/// `previous` is exactly the state of the applied prefix and a retry only runs the
/// remainder (no element is created twice and orphaned). On success `previous` is `target`.
///
/// # Errors
/// The first error of `apply`; the operations after it are not run.
pub fn sync_with<E>(
    previous: &mut Vec<VideoNode>,
    target: &[VideoNode],
    mut apply: impl FnMut(&VideoOp) -> Result<(), E>,
) -> Result<(), E> {
    for op in reconcile(previous, target) {
        apply(&op)?;
        advance(previous, &op);
    }
    *previous = target.to_vec();
    Ok(())
}

/// The CSS `style` of an element laid over the canvas: `rect` in canvas pixels plus the
/// canvas `origin` in the page. The absolute position is against the element's containing
/// block, so the canvas's offset parent must be that block (a positioned ancestor).
/// A non-finite value becomes 0 and a negative size becomes 0, whatever the caller built.
#[must_use]
pub fn style_for(rect: [f32; 4], origin: [f64; 2]) -> String {
    let clean = |v: f64| if v.is_finite() { v } else { 0.0 };
    let [x, y, w, h] = rect.map(|v| clean(f64::from(v)));
    let left = clean(origin[0]) + x;
    let top = clean(origin[1]) + y;
    let (w, h) = (w.max(0.0), h.max(0.0));
    format!("position:absolute;left:{left}px;top:{top}px;width:{w}px;height:{h}px")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn node(id: &str) -> VideoNode {
        VideoNode {
            id: id.into(),
            rect: [1.0, 2.0, 30.0, 40.0],
            src: format!("{id}.mp4"),
            playing: false,
            muted: false,
        }
    }

    /// A model DOM: key -> (rect bits, src, playing, muted).
    type Dom = BTreeMap<String, ([u32; 4], String, bool, bool)>;

    fn apply(dom: &mut Dom, ops: &[VideoOp]) {
        for op in ops {
            match op {
                VideoOp::Create {
                    id,
                    rect,
                    src,
                    muted,
                } => {
                    dom.insert(
                        id.clone(),
                        (rect.map(f32::to_bits), src.clone(), false, *muted),
                    );
                }
                VideoOp::Move { id, rect } => {
                    dom.get_mut(id).expect("move target").0 = rect.map(f32::to_bits)
                }
                VideoOp::SetSrc { id, src } => {
                    // Browser behaviour: setting src runs the media load algorithm,
                    // which leaves the element paused.
                    let e = dom.get_mut(id).expect("src target");
                    e.1 = src.clone();
                    e.2 = false;
                }
                VideoOp::SetMuted { id, muted } => dom.get_mut(id).expect("mute target").3 = *muted,
                VideoOp::Play { id } => dom.get_mut(id).expect("play target").2 = true,
                VideoOp::Pause { id } => dom.get_mut(id).expect("pause target").2 = false,
                VideoOp::Remove { id } => {
                    dom.remove(id).expect("remove target");
                }
            }
        }
    }

    fn dom_of(nodes: &[VideoNode]) -> Dom {
        let mut dom = Dom::new();
        apply(&mut dom, &reconcile(&[], nodes));
        dom
    }

    #[test]
    fn nothing_changed_yields_no_op() {
        let set = vec![node("a"), node("b")];
        assert!(reconcile(&set, &set).is_empty());
        assert!(reconcile(&[], &[]).is_empty());
    }

    #[test]
    fn appearing_widget_is_one_create_and_playing_adds_play() {
        let mut p = node("a");
        p.playing = true;
        p.muted = true;
        let ops = reconcile(&[], &[p.clone()]);
        assert_eq!(
            ops,
            vec![
                VideoOp::Create {
                    id: "a".into(),
                    rect: p.rect,
                    src: "a.mp4".into(),
                    muted: true
                },
                VideoOp::Play { id: "a".into() },
            ]
        );
        assert_eq!(reconcile(&[], &[node("a")]).len(), 1);
    }

    #[test]
    fn removed_widget_is_exactly_one_remove() {
        let ops = reconcile(&[node("a"), node("b")], &[node("a")]);
        assert_eq!(ops, vec![VideoOp::Remove { id: "b".into() }]);
    }

    #[test]
    fn each_change_maps_to_its_op() {
        let old = node("a");
        let mut moved = old.clone();
        moved.rect[0] = 9.0;
        assert_eq!(
            reconcile(&[old.clone()], &[moved.clone()]),
            vec![VideoOp::Move {
                id: "a".into(),
                rect: moved.rect
            }]
        );
        let mut s = old.clone();
        s.src = "z.mp4".into();
        assert_eq!(
            reconcile(&[old.clone()], &[s]),
            vec![VideoOp::SetSrc {
                id: "a".into(),
                src: "z.mp4".into()
            }]
        );
        let mut m = old.clone();
        m.muted = true;
        assert_eq!(
            reconcile(&[old.clone()], &[m]),
            vec![VideoOp::SetMuted {
                id: "a".into(),
                muted: true
            }]
        );
        let mut pl = old.clone();
        pl.playing = true;
        assert_eq!(
            reconcile(&[old.clone()], &[pl.clone()]),
            vec![VideoOp::Play { id: "a".into() }]
        );
        assert_eq!(
            reconcile(&[pl], &[old]),
            vec![VideoOp::Pause { id: "a".into() }]
        );
    }

    #[test]
    fn model_dom_follows_every_kind_of_change() {
        let old = node("a");
        let mut new = old.clone();
        new.rect[1] = 7.0;
        new.src = "z.mp4".into();
        new.muted = true;
        new.playing = true;
        let mut dom = dom_of(std::slice::from_ref(&old));
        apply(
            &mut dom,
            &reconcile(std::slice::from_ref(&old), std::slice::from_ref(&new)),
        );
        assert_eq!(dom, dom_of(std::slice::from_ref(&new)));
        apply(
            &mut dom,
            &reconcile(std::slice::from_ref(&new), std::slice::from_ref(&old)),
        );
        assert_eq!(dom, dom_of(std::slice::from_ref(&old)));
    }

    #[test]
    fn play_pause_are_idempotent() {
        let mut pl = node("a");
        pl.playing = true;
        assert!(reconcile(&[pl.clone()], &[pl.clone()]).is_empty());
        let paused = node("a");
        assert!(reconcile(&[paused.clone()], &[paused]).is_empty());
        // applying the same target twice leaves the DOM as after the first time
        let mut dom = dom_of(&[node("a")]);
        let target = vec![pl.clone()];
        apply(&mut dom, &reconcile(&[node("a")], &target));
        let once = dom.clone();
        apply(&mut dom, &reconcile(&target, &target));
        assert_eq!(dom, once);
    }

    #[test]
    fn nan_rect_is_stable_not_a_move_every_frame() {
        let mut n = node("a");
        n.rect[2] = f32::NAN;
        assert!(reconcile(&[n.clone()], &[n]).is_empty());
    }

    #[test]
    fn duplicate_keys_are_one_element() {
        let ops = reconcile(&[], &[node("a"), node("a")]);
        assert_eq!(ops.len(), 1);
        assert_eq!(dom_of(&[node("a"), node("a")]).len(), 1);
    }

    #[test]
    fn element_count_equals_widget_count_after_any_transition() {
        let a = vec![node("a"), node("b"), node("c")];
        let b = vec![node("c"), node("d")];
        let mut dom = dom_of(&a);
        apply(&mut dom, &reconcile(&a, &b));
        assert_eq!(dom.len(), b.len());
        assert_eq!(dom, dom_of(&b));
        apply(&mut dom, &reconcile(&b, &[]));
        assert!(dom.is_empty());
    }

    #[test]
    fn from_video_reads_state_and_sanitises_bounds() {
        let mut v = Video::new("clip.mp4");
        let n = VideoNode::from_video("v", &v, [1.0, f32::NAN, -5.0, f32::INFINITY], true);
        assert_eq!(
            (n.rect, n.src.as_str(), n.playing, n.muted),
            ([1.0, 0.0, 0.0, 0.0], "clip.mp4", false, true)
        );
        v.play();
        assert!(VideoNode::from_video("v", &v, [0.0; 4], false).playing);
    }

    #[test]
    fn src_change_on_playing_video_is_followed_by_play() {
        let mut old = node("a");
        old.playing = true;
        let mut new = old.clone();
        new.src = "z.mp4".into();
        let ops = reconcile(&[old.clone()], &[new.clone()]);
        assert_eq!(
            ops,
            vec![
                VideoOp::SetSrc {
                    id: "a".into(),
                    src: "z.mp4".into()
                },
                VideoOp::Play { id: "a".into() },
            ]
        );
        // the model DOM pauses on SetSrc, so Play must restore the playing state
        let mut dom = dom_of(&[old.clone()]);
        apply(&mut dom, &reconcile(&[old], &[new.clone()]));
        assert_eq!(dom, dom_of(&[new]));
    }

    #[test]
    fn src_change_with_play_toggle_sends_one_play_and_paused_sends_none() {
        let mut old = node("a");
        let mut new = old.clone();
        new.src = "z.mp4".into();
        new.playing = true;
        let plays = |ops: &[VideoOp]| {
            ops.iter()
                .filter(|o| matches!(o, VideoOp::Play { .. }))
                .count()
        };
        assert_eq!(plays(&reconcile(&[old.clone()], &[new.clone()])), 1);
        // paused stays paused across a src change: no Play
        new.playing = false;
        assert_eq!(plays(&reconcile(&[old.clone()], &[new.clone()])), 0);
        // playing to paused with a src change: Pause, no Play
        old.playing = true;
        let ops = reconcile(&[old], &[new]);
        assert_eq!(plays(&ops), 0);
        assert!(ops.contains(&VideoOp::Pause { id: "a".into() }));
    }

    /// Run `sync_with` against a model DOM whose applier fails on the `fail_at`-th op
    /// (0-based) of this call; returns the result.
    fn sync_failing(
        dom: &mut Dom,
        previous: &mut Vec<VideoNode>,
        target: &[VideoNode],
        fail_at: Option<usize>,
    ) -> Result<(), &'static str> {
        let mut n = 0;
        sync_with(previous, target, |op| {
            let i = n;
            n += 1;
            if Some(i) == fail_at {
                return Err("boom");
            }
            apply(dom, std::slice::from_ref(op));
            Ok(())
        })
    }

    #[test]
    fn partial_failure_keeps_previous_consistent_with_the_dom() {
        let target = vec![node("a"), node("b"), node("c")];
        let mut dom = Dom::new();
        let mut previous = Vec::new();
        // Create a, Create b applied; Create c fails
        assert_eq!(
            sync_failing(&mut dom, &mut previous, &target, Some(2)),
            Err("boom")
        );
        assert_eq!(dom.len(), 2);
        assert_eq!(
            previous.len(),
            2,
            "previous records exactly the applied ops"
        );
        // retry: only c is created, nothing is created twice, the DOM converges
        let mut n = 0;
        sync_with(&mut previous, &target, |op| {
            n += 1;
            assert_eq!(
                *op,
                VideoOp::Create {
                    id: "c".into(),
                    rect: target[2].rect,
                    src: "c.mp4".into(),
                    muted: false
                }
            );
            apply(&mut dom, std::slice::from_ref(op));
            Ok::<(), ()>(())
        })
        .expect("retry succeeds");
        assert_eq!(n, 1);
        assert_eq!(dom, dom_of(&target));
        assert_eq!(previous, target);
    }

    #[test]
    fn failure_after_src_change_retries_play() {
        let mut old = node("a");
        old.playing = true;
        let mut new = old.clone();
        new.src = "z.mp4".into();
        let mut dom = dom_of(&[old.clone()]);
        let mut previous = vec![old];
        // SetSrc applies, Play fails
        assert_eq!(
            sync_failing(&mut dom, &mut previous, &[new.clone()], Some(1)),
            Err("boom")
        );
        assert!(!previous[0].playing, "previous knows the element is paused");
        assert_eq!(
            sync_failing(&mut dom, &mut previous, &[new.clone()], None),
            Ok(())
        );
        assert_eq!(dom, dom_of(&[new]));
    }

    #[test]
    fn partial_failure_property_any_failure_point_converges() {
        let a = vec![node("a"), node("b")];
        let mut b2 = node("b");
        b2.src = "q.mp4".into();
        b2.playing = true;
        b2.rect[0] = 5.0;
        let b = vec![b2, node("c"), node("d")];
        let total = reconcile(&a, &b).len();
        for k in 0..total {
            let mut dom = dom_of(&a);
            let mut previous = a.clone();
            assert!(sync_failing(&mut dom, &mut previous, &b, Some(k)).is_err());
            assert_eq!(sync_failing(&mut dom, &mut previous, &b, None), Ok(()));
            assert_eq!(dom, dom_of(&b), "fail point {k}");
        }
    }

    #[test]
    fn style_is_sanitised_and_offset() {
        assert_eq!(
            style_for([1.0, 2.0, 30.0, 40.0], [10.0, 20.0]),
            "position:absolute;left:11px;top:22px;width:30px;height:40px"
        );
        let s = style_for([f32::NAN, f32::INFINITY, -4.0, f32::NAN], [0.0, 0.0]);
        assert_eq!(s, "position:absolute;left:0px;top:0px;width:0px;height:0px");
        assert!(!s.contains("NaN") && !s.contains("inf"));
    }
}
