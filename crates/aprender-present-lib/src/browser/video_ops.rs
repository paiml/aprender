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
/// changed. Equal sets yield no operation.
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
        if old.playing != new.playing {
            ops.push(if new.playing {
                VideoOp::Play { id }
            } else {
                VideoOp::Pause { id }
            });
        }
    }
    ops
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
                VideoOp::SetSrc { id, src } => dom.get_mut(id).expect("src target").1 = src.clone(),
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
}
