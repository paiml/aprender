//! wasm32 applier for `browser::video_ops`: executes the operations against real `<video>`
//! elements positioned over the canvas (aprender#6890). It decides nothing.

use super::video_ops::{reconcile, VideoNode, VideoOp};
use std::collections::HashMap;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, HtmlElement, HtmlVideoElement};

/// The `<video>` elements of one canvas.
pub struct VideoLayer {
    canvas: HtmlCanvasElement,
    elements: HashMap<String, HtmlVideoElement>,
    previous: Vec<VideoNode>,
}

impl VideoLayer {
    /// A layer over `canvas`; elements are added next to it in the DOM.
    #[must_use]
    pub fn new(canvas: HtmlCanvasElement) -> Self {
        Self {
            canvas,
            elements: HashMap::new(),
            previous: Vec::new(),
        }
    }

    /// Make the elements match `nodes`.
    ///
    /// # Errors
    /// The first DOM call that fails; the remaining operations of this call are not run.
    pub fn sync(&mut self, nodes: &[VideoNode]) -> Result<(), JsValue> {
        for op in reconcile(&self.previous, nodes) {
            self.run(op)?;
        }
        self.previous = nodes.to_vec();
        Ok(())
    }

    /// `currentTime` in seconds of the element for `id`, if there is one.
    #[must_use]
    pub fn current_time(&self, id: &str) -> Option<f64> {
        self.elements.get(id).map(|el| el.current_time())
    }

    /// Whether the element for `id` is paused, if there is one.
    #[must_use]
    pub fn paused(&self, id: &str) -> Option<bool> {
        self.elements.get(id).map(|el| el.paused())
    }

    /// Number of live elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether there are no live elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    fn place(&self, el: &HtmlVideoElement, rect: [f32; 4]) -> Result<(), JsValue> {
        let [x, y, w, h] = rect;
        let left = f64::from(self.canvas.offset_left()) + f64::from(x);
        let top = f64::from(self.canvas.offset_top()) + f64::from(y);
        el.set_attribute(
            "style",
            &format!("position:absolute;left:{left}px;top:{top}px;width:{w}px;height:{h}px"),
        )
    }

    fn run(&mut self, op: VideoOp) -> Result<(), JsValue> {
        match op {
            VideoOp::Create {
                id,
                rect,
                src,
                muted,
            } => {
                let document = self
                    .canvas
                    .owner_document()
                    .ok_or("canvas has no document")?;
                let el: HtmlVideoElement = document.create_element("video")?.dyn_into()?;
                el.set_src(&src);
                el.set_muted(muted);
                self.place(&el, rect)?;
                let parent = self.canvas.parent_node().ok_or("canvas has no parent")?;
                parent.insert_before(&el, self.canvas.next_sibling().as_ref())?;
                self.elements.insert(id, el);
            }
            VideoOp::Move { id, rect } => {
                if let Some(el) = self.elements.get(&id) {
                    self.place(el, rect)?;
                }
            }
            VideoOp::SetSrc { id, src } => {
                if let Some(el) = self.elements.get(&id) {
                    el.set_src(&src);
                }
            }
            VideoOp::SetMuted { id, muted } => {
                if let Some(el) = self.elements.get(&id) {
                    el.set_muted(muted);
                }
            }
            VideoOp::Play { id } => {
                if let Some(el) = self.elements.get(&id) {
                    // The promise settles later; the state is read back from `paused()`.
                    let _ = el.play()?;
                }
            }
            VideoOp::Pause { id } => {
                if let Some(el) = self.elements.get(&id) {
                    el.pause()?;
                }
            }
            VideoOp::Remove { id } => {
                if let Some(el) = self.elements.remove(&id) {
                    let el: &HtmlElement = el.as_ref();
                    el.remove();
                }
            }
        }
        Ok(())
    }
}
