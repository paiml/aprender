//! wasm32 applier for `browser::video_ops`: executes the operations against real `<video>`
//! elements positioned over the canvas (aprender#6890). It decides nothing.

use super::video_ops::{style_for, sync_with, VideoNode, VideoOp};
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
    /// Progress is recorded op by op (`video_ops::sync_with`), so after a failure the layer
    /// still describes the DOM as it is and the next `sync` runs only the remainder.
    ///
    /// Play failures: `play()` returns a Promise that settles later (autoplay policy can
    /// reject it); a rejection is not seen here and shows only as `paused() == Some(true)`.
    ///
    /// # Errors
    /// The first DOM call that fails (including an operation on an id the layer does not
    /// hold, which means layer and model diverged); the remaining operations of this call
    /// are not run.
    pub fn sync(&mut self, nodes: &[VideoNode]) -> Result<(), JsValue> {
        let mut previous = std::mem::take(&mut self.previous);
        let result = sync_with(&mut previous, nodes, |op| self.run(op));
        self.previous = previous;
        result
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
        let origin = [
            f64::from(self.canvas.offset_left()),
            f64::from(self.canvas.offset_top()),
        ];
        el.set_attribute("style", &style_for(rect, origin))
    }

    fn element(&self, id: &str) -> Result<&HtmlVideoElement, JsValue> {
        self.elements
            .get(id)
            .ok_or_else(|| JsValue::from_str(&format!("no video element for id {id}")))
    }

    fn run(&mut self, op: &VideoOp) -> Result<(), JsValue> {
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
                el.set_src(src);
                el.set_muted(*muted);
                self.place(&el, *rect)?;
                let parent = self.canvas.parent_node().ok_or("canvas has no parent")?;
                parent.insert_before(&el, self.canvas.next_sibling().as_ref())?;
                // Recorded only once attached, so a failed Create leaves nothing behind.
                self.elements.insert(id.clone(), el);
            }
            VideoOp::Move { id, rect } => self.place(self.element(id)?, *rect)?,
            VideoOp::SetSrc { id, src } => self.element(id)?.set_src(src),
            VideoOp::SetMuted { id, muted } => self.element(id)?.set_muted(*muted),
            VideoOp::Play { id } => {
                // The promise settles later; the state is read back from `paused()`.
                let _ = self.element(id)?.play()?;
            }
            VideoOp::Pause { id } => self.element(id)?.pause()?,
            VideoOp::Remove { id } => {
                let el = self
                    .elements
                    .remove(id)
                    .ok_or_else(|| JsValue::from_str(&format!("no video element for id {id}")))?;
                let el: &HtmlElement = el.as_ref();
                el.remove();
            }
        }
        Ok(())
    }
}

impl Drop for VideoLayer {
    /// Removes every element the layer added, so none outlives it (still playing).
    fn drop(&mut self) {
        for (_, el) in self.elements.drain() {
            let el: &HtmlElement = el.as_ref();
            el.remove();
        }
    }
}
