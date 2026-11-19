//! Page driven by `tests/video_e2e_chrome.rs` (aprender#7261): Video widgets in a Grid and in a
//! Modal, mounted as `<video>` elements by `VideoLayer`, with play/pause and read-back exported.
#![cfg(target_arch = "wasm32")]

use aprender_present_lib::browser::{VideoLayer, VideoNode};
use aprender_present_lib::widgets::{Grid, Modal, Video};
use aprender_present_lib::{Rect, Widget};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen]
pub struct E2e {
    layer: VideoLayer,
    videos: Vec<(String, Video, [f32; 4])>,
}

fn clip() -> Video {
    Video::new("clip.webm").size(160.0, 120.0)
}

fn quad(r: Rect) -> [f32; 4] {
    [r.x, r.y, r.width, r.height]
}

#[wasm_bindgen]
impl E2e {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<E2e, JsValue> {
        let canvas: HtmlCanvasElement = web_sys::window()
            .and_then(|w| w.document()?.get_element_by_id("c"))
            .ok_or("no canvas")?
            .dyn_into()?;
        let grid = Grid::new(2)
            .gap(8.0)
            .row_height(120.0)
            .child(clip())
            .child(clip());
        let cells = grid.cells(Rect::new(0.0, 0.0, 800.0, 130.0));
        let mut modal = Modal::new().content(clip());
        modal.show();
        modal.layout(Rect::new(0.0, 0.0, 800.0, 600.0));
        let videos = vec![
            ("grid-0".to_string(), clip(), quad(cells[0])),
            ("grid-1".to_string(), clip(), quad(cells[1])),
            ("modal".to_string(), clip(), quad(modal.content_bounds())),
        ];
        let mut page = E2e {
            layer: VideoLayer::new(canvas),
            videos,
        };
        page.sync()?;
        Ok(page)
    }

    fn sync(&mut self) -> Result<(), JsValue> {
        let nodes: Vec<VideoNode> = (self.videos.iter())
            .map(|(id, v, r)| VideoNode::from_video(id.clone(), v, *r, true))
            .collect();
        self.layer.sync(&nodes)
    }

    /// Toggle the widget `id` between playing and paused.
    pub fn toggle(&mut self, id: &str) -> Result<(), JsValue> {
        self.videos
            .iter_mut()
            .find(|(i, ..)| i == id)
            .ok_or("no such video")?
            .1
            .toggle();
        self.sync()
    }

    /// `{"paused":..,"time":..}` read back from the element for `id`; `null` when none.
    pub fn state(&self, id: &str) -> String {
        match (self.layer.paused(id), self.layer.current_time(id)) {
            (Some(p), Some(t)) => format!("{{\"paused\":{p},\"time\":{t}}}"),
            _ => "null".into(),
        }
    }
}

fn main() {}
