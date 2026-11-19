//! Page driven by `tests/video_e2e_chrome.rs` (aprender#7261): Video widgets in a Grid and in a
//! Modal, mounted as real `<video>` elements by `VideoLayer`, with play/pause and read-back
//! exported to the page. Build: `cargo build --example video_e2e --target wasm32-unknown-unknown`.
#![cfg(target_arch = "wasm32")]

use aprender_present_lib::browser::{VideoLayer, VideoNode};
use aprender_present_lib::widgets::{Grid, Modal, Video};
use aprender_present_lib::{Rect, Widget};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

struct Page {
    layer: VideoLayer,
    videos: Vec<(String, Video, [f32; 4])>,
    modal: Modal,
}

thread_local! { static PAGE: RefCell<Option<Page>> = const { RefCell::new(None) }; }

fn clip() -> Video {
    Video::new("clip.webm").size(160.0, 120.0)
}

fn rect4(r: Rect) -> [f32; 4] {
    [r.x, r.y, r.width, r.height]
}

impl Page {
    fn sync(&mut self) -> Result<(), JsValue> {
        let open = self.modal.is_open();
        let nodes: Vec<VideoNode> = self
            .videos
            .iter()
            .filter(|(id, ..)| open || id != "modal")
            .map(|(id, v, r)| VideoNode::from_video(id.clone(), v, *r, true))
            .collect();
        self.layer.sync(&nodes)
    }
}

#[wasm_bindgen(start)]
fn start() -> Result<(), JsValue> {
    let canvas: HtmlCanvasElement = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("c"))
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
    let mut videos: Vec<(String, Video, [f32; 4])> = (0..2)
        .map(|i| (format!("grid-{i}"), clip(), rect4(cells[i])))
        .collect();
    videos.push(("modal".into(), clip(), rect4(modal.content_bounds())));
    let mut page = Page {
        layer: VideoLayer::new(canvas),
        videos,
        modal,
    };
    page.sync()?;
    PAGE.with(|p| *p.borrow_mut() = Some(page));
    Ok(())
}

fn with_page<T>(f: impl FnOnce(&mut Page) -> Result<T, JsValue>) -> Result<T, JsValue> {
    PAGE.with(|p| f(p.borrow_mut().as_mut().ok_or("page not started")?))
}

/// Toggle the widget `id` between playing and paused and sync the elements.
#[wasm_bindgen]
pub fn e2e_toggle(id: &str) -> Result<(), JsValue> {
    with_page(|p| {
        let (_, v, _) = p
            .videos
            .iter_mut()
            .find(|(i, ..)| i == id)
            .ok_or("no such video")?;
        v.toggle();
        p.sync()
    })
}

/// Show or hide the modal and sync the elements.
#[wasm_bindgen]
pub fn e2e_modal(open: bool) -> Result<(), JsValue> {
    with_page(|p| {
        if open {
            p.modal.show()
        } else {
            p.modal.hide()
        }
        p.sync()
    })
}

/// `{"paused":bool,"time":f64}` read back from the element for `id`, or `null` when none.
#[wasm_bindgen]
pub fn e2e_state(id: &str) -> Result<String, JsValue> {
    with_page(|p| {
        Ok(match (p.layer.paused(id), p.layer.current_time(id)) {
            (Some(paused), Some(time)) => format!("{{\"paused\":{paused},\"time\":{time}}}"),
            _ => "null".into(),
        })
    })
}

fn main() {}
