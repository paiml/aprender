//! wasm32 shell around `browser::mount`: DOM listeners and the `requestAnimationFrame`
//! loop, registered from Rust, so the page carries no script of its own (aprender#6893,
//! aprender#6892).
//!
//! Every decision (does an event owe a repaint, does the loop go on) is taken by
//! `Mounted`; this file only moves browser events in and draw commands out.

use super::canvas2d::Canvas2DRenderer;
use super::events::{keyboard_event_to_presentar, mouse_event_to_presentar};
use super::mount::{counter_tree, Frame, Mounted};
use presentar_core::{Event, Widget};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{window, EventTarget, HtmlCanvasElement, KeyboardEvent, MouseEvent};

/// Called once per widget message, with the mounted tree to change in response.
pub type MessageHandler = Box<dyn FnMut(&mut Mounted, Box<dyn Any + Send>)>;

/// The id of the canvas the built-in counter page mounts on.
pub const COUNTER_CANVAS_ID: &str = "presentar-counter";
/// The canvas attribute the counter page writes its state to, for a test to read back.
pub const COUNT_ATTRIBUTE: &str = "data-count";

struct Shared {
    mounted: RefCell<Mounted>,
    handler: RefCell<MessageHandler>,
    renderer: Canvas2DRenderer,
    running: Cell<bool>,
    pending: Cell<Option<i32>>,
    tick: RefCell<Option<Closure<dyn FnMut(f64)>>>,
}

impl Shared {
    /// Ask the browser for one frame. `running` stays false if it refuses.
    fn request(&self) {
        let id = window().and_then(|w| {
            let tick = self.tick.borrow();
            let tick = tick.as_ref()?;
            w.request_animation_frame(tick.as_ref().unchecked_ref())
                .ok()
        });
        self.pending.set(id);
        self.running.set(id.is_some());
    }

    /// Start the loop if it is stopped and a frame is owed.
    fn kick(&self) {
        if !self.running.get() && self.mounted.borrow().needs_frame() {
            self.request();
        }
    }

    fn on_tick(&self) {
        let frame = self.mounted.borrow_mut().frame();
        match &frame {
            Frame::Painted(commands) => {
                self.renderer.clear();
                self.renderer.render(commands);
            }
            Frame::Blocked(why) => {
                web_sys::console::error_1(&JsValue::from_str(&format!("JIDOKA: {why}")));
            }
            Frame::Idle => {}
        }
        if frame.wants_next() {
            self.request();
        } else {
            self.pending.set(None);
            self.running.set(false);
        }
    }

    fn on_event(&self, event: &Event) {
        let mut mounted = self.mounted.borrow_mut();
        mounted.dispatch(event);
        let mut handler = self.handler.borrow_mut();
        for message in mounted.take_messages() {
            handler(&mut mounted, message);
        }
        drop(handler);
        drop(mounted);
        self.kick();
    }
}

enum Listener {
    Mouse(Closure<dyn FnMut(MouseEvent)>),
    Key(Closure<dyn FnMut(KeyboardEvent)>),
}

impl Listener {
    fn function(&self) -> &js_sys::Function {
        match self {
            Self::Mouse(closure) => closure.as_ref().unchecked_ref(),
            Self::Key(closure) => closure.as_ref().unchecked_ref(),
        }
    }
}

/// A mounted app. Dropping it removes the listeners and cancels the pending frame;
/// `leak` keeps it for the life of the page.
pub struct MountHandle {
    shared: Rc<Shared>,
    listeners: Vec<(EventTarget, &'static str, Listener)>,
}

impl MountHandle {
    /// Keep the app mounted for the life of the page (what a start function wants).
    pub fn leak(self) {
        std::mem::forget(self);
    }

    /// Change the mounted tree from outside an event (a timer, a socket), then repaint.
    /// A repaint is always requested, so an in-place `root_mut()` change cannot be dropped.
    pub fn update(&self, change: impl FnOnce(&mut Mounted)) {
        {
            let mut mounted = self.shared.mounted.borrow_mut();
            change(&mut mounted);
            mounted.request_repaint();
        }
        self.shared.kick();
    }
}

impl Drop for MountHandle {
    fn drop(&mut self) {
        for (target, name, listener) in &self.listeners {
            let _ = target.remove_event_listener_with_callback(name, listener.function());
        }
        if let (Some(id), Some(w)) = (self.shared.pending.take(), window()) {
            let _ = w.cancel_animation_frame(id);
        }
        // The tick closure holds the only other Rc to `shared`; dropping it frees both.
        self.shared.tick.borrow_mut().take();
    }
}

/// Mount `root` on the canvas with id `canvas_id`. Mouse and keyboard listeners and the
/// frame loop are registered here, in Rust; `handler` receives every widget message.
///
/// # Errors
/// When there is no window or document, the element is missing or is not a canvas, the
/// 2D context cannot be made, or a listener cannot be added.
pub fn mount(
    canvas_id: &str,
    root: Box<dyn Widget>,
    handler: MessageHandler,
) -> Result<MountHandle, JsValue> {
    console_error_panic_hook::set_once();
    let document = window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| format!("canvas '{canvas_id}' not found"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| "element is not a canvas")?;
    let (width, height) = (canvas.width() as f32, canvas.height() as f32);
    let renderer = Canvas2DRenderer::new(canvas.clone()).map_err(|e| JsValue::from_str(&e))?;

    let shared = Rc::new(Shared {
        mounted: RefCell::new(Mounted::new(root, width, height)),
        handler: RefCell::new(handler),
        renderer,
        running: Cell::new(false),
        pending: Cell::new(None),
        tick: RefCell::new(None),
    });
    let for_tick = Rc::clone(&shared);
    *shared.tick.borrow_mut() = Some(Closure::new(move |_now: f64| for_tick.on_tick()));

    let mut listeners = Vec::new();
    let canvas_target: EventTarget = canvas.into();
    for name in [
        "mousedown",
        "mouseup",
        "mousemove",
        "mouseenter",
        "mouseleave",
    ] {
        let for_event = Rc::clone(&shared);
        let listener = Listener::Mouse(Closure::new(move |e: MouseEvent| {
            for_event.on_event(&mouse_event_to_presentar(&e, name));
        }));
        canvas_target.add_event_listener_with_callback(name, listener.function())?;
        listeners.push((canvas_target.clone(), name, listener));
    }
    let document_target: EventTarget = document.into();
    for name in ["keydown", "keyup"] {
        let for_event = Rc::clone(&shared);
        let listener = Listener::Key(Closure::new(move |e: KeyboardEvent| {
            for_event.on_event(&keyboard_event_to_presentar(&e, name));
        }));
        document_target.add_event_listener_with_callback(name, listener.function())?;
        listeners.push((document_target.clone(), name, listener));
    }

    shared.kick();
    Ok(MountHandle { shared, listeners })
}

/// Mount the built-in counter on `<canvas id="presentar-counter">` if the page has one.
/// Each click on "+" adds one and writes the count to the canvas's `data-count`.
/// Returns whether a counter was mounted.
///
/// # Errors
/// As `mount`.
pub fn mount_counter_if_declared() -> Result<bool, JsValue> {
    let Some(canvas) = window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(COUNTER_CANVAS_ID))
    else {
        return Ok(false);
    };
    canvas.set_attribute(COUNT_ATTRIBUTE, "0")?;
    let mut count = 0_i32;
    let handler: MessageHandler = Box::new(move |mounted, _message| {
        count += 1;
        let _ = canvas.set_attribute(COUNT_ATTRIBUTE, &count.to_string());
        mounted.set_root(counter_tree(count));
    });
    mount(COUNTER_CANVAS_ID, counter_tree(0), handler)?.leak();
    Ok(true)
}
