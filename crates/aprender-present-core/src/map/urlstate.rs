//! URL state codec and history policy for the map runtime (APR-MAP-001 MAP-6, contract
//! `aprender-viz-map-url-state-v1`).
//!
//! The state is written into the URL hash as `&`-separated `key=value` pairs, each key and value
//! percent-encoded. Typed keys (`cam`, `sel`, `lens`) are fixed in order, extension keys follow
//! sorted by key, so `encode` is a function of the state alone. Keys that the codec does not
//! type are kept as opaque pairs and re-emitted unchanged. `decode` never panics: every input
//! returns a state or a [`MapError`].
//!
//! Target-neutral plain Rust. Time is passed in by the caller.

use std::collections::BTreeMap;
use std::str::FromStr;

use super::{
    camera::Camera2D,
    layers::{EntityId, LensId},
    MapError,
};

/// Longest hash accepted by `decode` and produced by `encode`, in bytes.
pub const MAX_URL_LEN: usize = 4096;
/// Quiet time after the last camera move before the camera is written with a replace, in ms.
pub const CAMERA_DEBOUNCE_MS: u64 = 250;

const KEY_CAMERA: &str = "cam";
const KEY_SELECTION: &str = "sel";
const KEY_LENS: &str = "lens";

/// The state the URL carries: the typed map keys plus opaque extension pairs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UrlState {
    camera: Option<Camera2D>,
    selection: Option<EntityId>,
    lens: Option<LensId>,
    extra: BTreeMap<String, String>,
}

impl UrlState {
    /// Builds a state from the typed keys with no extension pairs.
    pub fn new(
        camera: Option<Camera2D>,
        selection: Option<EntityId>,
        lens: Option<LensId>,
    ) -> Self {
        Self {
            camera,
            selection,
            lens,
            extra: BTreeMap::new(),
        }
    }

    /// Camera, if the URL carries one.
    pub fn camera(&self) -> Option<Camera2D> {
        self.camera
    }

    /// Selected entity, if any.
    pub fn selection(&self) -> Option<EntityId> {
        self.selection
    }

    /// Active lens, if any.
    pub fn lens(&self) -> Option<LensId> {
        self.lens
    }

    /// Extension pairs, sorted by key.
    pub fn extra(&self) -> &BTreeMap<String, String> {
        &self.extra
    }

    /// Adds or replaces an extension pair. Rejects an empty key and the typed keys.
    pub fn insert_extra(&mut self, key: &str, value: &str) -> Result<(), MapError> {
        let reserved = matches!(key, KEY_CAMERA | KEY_SELECTION | KEY_LENS);
        if key.is_empty() || reserved {
            return Err(MapError::Malformed("url key"));
        }
        self.extra.insert(key.to_owned(), value.to_owned());
        Ok(())
    }
}

/// Encodes a state as a URL hash without the leading `#`.
///
/// Fails with `OutOfRange("url")` when the result would exceed [`MAX_URL_LEN`].
pub fn encode(state: &UrlState) -> Result<String, MapError> {
    let mut out = String::new();
    if let Some(cam) = state.camera {
        push_raw(&mut out, KEY_CAMERA, &camera_value(&cam));
    }
    if let Some(sel) = state.selection {
        push_raw(&mut out, KEY_SELECTION, &sel.to_string());
    }
    if let Some(lens) = state.lens {
        push_raw(&mut out, KEY_LENS, &lens.to_string());
    }
    for (key, value) in &state.extra {
        push_escaped(&mut out, key, value);
    }
    if out.len() > MAX_URL_LEN {
        return Err(MapError::OutOfRange("url"));
    }
    Ok(out)
}

/// Decodes a URL hash, with or without a leading `#`. Empty segments are skipped.
///
/// Fails with `OutOfRange("url")` above [`MAX_URL_LEN`] bytes, and with `Malformed` on a bad
/// percent escape, invalid UTF-8, a duplicate key, or a typed value that does not parse.
pub fn decode(hash: &str) -> Result<UrlState, MapError> {
    if hash.len() > MAX_URL_LEN {
        return Err(MapError::OutOfRange("url"));
    }
    let body = hash.strip_prefix('#').unwrap_or(hash);
    let mut state = UrlState::default();
    for segment in body.split('&').filter(|s| !s.is_empty()) {
        let (raw_key, raw_value) = segment.split_once('=').unwrap_or((segment, ""));
        let key = unescape(raw_key)?;
        let value = unescape(raw_value)?;
        apply_pair(&mut state, key, &value)?;
    }
    Ok(state)
}

/// Stores one decoded pair into `state`, typed by its key.
fn apply_pair(state: &mut UrlState, key: String, value: &str) -> Result<(), MapError> {
    if key == KEY_CAMERA {
        set_once(&mut state.camera, parse_camera(value)?, KEY_CAMERA)
    } else if key == KEY_SELECTION {
        set_once(
            &mut state.selection,
            parse_num(value, KEY_SELECTION)?,
            KEY_SELECTION,
        )
    } else if key == KEY_LENS {
        set_once(&mut state.lens, parse_num(value, KEY_LENS)?, KEY_LENS)
    } else if state.extra.insert(key, value.to_owned()).is_some() {
        Err(MapError::Malformed("duplicate url key"))
    } else {
        Ok(())
    }
}

/// Sets a typed slot, failing when the key already appeared in the hash.
fn set_once<T>(slot: &mut Option<T>, value: T, what: &'static str) -> Result<(), MapError> {
    if slot.is_some() {
        return Err(MapError::Malformed(what));
    }
    *slot = Some(value);
    Ok(())
}

/// Parses `cam=x,y,scale,viewport_w,viewport_h`, validated by `Camera2D::new`.
fn parse_camera(value: &str) -> Result<Camera2D, MapError> {
    let fields: Vec<&str> = value.split(',').collect();
    match fields.as_slice() {
        [x, y, scale, w, h] => Camera2D::new(
            (parse_num(x, KEY_CAMERA)?, parse_num(y, KEY_CAMERA)?),
            parse_num(scale, KEY_CAMERA)?,
            (parse_num(w, KEY_CAMERA)?, parse_num(h, KEY_CAMERA)?),
        ),
        _ => Err(MapError::Malformed(KEY_CAMERA)),
    }
}

fn parse_num<T: FromStr>(value: &str, what: &'static str) -> Result<T, MapError> {
    value.parse::<T>().map_err(|_| MapError::Malformed(what))
}

/// Camera value in the form `parse_camera` reads. `f64` `Display` is the shortest string that
/// parses back to the same value, so no precision is lost.
fn camera_value(cam: &Camera2D) -> String {
    let (x, y) = cam.centre();
    let (w, h) = cam.viewport();
    format!("{x},{y},{},{w},{h}", cam.scale())
}

/// Appends `key=value` with an already-safe key and value.
fn push_raw(out: &mut String, key: &str, value: &str) {
    if !out.is_empty() {
        out.push('&');
    }
    out.push_str(key);
    out.push('=');
    out.push_str(value);
}

/// Appends `key=value` with both parts percent-encoded.
fn push_escaped(out: &mut String, key: &str, value: &str) {
    if !out.is_empty() {
        out.push('&');
    }
    escape_into(out, key);
    out.push('=');
    escape_into(out, value);
}

/// Percent-encodes every byte outside the RFC 3986 unreserved set, so `&`, `=`, `#`, `%`, `+`,
/// space and every non-ASCII byte are all written as `%XX`.
fn escape_into(out: &mut String, raw: &str) {
    for byte in raw.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(hex_digit(byte >> 4));
            out.push(hex_digit(byte & 0x0F));
        }
    }
}

/// Upper-case hex digit for a nibble (`n` is masked to 0..=15 by the caller).
fn hex_digit(n: u8) -> char {
    match n & 0x0F {
        d @ 0..=9 => char::from(b'0' + d),
        d => char::from(b'A' + d - 10),
    }
}

/// Reverses `escape_into`. Accepts lower- and upper-case hex and rejects anything else after `%`.
fn unescape(raw: &str) -> Result<String, MapError> {
    let mut bytes = Vec::with_capacity(raw.len());
    let mut rest = raw.as_bytes();
    while let Some((&first, tail)) = rest.split_first() {
        if first == b'%' {
            let [h, l, after @ ..] = tail else {
                return Err(MapError::Malformed("percent escape"));
            };
            match (hex_val(*h), hex_val(*l)) {
                (Some(hi), Some(lo)) => bytes.push(hi * 16 + lo),
                _ => return Err(MapError::Malformed("percent escape")),
            }
            rest = after;
        } else {
            bytes.push(first);
            rest = tail;
        }
    }
    String::from_utf8(bytes).map_err(|_| MapError::Malformed("utf-8"))
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// What the caller must do with the browser history after an event.
#[derive(Debug, Clone, PartialEq)]
pub enum Nav {
    /// Nothing to write now.
    Nothing,
    /// `pushState` with this hash: one new history entry.
    Push(String),
    /// `replaceState` with this hash: no new entry.
    Replace(String),
}

/// An event that changes the URL-visible state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UrlEvent {
    /// The camera moved (pan, zoom or fly-to step).
    Camera(Camera2D),
    /// The selection changed.
    Select(Option<EntityId>),
    /// The lens switched.
    Lens(Option<LensId>),
}

/// History policy: selections and lens switches push one entry each; camera moves are debounced
/// into a single replace once the camera has been still for [`CAMERA_DEBOUNCE_MS`].
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryPolicy {
    state: UrlState,
    last_camera_ms: Option<u64>,
}

impl HistoryPolicy {
    /// Starts the policy from the state the page was loaded with.
    pub fn new(state: UrlState) -> Self {
        Self {
            state,
            last_camera_ms: None,
        }
    }

    /// The current state, including camera moves not yet written.
    pub fn state(&self) -> &UrlState {
        &self.state
    }

    /// True while a camera move is waiting for its debounced replace.
    pub fn camera_pending(&self) -> bool {
        self.last_camera_ms.is_some()
    }

    /// Applies an event at `now_ms`. On error the state is unchanged.
    pub fn apply(&mut self, event: UrlEvent, now_ms: u64) -> Result<Nav, MapError> {
        match event {
            UrlEvent::Camera(cam) => {
                self.state.camera = Some(cam);
                self.last_camera_ms = Some(now_ms);
                Ok(Nav::Nothing)
            }
            UrlEvent::Select(sel) => self.commit_push(|s| s.selection = sel),
            UrlEvent::Lens(lens) => self.commit_push(|s| s.lens = lens),
        }
    }

    /// Writes a debounced camera as a replace once the camera has been still long enough.
    /// A push already carries the latest camera, so it clears any pending move.
    pub fn poll(&mut self, now_ms: u64) -> Result<Nav, MapError> {
        match self.last_camera_ms {
            Some(at) if now_ms.saturating_sub(at) >= CAMERA_DEBOUNCE_MS => {
                let hash = encode(&self.state)?;
                self.last_camera_ms = None;
                Ok(Nav::Replace(hash))
            }
            _ => Ok(Nav::Nothing),
        }
    }

    fn commit_push(&mut self, change: impl FnOnce(&mut UrlState)) -> Result<Nav, MapError> {
        let mut next = self.state.clone();
        change(&mut next);
        let hash = encode(&next)?;
        self.state = next;
        self.last_camera_ms = None;
        Ok(Nav::Push(hash))
    }
}

#[cfg(test)]
#[path = "urlstate_tests.rs"]
mod tests;
