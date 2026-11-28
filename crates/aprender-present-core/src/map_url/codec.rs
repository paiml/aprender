//! `encode` and `decode` for the `k=v&...` map URL hash.
//!
//! Values and keys are percent-encoded: every byte outside the unreserved set
//! `A-Z a-z 0-9 - _ . ~` becomes `%XX` (uppercase hex). Camera floats use
//! Rust's shortest round-trip `Display`, never fixed decimals.

use super::state::{Camera, MapUrlError, MapUrlState, EXTENSION_KEYS, FIXED_KEYS};
use std::collections::BTreeSet;

const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Encodes `state` as the hash body, without the leading `#`.
///
/// Key order is fixed: `lens`, `city`, `state`, `at`, `advisor`, `layers`,
/// then registered extension keys sorted, then opaque keys sorted.
///
/// # Errors
/// Returns an error for a non-finite camera, an empty layer name, or a bad key.
pub fn encode(state: &MapUrlState) -> Result<String, MapUrlError> {
    let mut pairs: Vec<String> = Vec::new();
    push_text(&mut pairs, "lens", state.lens.as_deref());
    push_text(&mut pairs, "city", state.city.as_deref());
    push_text(&mut pairs, "state", state.state.as_deref());
    if let Some(camera) = state.camera {
        pairs.push(format!("at={}", encode_camera(camera)?));
    }
    push_text(&mut pairs, "advisor", state.advisor.as_deref());
    if !state.layers.is_empty() {
        pairs.push(format!("layers={}", encode_layers(&state.layers)?));
    }
    for (key, value) in &state.extensions {
        if !EXTENSION_KEYS.contains(&key.as_str()) {
            return Err(MapUrlError::UnregisteredExtension(key.clone()));
        }
        pairs.push(pair(key, value));
    }
    for (key, value) in &state.opaque {
        check_opaque_key(key)?;
        pairs.push(pair(key, value));
    }
    Ok(pairs.join("&"))
}

/// Decodes a hash, with or without the leading `#`. Unregistered keys are kept
/// in `opaque`; empty segments are skipped.
///
/// # Errors
/// Returns an error for a pair with no `=`, a duplicate or empty key, a bad
/// percent escape, invalid UTF-8, or a non-finite or malformed camera.
pub fn decode(hash: &str) -> Result<MapUrlState, MapUrlError> {
    let body = hash.strip_prefix('#').unwrap_or(hash);
    let mut state = MapUrlState::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for segment in body.split('&').filter(|s| !s.is_empty()) {
        let (raw_key, raw_value) = segment
            .split_once('=')
            .ok_or_else(|| MapUrlError::Malformed(format!("pair without '=': {segment}")))?;
        let key = unescape(raw_key)?;
        if key.is_empty() {
            return Err(MapUrlError::EmptyKey);
        }
        if !seen.insert(key.clone()) {
            return Err(MapUrlError::DuplicateKey(key));
        }
        apply(&mut state, key, raw_value)?;
    }
    Ok(state)
}

fn apply(state: &mut MapUrlState, key: String, raw: &str) -> Result<(), MapUrlError> {
    match key.as_str() {
        "lens" => state.lens = Some(unescape(raw)?),
        "city" => state.city = Some(unescape(raw)?),
        "state" => state.state = Some(unescape(raw)?),
        "advisor" => state.advisor = Some(unescape(raw)?),
        "at" => state.camera = Some(decode_camera(raw)?),
        "layers" => state.layers = decode_layers(raw)?,
        _ => {
            let value = unescape(raw)?;
            if EXTENSION_KEYS.contains(&key.as_str()) {
                state.extensions.insert(key, value);
            } else {
                state.opaque.insert(key, value);
            }
        }
    }
    Ok(())
}

fn push_text(pairs: &mut Vec<String>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        pairs.push(pair(key, value));
    }
}

fn pair(key: &str, value: &str) -> String {
    format!("{}={}", escape(key), escape(value))
}

fn check_opaque_key(key: &str) -> Result<(), MapUrlError> {
    if key.is_empty() {
        return Err(MapUrlError::EmptyKey);
    }
    if FIXED_KEYS.contains(&key) || EXTENSION_KEYS.contains(&key) {
        return Err(MapUrlError::KeyCollision(key.to_owned()));
    }
    Ok(())
}

fn encode_camera(camera: Camera) -> Result<String, MapUrlError> {
    if !camera.is_finite() {
        return Err(MapUrlError::NonFinite);
    }
    Ok(format!("{},{},{}", camera.x, camera.y, camera.log2_zoom))
}

fn encode_layers(layers: &[String]) -> Result<String, MapUrlError> {
    let mut parts = Vec::with_capacity(layers.len());
    for layer in layers {
        if layer.is_empty() {
            return Err(MapUrlError::EmptyLayer);
        }
        parts.push(escape(layer));
    }
    Ok(parts.join(","))
}

fn decode_camera(raw: &str) -> Result<Camera, MapUrlError> {
    let parts: Vec<&str> = raw.split(',').collect();
    match parts.as_slice() {
        [x, y, z] => Ok(Camera {
            x: parse_finite(x)?,
            y: parse_finite(y)?,
            log2_zoom: parse_finite(z)?,
        }),
        _ => Err(MapUrlError::Malformed(format!(
            "at needs three values: {raw}"
        ))),
    }
}

fn parse_finite(text: &str) -> Result<f64, MapUrlError> {
    let value: f64 = text
        .parse()
        .map_err(|_| MapUrlError::Malformed(format!("not a number: {text}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MapUrlError::NonFinite)
    }
}

fn decode_layers(raw: &str) -> Result<Vec<String>, MapUrlError> {
    raw.split(',')
        .filter(|s| !s.is_empty())
        .map(unescape)
        .collect()
}

/// Percent-encodes every byte outside the unreserved set.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
    out
}

/// Reverses `escape`. Rejects a `%` not followed by two hex digits, and
/// percent-decoded bytes that are not UTF-8.
fn unescape(text: &str) -> Result<String, MapUrlError> {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            out.push(hex_byte(bytes.get(i + 1..i + 3))?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out)
        .map_err(|_| MapUrlError::Malformed("percent-decoded value is not UTF-8".to_owned()))
}

fn hex_byte(digits: Option<&[u8]>) -> Result<u8, MapUrlError> {
    digits
        .filter(|d| d.iter().all(u8::is_ascii_hexdigit))
        .and_then(|d| std::str::from_utf8(d).ok())
        .and_then(|h| u8::from_str_radix(h, 16).ok())
        .ok_or_else(|| MapUrlError::Malformed("bad percent escape".to_owned()))
}
