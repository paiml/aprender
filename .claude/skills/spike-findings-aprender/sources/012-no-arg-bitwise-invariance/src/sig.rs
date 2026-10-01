//! A bit-exact signature over the deterministic fields of a `ForecastResponse`.
//!
//! `fit_seconds` / `predict_seconds` are wall-clock and are EXCLUDED — a signature that
//! includes them can never be stable, which would make the whole gate vacuous.
//! Everything else is included, f64s by their BITS (`to_bits`), not by a printed decimal:
//! a formatted comparison silently accepts a change below the print precision.

use aprender_forecast::types::ForecastResponse;

pub struct Hasher(u64);

impl Hasher {
    pub fn new() -> Self {
        Hasher(0xcbf2_9ce4_8422_2325)
    }
    pub fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }
    pub fn f64(&mut self, v: f64) {
        // Normalise the two zeros so -0.0 and 0.0 do not read as a change; every other
        // bit pattern, NaN payloads included, is significant.
        let v = if v == 0.0 { 0.0 } else { v };
        self.bytes(&v.to_bits().to_le_bytes());
    }
    pub fn f64s(&mut self, vs: &[f64]) {
        self.bytes(&(vs.len() as u64).to_le_bytes());
        for &v in vs {
            self.f64(v);
        }
    }
    pub fn str(&mut self, s: &str) {
        self.bytes(&(s.len() as u64).to_le_bytes());
        self.bytes(s.as_bytes());
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

fn json(h: &mut Hasher, v: &serde_json::Value) {
    match v {
        serde_json::Value::Null => h.bytes(b"n"),
        serde_json::Value::Bool(b) => {
            h.bytes(b"b");
            h.bytes(&[u8::from(*b)]);
        }
        serde_json::Value::Number(n) => {
            h.bytes(b"#");
            h.f64(n.as_f64().unwrap_or(f64::NAN));
        }
        serde_json::Value::String(s) => {
            h.bytes(b"s");
            h.str(s);
        }
        serde_json::Value::Array(a) => {
            h.bytes(b"[");
            h.bytes(&(a.len() as u64).to_le_bytes());
            for x in a {
                json(h, x);
            }
        }
        serde_json::Value::Object(o) => {
            // serde_json's default Map is a BTreeMap, so iteration is key-sorted and the
            // signature does not depend on insertion order.
            h.bytes(b"{");
            h.bytes(&(o.len() as u64).to_le_bytes());
            for (k, x) in o {
                h.str(k);
                json(h, x);
            }
        }
    }
}

#[must_use]
pub fn signature(r: &ForecastResponse) -> u64 {
    let mut h = Hasher::new();
    h.str(&r.model);
    h.str(&r.freq);
    h.bytes(&(r.n_history as u64).to_le_bytes());
    h.bytes(&(r.ds.len() as u64).to_le_bytes());
    for d in &r.ds {
        h.str(d);
    }
    h.f64s(&r.yhat);
    h.f64s(&r.yhat_lower);
    h.f64s(&r.yhat_upper);
    h.f64s(&r.trend);
    h.bytes(&(r.components.len() as u64).to_le_bytes());
    for (k, v) in &r.components {
        h.str(k);
        json(&mut h, v);
    }
    json(&mut h, &r.diagnostics);
    h.finish()
}
