//! MEAS-001 R1 (#4522): the measured `resources{}` block that `apr bench`,
//! `apr qa` and `apr profile` put in their `--json` documents.
//!
//! The sampler is `realizar::resources` (aprender-serve). Without the
//! `inference` feature there is no sampler, and every field is emitted as
//! null WITH a reason — never as a zero and never omitted, so a consumer
//! (R2 reads `peak_rss_bytes` / `vram_peak_bytes`) can tell "not measured"
//! from "measured zero".

/// The JSON keys of a `resources{}` block that carry a reading.
pub(crate) const FIELDS: [&str; 15] = [
    "peak_rss_bytes",
    "cpu_user_s",
    "cpu_sys_s",
    "minor_faults",
    "major_faults",
    "voluntary_ctx_switches",
    "involuntary_ctx_switches",
    "threads_peak",
    "io_read_bytes",
    "io_write_bytes",
    "io_rchar_bytes",
    "io_wchar_bytes",
    "vram_peak_bytes",
    "energy_j",
    "gpu_energy_j",
];

/// One measurement window: started before the work, finished after it.
pub(crate) struct Window {
    #[cfg(feature = "inference")]
    sampler: realizar::resources::ResourceSampler,
    #[cfg(not(feature = "inference"))]
    started: std::time::Instant,
}

/// Start sampling now.
pub(crate) fn start() -> Window {
    Window {
        #[cfg(feature = "inference")]
        sampler: realizar::resources::ResourceSampler::start(),
        #[cfg(not(feature = "inference"))]
        started: std::time::Instant::now(),
    }
}

impl Window {
    /// Stop sampling and return the `resources{}` JSON block.
    pub(crate) fn finish(self) -> serde_json::Value {
        #[cfg(feature = "inference")]
        {
            let usage = self.sampler.finish();
            serde_json::to_value(&usage)
                .unwrap_or_else(|e| unmeasured(0.0, &format!("resources did not serialize: {e}")))
        }
        #[cfg(not(feature = "inference"))]
        {
            unmeasured(
                self.started.elapsed().as_secs_f64(),
                "apr built without the `inference` feature: realizar::resources is absent",
            )
        }
    }
}

/// A block in which every field is null, each with `why`.
pub(crate) fn unmeasured(wall_s: f64, why: &str) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("wall_s".to_string(), serde_json::json!(wall_s));
    let mut reasons = serde_json::Map::new();
    for k in FIELDS {
        obj.insert(k.to_string(), serde_json::Value::Null);
        reasons.insert(k.to_string(), serde_json::Value::String(why.to_string()));
    }
    obj.insert("sources".to_string(), serde_json::json!({}));
    obj.insert(
        "null_reasons".to_string(),
        serde_json::Value::Object(reasons),
    );
    serde_json::Value::Object(obj)
}

/// CRUX-D-03: the one-line `--report-peak-memory` document. The two peaks
/// are lifted to the top level so the falsifier reads `.peak_vram_bytes`;
/// the whole block (with its sources and null reasons) rides beside them.
pub(crate) fn peak_memory_line(resources: &serde_json::Value) -> String {
    serde_json::json!({
        "peak_vram_bytes": resources.get("vram_peak_bytes").cloned().unwrap_or_default(),
        "peak_rss_bytes": resources.get("peak_rss_bytes").cloned().unwrap_or_default(),
        "resources": resources,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field is either set with a source or null with a reason.
    fn assert_accounted(v: &serde_json::Value) {
        let src = v["sources"].as_object().expect("sources");
        let why = v["null_reasons"].as_object().expect("null_reasons");
        for k in FIELDS {
            let val = v.get(k).unwrap_or_else(|| panic!("{k} missing: {v}"));
            if val.is_null() {
                assert!(why.contains_key(k) && !src.contains_key(k), "{k}: {v}");
            } else {
                assert!(src.contains_key(k) && !why.contains_key(k), "{k}: {v}");
            }
        }
    }

    #[test]
    fn unmeasured_block_nulls_every_field_with_its_reason() {
        let v = unmeasured(1.5, "because");
        assert_accounted(&v);
        for k in FIELDS {
            assert!(v[k].is_null());
            assert_eq!(v["null_reasons"][k], "because");
        }
    }

    #[test]
    fn a_window_accounts_for_every_field() {
        let w = start();
        let v = w.finish();
        assert_accounted(&v);
        assert!(v["wall_s"].as_f64().expect("wall_s") >= 0.0);
    }

    #[test]
    fn peak_memory_line_is_one_line_with_both_peaks_on_top() {
        let block = serde_json::json!({"vram_peak_bytes": 7, "peak_rss_bytes": 9});
        let line = peak_memory_line(&block);
        assert!(!line.contains('\n'), "{line}");
        let v: serde_json::Value = serde_json::from_str(&line).expect("json");
        assert_eq!(v["peak_vram_bytes"], 7);
        assert_eq!(v["peak_rss_bytes"], 9);
        assert_eq!(v["resources"], block);
        // Unmeasured stays null — never a zero.
        let v: serde_json::Value =
            serde_json::from_str(&peak_memory_line(&unmeasured(0.0, "x"))).expect("json");
        assert!(v["peak_vram_bytes"].is_null() && v["peak_rss_bytes"].is_null());
    }

    /// FIELDS must be exactly the reading keys the sampler serializes: a field
    /// added to `ResourceUsage` and not here would be dropped from the
    /// non-inference block.
    #[cfg(feature = "inference")]
    #[test]
    fn fields_match_the_sampler_schema() {
        let v =
            serde_json::to_value(realizar::resources::ResourceUsage::default()).expect("serialize");
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .filter(|k| !matches!(*k, "wall_s" | "sources" | "null_reasons"))
            .collect();
        keys.sort_unstable();
        let mut want = FIELDS.to_vec();
        want.sort_unstable();
        assert_eq!(keys, want);
    }
}
