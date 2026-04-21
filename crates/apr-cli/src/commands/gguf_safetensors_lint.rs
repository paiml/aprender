//! CRUX-B-02 — `apr gguf-safetensors-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches three pure classifiers in `gguf_to_safetensors.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "layout": {
//!     "listing": ["model.safetensors", "config.json", "tokenizer.json"]
//!   },
//!   "metadata": {
//!     "kv": {
//!       "general.architecture":       { "str": "llama" },
//!       "llama.embedding_length":     { "u32": 4096 },
//!       "llama.block_count":          { "u32": 32 },
//!       "llama.attention.head_count": { "u32": 32 }
//!     },
//!     "expected_outcome": "ok"   // ok | missing_key | wrong_type
//!   },
//!   "peft": {
//!     "tensor_names":     ["model.layers.0.self_attn.q_proj.weight", "..."],
//!     "target_modules":   ["q_proj", "v_proj"],
//!     "expected_outcome": "resolved"   // resolved | unresolved
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-02
//! stderr stamp on any failing gate.

use crate::commands::gguf_to_safetensors::{
    hf_required_files, missing_hf_files, peft_target_modules_resolve, translate_gguf_metadata,
    GgufValue, MetadataError, PeftResolution,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GgufSafetensorsLintArgs {
    pub observation_file: String,
    pub json: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: GgufSafetensorsLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-02: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-02: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-02: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-02: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("layout") {
        let (r, err) = run_layout_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("metadata") {
        let (r, err) = run_metadata_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("peft") {
        let (r, err) = run_peft_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-B-02: observation has none of layout/metadata/peft".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-02",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(failures.join("\n"));
    }
    Ok(())
}

fn run_layout_gate(v: &Value) -> (GateReport, Option<String>) {
    let listing: BTreeSet<String> = v
        .get("listing")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let missing = missing_hf_files(&listing);
    let required = hf_required_files();
    let passed = missing.is_empty();
    let desc = if passed {
        format!(
            "listing={} covers required trio ({})",
            listing.len(),
            required.len()
        )
    } else {
        let mut sorted: Vec<&&str> = missing.iter().collect();
        sorted.sort();
        format!("missing={sorted:?}")
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-02-001 layout gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "layout",
            falsify_id: "FALSIFY-CRUX-B-02-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_metadata_gate(v: &Value) -> (GateReport, Option<String>) {
    let kv = match parse_kv(v.get("kv")) {
        Ok(m) => m,
        Err(e) => {
            let desc = format!("kv parse error: {e}");
            return (
                GateReport {
                    gate: "metadata",
                    falsify_id: "FALSIFY-CRUX-B-02-003",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!(
                    "FALSIFY-CRUX-B-02-003 metadata gate failed: {desc}"
                )),
            );
        }
    };
    let result = translate_gguf_metadata(&kv);
    let got = match &result {
        Ok(_) => "ok",
        Err(MetadataError::MissingKey(_)) => "missing_key",
        Err(MetadataError::WrongType { .. }) => "wrong_type",
    };
    let expected = v
        .get("expected_outcome")
        .and_then(|x| x.as_str())
        .unwrap_or("ok");
    let passed = got == expected;
    let detail = match &result {
        Ok(cfg) => format!(
            "arch={:?} hidden={} layers={} heads={}",
            cfg.architectures, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads
        ),
        Err(MetadataError::MissingKey(k)) => format!("missing={k}"),
        Err(MetadataError::WrongType { key, got }) => format!("key={key} got={got}"),
    };
    let desc = format!("expected={expected} got={got} ({detail})");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-02-003 metadata gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "metadata",
            falsify_id: "FALSIFY-CRUX-B-02-003",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn parse_kv(v: Option<&Value>) -> Result<BTreeMap<String, GgufValue>, String> {
    let obj = v
        .and_then(|x| x.as_object())
        .ok_or_else(|| "kv must be a JSON object".to_string())?;
    let mut out = BTreeMap::new();
    for (k, val) in obj {
        let inner = val
            .as_object()
            .ok_or_else(|| format!("kv[{k}] must be an object with one of str|u32"))?;
        if let Some(s) = inner.get("str").and_then(|x| x.as_str()) {
            out.insert(k.clone(), GgufValue::Str(s.to_string()));
        } else if let Some(n) = inner.get("u32").and_then(|x| x.as_u64()) {
            let n: u32 = n
                .try_into()
                .map_err(|_| format!("kv[{k}].u32 out of range"))?;
            out.insert(k.clone(), GgufValue::U32(n));
        } else {
            return Err(format!("kv[{k}] must have a 'str' or 'u32' field"));
        }
    }
    Ok(out)
}

fn run_peft_gate(v: &Value) -> (GateReport, Option<String>) {
    let tensor_names: Vec<String> = v
        .get("tensor_names")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let target_owned: Vec<String> = v
        .get("target_modules")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let target_refs: Vec<&str> = target_owned.iter().map(|s| s.as_str()).collect();
    let res = peft_target_modules_resolve(&tensor_names, &target_refs);
    let got = match &res {
        PeftResolution::AllResolved => "resolved",
        PeftResolution::Unresolved { .. } => "unresolved",
    };
    let expected = v
        .get("expected_outcome")
        .and_then(|x| x.as_str())
        .unwrap_or("resolved");
    let passed = got == expected;
    let detail = match &res {
        PeftResolution::AllResolved => format!(
            "targets={} resolved over tensors={}",
            target_owned.len(),
            tensor_names.len()
        ),
        PeftResolution::Unresolved { missing } => format!("missing={missing:?}"),
    };
    let desc = format!("expected={expected} got={got} ({detail})");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-02-004 peft gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "peft",
            falsify_id: "FALSIFY-CRUX-B-02-004",
            outcome: desc,
            passed,
        },
        err,
    )
}
