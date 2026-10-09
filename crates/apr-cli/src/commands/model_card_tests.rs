//! EXT-16 (aprender#4398): FALSIFY-EXT-018 — a hand-typed card number fails the lint;
//! FALSIFY-EXT-024 — a ratio next to a competitor name is rejected.

use super::super::comparator::ComparatorBlock;
use super::super::model_gate::{M1Evidence, M3Evidence, Probe};
use super::super::model_gate_m1b::{M1bEvidence, QualityArm, QualityRecord};
use super::super::speed_arms::ArmMeasurement;
use super::*;

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn manifest() -> ReleaseManifest {
    serde_json::from_value(serde_json::json!({
        "line": "paiml/qwen3.5-4b-apr", "version": "0.1.0", "channel": "released",
        "files": [{"name": "model-q4k.gguf", "format": "gguf", "quant": "Q4_K_M",
                   "bytes": 2_700_000_000_u64, "sha256": sha(0xaa)},
                  {"name": "README.md", "format": "markdown", "bytes": 1, "sha256": sha(0xab)}],
        "base": {"hf_id": "Qwen/Qwen3.5-4B", "revision": "3".repeat(40), "sha256": sha(9)},
        "lineage": ["0192f000-0000-7000-8000-000000000001"], "datasets": [sha(5)],
        "engine": {"apr_version": "0.71.0", "crate_tarball_sha256": sha(8)},
        "license": {"spdx_or_name": "Apache-2.0", "upstream_notice_sha256": sha(7)},
    }))
    .expect("manifest")
}

fn qarm(name: &str, n: u8, kl: f64, top1: f64) -> QualityArm {
    QualityArm {
        arm: name.into(),
        file_sha256: sha(n),
        kl_mean: kl,
        top1,
    }
}

fn evidence() -> GateEvidence {
    GateEvidence {
        m1: Some(M1Evidence {
            cosine: 0.999_31,
            llama_cpp_commit: "d1d3c3396aa13a5f239109a822666c4870490ad5".into(),
            greedy_token_agreement: Some(0.97),
        }),
        m1b: Some(M1bEvidence {
            current: QualityRecord {
                version: "0.1.0".into(),
                llama_cpp_commit: "d1d3c3396aa13a5f239109a822666c4870490ad5".into(),
                corpus_sha256: sha(0xc0),
                reference_sha256: sha(0xbf),
                ours: qarm("ours", 0xaa, 0.031, 0.929),
                arms: vec![qarm("unsloth-q4_k_m", 0x11, 0.029, 0.931)],
            },
            baseline: None,
            receipt_id: "m1b-quality".into(),
        }),
        m3: Some(M3Evidence {
            probes: vec![
                Probe {
                    id: "hello".into(),
                    run_answered: true,
                    serve_answered: true,
                },
                Probe {
                    id: "tool-call-2".into(),
                    run_answered: true,
                    serve_answered: false,
                },
            ],
            parse_rate: None,
        }),
        receipt_ids: vec!["gate-ev-1".into(), "m1b-quality".into()],
        ..GateEvidence::default()
    }
}

fn block(version: &str, n: u8) -> ComparatorBlock {
    ComparatorBlock {
        command: vec!["llama-bench".into()],
        version: version.into(),
        env_sha256: sha(n),
        artifact_sha256: sha(n + 1),
        log_path: "/tmp/arm.log".into(),
        image: None,
        started_utc: "2026-09-27T08:00:00Z".into(),
        finished_utc: "2026-09-27T08:01:00Z".into(),
    }
}

fn speed_arms() -> Vec<ArmOutcome> {
    vec![
        ArmOutcome::Measured(ArmMeasurement {
            arm: "apr".into(),
            decode_tok_s: 41.2,
            comparator: block("apr 0.71.0", 0x20),
        }),
        ArmOutcome::Measured(ArmMeasurement {
            arm: "llama.cpp".into(),
            decode_tok_s: 44.9,
            comparator: block("version: 6000 (d1d3c339)", 0x30),
        }),
        ArmOutcome::NotRun {
            arm: "mistral.rs".into(),
            reason: "v0.9.4 refuses the qwen35 architecture".into(),
        },
    ]
}

fn rendered() -> RenderedCard {
    let (m, ev, arms) = (manifest(), evidence(), speed_arms());
    render_card(&CardSources {
        manifest: &m,
        evidence: &ev,
        evidence_receipt: "gate-ev-1",
        prereg: &M2Prereg::REX_001,
        speed: Some(SpeedTable {
            arms: &arms,
            receipt_id: "speed-1",
        }),
        datasets: &[DatasetFraction {
            canonical_sha256: sha(5),
            synthetic_fraction: 0.125,
            receipt_id: "ds-5".into(),
        }],
    })
}

fn lint(text: &str, card: &RenderedCard) -> CardLint {
    let known: BTreeSet<&str> = card.cited.iter().map(String::as_str).collect();
    lint_card("README.md", text, &known)
}

/// The control: a card rendered from receipts has every figure mapped and passes.
#[test]
fn rendered_card_is_clean_and_covers_every_section() {
    let card = rendered();
    let l = lint(&card.markdown, &card);
    assert!(l.is_clean(), "{:#?}\n{}", l.findings, card.markdown);
    assert_eq!(l.unmapped, 0);
    assert!(l.cited >= 7, "{} cited lines\n{}", l.cited, card.markdown);
    for section in [
        "## What",
        "## Lineage",
        "## Engine",
        "## Parity",
        "## Artifact quality",
        "## Sealed suites",
        "## Speed",
        "## Synthetic fraction",
        "## License",
        "## Known refusals",
    ] {
        assert!(card.markdown.contains(section), "missing {section}");
    }
    for want in [
        "0.999310",
        "41.2",
        "44.9",
        "0.1250",
        "tool-call-2",
        "mistral.rs | not run",
        "Apache-2.0",
    ] {
        assert!(card.markdown.contains(want), "missing {want:?}");
    }
    let ids: Vec<&str> = card.cited.iter().map(String::as_str).collect();
    assert_eq!(ids, ["ds-5", "gate-ev-1", "m1b-quality", "speed-1"]);
}

/// Absent evidence renders as not measured, never as a number.
#[test]
fn absent_evidence_renders_no_figure() {
    let m = manifest();
    let ev = GateEvidence::default();
    let card = render_card(&CardSources {
        manifest: &m,
        evidence: &ev,
        evidence_receipt: "gate-ev-1",
        prereg: &M2Prereg::REX_001,
        speed: None,
        datasets: &[],
    });
    let l = lint(&card.markdown, &card);
    assert!(l.is_clean(), "{:#?}", l.findings);
    assert_eq!(l.cited, 0, "{}", card.markdown);
    assert!(card.cited.is_empty());
    assert_eq!(card.markdown.matches("Not measured").count(), 5);
}

/// FALSIFY-EXT-018: a hand-typed number fails the lint, wherever it is typed.
#[test]
fn falsify_ext_018_hand_typed_number_fails() {
    let card = rendered();
    for typed in [
        "Decodes at 57 tok/s on an RTX 4090.",
        "- Context window: 262144 tokens",
        "| apr | 41.2 | apr v0.71.0 | sha256-202020202020 |",
        "Trained on 1.5 M rows.",
    ] {
        let text = format!("{}{typed}\n", card.markdown);
        let l = lint(&text, &card);
        assert_eq!(l.unmapped, 1, "{typed:?}");
        assert!(!l.is_clean(), "{typed:?}");
        assert!(
            l.findings
                .iter()
                .any(|f| f.contains("a figure with no receipt id")),
            "{typed:?}: {:?}",
            l.findings
        );
    }
    // A figure citing a receipt the release does not carry is also refused.
    let text = format!("{}MBPP 0.61 [receipt:made-up]\n", card.markdown);
    let l = lint(&text, &card);
    assert_eq!(l.unmapped, 0);
    assert!(l.findings.iter().any(|f| f.contains("\"made-up\"")));
}

/// FALSIFY-EXT-024: a ratio or comparative token next to a competitor name is rejected,
/// even when the line cites a known receipt.
#[test]
fn falsify_ext_024_ratio_next_to_competitor_rejected() {
    let card = rendered();
    let reject = [
        "apr decodes at 1.3x llama.cpp [receipt:speed-1]",
        "apr decodes at 1.3 x llama.cpp [receipt:speed-1]",
        "2× the throughput of Ollama [receipt:speed-1]",
        "30% faster than vLLM [receipt:speed-1]",
        "apr: 0.92 of mistral.rs (ratio) [receipt:speed-1]",
        "Speedup over llama.cpp: 1.1 [receipt:speed-1]",
        "apr beats Ollama on decode [receipt:speed-1]",
        "apr outperforms text-generation-inference [receipt:speed-1]",
        "| llama.cpp | 44.9 (0.9x) | [receipt:speed-1] |",
    ];
    for line in reject {
        let l = lint(&format!("{}{line}\n", card.markdown), &card);
        assert!(
            l.findings.iter().any(|f| f.contains("next to competitor")),
            "not rejected: {line:?}"
        );
    }
    let accept = [
        "| llama.cpp | 44.9 | version: 6000 (d1d3c339) | sha256-313131313131 [receipt:speed-1] |",
        "text-generation-inference: 12.0 tok/s [receipt:speed-1]",
        "Parity cosine 0.9993 vs llama.cpp [receipt:gate-ev-1]",
        "apr decodes 1.3x faster than v0.70 [receipt:speed-1]",
        "calibration generation on x86_64 [receipt:speed-1]",
        "llamacppish tooling ratio [receipt:speed-1]",
    ];
    for line in accept {
        let l = lint(&format!("{}{line}\n", card.markdown), &card);
        assert!(l.is_clean(), "{line:?}: {:?}", l.findings);
    }
}

/// `[X]` figures: labelled third-party, and never on a competitor line.
#[test]
fn x_figures_are_labelled_and_never_competitor_values() {
    let card = rendered();
    let cases = [
        (
            "Upstream reports 0.79 on MMLU [X] [receipt:gate-ev-1]",
            Some("not labelled"),
        ),
        (
            "Ollama reports 50 tok/s [X], third-party [receipt:gate-ev-1]",
            Some("competitor line"),
        ),
        (
            "Upstream reports 0.79 on MMLU [X], third-party [receipt:gate-ev-1]",
            None,
        ),
    ];
    for (line, want) in cases {
        let l = lint(&format!("{}{line}\n", card.markdown), &card);
        match want {
            Some(w) => assert!(
                l.findings.iter().any(|f| f.contains(w)),
                "{line:?}: {:?}",
                l.findings
            ),
            None => assert!(l.is_clean(), "{line:?}: {:?}", l.findings),
        }
    }
}

#[test]
fn competitor_names_match_on_word_boundaries() {
    assert_eq!(competitors_named("llama.cpp d1d3"), ["llama.cpp"]);
    assert_eq!(competitors_named("see ollama."), ["ollama"]);
    assert!(competitors_named("llamacppish").is_empty());
    assert!(competitors_named("mlxy candles").is_empty());
    assert_eq!(ratio_token("30% faster").as_deref(), Some("% faster"));
    assert_eq!(ratio_token("at 1.3x it").as_deref(), Some("1.3x"));
    assert_eq!(ratio_token("q4x 0x7f x86_64"), None);
    assert_eq!(ratio_token("text-generation"), None);
}
