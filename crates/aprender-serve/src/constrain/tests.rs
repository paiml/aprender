//! #3568 case rows. The schema source and the refusals run in every build. The
//! engine rows need `structured-output`, which a workspace build turns on (apr-cli's
//! `inference`), and which `-p aprender-serve --features structured-output` selects.

use super::*;

#[test]
fn load_schema_takes_the_schema_inline() {
    let v = load_schema(r#"{"type":"object"}"#).expect("an inline schema");
    assert_eq!(v["type"], "object");
}

#[test]
fn load_schema_takes_an_at_path() {
    let dir = std::env::temp_dir().join(format!("apr-3568-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let file = dir.join("schema.json");
    std::fs::write(&file, r#"{"type":"string"}"#).expect("write the schema");
    let v = load_schema(&format!("@{}", file.display())).expect("an @path schema");
    assert_eq!(v["type"], "string");
    std::fs::remove_file(&file).expect("remove the schema");
}

#[test]
fn load_schema_refuses_an_unreadable_path_by_name() {
    let e = load_schema("@/no/such/dir/schema.json").expect_err("an unreadable path must refuse");
    assert!(
        matches!(&e, ConstraintError::SchemaInvalid(why) if why.contains("/no/such/dir/schema.json")),
        "{e}"
    );
}

#[test]
fn load_schema_refuses_text_that_is_not_json() {
    // the mistake the design input names: a path passed where the schema was expected
    let e = load_schema("schema.json").expect_err("a bare path is not a schema");
    assert!(
        matches!(&e, ConstraintError::SchemaInvalid(why) if why.contains("not JSON")),
        "{e}"
    );
    assert!(e.to_string().starts_with("SchemaInvalid: "), "{e}");
}

#[test]
fn load_schema_refuses_json_that_is_not_a_schema() {
    for not_a_schema in ["[1, 2]", "42", r#""object""#, "null"] {
        let e = load_schema(not_a_schema).expect_err(not_a_schema);
        assert!(
            matches!(&e, ConstraintError::SchemaInvalid(why) if why.contains("not a schema")),
            "{e}"
        );
    }
}

#[test]
fn a_vocabulary_whose_parts_disagree_is_refused_before_any_engine_sees_it() {
    let good = ConstraintVocab {
        token_bytes: vec![b"a".to_vec(), b"<eos>".to_vec()],
        special: vec![false, true],
        eos: 1,
    };
    assert_eq!(good.validate(), Ok(()));
    let short_flags = ConstraintVocab {
        special: vec![false],
        ..good.clone()
    };
    assert!(matches!(
        short_flags.validate(),
        Err(ConstraintError::Vocab(_))
    ));
    let eos_outside = ConstraintVocab {
        eos: 2,
        ..good.clone()
    };
    assert!(matches!(
        eos_outside.validate(),
        Err(ConstraintError::Vocab(_))
    ));
    let empty = ConstraintVocab {
        token_bytes: vec![],
        special: vec![],
        eos: 0,
    };
    assert!(matches!(empty.validate(), Err(ConstraintError::Vocab(_))));
}

#[cfg(not(feature = "structured-output"))]
#[test]
fn without_the_feature_a_constraint_is_refused_never_ignored() {
    let vocab = ConstraintVocab {
        token_bytes: vec![b"a".to_vec(), b"<eos>".to_vec()],
        special: vec![false, true],
        eos: 1,
    };
    let e = ConstraintEnv::new(&vocab)
        .err()
        .expect("no engine compiled in: refuse");
    assert_eq!(e, ConstraintError::NotCompiled);
    assert!(
        e.to_string().starts_with("StructuredOutputNotCompiled: "),
        "{e}"
    );
}

#[cfg(feature = "structured-output")]
mod engine {
    use super::*;

    const EOS: u32 = 0;
    const IM_END: u32 = 1;

    /// A byte-level vocabulary: EOS and one more special token, then every single byte (so
    /// every string is reachable), then multi-byte tokens a real BPE vocabulary would carry.
    fn vocab() -> ConstraintVocab {
        let mut token_bytes = vec![b"<|endoftext|>".to_vec(), b"<|im_end|>".to_vec()];
        let mut special = vec![true, true];
        for b in 0..=255u8 {
            token_bytes.push(vec![b]);
            special.push(false);
        }
        for multi in [
            "{\"",
            "\":",
            "\",",
            "\"}",
            "\"verdict\"",
            "PASS",
            "FAIL",
            "xx",
            "\"findings\"",
            "summary",
            "  ",
        ] {
            token_bytes.push(multi.as_bytes().to_vec());
            special.push(false);
        }
        ConstraintVocab {
            token_bytes,
            special,
            eos: EOS,
        }
    }

    fn id_of(v: &ConstraintVocab, bytes: &[u8]) -> u32 {
        let pos = v
            .token_bytes
            .iter()
            .position(|t| t == bytes)
            .expect("token in the vocabulary");
        u32::try_from(pos).expect("id fits u32")
    }

    /// Greedy pick: the highest logit's id (a masked token is -inf and never wins).
    fn greedy(logits: &[f32]) -> u32 {
        let (next, _) =
            logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &l)| {
                    if l > best.1 {
                        (i, l)
                    } else {
                        best
                    }
                });
        u32::try_from(next).expect("id fits u32")
    }

    /// One decoding step, the way a constrained loop runs it: mask, the loop's own greedy
    /// sampler, then accept. `None` when the model chose end-of-sequence.
    fn step(
        constraint: &mut Option<&mut dyn TokenConstraint>,
        mut logits: Vec<f32>,
    ) -> Result<Option<u32>, ConstraintError> {
        if let Some(c) = constraint.as_deref_mut() {
            c.mask(&mut logits)?;
        }
        let next = greedy(&logits);
        if next == EOS {
            return Ok(None);
        }
        if let Some(c) = constraint.as_deref_mut() {
            c.accept(next)?;
        }
        Ok(Some(next))
    }

    /// A fake model that WANTS to violate any schema: `x` scores highest, then `xx`, then
    /// every token by a fixed arbitrary order. Greedy, like `--temperature 0`.
    fn generate(
        v: &ConstraintVocab,
        mut constraint: Option<&mut dyn TokenConstraint>,
        max: usize,
    ) -> Result<Vec<u8>, ConstraintError> {
        let x = id_of(v, b"x");
        let xx = id_of(v, b"xx");
        let score = |i: usize| {
            let t = u32::try_from(i).expect("id fits u32");
            if t == x {
                100.0
            } else if t == xx {
                90.0
            } else if t == EOS {
                80.0 // a finished model wants to stop; masked until the document is complete
            } else {
                ((i * 7919) % 1000) as f32 / 100.0
            }
        };
        let mut out = Vec::new();
        for _ in 0..max {
            let logits = (0..v.token_bytes.len()).map(score).collect();
            let Some(next) = step(&mut constraint, logits)? else {
                return Ok(out);
            };
            out.extend_from_slice(&v.token_bytes[next as usize]);
        }
        Ok(out)
    }

    /// How the intent model scores token `i` when `rest` is what it still means to write:
    /// end-of-sequence once the intent is written; else the longest allowed prefix of `rest`;
    /// else `fallback`'s fixed order, which closes structures.
    fn intent_score(v: &ConstraintVocab, fallback: &[u32], rest: &[u8], i: usize) -> f32 {
        let t = u32::try_from(i).expect("id fits u32");
        let bytes = &v.token_bytes[i];
        if t == EOS {
            return if rest.is_empty() { 2000.0 } else { -1.0 };
        }
        if !v.special[i] && !bytes.is_empty() && rest.starts_with(bytes) {
            return 1000.0 + bytes.len() as f32;
        }
        fallback
            .iter()
            .position(|&f| f == t)
            .map_or(0.0, |rank| 500.0 - rank as f32)
    }

    /// A fake model with an INTENT: it prefers the allowed token that spells the longest prefix
    /// of what it still means to write; when the constraint forbids its next intended byte it
    /// falls back to a fixed order that closes structures. It stops once its intent is written.
    fn generate_intent(
        v: &ConstraintVocab,
        mut constraint: Option<&mut dyn TokenConstraint>,
        intended: &[u8],
        max: usize,
    ) -> Result<Vec<u8>, ConstraintError> {
        let fallback: Vec<u32> = ["}", "]", ",", "\"findings\"", ":", "[", "\""]
            .iter()
            .map(|t| id_of(v, t.as_bytes()))
            .collect();
        let mut out = Vec::new();
        let mut done = 0usize; // bytes of the intent already written
        for _ in 0..max {
            let rest = &intended[done..];
            let logits = (0..v.token_bytes.len())
                .map(|i| intent_score(v, &fallback, rest, i))
                .collect();
            let Some(next) = step(&mut constraint, logits)? else {
                return Ok(out);
            };
            let bytes = &v.token_bytes[next as usize];
            if rest.starts_with(bytes) {
                done += bytes.len();
            }
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    fn verdict_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["verdict"],
            "additionalProperties": false,
            "properties": {"verdict": {"type": "string", "enum": ["PASS", "FAIL", "do-not-implement-as-written"]}}
        })
    }

    #[test]
    fn a_model_that_wants_to_violate_the_schema_is_held_to_it() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");

        // the positive control: unconstrained, this model writes no JSON at all
        let free = generate(&v, None, 16).expect("unconstrained run");
        assert!(
            serde_json::from_slice::<serde_json::Value>(&free).is_err(),
            "{}",
            String::from_utf8_lossy(&free)
        );

        let mut c = env
            .json_schema(&verdict_schema())
            .expect("compile the schema");
        let out = generate(&v, Some(c.as_mut()), 64).expect("constrained run");
        let doc: serde_json::Value = serde_json::from_slice(&out)
            .unwrap_or_else(|e| panic!("not JSON: {e}: {}", String::from_utf8_lossy(&out)));
        let verdict = doc["verdict"]
            .as_str()
            .expect("verdict is required and a string");
        assert!(
            ["PASS", "FAIL", "do-not-implement-as-written"].contains(&verdict),
            "{doc}"
        );
    }

    #[test]
    fn the_rah_quorum_lane_schema_is_enforced_end_to_end() {
        // paiml-implement agy/quorum-lane-schema.json, the first consumer (#3716)
        let schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "paiml-implement quorum lane output",
            "type": "object",
            "required": ["verdict", "summary", "findings"],
            "properties": {
                "verdict": {"type": "string", "enum": ["PASS", "FAIL", "do-not-implement-as-written"]},
                "summary": {"type": "string"},
                "tree_witness": {"type": "string", "description": "copied verbatim"},
                "findings": {"type": "array", "items": {
                    "type": "object",
                    "required": ["file", "claim", "grounding"],
                    "properties": {
                        "file": {"type": "string"}, "line": {"type": "integer"}, "claim": {"type": "string"},
                        "grounding": {"type": "string", "enum": ["cited", "measured", "asserted"]},
                        "fix": {"type": "string"}
                    }
                }}
            }
        });
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env.json_schema(&schema).expect("compile the lane schema");
        // the 2026-09-21 local-lane answer: right substance, `finding` for `findings`, a NULL vote
        let intended = br#"{"verdict":"PASS","summary":"ok","finding":[]}"#;
        let free = generate_intent(&v, None, intended, 128).expect("unconstrained run");
        assert_eq!(
            free,
            intended.to_vec(),
            "unconstrained, the model writes the NULL vote"
        );
        let out = generate_intent(&v, Some(c.as_mut()), intended, 128).expect("constrained run");
        let doc: serde_json::Value = serde_json::from_slice(&out)
            .unwrap_or_else(|e| panic!("not JSON: {e}: {}", String::from_utf8_lossy(&out)));
        for key in ["verdict", "summary", "findings"] {
            assert!(doc.get(key).is_some(), "required {key} missing: {doc}");
        }
        assert!(doc["findings"].is_array(), "{doc}");
        assert!(
            doc.get("finding").is_none(),
            "the 2026-09-21 NULL vote's key: {doc}"
        );
    }

    #[test]
    fn special_tokens_are_never_allowed_as_constrained_text() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env
            .json_schema(&serde_json::json!({"type": "string"}))
            .expect("compile");
        // INSIDE an open string, where `<|im_end|>` would be legal TEXT if it were not special
        c.accept(id_of(&v, b"\""))
            .expect("a string opens with a quote");
        let mut logits = vec![0.0f32; v.token_bytes.len()];
        c.mask(&mut logits).expect("string content can follow");
        assert!(
            logits[id_of(&v, b"<") as usize].is_finite(),
            "the bytes of `<|im_end|>` are legal string text"
        );
        assert_eq!(
            logits[IM_END as usize],
            f32::NEG_INFINITY,
            "a control token must never be string text"
        );
        assert_eq!(
            logits[EOS as usize],
            f32::NEG_INFINITY,
            "nor may end-of-sequence close an open string"
        );
    }

    #[test]
    fn a_complete_document_allows_end_of_sequence() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env
            .json_schema(&serde_json::json!({"type": "boolean"}))
            .expect("compile");
        assert!(!c.is_complete(), "nothing generated yet");
        for b in b"true" {
            c.accept(id_of(&v, &[*b])).expect("true is a boolean");
        }
        assert!(c.is_complete(), "`true` is a complete boolean document");
        let mut logits = vec![0.0f32; v.token_bytes.len()];
        c.mask(&mut logits)
            .expect("the mask after a complete document");
        assert!(
            logits[EOS as usize].is_finite(),
            "EOS must be allowed after a complete document"
        );
    }

    #[test]
    fn a_token_the_mask_forbade_is_rejected_by_name() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env.json_schema(&verdict_schema()).expect("compile");
        let e = c
            .accept(id_of(&v, b"x"))
            .expect_err("an object cannot start with x");
        assert!(
            matches!(e, ConstraintError::Rejected { position: 0, .. }),
            "{e}"
        );
        assert!(e.to_string().starts_with("ConstraintRejected: "), "{e}");
    }

    #[test]
    fn a_vocabulary_that_cannot_spell_the_schema_is_a_dead_end_never_unconstrained() {
        // only EOS and `a`: no digit exists, so an integer cannot even start
        let v = ConstraintVocab {
            token_bytes: vec![b"<|endoftext|>".to_vec(), b"a".to_vec()],
            special: vec![true, false],
            eos: EOS,
        };
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env
            .json_schema(&serde_json::json!({"type": "integer"}))
            .expect("compile");
        let mut logits = vec![1.0f32; 2];
        let e = c
            .mask(&mut logits)
            .expect_err("an integer cannot start from `a`");
        assert!(
            matches!(e, ConstraintError::DeadEnd { position: 0, .. }),
            "{e}"
        );
        assert!(e.to_string().starts_with("ConstraintDeadEnd: "), "{e}");
    }

    #[test]
    fn a_keyword_the_engine_cannot_enforce_is_refused_by_name_before_generation() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        for (schema, keyword) in [
            (
                serde_json::json!({"type": "array", "uniqueItems": true}),
                "uniqueItems",
            ),
            (serde_json::json!({"not": {"type": "string"}}), "not"),
            (
                serde_json::json!({"type": "array", "contains": {"type": "integer"}}),
                "contains",
            ),
        ] {
            match env.json_schema(&schema) {
                Ok(_) => panic!("{keyword}: compiled, but the engine cannot enforce it"),
                Err(e) => assert!(
                    matches!(&e, ConstraintError::SchemaUnsupported(why) if why.contains(keyword)),
                    "{keyword}: {e}"
                ),
            }
        }
    }

    #[test]
    fn json_is_compact_by_default_and_the_callers_x_guidance_wins() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let space = id_of(&v, b" ");
        let brace = id_of(&v, b"{");
        let after_brace = |schema: serde_json::Value| {
            let mut c = env.json_schema(&schema).expect("compile");
            c.accept(brace).expect("an object starts with {");
            let mut logits = vec![0.0f32; v.token_bytes.len()];
            c.mask(&mut logits).expect("the mask after {");
            logits[space as usize].is_finite()
        };
        // default: no free whitespace, so no whitespace runaway
        assert!(
            !after_brace(serde_json::json!({"type": "object"})),
            "compact JSON admits no space after {{"
        );
        // a caller that chose flexible whitespace gets it
        assert!(after_brace(
            serde_json::json!({"type": "object", "x-guidance": {"whitespace_flexible": true}})
        ));
    }

    /// Feed `doc` through the constraint one byte token at a time (the vocabulary carries every
    /// byte at id 2 + byte). Ok(complete) when every byte is accepted, or the first refusal.
    fn feed(c: &mut dyn TokenConstraint, doc: &[u8]) -> Result<bool, (usize, ConstraintError)> {
        for (i, b) in doc.iter().enumerate() {
            c.accept(2 + u32::from(*b)).map_err(|e| (i, e))?;
        }
        Ok(c.is_complete())
    }

    fn consumer_schemas() -> Vec<(&'static str, serde_json::Value)> {
        vec![
            // paiml-implement agy/quorum-lane-schema.json (#3716, RAH)
            (
                "rah-lane",
                serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
                    "required": ["verdict", "summary", "findings"],
                    "properties": {
                        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "do-not-implement-as-written"]},
                        "summary": {"type": "string"}, "tree_witness": {"type": "string"},
                        "findings": {"type": "array", "items": {"type": "object", "required": ["file", "claim", "grounding"],
                            "properties": {"file": {"type": "string"}, "line": {"type": "integer"}, "claim": {"type": "string"},
                                "grounding": {"type": "string", "enum": ["cited", "measured", "asserted"]}, "fix": {"type": "string"}}}}
                    }
                }),
            ),
            // rmedia #3716 issuecomment-5764061597: quiz
            (
                "rmedia-quiz",
                serde_json::json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": false,
                    "required": ["quiz", "graded", "n", "stem", "options", "correct_index", "grounded_in"],
                    "properties": {
                        "quiz": {"type": "string", "minLength": 1, "maxLength": 120}, "graded": {"type": "boolean"},
                        "n": {"type": "integer", "minimum": 1}, "stem": {"type": "string", "minLength": 1, "maxLength": 300},
                        "options": {"type": "array", "minItems": 2, "maxItems": 6, "items": {"type": "string", "minLength": 1, "maxLength": 200}},
                        "correct_index": {"type": "integer", "minimum": 0, "maximum": 5},
                        "grounded_in": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+-.*\\.srt$"},
                        "option_feedback": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 300}},
                        "explanation": {"type": "string", "maxLength": 300}
                    }
                }),
            ),
            // rmedia: outline
            (
                "rmedia-outline",
                serde_json::json!({
                    "type": "object", "additionalProperties": false,
                    "required": ["kind", "course", "module", "index", "lesson", "description", "runtime", "seconds", "file", "rendered"],
                    "properties": {
                        "kind": {"const": "lesson"}, "course": {"type": "string", "minLength": 1},
                        "module": {"type": "string", "pattern": "^Module [0-9]+: .+"}, "index": {"type": "integer", "minimum": 1},
                        "lesson": {"type": "string", "minLength": 1}, "description": {"type": "string", "minLength": 1, "maxLength": 200},
                        "runtime": {"type": "string", "pattern": "^[0-9]+:[0-9]{2}$"}, "seconds": {"type": "number", "exclusiveMinimum": 0},
                        "file": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+-.*\\.mp4$"}, "rendered": {"type": "boolean"},
                        "instructor": {"type": "string"}, "level": {"enum": ["Beginner", "Intermediate", "Advanced"]}, "repo": {"type": "string"}
                    }
                }),
            ),
            // rmedia: key terms + reflection
            (
                "rmedia-keyterms",
                serde_json::json!({
                    "type": "object", "additionalProperties": false, "required": ["lesson", "key_terms", "reflection"],
                    "properties": {
                        "lesson": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+ — .+"},
                        "key_terms": {"type": "array", "minItems": 3, "maxItems": 8, "items": {"type": "object", "additionalProperties": false,
                            "required": ["term", "definition"], "properties": {"term": {"type": "string", "maxLength": 60},
                                "definition": {"type": "string", "minLength": 40, "maxLength": 600}}}},
                        "reflection": {"type": "string", "minLength": 80, "maxLength": 1200}
                    }
                }),
            ),
        ]
    }

    #[test]
    fn every_consumer_schema_compiles_exactly_and_admits_its_real_documents() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let def = "a definition that is comfortably longer than forty bytes";
        let keyterms = format!(
            r#"{{"lesson":"1.1 — Opening","key_terms":[{{"term":"count","definition":"{def}"}},{{"term":"drawer","definition":"{def}"}},{{"term":"definition","definition":"{def}"}}],"reflection":"{}"}}"#,
            "r".repeat(90)
        );
        let accepted: Vec<(&str, String)> = vec![
            ("rah-lane", r#"{"verdict":"PASS","summary":"ok","findings":[{"file":"a.rs","line":3,"claim":"c","grounding":"cited"}]}"#.to_string()),
            ("rmedia-quiz", r#"{"quiz":"Module 1: Counting honestly","graded":false,"n":1,"stem":"What is missing?","options":["A third count","A written definition"],"correct_index":1,"grounded_in":"1.1.1-first-count.srt"}"#.to_string()),
            ("rmedia-outline", r#"{"kind":"lesson","course":"Counting Honestly","module":"Module 1: Counting honestly","index":1,"lesson":"Opening","description":"Why two counts disagree","runtime":"2:10","seconds":130.0,"file":"1.1.0-opening.mp4","rendered":true,"level":"Intermediate"}"#.to_string()),
            ("rmedia-keyterms", keyterms),
        ];
        for (name, schema) in consumer_schemas() {
            let doc = &accepted
                .iter()
                .find(|(n, _)| *n == name)
                .expect("a document per schema")
                .1;
            let mut c = env
                .json_schema(&schema)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            match feed(c.as_mut(), doc.as_bytes()) {
                Ok(true) => {},
                Ok(false) => panic!("{name}: the real document was accepted but is not complete"),
                Err((at, e)) => panic!("{name}: byte {at} of a real document refused: {e}\n{doc}"),
            }
            // rmedia's framing request: ONE top-level value, so nothing may follow a complete one
            let mut logits = vec![0.0f32; v.token_bytes.len()];
            c.mask(&mut logits)
                .expect("the mask after a complete document");
            assert!(
                logits[EOS as usize].is_finite(),
                "{name}: EOS must be allowed"
            );
            assert_eq!(
                logits[2 + usize::from(b'{')],
                f32::NEG_INFINITY,
                "{name}: a second value must not start"
            );
        }
    }

    #[test]
    fn every_consumer_schema_refuses_its_violations_at_the_byte() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let schema = |name: &str| {
            consumer_schemas()
                .into_iter()
                .find(|(n, _)| *n == name)
                .expect("schema")
                .1
        };
        for (name, violating, why) in [
            ("rah-lane", r#"{"verdict":"MAYBE""#, "enum"),
            (
                "rmedia-quiz",
                r#"{"quiz":"q","graded":false,"n":0"#,
                "minimum 1",
            ),
            (
                "rmedia-quiz",
                r#"{"quiz":"q","graded":false,"n":1,"stem":"s","options":["a","b"],"correct_index":1,"grounded_in":"notes.txt""#,
                "pattern",
            ),
            (
                "rmedia-quiz",
                r#"{"quiz":"q","graded":false,"n":1,"stem":"s","options":["a","b"],"correct_index":1,"grounded_in":"1.1.1-a.srt","difficulty":"#,
                "additionalProperties false",
            ),
            ("rmedia-outline", r#"{"kind":"lecture""#, "const"),
            (
                "rmedia-outline",
                r#"{"kind":"lesson","course":"c","module":"Chapter 1""#,
                "pattern",
            ),
            (
                "rmedia-keyterms",
                r#"{"lesson":"1.1 — O","key_terms":[{"term":"t","definition":"too short"}"#,
                "minLength 40",
            ),
        ] {
            let mut c = env
                .json_schema(&schema(name))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            match feed(c.as_mut(), violating.as_bytes()) {
                Err((_, ConstraintError::Rejected { .. })) => {},
                other => panic!("{name} ({why}): the violation was not refused: {other:?}"),
            }
        }
    }

    #[test]
    fn a_caller_cannot_buy_an_approximation_through_x_guidance() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        for schema in [
            serde_json::json!({"type": "string", "format": "not-a-format", "x-guidance": {"lenient": true}}),
            serde_json::json!({"oneOf": [{"type": "integer"}, {"type": "number"}], "x-guidance": {"coerce_one_of": true}}),
        ] {
            match env.json_schema(&schema) {
                Ok(_) => panic!("compiled approximately and was accepted: {schema}"),
                Err(e) => assert!(
                    matches!(&e, ConstraintError::SchemaUnsupported(why) if why.contains("only approximately")),
                    "{e}"
                ),
            }
        }
    }

    #[test]
    fn boolean_schemas_compile_compact_or_refuse() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        // `true` is any JSON value, and it must still be compact: no whitespace after `{`
        let mut c = env
            .json_schema(&serde_json::json!(true))
            .expect("`true` compiles");
        c.accept(id_of(&v, b"{"))
            .expect("any value may be an object");
        let mut logits = vec![0.0f32; v.token_bytes.len()];
        c.mask(&mut logits).expect("the mask after {");
        assert_eq!(
            logits[id_of(&v, b" ") as usize],
            f32::NEG_INFINITY,
            "`true` must compile compact"
        );
        // `false` admits nothing: refused before the first token, never a dead end at token 0
        let e = env
            .json_schema(&serde_json::json!(false))
            .err()
            .expect("`false` is refused");
        assert!(
            matches!(&e, ConstraintError::SchemaInvalid(why) if why.contains("admits no document")),
            "{e}"
        );
    }

    #[test]
    fn a_lark_grammar_is_enforced_too() {
        let v = vocab();
        let env = ConstraintEnv::new(&v).expect("index the vocabulary");
        let mut c = env
            .lark(r#"start: "PASS" | "FAIL""#)
            .expect("compile the grammar");
        let out = generate(&v, Some(c.as_mut()), 16).expect("constrained run");
        assert!(
            out == b"PASS" || out == b"FAIL",
            "{}",
            String::from_utf8_lossy(&out)
        );
    }
}
