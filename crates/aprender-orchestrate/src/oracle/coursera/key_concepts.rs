//! Key Concepts reading generation
//!
//! Extracts technical concepts with definitions, categorizes them,
//! and inserts code examples when transcript mentions programming patterns.

use std::collections::HashMap;

use super::types::{CodeExample, Concept, ConceptCategory, KeyConceptsReading, TranscriptInput};
use super::vocabulary;

/// Generate a key concepts reading from a transcript.
pub fn generate_key_concepts(transcript: &TranscriptInput) -> KeyConceptsReading {
    let concepts = extract_concepts(transcript);
    let code_examples = extract_code_examples(transcript, &concepts);

    KeyConceptsReading { concepts, code_examples }
}

/// Render key concepts as Markdown.
pub fn render_key_concepts_markdown(reading: &KeyConceptsReading) -> String {
    let mut md = String::new();
    md.push_str("# Key Concepts\n\n");

    if reading.concepts.is_empty() {
        md.push_str("No key concepts extracted from this transcript.\n");
        return md;
    }

    // Group by category
    let mut by_category: HashMap<&str, Vec<&Concept>> = HashMap::new();
    for concept in &reading.concepts {
        by_category.entry(concept.category.as_str()).or_default().push(concept);
    }

    let mut categories: Vec<&&str> = by_category.keys().collect();
    categories.sort();

    for cat in categories {
        let cat_concepts = &by_category[*cat];
        md.push_str(&format!("## {}\n\n", cat));
        md.push_str("| Concept | Definition |\n");
        md.push_str("|---------|------------|\n");

        for concept in cat_concepts {
            md.push_str(&format!("| **{}** | {} |\n", concept.term, concept.definition));
        }
        md.push('\n');

        // Add context quotes
        for concept in cat_concepts {
            if !concept.context.is_empty() {
                md.push_str(&format!("> *\"{}\"*\n\n", concept.context));
            }
        }
    }

    // Code examples
    if !reading.code_examples.is_empty() {
        md.push_str("## Code Examples\n\n");
        for example in &reading.code_examples {
            md.push_str(&format!(
                "### {} ({})\n\n```{}\n{}\n```\n\n",
                example.related_concept, example.language, example.language, example.code
            ));
        }
    }

    md
}

// ============================================================================
// Internal
// ============================================================================

fn extract_concepts(transcript: &TranscriptInput) -> Vec<Concept> {
    let vocab = vocabulary::extract_vocabulary(std::slice::from_ref(transcript));
    let sentences = split_sentences(&transcript.text);

    let mut concepts: Vec<Concept> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in &vocab {
        let lower = entry.term.to_lowercase();
        if seen.contains(&lower) {
            continue;
        }
        seen.insert(lower.clone());

        let context = find_best_context(&sentences, &entry.term);
        let definition = if entry.definition.len() > 10 {
            entry.definition.clone()
        } else {
            derive_concept_definition(&sentences, &entry.term)
        };

        concepts.push(Concept {
            term: entry.term.clone(),
            definition,
            context,
            category: entry.category,
        });
    }

    // Also extract definition patterns not caught by vocab extraction
    for sentence in &sentences {
        if let Some(concept) = try_extract_definition_pattern(sentence) {
            let lower = concept.term.to_lowercase();
            if !seen.contains(&lower) {
                seen.insert(lower);
                concepts.push(concept);
            }
        }
    }

    // Limit to top 15 concepts
    concepts.truncate(15);
    concepts
}

fn find_best_context(sentences: &[String], term: &str) -> String {
    let lower_term = term.to_lowercase();

    // Prefer sentences with definitional patterns
    for s in sentences {
        let lower = s.to_lowercase();
        if lower.contains(&lower_term) && (lower.contains(" is ") || lower.contains(" are ")) {
            return truncate(s, 150);
        }
    }

    // Fall back to first mention
    for s in sentences {
        if s.to_lowercase().contains(&lower_term) {
            return truncate(s, 150);
        }
    }

    String::new()
}

fn derive_concept_definition(sentences: &[String], term: &str) -> String {
    let lower_term = term.to_lowercase();

    for sentence in sentences {
        let lower = sentence.to_lowercase();

        // "X is ..."
        if let Some(pos) = lower.find(&format!("{} is ", lower_term)) {
            let start = pos + lower_term.len() + 4;
            if let Some(def) = sentence.get(start..) {
                let end = def.find('.').unwrap_or(def.len()).min(120);
                return capitalize_first(safe_truncate_bytes(def, end).trim());
            }
        }

        // "X, also known as ..."
        if let Some(pos) = lower.find(&format!("{}, also known as ", lower_term)) {
            let start = pos + lower_term.len() + 17;
            if let Some(def) = sentence.get(start..) {
                let end = def.find('.').unwrap_or(def.len()).min(120);
                return format!("Also known as {}", safe_truncate_bytes(def, end).trim());
            }
        }
    }

    format!("Technical concept: {term}")
}

fn try_extract_definition_pattern(sentence: &str) -> Option<Concept> {
    let patterns = [" is a ", " is an ", " is the ", " refers to "];
    let lower = sentence.to_lowercase();

    patterns.iter().find_map(|pat| try_match_definition(sentence, &lower, pat))
}

fn try_match_definition(sentence: &str, lower: &str, pat: &str) -> Option<Concept> {
    let pos = lower.find(pat)?;

    let term = extract_term_before(sentence, pos);
    if term.len() < 3 || term.chars().next().is_some_and(|c| c.is_lowercase()) {
        return None;
    }

    let def_start = pos + pat.len();
    let definition = sentence.get(def_start..)?;
    let end = definition.find('.').unwrap_or(definition.len()).min(120);
    let definition = capitalize_first(safe_truncate_bytes(definition, end).trim());

    if definition.len() < 5 {
        return None;
    }

    Some(Concept {
        term: term.trim().to_string(),
        definition,
        context: truncate(sentence, 150),
        category: ConceptCategory::General,
    })
}

fn extract_term_before(sentence: &str, pos: usize) -> String {
    sentence
        .get(..pos)
        .unwrap_or("")
        .split_whitespace()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_code_examples(transcript: &TranscriptInput, concepts: &[Concept]) -> Vec<CodeExample> {
    let mut examples = Vec::new();
    let text_lower = transcript.text.to_lowercase();

    extract_bash_examples(&text_lower, concepts, &mut examples);
    extract_language_example(
        &text_lower,
        concepts,
        &mut examples,
        &["python", "import", "pytorch"],
        "python",
        &["python", "pytorch", "model"],
        "Python",
        "import torch\nmodel = torch.load(\"model.pt\")\noutput = model(input_tensor)",
    );
    extract_language_example(
        &text_lower,
        concepts,
        &mut examples,
        &["rust", "cargo", "trueno"],
        "rust",
        &["rust", "cargo", "trueno"],
        "Rust",
        "use trueno::Tensor;\nlet data = Tensor::from_slice(&[1.0, 2.0, 3.0]);\nlet result = data.matmul(&weights)?;",
    );

    examples.truncate(5);
    examples
}

fn extract_bash_examples(text_lower: &str, concepts: &[Concept], examples: &mut Vec<CodeExample>) {
    let bash_patterns: &[(&str, &str)] = &[
        ("docker", "docker run -p 8080:8080 model-server"),
        ("pip", "pip install torch transformers"),
        ("cargo", "cargo build --release"),
        ("kubectl", "kubectl apply -f deployment.yaml"),
        ("curl", "curl -X POST http://localhost:8080/predict -d '{\"input\": \"text\"}'"),
        ("git", "git clone https://github.com/org/repo.git"),
    ];

    for (keyword, code) in bash_patterns {
        if text_lower.contains(keyword) {
            let related = find_related_concept(concepts, &[keyword])
                .unwrap_or_else(|| (*keyword).to_string());
            examples.push(CodeExample {
                language: "bash".to_string(),
                code: (*code).to_string(),
                related_concept: related,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_language_example(
    text_lower: &str,
    concepts: &[Concept],
    examples: &mut Vec<CodeExample>,
    triggers: &[&str],
    language: &str,
    concept_keywords: &[&str],
    fallback_name: &str,
    code: &str,
) {
    if triggers.iter().any(|t| text_lower.contains(t)) {
        let related = find_related_concept(concepts, concept_keywords)
            .unwrap_or_else(|| fallback_name.to_string());
        examples.push(CodeExample {
            language: language.to_string(),
            code: code.to_string(),
            related_concept: related,
        });
    }
}

fn find_related_concept(concepts: &[Concept], keywords: &[&str]) -> Option<String> {
    concepts
        .iter()
        .find(|c| {
            let cl = c.term.to_lowercase();
            keywords.iter().any(|kw| cl.contains(kw))
        })
        .map(|c| c.term.clone())
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", safe_truncate_bytes(s, max))
    }
}

/// Truncate a string at the nearest char boundary at or before `max_bytes`.
fn safe_truncate_bytes(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::coursera::types::TranscriptSegment;

    fn make_transcript(text: &str) -> TranscriptInput {
        TranscriptInput {
            text: text.to_string(),
            language: "en".to_string(),
            segments: vec![],
            source_path: "test.txt".to_string(),
        }
    }

    #[test]
    fn test_generate_key_concepts() {
        let t = make_transcript(
            "MLOps is the practice of deploying ML models to production. \
             MLOps combines DevOps and machine learning workflows. \
             CI/CD pipelines automate the deployment process. \
             CI/CD is essential for reliable software delivery. \
             GPU acceleration speeds up model inference. \
             GPU computing enables parallel processing.",
        );
        let reading = generate_key_concepts(&t);
        assert!(!reading.concepts.is_empty());
    }

    #[test]
    fn test_generate_key_concepts_empty() {
        let t = make_transcript("The cat sat on the mat.");
        let reading = generate_key_concepts(&t);
        assert!(reading.concepts.is_empty());
    }

    #[test]
    fn test_render_key_concepts_markdown() {
        let reading = KeyConceptsReading {
            concepts: vec![Concept {
                term: "MLOps".to_string(),
                definition: "Machine Learning Operations".to_string(),
                context: "MLOps combines ML and DevOps.".to_string(),
                category: ConceptCategory::Pattern,
            }],
            code_examples: vec![CodeExample {
                language: "bash".to_string(),
                code: "docker run app".to_string(),
                related_concept: "Docker".to_string(),
            }],
        };
        let md = render_key_concepts_markdown(&reading);
        assert!(md.contains("# Key Concepts"));
        assert!(md.contains("MLOps"));
        assert!(md.contains("## Code Examples"));
        assert!(md.contains("```bash"));
    }

    #[test]
    fn test_render_key_concepts_empty() {
        let reading = KeyConceptsReading { concepts: vec![], code_examples: vec![] };
        let md = render_key_concepts_markdown(&reading);
        assert!(md.contains("No key concepts extracted"));
    }

    #[test]
    fn test_extract_code_examples_bash() {
        let t = make_transcript(
            "We use docker to deploy our models. Docker containers are lightweight.",
        );
        let concepts = vec![Concept {
            term: "Docker".to_string(),
            definition: "Container runtime".to_string(),
            context: "".to_string(),
            category: ConceptCategory::Tool,
        }];
        let examples = extract_code_examples(&t, &concepts);
        assert!(!examples.is_empty());
        assert_eq!(examples[0].language, "bash");
    }

    #[test]
    fn test_extract_code_examples_python() {
        let t = make_transcript("Python and PyTorch are used for model training. Python scripts handle data processing.");
        let concepts = vec![];
        let examples = extract_code_examples(&t, &concepts);
        let python_example = examples.iter().find(|e| e.language == "python");
        assert!(python_example.is_some());
    }

    #[test]
    fn test_extract_code_examples_rust() {
        let t = make_transcript(
            "Rust and cargo are used for high-performance computing. Rust provides memory safety.",
        );
        let concepts = vec![];
        let examples = extract_code_examples(&t, &concepts);
        let rust_example = examples.iter().find(|e| e.language == "rust");
        assert!(rust_example.is_some());
    }

    #[test]
    fn test_try_extract_definition_pattern() {
        let result = try_extract_definition_pattern(
            "Batch Normalization is a technique that normalizes layer inputs.",
        );
        assert!(result.is_some());
        let concept = result.expect("operation failed");
        assert!(concept.term.contains("Normalization"));
    }

    #[test]
    fn test_duplicate_terms_deduplicated() {
        // Triggers line 94: duplicate term skip via `seen`
        let t = make_transcript(
            "MLOps is the practice of deploying ML models. MLOps automates deployment. \
             MLOps combines DevOps and ML. MLOps pipelines handle continuous delivery. \
             MLOps teams build reliable systems.",
        );
        let reading = generate_key_concepts(&t);
        let mlops_count =
            reading.concepts.iter().filter(|c| c.term.to_lowercase() == "mlops").count();
        assert!(mlops_count <= 1, "MLOps should appear at most once");
    }

    #[test]
    fn test_derive_concept_definition_is_pattern() {
        // Triggers derive_concept_definition "X is ..." pattern (lines 157-163)
        let sentences =
            vec!["Kubernetes is an open-source container orchestration platform.".to_string()];
        let def = super::derive_concept_definition(&sentences, "Kubernetes");
        assert!(def.contains("open-source") || def.contains("container"), "Got: {def}");
    }

    #[test]
    fn test_derive_concept_definition_also_known_as() {
        // Triggers "also known as" pattern (lines 166-172)
        let sentences = vec!["K8s, also known as Kubernetes container orchestration.".to_string()];
        let def = super::derive_concept_definition(&sentences, "K8s");
        assert!(def.starts_with("Also known as"), "Got: {def}");
    }

    #[test]
    fn test_derive_concept_definition_fallback() {
        // Triggers fallback path (line 175) when no definition pattern matches
        let sentences = vec!["Random text about something.".to_string()];
        let def = super::derive_concept_definition(&sentences, "QUIC");
        assert!(def.contains("Technical concept: QUIC"), "Got: {def}");
    }

    #[test]
    fn test_find_best_context_no_match() {
        // Triggers line 147: no matching sentence returns empty string
        let sentences = vec!["The cat sat on the mat.".to_string()];
        let ctx = super::find_best_context(&sentences, "kubernetes");
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_try_match_definition_short_term_rejected() {
        // Triggers line 192: term too short (< 3 chars)
        let result = super::try_match_definition("It is a test.", "it is a test.", " is a ");
        assert!(result.is_none());
    }

    #[test]
    fn test_try_match_definition_short_definition_rejected() {
        // Triggers line 200-201: definition too short (< 5 chars)
        let result = try_extract_definition_pattern("BigThing is a ok.");
        assert!(result.is_none());
    }

    #[test]
    fn test_truncate_long_string() {
        // Triggers line 341: format!("{}...", safe_truncate_bytes(s, max))
        let long = "a".repeat(200);
        let result = super::truncate(&long, 50);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 54); // 50 + "..."
    }

    #[test]
    fn test_safe_truncate_bytes_multibyte() {
        // Triggers lines 348-353: char boundary adjustment
        let s = "héllo wörld";
        let truncated = super::safe_truncate_bytes(s, 3);
        // Should not panic; 'é' is 2 bytes, so position 3 may fall mid-char
        assert!(!truncated.is_empty());
        assert!(s.is_char_boundary(truncated.len()));
    }

    #[test]
    fn test_capitalize_first_empty() {
        // Triggers line 360: empty string case
        assert_eq!(super::capitalize_first(""), "");
    }

    #[test]
    fn test_split_sentences_trailing_text() {
        // Triggers lines 329-331: trailing text without terminal punctuation
        let sentences = super::split_sentences("Hello world. This has no period");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[1], "This has no period");
    }

    #[test]
    fn test_definition_pattern_refers_to() {
        // Triggers " refers to " pattern in try_extract_definition_pattern
        let result =
            try_extract_definition_pattern("MLOps refers to the practice of operationalizing ML.");
        assert!(result.is_some());
        let concept = result.expect("operation failed");
        assert!(
            concept.definition.contains("practice")
                || concept.definition.contains("operationalizing")
        );
    }

    #[test]
    fn test_concepts_with_segments() {
        let t = TranscriptInput {
            text: "API endpoints serve ML predictions. The API handles inference. \
                   GPU acceleration is critical. GPU kernels run fast."
                .to_string(),
            language: "en".to_string(),
            segments: vec![TranscriptSegment {
                start: 0.0,
                end: 10.0,
                text: "API endpoints serve ML predictions.".to_string(),
            }],
            source_path: "test.json".to_string(),
        };
        let reading = generate_key_concepts(&t);
        assert!(!reading.concepts.is_empty());
    }
}
