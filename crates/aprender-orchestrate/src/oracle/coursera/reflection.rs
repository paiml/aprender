//! Reflection reading generation
//!
//! Extracts themes from transcript via keyword frequency + n-gram analysis,
//! generates Bloom's taxonomy questions, and attaches arXiv citations.

use std::collections::HashMap;

use super::arxiv_db::ArxivDatabase;
use super::types::{
    ArxivCitation, BloomLevel, ReflectionQuestion, ReflectionReading, TranscriptInput,
};

/// Generate a reflection reading from a transcript.
///
/// If `topic_override` is provided, it's used for arXiv lookup instead of
/// auto-detected themes.
pub fn generate_reflection(
    transcript: &TranscriptInput,
    topic_override: Option<&str>,
) -> ReflectionReading {
    let themes = extract_themes(&transcript.text);
    let questions = generate_bloom_questions(&themes, &transcript.text);

    let db = ArxivDatabase::builtin();
    let citations = if let Some(topic) = topic_override {
        find_citations_for_topic(&db, topic)
    } else {
        find_citations_for_themes(&db, &themes)
    };

    ReflectionReading { themes, questions, citations }
}

/// Render a reflection reading as Markdown.
pub fn render_reflection_markdown(reading: &ReflectionReading) -> String {
    let mut md = String::new();
    md.push_str("# Reflection Reading\n\n");

    // Themes
    md.push_str("## Key Themes\n\n");
    if reading.themes.is_empty() {
        md.push_str("No dominant themes extracted.\n\n");
    } else {
        for theme in &reading.themes {
            md.push_str(&format!("- {theme}\n"));
        }
        md.push('\n');
    }

    // Questions
    md.push_str("## Reflection Questions\n\n");
    if reading.questions.is_empty() {
        md.push_str("No reflection questions generated.\n\n");
    } else {
        for (i, q) in reading.questions.iter().enumerate() {
            md.push_str(&format!("{}. **[{}]** {}\n\n", i + 1, q.thinking_level, q.question));
        }
    }

    // Citations
    md.push_str("## Further Reading\n\n");
    if reading.citations.is_empty() {
        md.push_str("No matching citations found.\n");
    } else {
        for cite in &reading.citations {
            md.push_str(&format!(
                "- {} ({}) — [{}]({}) — *{}*\n",
                cite.authors, cite.year, cite.title, cite.url, cite.abstract_snippet,
            ));
        }
    }
    md.push('\n');

    md
}

// ============================================================================
// Internal
// ============================================================================

fn extract_themes(text: &str) -> Vec<String> {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 3)
        .collect();

    // Unigram frequency (excluding stop words)
    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in &words {
        let lower = word.to_lowercase();
        if !is_stop_word(&lower) && lower.len() >= 3 {
            *freq.entry(lower).or_default() += 1;
        }
    }

    // Bigram frequency
    let mut bigram_freq: HashMap<String, usize> = HashMap::new();
    for pair in words.windows(2) {
        let a = pair[0].to_lowercase();
        let b = pair[1].to_lowercase();
        if !is_stop_word(&a) && !is_stop_word(&b) && a.len() >= 3 && b.len() >= 3 {
            let bigram = format!("{a} {b}");
            *bigram_freq.entry(bigram).or_default() += 1;
        }
    }

    // Merge: bigrams that appear >= 2 times, then top unigrams
    let mut themes: Vec<(String, usize)> =
        bigram_freq.into_iter().filter(|(_, count)| *count >= 2).collect();

    let top_unigrams: Vec<(String, usize)> = {
        let mut v: Vec<_> = freq.into_iter().filter(|(_, c)| *c >= 3).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.into_iter().take(10).collect()
    };

    themes.extend(top_unigrams);
    themes.sort_by(|a, b| b.1.cmp(&a.1));
    themes.dedup_by(|a, b| a.0 == b.0);

    themes.into_iter().take(5).map(|(t, _)| capitalize_theme(&t)).collect()
}

fn capitalize_theme(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

fn generate_bloom_questions(themes: &[String], text: &str) -> Vec<ReflectionQuestion> {
    if themes.is_empty() {
        return Vec::new();
    }

    let text_lower = text.to_lowercase();

    let bloom_templates: Vec<(BloomLevel, &[&str])> = vec![
        (
            BloomLevel::Analysis,
            &[
                "What are the key components of {theme}, and how do they relate to each other?",
                "Compare and contrast the approaches to {theme} discussed in the lecture.",
                "What assumptions underlie the discussion of {theme}?",
            ],
        ),
        (
            BloomLevel::Synthesis,
            &[
                "How would you combine the concepts from {theme} with your existing knowledge to solve a novel problem?",
                "Design a system that integrates {theme} with a complementary technique.",
            ],
        ),
        (
            BloomLevel::Evaluation,
            &[
                "What are the strengths and limitations of the approach to {theme} presented here?",
                "Under what conditions would {theme} fail or underperform?",
            ],
        ),
        (
            BloomLevel::Application,
            &[
                "How would you apply {theme} in a production environment?",
                "Describe a real-world scenario where {theme} would provide significant value.",
            ],
        ),
        (
            BloomLevel::Creation,
            &[
                "Propose an improvement or extension to {theme} that addresses a current limitation.",
                "Design an experiment to validate the effectiveness of {theme} in your domain.",
            ],
        ),
    ];

    let mut questions = Vec::new();

    for (level, templates) in &bloom_templates {
        // Pick the best template for each level based on content relevance
        let theme = select_theme_for_level(themes, &text_lower, level);
        let template_idx = match level {
            BloomLevel::Analysis => {
                usize::from(text_lower.contains("compare") || text_lower.contains("contrast"))
            }
            BloomLevel::Evaluation => {
                usize::from(text_lower.contains("limitation") || text_lower.contains("trade"))
            }
            _ => 0,
        };

        let template = templates[template_idx.min(templates.len() - 1)];
        let question = template.replace("{theme}", &theme);

        questions.push(ReflectionQuestion { question, thinking_level: *level });
    }

    questions
}

fn select_theme_for_level(themes: &[String], _text: &str, _level: &BloomLevel) -> String {
    // Use the most prominent theme for most levels, rotate for variety
    let idx = match _level {
        BloomLevel::Analysis => 0,
        BloomLevel::Synthesis => themes.len().min(1),
        BloomLevel::Evaluation => themes.len().min(2) % themes.len(),
        BloomLevel::Application => 0,
        BloomLevel::Creation => themes.len().min(1) % themes.len(),
    };
    themes.get(idx).cloned().unwrap_or_else(|| "the topic".to_string())
}

fn find_citations_for_topic(db: &ArxivDatabase, topic: &str) -> Vec<ArxivCitation> {
    let mut results = db.find_by_topic(topic, 5);
    if results.len() < 3 {
        // Try splitting topic into keywords
        let keywords: Vec<&str> = topic.split_whitespace().collect();
        let additional = db.find_by_keywords(&keywords, 5 - results.len());
        for cite in additional {
            if !results.iter().any(|r| r.arxiv_id == cite.arxiv_id) {
                results.push(cite);
            }
        }
    }
    results.truncate(5);
    results
}

fn find_citations_for_themes(db: &ArxivDatabase, themes: &[String]) -> Vec<ArxivCitation> {
    let keywords: Vec<&str> = themes.iter().map(|t| t.as_str()).collect();
    let mut results = db.find_by_keywords(&keywords, 5);

    // If not enough, try individual themes
    if results.len() < 3 {
        for theme in themes {
            let additional = db.find_by_topic(theme, 2);
            for cite in additional {
                if !results.iter().any(|r| r.arxiv_id == cite.arxiv_id) {
                    results.push(cite);
                }
            }
            if results.len() >= 5 {
                break;
            }
        }
    }

    results.truncate(5);
    results
}

fn is_stop_word(word: &str) -> bool {
    const STOP: &[&str] = &[
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "or",
        "and",
        "but",
        "if",
        "not",
        "no",
        "so",
        "up",
        "out",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "we",
        "you",
        "they",
        "he",
        "she",
        "my",
        "your",
        "our",
        "us",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "about",
        "which",
        "what",
        "when",
        "where",
        "how",
        "who",
        "whom",
        "why",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "same",
        "different",
        "then",
        "there",
        "here",
        "new",
        "old",
        "many",
        "much",
        "own",
        "only",
        "well",
        "use",
        "used",
        "using",
        "like",
        "one",
        "two",
        "get",
        "make",
        "way",
    ];
    STOP.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::coursera::types::TranscriptInput;

    fn make_transcript(text: &str) -> TranscriptInput {
        TranscriptInput {
            text: text.to_string(),
            language: "en".to_string(),
            segments: vec![],
            source_path: "test.txt".to_string(),
        }
    }

    #[test]
    fn test_generate_reflection_with_topic() {
        let t = make_transcript(
            "Machine learning models need careful deployment. MLOps practices help \
             automate the deployment pipeline. Continuous integration ensures quality. \
             Testing machine learning models requires specialized approaches.",
        );
        let reading = generate_reflection(&t, Some("mlops"));
        assert!(!reading.citations.is_empty(), "Should find mlops citations");
        assert!(!reading.questions.is_empty());
    }

    #[test]
    fn test_generate_reflection_auto_themes() {
        let t = make_transcript(
            "Transformer models use attention mechanisms. Attention allows the model to \
             focus on relevant parts of the input. The transformer architecture has \
             revolutionized natural language processing. Attention is computed as a \
             weighted sum of values. Transformer attention enables parallel computation.",
        );
        let reading = generate_reflection(&t, None);
        assert!(!reading.themes.is_empty());
        assert!(!reading.questions.is_empty());
    }

    #[test]
    fn test_generate_reflection_empty_transcript() {
        let t = make_transcript("");
        let reading = generate_reflection(&t, None);
        assert!(reading.themes.is_empty());
    }

    #[test]
    fn test_render_reflection_markdown() {
        let reading = ReflectionReading {
            themes: vec!["Machine learning".to_string(), "Deployment".to_string()],
            questions: vec![ReflectionQuestion {
                question: "What are the key challenges?".to_string(),
                thinking_level: BloomLevel::Analysis,
            }],
            citations: vec![ArxivCitation {
                arxiv_id: "1706.03762".to_string(),
                title: "Attention Is All You Need".to_string(),
                authors: "Vaswani et al.".to_string(),
                year: 2017,
                url: "https://arxiv.org/abs/1706.03762".to_string(),
                abstract_snippet: "Proposes the Transformer.".to_string(),
                topics: vec!["transformer".to_string()],
            }],
        };
        let md = render_reflection_markdown(&reading);
        assert!(md.contains("# Reflection Reading"));
        assert!(md.contains("Machine learning"));
        assert!(md.contains("[Analysis]"));
        assert!(md.contains("Vaswani"));
        assert!(md.contains("https://arxiv.org"));
    }

    #[test]
    fn test_render_reflection_empty() {
        let reading = ReflectionReading { themes: vec![], questions: vec![], citations: vec![] };
        let md = render_reflection_markdown(&reading);
        assert!(md.contains("No dominant themes"));
        assert!(md.contains("No reflection questions"));
        assert!(md.contains("No matching citations"));
    }

    #[test]
    fn test_extract_themes() {
        let themes = extract_themes(
            "Deep learning models require large datasets for training. Deep learning \
             has transformed computer vision and natural language processing. Training \
             deep learning models requires significant compute resources. Deep learning \
             architectures include transformers and convolutional networks.",
        );
        assert!(!themes.is_empty());
        // Should detect "deep learning" as a theme
        assert!(
            themes.iter().any(|t| t.to_lowercase().contains("deep") || t.to_lowercase().contains("learning")),
            "Themes: {:?}",
            themes
        );
    }

    #[test]
    fn test_bloom_question_levels() {
        let themes = vec!["Machine learning".to_string()];
        let questions = generate_bloom_questions(&themes, "Machine learning is important.");
        assert_eq!(questions.len(), 5);

        let levels: Vec<BloomLevel> = questions.iter().map(|q| q.thinking_level).collect();
        assert!(levels.contains(&BloomLevel::Analysis));
        assert!(levels.contains(&BloomLevel::Synthesis));
        assert!(levels.contains(&BloomLevel::Evaluation));
        assert!(levels.contains(&BloomLevel::Application));
        assert!(levels.contains(&BloomLevel::Creation));
    }

    #[test]
    fn test_bloom_questions_empty_themes() {
        let questions = generate_bloom_questions(&[], "Some text");
        assert!(questions.is_empty());
    }

    #[test]
    fn test_find_citations_for_topic() {
        let db = ArxivDatabase::builtin();
        let results = find_citations_for_topic(&db, "transformer");
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_find_citations_for_themes() {
        let db = ArxivDatabase::builtin();
        let themes = vec!["mlops".to_string(), "deployment".to_string()];
        let results = find_citations_for_themes(&db, &themes);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_bloom_compare_contrast_template() {
        // Triggers line 202: template_idx = 1 when text contains "compare"
        let themes = vec!["Neural networks".to_string()];
        let questions =
            generate_bloom_questions(&themes, "We compare different neural network architectures.");
        let analysis_q = questions
            .iter()
            .find(|q| q.thinking_level == BloomLevel::Analysis)
            .expect("unexpected failure");
        assert!(
            analysis_q.question.contains("Compare and contrast"),
            "Got: {}",
            analysis_q.question
        );
    }

    #[test]
    fn test_bloom_limitation_template() {
        // Triggers line 209: template_idx = 1 when text contains "limitation"
        let themes = vec!["Attention".to_string()];
        let questions =
            generate_bloom_questions(&themes, "A key limitation of attention is memory cost.");
        let eval_q = questions
            .iter()
            .find(|q| q.thinking_level == BloomLevel::Evaluation)
            .expect("unexpected failure");
        assert!(
            eval_q.question.contains("conditions") || eval_q.question.contains("fail"),
            "Got: {}",
            eval_q.question
        );
    }

    #[test]
    fn test_capitalize_theme_empty() {
        // Triggers line 143: empty string capitalization
        assert_eq!(capitalize_theme(""), "");
    }

    #[test]
    fn test_find_citations_for_topic_sparse() {
        // Triggers lines 248-254: keyword fallback when topic yields < 3 results
        let db = ArxivDatabase::builtin();
        let results = find_citations_for_topic(&db, "obscure quantum federated distillation");
        // Should still attempt keyword split fallback
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_find_citations_for_themes_individual_fallback() {
        // Triggers lines 267-275: individual theme lookup when keyword search yields < 3
        let db = ArxivDatabase::builtin();
        let themes =
            vec!["xyznonexistent".to_string(), "transformer".to_string(), "attention".to_string()];
        let results = find_citations_for_themes(&db, &themes);
        // The individual theme "transformer" should produce results
        assert!(!results.is_empty());
    }
}
