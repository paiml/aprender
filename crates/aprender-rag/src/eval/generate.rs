//! Synthetic ground truth generation from corpus chunks

use super::client::AnthropicClient;
use super::domain::{classify_domain, extract_course_dir};
use super::types::GroundTruthEntry;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;

const SYSTEM_PROMPT: &str = "You generate evaluation questions from video transcript chunks.
Given a transcript chunk, generate ONE specific question this text answers.
Rules:
(1) The question must be answerable only from the provided text.
(2) Write a student-style query, 8-20 words long.
(3) Do NOT reference \"the video\", \"the instructor\", \"the speaker\", or \"this lecture\".
(4) Do NOT ask yes/no questions.
(5) If the text is too vague or navigational to generate a good question, respond with exactly: SKIP";

/// Chunk data extracted from a PersistedIndex
#[derive(Debug, Clone)]
pub struct IndexChunk {
    /// Chunk text content
    pub content: String,
    /// Source file path
    pub source: String,
    /// Optional title
    pub title: Option<String>,
    /// Start timestamp
    pub start_secs: Option<f64>,
    /// End timestamp
    pub end_secs: Option<f64>,
}

/// Generator for synthetic ground truth
pub struct GroundTruthGenerator {
    client: AnthropicClient,
    model: String,
    sample_size: usize,
    seed: u64,
}

impl GroundTruthGenerator {
    /// Create a new generator
    pub fn new(client: AnthropicClient, model: &str, sample_size: usize, seed: u64) -> Self {
        Self { client, model: model.to_string(), sample_size, seed }
    }

    /// Sample chunks using stratified sampling by course directory
    pub fn sample_chunks(&self, chunks: &[IndexChunk]) -> Vec<SampledChunk> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        // Group by course
        let mut by_course: HashMap<String, Vec<&IndexChunk>> = HashMap::new();
        for chunk in chunks {
            let course = extract_course_dir(&chunk.source).to_string();
            by_course.entry(course).or_default().push(chunk);
        }

        // Sort courses by chunk count descending, then by name for determinism
        let mut courses: Vec<(String, Vec<&IndexChunk>)> = by_course.into_iter().collect();
        courses.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

        let mut sampled = Vec::new();

        for (course, course_chunks) in &courses {
            // Filter eligible chunks
            let eligible: Vec<&&IndexChunk> =
                course_chunks.iter().filter(|c| is_eligible(c)).collect();

            if eligible.len() < 2 {
                continue;
            }

            // Sample 2-3 chunks per course
            let n = eligible.len().min(3);
            let mut indices: Vec<usize> = (0..eligible.len()).collect();
            indices.shuffle(&mut rng);

            for &idx in indices.iter().take(n) {
                let chunk = eligible[idx];
                sampled.push(SampledChunk {
                    content: chunk.content.clone(),
                    source: chunk.source.clone(),
                    start_secs: chunk.start_secs,
                    end_secs: chunk.end_secs,
                    course: course.clone(),
                    domain: classify_domain(course).to_string(),
                });
            }

            if sampled.len() >= self.sample_size {
                break;
            }
        }

        // Trim to exact size
        sampled.truncate(self.sample_size);

        // Report distribution
        let mut domain_counts: HashMap<&str, usize> = HashMap::new();
        for s in &sampled {
            *domain_counts.entry(&s.domain).or_default() += 1;
        }
        eprintln!(
            "Sampled {} chunks from {} courses",
            sampled.len(),
            courses.len().min(sampled.len())
        );
        let mut sorted_domains: Vec<_> = domain_counts.into_iter().collect();
        sorted_domains.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (domain, count) in &sorted_domains {
            eprintln!("  {domain}: {count}");
        }

        sampled
    }

    /// Generate a question for a single chunk
    pub async fn generate_question(&self, content: &str) -> Result<Option<String>, String> {
        let user_msg = format!("Transcript chunk:\n---\n{content}\n---");

        let result = self.client.complete(&self.model, Some(SYSTEM_PROMPT), &user_msg, 150).await?;

        let text = result.text.trim().to_string();
        if text == "SKIP" || text.starts_with("SKIP") {
            return Ok(None);
        }

        // Clean up
        let mut question = text.trim_matches('"').trim_matches('\'').trim().to_string();
        if !question.ends_with('?') {
            question.push('?');
        }

        Ok(Some(question))
    }

    /// Generate ground truth for all sampled chunks
    pub async fn generate(&self, chunks: &[IndexChunk]) -> Result<Vec<GroundTruthEntry>, String> {
        let sampled = self.sample_chunks(chunks);
        let total = sampled.len();
        let mut results = Vec::new();
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for (i, sample) in sampled.iter().enumerate() {
            eprint!("[{}/{}] {} ({})...", i + 1, total, sample.course, sample.domain);

            match self.generate_question(&sample.content).await {
                Ok(Some(question)) => {
                    eprintln!(" {}", &question[..question.len().min(60)]);
                    results.push(GroundTruthEntry {
                        query: question,
                        chunk_content: sample.content.clone(),
                        chunk_source: sample.source.clone(),
                        chunk_start_secs: sample.start_secs,
                        chunk_end_secs: sample.end_secs,
                        domain: sample.domain.clone(),
                        course: sample.course.clone(),
                    });
                }
                Ok(None) => {
                    eprintln!(" SKIP");
                    skipped += 1;
                }
                Err(e) => {
                    eprintln!(" ERROR: {e}");
                    errors += 1;
                }
            }
        }

        eprintln!("\nGenerated {} queries, {} skipped, {} errors", results.len(), skipped, errors);

        Ok(results)
    }
}

/// A sampled chunk with its metadata
#[derive(Debug, Clone)]
pub struct SampledChunk {
    /// Chunk text
    pub content: String,
    /// Source path
    pub source: String,
    /// Start time
    pub start_secs: Option<f64>,
    /// End time
    pub end_secs: Option<f64>,
    /// Course directory
    pub course: String,
    /// Domain
    pub domain: String,
}

/// Check if a chunk is eligible for question generation
fn is_eligible(chunk: &IndexChunk) -> bool {
    let words: Vec<&str> = chunk.content.split_whitespace().collect();
    if words.len() < 50 {
        return false;
    }
    let lowered: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let unique: std::collections::HashSet<&str> = lowered.iter().map(|w| w.as_str()).collect();
    if unique.len() < 15 {
        return false;
    }

    // Skip navigational boilerplate
    let lower = chunk.content.to_lowercase();
    let nav_phrases = [
        "welcome back",
        "in this video",
        "let's go ahead",
        "see you in the next",
        "don't forget to subscribe",
        "click the link",
        "table of contents",
    ];
    let nav_count = nav_phrases.iter().filter(|p| lower.contains(*p)).count();
    nav_count < 3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(content: &str, source: &str) -> IndexChunk {
        IndexChunk {
            content: content.to_string(),
            source: source.to_string(),
            title: None,
            start_secs: Some(0.0),
            end_secs: Some(30.0),
        }
    }

    #[test]
    fn test_is_eligible_short() {
        let chunk = make_chunk("too short", "/data/courses/test/build/a.srt");
        assert!(!is_eligible(&chunk));
    }

    #[test]
    fn test_is_eligible_valid() {
        let words: Vec<String> = (0..60).map(|i| format!("word{i}")).collect();
        let content = words.join(" ");
        let chunk = make_chunk(&content, "/data/courses/test/build/a.srt");
        assert!(is_eligible(&chunk));
    }

    #[test]
    fn test_sampling_deterministic() {
        let chunks: Vec<IndexChunk> = (0..100)
            .map(|i| {
                let words: Vec<String> = (0..60).map(|j| format!("w{j}c{i}")).collect();
                make_chunk(
                    &words.join(" "),
                    &format!("/data/courses/course-{}/build/vid.srt", i / 5),
                )
            })
            .collect();

        let gen1 = GroundTruthGenerator::new(AnthropicClient::new("fake"), "model", 20, 42);
        let gen2 = GroundTruthGenerator::new(AnthropicClient::new("fake"), "model", 20, 42);

        let s1 = gen1.sample_chunks(&chunks);
        let s2 = gen2.sample_chunks(&chunks);

        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.course, b.course);
        }
    }
}
