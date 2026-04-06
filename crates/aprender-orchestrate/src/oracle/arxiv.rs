//! arXiv Paper Enrichment for Oracle Mode
//!
//! Two-tier enrichment:
//! - **Builtin** (`--arxiv`): Instant results from the curated `ArxivDatabase` (no network)
//! - **Live** (`--arxiv-live`): Fresh papers from `export.arxiv.org/api/query` (Atom XML)

use super::coursera::arxiv_db::ArxivDatabase;
use super::coursera::types::ArxivCitation;
use super::query_engine::ParsedQuery;
use super::types::ProblemDomain;
use serde::Serialize;

// =============================================================================
// Types
// =============================================================================

/// A paper from arXiv (either curated or fetched live).
#[derive(Debug, Clone, Serialize)]
pub struct ArxivPaper {
    /// arXiv identifier (e.g., "2212.04356")
    pub arxiv_id: String,
    /// Paper title
    pub title: String,
    /// Author list (abbreviated, e.g., "Radford et al.")
    pub authors: String,
    /// Publication year
    pub year: u16,
    /// Brief summary (truncated to ~200 chars)
    pub summary: String,
    /// Canonical URL (e.g., `https://arxiv.org/abs/2212.04356`)
    pub url: String,
    /// PDF download URL
    pub pdf_url: Option<String>,
    /// ISO date string from arXiv (e.g., "2022-12-08T...")
    pub published: Option<String>,
}

/// Indicates whether results came from the builtin DB or a live fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArxivSource {
    Builtin,
    Live,
}

/// Enrichment result returned by the enricher.
#[derive(Debug, Clone, Serialize)]
pub struct ArxivEnrichment {
    pub papers: Vec<ArxivPaper>,
    pub source: ArxivSource,
    pub query_terms: Vec<String>,
}

// =============================================================================
// ArxivPaper conversions
// =============================================================================

impl ArxivPaper {
    /// Convert from the existing curated `ArxivCitation` type.
    pub fn from_citation(c: &ArxivCitation) -> Self {
        Self {
            arxiv_id: c.arxiv_id.clone(),
            title: c.title.clone(),
            authors: c.authors.clone(),
            year: c.year,
            summary: c.abstract_snippet.clone(),
            url: c.url.clone(),
            pdf_url: Some(format!("https://arxiv.org/pdf/{}.pdf", c.arxiv_id)),
            published: None,
        }
    }
}

// =============================================================================
// Search term derivation
// =============================================================================

/// Component-name to arXiv search-term mapping.
const COMPONENT_TERMS: &[(&str, &[&str])] = &[
    ("whisper-apr", &["whisper", "speech recognition"]),
    ("realizar", &["model inference", "optimization"]),
    ("trueno", &["SIMD", "GPU compute"]),
    ("aprender", &["machine learning", "algorithms"]),
    ("entrenar", &["training", "fine-tuning", "LoRA"]),
    ("repartir", &["distributed computing", "parallelism"]),
    ("trueno-db", &["database", "analytics"]),
    ("trueno-graph", &["graph neural network", "graph analytics"]),
    ("trueno-rag", &["retrieval augmented generation"]),
    ("simular", &["simulation", "monte carlo"]),
    ("jugar", &["game engine", "reinforcement learning"]),
    ("alimentar", &["data loading", "parquet"]),
    ("pacha", &["model registry", "model management"]),
    ("depyler", &["python", "transpilation"]),
    ("decy", &["C++", "transpilation"]),
    ("bashrs", &["shell", "scripting"]),
    ("pepita", &["kernel", "operating system"]),
];

/// Domain-to-arXiv search-term mapping.
const DOMAIN_TERMS: &[(ProblemDomain, &[&str])] = &[
    (ProblemDomain::SpeechRecognition, &["speech recognition", "ASR"]),
    (ProblemDomain::Inference, &["model inference", "serving"]),
    (ProblemDomain::DeepLearning, &["deep learning", "transformer"]),
    (ProblemDomain::SupervisedLearning, &["supervised learning", "classification"]),
    (ProblemDomain::UnsupervisedLearning, &["unsupervised learning", "clustering"]),
    (ProblemDomain::LinearAlgebra, &["linear algebra", "SIMD"]),
    (ProblemDomain::VectorSearch, &["vector search", "embedding"]),
    (ProblemDomain::GraphAnalytics, &["graph neural network"]),
    (ProblemDomain::DistributedCompute, &["distributed computing"]),
    (ProblemDomain::PythonMigration, &["python", "machine learning"]),
    (ProblemDomain::CMigration, &["systems programming"]),
    (ProblemDomain::ShellMigration, &["automation"]),
    (ProblemDomain::DataPipeline, &["data pipeline", "ETL"]),
    (ProblemDomain::ModelServing, &["model serving", "edge deployment"]),
    (ProblemDomain::Testing, &["mutation testing", "software testing"]),
    (ProblemDomain::Profiling, &["profiling", "tracing"]),
    (ProblemDomain::Validation, &["validation", "quality"]),
];

/// Derive search terms from a parsed oracle query.
///
/// Priority: mentioned components > detected domains > fallback to keywords.
pub fn derive_search_terms(parsed: &ParsedQuery) -> Vec<String> {
    let mut terms = Vec::new();

    // 1. Component mentions → mapped terms
    for comp in &parsed.mentioned_components {
        if let Some(&(_, mapped)) = COMPONENT_TERMS.iter().find(|&&(name, _)| name == comp) {
            terms.extend(mapped.iter().map(|s| (*s).to_string()));
        }
    }

    // 2. Detected domains → mapped terms
    for domain in &parsed.domains {
        if let Some(&(_, mapped)) = DOMAIN_TERMS.iter().find(|&&(d, _)| d == *domain) {
            for t in mapped {
                if !terms.iter().any(|existing| existing == *t) {
                    terms.push((*t).to_string());
                }
            }
        }
    }

    // 3. Detected algorithms → direct use
    for algo in &parsed.algorithms {
        let readable = algo.replace('_', " ");
        if !terms.iter().any(|existing| existing == &readable) {
            terms.push(readable);
        }
    }

    // 4. Fallback: use first 3 keywords if nothing was derived
    if terms.is_empty() {
        terms.extend(parsed.keywords.iter().take(3).cloned());
    }

    terms
}

// =============================================================================
// ArxivEnricher
// =============================================================================

/// Enricher that can query the builtin curated database or fetch live from arXiv.
#[derive(Default)]
pub struct ArxivEnricher;

impl ArxivEnricher {
    pub fn new() -> Self {
        Self
    }

    /// Enrich from the builtin curated database (instant, no network).
    pub fn enrich_builtin(&self, parsed: &ParsedQuery, max: usize) -> ArxivEnrichment {
        let terms = derive_search_terms(parsed);
        let db = ArxivDatabase::builtin();

        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
        let citations = db.find_by_keywords(&term_refs, max);
        let papers = citations.iter().map(ArxivPaper::from_citation).collect();

        ArxivEnrichment { papers, source: ArxivSource::Builtin, query_terms: terms }
    }

    /// Enrich from the live arXiv API. Falls back to builtin on error.
    #[cfg(feature = "native")]
    pub async fn enrich_live(&self, parsed: &ParsedQuery, max: usize) -> ArxivEnrichment {
        let terms = derive_search_terms(parsed);
        let search_query = terms.join(" ");

        match fetch_arxiv_api(&search_query, max).await {
            Ok(papers) if !papers.is_empty() => {
                ArxivEnrichment { papers, source: ArxivSource::Live, query_terms: terms }
            }
            _ => {
                // Fallback to builtin on error or empty results
                self.enrich_builtin(parsed, max)
            }
        }
    }
}

// =============================================================================
// Live arXiv API
// =============================================================================

/// Fetch papers from the arXiv Atom API.
#[cfg(feature = "native")]
pub async fn fetch_arxiv_api(query: &str, max: usize) -> anyhow::Result<Vec<ArxivPaper>> {
    let encoded = query.replace(' ', "+");
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}",
        encoded, max
    );

    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(10)).build()?;

    let body = client.get(&url).send().await?.text().await?;
    parse_arxiv_atom_xml(&body)
}

/// Per-entry accumulator for XML parsing state.
#[cfg(feature = "native")]
struct EntryAccum {
    id: String,
    title: String,
    summary: String,
    published: String,
    authors: Vec<String>,
    pdf_url: Option<String>,
}

#[cfg(feature = "native")]
impl EntryAccum {
    fn new() -> Self {
        Self {
            id: String::new(),
            title: String::new(),
            summary: String::new(),
            published: String::new(),
            authors: Vec::new(),
            pdf_url: None,
        }
    }

    fn clear(&mut self) {
        self.id.clear();
        self.title.clear();
        self.summary.clear();
        self.published.clear();
        self.authors.clear();
        self.pdf_url = None;
    }

    fn into_paper(self) -> Option<ArxivPaper> {
        if self.id.is_empty() || self.title.is_empty() {
            return None;
        }
        let arxiv_id = extract_arxiv_id(&self.id);
        let year = extract_year(&self.published);
        Some(ArxivPaper {
            url: format!("https://arxiv.org/abs/{}", arxiv_id),
            pdf_url: self.pdf_url,
            arxiv_id,
            title: normalize_whitespace(&self.title),
            authors: format_authors(&self.authors),
            year,
            summary: truncate_summary(&self.summary),
            published: if self.published.is_empty() { None } else { Some(self.published) },
        })
    }

    fn push_text(&mut self, tag: &str, text: String, in_author: bool) {
        match tag {
            "id" => self.id.push_str(&text),
            "title" => self.title.push_str(&text),
            "summary" => self.summary.push_str(&text),
            "published" => self.published.push_str(&text),
            "name" if in_author => self.authors.push(text),
            _ => {}
        }
    }
}

/// Extract PDF href from a `<link>` element's attributes, if present.
#[cfg(feature = "native")]
fn extract_pdf_href(attrs: quick_xml::events::attributes::Attributes) -> Option<String> {
    let mut href = String::new();
    let mut is_pdf = false;
    for attr in attrs.flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref());
        let val = String::from_utf8_lossy(&attr.value);
        if key == "title" && val == "pdf" {
            is_pdf = true;
        }
        if key == "href" {
            href = val.to_string();
        }
    }
    (is_pdf && !href.is_empty()).then_some(href)
}

/// Atom XML state machine for parsing arXiv feeds.
#[cfg(feature = "native")]
struct AtomParser {
    papers: Vec<ArxivPaper>,
    accum: EntryAccum,
    current_tag: String,
    in_entry: bool,
    in_author: bool,
}

#[cfg(feature = "native")]
impl AtomParser {
    fn new() -> Self {
        Self {
            papers: Vec::new(),
            accum: EntryAccum::new(),
            current_tag: String::new(),
            in_entry: false,
            in_author: false,
        }
    }

    fn handle_start(&mut self, e: &quick_xml::events::BytesStart<'_>) {
        let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
        match tag.as_str() {
            "entry" => {
                self.in_entry = true;
                self.accum.clear();
            }
            "author" if self.in_entry => {
                self.in_author = true;
            }
            "link" if self.in_entry => {
                if let Some(href) = extract_pdf_href(e.attributes()) {
                    self.accum.pdf_url = Some(href);
                }
            }
            _ if self.in_entry => {
                self.current_tag = tag;
            }
            _ => {}
        }
    }

    fn handle_empty(&mut self, e: &quick_xml::events::BytesStart<'_>) {
        if !self.in_entry {
            return;
        }
        let name = e.name();
        let tag = String::from_utf8_lossy(name.as_ref());
        if tag == "link" {
            if let Some(href) = extract_pdf_href(e.attributes()) {
                self.accum.pdf_url = Some(href);
            }
        }
    }

    fn handle_text(&mut self, e: &quick_xml::events::BytesText<'_>) {
        if !self.in_entry {
            return;
        }
        let text = e.unescape().unwrap_or_default().to_string();
        self.accum.push_text(&self.current_tag, text, self.in_author);
    }

    fn handle_end(&mut self, e: &quick_xml::events::BytesEnd<'_>) {
        let name = e.name();
        let tag = String::from_utf8_lossy(name.as_ref());
        match tag.as_ref() {
            "entry" => {
                let finished = std::mem::replace(&mut self.accum, EntryAccum::new());
                if let Some(paper) = finished.into_paper() {
                    self.papers.push(paper);
                }
                self.in_entry = false;
                self.current_tag.clear();
            }
            "author" => {
                self.in_author = false;
            }
            _ => {
                self.current_tag.clear();
            }
        }
    }
}

/// Parse arXiv Atom XML feed into `ArxivPaper` structs.
#[cfg(feature = "native")]
pub fn parse_arxiv_atom_xml(xml: &str) -> anyhow::Result<Vec<ArxivPaper>> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    let mut parser = AtomParser::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => parser.handle_start(e),
            Ok(Event::Empty(ref e)) => parser.handle_empty(e),
            Ok(Event::Text(ref e)) => parser.handle_text(e),
            Ok(Event::End(ref e)) => parser.handle_end(e),
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }

    Ok(parser.papers)
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract the arXiv ID from a full URL like `http://arxiv.org/abs/2212.04356v1`.
#[cfg(feature = "native")]
fn extract_arxiv_id(url: &str) -> String {
    let base = url.rsplit('/').next().unwrap_or(url);
    // Strip trailing version suffix (e.g., "v2")
    if let Some(idx) = base.rfind('v') {
        if base[idx + 1..].chars().all(|c| c.is_ascii_digit()) && !base[idx + 1..].is_empty() {
            return base[..idx].to_string();
        }
    }
    base.to_string()
}

/// Extract a 4-digit year from an ISO date string like "2022-12-08T...".
#[cfg(feature = "native")]
fn extract_year(published: &str) -> u16 {
    published.get(..4).and_then(|s| s.parse::<u16>().ok()).unwrap_or(0)
}

/// Format author list: 1 author → "Name", 2 → "A & B", 3+ → "A et al."
pub fn format_authors(authors: &[String]) -> String {
    match authors.len() {
        0 => "Unknown".to_string(),
        1 => authors[0].clone(),
        2 => format!("{} & {}", authors[0], authors[1]),
        _ => format!("{} et al.", authors[0]),
    }
}

/// Truncate summary to ~200 chars at a word boundary.
#[cfg(feature = "native")]
fn truncate_summary(s: &str) -> String {
    let normalized = normalize_whitespace(s);
    if normalized.len() <= 200 {
        return normalized;
    }
    // Find a char boundary at or before byte 200
    let boundary =
        normalized.char_indices().take_while(|&(i, _)| i <= 200).last().map_or(0, |(i, _)| i);
    let truncated = &normalized[..boundary];
    match truncated.rfind(' ') {
        Some(idx) => format!("{}...", &truncated[..idx]),
        None => format!("{}...", truncated),
    }
}

/// Collapse consecutive whitespace (newlines, tabs, spaces) into a single space.
#[cfg(feature = "native")]
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::QueryEngine;

    #[test]
    fn test_derive_terms_from_component() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("improve whisper-apr performance");
        let terms = derive_search_terms(&parsed);
        assert!(
            terms.iter().any(|t| t.contains("whisper")),
            "Expected 'whisper' in terms: {:?}",
            terms
        );
    }

    #[test]
    fn test_derive_terms_from_domain() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("train a classifier with supervised learning");
        let terms = derive_search_terms(&parsed);
        assert!(!terms.is_empty(), "Expected non-empty terms for supervised learning query");
    }

    #[test]
    fn test_derive_terms_fallback() {
        let engine = QueryEngine::new();
        let parsed = engine.parse("completely unknown xyzzy topic");
        let terms = derive_search_terms(&parsed);
        // Should fallback to keywords
        assert!(!terms.is_empty(), "Expected keyword fallback for unknown query");
    }

    #[test]
    fn test_enrich_builtin_returns_papers() {
        let enricher = ArxivEnricher::new();
        let engine = QueryEngine::new();
        let parsed = engine.parse("whisper speech recognition");
        let result = enricher.enrich_builtin(&parsed, 5);
        assert!(!result.papers.is_empty(), "Expected papers for 'whisper speech recognition'");
        assert_eq!(result.source, ArxivSource::Builtin);
    }

    #[test]
    fn test_enrich_builtin_respects_max() {
        let enricher = ArxivEnricher::new();
        let engine = QueryEngine::new();
        let parsed = engine.parse("deep learning transformer attention");
        let result = enricher.enrich_builtin(&parsed, 1);
        assert!(result.papers.len() <= 1, "Expected at most 1 paper, got {}", result.papers.len());
    }

    #[test]
    fn test_enrich_builtin_no_results() {
        let enricher = ArxivEnricher::new();
        let engine = QueryEngine::new();
        let parsed = engine.parse("xyzzy nonexistent gibberish");
        let result = enricher.enrich_builtin(&parsed, 5);
        // May or may not find results (keywords might match something), but shouldn't panic
        assert_eq!(result.source, ArxivSource::Builtin);
    }

    #[test]
    fn test_from_citation() {
        let citation = ArxivCitation {
            arxiv_id: "2212.04356".to_string(),
            title: "Whisper: Robust Speech Recognition".to_string(),
            authors: "Radford et al.".to_string(),
            year: 2022,
            url: "https://arxiv.org/abs/2212.04356".to_string(),
            abstract_snippet: "Multitask speech model.".to_string(),
            topics: vec!["speech".to_string()],
        };
        let paper = ArxivPaper::from_citation(&citation);
        assert_eq!(paper.arxiv_id, "2212.04356");
        assert_eq!(paper.year, 2022);
        assert_eq!(paper.url, "https://arxiv.org/abs/2212.04356");
        assert!(paper.pdf_url.is_some());
        assert!(paper.pdf_url.expect("unexpected failure").contains("2212.04356"));
    }

    #[test]
    fn test_format_authors_zero() {
        assert_eq!(format_authors(&[]), "Unknown");
    }

    #[test]
    fn test_format_authors_one() {
        assert_eq!(format_authors(&["Alice".to_string()]), "Alice");
    }

    #[test]
    fn test_format_authors_two() {
        assert_eq!(format_authors(&["Alice".to_string(), "Bob".to_string()]), "Alice & Bob");
    }

    #[test]
    fn test_format_authors_three() {
        assert_eq!(
            format_authors(&["Alice".to_string(), "Bob".to_string(), "Carol".to_string(),]),
            "Alice et al."
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_parse_arxiv_atom_xml() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2212.04356v2</id>
    <title>Robust Speech Recognition via Large-Scale Weak Supervision</title>
    <summary>We study the capabilities of speech processing systems.</summary>
    <published>2022-12-06T00:00:00Z</published>
    <author><name>Alec Radford</name></author>
    <author><name>Jong Wook Kim</name></author>
    <author><name>Tao Xu</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2212.04356v2" rel="related" type="application/pdf"/>
  </entry>
</feed>"#;

        let papers = parse_arxiv_atom_xml(xml).expect("unexpected failure");
        assert_eq!(papers.len(), 1);
        let p = &papers[0];
        assert_eq!(p.arxiv_id, "2212.04356");
        assert!(p.title.contains("Robust Speech Recognition"));
        assert_eq!(p.year, 2022);
        assert_eq!(p.authors, "Alec Radford et al.");
        assert!(p.pdf_url.is_some());
        assert!(p.published.is_some());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_parse_arxiv_atom_xml_empty() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"#;
        let papers = parse_arxiv_atom_xml(xml).expect("unexpected failure");
        assert!(papers.is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_extract_arxiv_id_with_version() {
        assert_eq!(extract_arxiv_id("http://arxiv.org/abs/2212.04356v2"), "2212.04356");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_extract_arxiv_id_without_version() {
        assert_eq!(extract_arxiv_id("http://arxiv.org/abs/2212.04356"), "2212.04356");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_truncate_summary_short() {
        assert_eq!(truncate_summary("Short text."), "Short text.");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_truncate_summary_long() {
        let long = "a ".repeat(200);
        let truncated = truncate_summary(&long);
        assert!(truncated.len() <= 210);
        assert!(truncated.ends_with("..."));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("hello\n  world\t\tfoo"), "hello world foo");
    }

    #[ignore = "requires network access to arXiv API"]
    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_live_arxiv_query() {
        let papers =
            fetch_arxiv_api("whisper speech recognition", 3).await.expect("unexpected failure");
        assert!(!papers.is_empty());
        for p in &papers {
            assert!(!p.title.is_empty());
            assert!(!p.arxiv_id.is_empty());
        }
    }
}
