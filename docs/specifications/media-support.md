# Trueno-RAG: Media Support Specification

**Version:** 1.0.0
**Status:** Draft
**Authors:** Pragmatic AI Labs
**References:** TRUENO-RAG-003
**Feature Flag:** `transcription` (for whisper-apr-based speech-to-text)

## Abstract

This specification extends trueno-rag with media file support — enabling video and audio
content to participate in RAG pipelines alongside text documents. The design introduces a
`DocumentLoader` trait abstraction for pluggable file format support, built-in SRT/VTT
subtitle parsing (zero additional dependencies), optional whisper-apr integration for
GPU-accelerated speech-to-text transcription via .apr model format (feature-gated),
timestamp-aware chunking that preserves temporal context, and a batch processing strategy
suitable for large media corpora on GPU-equipped hardware.

## 1. Introduction

### 1.1 Motivation

trueno-rag currently operates exclusively on text — `.txt` and `.md` files loaded via
`std::fs::read_to_string`. This excludes a significant class of knowledge assets:
recorded lectures, conference talks, screencasts, podcasts, and other audio/video
content. Most of this content lacks transcripts.

The gap is architectural: trueno-rag has no abstraction for "how to turn a file into
a `Document`." Text files happen to work because the CLI reads them directly. Adding
media support requires:

1. **Format abstraction** — A trait that decouples file loading from the pipeline
2. **Subtitle parsing** — SRT/VTT are the lingua franca of timed text; parsing them requires zero heavy dependencies
3. **Transcription** — For media without subtitles, whisper-apr provides
   GPU-accelerated Whisper inference via `.apr` models
4. **Temporal chunking** — Chunks from media should carry timestamp metadata for citation and navigation

### 1.2 Design Principles

- **Zero mandatory heavy deps** — Subtitle parsing works with no new dependencies
- **Feature-gated transcription** — whisper-apr adds ML inference deps; opt-in only
- **Trait-based extensibility** — Third parties can implement `DocumentLoader` for any format
- **Timestamp preservation** — Temporal metadata flows through chunking, indexing, and retrieval
- **Batch-friendly** — Designed for corpora of thousands of files on multi-core hardware

### 1.3 Integration

This specification builds on existing trueno-rag components:

- **`Document`** — Extended with media-specific metadata (duration, timestamps)
- **`Chunk` / `ChunkMetadata`** — `custom` field carries temporal offsets
- **`Chunker` trait** — New `TimestampChunker` implementation
- **CLI `index` command** — Extended with `--recursive` and media format support
- **whisper-apr** — Optional dependency for GPU-accelerated transcription (feature-gated)

## 2. Architecture

### 2.1 Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Media Ingestion Pipeline                      │
├──────────────┬───────────────┬────────────────┬──────────────────────┤
│  Discovery   │    Loading    │   Chunking     │     Indexing         │
│  ──────────  │  ───────────  │  ──────────    │    ─────────         │
│  Recursive   │  DocumentLoader│ Timestamp-    │  Standard RAG        │
│  file walk   │  trait dispatch│ aware split   │  pipeline            │
├──────────────┴───────────────┴────────────────┴──────────────────────┤
│                    Existing trueno-rag Pipeline                       │
│              (Embed → Index → Retrieve → Rerank → Assemble)          │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
media file (.mp4, .wav, .srt, .vtt, .txt, .md)
    │
    ▼
DocumentLoader::load(path) → Document
    │                          ├── content: String (transcript text)
    │                          ├── source: path
    │                          └── metadata: { "duration_secs", "timestamps", "format" }
    ▼
TimestampChunker::chunk(doc) → Vec<Chunk>
    │                            └── metadata.custom: { "start_secs", "end_secs" }
    ▼
Standard RAG pipeline (embed → index → retrieve)
    │
    ▼
RetrievalResult with temporal context
    └── "At 14:32 in lecture.mp4: ..."
```

## 3. `DocumentLoader` Trait

### 3.1 Trait Definition

```rust
/// Abstraction for loading files of any format into Documents.
///
/// Implementors handle format detection, parsing, and conversion
/// to the standard Document representation. A loader may support
/// multiple file extensions.
pub trait DocumentLoader: Send + Sync {
    /// File extensions this loader handles (lowercase, without dot).
    ///
    /// Example: `vec!["srt", "vtt"]`
    fn supported_extensions(&self) -> Vec<&str>;

    /// Returns true if this loader can handle the given path.
    ///
    /// Default implementation checks the file extension against
    /// `supported_extensions()`. Implementors may override for
    /// content-based detection (magic bytes, etc.).
    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                let lower = ext.to_lowercase();
                self.supported_extensions().iter().any(|s| *s == lower)
            })
            .unwrap_or(false)
    }

    /// Load a file and produce a Document.
    ///
    /// The returned Document should have:
    /// - `content`: The extracted text (transcript, subtitle text, etc.)
    /// - `source`: The file path
    /// - `title`: Derived from filename or embedded metadata
    /// - `metadata`: Format-specific fields (duration, timestamps, etc.)
    fn load(&self, path: &Path) -> Result<Document>;

    /// Load a file asynchronously.
    ///
    /// Default implementation calls `load()` on a blocking thread.
    /// Implementors may override for truly async I/O.
    async fn load_async(&self, path: &Path) -> Result<Document> {
        let path = path.to_path_buf();
        let loader = self; // requires Send + Sync
        tokio::task::spawn_blocking(move || loader.load(&path))
            .await
            .map_err(|e| Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other, e
            )))?
    }
}
```

### 3.2 Loader Registry

```rust
/// Registry that dispatches file loading to the appropriate DocumentLoader.
pub struct LoaderRegistry {
    loaders: Vec<Box<dyn DocumentLoader>>,
}

impl LoaderRegistry {
    /// Create a registry with default loaders (text, subtitle).
    pub fn new() -> Self {
        let mut registry = Self { loaders: Vec::new() };
        registry.register(Box::new(TextLoader));
        registry.register(Box::new(SubtitleLoader));
        registry
    }

    /// Register a custom loader.
    pub fn register(&mut self, loader: Box<dyn DocumentLoader>) {
        self.loaders.push(loader);
    }

    /// Find the first loader that can handle the given path.
    pub fn loader_for(&self, path: &Path) -> Option<&dyn DocumentLoader> {
        self.loaders.iter()
            .find(|l| l.can_load(path))
            .map(|l| l.as_ref())
    }

    /// Load a document, selecting the appropriate loader automatically.
    pub fn load(&self, path: &Path) -> Result<Document> {
        let loader = self.loader_for(path)
            .ok_or_else(|| Error::InvalidInput(
                format!("No loader for: {}", path.display())
            ))?;
        loader.load(path)
    }

    /// All supported extensions across all registered loaders.
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.loaders.iter()
            .flat_map(|l| l.supported_extensions())
            .collect()
    }
}
```

### 3.3 Built-in Loaders

#### TextLoader

Handles `.txt` and `.md` files. This extracts the existing CLI logic into a proper loader.

```rust
pub struct TextLoader;

impl DocumentLoader for TextLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["txt", "md"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        let content = std::fs::read_to_string(path)
            .map_err(Error::Io)?;
        let title = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Untitled")
            .to_string();

        Ok(Document::new(content)
            .with_title(title)
            .with_source(path.to_string_lossy()))
    }
}
```

#### SubtitleLoader

See Section 4.

## 4. SRT/VTT Subtitle Parser

### 4.1 Supported Formats

**SRT (SubRip Text):**
```
1
00:00:01,000 --> 00:00:04,500
Welcome to this lecture on machine learning.

2
00:00:05,000 --> 00:00:09,200
Today we'll cover the fundamentals of
supervised learning algorithms.
```

**VTT (WebVTT):**
```
WEBVTT

00:00:01.000 --> 00:00:04.500
Welcome to this lecture on machine learning.

00:00:05.000 --> 00:00:09.200
Today we'll cover the fundamentals of
supervised learning algorithms.
```

### 4.2 Data Model

```rust
/// A single timed text cue from a subtitle file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubtitleCue {
    /// Sequence number (SRT) or index
    pub index: usize,
    /// Start time in seconds
    pub start_secs: f64,
    /// End time in seconds
    pub end_secs: f64,
    /// Text content (may contain multiple lines)
    pub text: String,
}

/// Parsed subtitle file.
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// Format detected
    pub format: SubtitleFormat,
    /// Ordered cues
    pub cues: Vec<SubtitleCue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    Vtt,
}

impl SubtitleTrack {
    /// Total duration based on last cue end time.
    pub fn duration_secs(&self) -> f64 {
        self.cues.last().map(|c| c.end_secs).unwrap_or(0.0)
    }

    /// Concatenate all cue text into a plain transcript.
    pub fn to_plain_text(&self) -> String {
        self.cues.iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get cues within a time range.
    pub fn cues_in_range(&self, start: f64, end: f64) -> Vec<&SubtitleCue> {
        self.cues.iter()
            .filter(|c| c.end_secs > start && c.start_secs < end)
            .collect()
    }
}
```

### 4.3 Parser Implementation

```rust
/// Parse SRT or VTT from a string, auto-detecting format.
pub fn parse_subtitles(input: &str) -> Result<SubtitleTrack> {
    let trimmed = input.trim_start_with_bom();
    if trimmed.starts_with("WEBVTT") {
        parse_vtt(trimmed)
    } else {
        parse_srt(trimmed)
    }
}

/// Parse SRT format.
fn parse_srt(input: &str) -> Result<SubtitleTrack> {
    let mut cues = Vec::new();
    // Split on blank lines to get cue blocks
    for block in input.split("\n\n").filter(|b| !b.trim().is_empty()) {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 { continue; }

        // Line 0: sequence number
        let index: usize = lines[0].trim().parse()
            .map_err(|_| Error::InvalidInput("Bad SRT index".into()))?;

        // Line 1: timestamp line "HH:MM:SS,mmm --> HH:MM:SS,mmm"
        let (start, end) = parse_srt_timestamp_line(lines[1])?;

        // Lines 2+: text content
        let text = lines[2..].join("\n").trim().to_string();

        cues.push(SubtitleCue { index, start_secs: start, end_secs: end, text });
    }

    Ok(SubtitleTrack { format: SubtitleFormat::Srt, cues })
}

/// Parse VTT format.
fn parse_vtt(input: &str) -> Result<SubtitleTrack> {
    let mut cues = Vec::new();
    let mut index = 0usize;

    // Skip header and metadata
    let body = input.splitn(2, "\n\n").nth(1).unwrap_or("");

    for block in body.split("\n\n").filter(|b| !b.trim().is_empty()) {
        let lines: Vec<&str> = block.lines().collect();

        // Find the timestamp line (contains "-->")
        let ts_line_idx = lines.iter().position(|l| l.contains("-->"));
        let Some(ts_idx) = ts_line_idx else { continue };

        let (start, end) = parse_vtt_timestamp_line(lines[ts_idx])?;
        let text = lines[ts_idx + 1..].join("\n").trim().to_string();

        if text.is_empty() { continue; }

        cues.push(SubtitleCue { index, start_secs: start, end_secs: end, text });
        index += 1;
    }

    Ok(SubtitleTrack { format: SubtitleFormat::Vtt, cues })
}

/// Parse "HH:MM:SS,mmm" to seconds (SRT uses comma).
fn parse_srt_time(s: &str) -> Result<f64> {
    // "00:14:32,500" → 872.5
    let s = s.trim().replace(',', ".");
    parse_timestamp(&s)
}

/// Parse "HH:MM:SS.mmm" to seconds (VTT uses dot).
fn parse_vtt_time(s: &str) -> Result<f64> {
    parse_timestamp(s.trim())
}

/// Common timestamp parser for "HH:MM:SS.mmm" or "MM:SS.mmm".
fn parse_timestamp(s: &str) -> Result<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        // MM:SS.mmm
        2 => {
            let mins: f64 = parts[0].parse().map_err(|_| Error::InvalidInput("Bad timestamp".into()))?;
            let secs: f64 = parts[1].parse().map_err(|_| Error::InvalidInput("Bad timestamp".into()))?;
            Ok(mins * 60.0 + secs)
        }
        // HH:MM:SS.mmm
        3 => {
            let hrs: f64 = parts[0].parse().map_err(|_| Error::InvalidInput("Bad timestamp".into()))?;
            let mins: f64 = parts[1].parse().map_err(|_| Error::InvalidInput("Bad timestamp".into()))?;
            let secs: f64 = parts[2].parse().map_err(|_| Error::InvalidInput("Bad timestamp".into()))?;
            Ok(hrs * 3600.0 + mins * 60.0 + secs)
        }
        _ => Err(Error::InvalidInput(format!("Invalid timestamp: {s}"))),
    }
}
```

### 4.4 SubtitleLoader

```rust
pub struct SubtitleLoader;

impl DocumentLoader for SubtitleLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["srt", "vtt"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        let raw = std::fs::read_to_string(path).map_err(Error::Io)?;
        let track = parse_subtitles(&raw)?;

        let title = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Untitled")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert(
            "duration_secs".into(),
            serde_json::json!(track.duration_secs()),
        );
        metadata.insert(
            "format".into(),
            serde_json::json!(match track.format {
                SubtitleFormat::Srt => "srt",
                SubtitleFormat::Vtt => "vtt",
            }),
        );
        // Serialize cue timestamps for downstream chunking
        metadata.insert(
            "subtitle_cues".into(),
            serde_json::to_value(&track.cues)
                .map_err(Error::Serialization)?,
        );

        let mut doc = Document::new(track.to_plain_text())
            .with_title(title)
            .with_source(path.to_string_lossy());
        doc.metadata = metadata;
        Ok(doc)
    }
}
```

## 5. Whisper-APR Transcription Integration (Feature-Gated)

The sovereign stack approach: whisper-apr provides GPU-accelerated Whisper ASR inference
using `.apr` model files. trueno-rag uses whisper-apr for speech-to-text, keeping the
entire pipeline in pure Rust with zero Python/C++ dependencies.

### 5.1 Feature Flag

```toml
# In Cargo.toml
[dependencies]
whisper-apr = { version = "0.2", optional = true, default-features = false, features = ["std"] }

[features]
# GPU-accelerated speech-to-text via whisper-apr (Whisper .apr models)
# Adds ML inference dependency — opt-in only
transcription = ["dep:whisper-apr"]
```

### 5.2 TranscriptionConfig

```rust
#[cfg(feature = "transcription")]
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Language hint (ISO 639-1, e.g., "en"). None for auto-detect.
    pub language: Option<String>,
    /// Beam size for decoding (1 = greedy, 5 = default).
    pub beam_size: usize,
    /// Enable word-level timestamps (more precise but slower).
    pub word_timestamps: bool,
    /// Write .srt sidecar files after transcription for caching.
    pub write_sidecar: bool,
    /// Backend selection for whisper-apr inference.
    pub backend: TranscriptionBackend,
}

#[cfg(feature = "transcription")]
#[derive(Debug, Clone, Copy, Default)]
pub enum TranscriptionBackend {
    /// CPU with SIMD acceleration via trueno
    #[default]
    Cpu,
    /// GPU via wgpu (cross-platform)
    Gpu,
    /// NVIDIA CUDA (Linux/Windows)
    Cuda,
}
```

### 5.3 TranscriptionLoader

```rust
#[cfg(feature = "transcription")]
pub struct TranscriptionLoader {
    config: TranscriptionConfig,
}

#[cfg(feature = "transcription")]
impl DocumentLoader for TranscriptionLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["mp4", "mp3", "wav", "m4a", "ogg", "flac", "webm"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // 1. Check for sidecar — skip transcription if .srt/.vtt exists
        if let Some(sidecar) = LoaderRegistry::find_sidecar(path) {
            return SubtitleLoader.load(&sidecar);
        }

        // 2. Parse WAV and resample to 16kHz mono
        let wav = whisper_apr::audio::wav::parse_wav_file(path)?;
        let samples = whisper_apr::audio::wav::resample(
            &wav.data, wav.original_channels, wav.sample_rate, 16000
        );

        // 3. Run Whisper inference via WhisperApr
        let mut options = TranscribeOptions::default();
        options.language = self.config.language.clone();
        let result = self.whisper.as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .transcribe(&samples, options)?;

        // 4. Convert whisper-apr Segments → SubtitleTrack
        let track = segments_to_track(&result.segments);

        // 5. Write sidecar if configured
        if self.config.write_sidecar {
            let _ = write_sidecar(path, &track);
        }

        // 6. Build Document with timestamp metadata
        build_transcription_document(path, &track)
    }
}

/// Convert whisper-apr Segments to a SubtitleTrack.
fn segments_to_track(segments: &[whisper_apr::Segment]) -> SubtitleTrack {
    let cues: Vec<SubtitleCue> = segments.iter().enumerate().map(|(i, seg)| {
        SubtitleCue {
            index: i,
            start_secs: seg.start as f64,
            end_secs: seg.end as f64,
            text: seg.text.trim().to_string(),
        }
    }).collect();
    SubtitleTrack { format: SubtitleFormat::Srt, cues }
}
```

### 5.4 Audio Pipeline

The transcription pipeline leverages whisper-apr's audio primitives:

```
Input File (.wav)
    │
    ▼
whisper_apr::audio::wav::parse_wav_file(path) → WavData
    │
    ▼
whisper_apr::audio::wav::resample(data, channels, orig_rate, 16000)
    │
    ▼
WhisperApr::transcribe(&samples, options)
    │
    ▼
TranscriptionResult { text, segments: Vec<Segment { start: f32, end: f32, text }> }
    │
    ▼
segments_to_track() → SubtitleTrack → Document
```

### 5.5 Backend Selection

whisper-apr supports multiple compute backends via `TranscriptionBackend`:

```rust
/// Compute backend for transcription inference.
pub enum TranscriptionBackend {
    Cpu,   // CPU with SIMD acceleration via trueno
    Gpu,   // GPU via wgpu (cross-platform)
    Cuda,  // NVIDIA CUDA (Linux/Windows)
}
```

Model loading uses the `.apr` format — use the best available Whisper model variant
(e.g., `base.apr`, `large-v3-turbo.apr` for maximum accuracy on GPU hardware).

### 5.3 Sidecar Strategy

For large corpora, transcription is expensive. trueno-rag supports a **sidecar**
pattern: if a `.srt` or `.vtt` file exists alongside a media file, the
`SubtitleLoader` is used instead of invoking Whisper.

```
videos/
├── lecture-01.mp4          # Media file
├── lecture-01.srt          # ← Sidecar: SubtitleLoader used, Whisper skipped
├── lecture-02.mp4          # No sidecar → TranscriptionLoader invoked
└── lecture-03.vtt          # Standalone subtitle → SubtitleLoader
```

Resolution order in `LoaderRegistry`:
1. Check for sidecar subtitle (`.srt`, `.vtt`) adjacent to media file
2. If sidecar exists, load via `SubtitleLoader`
3. If no sidecar and `transcription` feature is enabled, use `TranscriptionLoader`
4. If no sidecar and feature disabled, skip file with warning

This allows incremental transcription: run whisper-apr transcription on a batch,
save `.srt` sidecars, then index everything cheaply.

## 6. Timestamp-Aware Chunking

### 6.1 TimestampChunker

Standard text chunkers split on character boundaries. Media transcripts have a
natural temporal structure that should be preserved.

```rust
/// Chunker that respects subtitle cue boundaries and preserves timestamps.
pub struct TimestampChunker {
    /// Target chunk duration in seconds
    pub target_duration_secs: f64,
    /// Minimum chunk duration (avoid tiny fragments)
    pub min_duration_secs: f64,
    /// Maximum chunk duration (hard limit)
    pub max_duration_secs: f64,
    /// Overlap duration for context continuity
    pub overlap_secs: f64,
}

impl Default for TimestampChunker {
    fn default() -> Self {
        Self {
            target_duration_secs: 60.0,  // ~1 minute chunks
            min_duration_secs: 10.0,
            max_duration_secs: 120.0,
            overlap_secs: 5.0,
        }
    }
}
```

### 6.2 Chunking Algorithm

```rust
impl Chunker for TimestampChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        // Extract subtitle cues from document metadata
        let cues: Vec<SubtitleCue> = document.metadata
            .get("subtitle_cues")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        if cues.is_empty() {
            // No timestamp data — fall back to RecursiveChunker behavior
            return RecursiveChunker::new(512, 50).chunk(document);
        }

        let mut chunks = Vec::new();
        let mut current_cues: Vec<&SubtitleCue> = Vec::new();
        let mut chunk_start = cues[0].start_secs;

        for cue in &cues {
            let current_duration = cue.end_secs - chunk_start;

            if current_duration >= self.target_duration_secs
                && !current_cues.is_empty()
            {
                // Emit chunk
                chunks.push(self.build_chunk(
                    document, &current_cues, chunk_start,
                ));

                // Start next chunk with overlap
                let overlap_start = cue.start_secs - self.overlap_secs;
                current_cues.retain(|c| c.start_secs >= overlap_start);
                chunk_start = current_cues.first()
                    .map(|c| c.start_secs)
                    .unwrap_or(cue.start_secs);
            }

            current_cues.push(cue);
        }

        // Emit final chunk
        if !current_cues.is_empty() {
            // Merge into previous if too short
            if current_cues.last().unwrap().end_secs - chunk_start
                < self.min_duration_secs
                && !chunks.is_empty()
            {
                // Extend the previous chunk
                if let Some(last) = chunks.last_mut() {
                    let text: String = current_cues.iter()
                        .map(|c| c.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    last.content.push(' ');
                    last.content.push_str(&text);
                    last.end_offset = current_cues.last().unwrap().end_secs as usize;
                    if let Some(end) = last.metadata.custom.get_mut("end_secs") {
                        *end = serde_json::json!(current_cues.last().unwrap().end_secs);
                    }
                }
            } else {
                chunks.push(self.build_chunk(
                    document, &current_cues, chunk_start,
                ));
            }
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        let duration = document.metadata
            .get("duration_secs")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        (duration / self.target_duration_secs).ceil() as usize
    }
}
```

### 6.3 Chunk Metadata

Each media-derived chunk carries temporal metadata in `ChunkMetadata.custom`:

```json
{
    "start_secs": 872.5,
    "end_secs": 932.1,
    "start_display": "14:32",
    "end_display": "15:32",
    "source_format": "srt",
    "cue_count": 12
}
```

This enables timestamp-aware citation in retrieval results:

```
[Score: 0.847] "Supervised learning uses labeled training data..."
  → lecture-03.mp4 at 14:32–15:32
```

## 7. CLI Extensions

### 7.1 Recursive Directory Walking

The `index` command gains `--recursive` for deep directory traversal:

```
trueno-rag index --path /data/courses/ --output index/ --recursive
```

```rust
// In Commands::Index
/// Recursively scan subdirectories
#[arg(short, long, default_value = "false")]
recursive: bool,
```

Implementation uses `walkdir` (or `std::fs` recursive read) filtered through `LoaderRegistry::supported_extensions()`.

### 7.2 Media-Aware Indexing

```
# Index a directory containing mixed text and subtitle files
trueno-rag index --path /data/ --output index/ --recursive

# Index with timestamp-aware chunking (auto-selected for media)
trueno-rag index --path /data/ --output index/ --recursive --chunk-strategy timestamp

# With transcription (requires --features transcription)
trueno-rag index --path /data/ --output index/ --recursive \
    --model /models/whisper-large-v3.gguf --backend gpu

# Control parallelism
trueno-rag index --path /data/ --output index/ --recursive --jobs 16
```

New CLI arguments:

```rust
/// Chunking strategy for media files
#[arg(long, value_enum, default_value = "auto")]
chunk_strategy: ChunkStrategy,

/// Path to Whisper model (GGUF/safetensors/APR format)
#[arg(long)]
model: Option<String>,

/// Transcription backend (cpu, gpu, cuda)
#[arg(long, default_value = "cpu")]
backend: Option<String>,

/// Number of parallel transcription/loading jobs
#[arg(short, long, default_value = "4")]
jobs: usize,
```

`ChunkStrategy::Auto` selects `TimestampChunker` for files with timestamp metadata, `RecursiveChunker` for plain text.

### 7.3 Progress Reporting

For large corpora, the CLI reports progress:

```
Scanning /data/courses/... found 5,247 files
  4,076 media files (mp4)
  1,171 text files (md, txt)
  0 subtitle sidecars

Processing [████████░░░░░░░░░░░░] 2,038/5,247 (38.8%)
  Transcribing: lecture-2041.mp4 [3:42 remaining]
  Indexed: 48,291 chunks | 127.3 MB embeddings
```

## 8. Batch Processing Strategy

### 8.1 Pipeline Architecture

For large corpora, the bottleneck is transcription (Whisper inference). The batch
pipeline separates discovery, transcription, and indexing into stages. With
whisper-apr's GPU support, transcription throughput scales with GPU capability
rather than CPU core count.

```
Stage 1: Discovery (single-threaded, fast)
    Walk directory → classify files → check for sidecars
    Output: manifest.json (file list with loader assignments)

Stage 2: Transcription (GPU-accelerated, parallelizable)
    For each media file without sidecar:
        whisper-apr inference (.apr model) → write .srt sidecar
    GPU: single model, sequential files (GPU memory bound)
    CPU: --jobs N for parallel SIMD inference

Stage 3: Indexing (parallel, moderate)
    Load all documents (text + sidecars) via LoaderRegistry
    Chunk → Embed → Index
    Parallelism: bounded by embedder throughput
```

### 8.2 Parallelism Model

```rust
/// Batch processing configuration.
pub struct BatchConfig {
    /// Max concurrent transcription jobs (CPU mode)
    pub transcription_jobs: usize,
    /// Max concurrent embedding jobs
    pub embedding_jobs: usize,
    /// Write .srt sidecars after transcription (for caching)
    pub write_sidecars: bool,
    /// Skip files that already have sidecars
    pub skip_existing_sidecars: bool,
    /// Progress callback
    pub on_progress: Option<Box<dyn Fn(BatchProgress) + Send>>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        let cpus = num_cpus::get();
        Self {
            transcription_jobs: cpus / 2,  // CPU mode; GPU mode uses 1 job
            embedding_jobs: cpus,
            write_sidecars: true,
            skip_existing_sidecars: true,
            on_progress: None,
        }
    }
}
```

### 8.3 Sidecar Caching

Transcription results are persisted as `.srt` sidecars beside the original media files. This means:

- **Re-indexing is cheap** — Only subtitle loading, no re-transcription
- **Incremental updates** — New files get transcribed; existing sidecars are reused
- **Portability** — `.srt` files are human-readable and editable
- **Interop** — Standard format consumable by any subtitle tool

```rust
/// Write transcription result as SRT sidecar.
fn write_sidecar(media_path: &Path, track: &SubtitleTrack) -> Result<PathBuf> {
    let sidecar_path = media_path.with_extension("srt");
    let srt_content = track.to_srt_string();
    std::fs::write(&sidecar_path, srt_content).map_err(Error::Io)?;
    Ok(sidecar_path)
}
```

### 8.4 Estimated Throughput

Rough throughput targets for planning (actual numbers depend on hardware and model size):

| Stage | Per-file (GPU) | Per-file (CPU) | 5,000 files |
|-------|----------------|----------------|-------------|
| Discovery + classify | < 1ms | < 1ms | < 5s total |
| Whisper large-v3 (GPU) | ~10-30x realtime | N/A | hours (sequential) |
| Whisper base.en (CPU) | N/A | ~1x realtime | ~16 concurrent = hours |
| Subtitle load + parse | < 10ms | < 10ms | < 50s total |
| Embed (MiniLM, batch) | ~5ms/chunk | ~5ms/chunk | minutes |
| BM25 index | ~1ms/chunk | ~1ms/chunk | seconds |

**Recommended strategy for large corpora:**
1. Use the best Whisper model available (large-v3 GGUF on GPU for accuracy)
2. Run transcription as a background batch job via `trueno-rag transcribe`
3. Sidecars accumulate on disk as `.srt` files
4. Run indexing separately once transcription is complete (minutes)
5. Re-index incrementally as new content is added

## 9. Module Organization

### 9.1 New Modules

```
src/
├── loader/
│   ├── mod.rs          # DocumentLoader trait, LoaderRegistry
│   ├── text.rs         # TextLoader (.txt, .md)
│   ├── subtitle.rs     # SubtitleLoader, SRT/VTT parser
│   └── transcription.rs # TranscriptionLoader (feature-gated)
├── media.rs            # SubtitleCue, SubtitleTrack, SubtitleFormat
├── chunk.rs            # + TimestampChunker
└── ...
```

### 9.2 Public API Additions

```rust
// lib.rs additions
pub mod loader;
pub mod media;

pub use loader::{DocumentLoader, LoaderRegistry, TextLoader, SubtitleLoader};
pub use media::{SubtitleCue, SubtitleTrack, SubtitleFormat, parse_subtitles};

#[cfg(feature = "transcription")]
pub use loader::TranscriptionLoader;

pub use chunk::TimestampChunker;
```

### 9.3 Feature Flags

```toml
[features]
default = ["sqlite"]
# ... existing features ...

# GPU-accelerated speech-to-text via whisper-apr (Whisper .apr models)
transcription = ["dep:whisper-apr"]
```

## 10. Error Handling

New error variants:

```rust
pub enum Error {
    // ... existing variants ...

    /// Subtitle parsing failed
    SubtitleParse(String),

    /// No suitable loader found for file format
    UnsupportedFormat(String),

    /// Transcription failed (whisper-apr inference error)
    #[cfg(feature = "transcription")]
    Transcription(String),
}
```

## 11. Testing Strategy

### 11.1 Unit Tests

- **SRT parser**: Valid SRT, malformed timestamps, missing sequence numbers, UTF-8 with BOM, empty cues
- **VTT parser**: Valid VTT, WEBVTT header variations, metadata blocks, styling tags (stripped)
- **Timestamp parsing**: Edge cases (0:00:00, 99:59:59, millisecond precision)
- **TimestampChunker**: Chunk boundary alignment, overlap correctness, min/max duration enforcement, empty input
- **LoaderRegistry**: Extension matching, sidecar resolution, unknown format handling

### 11.2 Property-Based Tests

```rust
proptest! {
    #[test]
    fn srt_roundtrip(cues in vec(subtitle_cue_strategy(), 1..100)) {
        let track = SubtitleTrack { format: SubtitleFormat::Srt, cues };
        let srt_string = track.to_srt_string();
        let parsed = parse_srt(&srt_string).unwrap();
        assert_eq!(parsed.cues.len(), track.cues.len());
    }

    #[test]
    fn timestamp_chunks_cover_full_duration(
        cues in vec(subtitle_cue_strategy(), 10..200)
    ) {
        let doc = document_from_cues(&cues);
        let chunker = TimestampChunker::default();
        let chunks = chunker.chunk(&doc).unwrap();

        // All text should be represented in at least one chunk
        for cue in &cues {
            assert!(chunks.iter().any(|c| c.content.contains(&cue.text)));
        }
    }
}
```

### 11.3 Integration Tests

- Load a real `.srt` file → chunk → embed → retrieve by query
- Verify timestamp metadata survives the full pipeline
- Sidecar resolution: place `.mp4` + `.srt` in temp dir, verify subtitle loader is selected

## 12. Implementation Phases

### Phase 1: Foundation (no new dependencies)

1. `DocumentLoader` trait and `LoaderRegistry`
2. `TextLoader` (refactor existing CLI logic)
3. SRT/VTT parser and `SubtitleLoader`
4. `TimestampChunker`
5. CLI `--recursive` flag
6. Tests for all above

### Phase 2: CLI Integration

1. Wire `LoaderRegistry` into CLI `index` command
2. `--chunk-strategy auto|timestamp|recursive` flag
3. `--jobs` parallel loading
4. Progress reporting
5. JSON manifest output

### Phase 3: Transcription (feature-gated)

1. `whisper-apr` dependency (optional, `std` feature)
2. `TranscriptionConfig` and `TranscriptionLoader`
3. Audio→transcribe pipeline using whisper-apr primitives
4. Sidecar write/read logic
5. CLI `--model` and `--backend` flags

### Phase 4: Batch Tooling

1. `trueno-rag transcribe` subcommand (batch transcription only, no indexing)
2. Manifest-based resume (skip completed files)
3. Progress persistence across restarts
4. Throughput reporting

## 13. References

[1] Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale
Weak Supervision." *Proceedings of ICML*. arXiv:2212.04356

[2] W3C. (2019). "WebVTT: The Web Video Text Tracks Format." https://www.w3.org/TR/webvtt1/

[3] Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997

---

## Appendix A: SRT Format Reference

```
<sequence_number>\n
<start_time> --> <end_time>\n
<text_line_1>\n
[<text_line_2>\n]
\n
```

Timestamp format: `HH:MM:SS,mmm` (comma separator for milliseconds)

## Appendix B: VTT Format Reference

```
WEBVTT\n
[metadata headers]\n
\n
[<cue_id>\n]
<start_time> --> <end_time> [<settings>]\n
<text_line_1>\n
[<text_line_2>\n]
\n
```

Timestamp format: `HH:MM:SS.mmm` or `MM:SS.mmm` (dot separator)

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-16 | Initial specification |
