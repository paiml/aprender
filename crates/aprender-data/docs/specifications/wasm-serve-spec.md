# WASM Serve Specification v1.0

> **Toyota Way Review Summary**
> * **Reviewer:** Gemini (Acting as Technical Architect)
> * **Date:** 2025-11-25
> * **Philosophy:** The Toyota Way & Lean Product Development
>
> **Executive Summary:**
> This specification enables **sovereign data sharing** through browser-based WASM serving. The plugin architecture for course types exemplifies **Principle 6: Standardized Work as a Foundation for Continuous Improvement** while the peer-to-peer capabilities reduce cloud dependency, supporting **Principle 1: Long-term Philosophy**.

**Browser-Based Data Serving, Sharing, and Course Delivery**

## Overview

WASM Serve extends alimentar's core data loading capabilities to enable browser-based serving of datasets, parquet files, and structured course content. This specification defines a plugin-based type system that supports arbitrary content types (including educational courses from assetgen) while maintaining alimentar's sovereignty-first principles.

> **Review Note (Principle 1: Long-Term Philosophy):**
> Browser-based serving eliminates mandatory server infrastructure, enabling true data sovereignty. Users can share datasets directly from their browsers without cloud intermediaries.
> * *Reference: Liker, J. K. (2004). The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer. McGraw-Hill. [1]*

## Design Principles

1. **Browser-First** - Full functionality in WASM without server dependencies
2. **Plugin Architecture** - Extensible type system for arbitrary content (courses, datasets, models)
3. **Zero-Server Option** - P2P sharing via WebRTC without mandatory infrastructure
4. **Schema-Driven** - YAML/JSON configuration for type definitions
5. **Ecosystem Integration** - Native interop with trueno-viz, trueno-db, assetgen

> **Review Note (Principle 7: Visual Control):**
> The architecture diagram below serves as a visual management tool, enabling any team member to understand the system at a glance—reducing cognitive burden (Muri).
> * *Reference: Poppendieck, M., & Poppendieck, T. (2003). Lean Software Development: An Agile Toolkit. Addison-Wesley. [2]*

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         WASM Serve Architecture                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     Type Plugin Registry                             │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │ │
│  │  │  Dataset │ │  Course  │ │  Model   │ │ Registry │ │  Custom  │  │ │
│  │  │  Plugin  │ │  Plugin  │ │  Plugin  │ │  Plugin  │ │  Plugin  │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐│
│  │                    Content Abstraction Layer                          ││
│  │  ┌────────────────────────────────────────────────────────────────┐  ││
│  │  │  ServeableContent trait                                        │  ││
│  │  │  • schema() → ContentSchema                                    │  ││
│  │  │  • validate() → Result<()>                                     │  ││
│  │  │  • to_arrow() → RecordBatch                                    │  ││
│  │  │  • metadata() → ContentMetadata                                │  ││
│  │  └────────────────────────────────────────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                    │                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                      Transport Layer                                  ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  ││
│  │  │    HTTP(S)   │ │    WebRTC    │ │  IndexedDB   │                  ││
│  │  │   Serving    │ │     P2P      │ │    Cache     │                  ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                    │                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                      Rendering Integration                            ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  ││
│  │  │  trueno-viz  │ │  trueno-db   │ │   assetgen   │                  ││
│  │  │   Charts     │ │   Queries    │ │   Courses    │                  ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  ││
│  └──────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

## Core Types

### ServeableContent Trait

> **Review Note (Principle 2: Create Continuous Process Flow):**
> The `ServeableContent` trait establishes a single-piece flow for content processing—any content type flows through the same pipeline, minimizing inventory (buffered state) and maximizing throughput.
> * *Reference: Womack, J. P., & Jones, D. T. (1996). Lean Thinking: Banish Waste and Create Wealth in Your Corporation. Simon & Schuster. [3]*

```rust
/// Trait for any content that can be served via WASM
///
/// This trait provides the abstraction layer between content types
/// (datasets, courses, models) and the serving infrastructure.
pub trait ServeableContent: Send + Sync {
    /// Returns the content schema for validation and UI generation
    fn schema(&self) -> ContentSchema;

    /// Validates content integrity
    fn validate(&self) -> Result<ValidationReport>;

    /// Converts content to Arrow RecordBatch for efficient transfer
    fn to_arrow(&self) -> Result<RecordBatch>;

    /// Returns content metadata for indexing and discovery
    fn metadata(&self) -> ContentMetadata;

    /// Returns content type identifier
    fn content_type(&self) -> ContentTypeId;

    /// Chunk iterator for streaming large content
    fn chunks(&self, chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>>>;
}

/// Unique identifier for content types
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContentTypeId(String);

impl ContentTypeId {
    pub const DATASET: &'static str = "alimentar.dataset";
    pub const COURSE: &'static str = "assetgen.course";
    pub const MODEL: &'static str = "aprender.model";
    pub const REGISTRY: &'static str = "alimentar.registry";
}
```

### Content Schema

```rust
/// Schema definition for content validation and UI generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSchema {
    /// Schema version for compatibility checking
    pub version: String,

    /// Content type identifier
    pub content_type: ContentTypeId,

    /// Field definitions with types and constraints
    pub fields: Vec<FieldDefinition>,

    /// Required fields for validation
    pub required: Vec<String>,

    /// Custom validators (regex patterns, range checks, etc.)
    pub validators: Vec<ValidatorDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    pub description: Option<String>,
    pub default: Option<serde_json::Value>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Binary,
    Array { item_type: Box<FieldType> },
    Object { schema: Box<ContentSchema> },
    Reference { content_type: ContentTypeId },
}

/// Constraint definition for field validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub kind: String,
    pub value: serde_json::Value,
}

/// Validator definition for custom validation logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorDefinition {
    pub validator_type: String,
    pub name: String,
    pub message: String,
    pub check: String,
}
```

## Plugin System

### Type Plugin Trait

> **Review Note (Principle 11: Respect Your Extended Network):**
> The plugin architecture enables third-party extensions without modifying core code—respecting the principle that value creation extends beyond organizational boundaries.
> * *Reference: Dyer, J. H., & Nobeoka, K. (2000). Creating and Managing a High-Performance Knowledge-Sharing Network: The Toyota Case. Strategic Management Journal, 21(3), 345-367. [4]*

```rust
/// Plugin interface for extending alimentar with new content types
pub trait ContentPlugin: Send + Sync {
    /// Returns the content type ID this plugin handles
    fn content_type(&self) -> ContentTypeId;

    /// Returns the schema for this content type
    fn schema(&self) -> ContentSchema;

    /// Parses raw content into ServeableContent
    fn parse(&self, data: &[u8]) -> Result<Box<dyn ServeableContent>>;

    /// Serializes ServeableContent back to bytes
    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>>;

    /// Returns UI rendering hints for trueno-viz integration
    fn render_hints(&self) -> RenderHints;

    /// Plugin version for compatibility
    fn version(&self) -> &str;
}

/// Registry for content plugins
pub struct PluginRegistry {
    plugins: HashMap<ContentTypeId, Box<dyn ContentPlugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        let mut registry = Self { plugins: HashMap::new() };

        // Register built-in plugins
        registry.register(Box::new(DatasetPlugin::new()));
        registry.register(Box::new(CoursePlugin::new()));
        registry.register(Box::new(ModelPlugin::new()));
        registry.register(Box::new(RegistryPlugin::new()));

        registry
    }

    pub fn register(&mut self, plugin: Box<dyn ContentPlugin>) {
        self.plugins.insert(plugin.content_type(), plugin);
    }

    pub fn get(&self, content_type: &ContentTypeId) -> Option<&dyn ContentPlugin> {
        self.plugins.get(content_type).map(|p| p.as_ref())
    }
}
```

### YAML Configuration

> **Review Note (Principle 6: Standardized Work):**
> YAML configuration files establish standardized work for content definitions—providing a baseline from which continuous improvement (Kaizen) can occur.
> * *Reference: Spear, S., & Bowen, H. K. (1999). Decoding the DNA of the Toyota Production System. Harvard Business Review, 77(5), 96-106. [5]*

```yaml
# alimentar-serve.yml - Configuration for WASM serve functionality
version: "1.0"

# Plugin configuration
plugins:
  # Built-in dataset plugin
  - type: alimentar.dataset
    enabled: true
    config:
      max_chunk_size: 1048576  # 1MB chunks for streaming
      compression: snappy

  # Course plugin (assetgen integration)
  - type: assetgen.course
    enabled: true
    config:
      validate_assets: true
      render_markdown: true
      schema_path: "./schemas/course.yml"

  # Custom plugin (user-defined)
  - type: custom.my_content
    enabled: true
    wasm_path: "./plugins/my_content.wasm"
    schema_path: "./schemas/my_content.yml"

# Server configuration
server:
  # HTTP serving (optional)
  http:
    enabled: true
    port: 8080
    cors_origins: ["*"]

  # WebRTC P2P (optional)
  webrtc:
    enabled: true
    signaling_server: "wss://signal.example.com"
    ice_servers:
      - urls: ["stun:stun.l.google.com:19302"]

  # IndexedDB caching
  cache:
    enabled: true
    max_size_mb: 500
    ttl_hours: 24

# Content types to serve
content:
  - path: "./data/train.parquet"
    type: alimentar.dataset
    public: true

  - path: "./courses/rust-fundamentals"
    type: assetgen.course
    public: true
    metadata:
      title: "Rust Fundamentals"
      categories: ["programming", "rust"]
```

## Course Plugin (assetgen Integration)

### Course Content Type

> **Review Note (Principle 12: Go and See - Genchi Genbutsu):**
> The course schema is derived from actual assetgen course structures through direct observation of production systems, not assumptions.
> * *Reference: Fagan, M. E. (1976). Design and Code Inspections to Reduce Errors in Program Development. IBM Systems Journal, 15(3), 182-211. [6]*

```rust
/// Course content type for assetgen integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseContent {
    pub info: CourseInfo,
    pub outline: CourseOutline,
    pub assets: Vec<Asset>,
    pub metadata: ContentMetadata,
}

/// Course information (aligned with assetgen CourseInfo)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseInfo {
    pub id: String,
    pub title: String,
    pub description: String,
    pub short_description: String,
    pub categories: Vec<String>,
    pub weeks: u32,
    pub featured: bool,
    pub difficulty: Option<String>,
    pub prerequisites: Vec<String>,
}

/// Course outline structure (aligned with assetgen CourseOutline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseOutline {
    pub title: String,
    pub weeks: Vec<Week>,
    pub required_files: Vec<RequiredFile>,
    pub validation_rules: ValidationRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Week {
    pub number: u32,
    pub title: String,
    pub lessons: Vec<Lesson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    pub number: String,
    pub title: String,
    pub assets: Vec<Asset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub filename: String,
    pub asset_type: AssetType,
    pub description: Option<String>,
    pub generates: Option<Vec<GenerationType>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetType {
    Video,
    KeyTerms,
    Quiz,
    Lab,
    Reflection,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationType {
    Yml,
    Md,
}
```

### Course Schema YAML

```yaml
# schemas/course.yml - Course content type schema
version: "1.0"
content_type: assetgen.course

fields:
  - name: id
    type: string
    description: "Unique course identifier"
    constraints:
      - pattern: "^[a-z0-9-]+$"
      - max_length: 64

  - name: title
    type: string
    description: "Course title"
    constraints:
      - min_length: 1
      - max_length: 256

  - name: description
    type: string
    description: "Full course description"

  - name: short_description
    type: string
    description: "Brief course summary"
    constraints:
      - max_length: 500

  - name: categories
    type: array
    item_type: string
    description: "Course categories for discovery"

  - name: weeks
    type: integer
    description: "Number of weeks in course"
    constraints:
      - min: 1
      - max: 52

  - name: outline
    type: object
    description: "Course outline structure"
    schema:
      fields:
        - name: weeks
          type: array
          item_type:
            type: object
            fields:
              - name: number
                type: integer
              - name: title
                type: string
              - name: lessons
                type: array
                item_type:
                  type: object
                  fields:
                    - name: number
                      type: string
                    - name: title
                      type: string
                    - name: assets
                      type: array

required:
  - id
  - title
  - description
  - weeks

validators:
  - type: custom
    name: valid_asset_types
    message: "Asset types must be valid"
    check: "assets.*.asset_type in ['video', 'keyterms', 'quiz', 'lab', 'reflection']"
```

## WASM Server

### WasmServeServer

> **Review Note (Principle 3: Use Pull Systems):**
> Chunk-based streaming implements a pull system where consumers request data as needed, preventing overproduction (downloading unnecessary data) and reducing memory pressure.
> * *Reference: Rother, M., & Shook, J. (1999). Learning to See: Value-Stream Mapping to Create Value and Eliminate Muda. Lean Enterprise Institute. [7]*

```rust
/// WASM-compatible content server
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmServeServer {
    registry: PluginRegistry,
    content: HashMap<String, Box<dyn ServeableContent>>,
    config: ServeConfig,
}

#[wasm_bindgen]
impl WasmServeServer {
    /// Create a new WASM server instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
            content: HashMap::new(),
            config: ServeConfig::default(),
        }
    }

    /// Load content from URL
    #[wasm_bindgen]
    pub async fn load_url(&mut self, url: &str, content_type: &str) -> Result<(), JsValue> {
        let response = fetch(url).await?;
        let data = response.array_buffer().await?;
        let bytes = js_sys::Uint8Array::new(&data).to_vec();

        let type_id = ContentTypeId(content_type.to_string());
        let plugin = self.registry.get(&type_id)
            .ok_or_else(|| JsValue::from_str("Unknown content type"))?;

        let content = plugin.parse(&bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.content.insert(url.to_string(), content);
        Ok(())
    }

    /// Load content from IndexedDB cache
    #[wasm_bindgen]
    pub async fn load_cached(&mut self, key: &str) -> Result<bool, JsValue> {
        // IndexedDB retrieval implementation
        todo!()
    }

    /// Get content as Arrow IPC bytes
    #[wasm_bindgen]
    pub fn get_arrow(&self, key: &str) -> Result<js_sys::Uint8Array, JsValue> {
        let content = self.content.get(key)
            .ok_or_else(|| JsValue::from_str("Content not found"))?;

        let batch = content.to_arrow()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Convert to Arrow IPC format
        let ipc_bytes = arrow_ipc_encode(&batch)?;
        Ok(js_sys::Uint8Array::from(&ipc_bytes[..]))
    }

    /// Stream content in chunks
    #[wasm_bindgen]
    pub fn stream_chunks(&self, key: &str, chunk_size: usize) -> ChunkIterator {
        // Returns async iterator for streaming
        todo!()
    }

    /// Share content via WebRTC
    #[wasm_bindgen]
    pub async fn share_p2p(&self, key: &str) -> Result<String, JsValue> {
        // Returns sharing URL/code
        todo!()
    }

    /// Render content with trueno-viz
    #[wasm_bindgen]
    pub fn render(&self, key: &str, target: &str) -> Result<(), JsValue> {
        let content = self.content.get(key)
            .ok_or_else(|| JsValue::from_str("Content not found"))?;

        let plugin = self.registry.get(&content.content_type())
            .ok_or_else(|| JsValue::from_str("Plugin not found"))?;

        let hints = plugin.render_hints();

        // Integrate with trueno-viz for rendering
        // trueno_viz::render(content.to_arrow()?, hints, target)?;

        Ok(())
    }
}
```

### P2P Sharing (WebRTC)

> **Review Note (Principle 9: Grow Leaders Who Live the Philosophy):**
> P2P sharing decentralizes data distribution, empowering individual users to become data stewards rather than depending on centralized authorities.
> * *Reference: Nonaka, I., & Takeuchi, H. (1995). The Knowledge-Creating Company: How Japanese Companies Create the Dynamics of Innovation. Oxford University Press. [8]*

```rust
/// WebRTC-based P2P content sharing
#[cfg(target_arch = "wasm32")]
pub struct P2PSharing {
    peer_connection: web_sys::RtcPeerConnection,
    data_channel: Option<web_sys::RtcDataChannel>,
    content_key: String,
}

impl P2PSharing {
    /// Create a sharing offer
    pub async fn create_offer(&mut self, content: &dyn ServeableContent) -> Result<String> {
        // Create WebRTC offer with content metadata
        let offer = self.peer_connection.create_offer().await?;
        self.peer_connection.set_local_description(&offer).await?;

        let sharing_code = ShareCode {
            sdp: offer.sdp(),
            content_type: content.content_type(),
            metadata: content.metadata(),
            chunks: content.to_arrow()?.num_rows() / CHUNK_SIZE,
        };

        Ok(base64_encode(&serde_json::to_vec(&sharing_code)?))
    }

    /// Accept a sharing offer
    pub async fn accept_offer(&mut self, offer_code: &str) -> Result<()> {
        let sharing_code: ShareCode = serde_json::from_slice(&base64_decode(offer_code)?)?;

        let offer = web_sys::RtcSessionDescriptionInit::new(
            web_sys::RtcSdpType::Offer
        );
        offer.set_sdp(&sharing_code.sdp);

        self.peer_connection.set_remote_description(&offer).await?;
        let answer = self.peer_connection.create_answer().await?;
        self.peer_connection.set_local_description(&answer).await?;

        Ok(())
    }

    /// Receive content chunks
    pub async fn receive_content(&mut self) -> Result<Box<dyn ServeableContent>> {
        // Receive Arrow IPC chunks via data channel
        todo!()
    }
}
```

## trueno-db Integration

### Query Support

> **Review Note (Principle 5: Build a Culture of Stopping to Fix Problems - Jidoka):**
> Query validation stops execution immediately upon detecting malformed queries, preventing downstream errors—embodying the principle of building quality in at the source.
> * *Reference: Beck, K. (2002). Test Driven Development: By Example. Addison-Wesley Professional. [9]*

```rust
/// trueno-db integration for queryable content
impl WasmServeServer {
    /// Load content into trueno-db for SQL queries
    pub fn enable_queries(&mut self, key: &str) -> Result<QueryHandle> {
        let content = self.content.get(key)
            .ok_or(Error::NotFound)?;

        let batch = content.to_arrow()?;

        // Create trueno-db instance
        let db = trueno_db::Database::new()?;
        db.insert_batch("content", batch)?;

        Ok(QueryHandle { db, table: "content".to_string() })
    }

    /// Execute SQL query on content
    pub fn query(&self, handle: &QueryHandle, sql: &str) -> Result<RecordBatch> {
        handle.db.query(sql)
    }

    /// Filter content with predicates
    pub fn filter(&self, key: &str, predicate: &str) -> Result<Box<dyn ServeableContent>> {
        // trueno-db predicate pushdown
        todo!()
    }
}

/// Handle for queryable content
pub struct QueryHandle {
    db: trueno_db::Database,
    table: String,
}
```

## trueno-viz Integration

### Visualization Rendering

```rust
/// trueno-viz integration for content visualization
pub struct ContentVisualizer {
    viz: trueno_viz::Context,
}

impl ContentVisualizer {
    /// Render dataset as chart
    pub fn render_dataset(&self, content: &ArrowDataset, config: &VizConfig) -> Result<()> {
        let batch = content.get(0)?;

        // Auto-detect visualization type based on schema
        let plot = match self.detect_plot_type(&batch.schema()) {
            PlotType::Scatter => trueno_viz::scatter(&batch, &config.x, &config.y),
            PlotType::Line => trueno_viz::line(&batch, &config.x, &config.y),
            PlotType::Histogram => trueno_viz::histogram(&batch, &config.x),
            PlotType::Heatmap => trueno_viz::heatmap(&batch),
        };

        plot.render(config.target)?;
        Ok(())
    }

    /// Render course progress
    pub fn render_course_progress(&self, course: &CourseContent) -> Result<()> {
        // Visualize course completion, quiz scores, etc.
        todo!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizConfig {
    pub x: String,
    pub y: Option<String>,
    pub color: Option<String>,
    pub target: String,  // DOM element ID
}
```

## API Reference

### JavaScript/TypeScript API

```typescript
// TypeScript definitions for WASM serve API
declare module 'alimentar-wasm' {
    export class WasmServeServer {
        constructor();

        // Content loading
        loadUrl(url: string, contentType: string): Promise<void>;
        loadCached(key: string): Promise<boolean>;
        loadCourse(url: string): Promise<void>;
        loadDataset(url: string): Promise<void>;

        // Content access
        getArrow(key: string): Uint8Array;
        getMetadata(key: string): ContentMetadata;
        streamChunks(key: string, chunkSize: number): AsyncIterator<Uint8Array>;

        // P2P sharing
        shareP2P(key: string): Promise<string>;
        receiveP2P(shareCode: string): Promise<void>;

        // Querying (trueno-db)
        enableQueries(key: string): QueryHandle;
        query(handle: QueryHandle, sql: string): RecordBatch;

        // Visualization (trueno-viz)
        render(key: string, target: string, config?: VizConfig): void;
    }

    export interface ContentMetadata {
        contentType: string;
        title: string;
        description?: string;
        size: number;
        rowCount?: number;
        schema?: object;
    }

    export interface CourseContent {
        info: CourseInfo;
        outline: CourseOutline;
        assets: CourseAsset[];
    }
}
```

### CLI Commands

```bash
# Serve content locally
alimentar serve ./data --port 8080

# Serve course content
alimentar serve ./courses/rust-fundamentals --type course

# Generate sharing code
alimentar share ./data/train.parquet

# Start P2P sharing server
alimentar serve --p2p --signaling wss://signal.example.com

# Convert course for WASM serving
alimentar convert-course ./course-dir --output ./serve/course.bin
```

## Quality Standards

> **Review Note (Principle 5: Jidoka - Build Quality In):**
> Zero-tolerance quality gates ensure defects are caught at the source, not propagated downstream—the essence of built-in quality.
> * *Reference: Humble, J., & Farley, D. (2010). Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation. Addison-Wesley. [10]*

### Metrics Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Test coverage | ≥90% | `make coverage` (blocks CI) |
| WASM binary size | <300KB | CI size check |
| First render | <100ms | Performance test |
| P2P latency | <500ms | Integration test |
| Memory usage | <50MB | WASM memory limit |

### Test Categories

```rust
#[cfg(test)]
mod tests {
    // Plugin registration tests
    #[test]
    fn test_plugin_registration() {
        let registry = PluginRegistry::new();
        assert!(registry.get(&ContentTypeId::DATASET).is_some());
        assert!(registry.get(&ContentTypeId::COURSE).is_some());
    }

    // Content serialization roundtrip
    #[test]
    fn test_content_roundtrip() {
        let plugin = DatasetPlugin::new();
        let content = create_test_dataset();

        let bytes = plugin.serialize(&content).unwrap();
        let restored = plugin.parse(&bytes).unwrap();

        assert_eq!(content.metadata(), restored.metadata());
    }

    // WASM-specific tests
    #[wasm_bindgen_test]
    async fn test_wasm_serve() {
        let server = WasmServeServer::new();
        server.load_url("./test.parquet", "alimentar.dataset").await.unwrap();

        let arrow = server.get_arrow("./test.parquet").unwrap();
        assert!(arrow.length() > 0);
    }
}
```

## Feature Flags

```toml
[features]
default = ["local", "tokio-runtime"]

# WASM serve features
wasm-serve = ["wasm", "wasm-bindgen", "js-sys", "web-sys"]
webrtc-p2p = ["wasm-serve", "web-sys/RtcPeerConnection"]
indexeddb-cache = ["wasm-serve", "web-sys/IdbDatabase"]

# Integration features
trueno-viz-integration = ["dep:trueno-viz"]
trueno-db-integration = ["dep:trueno-db"]
course-plugin = []  # Enables assetgen course type
```

## Roadmap

### v0.5.0 - WASM Serve Core
- [ ] ServeableContent trait and base implementations
- [ ] Plugin registry with built-in plugins
- [ ] YAML configuration parsing
- [ ] Basic HTTP serving in browser
- [ ] IndexedDB caching

### v0.6.0 - Course Plugin
- [ ] Course content type (assetgen integration)
- [ ] Course schema validation
- [ ] Course outline rendering
- [ ] Asset streaming

### v0.7.0 - P2P Sharing
- [ ] WebRTC data channel implementation
- [ ] Signaling server integration
- [ ] Share code generation
- [ ] NAT traversal (STUN/TURN)

### v0.8.0 - Ecosystem Integration
- [ ] trueno-viz chart rendering
- [ ] trueno-db query support
- [ ] Visualization auto-detection
- [ ] Interactive exploration

## References

1. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.

2. Poppendieck, M., & Poppendieck, T. (2003). *Lean Software Development: An Agile Toolkit*. Addison-Wesley.

3. Womack, J. P., & Jones, D. T. (1996). *Lean Thinking: Banish Waste and Create Wealth in Your Corporation*. Simon & Schuster.

4. Dyer, J. H., & Nobeoka, K. (2000). Creating and Managing a High-Performance Knowledge-Sharing Network: The Toyota Case. *Strategic Management Journal*, 21(3), 345-367.

5. Spear, S., & Bowen, H. K. (1999). Decoding the DNA of the Toyota Production System. *Harvard Business Review*, 77(5), 96-106.

6. Fagan, M. E. (1976). Design and Code Inspections to Reduce Errors in Program Development. *IBM Systems Journal*, 15(3), 182-211.

7. Rother, M., & Shook, J. (1999). *Learning to See: Value-Stream Mapping to Create Value and Eliminate Muda*. Lean Enterprise Institute.

8. Nonaka, I., & Takeuchi, H. (1995). *The Knowledge-Creating Company: How Japanese Companies Create the Dynamics of Innovation*. Oxford University Press.

9. Beck, K. (2002). *Test Driven Development: By Example*. Addison-Wesley Professional.

10. Humble, J., & Farley, D. (2010). *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Addison-Wesley.