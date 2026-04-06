//! Fixture loading system for Presentar tests.
//!
//! Provides utilities for loading test fixtures from tar archives,
//! directories, and inline definitions. Zero external dependencies.
//!
//! # Example
//!
//! ```ignore
//! use presentar_test::fixture::{Fixture, FixtureBuilder};
//!
//! // Load from embedded tar archive
//! let fixture = Fixture::from_tar(include_bytes!("fixtures/app.tar"))?;
//!
//! // Access fixture files
//! let app_yaml = fixture.get_file("app.yaml")?;
//! let data = fixture.get_data("metrics.ald")?;
//! ```

use std::collections::HashMap;

/// Error type for fixture operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureError {
    /// Failed to parse tar archive
    InvalidTar(String),
    /// File not found in fixture
    FileNotFound(String),
    /// Invalid fixture format
    InvalidFormat(String),
    /// IO error (represented as string for no-std compatibility)
    IoError(String),
    /// YAML parsing error
    YamlError(String),
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTar(msg) => write!(f, "invalid tar archive: {msg}"),
            Self::FileNotFound(path) => write!(f, "file not found: {path}"),
            Self::InvalidFormat(msg) => write!(f, "invalid fixture format: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::YamlError(msg) => write!(f, "YAML error: {msg}"),
        }
    }
}

impl std::error::Error for FixtureError {}

/// A test fixture containing files and configuration.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Files by path
    files: HashMap<String, Vec<u8>>,
    /// Fixture manifest
    manifest: FixtureManifest,
}

/// Fixture manifest describing contents.
#[derive(Debug, Clone, Default)]
pub struct FixtureManifest {
    /// Fixture name
    pub name: String,
    /// App YAML path (if present)
    pub app_yaml: Option<String>,
    /// Data files (.ald)
    pub data_files: Vec<String>,
    /// Model files (.apr)
    pub model_files: Vec<String>,
    /// Asset files (images, fonts)
    pub assets: Vec<String>,
    /// Snapshot baselines
    pub snapshots: Vec<String>,
}

impl Fixture {
    /// Create a new empty fixture.
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            manifest: FixtureManifest::default(),
        }
    }

    /// Create fixture from a tar archive (embedded bytes).
    ///
    /// # Errors
    ///
    /// Returns error if tar parsing fails.
    pub fn from_tar(data: &[u8]) -> Result<Self, FixtureError> {
        let mut fixture = Self::new();
        fixture.parse_tar(data)?;
        fixture.build_manifest();
        Ok(fixture)
    }

    /// Create fixture from inline files.
    #[must_use]
    pub fn from_files(files: Vec<(&str, &[u8])>) -> Self {
        let mut fixture = Self::new();
        for (path, content) in files {
            fixture.files.insert(path.to_string(), content.to_vec());
        }
        fixture.build_manifest();
        fixture
    }

    /// Get the fixture manifest.
    #[must_use]
    pub fn manifest(&self) -> &FixtureManifest {
        &self.manifest
    }

    /// List all file paths in the fixture.
    #[must_use]
    pub fn list_files(&self) -> Vec<&str> {
        self.files.keys().map(String::as_str).collect()
    }

    /// Check if a file exists.
    #[must_use]
    pub fn has_file(&self, path: &str) -> bool {
        self.files.contains_key(path)
    }

    /// Get file contents as bytes.
    pub fn get_file(&self, path: &str) -> Result<&[u8], FixtureError> {
        self.files
            .get(path)
            .map(Vec::as_slice)
            .ok_or_else(|| FixtureError::FileNotFound(path.to_string()))
    }

    /// Get file contents as string.
    pub fn get_file_str(&self, path: &str) -> Result<&str, FixtureError> {
        let bytes = self.get_file(path)?;
        std::str::from_utf8(bytes).map_err(|e| FixtureError::InvalidFormat(e.to_string()))
    }

    /// Get the app.yaml content if present.
    pub fn get_app_yaml(&self) -> Result<&str, FixtureError> {
        let path = self
            .manifest
            .app_yaml
            .as_deref()
            .ok_or_else(|| FixtureError::FileNotFound("app.yaml".to_string()))?;
        self.get_file_str(path)
    }

    /// Get a data file (.ald) content.
    pub fn get_data(&self, name: &str) -> Result<&[u8], FixtureError> {
        let path = if name.ends_with(".ald") {
            name.to_string()
        } else {
            format!("{name}.ald")
        };
        self.get_file(&path)
    }

    /// Get a model file (.apr) content.
    pub fn get_model(&self, name: &str) -> Result<&[u8], FixtureError> {
        let path = if name.ends_with(".apr") {
            name.to_string()
        } else {
            format!("{name}.apr")
        };
        self.get_file(&path)
    }

    /// Get a snapshot baseline image.
    pub fn get_snapshot(&self, name: &str) -> Result<&[u8], FixtureError> {
        let path = if name.contains('/') {
            name.to_string()
        } else {
            format!("snapshots/{name}.png")
        };
        self.get_file(&path)
    }

    /// Add a file to the fixture.
    pub fn add_file(&mut self, path: &str, content: Vec<u8>) {
        self.files.insert(path.to_string(), content);
    }

    /// Remove a file from the fixture.
    pub fn remove_file(&mut self, path: &str) -> Option<Vec<u8>> {
        self.files.remove(path)
    }

    /// Parse a minimal tar archive format.
    fn parse_tar(&mut self, data: &[u8]) -> Result<(), FixtureError> {
        const BLOCK_SIZE: usize = 512;

        let mut pos = 0;
        while pos + BLOCK_SIZE <= data.len() {
            let header = &data[pos..pos + BLOCK_SIZE];

            // Check for end of archive (two empty blocks)
            if header.iter().all(|&b| b == 0) {
                break;
            }

            // Parse tar header
            let name = Self::parse_tar_string(&header[0..100]);
            if name.is_empty() {
                break;
            }

            // Parse file size (octal)
            let size_str = Self::parse_tar_string(&header[124..136]);
            let size = usize::from_str_radix(size_str.trim(), 8)
                .map_err(|_| FixtureError::InvalidTar("invalid file size".to_string()))?;

            // Parse file type (0 = regular file, 5 = directory)
            let typeflag = header[156];

            pos += BLOCK_SIZE; // Move past header

            // Only process regular files
            if typeflag == b'0' || typeflag == 0 {
                let content = if pos + size <= data.len() {
                    data[pos..pos + size].to_vec()
                } else {
                    return Err(FixtureError::InvalidTar(
                        "truncated file content".to_string(),
                    ));
                };

                self.files.insert(name, content);
            }

            // Move to next block boundary
            pos += (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        }

        Ok(())
    }

    /// Parse a null-terminated string from tar header.
    fn parse_tar_string(data: &[u8]) -> String {
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        String::from_utf8_lossy(&data[..end]).trim().to_string()
    }

    /// Build manifest from loaded files.
    fn build_manifest(&mut self) {
        // Find app.yaml
        for name in &["app.yaml", "app.yml", "presentar.yaml", "presentar.yml"] {
            if self.files.contains_key(*name) {
                self.manifest.app_yaml = Some((*name).to_string());
                break;
            }
        }

        // Categorize files
        for path in self.files.keys() {
            if path.ends_with(".ald") {
                self.manifest.data_files.push(path.clone());
            } else if path.ends_with(".apr") {
                self.manifest.model_files.push(path.clone());
            } else if path.ends_with(".png") || path.ends_with(".jpg") || path.ends_with(".svg") {
                if path.starts_with("snapshots/") {
                    self.manifest.snapshots.push(path.clone());
                } else {
                    self.manifest.assets.push(path.clone());
                }
            } else if path.ends_with(".ttf") || path.ends_with(".otf") || path.ends_with(".woff2") {
                self.manifest.assets.push(path.clone());
            }
        }
    }

    /// Get fixture file count.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Get total fixture size in bytes.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.files.values().map(Vec::len).sum()
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating fixtures programmatically.
#[derive(Debug, Default)]
pub struct FixtureBuilder {
    files: Vec<(String, Vec<u8>)>,
    name: String,
}

impl FixtureBuilder {
    /// Create a new fixture builder.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            files: Vec::new(),
            name: name.into(),
        }
    }

    /// Add a file with string content.
    #[must_use]
    pub fn file(mut self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.files.push((path.into(), content.into().into_bytes()));
        self
    }

    /// Add a file with binary content.
    #[must_use]
    pub fn binary_file(mut self, path: impl Into<String>, content: Vec<u8>) -> Self {
        self.files.push((path.into(), content));
        self
    }

    /// Add app.yaml configuration.
    #[must_use]
    pub fn app_yaml(self, content: impl Into<String>) -> Self {
        self.file("app.yaml", content)
    }

    /// Add a data file.
    #[must_use]
    pub fn data(self, name: impl Into<String>, content: Vec<u8>) -> Self {
        let name = name.into();
        let path = if name.ends_with(".ald") {
            name
        } else {
            format!("{name}.ald")
        };
        self.binary_file(path, content)
    }

    /// Add a snapshot baseline.
    #[must_use]
    pub fn snapshot(self, name: impl Into<String>, png_data: Vec<u8>) -> Self {
        let name = name.into();
        let path = format!("snapshots/{name}.png");
        self.binary_file(path, png_data)
    }

    /// Build the fixture.
    #[must_use]
    pub fn build(self) -> Fixture {
        let files: Vec<(&str, &[u8])> = self
            .files
            .iter()
            .map(|(p, c)| (p.as_str(), c.as_slice()))
            .collect();
        let mut fixture = Fixture::from_files(files);
        fixture.manifest.name = self.name;
        fixture
    }
}

/// Test data generator for common test scenarios.
pub struct TestData;

impl TestData {
    /// Generate sample metrics data.
    #[must_use]
    pub fn metrics_json(count: usize) -> String {
        let mut data = String::from("[");
        for i in 0..count {
            if i > 0 {
                data.push(',');
            }
            data.push_str(&format!(
                r#"{{"timestamp":{},"cpu":{},"memory":{},"requests":{}}}"#,
                1700000000 + i * 60,
                20.0 + (i as f64 * 0.5).sin() * 10.0,
                45.0 + (i as f64 * 0.3).cos() * 15.0,
                100 + (i % 50)
            ));
        }
        data.push(']');
        data
    }

    /// Generate sample chart data.
    #[must_use]
    pub fn chart_points(count: usize) -> Vec<(f64, f64)> {
        (0..count)
            .map(|i| {
                let x = i as f64;
                let y = (x * 0.1).sin() * 50.0 + 50.0;
                (x, y)
            })
            .collect()
    }

    /// Generate sample table data as CSV.
    #[must_use]
    pub fn table_csv(rows: usize, cols: usize) -> String {
        let mut csv = String::new();

        // Header
        for c in 0..cols {
            if c > 0 {
                csv.push(',');
            }
            csv.push_str(&format!("col{c}"));
        }
        csv.push('\n');

        // Rows
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    csv.push(',');
                }
                csv.push_str(&format!("r{r}c{c}"));
            }
            csv.push('\n');
        }

        csv
    }

    /// Generate minimal PNG image data.
    #[must_use]
    pub fn minimal_png(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
        // This is a minimal valid PNG with solid color
        // In real implementation would use proper PNG encoding
        let mut png = Vec::new();

        // PNG signature
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

        // IHDR chunk
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(8); // bit depth
        ihdr.push(6); // color type (RGBA)
        ihdr.push(0); // compression
        ihdr.push(0); // filter
        ihdr.push(0); // interlace

        png.extend_from_slice(&(ihdr.len() as u32).to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&Self::crc32(b"IHDR", &ihdr).to_be_bytes());

        // IDAT chunk (minimal - just one pixel repeated conceptually)
        // This is simplified; real PNG needs proper zlib compression
        let mut idat = vec![0x08, 0x1D]; // zlib header
        let pixel_row_size = 1 + width as usize * 4; // filter byte + RGBA
        let raw_size = pixel_row_size * height as usize;

        // Simplified: store uncompressed (this is not valid zlib but demonstrates structure)
        idat.push(0x01); // final block, uncompressed
        idat.extend_from_slice(&(raw_size as u16).to_le_bytes());
        idat.extend_from_slice(&(!(raw_size as u16)).to_le_bytes());

        for _ in 0..height {
            idat.push(0); // filter: none
            for _ in 0..width {
                idat.extend_from_slice(&color);
            }
        }

        // Adler32 checksum (simplified)
        let adler = Self::adler32(&idat[2..]);
        idat.extend_from_slice(&adler.to_be_bytes());

        png.extend_from_slice(&(idat.len() as u32).to_be_bytes());
        png.extend_from_slice(b"IDAT");
        png.extend_from_slice(&idat);
        png.extend_from_slice(&Self::crc32(b"IDAT", &idat).to_be_bytes());

        // IEND chunk
        png.extend_from_slice(&0u32.to_be_bytes());
        png.extend_from_slice(b"IEND");
        png.extend_from_slice(&Self::crc32(b"IEND", &[]).to_be_bytes());

        png
    }

    /// Compute CRC32 for PNG chunk.
    fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
        const CRC_TABLE: [u32; 256] = {
            let mut table = [0u32; 256];
            let mut i = 0;
            while i < 256 {
                let mut c = i as u32;
                let mut k = 0;
                while k < 8 {
                    if c & 1 != 0 {
                        c = 0xedb8_8320 ^ (c >> 1);
                    } else {
                        c >>= 1;
                    }
                    k += 1;
                }
                table[i] = c;
                i += 1;
            }
            table
        };

        let mut crc = 0xFFFF_FFFF_u32;
        for &byte in chunk_type.iter().chain(data.iter()) {
            crc = CRC_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        !crc
    }

    /// Compute Adler32 for zlib.
    fn adler32(data: &[u8]) -> u32 {
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }
}

/// Context for fixture-based tests.
#[derive(Debug)]
pub struct FixtureContext {
    /// The loaded fixture
    pub fixture: Fixture,
    /// Test name
    pub test_name: String,
    /// Output directory for generated files
    pub output_dir: Option<String>,
}

impl FixtureContext {
    /// Create a new fixture context.
    #[must_use]
    pub fn new(fixture: Fixture, test_name: impl Into<String>) -> Self {
        Self {
            fixture,
            test_name: test_name.into(),
            output_dir: None,
        }
    }

    /// Set output directory.
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }

    /// Get the fixture.
    #[must_use]
    pub fn fixture(&self) -> &Fixture {
        &self.fixture
    }

    /// Get app yaml.
    pub fn app_yaml(&self) -> Result<&str, FixtureError> {
        self.fixture.get_app_yaml()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Fixture Tests
    // =========================================================================

    #[test]
    fn test_fixture_new() {
        let fixture = Fixture::new();
        assert_eq!(fixture.file_count(), 0);
        assert_eq!(fixture.total_size(), 0);
    }

    #[test]
    fn test_fixture_from_files() {
        let fixture = Fixture::from_files(vec![
            ("app.yaml", b"name: test" as &[u8]),
            ("data.ald", b"binary data"),
        ]);

        assert_eq!(fixture.file_count(), 2);
        assert!(fixture.has_file("app.yaml"));
        assert!(fixture.has_file("data.ald"));
    }

    #[test]
    fn test_fixture_get_file() {
        let fixture = Fixture::from_files(vec![("test.txt", b"hello world" as &[u8])]);

        let content = fixture.get_file("test.txt").unwrap();
        assert_eq!(content, b"hello world");
    }

    #[test]
    fn test_fixture_get_file_str() {
        let fixture = Fixture::from_files(vec![("test.txt", b"hello world" as &[u8])]);

        let content = fixture.get_file_str("test.txt").unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_fixture_get_file_not_found() {
        let fixture = Fixture::new();
        let result = fixture.get_file("missing.txt");
        assert!(matches!(result, Err(FixtureError::FileNotFound(_))));
    }

    #[test]
    fn test_fixture_list_files() {
        let fixture = Fixture::from_files(vec![
            ("a.txt", b"a" as &[u8]),
            ("b.txt", b"b"),
            ("c.txt", b"c"),
        ]);

        let files = fixture.list_files();
        assert_eq!(files.len(), 3);
        assert!(files.contains(&"a.txt"));
        assert!(files.contains(&"b.txt"));
        assert!(files.contains(&"c.txt"));
    }

    #[test]
    fn test_fixture_add_remove_file() {
        let mut fixture = Fixture::new();

        fixture.add_file("test.txt", b"content".to_vec());
        assert!(fixture.has_file("test.txt"));

        let removed = fixture.remove_file("test.txt");
        assert_eq!(removed, Some(b"content".to_vec()));
        assert!(!fixture.has_file("test.txt"));
    }

    #[test]
    fn test_fixture_total_size() {
        let fixture = Fixture::from_files(vec![("a.txt", b"12345" as &[u8]), ("b.txt", b"67890")]);

        assert_eq!(fixture.total_size(), 10);
    }

    // =========================================================================
    // Manifest Tests
    // =========================================================================

    #[test]
    fn test_manifest_app_yaml_detection() {
        let fixture = Fixture::from_files(vec![("app.yaml", b"name: test" as &[u8])]);

        assert_eq!(fixture.manifest().app_yaml, Some("app.yaml".to_string()));
    }

    #[test]
    fn test_manifest_app_yml_detection() {
        let fixture = Fixture::from_files(vec![("app.yml", b"name: test" as &[u8])]);

        assert_eq!(fixture.manifest().app_yaml, Some("app.yml".to_string()));
    }

    #[test]
    fn test_manifest_data_files() {
        let fixture = Fixture::from_files(vec![
            ("metrics.ald", b"data" as &[u8]),
            ("users.ald", b"data"),
        ]);

        assert_eq!(fixture.manifest().data_files.len(), 2);
    }

    #[test]
    fn test_manifest_model_files() {
        let fixture = Fixture::from_files(vec![("model.apr", b"data" as &[u8])]);

        assert_eq!(fixture.manifest().model_files.len(), 1);
    }

    #[test]
    fn test_manifest_snapshots() {
        let fixture = Fixture::from_files(vec![
            ("snapshots/button.png", b"png" as &[u8]),
            ("snapshots/chart.png", b"png"),
        ]);

        assert_eq!(fixture.manifest().snapshots.len(), 2);
    }

    #[test]
    fn test_manifest_assets() {
        let fixture = Fixture::from_files(vec![
            ("assets/logo.png", b"png" as &[u8]),
            ("fonts/roboto.ttf", b"ttf"),
        ]);

        assert_eq!(fixture.manifest().assets.len(), 2);
    }

    // =========================================================================
    // Get Data/Model Tests
    // =========================================================================

    #[test]
    fn test_get_data_with_extension() {
        let fixture = Fixture::from_files(vec![("metrics.ald", b"data" as &[u8])]);

        let data = fixture.get_data("metrics.ald").unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn test_get_data_without_extension() {
        let fixture = Fixture::from_files(vec![("metrics.ald", b"data" as &[u8])]);

        let data = fixture.get_data("metrics").unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn test_get_model() {
        let fixture = Fixture::from_files(vec![("model.apr", b"model data" as &[u8])]);

        let data = fixture.get_model("model").unwrap();
        assert_eq!(data, b"model data");
    }

    #[test]
    fn test_get_snapshot() {
        let fixture = Fixture::from_files(vec![("snapshots/button.png", b"png" as &[u8])]);

        let data = fixture.get_snapshot("button").unwrap();
        assert_eq!(data, b"png");
    }

    // =========================================================================
    // FixtureBuilder Tests
    // =========================================================================

    #[test]
    fn test_builder_new() {
        let fixture = FixtureBuilder::new("test-fixture").build();
        assert_eq!(fixture.manifest().name, "test-fixture");
    }

    #[test]
    fn test_builder_file() {
        let fixture = FixtureBuilder::new("test")
            .file("test.txt", "hello")
            .build();

        assert_eq!(fixture.get_file_str("test.txt").unwrap(), "hello");
    }

    #[test]
    fn test_builder_binary_file() {
        let fixture = FixtureBuilder::new("test")
            .binary_file("data.bin", vec![1, 2, 3, 4])
            .build();

        assert_eq!(fixture.get_file("data.bin").unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_builder_app_yaml() {
        let fixture = FixtureBuilder::new("test")
            .app_yaml("name: my-app\nversion: 1.0")
            .build();

        assert!(fixture.get_app_yaml().is_ok());
        assert!(fixture.get_app_yaml().unwrap().contains("my-app"));
    }

    #[test]
    fn test_builder_data() {
        let fixture = FixtureBuilder::new("test")
            .data("metrics", vec![1, 2, 3])
            .build();

        assert_eq!(fixture.get_data("metrics").unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_builder_snapshot() {
        let fixture = FixtureBuilder::new("test")
            .snapshot("button", vec![0x89, b'P', b'N', b'G'])
            .build();

        assert!(fixture.get_snapshot("button").is_ok());
    }

    #[test]
    fn test_builder_chaining() {
        let fixture = FixtureBuilder::new("complex-fixture")
            .app_yaml("name: test")
            .file("readme.md", "# Test")
            .data("data1", vec![1, 2, 3])
            .data("data2", vec![4, 5, 6])
            .snapshot("main", vec![0])
            .build();

        assert_eq!(fixture.file_count(), 5);
    }

    // =========================================================================
    // TestData Tests
    // =========================================================================

    #[test]
    fn test_metrics_json() {
        let json = TestData::metrics_json(5);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("timestamp"));
        assert!(json.contains("cpu"));
        assert!(json.contains("memory"));
    }

    #[test]
    fn test_metrics_json_count() {
        let json = TestData::metrics_json(10);
        let count = json.matches("timestamp").count();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_chart_points() {
        let points = TestData::chart_points(100);
        assert_eq!(points.len(), 100);
        assert_eq!(points[0].0, 0.0);
        assert_eq!(points[99].0, 99.0);
    }

    #[test]
    fn test_chart_points_values() {
        let points = TestData::chart_points(10);
        for (x, y) in &points {
            assert!(*y >= 0.0 && *y <= 100.0, "y={y} should be in [0, 100]");
            assert!(*x >= 0.0);
        }
    }

    #[test]
    fn test_table_csv() {
        let csv = TestData::table_csv(3, 2);
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 4); // header + 3 rows
        assert_eq!(lines[0], "col0,col1");
        assert_eq!(lines[1], "r0c0,r0c1");
    }

    #[test]
    fn test_table_csv_dimensions() {
        let csv = TestData::table_csv(5, 4);
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 6); // header + 5 rows

        // Check column count
        assert_eq!(lines[0].split(',').count(), 4);
    }

    #[test]
    fn test_minimal_png_structure() {
        let png = TestData::minimal_png(2, 2, [255, 0, 0, 255]);

        // Check PNG signature
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn test_minimal_png_has_ihdr() {
        let png = TestData::minimal_png(4, 4, [0, 255, 0, 255]);

        // Find IHDR chunk
        let ihdr_pos = png.windows(4).position(|w| w == b"IHDR");
        assert!(ihdr_pos.is_some());
    }

    #[test]
    fn test_minimal_png_has_iend() {
        let png = TestData::minimal_png(4, 4, [0, 0, 255, 255]);

        // Find IEND chunk
        let iend_pos = png.windows(4).position(|w| w == b"IEND");
        assert!(iend_pos.is_some());
    }

    // =========================================================================
    // Tar Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_tar_string() {
        let data = b"hello\0\0\0\0\0";
        let result = Fixture::parse_tar_string(data);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_parse_tar_string_empty() {
        let data = b"\0\0\0\0\0";
        let result = Fixture::parse_tar_string(data);
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_tar_string_no_null() {
        let data = b"hello";
        let result = Fixture::parse_tar_string(data);
        assert_eq!(result, "hello");
    }

    // =========================================================================
    // FixtureContext Tests
    // =========================================================================

    #[test]
    fn test_context_new() {
        let fixture = FixtureBuilder::new("test").app_yaml("name: test").build();
        let context = FixtureContext::new(fixture, "my_test");

        assert_eq!(context.test_name, "my_test");
        assert!(context.output_dir.is_none());
    }

    #[test]
    fn test_context_with_output_dir() {
        let fixture = Fixture::new();
        let context = FixtureContext::new(fixture, "test").with_output_dir("/tmp/output");

        assert_eq!(context.output_dir, Some("/tmp/output".to_string()));
    }

    #[test]
    fn test_context_app_yaml() {
        let fixture = FixtureBuilder::new("test").app_yaml("name: my-app").build();
        let context = FixtureContext::new(fixture, "test");

        let yaml = context.app_yaml().unwrap();
        assert!(yaml.contains("my-app"));
    }

    // =========================================================================
    // FixtureError Tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let errors = vec![
            (
                FixtureError::InvalidTar("bad header".to_string()),
                "invalid tar archive: bad header",
            ),
            (
                FixtureError::FileNotFound("test.txt".to_string()),
                "file not found: test.txt",
            ),
            (
                FixtureError::InvalidFormat("bad format".to_string()),
                "invalid fixture format: bad format",
            ),
            (
                FixtureError::IoError("read failed".to_string()),
                "IO error: read failed",
            ),
            (
                FixtureError::YamlError("parse error".to_string()),
                "YAML error: parse error",
            ),
        ];

        for (error, expected) in errors {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn test_error_equality() {
        let e1 = FixtureError::FileNotFound("test.txt".to_string());
        let e2 = FixtureError::FileNotFound("test.txt".to_string());
        let e3 = FixtureError::FileNotFound("other.txt".to_string());

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_fixture_workflow() {
        // Build a complete fixture
        let fixture = FixtureBuilder::new("dashboard-fixture")
            .app_yaml(
                r#"
name: Dashboard
version: 1.0.0
layout:
  type: column
  children:
    - type: chart
      data: "{{ metrics }}"
"#,
            )
            .data("metrics", TestData::metrics_json(100).into_bytes())
            .snapshot(
                "dashboard",
                TestData::minimal_png(100, 100, [255, 255, 255, 255]),
            )
            .build();

        // Verify structure
        assert!(fixture.has_file("app.yaml"));
        assert!(fixture.has_file("metrics.ald"));
        assert!(fixture.has_file("snapshots/dashboard.png"));

        // Verify manifest
        let manifest = fixture.manifest();
        assert_eq!(manifest.name, "dashboard-fixture");
        assert!(manifest.app_yaml.is_some());
        assert_eq!(manifest.data_files.len(), 1);
        assert_eq!(manifest.snapshots.len(), 1);

        // Verify app yaml access
        let yaml = fixture.get_app_yaml().unwrap();
        assert!(yaml.contains("Dashboard"));
        assert!(yaml.contains("metrics"));
    }

    #[test]
    fn test_fixture_as_harness_input() {
        // Simulate how fixtures would be used with test harness
        let fixture = FixtureBuilder::new("button-test")
            .app_yaml(
                r#"
name: Button Test
widgets:
  - type: button
    text: "Click Me"
    test-id: submit-btn
"#,
            )
            .build();

        let context = FixtureContext::new(fixture, "test_button_click");
        let yaml = context.app_yaml().unwrap();

        assert!(yaml.contains("submit-btn"));
    }
}
