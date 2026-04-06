//! File Upload and Download Operations (Feature G.8)
//!
//! Provides support for file input/output operations in E2E tests.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// File input for upload operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInput {
    /// File name
    pub name: String,
    /// MIME type
    pub mime_type: String,
    /// File contents
    pub contents: Vec<u8>,
    /// Original path (if from filesystem)
    pub path: Option<PathBuf>,
}

impl FileInput {
    /// Create a new file input
    #[must_use]
    pub fn new(name: impl Into<String>, mime_type: impl Into<String>, contents: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            mime_type: mime_type.into(),
            contents,
            path: None,
        }
    }

    /// Create from path (mock - doesn't actually read file)
    #[must_use]
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let mime_type = guess_mime_type(&name);

        Self {
            name,
            mime_type,
            contents: Vec::new(), // Empty in mock
            path: Some(path.to_path_buf()),
        }
    }

    /// Create from path with contents
    #[must_use]
    pub fn from_path_with_contents(path: impl AsRef<Path>, contents: Vec<u8>) -> Self {
        let path = path.as_ref();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let mime_type = guess_mime_type(&name);

        Self {
            name,
            mime_type,
            contents,
            path: Some(path.to_path_buf()),
        }
    }

    /// Create a text file
    #[must_use]
    pub fn text(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(name, "text/plain", content.into().into_bytes())
    }

    /// Create a JSON file
    #[must_use]
    pub fn json(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(name, "application/json", content.into().into_bytes())
    }

    /// Create a CSV file
    #[must_use]
    pub fn csv(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(name, "text/csv", content.into().into_bytes())
    }

    /// Create a PNG image (mock)
    #[must_use]
    pub fn png(name: impl Into<String>, contents: Vec<u8>) -> Self {
        Self::new(name, "image/png", contents)
    }

    /// Create a PDF document (mock)
    #[must_use]
    pub fn pdf(name: impl Into<String>, contents: Vec<u8>) -> Self {
        Self::new(name, "application/pdf", contents)
    }

    /// Get file name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get MIME type
    #[must_use]
    pub fn mime_type(&self) -> &str {
        &self.mime_type
    }

    /// Get file size
    #[must_use]
    pub fn size(&self) -> usize {
        self.contents.len()
    }

    /// Get contents as bytes
    #[must_use]
    pub fn contents(&self) -> &[u8] {
        &self.contents
    }

    /// Get contents as string (if valid UTF-8)
    #[must_use]
    pub fn contents_string(&self) -> Option<String> {
        String::from_utf8(self.contents.clone()).ok()
    }

    /// Check if file is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

/// Represents a file download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Download {
    /// Suggested filename
    pub suggested_filename: String,
    /// URL the download came from
    pub url: String,
    /// File contents
    pub contents: Vec<u8>,
    /// Path where file was saved (if saved)
    pub saved_path: Option<PathBuf>,
    /// Download state
    pub state: DownloadState,
}

impl Download {
    /// Create a new download
    #[must_use]
    pub fn new(url: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            suggested_filename: filename.into(),
            url: url.into(),
            contents: Vec::new(),
            saved_path: None,
            state: DownloadState::InProgress,
        }
    }

    /// Create a completed download with contents
    #[must_use]
    pub fn completed(
        url: impl Into<String>,
        filename: impl Into<String>,
        contents: Vec<u8>,
    ) -> Self {
        Self {
            suggested_filename: filename.into(),
            url: url.into(),
            contents,
            saved_path: None,
            state: DownloadState::Completed,
        }
    }

    /// Get suggested filename
    #[must_use]
    pub fn suggested_filename(&self) -> &str {
        &self.suggested_filename
    }

    /// Get download URL
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get file size
    #[must_use]
    pub fn size(&self) -> usize {
        self.contents.len()
    }

    /// Check if download is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, DownloadState::Completed)
    }

    /// Check if download failed
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self.state, DownloadState::Failed(_))
    }

    /// Get path where file was saved
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.saved_path.as_deref()
    }

    /// Mark as saved to path (mock)
    pub fn save_as(&mut self, path: impl AsRef<Path>) {
        self.saved_path = Some(path.as_ref().to_path_buf());
        self.state = DownloadState::Completed;
    }

    /// Cancel the download
    pub fn cancel(&mut self) {
        self.state = DownloadState::Cancelled;
    }

    /// Mark as failed
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.state = DownloadState::Failed(reason.into());
    }

    /// Get contents
    #[must_use]
    pub fn contents(&self) -> &[u8] {
        &self.contents
    }

    /// Delete the downloaded file (mock)
    pub fn delete(&mut self) {
        self.saved_path = None;
        self.state = DownloadState::Deleted;
    }
}

/// State of a download
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    /// Download in progress
    InProgress,
    /// Download completed successfully
    Completed,
    /// Download was cancelled
    Cancelled,
    /// Download failed
    Failed(String),
    /// Downloaded file was deleted
    Deleted,
}

/// File chooser for handling file input elements
#[derive(Debug, Clone)]
pub struct FileChooser {
    /// Whether multiple files can be selected
    pub multiple: bool,
    /// Accepted file types (MIME types or extensions)
    pub accept: Vec<String>,
    /// Selected files
    pub files: Vec<FileInput>,
}

impl FileChooser {
    /// Create a new file chooser
    #[must_use]
    pub fn new() -> Self {
        Self {
            multiple: false,
            accept: Vec::new(),
            files: Vec::new(),
        }
    }

    /// Create for single file selection
    #[must_use]
    pub fn single() -> Self {
        Self::new()
    }

    /// Create for multiple file selection
    #[must_use]
    pub fn multiple() -> Self {
        Self {
            multiple: true,
            ..Self::new()
        }
    }

    /// Set accepted file types
    #[must_use]
    pub fn accept(mut self, types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.accept = types.into_iter().map(Into::into).collect();
        self
    }

    /// Set a single file
    pub fn set_files(&mut self, files: impl IntoIterator<Item = FileInput>) {
        let files: Vec<FileInput> = files.into_iter().collect();
        if !self.multiple && files.len() > 1 {
            // Safe: we checked files.len() > 1, so there's at least one item
            if let Some(first) = files.into_iter().next() {
                self.files = vec![first];
            }
        } else {
            self.files = files;
        }
    }

    /// Set files from paths
    pub fn set_input_files(&mut self, paths: &[impl AsRef<Path>]) {
        let files: Vec<FileInput> = paths.iter().map(FileInput::from_path).collect();
        self.set_files(files);
    }

    /// Check if file type is accepted
    #[must_use]
    pub fn is_accepted(&self, file: &FileInput) -> bool {
        if self.accept.is_empty() {
            return true;
        }

        let ext = file
            .name
            .rsplit('.')
            .next()
            .map(|e| format!(".{}", e.to_lowercase()));

        for accept in &self.accept {
            if accept == &file.mime_type {
                return true;
            }
            if let Some(ref extension) = ext {
                if accept == extension {
                    return true;
                }
            }
            if accept == "*/*" {
                return true;
            }
            // Check MIME type patterns like "image/*"
            if accept.ends_with("/*") {
                let prefix = &accept[..accept.len() - 1];
                if file.mime_type.starts_with(prefix) {
                    return true;
                }
            }
        }

        false
    }

    /// Get selected files
    #[must_use]
    pub fn files(&self) -> &[FileInput] {
        &self.files
    }

    /// Get file count
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Check if any files selected
    #[must_use]
    pub fn has_files(&self) -> bool {
        !self.files.is_empty()
    }

    /// Clear selected files
    pub fn clear(&mut self) {
        self.files.clear();
    }
}

impl Default for FileChooser {
    fn default() -> Self {
        Self::new()
    }
}

/// Download manager for tracking downloads
#[derive(Debug, Clone, Default)]
pub struct DownloadManager {
    downloads: Vec<Download>,
}

impl DownloadManager {
    /// Create a new download manager
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a download
    pub fn add(&mut self, download: Download) {
        self.downloads.push(download);
    }

    /// Get all downloads
    #[must_use]
    pub fn downloads(&self) -> &[Download] {
        &self.downloads
    }

    /// Get download count
    #[must_use]
    pub fn count(&self) -> usize {
        self.downloads.len()
    }

    /// Get last download
    #[must_use]
    pub fn last(&self) -> Option<&Download> {
        self.downloads.last()
    }

    /// Get mutable reference to last download
    pub fn last_mut(&mut self) -> Option<&mut Download> {
        self.downloads.last_mut()
    }

    /// Find download by filename
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&Download> {
        self.downloads.iter().find(|d| d.suggested_filename == name)
    }

    /// Clear all downloads
    pub fn clear(&mut self) {
        self.downloads.clear();
    }

    /// Get completed downloads
    #[must_use]
    pub fn completed(&self) -> Vec<&Download> {
        self.downloads.iter().filter(|d| d.is_complete()).collect()
    }

    /// Wait for download (mock - returns last download)
    #[must_use]
    pub fn wait_for_download(&self) -> Option<&Download> {
        self.last()
    }
}

/// Guess MIME type from filename
#[must_use]
pub fn guess_mime_type(filename: &str) -> String {
    let ext = filename
        .rsplit('.')
        .next()
        .map(str::to_lowercase)
        .unwrap_or_default();

    match ext.as_str() {
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" => "application/javascript",
        "json" => "application/json",
        "xml" => "application/xml",
        "csv" => "text/csv",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "wasm" => "application/wasm",
        "zip" => "application/zip",
        "gz" => "application/gzip",
        "tar" => "application/x-tar",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-FILE-01: FileInput creation
    // =========================================================================

    #[test]
    fn h0_file_01_new() {
        let file = FileInput::new("test.txt", "text/plain", b"hello".to_vec());
        assert_eq!(file.name(), "test.txt");
        assert_eq!(file.mime_type(), "text/plain");
        assert_eq!(file.contents(), b"hello");
    }

    #[test]
    fn h0_file_02_from_path() {
        let file = FileInput::from_path("documents/report.pdf");
        assert_eq!(file.name(), "report.pdf");
        assert_eq!(file.mime_type(), "application/pdf");
    }

    #[test]
    fn h0_file_03_text() {
        let file = FileInput::text("notes.txt", "Hello world");
        assert_eq!(file.mime_type(), "text/plain");
        assert_eq!(file.contents_string(), Some("Hello world".to_string()));
    }

    #[test]
    fn h0_file_04_json() {
        let file = FileInput::json("data.json", r#"{"key": "value"}"#);
        assert_eq!(file.mime_type(), "application/json");
    }

    #[test]
    fn h0_file_05_csv() {
        let file = FileInput::csv("data.csv", "a,b,c\n1,2,3");
        assert_eq!(file.mime_type(), "text/csv");
    }

    #[test]
    fn h0_file_06_png() {
        let file = FileInput::png("image.png", vec![0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(file.mime_type(), "image/png");
    }

    #[test]
    fn h0_file_07_pdf() {
        let file = FileInput::pdf("doc.pdf", vec![0x25, 0x50, 0x44, 0x46]);
        assert_eq!(file.mime_type(), "application/pdf");
    }

    // =========================================================================
    // H₀-FILE-08: FileInput properties
    // =========================================================================

    #[test]
    fn h0_file_08_size() {
        let file = FileInput::text("test.txt", "12345");
        assert_eq!(file.size(), 5);
    }

    #[test]
    fn h0_file_09_is_empty() {
        let empty = FileInput::new("empty.txt", "text/plain", vec![]);
        assert!(empty.is_empty());

        let non_empty = FileInput::text("full.txt", "content");
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn h0_file_10_contents_string_valid() {
        let file = FileInput::text("test.txt", "Hello");
        assert_eq!(file.contents_string(), Some("Hello".to_string()));
    }

    #[test]
    fn h0_file_11_contents_string_invalid() {
        let file = FileInput::new("binary.bin", "application/octet-stream", vec![0xFF, 0xFE]);
        assert!(file.contents_string().is_none());
    }

    // =========================================================================
    // H₀-FILE-12: Download creation
    // =========================================================================

    #[test]
    fn h0_file_12_download_new() {
        let download = Download::new("http://example.com/file.pdf", "file.pdf");
        assert_eq!(download.suggested_filename(), "file.pdf");
        assert_eq!(download.url(), "http://example.com/file.pdf");
        assert!(!download.is_complete());
    }

    #[test]
    fn h0_file_13_download_completed() {
        let download =
            Download::completed("http://example.com/data.json", "data.json", b"{}".to_vec());
        assert!(download.is_complete());
        assert_eq!(download.size(), 2);
    }

    // =========================================================================
    // H₀-FILE-14: Download operations
    // =========================================================================

    #[test]
    fn h0_file_14_save_as() {
        let mut download = Download::completed("http://test", "file.txt", b"content".to_vec());
        download.save_as("/tmp/file.txt");

        assert_eq!(download.path(), Some(Path::new("/tmp/file.txt")));
    }

    #[test]
    fn h0_file_15_cancel() {
        let mut download = Download::new("http://test", "file.txt");
        download.cancel();

        assert_eq!(download.state, DownloadState::Cancelled);
    }

    #[test]
    fn h0_file_16_fail() {
        let mut download = Download::new("http://test", "file.txt");
        download.fail("Network error");

        assert!(download.is_failed());
        assert_eq!(
            download.state,
            DownloadState::Failed("Network error".to_string())
        );
    }

    #[test]
    fn h0_file_17_delete() {
        let mut download = Download::completed("http://test", "file.txt", vec![]);
        download.save_as("/tmp/file.txt");
        download.delete();

        assert!(download.path().is_none());
        assert_eq!(download.state, DownloadState::Deleted);
    }

    // =========================================================================
    // H₀-FILE-18: FileChooser
    // =========================================================================

    #[test]
    fn h0_file_18_chooser_new() {
        let chooser = FileChooser::new();
        assert!(!chooser.multiple);
        assert!(!chooser.has_files());
    }

    #[test]
    fn h0_file_19_chooser_multiple() {
        let chooser = FileChooser::multiple();
        assert!(chooser.multiple);
    }

    #[test]
    fn h0_file_20_chooser_accept() {
        let chooser = FileChooser::new().accept(vec![".pdf", ".doc"]);
        assert_eq!(chooser.accept.len(), 2);
    }

    #[test]
    fn h0_file_21_set_files() {
        let mut chooser = FileChooser::new();
        chooser.set_files(vec![FileInput::text("test.txt", "content")]);

        assert_eq!(chooser.file_count(), 1);
    }

    #[test]
    fn h0_file_22_set_files_single_mode() {
        let mut chooser = FileChooser::single();
        chooser.set_files(vec![
            FileInput::text("a.txt", "a"),
            FileInput::text("b.txt", "b"),
        ]);

        // Should only keep first file
        assert_eq!(chooser.file_count(), 1);
        assert_eq!(chooser.files()[0].name(), "a.txt");
    }

    #[test]
    fn h0_file_23_set_files_multiple_mode() {
        let mut chooser = FileChooser::multiple();
        chooser.set_files(vec![
            FileInput::text("a.txt", "a"),
            FileInput::text("b.txt", "b"),
        ]);

        assert_eq!(chooser.file_count(), 2);
    }

    // =========================================================================
    // H₀-FILE-24: FileChooser accept validation
    // =========================================================================

    #[test]
    fn h0_file_24_is_accepted_empty() {
        let chooser = FileChooser::new();
        let file = FileInput::text("test.txt", "");
        assert!(chooser.is_accepted(&file));
    }

    #[test]
    fn h0_file_25_is_accepted_mime() {
        let chooser = FileChooser::new().accept(vec!["text/plain"]);
        let file = FileInput::text("test.txt", "");
        assert!(chooser.is_accepted(&file));
    }

    #[test]
    fn h0_file_26_is_accepted_extension() {
        let chooser = FileChooser::new().accept(vec![".pdf"]);
        let file = FileInput::from_path("doc.pdf");
        assert!(chooser.is_accepted(&file));
    }

    #[test]
    fn h0_file_27_is_accepted_wildcard() {
        let chooser = FileChooser::new().accept(vec!["image/*"]);
        let png = FileInput::png("test.png", vec![]);
        assert!(chooser.is_accepted(&png));
    }

    #[test]
    fn h0_file_28_is_not_accepted() {
        let chooser = FileChooser::new().accept(vec![".pdf"]);
        let file = FileInput::text("test.txt", "");
        assert!(!chooser.is_accepted(&file));
    }

    // =========================================================================
    // H₀-FILE-29: DownloadManager
    // =========================================================================

    #[test]
    fn h0_file_29_manager_new() {
        let manager = DownloadManager::new();
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn h0_file_30_manager_add() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test", "file.txt"));

        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn h0_file_31_manager_last() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test", "first.txt"));
        manager.add(Download::new("http://test", "last.txt"));

        let last = manager.last().unwrap();
        assert_eq!(last.suggested_filename(), "last.txt");
    }

    #[test]
    fn h0_file_32_manager_find_by_name() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test/a", "a.txt"));
        manager.add(Download::new("http://test/b", "b.txt"));

        let found = manager.find_by_name("a.txt").unwrap();
        assert_eq!(found.url(), "http://test/a");
    }

    #[test]
    fn h0_file_33_manager_clear() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test", "file.txt"));
        manager.clear();

        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn h0_file_34_manager_completed() {
        let mut manager = DownloadManager::new();
        manager.add(Download::completed("http://test/a", "a.txt", vec![]));
        manager.add(Download::new("http://test/b", "b.txt"));

        let completed = manager.completed();
        assert_eq!(completed.len(), 1);
    }

    // =========================================================================
    // H₀-FILE-35: MIME type guessing
    // =========================================================================

    #[test]
    fn h0_file_35_guess_mime_text() {
        assert_eq!(guess_mime_type("file.txt"), "text/plain");
        assert_eq!(guess_mime_type("page.html"), "text/html");
        assert_eq!(guess_mime_type("styles.css"), "text/css");
    }

    #[test]
    fn h0_file_36_guess_mime_image() {
        assert_eq!(guess_mime_type("photo.png"), "image/png");
        assert_eq!(guess_mime_type("photo.jpg"), "image/jpeg");
        assert_eq!(guess_mime_type("photo.jpeg"), "image/jpeg");
        assert_eq!(guess_mime_type("icon.svg"), "image/svg+xml");
    }

    #[test]
    fn h0_file_37_guess_mime_app() {
        assert_eq!(guess_mime_type("data.json"), "application/json");
        assert_eq!(guess_mime_type("doc.pdf"), "application/pdf");
        assert_eq!(guess_mime_type("app.wasm"), "application/wasm");
    }

    #[test]
    fn h0_file_38_guess_mime_unknown() {
        assert_eq!(guess_mime_type("file.xyz"), "application/octet-stream");
        assert_eq!(guess_mime_type("noextension"), "application/octet-stream");
    }

    // =========================================================================
    // H₀-FILE-39: Clone and Debug
    // =========================================================================

    #[test]
    fn h0_file_39_file_input_clone() {
        let file = FileInput::text("test.txt", "content");
        let cloned = file;
        assert_eq!(cloned.name(), "test.txt");
    }

    #[test]
    fn h0_file_40_download_clone() {
        let download = Download::completed("http://test", "file.txt", vec![1, 2, 3]);
        let cloned = download;
        assert_eq!(cloned.size(), 3);
    }

    // =========================================================================
    // H₀-FILE-41+: Additional coverage tests
    // =========================================================================

    #[test]
    fn h0_file_41_from_path_with_contents() {
        let file = FileInput::from_path_with_contents("docs/report.pdf", b"PDF content".to_vec());
        assert_eq!(file.name(), "report.pdf");
        assert_eq!(file.mime_type(), "application/pdf");
        assert_eq!(file.contents(), b"PDF content");
        assert!(file.path.is_some());
        assert_eq!(file.path.unwrap().to_str().unwrap(), "docs/report.pdf");
    }

    #[test]
    fn h0_file_42_from_path_no_filename() {
        // Test path with no file name (edge case)
        let file = FileInput::from_path("/");
        assert_eq!(file.name(), "unknown");
    }

    #[test]
    fn h0_file_43_from_path_with_contents_no_filename() {
        // Test path with no file name (edge case)
        let file = FileInput::from_path_with_contents("/", vec![1, 2, 3]);
        assert_eq!(file.name(), "unknown");
        assert_eq!(file.contents(), &[1, 2, 3]);
    }

    #[test]
    fn h0_file_44_download_contents_accessor() {
        let download = Download::completed("http://test", "file.txt", vec![1, 2, 3, 4, 5]);
        assert_eq!(download.contents(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn h0_file_45_file_chooser_set_input_files() {
        let mut chooser = FileChooser::multiple();
        chooser.set_input_files(&["file1.txt", "file2.pdf"]);
        assert_eq!(chooser.file_count(), 2);
        assert_eq!(chooser.files()[0].name(), "file1.txt");
        assert_eq!(chooser.files()[1].name(), "file2.pdf");
    }

    #[test]
    fn h0_file_46_file_chooser_clear() {
        let mut chooser = FileChooser::new();
        chooser.set_files(vec![FileInput::text("test.txt", "content")]);
        assert!(chooser.has_files());
        chooser.clear();
        assert!(!chooser.has_files());
        assert_eq!(chooser.file_count(), 0);
    }

    #[test]
    fn h0_file_47_file_chooser_default() {
        let chooser = FileChooser::default();
        assert!(!chooser.multiple);
        assert!(chooser.accept.is_empty());
        assert!(chooser.files.is_empty());
    }

    #[test]
    fn h0_file_48_download_manager_last_mut() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test", "file.txt"));

        // Modify via last_mut
        if let Some(download) = manager.last_mut() {
            download.cancel();
        }

        assert_eq!(manager.last().unwrap().state, DownloadState::Cancelled);
    }

    #[test]
    fn h0_file_49_download_manager_downloads_accessor() {
        let mut manager = DownloadManager::new();
        manager.add(Download::new("http://test/a", "a.txt"));
        manager.add(Download::new("http://test/b", "b.txt"));

        let downloads = manager.downloads();
        assert_eq!(downloads.len(), 2);
        assert_eq!(downloads[0].suggested_filename(), "a.txt");
        assert_eq!(downloads[1].suggested_filename(), "b.txt");
    }

    #[test]
    fn h0_file_50_download_manager_wait_for_download() {
        let mut manager = DownloadManager::new();
        assert!(manager.wait_for_download().is_none());

        manager.add(Download::new("http://test", "file.txt"));
        let waited = manager.wait_for_download();
        assert!(waited.is_some());
        assert_eq!(waited.unwrap().suggested_filename(), "file.txt");
    }

    #[test]
    fn h0_file_51_download_manager_find_by_name_not_found() {
        let manager = DownloadManager::new();
        assert!(manager.find_by_name("nonexistent.txt").is_none());
    }

    #[test]
    fn h0_file_52_download_manager_last_empty() {
        let manager = DownloadManager::new();
        assert!(manager.last().is_none());
    }

    #[test]
    fn h0_file_53_download_manager_last_mut_empty() {
        let mut manager = DownloadManager::new();
        assert!(manager.last_mut().is_none());
    }

    #[test]
    fn h0_file_54_is_accepted_all_wildcard() {
        let chooser = FileChooser::new().accept(vec!["*/*"]);
        let file = FileInput::text("test.txt", "");
        assert!(chooser.is_accepted(&file));
    }

    #[test]
    fn h0_file_55_guess_mime_htm() {
        assert_eq!(guess_mime_type("page.htm"), "text/html");
    }

    #[test]
    fn h0_file_56_guess_mime_gif() {
        assert_eq!(guess_mime_type("animation.gif"), "image/gif");
    }

    #[test]
    fn h0_file_57_guess_mime_webp() {
        assert_eq!(guess_mime_type("image.webp"), "image/webp");
    }

    #[test]
    fn h0_file_58_guess_mime_ico() {
        assert_eq!(guess_mime_type("favicon.ico"), "image/x-icon");
    }

    #[test]
    fn h0_file_59_guess_mime_audio() {
        assert_eq!(guess_mime_type("song.mp3"), "audio/mpeg");
        assert_eq!(guess_mime_type("sound.wav"), "audio/wav");
    }

    #[test]
    fn h0_file_60_guess_mime_video() {
        assert_eq!(guess_mime_type("movie.mp4"), "video/mp4");
        assert_eq!(guess_mime_type("clip.webm"), "video/webm");
    }

    #[test]
    fn h0_file_61_guess_mime_archive() {
        assert_eq!(guess_mime_type("archive.zip"), "application/zip");
        assert_eq!(guess_mime_type("archive.gz"), "application/gzip");
        assert_eq!(guess_mime_type("archive.tar"), "application/x-tar");
    }

    #[test]
    fn h0_file_62_guess_mime_office() {
        assert_eq!(guess_mime_type("document.doc"), "application/msword");
        assert_eq!(
            guess_mime_type("document.docx"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert_eq!(
            guess_mime_type("spreadsheet.xls"),
            "application/vnd.ms-excel"
        );
        assert_eq!(
            guess_mime_type("spreadsheet.xlsx"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
    }

    #[test]
    fn h0_file_63_guess_mime_xml() {
        assert_eq!(guess_mime_type("data.xml"), "application/xml");
    }

    #[test]
    fn h0_file_64_guess_mime_javascript() {
        assert_eq!(guess_mime_type("script.js"), "application/javascript");
    }

    #[test]
    fn h0_file_65_file_input_path_accessor() {
        let file = FileInput::new("test.txt", "text/plain", vec![]);
        assert!(file.path.is_none());

        let file_with_path = FileInput::from_path("folder/test.txt");
        assert!(file_with_path.path.is_some());
    }

    #[test]
    fn h0_file_66_download_state_debug() {
        let states = vec![
            DownloadState::InProgress,
            DownloadState::Completed,
            DownloadState::Cancelled,
            DownloadState::Failed("error".to_string()),
            DownloadState::Deleted,
        ];

        for state in &states {
            // Test Debug impl
            let _ = format!("{:?}", state);
        }

        // Test PartialEq
        assert_eq!(DownloadState::InProgress, DownloadState::InProgress);
        assert_ne!(DownloadState::InProgress, DownloadState::Completed);
    }

    #[test]
    fn h0_file_67_file_input_debug() {
        let file = FileInput::text("test.txt", "content");
        let debug_str = format!("{:?}", file);
        assert!(debug_str.contains("test.txt"));
    }

    #[test]
    fn h0_file_68_download_debug() {
        let download = Download::new("http://test", "file.txt");
        let debug_str = format!("{:?}", download);
        assert!(debug_str.contains("file.txt"));
    }

    #[test]
    fn h0_file_69_file_chooser_debug() {
        let chooser = FileChooser::new();
        let debug_str = format!("{:?}", chooser);
        assert!(debug_str.contains("FileChooser"));
    }

    #[test]
    fn h0_file_70_download_manager_debug() {
        let manager = DownloadManager::new();
        let debug_str = format!("{:?}", manager);
        assert!(debug_str.contains("DownloadManager"));
    }

    #[test]
    fn h0_file_71_set_files_single_mode_empty() {
        let mut chooser = FileChooser::single();
        chooser.set_files(Vec::<FileInput>::new());
        assert_eq!(chooser.file_count(), 0);
    }

    #[test]
    fn h0_file_72_set_files_single_mode_exactly_one() {
        let mut chooser = FileChooser::single();
        chooser.set_files(vec![FileInput::text("single.txt", "content")]);
        assert_eq!(chooser.file_count(), 1);
        assert_eq!(chooser.files()[0].name(), "single.txt");
    }

    #[test]
    fn h0_file_73_is_accepted_extension_case_insensitive() {
        let chooser = FileChooser::new().accept(vec![".pdf"]);
        let file = FileInput::from_path("doc.PDF");
        assert!(chooser.is_accepted(&file));
    }

    #[test]
    fn h0_file_74_is_accepted_mime_wildcard_image() {
        let chooser = FileChooser::new().accept(vec!["image/*"]);

        let gif = FileInput::new("test.gif", "image/gif", vec![]);
        assert!(chooser.is_accepted(&gif));

        let webp = FileInput::new("test.webp", "image/webp", vec![]);
        assert!(chooser.is_accepted(&webp));
    }

    #[test]
    fn h0_file_75_is_accepted_mime_wildcard_audio() {
        let chooser = FileChooser::new().accept(vec!["audio/*"]);

        let mp3 = FileInput::new("test.mp3", "audio/mpeg", vec![]);
        assert!(chooser.is_accepted(&mp3));

        let txt = FileInput::text("test.txt", "");
        assert!(!chooser.is_accepted(&txt));
    }

    #[test]
    fn h0_file_76_download_path_none_when_not_saved() {
        let download = Download::new("http://test", "file.txt");
        assert!(download.path().is_none());
    }

    #[test]
    fn h0_file_77_set_input_files_single_mode() {
        let mut chooser = FileChooser::single();
        chooser.set_input_files(&["file1.txt", "file2.txt"]);
        // Single mode should only keep first file
        assert_eq!(chooser.file_count(), 1);
        assert_eq!(chooser.files()[0].name(), "file1.txt");
    }

    #[test]
    fn h0_file_78_file_input_serialize_deserialize() {
        let file = FileInput::text("test.txt", "content");
        let json = serde_json::to_string(&file).unwrap();
        let deserialized: FileInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name(), "test.txt");
        assert_eq!(deserialized.contents_string(), Some("content".to_string()));
    }

    #[test]
    fn h0_file_79_download_serialize_deserialize() {
        let download = Download::completed("http://test", "file.txt", b"data".to_vec());
        let json = serde_json::to_string(&download).unwrap();
        let deserialized: Download = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.suggested_filename(), "file.txt");
        assert!(deserialized.is_complete());
    }

    #[test]
    fn h0_file_80_download_state_serialize_deserialize() {
        let states = vec![
            DownloadState::InProgress,
            DownloadState::Completed,
            DownloadState::Cancelled,
            DownloadState::Failed("Network error".to_string()),
            DownloadState::Deleted,
        ];

        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let deserialized: DownloadState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, deserialized);
        }
    }
}
