//! FTS5 full-text search with BM25 ranking.
//!
//! Implements search over the `chunks_fts` virtual table using SQLite's
//! built-in `bm25()` ranking function (Robertson & Zaragoza, 2009).

use crate::Result;
use rusqlite::Connection;

/// A single FTS5 search result with BM25 score.
#[derive(Debug, Clone)]
pub struct FtsResult {
    /// Chunk ID from the chunks table.
    pub chunk_id: String,
    /// Document ID the chunk belongs to.
    pub doc_id: String,
    /// Chunk text content.
    pub content: String,
    /// BM25 relevance score (lower is more relevant in SQLite's convention).
    /// We negate it so higher = more relevant.
    pub score: f64,
    /// Chunk position within its document.
    pub position: i64,
}

/// Escape user input for FTS5 MATCH syntax.
///
/// FTS5 treats certain characters as operators (AND, OR, NOT, *, ^, NEAR).
/// We wrap each token in double quotes to treat them as literal terms.
fn escape_fts5_query(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| {
            // Strip any existing quotes and re-wrap
            let clean = token.replace('"', "");
            if clean.is_empty() {
                return String::new();
            }
            format!("\"{clean}\"")
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Search the FTS5 index with BM25 ranking.
///
/// Returns up to `k` results ordered by descending relevance.
/// SQLite's `bm25()` returns negative scores where more negative = more
/// relevant, so we negate to produce positive scores where higher = better.
pub fn search(conn: &Connection, query: &str, k: usize) -> Result<Vec<FtsResult>> {
    let escaped = escape_fts5_query(query);
    if escaped.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn
        .prepare_cached(
            "SELECT c.id, c.doc_id, c.content, -bm25(chunks_fts) AS score, c.position
             FROM chunks_fts
             JOIN chunks c ON chunks_fts.rowid = c.rowid
             WHERE chunks_fts MATCH ?1
             ORDER BY score DESC
             LIMIT ?2",
        )
        .map_err(|e| crate::Error::Query(format!("Failed to prepare FTS5 search: {e}")))?;

    let results = stmt
        .query_map(rusqlite::params![escaped, k as i64], |row| {
            Ok(FtsResult {
                chunk_id: row.get(0)?,
                doc_id: row.get(1)?,
                content: row.get(2)?,
                score: row.get(3)?,
                position: row.get(4)?,
            })
        })
        .map_err(|e| crate::Error::Query(format!("FTS5 search failed: {e}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| crate::Error::Query(format!("FTS5 result mapping failed: {e}")))?;

    Ok(results)
}

/// Optimize the FTS5 index by merging segments.
///
/// Should be called after large batch inserts, not on every query.
pub fn optimize(conn: &Connection) -> Result<()> {
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')", [])
        .map_err(|e| crate::Error::Query(format!("FTS5 optimize failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::schema;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        schema::initialize(&conn).unwrap();
        conn
    }

    fn insert_chunk(conn: &Connection, doc_id: &str, chunk_id: &str, content: &str, pos: i64) {
        // Ensure document exists
        conn.execute("INSERT OR IGNORE INTO documents (id, content) VALUES (?1, '')", [doc_id])
            .unwrap();
        conn.execute(
            "INSERT INTO chunks (id, doc_id, content, position) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![chunk_id, doc_id, content, pos],
        )
        .unwrap();
    }

    #[test]
    fn test_search_returns_results() {
        let conn = setup();
        insert_chunk(&conn, "doc1", "c1", "SIMD vector operations for tensor math", 0);
        insert_chunk(&conn, "doc1", "c2", "GPU kernel dispatch and scheduling", 1);

        let results = search(&conn, "SIMD tensor", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[test]
    fn test_search_bm25_ordering() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "machine learning algorithms", 0);
        insert_chunk(
            &conn,
            "d2",
            "c2",
            "machine learning machine learning machine learning deep learning",
            0,
        );
        insert_chunk(&conn, "d3", "c3", "cooking recipes for dinner", 0);

        let results = search(&conn, "machine learning", 10).unwrap();
        // c2 has higher TF for "machine learning" so should rank higher
        assert!(results.len() >= 2);
        assert_eq!(results[0].chunk_id, "c2");
        assert_eq!(results[1].chunk_id, "c1");
        // "cooking recipes" should not match
        assert!(results.iter().all(|r| r.chunk_id != "c3"));
    }

    #[test]
    fn test_search_empty_query() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "some content", 0);
        let results = search(&conn, "", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_no_matches() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "SIMD vector operations", 0);
        let results = search(&conn, "cryptocurrency blockchain", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_escape_fts5_query_special_chars() {
        // Should wrap tokens in quotes to prevent FTS5 operator interpretation
        let escaped = escape_fts5_query("hello AND world");
        assert!(escaped.contains("\"hello\""));
        assert!(escaped.contains("\"AND\""));
        assert!(escaped.contains("\"world\""));
    }

    #[test]
    fn test_porter_stemming() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "tokenizer tokenization tokenizing", 0);

        // "tokenize" should match via Porter stemming conflation
        let results = search(&conn, "tokenize", 10).unwrap();
        assert!(!results.is_empty(), "Porter stemmer should conflate 'tokenize' variants");
    }

    #[test]
    fn test_optimize_does_not_error() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "some content", 0);
        optimize(&conn).unwrap();
    }

    #[test]
    fn test_scores_are_positive() {
        let conn = setup();
        insert_chunk(&conn, "d1", "c1", "machine learning algorithms", 0);
        let results = search(&conn, "machine learning", 10).unwrap();
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.score > 0.0, "Negated BM25 scores should be positive");
        }
    }
}
