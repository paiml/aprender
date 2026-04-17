#![allow(clippy::disallowed_methods)]
//! Benchmarks for retrieval operations

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use trueno_rag::{
    chunk::{Chunk, Chunker, RecursiveChunker},
    embed::MockEmbedder,
    index::{BM25Index, SparseIndex, VectorStore},
    Document, DocumentId, Embedder,
};

fn create_test_chunk(content: &str, embedding: Vec<f32>) -> Chunk {
    let mut chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
    chunk.set_embedding(embedding);
    chunk
}

fn bench_bm25_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_indexing");

    let chunks: Vec<_> = (0..1000)
        .map(|i| {
            Chunk::new(
                DocumentId::new(),
                format!("Document {i} contains information about machine learning and artificial intelligence"),
                0,
                80,
            )
        })
        .collect();

    group.bench_function("index_1000_chunks", |b| {
        b.iter(|| {
            let mut index = BM25Index::new();
            for chunk in &chunks {
                index.add(black_box(chunk));
            }
            index
        });
    });

    group.finish();
}

fn bench_bm25_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_search");

    // Pre-build index
    let mut index = BM25Index::new();
    for i in 0..1000 {
        let chunk = Chunk::new(
            DocumentId::new(),
            format!("Document {i} about topic {} with keywords", i % 100),
            0,
            50,
        );
        index.add(&chunk);
    }

    group.bench_function("search_top_10", |b| {
        b.iter(|| index.search(black_box("topic keywords"), 10));
    });

    group.bench_function("search_top_100", |b| {
        b.iter(|| index.search(black_box("topic keywords"), 100));
    });

    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");

    // Pre-build store
    let mut store = VectorStore::with_dimension(128);
    for i in 0..1000 {
        let mut embedding = vec![0.0f32; 128];
        embedding[i % 128] = 1.0;
        let chunk = create_test_chunk(&format!("document {i}"), embedding);
        store.insert(chunk).unwrap();
    }

    let query = vec![1.0f32; 128];

    group.bench_function("search_top_10", |b| {
        b.iter(|| store.search(black_box(&query), 10));
    });

    group.bench_function("search_top_100", |b| {
        b.iter(|| store.search(black_box(&query), 100));
    });

    group.finish();
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    let long_doc = Document::new("Lorem ipsum dolor sit amet. ".repeat(1000));
    let chunker = RecursiveChunker::new(512, 50);

    group.bench_function("chunk_large_doc", |b| {
        b.iter(|| chunker.chunk(black_box(&long_doc)));
    });

    group.finish();
}

fn bench_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    let embedder = MockEmbedder::new(384);
    let texts: Vec<&str> = (0..100).map(|_| "This is a test sentence for embedding").collect();

    group.bench_function("embed_100_texts", |b| {
        b.iter(|| embedder.embed_batch(black_box(&texts)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bm25_indexing,
    bench_bm25_search,
    bench_vector_search,
    bench_chunking,
    bench_embedding,
);

criterion_main!(benches);
