# Case Study: Content-Based Recommendations

Build a recommendation engine using text similarity and HNSW indexing.

## Use Case

Find similar movies based on plot descriptions.

## Implementation

```rust,ignore
use aprender::recommend::ContentRecommender;

fn main() {
    // Create recommender with HNSW parameters:
    // - M=16: connections per node
    // - ef_construction=200: build quality
    // - decay_factor=0.95: IDF decay
    let mut recommender = ContentRecommender::new(16, 200, 0.95);

    // Add movie descriptions
    let movies = vec![
        ("inception", "A thief steals secrets through dream-sharing technology"),
        ("matrix", "A hacker discovers reality is a simulation"),
        ("interstellar", "Astronauts travel through a wormhole to save humanity"),
        ("avatar", "A marine explores an alien world called Pandora"),
        ("terminator", "A cyborg assassin is sent back in time"),
        ("blade_runner", "A detective hunts rogue replicants in dystopian future"),
    ];

    for (id, description) in &movies {
        recommender.add_item(id, description);
    }

    // Build the index
    recommender.build_index();

    // Find similar movies
    let query = "science fiction about artificial intelligence and reality";
    let recommendations = recommender.recommend(query, 3);

    println!("Query: {}\n", query);
    println!("Recommendations:");
    for (id, score) in recommendations {
        println!("  {} (score: {:.3})", id, score);
    }
}
```

Output:
```text
Query: science fiction about artificial intelligence and reality

Recommendations:
  matrix (score: 0.847)
  blade_runner (score: 0.723)
  terminator (score: 0.691)
```

## How It Works

1. **TF-IDF Vectorization**: Convert descriptions to sparse vectors
2. **Incremental IDF**: Update vocabulary as items are added
3. **HNSW Index**: Fast approximate nearest neighbor search
4. **Cosine Similarity**: Rank by vector similarity

## Key Features

- **Incremental updates**: Add items without rebuilding
- **Scalable**: HNSW provides O(log n) search
- **No training required**: Pure content-based filtering

## Run

```bash
cargo run --example recommend_content
```
