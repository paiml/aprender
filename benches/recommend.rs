use aprender::recommend::ContentRecommender;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_movie_descriptions(n: usize) -> Vec<(String, String)> {
    let genres = [
        "action",
        "comedy",
        "drama",
        "thriller",
        "horror",
        "romance",
        "scifi",
        "fantasy",
        "mystery",
        "adventure",
    ];
    let adjectives = [
        "epic",
        "thrilling",
        "emotional",
        "intense",
        "hilarious",
        "dark",
        "heartwarming",
        "suspenseful",
        "mysterious",
        "explosive",
    ];
    let nouns = [
        "story",
        "journey",
        "adventure",
        "tale",
        "saga",
        "quest",
        "mission",
        "odyssey",
        "expedition",
        "voyage",
    ];

    (0..n)
        .map(|i| {
            let genre = genres[i % genres.len()];
            let adj = adjectives[(i / 10) % adjectives.len()];
            let noun = nouns[(i / 100) % nouns.len()];
            let id = format!("movie_{}", i);
            let desc = format!("{} {} {} about heroes and villains", adj, genre, noun);
            (id, desc)
        })
        .collect()
}

fn bench_add_items(c: &mut Criterion) {
    let mut group = c.benchmark_group("recommend_add");

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut rec = ContentRecommender::new(16, 200, 0.95);
                let items = generate_movie_descriptions(size);
                for (id, desc) in items {
                    rec.add_item(black_box(id), black_box(desc));
                }
                rec
            });
        });
    }

    group.finish();
}

fn bench_recommend(c: &mut Criterion) {
    let mut group = c.benchmark_group("recommend_search");
    group.sample_size(50); // Reduce samples for large datasets

    for size in [100, 1_000, 10_000].iter() {
        // Pre-build recommender
        let mut rec = ContentRecommender::new(16, 200, 0.95);
        let items = generate_movie_descriptions(*size);
        for (id, desc) in items {
            rec.add_item(id, desc);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Query for similar items
                rec.recommend(black_box("movie_0"), black_box(10))
                    .expect("should succeed")
            });
        });
    }

    group.finish();
}

fn bench_recommend_latency_target(c: &mut Criterion) {
    // Specific benchmark to verify <100ms latency for large dataset
    let mut rec = ContentRecommender::new(16, 200, 0.95);
    let items = generate_movie_descriptions(10_000);
    for (id, desc) in items {
        rec.add_item(id, desc);
    }

    c.bench_function("recommend_10k_latency", |b| {
        b.iter(|| {
            rec.recommend(black_box("movie_5000"), black_box(10))
                .expect("should succeed")
        });
    });
}

criterion_group!(
    benches,
    bench_add_items,
    bench_recommend,
    bench_recommend_latency_target
);
criterion_main!(benches);
