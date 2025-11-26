//! Content-Based Recommendation Example
//!
//! This example demonstrates how to use the ContentRecommender for
//! finding similar items based on text descriptions.
//!
//! Run with: cargo run --example recommend_content

use aprender::recommend::ContentRecommender;

fn main() {
    println!("Content-Based Recommendation Example\n");
    println!("======================================\n");

    // Create recommender with:
    // - M=16 connections per node in HNSW graph
    // - ef_construction=200 for quality
    // - decay_factor=0.95 for IDF (half-life ~14 documents)
    let mut recommender = ContentRecommender::new(16, 200, 0.95);

    // Add movie descriptions
    println!("Adding movie descriptions...\n");

    let movies = vec![
        (
            "inception",
            "A thief who steals corporate secrets through dream-sharing technology",
        ),
        (
            "matrix",
            "A computer hacker learns about the true nature of reality and his role in the war against its controllers",
        ),
        (
            "interstellar",
            "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival",
        ),
        (
            "dark_knight",
            "Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into chaos",
        ),
        (
            "shawshank",
            "Two imprisoned men bond over years, finding redemption through acts of common decency",
        ),
        (
            "goodfellas",
            "The story of Henry Hill and his life in the mob, covering his relationship with his wife and partners",
        ),
        (
            "pulp_fiction",
            "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption",
        ),
        (
            "fight_club",
            "An insomniac office worker and a soap salesman form an underground fight club that evolves into much more",
        ),
        (
            "forrest_gump",
            "The presidencies of Kennedy and Johnson unfold through the perspective of an Alabama man with an IQ of 75",
        ),
        (
            "avatar",
            "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world",
        ),
    ];

    for (id, description) in &movies {
        recommender.add_item(*id, *description);
        println!("Added: {id} - {description}");
    }

    println!("\n{} movies added to recommender\n", recommender.len());
    println!("======================================\n");

    // Get recommendations for different movies
    let query_movies = vec!["inception", "shawshank", "avatar"];

    for query_id in query_movies {
        println!("Finding movies similar to '{query_id}':");

        match recommender.recommend(query_id, 3) {
            Ok(recommendations) => {
                for (rank, (rec_id, similarity)) in recommendations.iter().enumerate() {
                    println!("  {}. {} (similarity: {:.3})", rank + 1, rec_id, similarity);
                }
            }
            Err(e) => {
                println!("Error getting recommendations: {e}");
            }
        }

        println!();
    }

    // Demonstrate adding new items incrementally
    println!("======================================\n");
    println!("Adding a new sci-fi movie...\n");

    recommender.add_item(
        "blade_runner",
        "A blade runner must pursue and terminate four replicants who stole a ship in space and have returned to Earth",
    );

    println!("Now recommending similar movies to 'blade_runner':");

    match recommender.recommend("blade_runner", 3) {
        Ok(recommendations) => {
            for (rank, (rec_id, similarity)) in recommendations.iter().enumerate() {
                println!("  {}. {} (similarity: {:.3})", rank + 1, rec_id, similarity);
            }
        }
        Err(e) => {
            println!("Error: {e}");
        }
    }

    println!("\nTotal movies in recommender: {}", recommender.len());
}
