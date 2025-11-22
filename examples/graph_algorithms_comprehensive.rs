//! Comprehensive Graph Algorithms Demo
//!
//! This example demonstrates all graph algorithms from v0.5.1:
//!
//! **Phase 1: Pathfinding**
//! - shortest_path: BFS-based unweighted shortest path
//! - dijkstra: Weighted shortest path with priority queue
//! - a_star: Heuristic-guided pathfinding
//! - all_pairs_shortest_paths: Distance matrix computation
//!
//! **Phase 2: Components & Traversal**
//! - dfs: Depth-first search exploration
//! - connected_components: Find groups in undirected graphs
//! - strongly_connected_components: Find cycles in directed graphs
//! - topological_sort: Order DAG nodes by dependencies
//!
//! **Phase 3: Community & Link Analysis**
//! - label_propagation: Iterative community detection
//! - common_neighbors: Link prediction metric
//! - adamic_adar_index: Weighted link prediction
//!
//! Run with: `cargo run --example graph_algorithms_comprehensive`

use aprender::graph::Graph;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Comprehensive Graph Algorithms Demo - Aprender v0.5.1");
    println!("═══════════════════════════════════════════════════════════════\n");

    demo_pathfinding();
    demo_components_traversal();
    demo_community_link_analysis();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo Complete! All 11 algorithms demonstrated.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn demo_pathfinding() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║ PHASE 1: PATHFINDING ALGORITHMS                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Build a weighted graph representing a road network
    // Nodes: Cities (0=A, 1=B, 2=C, 3=D, 4=E, 5=F)
    // Edges: Roads with distances in km
    println!("📍 Building road network graph:");
    println!("   Cities: A(0), B(1), C(2), D(3), E(4), F(5)");
    println!("   Roads with distances (km):\n");

    let weighted_edges = vec![
        (0, 1, 4.0),  // A-B: 4km
        (0, 2, 2.0),  // A-C: 2km
        (1, 2, 1.0),  // B-C: 1km
        (1, 3, 5.0),  // B-D: 5km
        (2, 3, 8.0),  // C-D: 8km
        (2, 4, 10.0), // C-E: 10km
        (3, 4, 2.0),  // D-E: 2km
        (3, 5, 6.0),  // D-F: 6km
        (4, 5, 3.0),  // E-F: 3km
    ];

    let cities = vec!["A", "B", "C", "D", "E", "F"];
    let g_weighted = Graph::from_weighted_edges(&weighted_edges, false);

    // For unweighted algorithms, create an unweighted version
    let unweighted_edges: Vec<(usize, usize)> =
        weighted_edges.iter().map(|(u, v, _)| (*u, *v)).collect();
    let g_unweighted = Graph::from_edges(&unweighted_edges, false);

    // 1. Shortest Path (BFS) - unweighted
    println!("─────────────────────────────────────────────────────────────────");
    println!("1️⃣  Shortest Path (BFS - unweighted hops)");
    println!("─────────────────────────────────────────────────────────────────");

    let path = g_unweighted
        .shortest_path(0, 5)
        .expect("Path should exist");

    print!("   Route from {} to {}: ", cities[0], cities[5]);
    for (i, &node) in path.iter().enumerate() {
        if i > 0 {
            print!(" → ");
        }
        print!("{}", cities[node]);
    }
    println!(" ({} hops)\n", path.len() - 1);

    // 2. Dijkstra's Algorithm - weighted shortest path
    println!("─────────────────────────────────────────────────────────────────");
    println!("2️⃣  Dijkstra's Algorithm (weighted shortest path)");
    println!("─────────────────────────────────────────────────────────────────");

    let (dijkstra_path, distance) = g_weighted.dijkstra(0, 5).expect("Path should exist");

    print!("   Shortest route from {} to {}: ", cities[0], cities[5]);
    for (i, &node) in dijkstra_path.iter().enumerate() {
        if i > 0 {
            print!(" → ");
        }
        print!("{}", cities[node]);
    }
    println!(" ({:.1} km)\n", distance);

    // 3. A* Search - heuristic-guided pathfinding
    println!("─────────────────────────────────────────────────────────────────");
    println!("3️⃣  A* Search (heuristic-guided pathfinding)");
    println!("─────────────────────────────────────────────────────────────────");

    // Simple heuristic: estimated remaining distance (straight-line distance approximation)
    let heuristic = |node: usize| match node {
        0 => 10.0, // A to F: ~10km estimate
        1 => 8.0,  // B to F: ~8km
        2 => 9.0,  // C to F: ~9km
        3 => 5.0,  // D to F: ~5km
        4 => 3.0,  // E to F: ~3km
        5 => 0.0,  // F to F: 0km
        _ => 0.0,
    };

    let astar_path = g_weighted
        .a_star(0, 5, heuristic)
        .expect("Path should exist");

    print!("   A* route from {} to {}: ", cities[0], cities[5]);
    for (i, &node) in astar_path.iter().enumerate() {
        if i > 0 {
            print!(" → ");
        }
        print!("{}", cities[node]);
    }
    println!(" (heuristic-guided)\n");

    // 4. All-Pairs Shortest Paths
    println!("─────────────────────────────────────────────────────────────────");
    println!("4️⃣  All-Pairs Shortest Paths (distance matrix)");
    println!("─────────────────────────────────────────────────────────────────");

    let dist_matrix = g_unweighted.all_pairs_shortest_paths();

    println!("   Distance matrix (hops):");
    print!("      ");
    for city in &cities {
        print!("{:>3} ", city);
    }
    println!();

    for (i, row) in dist_matrix.iter().enumerate() {
        print!("   {:>2} ", cities[i]);
        for dist in row {
            match dist {
                Some(d) => print!("{:>3} ", d),
                None => print!("  - "),
            }
        }
        println!();
    }
    println!();
}

fn demo_components_traversal() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ PHASE 2: COMPONENTS & TRAVERSAL ALGORITHMS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Depth-First Search
    println!("─────────────────────────────────────────────────────────────────");
    println!("1️⃣  Depth-First Search (DFS)");
    println!("─────────────────────────────────────────────────────────────────");

    let tree_edges = vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)];
    let tree = Graph::from_edges(&tree_edges, false);

    println!("   Tree structure:");
    println!("        0");
    println!("       / \\");
    println!("      1   2");
    println!("     / \\   \\");
    println!("    3   4   5\n");

    let dfs_order = tree.dfs(0).expect("DFS from root");
    print!("   DFS traversal from root: ");
    for (i, &node) in dfs_order.iter().enumerate() {
        if i > 0 {
            print!(" → ");
        }
        print!("{}", node);
    }
    println!("\n");

    // 2. Connected Components
    println!("─────────────────────────────────────────────────────────────────");
    println!("2️⃣  Connected Components (undirected graphs)");
    println!("─────────────────────────────────────────────────────────────────");

    // Three separate components
    let component_edges = vec![
        (0, 1),
        (1, 2), // Component 1: {0,1,2}
        (3, 4), // Component 2: {3,4}
                // Node 5 is isolated (Component 3)
    ];
    let g_components = Graph::from_edges(&component_edges, false);

    let components = g_components.connected_components();
    use std::collections::HashMap;

    let mut comp_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &comp_id) in components.iter().enumerate().take(6) {
        comp_groups.entry(comp_id).or_default().push(node);
    }

    println!("   Graph with {} nodes, {} edges", 6, component_edges.len());
    println!("   Found {} connected components:", comp_groups.len());
    for (i, nodes) in comp_groups.values().enumerate() {
        println!("   Component {}: {:?}", i + 1, nodes);
    }
    println!();

    // 3. Strongly Connected Components (SCCs)
    println!("─────────────────────────────────────────────────────────────────");
    println!("3️⃣  Strongly Connected Components (directed graphs)");
    println!("─────────────────────────────────────────────────────────────────");

    // Directed graph with two SCCs
    let scc_edges = vec![
        (0, 1),
        (1, 2),
        (2, 0), // SCC 1: {0,1,2} (cycle)
        (2, 3),
        (3, 4),
        (4, 3), // SCC 2: {3,4} (cycle)
    ];
    let g_directed = Graph::from_edges(&scc_edges, true);

    println!("   Directed graph:");
    println!("   0 → 1 → 2 → 0  (cycle 1)");
    println!("           ↓");
    println!("           3 ⇄ 4  (cycle 2)\n");

    let sccs = g_directed.strongly_connected_components();
    let mut scc_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &scc_id) in sccs.iter().enumerate().take(5) {
        scc_groups.entry(scc_id).or_default().push(node);
    }

    println!("   Found {} strongly connected components:", scc_groups.len());
    for (i, nodes) in scc_groups.values().enumerate() {
        println!("   SCC {}: {:?}", i + 1, nodes);
    }
    println!();

    // 4. Topological Sort
    println!("─────────────────────────────────────────────────────────────────");
    println!("4️⃣  Topological Sort (DAG ordering)");
    println!("─────────────────────────────────────────────────────────────────");

    // DAG representing task dependencies
    let dag_edges = vec![
        (0, 1), // Task 0 → Task 1
        (0, 2), // Task 0 → Task 2
        (1, 3), // Task 1 → Task 3
        (2, 3), // Task 2 → Task 3
        (3, 4), // Task 3 → Task 4
    ];
    let dag = Graph::from_edges(&dag_edges, true);

    let tasks = vec![
        "Setup Environment",
        "Install Dependencies",
        "Configure System",
        "Build Project",
        "Run Tests",
    ];

    println!("   Task dependency graph (DAG):");
    for (u, v) in &dag_edges {
        println!("   {} → {}", tasks[*u], tasks[*v]);
    }
    println!();

    match dag.topological_sort() {
        Some(order) => {
            println!("   Valid execution order:");
            for (i, &task_id) in order.iter().enumerate() {
                println!("   {}. {}", i + 1, tasks[task_id]);
            }
        }
        None => {
            println!("   ❌ Cycle detected! No valid ordering.");
        }
    }
    println!();
}

fn demo_community_link_analysis() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ PHASE 3: COMMUNITY & LINK ANALYSIS                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Build a social network with two communities
    let social_edges = vec![
        // Community 1: {0,1,2,3}
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 2), // Internal shortcuts
        // Bridge
        (3, 4),
        // Community 2: {4,5,6,7}
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (4, 6), // Internal shortcuts
    ];
    let g_social = Graph::from_edges(&social_edges, false);

    // 1. Label Propagation
    println!("─────────────────────────────────────────────────────────────────");
    println!("1️⃣  Label Propagation (community detection)");
    println!("─────────────────────────────────────────────────────────────────");

    println!("   Social network with 2 communities connected by a bridge:");
    println!("   Community A: {{0,1,2,3}}  ←bridge(3-4)→  Community B: {{4,5,6,7}}\n");

    let communities = g_social.label_propagation(10, Some(42));

    let mut comm_groups: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (node, &comm_id) in communities.iter().enumerate().take(8) {
        comm_groups.entry(comm_id).or_default().push(node);
    }

    println!("   Detected {} communities:", comm_groups.len());
    for (i, nodes) in comm_groups.values().enumerate() {
        println!("   Community {}: {:?}", i + 1, nodes);
    }
    println!();

    // 2. Common Neighbors
    println!("─────────────────────────────────────────────────────────────────");
    println!("2️⃣  Common Neighbors (link prediction)");
    println!("─────────────────────────────────────────────────────────────────");

    println!("   Link prediction: Will nodes 1 and 3 become friends?\n");

    let cn_1_3 = g_social.common_neighbors(1, 3).expect("Nodes exist");
    println!("   Common neighbors of 1 and 3: {}", cn_1_3);

    // Check who those common neighbors are
    let neighbors_1: Vec<usize> = g_social.neighbors(1).to_vec();
    let neighbors_3: Vec<usize> = g_social.neighbors(3).to_vec();

    println!("   Node 1 neighbors: {:?}", neighbors_1);
    println!("   Node 3 neighbors: {:?}", neighbors_3);

    let common: Vec<usize> = neighbors_1
        .iter()
        .filter(|n| neighbors_3.contains(n))
        .copied()
        .collect();
    println!("   Actual common neighbors: {:?}", common);
    println!("   → High common neighbor count suggests likely future connection\n");

    // 3. Adamic-Adar Index
    println!("─────────────────────────────────────────────────────────────────");
    println!("3️⃣  Adamic-Adar Index (weighted link prediction)");
    println!("─────────────────────────────────────────────────────────────────");

    let aa_1_3 = g_social.adamic_adar_index(1, 3).expect("Nodes exist");
    println!("   Adamic-Adar score for nodes 1 and 3: {:.4}", aa_1_3);

    // Compare with another pair
    let aa_0_7 = g_social.adamic_adar_index(0, 7).expect("Nodes exist");
    println!("   Adamic-Adar score for nodes 0 and 7: {:.4}", aa_0_7);

    println!("\n   💡 Interpretation:");
    println!("   - Higher score = stronger prediction for future link");
    println!("   - Nodes 1-3 (same community): {:.4}", aa_1_3);
    println!("   - Nodes 0-7 (different communities): {:.4}", aa_0_7);
    println!("   → Algorithm correctly identifies within-community links as more likely\n");
}
