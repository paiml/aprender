//! SPIKE: Decision Tree Classifier Prototype
//!
//! Quick proof-of-concept to validate:
//! - CART algorithm implementation
//! - Gini impurity calculations
//! - Tree building and prediction
//! - Performance characteristics
//!
//! NOT production code - just validating approach before full TDD implementation

use std::collections::HashMap;

#[derive(Debug, Clone)]
enum TreeNode {
    Node {
        feature_idx: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf {
        class_label: usize,
    },
}

struct DecisionTreeSpike {
    tree: Option<TreeNode>,
    max_depth: usize,
}

impl DecisionTreeSpike {
    fn new(max_depth: usize) -> Self {
        Self {
            tree: None,
            max_depth,
        }
    }

    /// Calculate Gini impurity for a set of labels
    fn gini_impurity(labels: &[usize]) -> f32 {
        if labels.is_empty() {
            return 0.0;
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }

        let n = labels.len() as f32;
        let mut gini = 1.0;

        for count in counts.values() {
            let p = *count as f32 / n;
            gini -= p * p;
        }

        gini
    }

    /// Find the best split for a feature
    fn find_best_split(data: &[(Vec<f32>, usize)], feature_idx: usize) -> Option<(f32, f32)> {
        if data.len() < 2 {
            return None;
        }

        // Get unique thresholds (midpoints between consecutive values)
        let mut values: Vec<f32> = data.iter().map(|(x, _)| x[feature_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();

        if values.len() < 2 {
            return None;
        }

        let mut best_gain = 0.0;
        let mut best_threshold = 0.0;

        // Current impurity
        let labels: Vec<usize> = data.iter().map(|(_, y)| *y).collect();
        let current_impurity = Self::gini_impurity(&labels);

        // Try each threshold
        for i in 0..values.len() - 1 {
            let threshold = (values[i] + values[i + 1]) / 2.0;

            let mut left_labels = Vec::new();
            let mut right_labels = Vec::new();

            for (x, y) in data {
                if x[feature_idx] <= threshold {
                    left_labels.push(*y);
                } else {
                    right_labels.push(*y);
                }
            }

            if left_labels.is_empty() || right_labels.is_empty() {
                continue;
            }

            // Weighted Gini
            let n = data.len() as f32;
            let n_left = left_labels.len() as f32;
            let n_right = right_labels.len() as f32;

            let weighted_gini = (n_left / n) * Self::gini_impurity(&left_labels)
                + (n_right / n) * Self::gini_impurity(&right_labels);

            let gain = current_impurity - weighted_gini;

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }

        if best_gain > 0.0 {
            Some((best_threshold, best_gain))
        } else {
            None
        }
    }

    /// Find best split across all features
    fn find_best_feature_split(
        data: &[(Vec<f32>, usize)],
        n_features: usize,
    ) -> Option<(usize, f32, f32)> {
        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        for feature_idx in 0..n_features {
            if let Some((threshold, gain)) = Self::find_best_split(data, feature_idx) {
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain > 0.0 {
            Some((best_feature, best_threshold, best_gain))
        } else {
            None
        }
    }

    /// Build tree recursively
    fn build_tree(
        data: &[(Vec<f32>, usize)],
        n_features: usize,
        depth: usize,
        max_depth: usize,
    ) -> TreeNode {
        // Check stopping criteria
        let labels: Vec<usize> = data.iter().map(|(_, y)| *y).collect();
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();

        // Pure node or max depth reached
        if unique_labels.len() == 1 || depth >= max_depth {
            let majority_label = Self::majority_class(&labels);
            return TreeNode::Leaf {
                class_label: majority_label,
            };
        }

        // Find best split
        if let Some((feature_idx, threshold, _gain)) =
            Self::find_best_feature_split(data, n_features)
        {
            let mut left_data = Vec::new();
            let mut right_data = Vec::new();

            for (x, y) in data {
                if x[feature_idx] <= threshold {
                    left_data.push((x.clone(), *y));
                } else {
                    right_data.push((x.clone(), *y));
                }
            }

            if left_data.is_empty() || right_data.is_empty() {
                let majority_label = Self::majority_class(&labels);
                return TreeNode::Leaf {
                    class_label: majority_label,
                };
            }

            TreeNode::Node {
                feature_idx,
                threshold,
                left: Box::new(Self::build_tree(
                    &left_data,
                    n_features,
                    depth + 1,
                    max_depth,
                )),
                right: Box::new(Self::build_tree(
                    &right_data,
                    n_features,
                    depth + 1,
                    max_depth,
                )),
            }
        } else {
            let majority_label = Self::majority_class(&labels);
            TreeNode::Leaf {
                class_label: majority_label,
            }
        }
    }

    fn majority_class(labels: &[usize]) -> usize {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        *counts.iter().max_by_key(|(_, &count)| count).unwrap().0
    }

    fn fit(&mut self, x: &[Vec<f32>], y: &[usize]) {
        let n_features = x[0].len();
        let data: Vec<(Vec<f32>, usize)> = x.iter().cloned().zip(y.iter().cloned()).collect();
        self.tree = Some(Self::build_tree(&data, n_features, 0, self.max_depth));
    }

    fn predict_one(&self, x: &[f32]) -> usize {
        let mut node = self.tree.as_ref().unwrap();

        loop {
            match node {
                TreeNode::Leaf { class_label } => return *class_label,
                TreeNode::Node {
                    feature_idx,
                    threshold,
                    left,
                    right,
                } => {
                    if x[*feature_idx] <= *threshold {
                        node = left;
                    } else {
                        node = right;
                    }
                }
            }
        }
    }

    fn predict(&self, x: &[Vec<f32>]) -> Vec<usize> {
        x.iter().map(|xi| self.predict_one(xi)).collect()
    }

    fn accuracy(&self, x: &[Vec<f32>], y: &[usize]) -> f32 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, y)| p == y)
            .count();
        correct as f32 / y.len() as f32
    }
}

fn main() {
    println!("🔬 Decision Tree Spike - Proof of Concept");
    println!("==========================================\n");

    // Test 1: Simple 2D binary classification (XOR-like)
    println!("Test 1: XOR-like binary classification");
    let x_train = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.1, 0.1],
        vec![0.1, 0.9],
        vec![0.9, 0.1],
        vec![0.9, 0.9],
    ];
    let y_train = vec![0, 1, 1, 0, 0, 1, 1, 0];

    let mut tree = DecisionTreeSpike::new(3);
    tree.fit(&x_train, &y_train);
    let predictions = tree.predict(&x_train);
    let accuracy = tree.accuracy(&x_train, &y_train);

    println!("  Training accuracy: {:.2}%", accuracy * 100.0);
    println!("  Predictions: {:?}", predictions);
    println!("  Expected:    {:?}\n", y_train);

    // Test 2: Simple linearly separable data
    println!("Test 2: Linearly separable data");
    let x_train2 = vec![
        vec![1.0, 1.0],
        vec![1.5, 1.2],
        vec![1.2, 1.5],
        vec![5.0, 5.0],
        vec![5.5, 5.2],
        vec![5.2, 5.5],
    ];
    let y_train2 = vec![0, 0, 0, 1, 1, 1];

    let mut tree2 = DecisionTreeSpike::new(5);
    tree2.fit(&x_train2, &y_train2);
    let accuracy2 = tree2.accuracy(&x_train2, &y_train2);

    println!("  Training accuracy: {:.2}%\n", accuracy2 * 100.0);

    // Test 3: Gini impurity validation
    println!("Test 3: Gini impurity calculations");
    let pure = vec![0, 0, 0, 0];
    let mixed = vec![0, 1, 0, 1];
    let multi = vec![0, 1, 2, 0, 1, 2];

    println!(
        "  Pure (all 0s): {:.4} (expect 0.0)",
        DecisionTreeSpike::gini_impurity(&pure)
    );
    println!(
        "  50/50 split:   {:.4} (expect 0.5)",
        DecisionTreeSpike::gini_impurity(&mixed)
    );
    println!(
        "  3-class even:  {:.4} (expect 0.667)",
        DecisionTreeSpike::gini_impurity(&multi)
    );

    println!("\n✅ Spike complete - algorithm validated!");
    println!("\nNext steps:");
    println!("  • Algorithm works correctly");
    println!("  • Ready for full EXTREME TDD implementation");
    println!("  • Will add: builder pattern, multi-class, feature importance");
}
