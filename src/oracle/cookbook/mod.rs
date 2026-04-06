//! Cookbook - Practical recipes for common Sovereign AI Stack patterns
//!
//! Each recipe includes:
//! - Problem description
//! - Components involved
//! - Code example
//! - Related recipes

mod recipes;
mod recipes_more;
mod recipes_rlhf_alignment;
mod recipes_rlhf_efficiency;
mod recipes_rlhf_training;

use serde::{Deserialize, Serialize};

/// A cookbook recipe for a common pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    /// Unique recipe ID
    pub id: String,
    /// Recipe title
    pub title: String,
    /// Problem this recipe solves
    pub problem: String,
    /// Components used
    pub components: Vec<String>,
    /// Tags for discovery
    pub tags: Vec<String>,
    /// Complete code example
    pub code: String,
    /// TDD test companion for the code example
    pub test_code: String,
    /// Related recipe IDs
    pub related: Vec<String>,
}

impl Recipe {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            problem: String::new(),
            components: Vec::new(),
            tags: Vec::new(),
            code: String::new(),
            test_code: String::new(),
            related: Vec::new(),
        }
    }

    pub fn with_problem(mut self, problem: impl Into<String>) -> Self {
        self.problem = problem.into();
        self
    }

    pub fn with_components(mut self, components: Vec<&str>) -> Self {
        self.components = components.into_iter().map(String::from).collect();
        self
    }

    pub fn with_tags(mut self, tags: Vec<&str>) -> Self {
        self.tags = tags.into_iter().map(String::from).collect();
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = code.into();
        self
    }

    pub fn with_test_code(mut self, test_code: impl Into<String>) -> Self {
        self.test_code = test_code.into();
        self
    }

    pub fn with_related(mut self, related: Vec<&str>) -> Self {
        self.related = related.into_iter().map(String::from).collect();
        self
    }
}

/// Cookbook containing all recipes
#[derive(Debug, Clone, Default)]
pub struct Cookbook {
    pub(crate) recipes: Vec<Recipe>,
}

impl Cookbook {
    /// Create a new empty cookbook
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cookbook with all standard recipes
    pub fn standard() -> Self {
        let mut cookbook = Self::new();
        recipes::register_all(&mut cookbook);
        cookbook
    }

    /// Get all recipes
    pub fn recipes(&self) -> &[Recipe] {
        &self.recipes
    }

    /// Find recipes by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Recipe> {
        self.recipes.iter().filter(|r| r.tags.iter().any(|t| t.eq_ignore_ascii_case(tag))).collect()
    }

    /// Find recipes by component
    pub fn find_by_component(&self, component: &str) -> Vec<&Recipe> {
        self.recipes
            .iter()
            .filter(|r| r.components.iter().any(|c| c.eq_ignore_ascii_case(component)))
            .collect()
    }

    /// Get recipe by ID
    pub fn get(&self, id: &str) -> Option<&Recipe> {
        self.recipes.iter().find(|r| r.id == id)
    }

    /// Search recipes by keyword
    pub fn search(&self, query: &str) -> Vec<&Recipe> {
        let query_lower = query.to_lowercase();
        self.recipes
            .iter()
            .filter(|r| {
                r.title.to_lowercase().contains(&query_lower)
                    || r.problem.to_lowercase().contains(&query_lower)
                    || r.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Add a recipe to the cookbook (used by recipes module)
    pub(crate) fn add(&mut self, recipe: Recipe) {
        self.recipes.push(recipe);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cookbook_standard() {
        let cookbook = Cookbook::standard();
        assert!(!cookbook.recipes().is_empty());
    }

    #[test]
    fn test_find_by_tag() {
        let cookbook = Cookbook::standard();
        let wasm_recipes = cookbook.find_by_tag("wasm");
        assert!(!wasm_recipes.is_empty());
    }

    #[test]
    fn test_find_by_component() {
        let cookbook = Cookbook::standard();
        let simular_recipes = cookbook.find_by_component("simular");
        assert!(!simular_recipes.is_empty());
    }

    #[test]
    fn test_get_by_id() {
        let cookbook = Cookbook::standard();
        let recipe = cookbook.get("wasm-zero-js");
        assert!(recipe.is_some());
        assert_eq!(recipe.expect("unexpected failure").title, "Zero-JS WASM Application");
    }

    #[test]
    fn test_search() {
        let cookbook = Cookbook::standard();
        let results = cookbook.search("random forest");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_all_recipes_have_code() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(!recipe.code.is_empty(), "Recipe '{}' has empty code field", recipe.id);
        }
    }

    #[test]
    fn test_recipe_code_contains_rust() {
        let cookbook = Cookbook::standard();
        let recipe = cookbook.get("ml-random-forest").expect("ml-random-forest must exist");
        assert!(recipe.code.contains("use "), "ml-random-forest code should contain 'use ' import");
        assert!(
            recipe.code.contains("aprender"),
            "ml-random-forest code should reference aprender"
        );
    }

    #[test]
    fn test_all_recipes_have_test_code() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                !recipe.test_code.is_empty(),
                "Recipe '{}' has empty test_code field",
                recipe.id
            );
        }
    }

    #[test]
    fn test_test_code_has_cfg_test() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                recipe.test_code.contains("#[cfg("),
                "Recipe '{}' test_code should contain #[cfg(test)] or #[cfg(all(test, ...))]",
                recipe.id
            );
        }
    }

    #[test]
    fn test_test_code_has_test_attr() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                recipe.test_code.contains("#[test]"),
                "Recipe '{}' test_code should contain #[test] attribute",
                recipe.id
            );
        }
    }
}
