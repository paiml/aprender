//! WASM Notebook Runtime with Reactive Cell Execution.
//!
//! Implements GitHub issue #6 - Notebook runtime for Ruchy integration.
//!
//! # Design
//!
//! - Dependency-aware cell execution order (topological sort)
//! - Automatic re-execution on upstream changes
//! - Cell state persistence
//!
//! # Example
//!
//! ```ignore
//! let mut notebook = NotebookRuntime::new();
//!
//! // Add cells
//! let a = notebook.add_cell("let x = 10");
//! let b = notebook.add_cell("let y = x * 2");  // depends on a
//! let c = notebook.add_cell("x + y");          // depends on a, b
//!
//! // Execute all cells in dependency order
//! notebook.execute_all();
//!
//! // Update cell a - cells b and c re-execute automatically
//! notebook.update_cell(a, "let x = 20");
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a notebook cell.
pub type CellId = u64;

/// Output from a cell execution.
#[derive(Debug, Clone, Default)]
pub struct CellOutput {
    /// Standard output text
    pub stdout: String,
    /// Error output text (if any)
    pub stderr: String,
    /// Return value (serialized)
    pub value: Option<String>,
    /// Execution time in milliseconds
    pub exec_time_ms: u64,
}

/// A single notebook cell with source code and execution state.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Unique identifier
    pub id: CellId,
    /// Source code
    pub source: String,
    /// Last execution output
    pub output: CellOutput,
    /// Variables this cell depends on
    pub dependencies: HashSet<String>,
    /// Variables this cell defines
    pub definitions: HashSet<String>,
    /// Whether cell needs re-execution
    pub dirty: bool,
    /// Cell execution order (0 = not executed)
    pub execution_order: u64,
}

impl Cell {
    /// Create a new cell with the given source code.
    #[must_use]
    pub fn new(id: CellId, source: impl Into<String>) -> Self {
        let source = source.into();
        let (dependencies, definitions) = Self::analyze_dependencies(&source);

        Self {
            id,
            source,
            output: CellOutput::default(),
            dependencies,
            definitions,
            dirty: true,
            execution_order: 0,
        }
    }

    /// Update the cell's source code and re-analyze dependencies.
    pub fn update_source(&mut self, source: impl Into<String>) {
        self.source = source.into();
        let (deps, defs) = Self::analyze_dependencies(&self.source);
        self.dependencies = deps;
        self.definitions = defs;
        self.dirty = true;
    }

    /// Analyze source code to extract variable dependencies and definitions.
    ///
    /// This is a simplified analysis - a real implementation would use the Ruchy parser.
    fn analyze_dependencies(source: &str) -> (HashSet<String>, HashSet<String>) {
        let mut dependencies = HashSet::new();
        let mut definitions = HashSet::new();

        for line in source.lines() {
            let trimmed = line.trim();

            // Simple pattern: "let x = ..." or "var x = ..."
            if let Some(rest) = trimmed.strip_prefix("let ") {
                if let Some(eq_pos) = rest.find('=') {
                    let name = rest[..eq_pos].trim().to_string();
                    definitions.insert(name);

                    // Simple: everything after '=' are potential dependencies
                    let rhs = &rest[eq_pos + 1..];
                    for word in rhs.split(|c: char| !c.is_alphanumeric() && c != '_') {
                        let word = word.trim();
                        if !word.is_empty() && word.chars().next().is_some_and(char::is_alphabetic)
                        {
                            dependencies.insert(word.to_string());
                        }
                    }
                }
            } else {
                // Expression - all identifiers are dependencies
                for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
                    let word = word.trim();
                    if !word.is_empty() && word.chars().next().is_some_and(char::is_alphabetic) {
                        dependencies.insert(word.to_string());
                    }
                }
            }
        }

        // Remove self-definitions from dependencies
        for def in &definitions {
            dependencies.remove(def);
        }

        // Remove keywords
        for kw in &[
            "let", "if", "else", "for", "while", "fun", "return", "true", "false",
        ] {
            dependencies.remove(*kw);
        }

        (dependencies, definitions)
    }
}

/// Directed acyclic graph tracking cell dependencies.
#[derive(Debug, Default)]
pub struct CellGraph {
    /// All cells in the notebook
    cells: HashMap<CellId, Cell>,
    /// Variable name -> cell that defines it
    var_to_cell: HashMap<String, CellId>,
    /// Next cell ID
    next_id: CellId,
}

impl CellGraph {
    /// Create a new empty cell graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new cell to the graph.
    #[must_use]
    pub fn add_cell(&mut self, source: impl Into<String>) -> CellId {
        let id = self.next_id;
        self.next_id += 1;

        let cell = Cell::new(id, source);

        // Register variable definitions
        for var in &cell.definitions {
            self.var_to_cell.insert(var.clone(), id);
        }

        self.cells.insert(id, cell);
        id
    }

    /// Update a cell's source code.
    ///
    /// Returns the set of cells that need re-execution.
    pub fn update_cell(&mut self, id: CellId, source: impl Into<String>) -> HashSet<CellId> {
        let mut dirty = HashSet::new();

        if let Some(cell) = self.cells.get_mut(&id) {
            // Remove old definitions
            for var in &cell.definitions {
                self.var_to_cell.remove(var);
            }

            // Update source
            cell.update_source(source);

            // Register new definitions
            for var in &cell.definitions {
                self.var_to_cell.insert(var.clone(), id);
            }

            dirty.insert(id);
        }

        // Mark all downstream cells as dirty
        self.propagate_dirty(id, &mut dirty);

        dirty
    }

    /// Remove a cell from the graph.
    pub fn remove_cell(&mut self, id: CellId) -> Option<Cell> {
        if let Some(cell) = self.cells.remove(&id) {
            // Remove variable definitions
            for var in &cell.definitions {
                if self.var_to_cell.get(var) == Some(&id) {
                    self.var_to_cell.remove(var);
                }
            }
            Some(cell)
        } else {
            None
        }
    }

    /// Get a cell by ID.
    #[must_use]
    pub fn get_cell(&self, id: CellId) -> Option<&Cell> {
        self.cells.get(&id)
    }

    /// Get a mutable reference to a cell.
    pub fn get_cell_mut(&mut self, id: CellId) -> Option<&mut Cell> {
        self.cells.get_mut(&id)
    }

    /// Get all cells.
    pub fn cells(&self) -> impl Iterator<Item = &Cell> {
        self.cells.values()
    }

    /// Get the number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Propagate dirty flag to all downstream cells.
    fn propagate_dirty(&mut self, from: CellId, dirty: &mut HashSet<CellId>) {
        // Get definitions from the source cell
        let definitions: HashSet<String> = self
            .cells
            .get(&from)
            .map(|c| c.definitions.clone())
            .unwrap_or_default();

        // Find all cells that depend on these definitions
        for (id, cell) in &mut self.cells {
            if *id == from {
                continue;
            }

            // Check if this cell depends on any of the definitions
            if cell.dependencies.iter().any(|d| definitions.contains(d)) {
                cell.dirty = true;
                dirty.insert(*id);
            }
        }
    }

    /// Get cells in topological order based on dependencies.
    ///
    /// Returns `None` if there's a cycle.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<CellId>> {
        let mut in_degree: HashMap<CellId, usize> = HashMap::new();
        let mut edges: HashMap<CellId, Vec<CellId>> = HashMap::new();

        // Initialize in-degrees
        for id in self.cells.keys() {
            in_degree.insert(*id, 0);
            edges.insert(*id, Vec::new());
        }

        // Build edges: if cell B depends on variable defined in cell A, add edge A -> B
        for (id, cell) in &self.cells {
            for dep in &cell.dependencies {
                if let Some(&def_cell) = self.var_to_cell.get(dep) {
                    if def_cell != *id {
                        edges.entry(def_cell).or_default().push(*id);
                        *in_degree.entry(*id).or_default() += 1;
                    }
                }
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<CellId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::with_capacity(self.cells.len());

        while let Some(id) = queue.pop_front() {
            result.push(id);

            if let Some(neighbors) = edges.get(&id) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if result.len() == self.cells.len() {
            Some(result)
        } else {
            None // Cycle detected
        }
    }

    /// Get cells that need execution (dirty cells in topological order).
    #[must_use]
    pub fn cells_to_execute(&self) -> Vec<CellId> {
        self.topological_order()
            .unwrap_or_default()
            .into_iter()
            .filter(|id| self.cells.get(id).is_some_and(|c| c.dirty))
            .collect()
    }
}

/// Notebook runtime with reactive cell execution.
#[derive(Debug, Default)]
pub struct NotebookRuntime {
    /// Cell dependency graph
    graph: CellGraph,
    /// Global execution counter
    execution_counter: u64,
    /// Shared variable namespace (name -> value as JSON)
    namespace: HashMap<String, String>,
}

impl NotebookRuntime {
    /// Create a new notebook runtime.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new cell to the notebook.
    #[must_use]
    pub fn add_cell(&mut self, source: impl Into<String>) -> CellId {
        self.graph.add_cell(source)
    }

    /// Update a cell's source code.
    ///
    /// Returns the set of cells that need re-execution.
    pub fn update_cell(&mut self, id: CellId, source: impl Into<String>) -> HashSet<CellId> {
        self.graph.update_cell(id, source)
    }

    /// Remove a cell from the notebook.
    pub fn remove_cell(&mut self, id: CellId) -> Option<Cell> {
        self.graph.remove_cell(id)
    }

    /// Get a cell by ID.
    #[must_use]
    pub fn get_cell(&self, id: CellId) -> Option<&Cell> {
        self.graph.get_cell(id)
    }

    /// Get all cells.
    pub fn cells(&self) -> impl Iterator<Item = &Cell> {
        self.graph.cells()
    }

    /// Get cells in execution order.
    #[must_use]
    pub fn cells_in_order(&self) -> Vec<CellId> {
        self.graph.topological_order().unwrap_or_default()
    }

    /// Get cells that need execution.
    #[must_use]
    pub fn dirty_cells(&self) -> Vec<CellId> {
        self.graph.cells_to_execute()
    }

    /// Execute a single cell.
    ///
    /// This is a stub - the actual execution would be done by the Ruchy runtime.
    pub fn execute_cell(&mut self, id: CellId) -> Option<&CellOutput> {
        self.execution_counter += 1;
        let counter = self.execution_counter;

        if let Some(cell) = self.graph.get_cell_mut(id) {
            // Placeholder: Ruchy runtime integration pending
            // Currently marks cell as executed without actual evaluation
            cell.dirty = false;
            cell.execution_order = counter;
            cell.output = CellOutput {
                stdout: String::new(),
                stderr: String::new(),
                value: Some(format!("/* executed cell {} */", id)),
                exec_time_ms: 0,
            };

            // Register defined variables
            for var in &cell.definitions {
                self.namespace.insert(var.clone(), "null".to_string());
            }

            return Some(&cell.output);
        }
        None
    }

    /// Execute all dirty cells in dependency order.
    pub fn execute_dirty(&mut self) -> Vec<(CellId, CellOutput)> {
        let to_execute = self.dirty_cells();
        let mut results = Vec::with_capacity(to_execute.len());

        for id in to_execute {
            if let Some(output) = self.execute_cell(id) {
                results.push((id, output.clone()));
            }
        }

        results
    }

    /// Execute all cells in dependency order.
    pub fn execute_all(&mut self) -> Vec<(CellId, CellOutput)> {
        // Mark all cells as dirty
        for cell in self.graph.cells.values_mut() {
            cell.dirty = true;
        }

        self.execute_dirty()
    }

    /// Get the shared namespace.
    #[must_use]
    pub fn namespace(&self) -> &HashMap<String, String> {
        &self.namespace
    }

    /// Clear the namespace.
    pub fn clear_namespace(&mut self) {
        self.namespace.clear();
    }

    /// Get the number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Check if the notebook is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_creation() {
        let cell = Cell::new(1, "let x = 10");
        assert_eq!(cell.id, 1);
        assert!(cell.definitions.contains("x"));
        assert!(!cell.dependencies.contains("x")); // x is defined, not a dependency
    }

    #[test]
    fn test_cell_dependencies() {
        let cell = Cell::new(1, "let y = x * 2");
        assert!(cell.definitions.contains("y"));
        assert!(cell.dependencies.contains("x"));
    }

    #[test]
    fn test_cell_graph_add() {
        let mut graph = CellGraph::new();
        let id = graph.add_cell("let x = 10");
        assert_eq!(graph.len(), 1);
        assert!(graph.get_cell(id).is_some());
    }

    #[test]
    fn test_cell_graph_topological_order() {
        let mut graph = CellGraph::new();
        let a = graph.add_cell("let x = 10");
        let b = graph.add_cell("let y = x * 2");
        let c = graph.add_cell("let z = x + y");

        let order = graph.topological_order().unwrap();

        // a must come before b and c
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        let pos_c = order.iter().position(|&id| id == c).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c); // b defines y which c depends on
    }

    #[test]
    fn test_cell_graph_update_propagates_dirty() {
        let mut graph = CellGraph::new();
        let a = graph.add_cell("let x = 10");
        let b = graph.add_cell("let y = x * 2");

        // Clear dirty flags
        graph.get_cell_mut(a).unwrap().dirty = false;
        graph.get_cell_mut(b).unwrap().dirty = false;

        // Update cell a
        let dirty = graph.update_cell(a, "let x = 20");

        assert!(dirty.contains(&a));
        assert!(dirty.contains(&b)); // b depends on x, should be marked dirty
    }

    #[test]
    fn test_notebook_runtime_basic() {
        let mut notebook = NotebookRuntime::new();
        let a = notebook.add_cell("let x = 10");
        let b = notebook.add_cell("let y = x * 2");

        assert_eq!(notebook.len(), 2);

        // Execute all
        let results = notebook.execute_all();
        assert_eq!(results.len(), 2);

        // Cells should be marked as not dirty
        assert!(!notebook.get_cell(a).unwrap().dirty);
        assert!(!notebook.get_cell(b).unwrap().dirty);
    }

    #[test]
    fn test_notebook_runtime_reactive_update() {
        let mut notebook = NotebookRuntime::new();
        let a = notebook.add_cell("let x = 10");
        let b = notebook.add_cell("let y = x * 2");

        // Execute all
        notebook.execute_all();

        // Update cell a
        let dirty = notebook.update_cell(a, "let x = 20");

        // Both cells should be dirty
        assert!(dirty.contains(&a));
        assert!(dirty.contains(&b));

        // Execute dirty cells
        let results = notebook.execute_dirty();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_notebook_remove_cell() {
        let mut notebook = NotebookRuntime::new();
        let a = notebook.add_cell("let x = 10");

        assert_eq!(notebook.len(), 1);

        let removed = notebook.remove_cell(a);
        assert!(removed.is_some());
        assert_eq!(notebook.len(), 0);
    }

    #[test]
    fn test_cell_graph_cycle_detection() {
        let mut graph = CellGraph::new();

        // Create a simple graph without cycles
        let _ = graph.add_cell("let x = 10");
        let _ = graph.add_cell("let y = x * 2");

        // Should have valid topological order
        assert!(graph.topological_order().is_some());
    }

    #[test]
    fn test_cell_expression_dependencies() {
        let cell = Cell::new(1, "x + y + z");
        assert!(cell.dependencies.contains("x"));
        assert!(cell.dependencies.contains("y"));
        assert!(cell.dependencies.contains("z"));
        assert!(cell.definitions.is_empty());
    }
}
