//! Validation and tour-length checks for `TspInstanceYaml` (split out of tsp_instance.rs for the pre-commit complexity limit, #6025; code unchanged).

use super::{TspInstanceError, TspInstanceYaml};

impl TspInstanceYaml {
    /// Validate the instance (Jidoka).
    ///
    /// # Errors
    ///
    /// Returns error if validation fails.
    pub fn validate(&self) -> Result<(), TspInstanceError> {
        self.check_matrix_dimensions()?;
        self.check_row_lengths()?;
        self.check_city_ids()
    }

    fn check_matrix_dimensions(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        if self.matrix.len() != n {
            return Err(TspInstanceError::MatrixDimensionMismatch {
                expected: n,
                got_rows: self.matrix.len(),
            });
        }
        Ok(())
    }

    fn check_row_lengths(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for (i, row) in self.matrix.iter().enumerate() {
            if row.len() != n {
                return Err(TspInstanceError::MatrixRowMismatch {
                    row: i,
                    expected: n,
                    got: row.len(),
                });
            }
        }
        Ok(())
    }

    fn check_city_ids(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for city in &self.cities {
            if city.id >= n {
                return Err(TspInstanceError::InvalidCityId {
                    id: city.id,
                    max: n - 1,
                });
            }
        }
        Ok(())
    }

    /// Check if matrix is symmetric.
    ///
    /// # Errors
    ///
    /// Returns error with first asymmetry found.
    pub fn check_symmetry(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for i in 0..n {
            for j in (i + 1)..n {
                self.check_pair_symmetric(i, j)?;
            }
        }
        Ok(())
    }

    fn check_pair_symmetric(&self, i: usize, j: usize) -> Result<(), TspInstanceError> {
        let forward = self.matrix[i][j];
        let backward = self.matrix[j][i];
        if forward != backward {
            return Err(TspInstanceError::AsymmetricMatrix {
                i,
                j,
                forward,
                backward,
            });
        }
        Ok(())
    }

    /// Check triangle inequality for all city triples.
    ///
    /// # Errors
    ///
    /// Returns error with first violation found.
    pub fn check_triangle_inequality(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if self.violates_triangle(i, j, k) {
                        return Err(TspInstanceError::TriangleInequalityViolation { i, j, k });
                    }
                }
            }
        }
        Ok(())
    }

    fn violates_triangle(&self, i: usize, j: usize, k: usize) -> bool {
        if i == j || j == k || i == k {
            return false;
        }
        let direct = self.matrix[i][k];
        let via_j = self.matrix[i][j].saturating_add(self.matrix[j][k]);
        direct > via_j
    }

    /// Compute tour length for given city sequence.
    ///
    /// # Arguments
    ///
    /// * `tour` - Sequence of city indices (must visit each city once)
    ///
    /// # Returns
    ///
    /// Total tour length including return to start.
    #[must_use]
    pub fn tour_length(&self, tour: &[usize]) -> u32 {
        if tour.is_empty() {
            return 0;
        }
        let mut total = 0u32;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total = total.saturating_add(self.matrix[from][to]);
        }
        total
    }

    /// Compute tour length with step-by-step verification (Genchi Genbutsu).
    ///
    /// # Returns
    ///
    /// Tuple of (`total_length`, `step_descriptions`).
    #[must_use]
    pub fn tour_length_verified(&self, tour: &[usize]) -> (u32, Vec<String>) {
        let mut total = 0u32;
        let mut steps = Vec::new();

        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            let dist = self.matrix[from][to];
            total = total.saturating_add(dist);

            let from_name = self.cities.get(from).map_or("?", |c| &c.name);
            let to_name = self.cities.get(to).map_or("?", |c| &c.name);
            steps.push(format!(
                "Step {}: {} → {} = {} {} (total: {})",
                i + 1,
                from_name,
                to_name,
                dist,
                self.meta.units,
                total
            ));
        }

        (total, steps)
    }
}
