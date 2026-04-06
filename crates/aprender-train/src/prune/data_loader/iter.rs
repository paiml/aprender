//! Iterator over calibration batches.

use super::loader::CalibrationDataLoader;
use crate::train::Batch;

/// Iterator over calibration batches.
pub struct CalibrationDataIter<'a> {
    loader: &'a CalibrationDataLoader,
    position: usize,
}

impl<'a> CalibrationDataIter<'a> {
    /// Create a new iterator over calibration batches.
    pub(crate) fn new(loader: &'a CalibrationDataLoader) -> Self {
        Self { loader, position: 0 }
    }
}

impl<'a> Iterator for CalibrationDataIter<'a> {
    type Item = &'a Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.loader.get_batch(self.position)?;
        self.position += 1;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.loader.num_batches().saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CalibrationDataIter<'_> {}
