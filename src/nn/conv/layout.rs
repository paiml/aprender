//! Convolution layout types for layout-aware dispatch.
//!
//! Defines data layouts (NCHW, NHWC, etc.) for convolution inputs/outputs
//! and kernel layouts (OIHW, HWIO, etc.) for convolution weights.

/// Dimension indices for a given layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionIndices {
    /// Index of batch dimension.
    pub batch: usize,
    /// Index of channel dimension.
    pub channel: usize,
    /// Indices of spatial dimensions (length 1 for 1D, length 2 for 2D).
    spatial: [usize; 2],
    /// Number of spatial dimensions (1 or 2).
    spatial_len: usize,
}

impl DimensionIndices {
    /// Get spatial dimension indices as a slice.
    #[must_use]
    pub fn spatial(&self) -> &[usize] {
        &self.spatial[..self.spatial_len]
    }
}

/// Data layout for convolution inputs and outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvLayout {
    /// Batch, Channels, Height, Width (PyTorch default for 2D).
    NCHW,
    /// Batch, Height, Width, Channels (TensorFlow default for 2D).
    NHWC,
    /// Batch, Channels, Length (PyTorch default for 1D).
    NCL,
    /// Batch, Length, Channels (TensorFlow default for 1D).
    NLC,
}

impl ConvLayout {
    /// Get dimension indices for this layout.
    #[must_use]
    pub fn indices(self) -> DimensionIndices {
        match self {
            Self::NCHW => DimensionIndices {
                batch: 0,
                channel: 1,
                spatial: [2, 3],
                spatial_len: 2,
            },
            Self::NHWC => DimensionIndices {
                batch: 0,
                channel: 3,
                spatial: [1, 2],
                spatial_len: 2,
            },
            Self::NCL => DimensionIndices {
                batch: 0,
                channel: 1,
                spatial: [2, 0],
                spatial_len: 1,
            },
            Self::NLC => DimensionIndices {
                batch: 0,
                channel: 2,
                spatial: [1, 0],
                spatial_len: 1,
            },
        }
    }

    /// Parse a shape according to this layout.
    ///
    /// Returns `(batch, channels, spatial_dims)`.
    #[must_use]
    pub fn parse_shape(self, shape: &[usize]) -> (usize, usize, Vec<usize>) {
        let idx = self.indices();
        let batch = shape[idx.batch];
        let channels = shape[idx.channel];
        let spatial: Vec<usize> = idx.spatial().iter().map(|&i| shape[i]).collect();
        (batch, channels, spatial)
    }

    /// Build a shape from components according to this layout.
    #[must_use]
    pub fn build_shape(self, batch: usize, channels: usize, spatial: &[usize]) -> Vec<usize> {
        match self {
            Self::NCHW => vec![batch, channels, spatial[0], spatial[1]],
            Self::NHWC => vec![batch, spatial[0], spatial[1], channels],
            Self::NCL => vec![batch, channels, spatial[0]],
            Self::NLC => vec![batch, spatial[0], channels],
        }
    }

    /// Whether channels come before spatial dimensions.
    #[must_use]
    pub fn is_channels_first(self) -> bool {
        matches!(self, Self::NCHW | Self::NCL)
    }

    /// Compute the permutation needed to convert from `self` to `target`.
    ///
    /// Returns an array of indices such that `output[i] = input[perm[i]]`.
    #[must_use]
    pub fn permutation_to(self, target: Self) -> Vec<usize> {
        // Build mapping: for each semantic role, what position is it in?
        let src = self.indices();
        let tgt = target.indices();

        let ndim = match self {
            Self::NCHW | Self::NHWC => 4,
            Self::NCL | Self::NLC => 3,
        };

        // Map semantic positions: batch, channel, spatial[0], spatial[1]
        let mut src_pos = vec![src.batch, src.channel];
        src_pos.extend_from_slice(src.spatial());

        let mut tgt_pos = vec![tgt.batch, tgt.channel];
        tgt_pos.extend_from_slice(tgt.spatial());

        // perm[tgt_position] = src_position for each semantic role
        let mut perm = vec![0; ndim];
        for (semantic_idx, &tp) in tgt_pos.iter().enumerate() {
            perm[tp] = src_pos[semantic_idx];
        }
        perm
    }
}

/// Kernel (weight) layout for convolution filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelLayout {
    /// OutChannels, InChannels, Height, Width (PyTorch default for 2D).
    OIHW,
    /// Height, Width, InChannels, OutChannels (TensorFlow default for 2D).
    HWIO,
    /// OutChannels, InChannels, Length (PyTorch default for 1D).
    OIL,
    /// Length, InChannels, OutChannels (TensorFlow default for 1D).
    LIO,
}

/// Fully describes input, kernel, and output data format for a convolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvDimensionNumbers {
    /// Layout of the input tensor.
    pub input_layout: ConvLayout,
    /// Layout of the kernel tensor.
    pub kernel_layout: KernelLayout,
    /// Layout of the output tensor.
    pub output_layout: ConvLayout,
}

impl Default for ConvDimensionNumbers {
    fn default() -> Self {
        Self {
            input_layout: ConvLayout::NCHW,
            kernel_layout: KernelLayout::OIHW,
            output_layout: ConvLayout::NCHW,
        }
    }
}

impl std::fmt::Display for ConvLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NCHW => write!(f, "NCHW"),
            Self::NHWC => write!(f, "NHWC"),
            Self::NCL => write!(f, "NCL"),
            Self::NLC => write!(f, "NLC"),
        }
    }
}

impl std::fmt::Display for KernelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OIHW => write!(f, "OIHW"),
            Self::HWIO => write!(f, "HWIO"),
            Self::OIL => write!(f, "OIL"),
            Self::LIO => write!(f, "LIO"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nchw_indices() {
        let idx = ConvLayout::NCHW.indices();
        assert_eq!(idx.batch, 0);
        assert_eq!(idx.channel, 1);
        assert_eq!(idx.spatial(), &[2, 3]);
    }

    #[test]
    fn test_nhwc_indices() {
        let idx = ConvLayout::NHWC.indices();
        assert_eq!(idx.batch, 0);
        assert_eq!(idx.channel, 3);
        assert_eq!(idx.spatial(), &[1, 2]);
    }

    #[test]
    fn test_ncl_indices() {
        let idx = ConvLayout::NCL.indices();
        assert_eq!(idx.batch, 0);
        assert_eq!(idx.channel, 1);
        assert_eq!(idx.spatial(), &[2]);
    }

    #[test]
    fn test_nlc_indices() {
        let idx = ConvLayout::NLC.indices();
        assert_eq!(idx.batch, 0);
        assert_eq!(idx.channel, 2);
        assert_eq!(idx.spatial(), &[1]);
    }

    #[test]
    fn test_parse_shape_nchw() {
        let (b, c, s) = ConvLayout::NCHW.parse_shape(&[2, 3, 32, 32]);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
        assert_eq!(s, vec![32, 32]);
    }

    #[test]
    fn test_parse_shape_nhwc() {
        let (b, c, s) = ConvLayout::NHWC.parse_shape(&[2, 32, 32, 3]);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
        assert_eq!(s, vec![32, 32]);
    }

    #[test]
    fn test_parse_shape_ncl() {
        let (b, c, s) = ConvLayout::NCL.parse_shape(&[4, 16, 100]);
        assert_eq!(b, 4);
        assert_eq!(c, 16);
        assert_eq!(s, vec![100]);
    }

    #[test]
    fn test_build_shape_nchw() {
        let shape = ConvLayout::NCHW.build_shape(2, 3, &[32, 32]);
        assert_eq!(shape, vec![2, 3, 32, 32]);
    }

    #[test]
    fn test_build_shape_nhwc() {
        let shape = ConvLayout::NHWC.build_shape(2, 3, &[32, 32]);
        assert_eq!(shape, vec![2, 32, 32, 3]);
    }

    #[test]
    fn test_build_shape_ncl() {
        let shape = ConvLayout::NCL.build_shape(4, 16, &[100]);
        assert_eq!(shape, vec![4, 16, 100]);
    }

    #[test]
    fn test_build_shape_nlc() {
        let shape = ConvLayout::NLC.build_shape(4, 16, &[100]);
        assert_eq!(shape, vec![4, 100, 16]);
    }

    #[test]
    fn test_is_channels_first() {
        assert!(ConvLayout::NCHW.is_channels_first());
        assert!(ConvLayout::NCL.is_channels_first());
        assert!(!ConvLayout::NHWC.is_channels_first());
        assert!(!ConvLayout::NLC.is_channels_first());
    }

    #[test]
    fn test_permutation_nchw_to_nhwc() {
        let perm = ConvLayout::NCHW.permutation_to(ConvLayout::NHWC);
        // NCHW[0,1,2,3] -> NHWC: batch stays 0, H=2->1, W=3->2, C=1->3
        assert_eq!(perm, vec![0, 2, 3, 1]);
    }

    #[test]
    fn test_permutation_nhwc_to_nchw() {
        let perm = ConvLayout::NHWC.permutation_to(ConvLayout::NCHW);
        // NHWC[0,1,2,3] -> NCHW: batch stays 0, C=3->1, H=1->2, W=2->3
        assert_eq!(perm, vec![0, 3, 1, 2]);
    }

    #[test]
    fn test_permutation_identity() {
        let perm = ConvLayout::NCHW.permutation_to(ConvLayout::NCHW);
        assert_eq!(perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_permutation_ncl_to_nlc() {
        let perm = ConvLayout::NCL.permutation_to(ConvLayout::NLC);
        // NCL[0,1,2] -> NLC: batch stays 0, L=2->1, C=1->2
        assert_eq!(perm, vec![0, 2, 1]);
    }

    #[test]
    fn test_display_conv_layout() {
        assert_eq!(format!("{}", ConvLayout::NCHW), "NCHW");
        assert_eq!(format!("{}", ConvLayout::NHWC), "NHWC");
        assert_eq!(format!("{}", ConvLayout::NCL), "NCL");
        assert_eq!(format!("{}", ConvLayout::NLC), "NLC");
    }

    #[test]
    fn test_display_kernel_layout() {
        assert_eq!(format!("{}", KernelLayout::OIHW), "OIHW");
        assert_eq!(format!("{}", KernelLayout::HWIO), "HWIO");
        assert_eq!(format!("{}", KernelLayout::OIL), "OIL");
        assert_eq!(format!("{}", KernelLayout::LIO), "LIO");
    }

    #[test]
    fn test_conv_dimension_numbers_default() {
        let cdn = ConvDimensionNumbers::default();
        assert_eq!(cdn.input_layout, ConvLayout::NCHW);
        assert_eq!(cdn.kernel_layout, KernelLayout::OIHW);
        assert_eq!(cdn.output_layout, ConvLayout::NCHW);
    }
}
