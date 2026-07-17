//! Numeric dtypes that the model can request from a backend.
//!
//! Backends may not implement every variant; unsupported variants must
//! return a backend-specific error (no silent fallback).

/// Tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    U32,
}

impl Dtype {
    /// Element size in bytes.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I64 => 8,
        }
    }

    /// True if the dtype is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F16 | Dtype::BF16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_in_bytes_matches_native_types() {
        assert_eq!(Dtype::F32.size_in_bytes(), 4);
        assert_eq!(Dtype::F16.size_in_bytes(), 2);
        assert_eq!(Dtype::BF16.size_in_bytes(), 2);
        assert_eq!(Dtype::I64.size_in_bytes(), 8);
        assert_eq!(Dtype::U32.size_in_bytes(), 4);
    }

    #[test]
    fn is_float_classifies_correctly() {
        assert!(Dtype::F32.is_float());
        assert!(Dtype::F16.is_float());
        assert!(Dtype::BF16.is_float());
        assert!(!Dtype::I64.is_float());
        assert!(!Dtype::U32.is_float());
    }
}
