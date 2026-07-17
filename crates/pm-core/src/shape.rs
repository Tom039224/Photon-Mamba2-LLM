//! Tensor shape (ordered list of dimensions).

use std::fmt;

/// Owned tensor shape. Backends usually borrow `&[usize]` directly; this
/// type is for places where ownership is needed (configs, tests, errors).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    #[must_use]
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self(v.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Self(v.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_is_product() {
        assert_eq!(Shape::from([2, 3, 4]).numel(), 24);
        assert_eq!(Shape::from(Vec::<usize>::new()).numel(), 1); // scalar
    }

    #[test]
    fn display_matches_python_style() {
        assert_eq!(format!("{}", Shape::from([2, 3, 4])), "[2, 3, 4]");
    }
}
