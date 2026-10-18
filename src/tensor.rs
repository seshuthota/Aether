//! A minimal row-major f32 tensor: a flat buffer plus a shape.
//!
//! A tensor is an n-dimensional grid of `f32` numbers. This module stores one
//! the way numeric code traditionally does: not as nested vectors, but as a
//! single flat `Vec<f32>` plus metadata that turns a logical position like
//! `(2, 3, 1)` into an offset into that flat buffer. One allocation, and the
//! elements sit contiguously — which is what later kernels (matmul, quantized
//! dot products) will want to iterate over.
//!
//! Two pieces of metadata do the work:
//!
//! - `shape`: how many elements each dimension has;
//! - `strides`: how many buffer slots to skip to advance one step along each
//!   dimension.
//!
//! "Row-major" (also called C order) means the last dimension varies fastest
//! in memory. For shape `[2, 3]` (2 rows of 3), the buffer looks like:
//!
//! ```text
//! logical positions          flat data (row-major)
//! (0,0) (0,1) (0,2)     [ d0, d1, d2, | d3, d4, d5 ]
//! (1,0) (1,1) (1,2)       row 0           row 1
//! ```
//!
//! Sizes and indices are `usize` (they are never negative, and `usize` is what
//! Rust's slices are indexed with). This module only *addresses* elements;
//! arithmetic comes later, in the kernels that will consume this type.

/// Row-major `f32` tensor. Element `(i0, i1, ..)` lives at `sum(i_k * stride_k)`,
/// where each stride is the product of all dimension sizes to its right.
///
/// Worked example — shape `[4, 5, 6]`:
///
/// - the strides are `[30, 6, 1]` (a step along dim 0 crosses a whole 5×6
///   slab, along dim 1 one row of 6, along dim 2 one element);
/// - the position `(2, 3, 1)` lives at flat index `2*30 + 3*6 + 1*1 = 79`.
///
/// Invariants maintained by every constructor in this module:
///
/// - `shape.len() == strides.len()` — one stride per dimension; and
/// - `data.len() == shape.iter().product()` — the buffer holds exactly `numel`
///   elements.
///
/// Derived traits: `Clone` is a deep copy (metadata *and* the full data
/// buffer), and `PartialEq` compares shape, strides, and element values.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Size of each dimension, outermost first: `[4, 5, 6]` means
    /// 4 blocks of (5 rows of 6 elements).
    shape: Vec<usize>,
    /// Elements to skip in `data` to advance by one step along each dimension.
    /// In this row-major layout the last stride is always 1 (the innermost
    /// dimension is contiguous in memory).
    ///
    /// Today these are a pure function of `shape`, but the field is stored
    /// because later features (views, transpose, slicing) produce layouts
    /// whose strides no longer follow from the shape. `index` is written
    /// against the field so it keeps working then.
    strides: Vec<usize>,
    /// The flat element storage, in row-major order; length equals numel.
    data: Vec<f32>,
}

/// Strides for a row-major layout: the last dimension has stride 1.
///
/// Filled in right-to-left: each stride is the product of the dimension sizes
/// to its right, because that is how many slots one step along that dimension
/// skips. Starting from `shape = [4, 5, 6]`:
///
/// ```text
/// strides = [1, 1, 1]                       (all start at 1)
/// k = 1:  strides[1] = 1 * shape[2] = 6     (a step over dim 1 skips 6)
/// k = 0:  strides[0] = 6 * shape[1] = 30    (a step over dim 0 skips 30)
/// result: [30, 6, 1]
/// ```
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    // The last dimension keeps its initial 1; everything else gets computed.
    let mut strides = vec![1; shape.len()];
    // Walk right-to-left over every dimension except the last.
    // `saturating_sub(1)` keeps this range empty for 0-d (`[]`, a scalar) and
    // 1-d shapes; plain `len() - 1` would underflow on `[]`.
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

impl Tensor {
    /// A tensor of the given shape with every element set to `0.0`.
    /// Delegates to [`Tensor::from_vec`] so validation and stride building
    /// happen in exactly one place.
    pub fn zeros(shape: &[usize]) -> Tensor {
        // The product of an empty iterator is 1, which is exactly right for
        // the scalar shape `[]`: one element, not zero.
        let numel = shape.iter().product();
        Tensor::from_vec(shape, vec![0.0; numel])
    }

    /// Builds a tensor from an existing flat, row-major buffer.
    ///
    /// The elements themselves are moved in as-is — no copying or reordering —
    /// but the shape is copied (`to_vec()`), so the caller's slice can go away
    /// afterwards.
    ///
    /// Panics if `data.len()` doesn't match the shape: that's a programming error.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Tensor {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "shape {shape:?} needs {numel} elements");
        // Own the metadata: the caller's slice can go away, the caller's
        // buffer is moved in, and strides are derived from the shape.
        Tensor { shape: shape.to_vec(), strides: row_major_strides(shape), data }
    }

    /// The shape (one size per dimension, outermost first).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides (elements to skip per dimension; see the struct docs).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Total number of elements (`numel` = "number of elements"): the product
    /// of the shape. A 1000×1000 tensor has 1,000,000.
    pub fn numel(&self) -> usize {
        // Taken from `data.len()` rather than recomputed from `shape`: the
        // constructors already force the two to agree, and the buffer length
        // is the fact the rest of the program relies on.
        self.data.len()
    }

    /// Bytes of storage backing the tensor: one `f32` is 4 bytes, so this is
    /// `numel() * 4` (written via `size_of` so it stays correct if `f32` ever
    /// becomes something else).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Read-only access to the flat row-major buffer. The escape hatch used by
    /// tests, benchmarks, and (later) kernels that want to iterate linearly.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to the flat row-major buffer. Requiring `&mut self` is
    /// what lets Rust guarantee no one can read the buffer while it is being
    /// written.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Flat index of a multi-dimensional position. Panics if out of range.
    ///
    /// This is the whole addressing scheme in one function: the flat offset is
    /// the dot product of the position with the strides. For shape `[4, 5, 6]`
    /// (strides `[30, 6, 1]`) and position `(2, 3, 1)`:
    ///
    /// ```text
    /// flat = 2*30 + 3*6 + 1*1 = 79
    ///        ^^^^   ^^^   ^^^
    ///        2 steps 3     1
    ///        along   steps steps
    ///        dim 0   dim 1 dim 2
    /// ```
    ///
    /// For any valid position the result is `< numel()`.
    pub fn index(&self, pos: &[usize]) -> usize {
        // Rank check: the position must supply exactly one index per dimension.
        assert_eq!(pos.len(), self.shape.len(), "expected {} indices", self.shape.len());
        let mut flat = 0;
        for k in 0..pos.len() {
            // Bounds check: each index must be valid within its own dimension.
            assert!(pos[k] < self.shape[k], "index {} out of range for dim {} (size {})", pos[k], k, self.shape[k]);
            flat += pos[k] * self.strides[k];
        }
        flat
    }

    /// The element at `pos`. Panics if `pos` is out of range (same checks as
    /// [`Tensor::index`], with the same messages).
    pub fn get(&self, pos: &[usize]) -> f32 {
        // Slice indexing is bounds-checked too, but a valid `pos` already
        // guarantees the flat index is inside the buffer.
        self.data[self.index(pos)]
    }

    /// Writes `value` to the element at `pos`. Panics if `pos` is out of range.
    pub fn set(&mut self, pos: &[usize], value: f32) {
        // Resolve the read-only part of the work first (`index` only reads
        // `self`), then perform the write. One line would compile the same,
        // but the read-then-write order is spelled out here.
        let i = self.index(pos);
        self.data[i] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For shape [4, 5, 6], a step along dim 0 skips a 5×6 slab (30 elements),
    /// along dim 1 skips one row of 6, and along dim 2 (innermost) moves by 1.
    #[test]
    fn strides_are_products_of_dims_to_the_right() {
        let t = Tensor::zeros(&[4, 5, 6]);
        assert_eq!(t.strides(), &[30, 6, 1]);
    }

    /// Shape [3, 4]: three rows of four. The last dimension is contiguous
    /// (stride 1); a step to the next row skips those four elements.
    #[test]
    fn strides_of_3_by_4_are_4_and_1() {
        assert_eq!(Tensor::zeros(&[3, 4]).strides(), &[4, 1]);
    }

    /// One full address computation on that tensor: 2*30 + 3*6 + 1*1 = 79.
    #[test]
    fn index_of_2_3_1_is_79() {
        let t = Tensor::zeros(&[4, 5, 6]);
        assert_eq!(t.index(&[2, 3, 1]), 79);
    }

    /// A 2×3 tensor filled with 0..=5, laid out row by row:
    ///
    /// ```text
    /// (0,0)=0  (0,1)=1  (0,2)=2
    /// (1,0)=3  (1,1)=4  (1,2)=5
    /// ```
    ///
    /// `(1, 0)` must read the first element of row 1 (flat 3), and writing
    /// `(1, 2)` must land at flat 5, the last buffer slot.
    #[test]
    fn get_set_roundtrip_and_row_major_order() {
        let mut t = Tensor::from_vec(&[2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(t.get(&[1, 0]), 3.0); // second row starts after 3 elements
        t.set(&[1, 2], 9.0);
        assert_eq!(t.data()[5], 9.0);
    }

    /// 1000×1000 = 1,000,000 elements; at 4 bytes per `f32` that is 4 MB.
    #[test]
    fn size_in_bytes() {
        let t = Tensor::zeros(&[1000, 1000]);
        assert_eq!(t.numel(), 1_000_000);
        assert_eq!(t.size_bytes(), 4_000_000);
    }

    /// Shape edge cases: `[]` is a scalar (the empty product is 1), and a
    /// one-dimensional shape gets the single stride 1.
    #[test]
    fn scalar_and_vector_shapes() {
        assert_eq!(Tensor::zeros(&[]).numel(), 1); // empty shape = one scalar
        assert_eq!(Tensor::zeros(&[7]).strides(), &[1]);
    }

    /// `(0, 3)` on a 2×3 tensor: dim 1 has size 3, so index 3 is one past its
    /// end. `index` must reject it instead of computing a wrong flat offset.
    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_index_panics() {
        Tensor::zeros(&[2, 3]).index(&[0, 3]);
    }

    /// 2×3 needs 6 elements; a 5-element buffer is rejected at construction,
    /// before any accessor can rely on the shape.
    #[test]
    #[should_panic(expected = "needs 6 elements")]
    fn from_vec_rejects_wrong_length() {
        Tensor::from_vec(&[2, 3], vec![0.0; 5]);
    }


}
