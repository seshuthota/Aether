//! A heap buffer of `f32`s whose first element sits on a 64-byte boundary.
//!
//! `Vec<f32>` asks the allocator for 4-byte alignment, because that is all an
//! `f32` needs. Kernels and benchmarks want more: a buffer that starts on a
//! cache-line boundary, so every run sees the same layout and a 16-byte NEON
//! load never straddles two lines by accident. `Vec` cannot be given such
//! memory (it would free it with alignment 4, a different `Layout`), so this
//! type asks the allocator itself and remembers how it asked.
//!
//! Invariants kept by every function in this module:
//!
//! - **Allocation**: if the buffer owns memory, `layout` is exactly the
//!   `Layout` it was allocated with, and `Drop` frees it with that layout,
//!   exactly once.
//! - **Empty**: `layout.size() == 0` means nothing was allocated. The
//!   allocator is never called with size 0 and `Drop` does not free.
//! - **Initialised**: every element holds a valid `f32` from the start
//!   (`alloc_zeroed`: zero bits are `0.0`), so handing out slices is sound.
//! - **Aligned**: `ptr` is a multiple of [`ALIGN`], even for an empty buffer.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ptr::NonNull;

/// Alignment of every [`AlignedBuf`], in bytes: one cache line.
///
/// 64 is what almost every ARM64 core uses. The core reports its real line
/// size in the `CTR_EL0` register; checking this constant against it on the
/// phone is a later step.
pub const ALIGN: usize = 64;

/// An owned, zero-initialised, 64-byte-aligned buffer of `f32`.
///
/// Like `Vec<f32>` without growth: the length is fixed at construction.
pub struct AlignedBuf {
    /// Address of element 0. `NonNull` rather than `*mut f32`: a slice
    /// pointer must never be null, even for length 0, and the type says so.
    ptr: NonNull<f32>,
    /// Number of `f32` elements (not bytes).
    len: usize,
    /// The exact size and alignment passed to `alloc_zeroed`; `Drop` hands
    /// the same value back to `dealloc`. Size 0 marks "nothing allocated".
    layout: Layout,
}

impl AlignedBuf {
    /// A buffer of `len` floats, all `0.0`, starting on a 64-byte boundary.
    ///
    /// Panics if `len * 4` bytes cannot be described (larger than `isize::MAX`).
    /// Aborts, like `Vec`, if the allocator has no memory to give.
    pub fn zeros(len: usize) -> AlignedBuf {
        // `array` computes len * 4 with an overflow check and alignment 4;
        // `align_to` raises the alignment to 64 and leaves the size alone.
        let layout = Layout::array::<f32>(len)
            .and_then(|l| l.align_to(ALIGN))
            .expect("AlignedBuf size overflows");

        if layout.size() == 0 {
            // Calling the allocator with size 0 is undefined behaviour, so an
            // empty buffer allocates nothing. Its pointer still has to be
            // non-null and aligned; address 64 is both, and is never read
            // because the length is 0. (`NonNull::dangling()` would give 4.)
            let ptr = NonNull::new(std::ptr::without_provenance_mut(ALIGN)).unwrap();
            return AlignedBuf { ptr, len, layout };
        }

        // SAFETY: `layout.size()` is non-zero (checked above), which is the
        // one requirement `alloc_zeroed` puts on its caller.
        let raw = unsafe { alloc_zeroed(layout) };
        // A null return means out of memory; abort the way `Vec` does.
        let Some(ptr) = NonNull::new(raw.cast::<f32>()) else {
            handle_alloc_error(layout)
        };
        AlignedBuf { ptr, len, layout }
    }

    /// Number of `f32` elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the buffer holds no elements (and owns no memory).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Address of element 0. Always a multiple of [`ALIGN`].
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// The elements as an ordinary slice. The borrow is tied to `&self`, so
    /// the compiler refuses any use of it after the buffer is dropped.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `ptr` is valid for `len` initialised, aligned f32s (zeroed
        // at allocation; dangling-but-aligned when len is 0, which a slice
        // allows), and the shared borrow of `self` rules out writers.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The elements as a mutable slice. Takes `&mut self` so that while this
    /// slice exists, no other slice of the same buffer can.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as in `as_slice`; `&mut self` guarantees this is the only
        // live view of the memory.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// `Deref` lets a buffer be used wherever a slice is expected: `buf[3]`,
// `buf.iter()`, `&buf[a..b]`, `buf.len()`.
impl std::ops::Deref for AlignedBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

// A raw pointer makes the compiler assume the type cannot cross threads. This
// one owns its memory exclusively, exactly like `Vec<f32>`, and only hands out
// access through `&self` / `&mut self`, so the borrow rules that make `Vec`
// thread-safe apply unchanged.
// SAFETY: no shared mutable state exists outside those borrows.
unsafe impl Send for AlignedBuf {}
// SAFETY: `&AlignedBuf` only gives out `&[f32]`, which is `Sync`.
unsafe impl Sync for AlignedBuf {}

impl Drop for AlignedBuf {
    /// Gives the memory back. Rust calls this exactly once, when the buffer
    /// goes out of scope, so double frees and leaks cannot happen.
    fn drop(&mut self) {
        if self.layout.size() == 0 {
            return; // empty: nothing was allocated, so nothing to free
        }
        // SAFETY: `ptr` came from `alloc_zeroed(self.layout)` in `zeros`, the
        // layout is the same value, and `drop` runs once, so this frees a live
        // allocation exactly once with its original layout.
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The start is 64-aligned whatever the length: 1 float, a few short of
    /// and just past one cache line (16 floats), and large odd sizes where
    /// the allocator switches strategy (big blocks come from mmap).
    #[test]
    fn start_is_64_aligned_for_awkward_lengths() {
        for len in [1, 3, 15, 16, 17, 1000, 4097, (1 << 20) + 1] {
            let b = AlignedBuf::zeros(len);
            assert_eq!(b.len(), len);
            assert_eq!(b.as_ptr() as usize % ALIGN, 0, "len {len}");
        }
    }

    /// Length 0 never reaches the allocator, yet still has an aligned,
    /// non-null pointer, and dropping it must not try to free anything.
    #[test]
    fn empty_buffer_is_aligned_and_drops_cleanly() {
        let b = AlignedBuf::zeros(0);
        assert!(b.is_empty());
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
        drop(b);
    }

    /// `alloc_zeroed` means every element reads as 0.0 before any write; a
    /// write through the mutable view is seen by the shared view.
    #[test]
    fn starts_zeroed_and_writes_are_visible() {
        let mut b = AlignedBuf::zeros(37);
        assert!(b.iter().all(|&x| x == 0.0));
        b[36] = 2.5;
        b.as_mut_slice()[0] = -1.0;
        assert_eq!((b.as_slice()[0], b[36]), (-1.0, 2.5));
    }

    /// An empty buffer still yields a valid empty slice.
    #[test]
    fn empty_buffer_gives_empty_slice() {
        let mut b = AlignedBuf::zeros(0);
        assert!(b.as_slice().is_empty());
        assert!(b.as_mut_slice().is_empty());
    }

    /// Scoped threads can each fill their own chunk and then all read the
    /// whole buffer: this is how the parallel bandwidth sweep uses it.
    #[test]
    fn chunks_can_be_shared_across_threads() {
        let mut b = AlignedBuf::zeros(64);
        std::thread::scope(|s| {
            for (i, chunk) in b.chunks_mut(16).enumerate() {
                s.spawn(move || chunk.fill(i as f32));
            }
        });
        let shared = &b;
        std::thread::scope(|s| {
            s.spawn(move || assert_eq!(shared[63], 3.0));
        });
    }

    /// usize::MAX floats is far more than any address space; the size
    /// computation must refuse it instead of wrapping around to a small
    /// number and allocating a buffer shorter than `len` claims.
    #[test]
    #[should_panic(expected = "overflows")]
    fn huge_length_is_rejected() {
        AlignedBuf::zeros(usize::MAX);
    }
}
