//! Memory-bandwidth kernels for Phase 1: read, copy, scale, add.
//!
//! Each kernel does almost no arithmetic per byte, so its speed is set by how
//! fast memory can deliver data. Running one over buffers of growing size
//! shows the bandwidth of each level of the hierarchy (L1, L2, system cache,
//! DRAM). `read` matters most: decode reads every weight once per token and
//! writes almost nothing.
//!
//! Bytes are counted the way STREAM does: each element read or written once
//! counts once. A write to memory not already in cache really costs a read
//! too (the line is fetched before it is overwritten, "write-allocate"), so
//! measured copy/scale/add rates understate the true traffic by up to a third.
//!
//! The Phase 0 sum (`s += x` in a loop) ran at ~3.3 GB/s because every add
//! waited for the previous one: a dependency chain, limited by add latency,
//! not by memory. `read_sum` keeps 8 independent NEON accumulators of 4 lanes
//! each, so 32 additions are in flight and the loop waits on memory instead.

/// One benchmark kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `sum(a)`: read only, like decode's weight stream.
    Read,
    /// `b = a`
    Copy,
    /// `b = s * a`
    Scale,
    /// `c = a + b`
    Add,
}

impl Kernel {
    pub const ALL: [Kernel; 4] = [Kernel::Read, Kernel::Copy, Kernel::Scale, Kernel::Add];

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Read => "read",
            Kernel::Copy => "copy",
            Kernel::Scale => "scale",
            Kernel::Add => "add",
        }
    }

    pub fn parse(s: &str) -> Option<Kernel> {
        Kernel::ALL.into_iter().find(|k| k.name() == s)
    }

    /// How many arrays the kernel touches (inputs + output).
    pub fn arrays(self) -> usize {
        match self {
            Kernel::Read => 1,
            Kernel::Copy | Kernel::Scale => 2,
            Kernel::Add => 3,
        }
    }

    /// Bytes counted per pass over arrays of `n` floats each (STREAM rule:
    /// every array is read or written once).
    pub fn bytes_per_pass(self, n: usize) -> usize {
        self.arrays() * n * std::mem::size_of::<f32>()
    }
}

/// Sum of all elements, with 8 independent 4-lane accumulators.
///
/// The result's rounding differs from a left-to-right sum (32 partial sums
/// are combined at the end); for a bandwidth benchmark only "every element
/// was read" matters.
pub fn read_sum(a: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        read_sum_neon(a)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        read_sum_portable(a)
    }
}

#[cfg(target_arch = "aarch64")]
fn read_sum_neon(a: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let (chunks, tail) = a.as_chunks::<32>();
    // SAFETY: NEON is part of the aarch64 baseline. Each load reads 4 floats
    // at `p + k*4` for k < 8, inside one 32-float chunk.
    let mut total = unsafe {
        let mut acc = [vdupq_n_f32(0.0); 8];
        for c in chunks {
            let p = c.as_ptr();
            for (k, r) in acc.iter_mut().enumerate() {
                *r = vaddq_f32(*r, vld1q_f32(p.add(4 * k)));
            }
        }
        // Combine the 8 vectors pairwise, then the 4 lanes of the last one.
        let s01 = vaddq_f32(acc[0], acc[1]);
        let s23 = vaddq_f32(acc[2], acc[3]);
        let s45 = vaddq_f32(acc[4], acc[5]);
        let s67 = vaddq_f32(acc[6], acc[7]);
        vaddvq_f32(vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67)))
    };
    for &x in tail {
        total += x;
    }
    total
}

/// Host build (x86 tests): the same 32 independent lanes as plain arrays;
/// LLVM turns the lane loop into vector adds.
#[cfg(not(target_arch = "aarch64"))]
fn read_sum_portable(a: &[f32]) -> f32 {
    let (chunks, tail) = a.as_chunks::<32>();
    let mut acc = [0.0f32; 32];
    for c in chunks {
        for (r, &x) in acc.iter_mut().zip(c) {
            *r += x;
        }
    }
    acc.iter().sum::<f32>() + tail.iter().sum::<f32>()
}

/// `dst = src`. Written as a loop, not `copy_from_slice`, so it measures a
/// plain load/store stream rather than the C library's `memcpy` tricks
/// (though LLVM may still recognise the loop and call `memcpy`).
pub fn copy(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s;
    }
}

/// `dst = s * src`
pub fn scale(dst: &mut [f32], src: &[f32], s: f32) {
    assert_eq!(dst.len(), src.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = s * x;
    }
}

/// `dst = a + b`
pub fn add(dst: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(dst.len(), b.len());
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *d = x + y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small integers are exact in f32 and so are their sums (all far below
    /// 2^24), so any summation order gives the exact total. The lengths
    /// cover: empty, tail only, exactly one chunk, chunk + tail, many chunks.
    #[test]
    fn read_sum_is_exact_on_small_integers() {
        for n in [0, 1, 31, 32, 33, 1000, 4099] {
            let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
            let expected: u64 = (0..n as u64).map(|i| i % 7).sum();
            assert_eq!(read_sum(&a), expected as f32, "n = {n}");
        }
    }

    /// The +4-byte-shifted slice the alignment benchmark uses still sums
    /// every element it contains.
    #[test]
    fn read_sum_on_misaligned_slice() {
        let mut b = crate::aligned::AlignedBuf::zeros(100);
        b.fill(1.0);
        assert_eq!(read_sum(&b[1..]), 99.0);
    }

    #[test]
    fn copy_scale_add() {
        let a: Vec<f32> = (0..37).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..37).map(|i| 2.0 * i as f32).collect();
        let mut d = vec![0.0; 37];
        copy(&mut d, &a);
        assert_eq!(d, a);
        scale(&mut d, &a, 3.0);
        assert_eq!(d[36], 108.0);
        add(&mut d, &a, &b);
        assert_eq!(d[36], 108.0);
    }

    #[test]
    fn byte_counts_follow_stream_rule() {
        assert_eq!(Kernel::Read.bytes_per_pass(1000), 4000);
        assert_eq!(Kernel::Copy.bytes_per_pass(1000), 8000);
        assert_eq!(Kernel::Add.bytes_per_pass(1000), 12000);
        assert_eq!(Kernel::parse("scale"), Some(Kernel::Scale));
    }
}
