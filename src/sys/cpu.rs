//! What the CPU reports about itself.

/// Size in bytes of the smallest data-cache line, as the core reports it.
///
/// AArch64 exposes this in the `CTR_EL0` register (Cache Type Register).
/// Bits 19:16, `DminLine`, hold log2 of the line size counted in 4-byte
/// words, so the line is `4 << DminLine` bytes: `DminLine = 4` means 64.
/// Linux lets user programs read this register directly (no system call).
#[cfg(target_arch = "aarch64")]
pub fn dcache_line_bytes() -> usize {
    let ctr: u64;
    // SAFETY: `mrs` from CTR_EL0 only reads a register; it touches no memory,
    // no stack and no flags, as the options declare.
    unsafe {
        std::arch::asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags));
    }
    4 << ((ctr >> 16) & 0xF)
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// `AlignedBuf` assumes 64-byte lines; this checks the assumption
    /// against the hardware instead of trusting it.
    #[test]
    fn cache_line_matches_aligned_buf() {
        let line = dcache_line_bytes();
        println!("CTR_EL0 data cache line: {line} bytes");
        assert_eq!(line, crate::aligned::ALIGN);
    }
}
