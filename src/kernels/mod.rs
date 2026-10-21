//! The engine's op functions. Model code calls these and never touches
//! memory layouts or SIMD itself; faster versions (NEON, threads, quantized)
//! replace the bodies here without changing the callers.

pub mod scalar;
