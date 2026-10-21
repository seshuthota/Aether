//! The operating system and the hardware under it: which core a thread runs
//! on, what the CPU reports about itself, and the `/sys` files that expose
//! clock speeds, temperatures and battery readings.
//!
//! Everything here is Linux (Android is Linux); nothing is engine logic.

pub mod affinity;
pub mod cpu;
pub mod sysfs;
