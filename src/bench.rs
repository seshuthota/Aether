//! Minimal benchmark harness: warm-up, repeated timed runs, robust statistics.
//!
//! This module deliberately avoids pulling in a full benchmarking framework;
//! it provides just enough machinery to answer "how long does this operation
//! take on this device?" with numbers that survive measurement noise:
//!
//! 1. [`bench()`] warms the code path up (untimed) and then times each run
//!    individually with [`Instant`].
//! 2. [`Stats::from_durations`] reduces the raw timings to min/median/mean/max.
//!
//! `Instant` is monotonic, so the measurements are immune to wall-clock
//! adjustments while a benchmark is running.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Summary of repeated timings. Nanoseconds throughout.
///
/// All fields are `f64` nanoseconds rather than [`Duration`] so callers can
/// compare, average, and print them without unit-conversion friction.
#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    /// Number of timed runs that went into this summary.
    pub runs: usize,
    /// Fastest observed run. The best approximation of the code's true cost
    /// under ideal conditions (warm caches, no scheduler interference).
    pub min: f64,
    /// Middle value of the sorted samples. The headline number: unlike the
    /// mean, a single slow outlier cannot drag it far.
    pub median: f64,
    /// Arithmetic mean of all samples. Sensitive to outliers; useful mainly
    /// to spot them (a mean far above the median implies a heavy tail).
    pub mean: f64,
    /// Slowest observed run. Together with `min` it brackets the noise level
    /// of the measurement environment.
    pub max: f64,
}

impl Stats {
    /// Summarizes `durations` into min/median/mean/max, in nanoseconds.
    ///
    /// Input order does not matter: the statistics are computed from a sorted
    /// copy, so callers can push samples as they come.
    ///
    /// Panics on an empty slice: a benchmark with zero runs is a bug.
    pub fn from_durations(durations: &[Duration]) -> Stats {
        // A zero-run summary would leave every statistic below either
        // panicking or NaN, so fail loudly at the source instead.
        assert!(!durations.is_empty(), "no timings to summarize");

        // Convert to f64 nanoseconds once; every later step works on this
        // buffer, so each sample is converted exactly once.
        let mut ns: Vec<f64> = durations.iter().map(|d| d.as_nanos() as f64).collect();

        // Sort ascending so min/max/median are simple indexing operations.
        // `partial_cmp` cannot fail here: durations are finite, never NaN.
        ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = ns.len();
        // With an even count there is no middle value; use the mean of the middle two.
        let median = if n % 2 == 1 {
            ns[n / 2]
        } else {
            (ns[n / 2 - 1] + ns[n / 2]) / 2.0
        };
        Stats {
            runs: n,
            min: ns[0], // sorted ascending: the first element is the fastest run
            median,
            // Sum as f64: nanoseconds are far below f64's exact-integer range
            // (2^53 ns is over 100 days), and a mean is fractional anyway.
            mean: ns.iter().sum::<f64>() / n as f64,
            max: ns[n - 1], // sorted ascending: the last element is the slowest run
        }
    }
}

/// Runs `f` `warmup` times untimed, then `iters` times timed (one timing per call).
///
/// Why per-call timing instead of one timing around the whole batch? Per-call
/// timing keeps each slow iteration identifiable: an outlier shows up in `max`
/// and widens the min/max spread but leaves `median` (the headline number)
/// intact. It also keeps every `Duration` sample directly comparable.
///
/// Why warm up? The first executions of a code path pay one-time costs that
/// are real but not representative of steady-state operation: cold
/// instruction/data caches, branch predictors that have not learned the
/// branch history yet, and (on a phone) CPU frequency scaling that may not
/// have ramped up. Those runs are deliberately executed but not recorded.
///
/// `f` is `FnMut`, so it may mutate captured state — necessary for benchmarks
/// that fill or drain a buffer. `R` is passed through [`black_box`], so the
/// function cannot be optimized away regardless of what it returns.
///
/// Each call must take much longer than a clock read (~50 ns on the phone),
/// otherwise the clock overhead dominates the measurement.
///
/// # Panics
///
/// If `iters` is zero: a benchmark with no timed runs cannot be summarized
/// (see [`Stats::from_durations`]).
pub fn bench<R>(warmup: usize, iters: usize, mut f: impl FnMut() -> R) -> Stats {
    // Warm-up phase. The return values still go through `black_box` so each
    // call is genuinely executed and not elided as dead code.
    for _ in 0..warmup {
        black_box(f());
    }

    // Pre-allocate: growing the vec mid-measurement would fold an unrelated
    // malloc/realloc cost into one of the timed samples.
    let mut durations = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        black_box(f()); // black_box: the compiler must treat the result as used
        durations.push(start.elapsed());
    }

    // Reduce the raw samples to the summary statistics.
    Stats::from_durations(&durations)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds `Duration`s from millisecond values so the expected constants in
    /// the tests below can be written as small round numbers.
    fn ms(v: &[u64]) -> Vec<Duration> {
        v.iter().map(|&x| Duration::from_millis(x)).collect()
    }

    /// Odd sample count: the median is exactly the middle element after
    /// sorting (3 ms in ns). The unsorted input proves order does not matter.
    #[test]
    fn median_odd_count() {
        let s = Stats::from_durations(&ms(&[5, 1, 3]));
        assert_eq!(s.median, 3_000_000.0);
    }

    /// Even sample count: no middle element exists, so the median is the mean
    /// of the two middle values — (2 ms + 3 ms) / 2.
    #[test]
    fn median_even_count_averages_middle_two() {
        let s = Stats::from_durations(&ms(&[4, 1, 3, 2]));
        assert_eq!(s.median, 2_500_000.0);
    }

    /// One 100 ms outlier among four 10 ms samples: the mean jumps to
    /// (10*4 + 100) / 5 = 28 ms while the median stays at 10 ms. This is the
    /// behavior that makes median the headline number.
    #[test]
    fn outlier_moves_mean_but_not_median() {
        let s = Stats::from_durations(&ms(&[10, 10, 10, 10, 100]));
        assert_eq!(s.median, 10_000_000.0);
        assert_eq!(s.mean, 28_000_000.0);
        assert_eq!((s.min, s.max), (10_000_000.0, 100_000_000.0));
    }

    /// The closure must run `warmup + iters` times in total, but only the
    /// `iters` timed runs may be summarized.
    #[test]
    fn bench_runs_warmup_plus_iters() {
        let mut calls = 0;
        let s = bench(3, 7, || calls += 1);
        assert_eq!(calls, 10);
        assert_eq!(s.runs, 7);
    }
}
