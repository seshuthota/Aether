//! A persistent thread pool with pinned workers, for splitting one op
//! (a GEMV, a GEMM) across cores.
//!
//! Decode runs hundreds of small ops per token, each only microseconds long,
//! so the cost of handing work to threads must be far below a microsecond.
//! Spawning threads per op (or even a mutex + condition variable) costs
//! several microseconds. This pool keeps its workers alive and pinned, and
//! hands out work through two counters:
//!
//! - `epoch`: the caller publishes a job, then increments `epoch`. Each
//!   worker remembers the last epoch it ran; a new value means "new job".
//! - `done`: each worker increments it when its share is finished; the
//!   caller waits until all `n - 1` workers have reported.
//!
//! Waiting is spin-then-park: a worker first busy-polls `epoch` (a new job
//! usually arrives within microseconds during decode), and only after
//! `SPIN` polls does it `park()` (sleep in the kernel until `unpark`). The
//! calling thread does share 0 of every job itself, so `n` threads means the
//! caller plus `n - 1` workers.
//!
//! The counters live on separate 64-byte cache lines: a line written by one
//! core and polled by others moves between their caches on every write
//! ("false sharing" when unrelated data shares it).

use crate::sys::affinity;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

/// Polls before a waiting worker parks. ~tens of microseconds of spinning.
const SPIN: u32 = 20_000;

/// Pads a value to its own 64-byte cache line.
#[repr(align(64))]
struct Line<T>(T);

/// The job: a reference to the caller's closure, erased to a raw pointer so
/// it can sit in an atomic. Valid only between publication and the moment
/// the caller has seen every worker finish (see `run`).
type Job = dyn Fn(usize, usize) + Sync;

struct Shared {
    epoch: Line<AtomicU64>,
    done: Line<AtomicUsize>,
    stop: Line<AtomicBool>,
    /// Set while a job runs: the pool has one job slot, so a second caller
    /// at the same time is a bug, caught here instead of racing.
    busy: AtomicBool,
    /// Points at a `&Job` living on the caller's stack during `run`.
    job: Line<AtomicPtr<&'static Job>>,
}

pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    cores: Vec<usize>,
}

impl ThreadPool {
    /// One thread per entry of `cores`, each pinned to that core. The
    /// calling thread becomes share 0 and is pinned to `cores[0]`.
    pub fn new(cores: &[usize]) -> std::io::Result<ThreadPool> {
        assert!(!cores.is_empty(), "a pool needs at least one core");
        affinity::pin_current_thread(&cores[..1])?;
        let shared = Arc::new(Shared {
            epoch: Line(AtomicU64::new(0)),
            done: Line(AtomicUsize::new(0)),
            stop: Line(AtomicBool::new(false)),
            busy: AtomicBool::new(false),
            job: Line(AtomicPtr::new(std::ptr::null_mut())),
        });
        let n = cores.len();
        let workers = cores[1..]
            .iter()
            .enumerate()
            .map(|(k, &core)| {
                let shared = shared.clone();
                thread::Builder::new().name(format!("aether-w{}", k + 1)).spawn(move || worker(shared, k + 1, n, core))
            })
            .collect::<std::io::Result<_>>()?;
        Ok(ThreadPool { shared, workers, cores: cores.to_vec() })
    }

    /// Threads taking part in every job, the caller included.
    pub fn threads(&self) -> usize {
        self.cores.len()
    }

    pub fn cores(&self) -> &[usize] {
        &self.cores
    }

    /// Runs `f(share, n)` once for every `share` in `0..n`, in parallel, and
    /// returns when all have finished. Share 0 runs on the calling thread.
    pub fn run<'a>(&self, f: &'a (dyn Fn(usize, usize) + Sync + 'a)) {
        let n = self.threads();
        if n == 1 {
            return f(0, 1);
        }
        let s = &*self.shared;
        assert!(
            s.busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok(),
            "ThreadPool::run called from two threads at once"
        );
        // SAFETY (lifetime erasure): workers only dereference `job` after
        // seeing the new epoch and before incrementing `done`, and this
        // function does not return until `done == n - 1`; so the reference
        // never outlives `f`.
        let job = unsafe { std::mem::transmute::<&'a (dyn Fn(usize, usize) + Sync + 'a), &'static Job>(f) };
        s.job.0.store(&job as *const &Job as *mut &Job, Ordering::Relaxed);
        s.done.0.store(0, Ordering::Relaxed);
        // Release: the job pointer and the reset counter become visible to
        // any worker that observes the new epoch with Acquire.
        s.epoch.0.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark(); // no system call unless that worker is asleep
        }
        f(0, n);
        let mut spins = 0u32;
        while s.done.0.load(Ordering::Acquire) != n - 1 {
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
        s.busy.store(false, Ordering::Release);
    }

    /// Splits `0..len` into one contiguous range per thread, each a multiple
    /// of `align` long (except the last), and runs `f(range)` on them in
    /// parallel. Ranges may be empty when `len` is small.
    pub fn for_ranges(&self, len: usize, align: usize, f: &(dyn Fn(std::ops::Range<usize>) + Sync)) {
        self.run(&|share, n| {
            let r = split(len, align, share, n);
            if !r.is_empty() {
                f(r);
            }
        });
    }
}

/// Share `share` of `n` of `0..len`, in blocks of `align`: blocks are dealt
/// out evenly, the first `blocks % n` shares getting one extra block.
pub fn split(len: usize, align: usize, share: usize, n: usize) -> std::ops::Range<usize> {
    let blocks = len.div_ceil(align);
    let (base, extra) = (blocks / n, blocks % n);
    let start_block = share * base + share.min(extra);
    let my_blocks = base + usize::from(share < extra);
    (start_block * align).min(len)..((start_block + my_blocks) * align).min(len)
}

fn worker(s: Arc<Shared>, share: usize, n: usize, core: usize) {
    // A failed pin (e.g. core offline) leaves the worker unpinned: slower,
    // still correct.
    let _ = affinity::pin_current_thread(&[core]);
    let mut seen = 0u64;
    loop {
        // Spin, then park, until the epoch moves or the pool shuts down.
        let mut spins = 0u32;
        let epoch = loop {
            let e = s.epoch.0.load(Ordering::Acquire);
            if e != seen || s.stop.0.load(Ordering::Acquire) {
                break e;
            }
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
            } else {
                // `unpark` before `park` is remembered, so a job published
                // between the check above and this call is not missed.
                thread::park();
            }
        };
        if s.stop.0.load(Ordering::Acquire) {
            return;
        }
        seen = epoch;
        // SAFETY: the epoch moved, so `job` points at the caller's `&Job`,
        // which stays alive until this worker increments `done` below.
        let f: &Job = unsafe { *s.job.0.load(Ordering::Relaxed) };
        f(share, n);
        s.done.0.fetch_add(1, Ordering::Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.stop.0.store(true, Ordering::Release);
        for w in self.workers.drain(..) {
            w.thread().unpark();
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cores(n: usize) -> Vec<usize> {
        affinity::online_cpus().unwrap().into_iter().take(n).collect()
    }

    #[test]
    fn split_covers_everything_once() {
        for (len, align, n) in [(100, 4, 3), (7, 4, 4), (0, 4, 2), (1000, 16, 8), (64, 16, 8)] {
            let mut seen = vec![0u8; len];
            for share in 0..n {
                let r = split(len, align, share, n);
                // Starts on a block boundary; only the range ending at `len`
                // may be shorter than a whole number of blocks.
                if !r.is_empty() {
                    assert_eq!(r.start % align, 0, "{len} {align} {n}");
                    assert!(r.len().is_multiple_of(align) || r.end == len, "{len} {align} {n}");
                }
                for i in r {
                    seen[i] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{len} {align} {n}");
        }
    }

    /// Every share runs exactly once per job, over many consecutive jobs
    /// (the epoch handshake must not skip or repeat one).
    #[test]
    fn every_share_runs_once_per_job() {
        // Run in a thread: the pool pins its caller.
        thread::spawn(|| {
            let pool = ThreadPool::new(&cores(4)).unwrap();
            let n = pool.threads();
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..2000 {
                pool.run(&|share, total| {
                    assert_eq!(total, n);
                    counts[share].fetch_add(1, Ordering::Relaxed);
                });
            }
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 2000));
        })
        .join()
        .unwrap();
    }

    /// Workers that went to sleep (park) wake up for the next job.
    #[test]
    fn parked_workers_wake_up() {
        thread::spawn(|| {
            let pool = ThreadPool::new(&cores(3)).unwrap();
            let hits = AtomicUsize::new(0);
            pool.run(&|_, _| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
            thread::sleep(std::time::Duration::from_millis(200)); // long enough to park
            pool.run(&|_, _| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(hits.load(Ordering::Relaxed), 2 * pool.threads());
        })
        .join()
        .unwrap();
    }
}
