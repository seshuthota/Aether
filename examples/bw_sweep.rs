//! Phase 1 working-set sweep: memory bandwidth vs buffer size, per core set.
//!
//!   cargo run --release --target aarch64-linux-android --example bw_sweep -- \
//!       --kernel read --cores 7 > results/raw/sweep-read-core7.csv
//!
//! Flags (all optional):
//!   --kernel read|copy|scale|add   default read
//!   --cores LIST                   cores to use, kernel syntax "0-5,7"; default 7
//!   --parallel                     one thread per listed core, the working set
//!                                  split between them; otherwise one thread
//!                                  allowed on the listed cores
//!   --min-kb N   --max-mb N        sweep range, default 4 KB .. 256 MB
//!   --steps N                      sizes per doubling, default 4
//!   --target-mb N                  bytes moved per timed sample, default 256
//!   --samples N                    timed samples per size, default 7
//!   --offset N                     shift every array start by N floats
//!                                  (1 = 4 bytes off the 64-byte alignment)
//!
//! Output: CSV on stdout. Lines starting with `#` record the conditions
//! (cores, clocks, temperature, battery) at the start and the monitor
//! summary at the end. GB/s is decimal (1e9 bytes per second), bytes counted
//! by the STREAM rule (see `membench`).

use aether::aligned::AlignedBuf;
use aether::membench::{self, Kernel};
use aether::monitor::Monitor;
use aether::sys::{affinity, sysfs};
use std::hint::black_box;
use std::sync::Barrier;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Args {
    kernel: Kernel,
    cores: Vec<usize>,
    parallel: bool,
    min_bytes: usize,
    max_bytes: usize,
    steps: usize,
    target_bytes: usize,
    samples: usize,
    offset: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        kernel: Kernel::Read,
        cores: vec![7],
        parallel: false,
        min_bytes: 4 << 10,
        max_bytes: 256 << 20,
        steps: 4,
        target_bytes: 256 << 20,
        samples: 7,
        offset: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        let mut num = || val().parse::<usize>().unwrap_or_else(|_| panic!("bad number for {flag}"));
        match flag.as_str() {
            "--kernel" => a.kernel = Kernel::parse(&val()).unwrap_or_else(|| panic!("unknown kernel")),
            "--cores" => a.cores = affinity::parse_cpu_list(&val()).unwrap(),
            "--parallel" => a.parallel = true,
            "--min-kb" => a.min_bytes = num() << 10,
            "--max-mb" => a.max_bytes = num() << 20,
            "--steps" => a.steps = num(),
            "--target-mb" => a.target_bytes = num() << 20,
            "--samples" => a.samples = num(),
            "--offset" => a.offset = num(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

/// Working-set sizes from min to max, `steps` per doubling (geometric).
fn sizes(a: &Args) -> Vec<usize> {
    let ratio = 2f64.powf(1.0 / a.steps as f64);
    let mut out = Vec::new();
    let mut s = a.min_bytes as f64;
    while s <= a.max_bytes as f64 * 1.0001 {
        out.push(s.round() as usize);
        s *= ratio;
    }
    out
}

/// One thread's arrays for one working-set size: `arrays` buffers of `n`
/// floats each, shifted by `offset` floats, filled with 1.0 (read_sum's
/// result then doubles as a check that every element was read).
struct Arrays {
    bufs: Vec<AlignedBuf>,
    n: usize,
    offset: usize,
}

impl Arrays {
    fn new(kernel: Kernel, n: usize, offset: usize) -> Arrays {
        let bufs = (0..kernel.arrays())
            .map(|_| {
                let mut b = AlignedBuf::zeros(n + offset);
                b.fill(1.0);
                b
            })
            .collect();
        Arrays { bufs, n, offset }
    }

    /// One pass of the kernel over this thread's arrays.
    fn pass(&mut self, kernel: Kernel) {
        let (o, n) = (self.offset, self.n);
        match kernel {
            Kernel::Read => {
                black_box(membench::read_sum(black_box(&self.bufs[0][o..o + n])));
            }
            Kernel::Copy | Kernel::Scale => {
                let (src, dst) = self.bufs.split_at_mut(1);
                let (src, dst) = (&src[0][o..o + n], &mut dst[0][o..o + n]);
                if kernel == Kernel::Copy {
                    membench::copy(dst, black_box(src));
                } else {
                    membench::scale(dst, black_box(src), black_box(1.0001));
                }
                black_box(dst);
            }
            Kernel::Add => {
                let (ab, c) = self.bufs.split_at_mut(2);
                let dst = &mut c[0][o..o + n];
                membench::add(dst, black_box(&ab[0][o..o + n]), black_box(&ab[1][o..o + n]));
                black_box(dst);
            }
        }
    }
}

/// Measures one working-set size. Returns the wall time of each sample.
///
/// Every thread pins itself, allocates and touches its own arrays, warms up,
/// then all threads start each sample together at a barrier. The leader
/// times from that barrier to the next one, so a sample lasts as long as the
/// slowest thread: aggregate bandwidth is total bytes over wall time.
fn measure(a: &Args, threads: &[Vec<usize>], n: usize, reps: usize) -> Vec<Duration> {
    let barrier = Barrier::new(threads.len());
    let mut times = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = threads
            .iter()
            .enumerate()
            .map(|(t, cores)| {
                let barrier = &barrier;
                s.spawn(move || {
                    affinity::pin_current_thread(cores).expect("pin failed");
                    let mut arr = Arrays::new(a.kernel, n, a.offset);
                    if a.kernel == Kernel::Read {
                        let sum = membench::read_sum(&arr.bufs[0][a.offset..a.offset + n]);
                        assert_eq!(sum, n as f32, "read_sum missed elements");
                    }
                    for _ in 0..reps.clamp(1, 3) {
                        arr.pass(a.kernel); // warm-up: caches, TLB, clocks
                    }
                    let mut mine = Vec::new();
                    for _ in 0..a.samples {
                        barrier.wait();
                        let start = Instant::now();
                        for _ in 0..reps {
                            arr.pass(a.kernel);
                        }
                        barrier.wait();
                        if t == 0 {
                            mine.push(start.elapsed());
                        }
                    }
                    mine
                })
            })
            .collect();
        for (t, h) in handles.into_iter().enumerate() {
            let mine = h.join().unwrap();
            if t == 0 {
                times = mine;
            }
        }
    });
    times
}

fn print_conditions(a: &Args, threads: &[Vec<usize>]) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    println!("# aether bw_sweep, unix time {now}");
    println!(
        "# kernel {} | cores {:?} | {} thread(s) | offset {} floats | {} samples, ~{} MB per sample",
        a.kernel.name(),
        a.cores,
        threads.len(),
        a.offset,
        a.samples,
        a.target_bytes >> 20
    );
    #[cfg(target_arch = "aarch64")]
    println!("# CTR_EL0 cache line: {} bytes", aether::sys::cpu::dcache_line_bytes());
    for c in affinity::online_cpus().unwrap_or_default() {
        println!(
            "# cpu{c}: cur {} kHz, cap {} kHz, max {} kHz",
            sysfs::cpu_cur_freq_khz(c).map_or("?".into(), |v| v.to_string()),
            sysfs::cpu_scaling_max_freq_khz(c).map_or("?".into(), |v| v.to_string()),
            sysfs::cpu_max_freq_khz(c).map_or("?".into(), |v| v.to_string()),
        );
    }
    let temps: Vec<i64> = sysfs::thermal_zones().iter().filter_map(|z| z.temp_millic().ok()).collect();
    if let Some(max) = temps.iter().max() {
        println!("# start: max thermal zone {:.1} °C over {} zones", *max as f64 / 1000.0, temps.len());
    }
    if let Some(b) = sysfs::Battery::find() {
        println!("# battery: {}", b.status().unwrap_or_else(|e| e.to_string()));
    }
}

fn main() {
    let a = parse_args();
    // Parallel: one single-core thread per listed core. Otherwise one thread
    // that may run on any listed core (normally a single core is listed).
    let threads: Vec<Vec<usize>> = if a.parallel { a.cores.iter().map(|&c| vec![c]).collect() } else { vec![a.cores.clone()] };

    // Sample on a core the benchmark does not use, if there is one.
    let online = affinity::online_cpus().unwrap();
    let spare: Vec<usize> = online.iter().copied().filter(|c| !a.cores.contains(c)).collect();
    let monitor = Monitor::start(Duration::from_millis(100), if spare.is_empty() { None } else { Some(&spare[..1]) }).unwrap();

    print_conditions(&a, &threads);
    println!("kernel,threads,working_set_bytes,reps,median_gbps,best_gbps,worst_gbps");
    for ws in sizes(&a) {
        // Floats per array per thread, rounded down to whole 64-byte lines.
        let n = (ws / (a.kernel.arrays() * threads.len() * 4)) / 16 * 16;
        if n == 0 {
            continue;
        }
        let bytes_per_pass = a.kernel.bytes_per_pass(n) * threads.len();
        let reps = a.target_bytes.div_ceil(bytes_per_pass).max(1);
        let times = measure(&a, &threads, n, reps);
        let stats = aether::bench::Stats::from_durations(&times);
        let gbps = |ns: f64| (bytes_per_pass * reps) as f64 / ns; // bytes/ns = GB/s
        println!(
            "{},{},{},{},{:.2},{:.2},{:.2}",
            a.kernel.name(),
            threads.len(),
            a.kernel.arrays() * threads.len() * n * 4,
            reps,
            gbps(stats.median),
            gbps(stats.min),
            gbps(stats.max)
        );
    }
    println!("# {}", monitor.stop().summary());
}
