//! Phase 1 sustained bandwidth and energy per GB.
//!
//! Reads a DRAM-sized buffer over and over for `--seconds`, printing the
//! bandwidth of every one-second window while the monitor records clocks,
//! temperatures and battery power. Two uses:
//!
//! - throttling: `--cores 0-7 --parallel --seconds 300` shows how far
//!   all-core bandwidth falls as the phone heats up;
//! - energy per GB: `--cores 7 --seconds 30 --idle-s 30` vs `--cores 0 ...`
//!   compares a prime and a performance core. `--idle-s` first measures the
//!   phone doing nothing for that long, so the report can separate the
//!   benchmark's energy from the phone's background draw. Run unplugged
//!   (adb over Wi-Fi); otherwise the battery current is charging current.
//!
//! Flags: --cores LIST (default 7), --parallel, --seconds N (default 30),
//! --size-mb N total working set (default 256), --idle-s N (default 0).
//! Output: per-second CSV, a `#` summary, then the monitor's samples as CSV
//! after the line `# monitor csv`.

use aether::aligned::AlignedBuf;
use aether::membench;
use aether::monitor::Monitor;
use aether::sys::affinity;
use std::hint::black_box;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn main() {
    let (mut cores, mut parallel, mut seconds, mut size_mb, mut idle_s) = (vec![7], false, 30u64, 256usize, 0u64);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--cores" => cores = affinity::parse_cpu_list(&val()).unwrap(),
            "--parallel" => parallel = true,
            "--seconds" => seconds = val().parse().unwrap(),
            "--size-mb" => size_mb = val().parse().unwrap(),
            "--idle-s" => idle_s = val().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    let threads: Vec<Vec<usize>> = if parallel { cores.iter().map(|&c| vec![c]).collect() } else { vec![cores.clone()] };
    let n = (size_mb << 20) / 4 / threads.len() / 16 * 16; // floats per thread
    let spare: Vec<usize> = affinity::online_cpus().unwrap().into_iter().filter(|c| !cores.contains(c)).collect();
    let pin = if spare.is_empty() { None } else { Some(&spare[..1]) };

    println!("# aether bw_sustained: read, cores {cores:?}, {} thread(s), {} MB total, {seconds} s", threads.len(), size_mb);

    // Idle baseline: same monitor, nothing running.
    let idle = (idle_s > 0).then(|| {
        let m = Monitor::start(Duration::from_millis(100), pin).unwrap();
        std::thread::sleep(Duration::from_secs(idle_s));
        m.stop()
    });

    let bytes = AtomicU64::new(0); // bytes read so far, all threads
    let stop = AtomicBool::new(false);
    let barrier = Barrier::new(threads.len() + 1);
    let monitor = Monitor::start(Duration::from_millis(100), pin).unwrap();
    println!("t_s,gbps");
    let start = {
        let (bytes, stop, barrier) = (&bytes, &stop, &barrier);
        std::thread::scope(|s| {
            for cs in &threads {
                s.spawn(move || {
                    affinity::pin_current_thread(cs).unwrap();
                    let mut b = AlignedBuf::zeros(n);
                    b.fill(1.0);
                    membench::read_sum(&b); // warm-up pass, touches every page
                    barrier.wait();
                    while !stop.load(Ordering::Relaxed) {
                        black_box(membench::read_sum(black_box(&b)));
                        bytes.fetch_add((n * 4) as u64, Ordering::Relaxed);
                    }
                });
            }
            barrier.wait();
            let start = Instant::now();
            // Reporter: bandwidth per one-second window, counted in whole passes.
            let mut last = (Duration::ZERO, 0u64);
            while start.elapsed() < Duration::from_secs(seconds) {
                std::thread::sleep(Duration::from_secs(1));
                let (t, b) = (start.elapsed(), bytes.load(Ordering::Relaxed));
                println!("{:.1},{:.2}", t.as_secs_f64(), (b - last.1) as f64 / (t - last.0).as_nanos() as f64);
                last = (t, b);
            }
            stop.store(true, Ordering::Relaxed);
            start
        })
    };
    let elapsed = start.elapsed();
    let report = monitor.stop();
    let total = bytes.load(Ordering::Relaxed) as f64;
    println!("# total {:.1} GB in {:.1} s = {:.2} GB/s average", total / 1e9, elapsed.as_secs_f64(), total / elapsed.as_nanos() as f64);
    println!("# {}", report.summary());
    // Energy per GB = average power / bandwidth. Power comes from whichever
    // battery source is better on this device (see Report::best_avg_power_w).
    let gbps = total / elapsed.as_nanos() as f64;
    if let Some((p, source)) = report.best_avg_power_w() {
        println!("# run power {p:.3} W ({source}) -> {:.3} J/GB total", p / gbps);
        if let Some((p_idle, _)) = idle.as_ref().and_then(|r| r.best_avg_power_w()) {
            println!("# idle baseline: {}", idle.as_ref().unwrap().summary());
            println!("# idle power {p_idle:.3} W -> {:.3} J/GB above idle", (p - p_idle) / gbps);
        }
    } else {
        println!("# no usable battery power reading (run longer: the charge counter must step twice)");
    }
    println!("# monitor csv");
    report.write_csv(std::io::stdout().lock()).unwrap();
}
