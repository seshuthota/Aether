//! What it costs to hand one (empty) job to the other threads, three ways:
//! spawn and join fresh threads, wake persistent workers with a mutex and a
//! condition variable, and `ThreadPool::run` (spin, then park). Decode hands
//! out a few hundred jobs per token, so this is the floor under every one.
//!
//!   cargo run --release --target aarch64-linux-android --example pool_handoff -- 4-7
use aether::pool::ThreadPool;
use aether::sys::affinity;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Best of 5 runs of `n` calls, in microseconds per call.
fn per_call(n: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..n / 10 {
        f();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / n as f64);
    }
    best * 1e6
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "0-3".into());
    let cores = affinity::parse_cpu_list(&arg).unwrap();
    assert!(cores.len() >= 2, "give at least two cores");
    let others = cores.len() - 1;
    affinity::pin_current_thread(&cores[..1]).unwrap();

    let spawn = per_call(2_000, || {
        let hs: Vec<_> = cores[1..]
            .iter()
            .map(|&c| {
                std::thread::spawn(move || {
                    let _ = affinity::pin_current_thread(&[c]);
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    });

    // (job number, workers finished), "new job" and "a worker finished".
    let st = Arc::new((Mutex::new((0u64, 0usize)), Condvar::new(), Condvar::new()));
    for &c in &cores[1..] {
        let st = st.clone();
        std::thread::spawn(move || {
            let _ = affinity::pin_current_thread(&[c]);
            let (m, go, done) = &*st;
            let mut seen = 0;
            loop {
                let mut g = m.lock().unwrap();
                while g.0 == seen {
                    g = go.wait(g).unwrap();
                }
                seen = g.0;
                g.1 += 1;
                done.notify_one();
            }
        });
    }
    let condvar = per_call(20_000, || {
        let (m, go, done) = &*st;
        let mut g = m.lock().unwrap();
        g.0 += 1;
        g.1 = 0;
        go.notify_all();
        while g.1 != others {
            g = done.wait(g).unwrap();
        }
    });

    let pool = ThreadPool::new(&cores).unwrap();
    let run = per_call(200_000, || pool.run(&|share, _| _ = std::hint::black_box(share)));

    println!("# pool_handoff cores {cores:?} (caller + {others} others), best of 5, us per job");
    println!("spawn+join {spawn:.2} | condvar {condvar:.2} | pool.run {run:.3}");
}
