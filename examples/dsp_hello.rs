//! L.5 step 1: our library on the Hexagon DSP. Opens it, checks a ping and
//! an HVX add, and times round trips (`--features hexagon`).
//!
//!     cargo run --release --target aarch64-linux-android --features hexagon --example dsp_hello

use aether::kernels::hexagon::Dsp;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let t = Instant::now();
    let d = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    println!("session open + library load: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(d.ping(41).unwrap(), 42);

    // A call with no work: what FastRPC itself costs per round trip.
    for _ in 0..100 {
        d.ping(1).unwrap();
    }
    let n = 2000;
    let mut us: Vec<f64> = (0..n)
        .map(|i| {
            let t = Instant::now();
            black_box(d.ping(black_box(i)).unwrap());
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(f64::total_cmp);
    println!("ping round trip: median {:.1} us, p10 {:.1}, p90 {:.1} ({n} calls)", us[n as usize / 2], us[n as usize / 10], us[9 * n as usize / 10]);

    // The add with its buffers copied both ways, by size.
    for len in [1 << 10, 1 << 14, 1 << 18, 1 << 20] {
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b = vec![0.5f32; len];
        let mut c = vec![0.0f32; len];
        d.add(&a, &b, &mut c).unwrap();
        assert!(c.iter().enumerate().all(|(i, &v)| v == i as f32 + 0.5));
        let reps = 50;
        let t = Instant::now();
        for _ in 0..reps {
            d.add(black_box(&a), black_box(&b), &mut c).unwrap();
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!("add {len:>8} floats: {:8.1} us per call, {:5.2} GB/s moved (3 arrays)", s * 1e6, 12.0 * len as f64 / s / 1e9);
    }
}
