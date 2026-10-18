// Throwaway experiments for Concept 4 (measuring time). Not part of the engine.
use std::hint::black_box;
use std::time::Instant;

// A sequential f32 sum: the compiler may not reorder float adds, so this stays honest.
#[inline(never)]
fn sum(data: &[f32]) -> f32 {
    let mut s = 0.0;
    for &x in data {
        s += x;
    }
    s
}

fn main() {
    // --- Experiment A: how much does reading the clock itself cost?
    let n = 1_000_000;
    let t = Instant::now();
    for _ in 0..n {
        black_box(Instant::now());
    }
    let per_call = t.elapsed().as_nanos() as f64 / n as f64;
    println!("A. Instant::now() costs about {per_call:.0} ns per call\n");

    let data: Vec<f32> = (0..4_000_000).map(|i| (i % 10) as f32).collect(); // 16 MB

    // --- Experiment B: work whose result is never used
    let t = Instant::now();
    for _ in 0..10 {
        let _ = sum(&data); // result thrown away
    }
    let unused = t.elapsed();

    let t = Instant::now();
    for _ in 0..10 {
        black_box(sum(black_box(&data))); // result "used"
    }
    let used = t.elapsed();
    println!("B. 10 sums over 4M floats:");
    println!("   result unused:        {:>9.1} us total", unused.as_secs_f64() * 1e6);
    println!("   result in black_box:  {:>9.1} us total\n", used.as_secs_f64() * 1e6);

    // --- Experiment C: the same work, run 30 times in a row
    println!("C. Same work 30 times (microseconds each):");
    let mut times = Vec::new();
    for i in 0..30 {
        let t = Instant::now();
        black_box(sum(black_box(&data)));
        let us = t.elapsed().as_secs_f64() * 1e6;
        times.push(us);
        print!("{us:7.0}{}", if i % 10 == 9 { "\n" } else { " " });
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "   min {:.0}  median {:.0}  mean {:.0}  max {:.0}",
        sorted[0], sorted[sorted.len() / 2], mean, sorted[sorted.len() - 1]
    );
}
