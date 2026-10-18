// examples/bench_sum.rs
use aether::bench::bench;
use std::hint::black_box;

fn main() {
    let data: Vec<f32> = (0..4_000_000).map(|i| (i % 10) as f32).collect();
    let s = bench(0, 30, || black_box(&data).iter().sum::<f32>());
    println!("{s:#?}");
}