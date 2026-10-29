//! L.5 step 3: the DSP's Q4_0 matrix-vector product on Qwen2.5-0.5B's
//! shapes, by number of DSP threads (`--features hexagon`). Times are the
//! DSP's own (thread start to join), so the ~146 us call overhead is not in
//! them; the weights are random blocks.
//!
//!     cargo run --release --target aarch64-linux-android --features hexagon --example dsp_gemv [prefetch]

use aether::kernels::hexagon::Dsp;
use aether::quant::{BlockQ4_0, BlockQ8_0};

fn main() {
    let prefetch = std::env::args().any(|a| a == "prefetch");
    println!("l2fetch: {prefetch}");
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let d = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    let mut seed = 1u64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u8
    };
    // (name, rows, cols, times per token in the model: 24 layers)
    let shapes = [("q/k/v", 1152, 896, 24), ("wo", 896, 896, 24), ("gate/up", 9728, 896, 24), ("down", 896, 4864, 24), ("classifier", 151936, 896, 1)];
    let threads = [1, 2, 3, 4, 5, 6];
    print!("{:>10} {:>12}", "matrix", "MB");
    threads.iter().for_each(|t| print!(" {:>12}", format!("{t} thr us")));
    println!("   best GB/s");
    let mut per_token = vec![0.0f64; threads.len()];
    for (name, rows, cols, times) in shapes {
        let nb = cols / 32;
        // Small scales, so the f32 sums stay finite.
        let blocks: Vec<BlockQ4_0> = (0..rows * nb).map(|_| BlockQ4_0 { d: 0x1400, qs: std::array::from_fn(|_| byte()) }).collect();
        let w = d.q4_0_matrix(&blocks, rows).unwrap();
        let x: Vec<BlockQ8_0> = (0..nb).map(|_| BlockQ8_0 { d: 0x2000, qs: std::array::from_fn(|_| byte() as i8) }).collect();
        let mut xbuf = d.q8_0_vector(nb).unwrap();
        let mut out = d.shared(4 * rows.div_ceil(32) * 32).unwrap();
        print!("{name:>10} {:>12.1}", w.bytes() as f64 / 1e6);
        let mut best = f64::MAX;
        for (i, &t) in threads.iter().enumerate() {
            d.q4_0_gemv(&w, &x, &mut xbuf, &mut out, t, prefetch).unwrap();
            let mut us: Vec<u64> = (0..15).map(|_| d.q4_0_gemv(&w, &x, &mut xbuf, &mut out, t, prefetch).unwrap()).collect();
            us.sort();
            let m = us[us.len() / 2] as f64;
            best = best.min(m);
            per_token[i] += m * times as f64;
            print!(" {:>12.0}", m);
        }
        println!("   {:>8.1}", w.bytes() as f64 / best / 1e3);
    }
    print!("{:>10} {:>12}", "per token", "ms");
    per_token.iter().for_each(|us| print!(" {:>12.2}", us / 1e3));
    println!();
}
