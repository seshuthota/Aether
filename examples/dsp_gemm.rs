//! L.5 step 5: the DSP's Q4_0 matrix-matrix product on HMX for a prompt
//! of many tokens, on Qwen2.5-0.5B's layer shapes (`--features hexagon`).
//! Times are the DSP's own; the weights are random blocks.
//!
//!     cargo run --release --target aarch64-linux-android --features hexagon --example dsp_gemm [tokens [threads]]

use aether::kernels::hexagon::Dsp;
use aether::quant::BlockQ4_0;

fn main() {
    let arg = |i: usize, default| std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let (tokens, threads) = (arg(1, 128), arg(2, 6) as u32);
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let d = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    let mut seed = 1u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // (name, rows, cols): the products of one layer (24 layers).
    let shapes = [("q/k/v", 1152, 896), ("wo", 896, 896), ("gate/up", 9728, 896), ("down", 896, 4864)];
    println!("{tokens} tokens, {threads} threads");
    println!("{:>10} {:>10} {:>10} {:>10}", "matrix", "us", "GFLOP/s", "us/token");
    let mut layer = 0.0;
    for (name, rows, cols) in shapes {
        let nb = cols / 32;
        let blocks: Vec<BlockQ4_0> = (0..rows * nb).map(|_| BlockQ4_0 { d: 0x1400, qs: std::array::from_fn(|_| next() as u8) }).collect();
        let w = d.q4_0_matrix(&blocks, rows).unwrap();
        let mut x = d.shared(4 * tokens * cols).unwrap();
        x.as_mut_slice::<f32>().iter_mut().for_each(|v| *v = (next() % 2000) as f32 / 1000.0 - 1.0);
        let mut out = d.shared(4 * tokens * rows.div_ceil(32) * 32).unwrap();
        d.hmx_gemm(&w, &x, &mut out, tokens, threads).unwrap();
        let mut us: Vec<u64> = (0..9).map(|_| d.hmx_gemm(&w, &x, &mut out, tokens, threads).unwrap()).collect();
        us.sort();
        let m = us[us.len() / 2] as f64;
        layer += m;
        println!("{name:>10} {m:>10.0} {:>10.1} {:>10.1}", 2.0 * (rows * cols * tokens) as f64 / m / 1e3, m / tokens as f64);
    }
    println!("24 layers: {:.1} ms for {tokens} tokens, {:.0} tok/s (products only)", 24.0 * layer / 1e3, tokens as f64 / (24.0 * layer / 1e6));
}
