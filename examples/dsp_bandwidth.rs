//! L.5 step 2: how fast the Hexagon DSP reads memory shared with the CPU,
//! by buffer size and number of DSP threads (`--features hexagon`). Times
//! come from the DSP's own clock, so the ~146 us call overhead is not in
//! them.
//!
//!     cargo run --release --target aarch64-linux-android --features hexagon --example dsp_bandwidth

use aether::kernels::hexagon::Dsp;

fn main() {
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let d = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    println!("HVX contexts: {}", d.hvx_units().unwrap());
    // 256 KB fits the DSP's L2; 256 MB is about the size of Qwen2.5-0.5B
    // in Q4_0 and comes from DRAM.
    let max = 256 << 20;
    let mut buf = d.shared(max).unwrap();
    for (i, w) in buf.as_mut_slice::<u32>().iter_mut().enumerate() {
        *w = i as u32;
    }
    println!("{:>9} {:>7} {:>9} {:>14}", "bytes", "threads", "GB/s", "with l2fetch");
    for bytes in [256 << 10, 4 << 20, 64 << 20, 256 << 20] {
        for threads in [1, 2, 3, 4, 5, 6, 8] {
            // Each thread reads whole groups of four vectors.
            let bytes = bytes / (threads as usize * 512) * (threads as usize * 512);
            // About 0.5 s of reading at 10 GB/s, at least 2 reps.
            let reps = ((5 << 30) / bytes).max(2) as u32;
            let gbs = |prefetch| {
                d.read_bw(&buf, bytes, threads, 1, prefetch).unwrap(); // warm-up
                let (us, _) = d.read_bw(&buf, bytes, threads, reps, prefetch).unwrap();
                bytes as f64 * reps as f64 / (us as f64 / 1e6) / 1e9
            };
            println!("{:>9} {:>7} {:>9.2} {:>14.2}", bytes, threads, gbs(false), gbs(true));
        }
    }
}
