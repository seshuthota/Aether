//! K-quant decode GEMV benchmark (plans/kquant-decode.md, K0): one matmul
//! per distinct (format, shape) found in a K-quant GGUF, using the file's
//! real block bytes, next to the Q4_0 file's matrix at the same shape.
//!
//!   cargo run --release --target aarch64-linux-android --example kquant_bench -- \
//!       testdata/gguf/LFM3-1.7B-Q4_K_M.gguf
//!
//! Flags (all optional):
//!   --q4_0 FILE   the comparison file (default: the LFM3 QAD-Q4_0 file)
//!   --cores LIST  the single-thread arm's cores (default 7)
//!   --pool LIST   the multi-thread arm's cores (default 2-7)
//!   --only NAME   keep cases whose tensor name contains NAME
//!   --iters N     timed runs per case (default: ~4 GB of weights per case)
//!   --max-seq N   config only; the weights are what is measured (default 2048)
//!
//! GB/s = weight bytes / median time. ns/256 = median time divided by the
//! number of 256-weight super-blocks (rows × cols / 256), so shapes of
//! different sizes compare: it is the time to stream and dot one super-block
//! of one row, with the per-call activation quantization included.
//!
//! The ceiling to compare with is a pure read of the same bytes at the same
//! working-set size (`bw_sweep` / `bw_sustained`), and the Q4_0 line at the
//! same shape is the format the engine already decodes at the ceiling.

use aether::bench::bench;
use aether::model::load_any;
use aether::model::config::Config;
use aether::model::transformer::Weights;
use aether::pool::ThreadPool;
use aether::sys::affinity;
use aether::weight::Weight;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::path::PathBuf;

/// Columns of a matrix, from its llama.cpp tensor name. `None` for the
/// matrices this benchmark does not cover (the f32 depthwise kernel).
fn cols(name: &str, c: &Config) -> Option<usize> {
    Some(match name {
        "ffn_down" => c.hidden_dim,
        "ffn_gate" | "ffn_up" | "attn_q" | "attn_k" | "attn_v" | "attn_output" | "attn_gate" | "shortconv.in_proj" | "shortconv.out_proj" | "token_embd" | "output" => c.dim,
        _ => return None,
    })
}

/// Bytes one row of `w` occupies, for a row of `cols` values.
fn per_row_bytes(w: &Weight, cols: usize) -> usize {
    match w {
        Weight::F32(_) => cols * 4,
        Weight::Q8_0(_) | Weight::Q8_0x4(_) => cols / 32 * 34,
        Weight::Q4_0(_) | Weight::Q4_0x4(_) => cols / 32 * 18,
        Weight::Q5_0(_) | Weight::Q5_0x4(_) => cols / 32 * 22,
        Weight::Q4_K(_) => cols / 256 * 144,
        Weight::Q6_K(_) => cols / 256 * 210,
    }
}

fn rows(w: &Weight, cols: usize) -> usize {
    w.bytes() / per_row_bytes(w, cols)
}

struct Case<'a> {
    name: String,
    w: &'a Weight,
    rows: usize,
    cols: usize,
    format: &'static str,
}

#[allow(clippy::too_many_arguments)]
fn add<'a>(out: &mut Vec<Case<'a>>, seen: &mut HashSet<(&'static str, usize, usize)>, c: &Config, name: &str, w: &'a Weight, want: &dyn Fn(&Weight) -> bool) {
    if !want(w) {
        return;
    }
    let Some(cols) = cols(name, c) else { return };
    let rows = rows(w, cols);
    if seen.insert((w.format_name(), rows, cols)) {
        out.push(Case { name: name.to_string(), w, rows, cols, format: w.format_name() });
    }
}

fn cases<'a>(ws: &'a Weights, c: &Config, want: &dyn Fn(&Weight) -> bool) -> Vec<Case<'a>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    add(&mut out, &mut seen, c, "token_embd", &ws.token_embedding, want);
    if let Some(w) = &ws.classifier {
        add(&mut out, &mut seen, c, "output", w, want);
    }
    for l in &ws.layers {
        for (name, w) in l.matrices() {
            add(&mut out, &mut seen, c, name, w, want);
        }
    }
    out
}

/// One case's median and best GB/s, and ns per 256-weight super-block.
fn measure(c: &Case, x: &[f32], out: &mut [f32], pool: Option<&ThreadPool>, iters: Option<usize>) -> (f64, f64, f64, f64) {
    let bytes = c.w.bytes() as f64;
    let n = iters.unwrap_or_else(|| ((4e9 / bytes) as usize).clamp(5, 5000));
    let s = bench(3, n, || aether::kernels::linear_on(pool, black_box(out), black_box(x), black_box(c.w)));
    let blocks = (c.rows * c.cols / 256) as f64;
    (s.median / 1e3, bytes / s.median, bytes / s.min, s.median / blocks)
}

fn main() {
    let mut model = PathBuf::from("testdata/gguf/LFM3-1.7B-Q4_K_M.gguf");
    let mut q4 = PathBuf::from("testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf");
    let mut cores = vec![7];
    let mut pool_cores = vec![2, 3, 4, 5, 6, 7];
    let mut only = String::new();
    let mut iters = None;
    let mut max_seq = 2048;
    let mut positional = true;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--q4_0" => q4 = PathBuf::from(it.next().expect("--q4_0 FILE")),
            "--cores" => cores = affinity::parse_cpu_list(&it.next().expect("--cores LIST")).unwrap(),
            "--pool" => pool_cores = affinity::parse_cpu_list(&it.next().expect("--pool LIST")).unwrap(),
            "--only" => only = it.next().expect("--only NAME"),
            "--iters" => iters = Some(it.next().expect("--iters N").parse().expect("number")),
            "--max-seq" => max_seq = it.next().expect("--max-seq N").parse().expect("number"),
            other if positional && !other.starts_with('-') => {
                model = PathBuf::from(other);
                positional = false;
            }
            other => panic!("unknown flag {other}"),
        }
    }

    let (kc, kw) = load_any(&model, max_seq).expect("load the K-quant model");
    let (qc, qw) = load_any(&q4, max_seq).expect("load the Q4_0 model");
    let k_quants = |w: &Weight| matches!(w.format_name(), "q4_k" | "q6_k");
    let q4_0 = |w: &Weight| w.format_name() == "q4_0";
    let k: Vec<Case> = cases(&kw, &kc, &k_quants).into_iter().filter(|c| c.name.contains(&only)).collect();
    let q: Vec<Case> = cases(&qw, &qc, &q4_0).into_iter().filter(|c| c.name.contains(&only)).collect();
    let q_by_shape: HashMap<(usize, usize), &Case> = q.iter().map(|c| ((c.rows, c.cols), c)).collect();

    println!("# kquant_bench: model {}, q4_0 {}, cores {cores:?}, pool {pool_cores:?}", model.display(), q4.display());
    println!("# file,tensor,format,rows,cols,weight_bytes,threads,median_us,median_gbps,best_gbps,ns_per_256");
    let print = |file: &str, c: &Case, threads: usize, m: (f64, f64, f64, f64)| {
        println!("{file},{},{},{},{},{},{threads},{:.1},{:.2},{:.2},{:.2}", c.name, c.format, c.rows, c.cols, c.w.bytes(), m.0, m.1, m.2, m.3);
    };

    // Single thread, pinned to `cores`: the kernel's own cost.
    affinity::pin_current_thread(&cores).expect("pin the single-thread arm");
    let mut x = vec![0.0f32; 0];
    let mut out = Vec::new();
    let mut q4_printed = HashSet::new();
    for c in &k {
        x.resize(c.cols, 0.0);
        for (j, v) in x.iter_mut().enumerate() {
            *v = ((j % 13) as f32 - 6.0) * 0.01;
        }
        out.resize(c.rows, 0.0);
        let m = measure(c, &x, &mut out, None, iters);
        print("kquant", c, 1, m);
        if let Some(q) = q_by_shape.get(&(c.rows, c.cols)) {
            if q4_printed.insert((c.rows, c.cols)) {
                let m = measure(q, &x, &mut out, None, iters);
                print("q4_0", q, 1, m);
            }
        }
    }

    // The pool arm: the caller plus one worker per listed core, as a model
    // runs it (`ThreadPool::new` pins the caller to the first core).
    let pool = ThreadPool::new(&pool_cores).expect("create the thread pool");
    let mut q4_printed = HashSet::new();
    for c in &k {
        x.resize(c.cols, 0.0);
        for (j, v) in x.iter_mut().enumerate() {
            *v = ((j % 13) as f32 - 6.0) * 0.01;
        }
        out.resize(c.rows, 0.0);
        let m = measure(c, &x, &mut out, Some(&pool), iters);
        print("kquant", c, pool_cores.len(), m);
        if let Some(q) = q_by_shape.get(&(c.rows, c.cols)) {
            if q4_printed.insert((c.rows, c.cols)) {
                let m = measure(q, &x, &mut out, Some(&pool), iters);
                print("q4_0", q, pool_cores.len(), m);
            }
        }
    }
}
