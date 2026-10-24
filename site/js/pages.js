// The one list of pages. The side nav, the previous/next links, the
// breadcrumb, and the cards on the home page are all built from this.
// To add a page: copy _template.html, then add one line here.
window.AETHER_PAGES = [
  {
    section: "Start here",
    crumb: "Start here",
    pages: [
      { href: "index.html", title: "Home" },
      { href: "start/llm.html", num: "01", title: "How an LLM makes text", desc: "Weights, tokens, the forward pass, and why every new token re-reads the whole model." },
      { href: "start/roadmap.html", num: "02", title: "The roadmap", desc: "All nine phases on one page, and which part of the engine each one builds." },
      { href: "start/outline.html", num: "03", title: "Course outline", desc: "Every page still to write, phase by phase, and which of them the engine already has code and numbers for." },
      { href: "glossary.html", num: "04", title: "Glossary", desc: "Every term the notes use, in one or two plain sentences." }
    ]
  },
  {
    section: "Phase 0 · done",
    crumb: "Phase 0",
    pages: [
      { href: "phase-0/index.html", title: "Overview" },
      { href: "phase-0/cross-compilation.html", num: "01", title: "Cross-compilation", desc: "Host, target, the triple, and which half of the toolchain actually broke." },
      { href: "phase-0/android.html", num: "02", title: "The phone", desc: "adb, where a binary is allowed to run, and what this SoC reported." },
      { href: "phase-0/cargo.html", num: "03", title: "Cargo and the runner", desc: "Why cargo test is just an executable, and how the script puts it on the phone." },
      { href: "phase-0/measuring-time.html", num: "04", title: "Measuring time", desc: "The clock, the compiler, warm-up, and why the median is the headline." },
      { href: "phase-0/tensors.html", num: "05", title: "Tensors", desc: "A flat buffer, a shape, and strides. No arithmetic yet." },
      { href: "phase-0/decode-ceiling.html", num: "06", title: "Decode ceiling", desc: "Why bytes per weight set a hard cap, and why quantization is a bandwidth trick." }
    ]
  },
  {
    section: "Phase 1 · concepts",
    crumb: "Phase 1",
    pages: [
      { href: "phase-1/index.html", title: "Overview" },
      { href: "phase-1/memory.html", num: "01", title: "Memory hierarchy", desc: "Registers, caches, DRAM, and why a weight is not reused during decode." },
      { href: "phase-1/alignment.html", num: "02", title: "Alignment", desc: "Cache lines, split loads, and why Vec<f32> only promises 4 bytes." },
      { href: "phase-1/topology.html", num: "03", title: "Cores and pinning", desc: "The 6+2 core map already measured, and what sched_setaffinity is for." },
      { href: "phase-1/bandwidth.html", num: "04", title: "Bandwidth", desc: "Copy, scale, add, a pure read, and the working-set sweep." },
      { href: "phase-1/roofline.html", num: "05", title: "Roofline", desc: "Arithmetic intensity, and where the Phase 0 sum sits under the roof." },
      { href: "phase-1/power.html", num: "06", title: "Power and heat", desc: "Joules, the battery sysfs files, and why a 5-minute run is a different number." },
      { href: "phase-1/profiling.html", num: "07", title: "simpleperf", desc: "Why the release profile keeps symbols, and what a profile is for." }
    ]
  },
  {
    section: "Phase 1 · Rust tools",
    crumb: "Phase 1",
    pages: [
      { href: "phase-1/rust-memory.html", num: "08", title: "Raw memory in Rust", desc: "Layout, alloc and dealloc, what unsafe means, and why the allocator needs its own type." },
      { href: "phase-1/rust-pinning.html", num: "09", title: "Calling Linux from Rust", desc: "FFI, the libc crate, CPU sets as bitmasks, and sched_setaffinity." },
      { href: "phase-1/rust-threads.html", num: "10", title: "Threads in Rust", desc: "spawn and join, a stop flag for the sampler, and scoped threads for the parallel sweep." },
      { href: "phase-1/rust-sysfs.html", num: "11", title: "Reading /sys from Rust", desc: "Turning a sysfs file into a number with the right unit, and what each failure means." },
      { href: "phase-1/plotting.html", num: "12", title: "Plotting the sweep", desc: "A CSV with its conditions, a log size axis, and the matplotlib script." }
    ]
  },
  {
    section: "Phase 1 · results",
    crumb: "Phase 1",
    pages: [
      { href: "phase-1/results.html", num: "13", title: "What Phase 1 measured", desc: "The yardstick table: 102 GB/s in L1, 61 from DRAM on one core, 67 on all, and joules per gigabyte." }
    ]
  },
  {
    section: "Phase 2 · a whole model",
    crumb: "Phase 2",
    pages: [
      { href: "phase-2/index.html", title: "Overview" },
      { href: "phase-2/model-file.html", num: "01", title: "Inside a model file", desc: "A 28-byte header, then 15.2 million floats in a fixed order. Where every matrix lives." },
      { href: "phase-2/tokenizer.html", num: "02", title: "Text to tokens", desc: "The vocabulary, scores, BPE merges step by step, and byte fallback for anything unknown." },
      { href: "phase-2/embedding-norm.html", num: "03", title: "Embedding and RMSNorm", desc: "A token id picks a row; RMSNorm resets the vector's size before every block." },
      { href: "phase-2/qkv.html", num: "04", title: "Queries, keys, values, heads", desc: "Three matrix multiplies per token, and why one vector is split into six heads." },
      { href: "phase-2/rope.html", num: "05", title: "RoPE: position as rotation", desc: "Rotating pairs of numbers by position so attention can tell how far apart tokens are." },
      { href: "phase-2/attention.html", num: "06", title: "Attention, step by step", desc: "Scores, scaling, softmax, and the weighted sum of values. A toy you can steer." },
      { href: "phase-2/kv-cache.html", num: "07", title: "The KV cache", desc: "Why earlier keys and values are kept, how big that gets, and how it is laid out." },
      { href: "phase-2/mlp.html", num: "08", title: "The MLP and the residual stream", desc: "Gate, up, down, the SiLU curve, and the one vector every block adds into." },
      { href: "phase-2/sampling.html", num: "09", title: "Choosing the next token", desc: "Logits, greedy, temperature, top-k, top-p, and a random generator that can be replayed." },
      { href: "phase-2/forward.html", num: "10", title: "The forward pass in code", desc: "All of the above in one function, line by line, with the work each line costs." },
      { href: "phase-2/correctness.html", num: "11", title: "Bit-identical with llama2.c", desc: "Golden dumps of every activation, and why matching bit for bit means matching the order of additions." },
      { href: "phase-2/baseline.html", num: "12", title: "First speed: 88.6 tok/s", desc: "Where the time goes, why it is not memory yet, and the dependency chain Phase 3 removes." }
    ]
  },
  {
    section: "Phase 3 · NEON kernels",
    crumb: "Phase 3",
    pages: [
      { href: "phase-3/index.html", title: "Overview" },
      { href: "phase-3/simd.html", num: "01", title: "SIMD and NEON", desc: "Four floats per instruction, the registers, and the intrinsics Rust gives you." },
      { href: "phase-3/chains.html", num: "02", title: "Dependency chains", desc: "Latency vs throughput, and why four accumulators beat one even with the same instructions." },
      { href: "phase-3/gemv.html", num: "03", title: "GEMV in three steps", desc: "5 → 18 → 42 → 54 GB/s on one core, measured, and the case where the last step loses." },
      { href: "phase-3/tails.html", num: "04", title: "Tails and testing floats", desc: "Lengths that are not multiples of 4, and checking against numpy with a tolerance." },
      { href: "phase-3/prefill.html", num: "05", title: "Prefill: many tokens at once", desc: "The same weights reused by every prompt token, and the batched forward pass." },
      { href: "phase-3/gemm.html", num: "06", title: "Inside the GEMM micro-kernel", desc: "A 4 × 4 tile: 8 loads feed 16 FMAs, kept in 24 of 32 registers." },
      { href: "phase-3/dispatch.html", num: "07", title: "One API, two machines", desc: "NEON on the phone, scalar on the PC, feature detection, and when inline assembly is needed." }
    ]
  },
  {
    section: "Phase 4 · threads",
    crumb: "Phase 4",
    pages: [
      { href: "phase-4/index.html", title: "Overview" },
      { href: "phase-4/pool.html", num: "01", title: "Threads that stay", desc: "241 jobs per token, and why handing one out must cost well under a microsecond." },
      { href: "phase-4/split.html", num: "02", title: "Splitting the work", desc: "Rows, not pieces of rows; why equal shares made 8 cores slower than 2; blocks from a counter." },
      { href: "phase-4/waiting.html", num: "03", title: "Waiting: spin, then park", desc: "Reacting fast vs leaving the core free, the lost wake-up, and one counter per cache line." },
      { href: "phase-4/cores.html", num: "04", title: "Choosing cores", desc: "Prime and performance cores, the caller on the fastest, and what each core set measured." }
    ]
  },
  {
    section: "Phase 5 · real model files",
    crumb: "Phase 5",
    pages: [
      { href: "phase-5/index.html", title: "Overview" }
    ]
  },
  {
    section: "Phase 6 · quantization",
    crumb: "Phase 6",
    pages: [
      { href: "phase-6/index.html", title: "Overview" },
      { href: "phase-6/blocks.html", num: "01", title: "Block quantization", desc: "One scale per 32 weights: Q8_0 in 34 bytes, Q4_0 in 18, and what an outlier does to its block." },
      { href: "phase-6/dot-products.html", num: "02", title: "Multiplying quantized weights", desc: "Integer dot products with sdot, then smmla: two rows times two tokens per instruction, measured." },
      { href: "phase-6/perplexity.html", num: "03", title: "Perplexity and KL", desc: "How surprised the model is by real text, and how to tell whether quantization hurt it." }
    ]
  },
  {
    section: "Phase 7 · GGUF and Qwen",
    crumb: "Phase 7",
    pages: [
      { href: "phase-7/index.html", title: "Overview" },
      { href: "phase-7/gguf.html", num: "01", title: "GGUF: one file for everything", desc: "Header, typed metadata, the tensor table, alignment, and weights used straight from the file." },
      { href: "phase-7/attention-ab.html", num: "02", title: "NEON attention, measured honestly", desc: "A speed-up a single run got wrong, and the alternating A/B that got it right." }
    ]
  },
  {
    section: "Phase 8 · the app",
    crumb: "Phase 8",
    pages: [
      { href: "phase-8/index.html", title: "Overview" }
    ]
  }
];
