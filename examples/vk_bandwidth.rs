//! GPU memory bandwidth on the phone (the GPU side of Phase 1's
//! `bw_sweep`): what the Adreno GPU says about itself, then how fast a
//! compute shader can read, and copy, buffers of growing size.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_bandwidth
//!
//! Times are the GPU's own timestamps around back-to-back dispatches, so
//! they leave out submitting and waiting. A copy moves each byte twice
//! (read + write) and is counted that way. The GPU's clock is not readable
//! without root, so each size first runs ~0.3 s untimed (the governor
//! raises the clock under load), then 5 timed trials: median and best.

use aether::kernels::vulkan::{Buffer, Gpu, Kernel};
use aether::spirv;
use ash::vk::MemoryPropertyFlags as M;

/// GB/s of `bytes_per_rep` moved per dispatch: median and best of 5.
fn measure(gpu: &Gpu, k: &Kernel, bufs: &[&Buffer], n4: u32, groups: u32, bytes_per_rep: f64) -> (f64, f64) {
    let push = n4.to_le_bytes();
    // Reps per trial: about 20 ms at 50 GB/s.
    let reps = ((1e9 / bytes_per_rep) as u32).clamp(2, 2000);
    let mut warm = 0.0;
    while warm < 0.3 {
        warm += gpu.run(k, bufs, &push, [groups, 1, 1], reps).unwrap();
    }
    let mut r: Vec<f64> = (0..5).map(|_| bytes_per_rep * reps as f64 / gpu.run(k, bufs, &push, [groups, 1, 1], reps).unwrap() / 1e9).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (r[2], r[4])
}

fn main() {
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    let i = &gpu.info;
    println!("# {} | Vulkan {}.{}.{} | driver {:#x}", i.name, i.api_version.0, i.api_version.1, i.api_version.2, i.driver_version);
    println!("# subgroup operations: {:?}", i.subgroup_ops);
    println!("# subgroup {} | max workgroup {} | max storage buffer {} MB | timestamp {} ns", i.subgroup_size, i.max_workgroup_invocations, i.max_storage_buffer_range >> 20, i.timestamp_period);
    for (n, (size, flags)) in i.heaps.iter().enumerate() {
        println!("# heap {n}: {:.0} MB {flags:?}", *size as f64 / (1 << 20) as f64);
    }
    for (n, (heap, flags)) in i.types.iter().enumerate() {
        println!("# memory type {n}: heap {heap} {flags:?}");
    }

    let read = Kernel::new(&gpu, spirv!("read"), 2, 4).unwrap();
    let copy = Kernel::new(&gpu, spirv!("copy"), 2, 4).unwrap();
    let groups = std::env::args().nth(1).map_or(512, |g| g.parse().unwrap());
    let sums = gpu.buffer(groups as usize * 256 * 16).unwrap();
    println!("# {groups} workgroups of 256; storage buffers may use memory types {:#b}", sums.allowed_types);
    // The fixed cost of one dispatch: an empty one (n4 = 0, one workgroup),
    // back to back with a barrier between, as a model's layers would run.
    let mut warm = 0.0;
    while warm < 0.2 {
        warm += gpu.run(&read, &[&sums, &sums], &0u32.to_le_bytes(), [1, 1, 1], 1000).unwrap();
    }
    for g in [1, groups] {
        let t = gpu.run(&read, &[&sums, &sums], &0u32.to_le_bytes(), [g, 1, 1], 1000).unwrap();
        println!("# empty dispatch + barrier, {g} workgroups: {:.1} us", t / 1000.0 * 1e6);
    }
    println!("test,memory_type,MB,median_GB_per_s,best_GB_per_s");
    let mut seen = Vec::new();
    for flags in [M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT, M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_CACHED] {
        for mb in [1usize, 4, 16, 64, 256] {
            let bytes = mb << 20;
            let mut a = gpu.buffer_with(bytes, flags).unwrap();
            if mb == 1 {
                if seen.contains(&a.mem_type) {
                    break; // the same memory type as an earlier row set
                }
                seen.push(a.mem_type);
            }
            for (k, v) in a.as_mut_slice::<f32>().iter_mut().enumerate() {
                *v = (k % 1000) as f32;
            }
            let mut b = gpu.buffer_with(bytes, flags).unwrap();
            let n4 = (bytes / 16) as u32;
            let (med, best) = measure(&gpu, &read, &[&a, &sums], n4, groups, bytes as f64);
            println!("read,{},{mb},{med:.1},{best:.1}", a.mem_type);
            b.as_mut_slice::<f32>().fill(-1.0);
            let (med, best) = measure(&gpu, &copy, &[&a, &b], n4, groups, 2.0 * bytes as f64);
            assert!(a.as_slice::<f32>() == b.as_slice::<f32>(), "copy differs from its input");
            println!("copy,{},{mb},{med:.1},{best:.1}", a.mem_type);
        }
    }
}
