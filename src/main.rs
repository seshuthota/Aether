use aether::tensor::Tensor;

fn main() {
    // A weight-matrix-sized tensor: 4096 x 4096 f32 values.
    let t = Tensor::zeros(&[4096, 4096]);
    println!(
        "shape {:?}, {} elements, {} MiB",
        t.shape(),
        t.numel(),
        t.size_bytes() / (1024 * 1024)
    );
}
