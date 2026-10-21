pub mod aligned;
pub mod bench;
pub mod chat;
pub mod eval;
pub mod format;
pub mod kernels;
pub mod membench;
pub mod model;
pub mod monitor;
pub mod pool;
pub mod quant;
pub mod session;
pub mod sys;
pub mod tensor;
pub mod tokenizer;
pub mod weight;

#[cfg(test)]
pub(crate) mod testutil;

#[cfg(test)]
mod tests {
    #[test]
    fn prints_arch() {
        // Shows where the test is actually running.
        println!("running on: {}", std::env::consts::ARCH);
    }
}
