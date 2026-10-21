pub mod aligned;
pub mod bench;
pub mod membench;
pub mod monitor;
pub mod sys;
pub mod tensor;

/// Placeholder so the library crate exists. The Tensor goes here later.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    fn prints_arch() {
        // Shows where the test is actually running.
        println!("running on: {}", std::env::consts::ARCH);
    }
}
