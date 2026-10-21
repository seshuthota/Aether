//! Pinning threads to cores.
//!
//! The scheduler normally moves a thread between cores whenever it likes. On
//! this phone that matters twice over: cores 6–7 (prime, 4.32 GHz) and 0–5
//! (performance, 3.53 GHz) run at different speeds, and a migration throws
//! away the warm L1/L2 contents. A benchmark that should measure "one prime
//! core" must therefore tell the kernel which cores the thread may use.
//!
//! The kernel call is `sched_setaffinity(pid, size, mask)`: `mask` is a
//! bitmask (`cpu_set_t`) with bit *n* set for each allowed core *n*, and
//! `pid = 0` means "the calling thread".

use std::io;

/// Restricts the calling thread to `cores` (it may run on any of them).
///
/// Errors if a core number does not fit in `cpu_set_t` (1024 bits) or the
/// kernel refuses (e.g. none of the cores is online).
pub fn pin_current_thread(cores: &[usize]) -> io::Result<()> {
    // SAFETY: `cpu_set_t` is a plain bit array; all-zero is the empty set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let capacity = 8 * std::mem::size_of::<libc::cpu_set_t>();
    for &c in cores {
        if c >= capacity {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("core {c} out of range")));
        }
        // SAFETY: `c` is below the set's bit capacity (checked above), so the
        // bit it sets lies inside `set`.
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // SAFETY: `set` is a valid, initialised `cpu_set_t` and the size passed
    // is its exact size; the kernel only reads it.
    let r = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The core the calling thread is running on at this instant. After pinning
/// to a single core this is always that core.
pub fn current_cpu() -> usize {
    // SAFETY: no arguments; returns the core number or -1 on failure.
    let c = unsafe { libc::sched_getcpu() };
    assert!(c >= 0, "sched_getcpu failed: {}", io::Error::last_os_error());
    c as usize
}

/// Parses the kernel's CPU-list syntax, as used by
/// `/sys/devices/system/cpu/online` and by the `--cores` flags:
/// comma-separated single cores and inclusive ranges, e.g. `"0-5,7"` →
/// `[0, 1, 2, 3, 4, 5, 7]`.
pub fn parse_cpu_list(s: &str) -> Result<Vec<usize>, String> {
    let mut cores = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        let num = |t: &str| t.trim().parse::<usize>().map_err(|_| format!("bad core number {t:?} in {s:?}"));
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                if a > b {
                    return Err(format!("empty range {part:?}"));
                }
                cores.extend(a..=b);
            }
            None => cores.push(num(part)?),
        }
    }
    if cores.is_empty() {
        return Err(format!("no cores in {s:?}"));
    }
    Ok(cores)
}

/// The cores the kernel reports as online.
pub fn online_cpus() -> io::Result<Vec<usize>> {
    let s = std::fs::read_to_string("/sys/devices/system/cpu/online")?;
    parse_cpu_list(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_singles_ranges_and_newline() {
        assert_eq!(parse_cpu_list("0-5,7\n").unwrap(), vec![0, 1, 2, 3, 4, 5, 7]);
        assert_eq!(parse_cpu_list("6").unwrap(), vec![6]);
        assert!(parse_cpu_list("3-1").is_err());
        assert!(parse_cpu_list("x").is_err());
        assert!(parse_cpu_list("").is_err());
    }

    /// Pin the test thread to each online core in turn; the kernel must then
    /// report that exact core. Runs in its own thread so the pinning does not
    /// leak into whichever test reuses the test-harness thread.
    #[test]
    fn pinning_moves_the_thread() {
        std::thread::spawn(|| {
            for c in online_cpus().unwrap() {
                pin_current_thread(&[c]).unwrap();
                assert_eq!(current_cpu(), c);
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn rejects_core_beyond_cpu_set() {
        assert!(pin_current_thread(&[100_000]).is_err());
    }
}
