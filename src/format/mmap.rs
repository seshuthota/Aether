//! Read-only memory-mapped files.
//!
//! `mmap` asks the kernel to make a file appear in our address space: no
//! bytes are read up front; touching a page (4 KB, or 16 KB on some phones)
//! for the first time triggers a page fault and the kernel reads it from
//! storage. Because the mapping is read-only and backed by the file, the
//! kernel can drop those pages under memory pressure and re-read them later,
//! instead of killing the process (Android's low-memory killer acts on
//! anonymous memory such as heap copies). Several processes mapping the same
//! file share one copy in RAM.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

/// A whole file mapped read-only. The bytes stay valid until it is dropped.
pub struct Mmap {
    ptr: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only and owned by this value; sharing `&[u8]`
// views of it between threads is as safe as sharing any `&[u8]`.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Mmap> {
        let file = File::open(path.as_ref())?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            // mmap of length 0 is an error; an empty file maps to nothing.
            return Ok(Mmap { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0 });
        }
        // SAFETY: a fresh private read-only mapping of an open file; the
        // kernel checks the arguments and reports failure as MAP_FAILED.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // The mapping keeps the file's pages reachable after the descriptor
        // closes (when `file` drops at the end of this function).
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` maps `len` readable bytes until `drop` unmaps them;
        // the returned slice borrows `self`, so it cannot outlive that.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: exactly the region `open` mapped, unmapped once.
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_file_and_an_empty_file() {
        let dir = std::env::temp_dir().join(format!("aether-mmap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (full, empty) = (dir.join("full"), dir.join("empty"));
        std::fs::write(&full, b"hello mmap").unwrap();
        std::fs::write(&empty, b"").unwrap();
        assert_eq!(Mmap::open(&full).unwrap().as_bytes(), b"hello mmap");
        assert!(Mmap::open(&empty).unwrap().is_empty());
        assert!(Mmap::open(dir.join("missing")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
