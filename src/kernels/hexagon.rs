//! The Hexagon DSP (cDSP, the phone's NPU) through FastRPC (`--features
//! hexagon`). Our DSP library is `dsp/aether_dsp.c`, built by build.rs and
//! embedded here; the calls it offers are `dsp/aether_dsp.idl`.
//!
//! How a call reaches the DSP: the system library `libcdsprpc.so` (loaded
//! with `dlopen`, as ash loads `libvulkan.so`) opens a session on the cDSP,
//! which loads our `.so` from a directory in `ADSP_LIBRARY_PATH`. Each call
//! is `remote_handle64_invoke(handle, scalars, args)`: `scalars` packs the
//! method number and how many buffers go in and out, `args` points to those
//! buffers. The kernel driver copies (or maps) them across, the DSP's
//! generated skel unpacks them and calls our C function. Packing the
//! arguments is what `qaic`'s CPU-side stub would do; we do it here.
//!
//! Unsigned code: the session is marked as running an unsigned module
//! (`DSPRPC_CONTROL_UNSIGNED_MODULE`) before it opens, so our library needs
//! no signature. It then runs in an "unsigned protection domain", which
//! this phone allows (llama.cpp's NPU backend runs the same way).

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;
use std::sync::OnceLock;

pub type Result<T> = std::result::Result<T, String>;

/// Our DSP library, built for Hexagon v79 by build.rs.
pub static DSP_LIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libaether_dsp.so"));

/// From the SDK's remote.h.
const CDSP_DOMAIN_ID: c_int = 3;
const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;
const FASTRPC_MAP_FD: c_int = 2;
// rpcmem.h: the system heap, CPU-cached, not registered with FastRPC by
// rpcmem itself (we map it with fastrpc_mmap).
const RPCMEM_HEAP_ID_SYSTEM: c_int = 25;
const RPCMEM_FLAGS: u32 = 1 | 0x4000_0000;
const URI: &str = "file:///libaether_dsp.so?aether_dsp_skel_handle_invoke&_modver=1.0&_dom=cdsp";

/// remote.h's `remote_arg` as a buffer (the union's other members are not
/// used): 16 bytes on a 64-bit CPU.
#[repr(C)]
struct Arg {
    pv: *mut c_void,
    len: usize,
}

/// remote.h's `REMOTE_SCALARS_MAKEX(0, method, n_in, n_out, 0, 0)`.
const fn scalars(method: u32, n_in: u32, n_out: u32) -> u32 {
    (method & 0x1f) << 24 | (n_in & 0xff) << 16 | (n_out & 0xff) << 8
}

/// Method numbers: 0 and 1 are open and close, then the IDL's order.
const PING: u32 = 2;
const ADD: u32 = 3;
const ADD_SHARED: u32 = 4;
const READ_BW: u32 = 5;
const HVX_UNITS: u32 = 6;

/// The functions of `libcdsprpc.so` we use (remote.h, rpcmem.h).
struct Rpc {
    control: unsafe extern "C" fn(u32, *mut c_void, u32) -> c_int,
    open: unsafe extern "C" fn(*const c_char, *mut u64) -> c_int,
    invoke: unsafe extern "C" fn(u64, u32, *mut Arg) -> c_int,
    close: unsafe extern "C" fn(u64) -> c_int,
    alloc: unsafe extern "C" fn(c_int, u32, usize) -> *mut c_void,
    to_fd: unsafe extern "C" fn(*mut c_void) -> c_int,
    free: unsafe extern "C" fn(*mut c_void),
    mmap: unsafe extern "C" fn(c_int, c_int, *mut c_void, c_int, usize, c_int) -> c_int,
    munmap: unsafe extern "C" fn(c_int, c_int, *mut c_void, usize) -> c_int,
}

/// Loads `libcdsprpc.so` once, with our library's directory first in
/// `ADSP_LIBRARY_PATH` (read when the library loads).
fn rpc(dir: &Path) -> Result<&'static Rpc> {
    static RPC: OnceLock<Result<Rpc>> = OnceLock::new();
    RPC.get_or_init(|| {
        // The loader's search path; entries are separated by ';'.
        let old = std::env::var("ADSP_LIBRARY_PATH").unwrap_or_else(|_| "/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp".into());
        // SAFETY: set once, before libcdsprpc.so (the only reader) loads;
        // callers open the DSP before starting threads that read the
        // environment.
        unsafe { std::env::set_var("ADSP_LIBRARY_PATH", format!("{};{old}", dir.display())) };
        // SAFETY: loading the system's FastRPC library; its initialisers
        // only set up its own state. It is never unloaded: its threads
        // (the listener that serves the DSP's file reads) outlive a
        // session, and unloading under them crashes the process.
        let lib = unsafe { libc::dlopen(c"libcdsprpc.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return Err("dlopen libcdsprpc.so failed (no FastRPC on this device?)".into());
        }
        let sym = |name: &std::ffi::CStr| {
            // SAFETY: `lib` is a live handle from dlopen.
            let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
            if p.is_null() { Err(format!("libcdsprpc.so has no {name:?}")) } else { Ok(p) }
        };
        // SAFETY (all): the symbols have these C signatures (remote.h,
        // rpcmem.h).
        unsafe {
            Ok(Rpc {
                control: std::mem::transmute(sym(c"remote_session_control")?),
                open: std::mem::transmute(sym(c"remote_handle64_open")?),
                invoke: std::mem::transmute(sym(c"remote_handle64_invoke")?),
                close: std::mem::transmute(sym(c"remote_handle64_close")?),
                alloc: std::mem::transmute(sym(c"rpcmem_alloc2")?),
                to_fd: std::mem::transmute(sym(c"rpcmem_to_fd")?),
                free: std::mem::transmute(sym(c"rpcmem_free")?),
                mmap: std::mem::transmute(sym(c"fastrpc_mmap")?),
                munmap: std::mem::transmute(sym(c"fastrpc_munmap")?),
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// An open session on the cDSP with our library loaded.
pub struct Dsp {
    rpc: &'static Rpc,
    handle: u64,
}

impl Dsp {
    /// Writes our library into `dir` (which must be on a file system the
    /// FastRPC loader can read, e.g. /data/local/tmp/… or an app's files
    /// directory), puts `dir` first in `ADSP_LIBRARY_PATH`, and opens it on
    /// the cDSP as an unsigned module.
    pub fn open(dir: &Path) -> Result<Dsp> {
        std::fs::write(dir.join("libaether_dsp.so"), DSP_LIB).map_err(|e| format!("write the DSP library to {}: {e}", dir.display()))?;
        let rpc = rpc(dir)?;
        // remote.h's `struct remote_rpc_control_unsigned_module`.
        let mut unsigned = [CDSP_DOMAIN_ID, 1];
        // SAFETY: the request and its 8-byte struct match remote.h.
        let err = unsafe { (rpc.control)(DSPRPC_CONTROL_UNSIGNED_MODULE, unsigned.as_mut_ptr().cast(), 8) };
        if err != 0 {
            return Err(format!("remote_session_control(unsigned module): 0x{err:x}"));
        }
        let uri = CString::new(URI).unwrap();
        let mut handle = 0u64;
        // SAFETY: a NUL-terminated URI and a place for the handle.
        let err = unsafe { (rpc.open)(uri.as_ptr(), &mut handle) };
        if err != 0 {
            return Err(format!("remote_handle64_open({URI}): 0x{err:x}"));
        }
        Ok(Dsp { rpc, handle })
    }

    /// One call: `ins` are copied to the DSP, `outs` back from it.
    fn call(&self, method: u32, ins: &[&[u8]], outs: &mut [&mut [u8]]) -> Result<()> {
        let mut args: Vec<Arg> = ins.iter().map(|b| Arg { pv: b.as_ptr() as *mut c_void, len: b.len() }).collect();
        args.extend(outs.iter_mut().map(|b| Arg { pv: b.as_mut_ptr().cast(), len: b.len() }));
        // SAFETY: every Arg points to a live buffer of its length; the DSP
        // writes only into the output ones, which we borrow mutably.
        let err = unsafe { (self.rpc.invoke)(self.handle, scalars(method, ins.len() as u32, outs.len() as u32), args.as_mut_ptr()) };
        if err != 0 { Err(format!("DSP method {method}: 0x{err:x}")) } else { Ok(()) }
    }

    /// x + 1, computed on the DSP: one round trip.
    pub fn ping(&self, x: u32) -> Result<u32> {
        let mut y = [0u8; 4];
        self.call(PING, &[&x.to_le_bytes()], &mut [&mut y])?;
        Ok(u32::from_le_bytes(y))
    }

    /// c = a + b on the DSP's vector unit (HVX).
    pub fn add(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
        assert!(a.len() == b.len() && a.len() == c.len());
        // The scalar arguments travel in one buffer: the three lengths.
        let n = (a.len() as u32).to_le_bytes();
        let lens: Vec<u8> = [n, n, n].concat();
        let bytes = |x: &[f32]| unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), 4 * x.len()) };
        // SAFETY: c's bytes, borrowed mutably for the call.
        let cb = unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr().cast::<u8>(), 4 * c.len()) };
        self.call(ADD, &[&lens, bytes(a), bytes(b)], &mut [cb])
    }

    /// Memory the CPU and the DSP both see: `bytes` (rounded up to 4 KB)
    /// from rpcmem, mapped into the DSP once.
    pub fn shared(&self, bytes: usize) -> Result<Shared> {
        let len = bytes.div_ceil(4096) * 4096;
        // SAFETY: rpcmem_alloc2 returns a fresh mapping of `len` bytes or
        // null.
        let ptr = unsafe { (self.rpc.alloc)(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAGS, len) };
        if ptr.is_null() {
            return Err(format!("rpcmem_alloc2({len}) failed"));
        }
        // SAFETY: ptr came from rpcmem_alloc2.
        let fd = unsafe { (self.rpc.to_fd)(ptr) };
        // SAFETY: the buffer and its fd are live and `len` long.
        let err = unsafe { (self.rpc.mmap)(CDSP_DOMAIN_ID, fd, ptr, 0, len, FASTRPC_MAP_FD) };
        if fd < 0 || err != 0 {
            // SAFETY: not mapped, not used again.
            unsafe { (self.rpc.free)(ptr) };
            return Err(format!("fastrpc_mmap(fd {fd}, {len}): 0x{err:x}"));
        }
        Ok(Shared { rpc: self.rpc, ptr: ptr.cast(), len, fd })
    }

    /// c = a + b on the DSP, in shared buffers (n floats each): only the
    /// three file descriptors and n travel with the call.
    pub fn add_shared(&self, a: &Shared, b: &Shared, c: &mut Shared, n: usize) -> Result<()> {
        assert!(4 * n <= a.len.min(b.len).min(c.len));
        self.call(ADD_SHARED, &[&words(&[a.fd as u32, b.fd as u32, c.fd as u32, n as u32])], &mut [])
    }

    /// Reads `bytes` of `buf` `reps` times on `threads` DSP threads with
    /// HVX loads; returns the DSP's own time for it in microseconds and a
    /// checksum (wrapping sum of the 32-bit words read, every rep). With
    /// `prefetch`, the threads prefetch 16 KB ahead into the L2.
    pub fn read_bw(&self, buf: &Shared, bytes: usize, threads: u32, reps: u32, prefetch: bool) -> Result<(u64, u32)> {
        assert!(bytes <= buf.len);
        let mut out = [0u8; 16];
        self.call(READ_BW, &[&words(&[buf.fd as u32, bytes as u32, threads, reps, prefetch as u32])], &mut [&mut out])?;
        Ok((u64::from_le_bytes(out[..8].try_into().unwrap()), u32::from_le_bytes(out[8..12].try_into().unwrap())))
    }

    /// The number of 128-byte HVX contexts on the DSP.
    pub fn hvx_units(&self) -> Result<u32> {
        let mut out = [0u8; 4];
        self.call(HVX_UNITS, &[], &mut [&mut out])?;
        Ok(u32::from_le_bytes(out))
    }
}

/// u32 scalars as the bytes of one argument buffer.
fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A buffer the CPU and the DSP share ([`Dsp::shared`]).
pub struct Shared {
    rpc: &'static Rpc,
    ptr: *mut u8,
    len: usize,
    fd: c_int,
}

impl Shared {
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // SAFETY: `len` bytes, page-aligned; T is plain data.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len / size_of::<T>()) }
    }
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        // SAFETY: as above, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len / size_of::<T>()) }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: mapped by us, not used after this.
        unsafe {
            (self.rpc.munmap)(CDSP_DOMAIN_ID, self.fd, self.ptr.cast(), self.len);
            (self.rpc.free)(self.ptr.cast());
        }
    }
}

impl Drop for Dsp {
    fn drop(&mut self) {
        // SAFETY: the handle is ours and not used after this.
        unsafe { (self.rpc.close)(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dsp() -> Option<Dsp> {
        let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        match Dsp::open(&dir) {
            Ok(d) => Some(d),
            Err(e) => {
                assert!(!cfg!(target_os = "android"), "{e}");
                None
            }
        }
    }

    /// Our library loads on the cDSP and answers.
    #[test]
    fn dsp_answers_a_ping() {
        let Some(d) = dsp() else { return };
        assert_eq!(d.ping(41).unwrap(), 42);
    }

    /// Buffers in shared memory: the DSP reads what the CPU wrote, and the
    /// CPU reads what the DSP wrote, with no copies or cache flushes.
    #[test]
    fn dsp_adds_shared_buffers() {
        let Some(d) = dsp() else { return };
        let n = 100_000;
        let (mut a, mut b, mut c) = (d.shared(4 * n).unwrap(), d.shared(4 * n).unwrap(), d.shared(4 * n).unwrap());
        for round in 0..3 {
            for i in 0..n {
                a.as_mut_slice::<f32>()[i] = (i * (round + 1)) as f32;
                b.as_mut_slice::<f32>()[i] = 0.25;
            }
            d.add_shared(&a, &b, &mut c, n).unwrap();
            let c = &c.as_slice::<f32>()[..n];
            let bad = c.iter().enumerate().position(|(i, &v)| v != (i * (round + 1)) as f32 + 0.25);
            assert!(bad.is_none(), "round {round}: c[{0}] = {1}", bad.unwrap(), c[bad.unwrap()]);
        }
    }

    /// The DSP's read loop reads every word: its checksum equals the sum of
    /// the words, once per rep.
    #[test]
    fn dsp_reads_every_word() {
        let Some(d) = dsp() else { return };
        let bytes = 1 << 20;
        let mut buf = d.shared(bytes).unwrap();
        for (i, w) in buf.as_mut_slice::<u32>().iter_mut().enumerate() {
            *w = (i as u32).wrapping_mul(2_654_435_761);
        }
        let sum = buf.as_slice::<u32>().iter().fold(0u32, |s, &w| s.wrapping_add(w));
        for threads in [1, 2, 4] {
            for prefetch in [false, true] {
                let (_, check) = d.read_bw(&buf, bytes, threads, 3, prefetch).unwrap();
                assert_eq!(check, sum.wrapping_mul(3), "{threads} threads, prefetch {prefetch}");
            }
        }
    }

    /// HVX float adds are IEEE single precision: exactly a + b, for lengths
    /// with and without a tail past the last whole vector.
    #[test]
    fn dsp_adds_arrays_with_hvx() {
        let Some(d) = dsp() else { return };
        for n in [1, 31, 32, 1000, 4096] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 1e3).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).cos()).collect();
            let mut c = vec![0.0; n];
            d.add(&a, &b, &mut c).unwrap();
            for i in 0..n {
                assert_eq!(c[i].to_bits(), (a[i] + b[i]).to_bits(), "n {n}, i {i}");
            }
        }
    }
}
