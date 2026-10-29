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

use crate::quant::{BlockQ4_0, BlockQ8_0, QK, f16_to_f32};

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
const REFRESH: u32 = 7;
const Q4_0_GEMV: u32 = 8;
const MODEL_LOAD: u32 = 9;
const MODEL_DECODE: u32 = 10;
const MATH_TEST: u32 = 11;

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

    /// A Q4_0 matrix (`rows` rows of `blocks.len() / rows` blocks) in
    /// shared memory, in the DSP kernel's layout: groups of 32 rows (the
    /// last padded with zeros); per group and block, four 128-byte vectors
    /// whose 32-bit lane j holds 4 quant bytes of row j (vector k: bytes
    /// 4k..4k+3 of the block), then a vector of the 32 rows' f32 scales.
    pub fn q4_0_matrix(&self, blocks: &[BlockQ4_0], rows: usize) -> Result<DspMatQ4_0> {
        assert!(rows > 0 && blocks.len().is_multiple_of(rows));
        let nb = blocks.len() / rows;
        let groups = rows.div_ceil(32);
        let mut buf = self.shared(groups * nb * 640)?;
        repack_q4_0(blocks, rows, buf.as_mut_slice::<u8>());
        self.call(REFRESH, &[&words(&[buf.fd as u32, (groups * nb * 640) as u32])], &mut [])?;
        Ok(DspMatQ4_0 { buf, rows, nb })
    }

    /// Room for an activation of `nb` Q8_0 blocks ([`Dsp::q4_0_gemv`]).
    pub fn q8_0_vector(&self, nb: usize) -> Result<Shared> {
        self.shared(40 * nb)
    }

    /// out = W x on `threads` DSP threads, x already quantized to Q8_0;
    /// `xbuf` from [`Dsp::q8_0_vector`], `out` at least `rows` rounded up
    /// to 32 floats. With `prefetch`, each DSP thread prefetches its next
    /// rows into the L2. Returns the DSP's own time in microseconds.
    pub fn q4_0_gemv(&self, w: &DspMatQ4_0, x: &[BlockQ8_0], xbuf: &mut Shared, out: &mut Shared, threads: u32, prefetch: bool) -> Result<u64> {
        let nb = w.nb;
        assert!(x.len() == nb && xbuf.len >= 40 * nb && out.len >= 4 * w.rows.div_ceil(32) * 32);
        // The int8 values, then the scales, then each block's sum.
        let xb = xbuf.as_mut_slice::<u8>();
        for (i, blk) in x.iter().enumerate() {
            for (k, &q) in blk.qs.iter().enumerate() {
                xb[QK * i + k] = q as u8;
            }
            xb[32 * nb + 4 * i..][..4].copy_from_slice(&f16_to_f32(blk.d).to_le_bytes());
            let sum: i32 = blk.qs.iter().map(|&q| q as i32).sum();
            xb[36 * nb + 4 * i..][..4].copy_from_slice(&sum.to_le_bytes());
        }
        let mut t = [0u8; 8];
        self.call(Q4_0_GEMV, &[&words(&[w.buf.fd as u32, xbuf.fd as u32, out.fd as u32, w.rows as u32, nb as u32, threads, prefetch as u32])], &mut [&mut t])?;
        Ok(u64::from_le_bytes(t))
    }

    /// Hands the DSP a model (dsp/model.c's `desc` layout; built by
    /// `model::dsp::DspModel`).
    pub fn model_load(&self, desc: &[u32]) -> Result<()> {
        self.call(MODEL_LOAD, &[&(desc.len() as u32).to_le_bytes(), &words(desc)], &mut [])
    }

    /// One decode step of the loaded model at `pos` on `threads` DSP
    /// threads: the greedy next token and the DSP's time in microseconds.
    pub fn model_decode(&self, pos: u32, threads: u32) -> Result<(u32, u64)> {
        let mut out = [0u8; 16];
        self.call(MODEL_DECODE, &[&words(&[pos, threads])], &mut [&mut out])?;
        Ok((u32::from_le_bytes(out[..4].try_into().unwrap()), u64::from_le_bytes(out[8..].try_into().unwrap())))
    }

    /// The decode step's vector math on the DSP (dsp/model.c), for tests:
    /// e^a (op 0) or silu(a)·b (op 1), lengths a multiple of 32.
    pub fn math_test(&self, op: u32, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        assert!(a.len() == b.len() && a.len().is_multiple_of(32));
        let n = a.len() as u32;
        let mut y = vec![0f32; a.len()];
        let bytes = |x: &[f32]| unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), 4 * x.len()) };
        // SAFETY: y's bytes, borrowed mutably for the call.
        let yb = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr().cast::<u8>(), 4 * y.len()) };
        self.call(MATH_TEST, &[&words(&[op, n, n, n]), bytes(a), bytes(b)], &mut [yb])?;
        Ok(y)
    }

    /// The number of 128-byte HVX contexts on the DSP.
    pub fn hvx_units(&self) -> Result<u32> {
        let mut out = [0u8; 4];
        self.call(HVX_UNITS, &[], &mut [&mut out])?;
        Ok(u32::from_le_bytes(out))
    }
}

/// Writes `rows` rows of Q4_0 `blocks` in the DSP kernel's layout
/// ([`Dsp::q4_0_matrix`]) into `out` (at least [`q4_0_bytes`] long).
pub fn repack_q4_0(blocks: &[BlockQ4_0], rows: usize, out: &mut [u8]) {
    let nb = blocks.len() / rows;
    for g in 0..rows.div_ceil(32) {
        for b in 0..nb {
            let at = (g * nb + b) * 640;
            for j in 0..32.min(rows - 32 * g) {
                let blk = &blocks[(32 * g + j) * nb + b];
                for k in 0..4 {
                    out[at + 128 * k + 4 * j..][..4].copy_from_slice(&blk.qs[4 * k..4 * k + 4]);
                }
                out[at + 512 + 4 * j..][..4].copy_from_slice(&f16_to_f32(blk.d).to_le_bytes());
            }
            for j in 32.min(rows - 32 * g)..32 {
                for k in 0..5 {
                    out[at + 128 * k + 4 * j..][..4].fill(0);
                }
            }
        }
    }
}

/// Bytes of a Q4_0 matrix in the DSP kernel's layout.
pub fn q4_0_bytes(rows: usize, nb: usize) -> usize {
    rows.div_ceil(32) * nb * 640
}

/// u32 scalars as the bytes of one argument buffer.
fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A Q4_0 matrix in the DSP's layout ([`Dsp::q4_0_matrix`]).
pub struct DspMatQ4_0 {
    buf: Shared,
    pub rows: usize,
    pub nb: usize,
}

impl DspMatQ4_0 {
    /// Bytes the kernel reads per product.
    pub fn bytes(&self) -> usize {
        self.rows.div_ceil(32) * self.nb * 640
    }
}

/// A buffer the CPU and the DSP share ([`Dsp::shared`]).
pub struct Shared {
    rpc: &'static Rpc,
    ptr: *mut u8,
    len: usize,
    fd: c_int,
}

impl Shared {
    /// The buffer's file descriptor: how the DSP finds it.
    pub fn fd(&self) -> u32 {
        self.fd as u32
    }
    pub fn len(&self) -> usize {
        self.len
    }
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

    /// The decode step's vector e^x and SwiGLU (dsp/model.c: a series
    /// and Newton steps of our own) against the standard library's exp in
    /// f64, over the range the model uses them in.
    #[test]
    fn dsp_vector_exp_and_swiglu_match_libm() {
        let Some(d) = dsp() else { return };
        let n = 32 * 1024;
        let x: Vec<f32> = (0..n).map(|i| -80.0 + 90.0 * i as f32 / n as f32).collect();
        let u: Vec<f32> = (0..n).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
        let worst = |got: &[f32], want: &dyn Fn(usize) -> f64| {
            (0..n).map(|i| ((got[i] as f64 - want(i)).abs() / want(i).abs().max(1e-30), i)).fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a })
        };
        let e = d.math_test(0, &x, &u).unwrap();
        let (err, i) = worst(&e, &|i| (x[i] as f64).exp());
        assert!(err < 1e-6, "e^{}: {} against {} (relative error {err:.2e})", x[i], e[i], (x[i] as f64).exp());
        let g: Vec<f32> = x.iter().map(|&v| v * 0.9 + 9.0).collect(); // -63 .. 18
        let y = d.math_test(1, &g, &u).unwrap();
        let silu = |i: usize| g[i] as f64 / (1.0 + (-g[i] as f64).exp()) * u[i] as f64;
        let (err, i) = worst(&y, &silu);
        assert!(err < 2e-6 || silu(i).abs() < 1e-30, "silu({}) * {}: {} against {} (relative error {err:.2e})", g[i], u[i], y[i], silu(i));
    }

    /// The DSP's Q4_0 x Q8_0 product on llama.cpp's bytes (the weights
    /// and x as llama.cpp quantizes them) against dequant(x) . dequant(W)
    /// in float64 (`out_q4_0_b`, tools/gen_quant.py), on 1 and 5 threads.
    #[test]
    fn dsp_q4_0_gemv_matches_reference() {
        let Some(d) = dsp() else { return };
        let Some(root) = crate::testutil::golden("quant") else { return };
        let mut cases = 0;
        for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
            cases += 1;
            let (ws, _) = crate::testutil::read_npy(&dir.join("w.npy"));
            let (_, want) = crate::testutil::read_npy(&dir.join("out_q4_0_b.npy"));
            let bytes = std::fs::read(dir.join("w_q4_0.bin")).unwrap();
            let blocks: Vec<BlockQ4_0> = bytes.as_chunks::<18>().0.iter().map(|c| BlockQ4_0 { d: u16::from_le_bytes([c[0], c[1]]), qs: c[2..].try_into().unwrap() }).collect();
            let bytes = std::fs::read(dir.join("x_q8_0.bin")).unwrap();
            let x: Vec<BlockQ8_0> = bytes.as_chunks::<34>().0.iter().map(|c| BlockQ8_0 { d: u16::from_le_bytes([c[0], c[1]]), qs: c[2..].iter().map(|&b| b as i8).collect::<Vec<_>>().try_into().unwrap() }).collect();
            let w = d.q4_0_matrix(&blocks, ws[0]).unwrap();
            let mut xbuf = d.q8_0_vector(w.nb).unwrap();
            let mut out = d.shared(4 * ws[0].div_ceil(32) * 32).unwrap();
            for (threads, prefetch) in [(1, false), (5, false), (5, true)] {
                out.as_mut_slice::<f32>().fill(f32::NAN);
                d.q4_0_gemv(&w, &x, &mut xbuf, &mut out, threads, prefetch).unwrap();
                let (_, rel) = crate::testutil::max_errors(&out.as_slice::<f32>()[..ws[0]], &want);
                assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{} on {threads} threads (prefetch {prefetch}): rel {rel:.2e}", dir.display());
            }
        }
        assert!(cases >= 12, "only {cases} golden cases");
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
