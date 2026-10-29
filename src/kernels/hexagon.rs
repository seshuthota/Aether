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

pub type Result<T> = std::result::Result<T, String>;

/// Our DSP library, built for Hexagon v79 by build.rs.
pub static DSP_LIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libaether_dsp.so"));

/// From the SDK's remote.h.
const CDSP_DOMAIN_ID: c_int = 3;
const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;
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

type SessionControl = unsafe extern "C" fn(u32, *mut c_void, u32) -> c_int;
type Open = unsafe extern "C" fn(*const c_char, *mut u64) -> c_int;
type Invoke = unsafe extern "C" fn(u64, u32, *mut Arg) -> c_int;
type Close = unsafe extern "C" fn(u64) -> c_int;

/// An open session on the cDSP with our library loaded.
pub struct Dsp {
    handle: u64,
    invoke: Invoke,
    close: Close,
}

impl Dsp {
    /// Writes our library into `dir` (which must be on a file system the
    /// FastRPC loader can read, e.g. /data/local/tmp/… or an app's files
    /// directory), puts `dir` first in `ADSP_LIBRARY_PATH`, and opens it on
    /// the cDSP as an unsigned module.
    pub fn open(dir: &Path) -> Result<Dsp> {
        std::fs::write(dir.join("libaether_dsp.so"), DSP_LIB).map_err(|e| format!("write the DSP library to {}: {e}", dir.display()))?;
        // The loader's search path; entries are separated by ';'.
        let old = std::env::var("ADSP_LIBRARY_PATH").unwrap_or_else(|_| "/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp".into());
        // SAFETY: set before libcdsprpc.so is loaded, which is the only
        // reader; callers open the DSP before starting other threads.
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
        // SAFETY (all four): the symbols have these C signatures (remote.h).
        let control: SessionControl = unsafe { std::mem::transmute(sym(c"remote_session_control")?) };
        let open: Open = unsafe { std::mem::transmute(sym(c"remote_handle64_open")?) };
        let invoke: Invoke = unsafe { std::mem::transmute(sym(c"remote_handle64_invoke")?) };
        let close: Close = unsafe { std::mem::transmute(sym(c"remote_handle64_close")?) };

        // remote.h's `struct remote_rpc_control_unsigned_module`.
        let mut unsigned = [CDSP_DOMAIN_ID, 1];
        // SAFETY: the request and its 8-byte struct match remote.h.
        let err = unsafe { control(DSPRPC_CONTROL_UNSIGNED_MODULE, unsigned.as_mut_ptr().cast(), 8) };
        if err != 0 {
            return Err(format!("remote_session_control(unsigned module): 0x{err:x}"));
        }
        let uri = CString::new(URI).unwrap();
        let mut handle = 0u64;
        // SAFETY: a NUL-terminated URI and a place for the handle.
        let err = unsafe { open(uri.as_ptr(), &mut handle) };
        if err != 0 {
            return Err(format!("remote_handle64_open({URI}): 0x{err:x}"));
        }
        Ok(Dsp { handle, invoke, close })
    }

    /// One call: `ins` are copied to the DSP, `outs` back from it.
    fn call(&self, method: u32, ins: &[&[u8]], outs: &mut [&mut [u8]]) -> Result<()> {
        let mut args: Vec<Arg> = ins.iter().map(|b| Arg { pv: b.as_ptr() as *mut c_void, len: b.len() }).collect();
        args.extend(outs.iter_mut().map(|b| Arg { pv: b.as_mut_ptr().cast(), len: b.len() }));
        // SAFETY: every Arg points to a live buffer of its length; the DSP
        // writes only into the output ones, which we borrow mutably.
        let err = unsafe { (self.invoke)(self.handle, scalars(method, ins.len() as u32, outs.len() as u32), args.as_mut_ptr()) };
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
}

impl Drop for Dsp {
    fn drop(&mut self) {
        // SAFETY: the handle is ours and not used after this.
        unsafe { (self.close)(self.handle) };
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
