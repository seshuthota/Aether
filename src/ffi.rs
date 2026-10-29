//! A C ABI for the engine, so other languages can load a model and stream
//! generated text (the Android app calls it through a small JNI shim).
//! The header is `include/aether.h`.
//!
//! Rules every function here follows, because C cannot see Rust's:
//!
//! - **Opaque handle**: C gets a pointer to an `AetherSession` it cannot look
//!   inside; it must hand it back to `aether_close` exactly once.
//! - **Error codes, not panics**: a Rust panic unwinding into C is undefined
//!   behaviour, so every entry point runs its body under `catch_unwind` and
//!   turns a panic into `AETHER_ERR_PANIC`. The message of the last error on
//!   this thread is available from `aether_last_error`.
//! - **Borrowed strings**: C strings passed in must be NUL-terminated UTF-8
//!   and are only read during the call.

use crate::model::sampler::Sampler;
use crate::session::Session;
use crate::weight::Format;
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

pub const AETHER_OK: c_int = 0;
pub const AETHER_ERR_ARGUMENT: c_int = 1;
pub const AETHER_ERR_LOAD: c_int = 2;
pub const AETHER_ERR_PANIC: c_int = 3;

/// What C holds: a loaded model, and the prompt lookup settings
/// (`aether_set_lookup`; draft length 0 = off).
pub struct AetherSession {
    session: Session,
    lookup_k: usize,
    lookup_ngram: usize,
}

/// Timings of one `aether_generate` call (milliseconds).
#[repr(C)]
#[derive(Default)]
pub struct AetherStats {
    pub prefill_tokens: u32,
    pub prefill_ms: f64,
    pub decode_tokens: u32,
    pub decode_ms: f64,
}

/// Receives each piece of generated text: `len` bytes of UTF-8 (a piece can
/// end in the middle of a multi-byte character; the next piece completes
/// it). Return non-zero to continue, 0 to stop generating.
pub type AetherTokenCallback = extern "C" fn(bytes: *const u8, len: usize, user: *mut c_void) -> c_int;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl Into<Vec<u8>>) {
    let msg = CString::new(msg).unwrap_or_else(|_| CString::new("error message contained NUL").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

/// Runs `body`, converting a panic into `AETHER_ERR_PANIC`.
fn guard(body: impl FnOnce() -> c_int) -> c_int {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(p) => {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into());
            set_error(format!("internal error: {msg}"));
            AETHER_ERR_PANIC
        }
    }
}

/// # Safety
/// `s` must be null or a valid NUL-terminated string.
unsafe fn str_arg<'a>(s: *const c_char, what: &str) -> Result<&'a str, c_int> {
    if s.is_null() {
        set_error(format!("{what} is null"));
        return Err(AETHER_ERR_ARGUMENT);
    }
    // SAFETY: non-null and NUL-terminated per the caller's contract.
    unsafe { CStr::from_ptr(s) }.to_str().map_err(|_| {
        set_error(format!("{what} is not UTF-8"));
        AETHER_ERR_ARGUMENT
    })
}

/// Message describing the last error on this thread ("" if none). Valid
/// until the next aether call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn aether_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Loads a model (`.gguf`, Hugging Face directory, or llama2.c `.bin`).
/// `cores`: CPU list such as "2-7" (one thread per core), or null/"" for one
/// unpinned thread. `max_seq`: longest conversation in tokens (sizes the KV
/// cache). On success stores the handle in `*out`.
///
/// # Safety
/// `path` and `cores` (if non-null) must be NUL-terminated strings; `out`
/// must point to writable memory for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aether_open(path: *const c_char, cores: *const c_char, max_seq: u32, out: *mut *mut AetherSession) -> c_int {
    guard(|| {
        if out.is_null() {
            set_error("out is null");
            return AETHER_ERR_ARGUMENT;
        }
        // SAFETY: caller's contract.
        let path = match unsafe { str_arg(path, "path") } {
            Ok(p) => p,
            Err(code) => return code,
        };
        let cores = if cores.is_null() {
            None
        } else {
            // SAFETY: caller's contract.
            match unsafe { str_arg(cores, "cores") } {
                Ok("") => None,
                Ok(c) => match crate::sys::affinity::parse_cpu_list(c) {
                    Ok(list) => Some(list),
                    Err(e) => {
                        set_error(e);
                        return AETHER_ERR_ARGUMENT;
                    }
                },
                Err(code) => return code,
            }
        };
        match Session::open(Path::new(path), None, max_seq as usize, Format::F32, cores.as_deref()) {
            Ok(s) => {
                // SAFETY: `out` is non-null and writable (caller's contract).
                unsafe { *out = Box::into_raw(Box::new(AetherSession { session: s, lookup_k: 0, lookup_ngram: 2 })) };
                AETHER_OK
            }
            Err(e) => {
                set_error(format!("{path}: {e}"));
                AETHER_ERR_LOAD
            }
        }
    })
}

/// Generates a reply to `prompt`, calling `on_text` with each piece as soon
/// as it is decoded. `chat` non-zero wraps the prompt in the ChatML template
/// (instruct models). `temperature` 0 is greedy; `top_p` in (0, 1) enables
/// nucleus sampling; `seed` fixes the random choices. Stops at the model's
/// end token, after `max_tokens` new tokens, or when `on_text` returns 0.
/// `stats` may be null.
///
/// # Safety
/// `session` must come from `aether_open` and not be closed; `prompt` must
/// be a NUL-terminated string; `stats`, if non-null, must be writable.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn aether_generate(
    session: *mut AetherSession,
    prompt: *const c_char,
    chat: c_int,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    seed: u64,
    on_text: AetherTokenCallback,
    user: *mut c_void,
    stats: *mut AetherStats,
) -> c_int {
    guard(|| {
        if session.is_null() {
            set_error("session is null");
            return AETHER_ERR_ARGUMENT;
        }
        // SAFETY: a live handle from aether_open (caller's contract); the
        // caller does not use it from another thread during this call.
        let AetherSession { session: s, lookup_k, lookup_ngram } = unsafe { &mut *session };
        // SAFETY: caller's contract.
        let prompt = match unsafe { str_arg(prompt, "prompt") } {
            Ok(p) => p,
            Err(code) => return code,
        };
        let text = if chat != 0 { crate::chat::chatml(crate::chat::QWEN_DEFAULT_SYSTEM, prompt) } else { prompt.to_string() };
        let prompt_len = s.text.encode(&text).len();
        let steps = prompt_len + max_tokens as usize;
        if steps > s.model.config.seq_len {
            set_error(format!("prompt ({prompt_len} tokens) + max_tokens ({max_tokens}) exceeds the context ({})", s.model.config.seq_len));
            return AETHER_ERR_ARGUMENT;
        }
        let mut sampler = Sampler::new(temperature, top_p, 0, seed.max(1));
        let emit = |piece: &[u8]| on_text(piece.as_ptr(), piece.len(), user) != 0;
        let st = if *lookup_k > 0 { s.generate_lookup(&text, &mut sampler, steps, *lookup_k, *lookup_ngram, emit).0 } else { s.generate(&text, &mut sampler, steps, false, emit) };
        if !stats.is_null() {
            // SAFETY: non-null and writable (caller's contract).
            unsafe {
                *stats = AetherStats {
                    prefill_tokens: st.prefill_tokens as u32,
                    prefill_ms: st.prefill.as_secs_f64() * 1e3,
                    decode_tokens: st.decode_tokens as u32,
                    decode_ms: st.decode.as_secs_f64() * 1e3,
                }
            };
        }
        AETHER_OK
    })
}

/// Turns prompt lookup decoding on for later `aether_generate` calls:
/// after each token, up to `k` draft tokens are copied from the last
/// earlier place where the last `ngram` tokens occurred, and checked in one
/// batched pass (`Session::generate_lookup`). Faster when the reply repeats
/// the prompt (code edits, quoting); the output keeps the sampler's
/// distribution. `k` = 0 turns it off; `ngram` must be at least 1.
///
/// # Safety
/// `session` must come from `aether_open` and not be closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aether_set_lookup(session: *mut AetherSession, k: u32, ngram: u32) -> c_int {
    guard(|| {
        if session.is_null() || ngram == 0 {
            set_error(if session.is_null() { "session is null" } else { "ngram must be at least 1" });
            return AETHER_ERR_ARGUMENT;
        }
        // SAFETY: a live handle from aether_open (caller's contract).
        let s = unsafe { &mut *session };
        (s.lookup_k, s.lookup_ngram) = (k as usize, ngram as usize);
        AETHER_OK
    })
}

/// Frees a session. Null is allowed (does nothing).
///
/// # Safety
/// `session` must come from `aether_open` and not have been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aether_close(session: *mut AetherSession) {
    let _ = guard(|| {
        if !session.is_null() {
            // SAFETY: created by Box::into_raw in aether_open, freed once.
            drop(unsafe { Box::from_raw(session) });
        }
        AETHER_OK
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;

    extern "C" fn collect(bytes: *const u8, len: usize, user: *mut c_void) -> c_int {
        // SAFETY: `user` is the &mut Vec<u8> passed below; bytes/len valid.
        let out = unsafe { &mut *(user as *mut Vec<u8>) };
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(bytes, len) });
        1
    }

    /// Through the C ABI exactly as C would call it: load, generate greedy
    /// llama2.c text, check errors are reported, close.
    #[test]
    fn c_abi_round_trip() {
        let Some(model) = golden("models/stories15M/stories15M.bin") else { return };
        let path = CString::new(model.to_str().unwrap()).unwrap();
        let mut s: *mut AetherSession = std::ptr::null_mut();
        unsafe {
            assert_eq!(aether_open(path.as_ptr(), std::ptr::null(), 256, &mut s), AETHER_OK);
            let mut out = Vec::<u8>::new();
            let mut stats = AetherStats::default();
            let prompt = CString::new("Once upon a time").unwrap();
            let code = aether_generate(s, prompt.as_ptr(), 0, 0.0, 0.9, 59, 1, collect, &mut out as *mut _ as *mut c_void, &mut stats);
            assert_eq!(code, AETHER_OK);
            // llama2.c's greedy text minus the echoed prompt.
            let want = std::fs::read_to_string(golden("stories15M/greedy.txt").unwrap()).unwrap();
            assert_eq!(String::from_utf8(out).unwrap(), want.trim_end().trim_start_matches("Once upon a time"));
            assert_eq!(stats.prefill_tokens, 5);
            // Prompt lookup decoding gives the same greedy text.
            assert_eq!(aether_set_lookup(s, 4, 1), AETHER_OK);
            let mut out = Vec::<u8>::new();
            assert_eq!(aether_generate(s, prompt.as_ptr(), 0, 0.0, 0.9, 59, 1, collect, &mut out as *mut _ as *mut c_void, &mut stats), AETHER_OK);
            assert_eq!(String::from_utf8(out).unwrap(), want.trim_end().trim_start_matches("Once upon a time"));
            assert_eq!(aether_set_lookup(s, 4, 0), AETHER_ERR_ARGUMENT);
            // Errors come back as codes plus a message, never as a panic.
            let bad = CString::new("/nonexistent/model.gguf").unwrap();
            let mut s2: *mut AetherSession = std::ptr::null_mut();
            assert_eq!(aether_open(bad.as_ptr(), std::ptr::null(), 256, &mut s2), AETHER_ERR_LOAD);
            assert!(CStr::from_ptr(aether_last_error()).to_str().unwrap().contains("nonexistent"));
            assert_eq!(aether_generate(s, std::ptr::null(), 0, 0.0, 0.9, 8, 1, collect, std::ptr::null_mut(), std::ptr::null_mut()), AETHER_ERR_ARGUMENT);
            aether_close(s);
        }
    }
}
