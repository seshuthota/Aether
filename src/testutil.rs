//! Test-only helpers: finding golden files and reading reference dumps.

use std::collections::HashMap;
use std::path::PathBuf;

/// The `testdata/` directory. On the phone the cargo runner syncs it next
/// to the test binary and says where in `AETHER_TESTDATA`; on the host it is
/// the repo's own `testdata/`.
pub fn testdata() -> PathBuf {
    match std::env::var_os("AETHER_TESTDATA") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata"),
    }
}

/// `testdata/<rel>` if it exists. Tests that need large regenerated goldens
/// call this and skip (with a message) when they are missing, instead of
/// failing on a fresh checkout.
pub fn golden(rel: &str) -> Option<PathBuf> {
    let p = testdata().join(rel);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} missing (see tools/ to regenerate)", p.display());
        None
    }
}

/// `hostdata/<rel>` if it exists: large reference files that only host
/// tests use (f32 GGUF conversions), kept out of `testdata/` so the cargo
/// runner does not copy them to the phone.
pub fn host_golden(rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hostdata").join(rel);
    if p.exists() && std::env::var_os("AETHER_TESTDATA").is_none() {
        Some(p)
    } else {
        eprintln!("SKIP: host-only {} not available here", p.display());
        None
    }
}

/// Key of one activation record: (name, layer or -1, position).
pub type Key = (String, i32, usize);

/// Reads a dump written by the patched llama2.c (`tools/llama2c-dump.patch`):
/// records of `u32 name_len, name, i32 layer, i32 pos, u32 n, n × f32`.
/// Returns the records keyed by (name, layer, pos), plus the keys in file
/// order.
pub fn read_dump(path: &std::path::Path) -> (HashMap<Key, Vec<f32>>, Vec<Key>) {
    let b = std::fs::read(path).unwrap();
    let mut at = 0;
    let u32_at = |at: &mut usize| {
        let v = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    };
    let (mut map, mut order) = (HashMap::new(), Vec::new());
    while at < b.len() {
        let len = u32_at(&mut at) as usize;
        let name = String::from_utf8(b[at..at + len].to_vec()).unwrap();
        at += len;
        let layer = u32_at(&mut at) as i32;
        let pos = u32_at(&mut at) as usize;
        let n = u32_at(&mut at) as usize;
        let data = b[at..at + 4 * n].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        at += 4 * n;
        let key = (name, layer, pos);
        order.push(key.clone());
        map.insert(key, data);
    }
    (map, order)
}

/// Largest absolute difference between two vectors, and that difference
/// relative to the reference's largest magnitude.
///
/// Relative to the vector's scale rather than element by element: an
/// element that happens to be near zero would otherwise turn a harmless
/// 1e-9 difference into a huge "relative error".
pub fn max_errors(ours: &[f32], reference: &[f32]) -> (f32, f32) {
    assert_eq!(ours.len(), reference.len());
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let abs = ours.iter().zip(reference).fold(0.0f32, |m, (&a, &r)| m.max((a - r).abs()));
    (abs, if scale > 0.0 { abs / scale } else { abs })
}

/// Reads a little-endian float32, C-order `.npy` file (numpy's own format:
/// magic `\x93NUMPY`, version, header length, a Python-dict header with
/// `descr`, `fortran_order` and `shape`, then the raw data). Returns the
/// shape and the flat data.
pub fn read_npy(path: &std::path::Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not an .npy file", path.display());
    let (header_len, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12),
    };
    let header = std::str::from_utf8(&b[start..start + header_len]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "{}: only little-endian f32 supported: {header}", path.display());
    assert!(header.contains("'fortran_order': False"), "{}: only C order supported", path.display());
    let shape_txt = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = shape_txt.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let data: Vec<f32> = b[start + header_len..].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
    assert_eq!(data.len(), shape.iter().product::<usize>(), "{}: size does not match shape", path.display());
    (shape, data)
}
