//! Unicode Normalization Form C (NFC), which Qwen's tokenizer applies before
//! anything else.
//!
//! The same text can be written in more than one way: `é` is either the
//! single code point U+00E9 or `e` followed by U+0301 COMBINING ACUTE ACCENT.
//! NFC picks one canonical spelling — composed where possible — so that both
//! tokenize identically. The algorithm (Unicode Standard Annex #15):
//!
//! 1. **Decompose** every character fully (`é` → `e` + U+0301; Hangul
//!    syllables by a formula, since there are 11,172 of them).
//! 2. **Reorder**: within each run of combining marks, sort by combining
//!    class, so `a` + below-mark + above-mark has one order whatever the
//!    input order was.
//! 3. **Recompose**: each mark combines with the last *starter* (class-0
//!    character) if a composite exists and no mark in between has an equal
//!    or higher class ("blocked").
//!
//! Tables come from `tools/gen_unicode_tables.py` (Python's `unicodedata`).

use super::unicode_tables::{CCC, COMPOSE, DECOMP};
use std::borrow::Cow;

// Hangul syllable arithmetic (Unicode §3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn ccc(c: u32) -> u8 {
    CCC.binary_search_by_key(&c, |&(cp, _)| cp).map(|i| CCC[i].1).unwrap_or(0)
}

fn decompose(c: u32, out: &mut Vec<u32>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&c) {
        let s = c - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if s % T_COUNT != 0 {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    match DECOMP.binary_search_by_key(&c, |&(cp, _)| cp) {
        Ok(i) => out.extend_from_slice(DECOMP[i].1),
        Err(_) => out.push(c),
    }
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    // Hangul: L + V → LV, LV + T → LVT.
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a) && (a - S_BASE) % T_COUNT == 0 && (T_BASE + 1..T_BASE + T_COUNT).contains(&b) {
        return Some(a + (b - T_BASE));
    }
    COMPOSE.binary_search_by_key(&(a, b), |&(x, y, _)| (x, y)).ok().map(|i| COMPOSE[i].2)
}

/// `s` in NFC. Text that cannot change (ASCII) is returned as is.
pub fn nfc(s: &str) -> Cow<'_, str> {
    if s.is_ascii() {
        return Cow::Borrowed(s);
    }
    // 1. Full canonical decomposition.
    let mut d = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c as u32, &mut d);
    }
    // 2. Canonical ordering: stable sort of each run of non-starters by class
    //    (insertion sort: runs are a handful of marks at most).
    let classes: Vec<u8> = d.iter().map(|&c| ccc(c)).collect();
    let mut pairs: Vec<(u32, u8)> = d.into_iter().zip(classes).collect();
    let mut i = 0;
    while i < pairs.len() {
        if pairs[i].1 == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < pairs.len() && pairs[i].1 != 0 {
            i += 1;
        }
        pairs[start..i].sort_by_key(|&(_, k)| k); // stable
    }
    // 3. Canonical composition.
    let mut out: Vec<u32> = Vec::with_capacity(pairs.len());
    let mut starter: Option<usize> = None; // index in `out` of the last starter
    let mut last_class: Option<u8> = None; // class of the last char after it
    for (c, k) in pairs {
        if let Some(si) = starter {
            let blocked = matches!(last_class, Some(p) if p == 0 || p >= k);
            if !blocked && let Some(comp) = compose_pair(out[si], c) {
                out[si] = comp;
                continue;
            }
        }
        if k == 0 {
            starter = Some(out.len());
            last_class = None;
        } else {
            last_class = Some(k);
        }
        out.push(c);
    }
    Cow::Owned(out.into_iter().map(|c| char::from_u32(c).expect("valid code point")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::json::Json;

    #[test]
    fn known_cases() {
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("é"), "é");
        assert_eq!(nfc("a\u{323}\u{302}"), "ậ"); // below + above, in order
        assert_eq!(nfc("a\u{302}\u{323}"), "ậ"); // reordered first
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "각"); // Hangul jamo → syllable
        assert_eq!(nfc("\u{212B}"), "Å"); // ANGSTROM SIGN is a singleton → U+00C5
        assert_eq!(nfc("plain ascii"), "plain ascii");
    }

    /// Against Python's unicodedata.normalize("NFC", ·) on every character
    /// NFC changes and on random mark sequences (tools/gen_nfc_tests.py).
    #[test]
    fn matches_python_unicodedata() {
        let Some(p) = crate::testutil::golden("unicode/nfc.json") else { return };
        let cases = Json::parse(&std::fs::read_to_string(p).unwrap()).unwrap();
        let cases = cases.as_array().unwrap();
        for pair in cases {
            let pair = pair.as_array().unwrap();
            let (input, want) = (pair[0].as_str().unwrap(), pair[1].as_str().unwrap());
            assert_eq!(nfc(input), want, "NFC of {:?}", input.chars().map(|c| format!("U+{:04X}", c as u32)).collect::<Vec<_>>());
        }
        println!("{} NFC cases identical to Python's unicodedata", cases.len());
    }
}
