//! A small JSON parser: enough for `config.json`, safetensors headers and
//! `tokenizer.json`.
//!
//! JSON has six kinds of value: `null`, `true`/`false`, numbers, strings,
//! arrays `[..]` and objects `{"key": value, ..}`. The parser is recursive
//! descent: one function per kind, each consuming its own text and calling
//! [`Parser::value`] for nested values. Objects keep their keys in file
//! order (a `Vec` of pairs) and look keys up linearly: the files here have
//! at most a few dozen keys per object, except the tokenizer vocabulary,
//! which the tokenizer turns into a hash map once.
//!
//! Strings are where the details are: escapes `\" \\ \/ \b \f \n \r \t`, and
//! `\uXXXX`, where characters outside the Basic Multilingual Plane (emoji)
//! arrive as two `\u` escapes, a "surrogate pair", that combine into one
//! character.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// JSON has one number type; f64 holds every integer up to 2^53 exactly,
    /// which covers token ids, shapes and byte offsets of files < 8 PB.
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    pub at: usize,
    pub msg: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JSON error at byte {}: {}", self.at, self.msg)
    }
}

impl std::error::Error for Error {}

impl Json {
    pub fn parse(text: &str) -> Result<Json, Error> {
        let mut p = Parser { s: text.as_bytes(), at: 0 };
        let v = p.value()?;
        p.ws();
        if p.at != p.s.len() {
            return Err(p.err("trailing characters after the value"));
        }
        Ok(v)
    }

    /// The value under `key` if this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self { Some(*n) } else { None }
    }

    /// The number as a `usize`, if it is a non-negative integer.
    pub fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        (n >= 0.0 && n.fract() == 0.0 && n < 9.007e15).then_some(n as usize)
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self { Some(o) } else { None }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> Error {
        Error { at: self.at, msg: msg.to_string() }
    }

    fn ws(&mut self) {
        while self.at < self.s.len() && matches!(self.s[self.at], b' ' | b'\t' | b'\n' | b'\r') {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }

    fn expect(&mut self, b: u8) -> Result<(), Error> {
        if self.peek() == Some(b) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", b as char)))
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, Error> {
        if self.s[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(v)
        } else {
            Err(self.err(&format!("expected {word}")))
        }
    }

    fn value(&mut self) -> Result<Json, Error> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn object(&mut self) -> Result<Json, Error> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            pairs.push((key, self.value()?));
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Json, Error> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn number(&mut self) -> Result<Json, Error> {
        let start = self.at;
        while self.at < self.s.len() && matches!(self.s[self.at], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            self.at += 1;
        }
        // The input is &str, so this slice of ASCII bytes is valid UTF-8.
        let text = std::str::from_utf8(&self.s[start..self.at]).unwrap();
        text.parse().map(Json::Num).map_err(|_| Error { at: start, msg: format!("bad number {text:?}") })
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let h = self.s.get(self.at..self.at + 4).ok_or_else(|| self.err("short \\u escape"))?;
        let v = std::str::from_utf8(h).ok().and_then(|h| u32::from_str_radix(h, 16).ok()).ok_or_else(|| self.err("bad \\u escape"))?;
        self.at += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            // Copy the run of plain bytes up to the next quote or backslash
            // in one go (valid UTF-8, since the input is a &str).
            let run = self.at;
            while self.at < self.s.len() && !matches!(self.s[self.at], b'"' | b'\\') {
                self.at += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[run..self.at]).unwrap());
            match self.peek() {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let c = self.peek().ok_or_else(|| self.err("unfinished escape"))?;
                    self.at += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                // High surrogate: a low one must follow.
                                if !self.s[self.at..].starts_with(b"\\u") {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.at += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("bad low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(code).ok_or_else(|| self.err("invalid code point"))?);
                        }
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                _ => return Err(self.err("unterminated string")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_nesting() {
        let v = Json::parse(r#" {"a": [1, -2.5e3, true, false, null], "b": {"c": "d"}, "e": []} "#).unwrap();
        assert_eq!(v.get("a").unwrap().as_array().unwrap()[1].as_f64(), Some(-2500.0));
        assert_eq!(v.get("a").unwrap().as_array().unwrap()[2].as_bool(), Some(true));
        assert!(v.get("a").unwrap().as_array().unwrap()[4].is_null());
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("d"));
        assert_eq!(v.get("e").unwrap().as_array().unwrap().len(), 0);
        assert_eq!(v.get("missing"), None);
    }

    #[test]
    fn string_escapes_and_surrogates() {
        let v = Json::parse(r#""q\" b\\ s\/ \n\t é 中 😀""#).unwrap();
        assert_eq!(v.as_str(), Some("q\" b\\ s/ \n\t é 中 😀"));
        // Raw (unescaped) UTF-8 passes through untouched.
        assert_eq!(Json::parse("\"Ġthe Ċ ñ\"").unwrap().as_str(), Some("Ġthe Ċ ñ"));
    }

    #[test]
    fn integers_for_shapes_and_ids() {
        let v = Json::parse("[0, 49151, 268435456, 1.5, -1]").unwrap();
        let a = v.as_array().unwrap();
        assert_eq!(a[2].as_usize(), Some(268_435_456));
        assert_eq!(a[3].as_usize(), None);
        assert_eq!(a[4].as_usize(), None);
    }

    #[test]
    fn errors_point_at_the_problem() {
        for bad in ["{", "[1,]", r#"{"a" 1}"#, r#""\ud83d""#, "tru", "1 2", r#""\x""#] {
            assert!(Json::parse(bad).is_err(), "{bad} should not parse");
        }
    }

    /// The real files: SmolLM2's config.json and its 2 MB tokenizer.json.
    #[test]
    fn parses_model_files() {
        let Some(dir) = crate::testutil::golden("models/SmolLM2-135M") else { return };
        let config = Json::parse(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(config.get("hidden_size").unwrap().as_usize(), Some(576));
        let tok = Json::parse(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap()).unwrap();
        let vocab = tok.get("model").unwrap().get("vocab").unwrap().as_object().unwrap();
        assert_eq!(vocab.len(), 49152);
    }
}
