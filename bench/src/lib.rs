//! Benchmark support library: a safe wrapper over the C++ simdjson FFI shim
//! (`cpp/shim.cpp`) plus dataset-loading helpers shared by the criterion
//! suite and the smoke tests.
//!
//! # Methodology notes
//!
//! - simdjson requires `SIMDJSON_PADDING` readable bytes after the document.
//!   [`PaddedBuf`] allocates and copies **once, outside the timed region**,
//!   so the timed call is exactly parse-to-tape (+ a linear tape walk that
//!   fills [`SjStats`], defeating dead-code elimination).
//! - **Symmetric tape walk**: the metal-json contender gets the *same*
//!   treatment — [`metal_stats`] walks metal-json's tape inside the timed
//!   closure, computing the equivalent [`SjStats`]. Both parsers therefore
//!   do the same end-to-end work (parse to tape + shallow stats walk), both
//!   sides defeat dead-code elimination, and an untimed per-dataset check
//!   asserts the two stats are equal — proof both parsers really parsed the
//!   same document to an equivalent tape.
//! - [`PageAligned`] loads a file into a 16 KiB page-aligned allocation —
//!   the layout metal-json's zero-copy `bytesNoCopy` input path wants once
//!   the GPU backend lands. The CPU backends accept it too, so all
//!   contenders parse byte-identical input.

use std::ffi::{c_int, c_void};
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use metal_json::Document;
use metal_json::tape::{
    self, TAG_DOUBLE, TAG_FALSE, TAG_INT64, TAG_NULL, TAG_START_ARRAY, TAG_START_OBJECT,
    TAG_STRING, TAG_TRUE, TAG_UINT64,
};

/// Page size used for metal-json-friendly aligned input (Apple Silicon
/// 16 KiB pages; `MTLBuffer bytesNoCopy` requires page alignment).
pub const PAGE_ALIGN: usize = 16384;

/// Stats returned by the shim's shallow tape walk. Mirrors `SjStats` in
/// `cpp/shim.cpp` — see there for the exact node/byte/xor semantics.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SjStats {
    /// Scalars + object keys + container openings.
    pub node_count: u64,
    /// Sum of unescaped byte lengths of all strings (keys and values).
    pub string_bytes: u64,
    /// XOR of the raw 64-bit payloads of all numbers.
    pub number_xor: u64,
}

/// The metal-json side of the [`SjStats`] contract: a shallow walk over a
/// [`Document`]'s tape computing the same stats the C++ shim's tape walk
/// computes (see `cpp/shim.cpp`). This function is **the** definition of the
/// metal-json ↔ simdjson stats mapping; the criterion harness runs it inside
/// the timed closure (symmetric DCE-defeating work) and asserts, untimed and
/// once per dataset, that it equals the shim's stats.
///
/// # Mapping
///
/// Tape format v1 (see `docs/tape-format.md`) deliberately mirrors
/// simdjson's tape: the same ASCII tags (`{ [ } ] " l u d t f n r`), the
/// same two-word number encoding (marker, then the raw `i64`/`u64`/`f64`
/// payload bits), the same `l`-vs-`u`-vs-`d` number-type-selection policy,
/// the same `[u32 LE length][bytes]` string records, and object keys on the
/// tape as ordinary `"` words. The mapping is therefore the **identity**:
///
/// - `node_count`  — every `"`, number marker, `t`/`f`/`n`, `{` and `[`
///   counts 1; closers and the root sentinels count 0. Identical counting
///   rule on both tapes, so the counts must be *equal*, not just mapped.
/// - `string_bytes` — sum of the `u32 LE` unescaped lengths of all string
///   records reached from `"` words (keys and values).
/// - `number_xor` — XOR of the raw 64-bit payload words following `l`/`u`/
///   `d` markers (two's-complement bits for integers, IEEE-754 bits for
///   doubles). Bit-exact f64 parsing on both sides makes this comparable.
///
/// No known divergences. If one appears (e.g. a number-type policy drift),
/// the per-dataset verification fails loudly; `string_bytes` and
/// `number_xor` must always match exactly, while `node_count` would get a
/// documented mapping here.
///
/// Like the shim's walk (and like `src/value.rs`), the number payload words
/// are *skipped* as tape entries — only their bits enter the XOR.
#[must_use]
pub fn metal_stats(doc: &Document) -> SjStats {
    let words = doc.tape();
    let strings = doc.strings().as_bytes();
    let end = tape::root_final_index(words[0]) as usize;
    let mut stats = SjStats::default();
    let mut i = 1;
    while i < end {
        let word = words[i];
        match tape::tag(word) {
            TAG_STRING => {
                let off = tape::string_offset(word) as usize;
                let len: [u8; 4] = strings[off..off + 4]
                    .try_into()
                    .expect("string record header in bounds");
                stats.string_bytes += u64::from(u32::from_le_bytes(len));
                stats.node_count += 1;
            }
            TAG_INT64 | TAG_UINT64 | TAG_DOUBLE => {
                // The payload lives in the next tape word; skip it as an
                // entry, fold its raw bits into the XOR.
                i += 1;
                stats.number_xor ^= words[i];
                stats.node_count += 1;
            }
            TAG_TRUE | TAG_FALSE | TAG_NULL | TAG_START_OBJECT | TAG_START_ARRAY => {
                stats.node_count += 1;
            }
            _ => {} // '}' ']' 'r': count 0, like the shim.
        }
        i += 1;
    }
    stats
}

unsafe extern "C" {
    fn sj_parser_new() -> *mut c_void;
    fn sj_parser_free(p: *mut c_void);
    fn sj_padding() -> usize;
    fn sj_alloc_padded(len: usize) -> *mut u8;
    fn sj_free_padded(p: *mut u8);
    fn sj_parse_tape(p: *mut c_void, ptr: *const u8, len: usize, out: *mut SjStats) -> c_int;
}

/// simdjson's required input padding, in bytes.
#[must_use]
pub fn simdjson_padding() -> usize {
    unsafe { sj_padding() }
}

/// A reusable `simdjson::dom::parser`. Like metal-json's `Parser`, its
/// internal tape buffers are allocated once and reused across parses.
pub struct SjParser {
    raw: NonNull<c_void>,
}

impl SjParser {
    /// Create a parser.
    ///
    /// # Panics
    ///
    /// Panics if the C++ allocation fails.
    #[must_use]
    pub fn new() -> Self {
        let raw = unsafe { sj_parser_new() };
        Self {
            raw: NonNull::new(raw).expect("simdjson parser allocation failed"),
        }
    }

    /// Parse a padded document to simdjson's tape and return the tape-walk
    /// stats. On failure returns the raw `simdjson::error_code` as `i32`.
    ///
    /// # Errors
    ///
    /// The nonzero `simdjson::error_code` for invalid documents.
    pub fn parse(&self, input: &PaddedBuf) -> Result<SjStats, i32> {
        let mut stats = SjStats::default();
        let rc = unsafe {
            sj_parse_tape(
                self.raw.as_ptr(),
                input.as_ptr(),
                input.len(),
                &raw mut stats,
            )
        };
        if rc == 0 { Ok(stats) } else { Err(rc) }
    }
}

impl Default for SjParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SjParser {
    fn drop(&mut self) {
        unsafe { sj_parser_free(self.raw.as_ptr()) };
    }
}

// A dom::parser is a plain heap object; moving it across threads is fine
// (criterion may run setup and measurement on different threads). It is NOT
// Sync — `parse` mutates internal buffers through &self on the C++ side.
unsafe impl Send for SjParser {}

/// Input buffer with `SIMDJSON_PADDING` trailing bytes, allocated via the
/// shim so padding is handled outside the timed parse call. Derefs to the
/// unpadded document bytes.
pub struct PaddedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

impl PaddedBuf {
    /// Copy `bytes` into a fresh padded allocation (padding zeroed).
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        let pad = simdjson_padding();
        let ptr = unsafe { sj_alloc_padded(bytes.len()) };
        let ptr = NonNull::new(ptr).expect("padded allocation failed");
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
            // Zero the padding: simdjson only requires it readable, but
            // deterministic content keeps runs reproducible.
            std::ptr::write_bytes(ptr.as_ptr().add(bytes.len()), 0, pad);
        }
        Self {
            ptr,
            len: bytes.len(),
        }
    }
}

impl Deref for PaddedBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PaddedBuf {
    fn drop(&mut self) {
        unsafe { sj_free_padded(self.ptr.as_ptr()) };
    }
}

unsafe impl Send for PaddedBuf {}
unsafe impl Sync for PaddedBuf {}

/// A 16 KiB page-aligned copy of a document — the input layout metal-json's
/// zero-copy path wants. Derefs to the document bytes.
pub struct PageAligned {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl PageAligned {
    /// Copy `bytes` into a fresh page-aligned allocation.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        // Round the allocation up to whole pages (bytesNoCopy also wants a
        // page-multiple length); never allocate zero bytes.
        let capacity = bytes.len().div_ceil(PAGE_ALIGN).max(1) * PAGE_ALIGN;
        let layout = std::alloc::Layout::from_size_align(capacity, PAGE_ALIGN)
            .expect("page-aligned layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("page-aligned allocation failed");
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
        }
        Self {
            ptr,
            len: bytes.len(),
            capacity,
        }
    }
}

impl Deref for PageAligned {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PageAligned {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.capacity, PAGE_ALIGN)
            .expect("page-aligned layout");
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

unsafe impl Send for PageAligned {}
unsafe impl Sync for PageAligned {}

/// The dataset directory: `$METAL_JSON_BENCH_DATA` if set, otherwise
/// `<workspace>/data/bench` (gitignored; populate with
/// `cargo run -p xtask -- fetch-data` / `gen-data`).
#[must_use]
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("METAL_JSON_BENCH_DATA") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate lives inside the workspace")
        .join("data")
        .join("bench")
}

/// All `*.json` files under [`data_dir`], as `(file_stem, path)` sorted by
/// file size (small datasets bench first). Empty when the directory is
/// missing or empty.
#[must_use]
pub fn list_datasets() -> Vec<(String, PathBuf)> {
    let dir = data_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<(u64, String, PathBuf)> = entries
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                return None;
            }
            let stem = path.file_stem()?.to_str()?.to_owned();
            let size = std::fs::metadata(&path).ok()?.len();
            Some((size, stem, path))
        })
        .collect();
    found.sort();
    found.into_iter().map(|(_, stem, p)| (stem, p)).collect()
}

/// Read a whole file.
///
/// # Errors
///
/// Propagates the underlying I/O error.
pub fn load(path: &Path) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Read a whole file into a simdjson-padded buffer.
///
/// # Errors
///
/// Propagates the underlying I/O error.
pub fn load_padded(path: &Path) -> io::Result<PaddedBuf> {
    Ok(PaddedBuf::from_slice(&std::fs::read(path)?))
}

/// Read a whole file into a page-aligned buffer (metal-json input layout).
///
/// # Errors
///
/// Propagates the underlying I/O error.
pub fn load_page_aligned(path: &Path) -> io::Result<PageAligned> {
    Ok(PageAligned::from_slice(&std::fs::read(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_is_nonzero() {
        assert!(simdjson_padding() >= 32);
    }

    #[test]
    fn padded_buf_roundtrips_bytes() {
        let buf = PaddedBuf::from_slice(b"[1,2,3]");
        assert_eq!(&*buf, b"[1,2,3]");
    }

    #[test]
    fn page_aligned_is_aligned_and_roundtrips() {
        let buf = PageAligned::from_slice(b"{\"a\":true}");
        assert_eq!(buf.as_ptr() as usize % PAGE_ALIGN, 0);
        assert_eq!(&*buf, b"{\"a\":true}");
    }

    #[test]
    fn shim_parses_a_tiny_document_with_exact_stats() {
        let parser = SjParser::new();
        let input = PaddedBuf::from_slice(br#"[1,2.5,"x"]"#);
        let stats = parser.parse(&input).expect("valid JSON must parse");
        // 1 array + 2 numbers + 1 string.
        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.string_bytes, 1);
        // 1i64 bits ^ 2.5f64 bits.
        assert_eq!(stats.number_xor, 1u64 ^ 2.5f64.to_bits());
    }

    #[test]
    fn shim_rejects_invalid_json() {
        let parser = SjParser::new();
        let input = PaddedBuf::from_slice(b"[1,");
        let err = parser.parse(&input).expect_err("must reject");
        assert_ne!(err, 0);
    }

    #[test]
    fn parser_is_reusable() {
        let parser = SjParser::new();
        let a = PaddedBuf::from_slice(b"{\"k\":[true,false,null]}");
        let b = PaddedBuf::from_slice(b"42");
        let sa = parser.parse(&a).unwrap();
        // k + array + true + false + null + object = 6 nodes.
        assert_eq!(sa.node_count, 6);
        let sb = parser.parse(&b).unwrap();
        assert_eq!(sb.node_count, 1);
        assert_eq!(sb.number_xor, 42);
    }

    /// The symmetric-stats contract on small documents: metal-json's tape
    /// walk ([`metal_stats`]) equals the C++ shim's tape walk, field for
    /// field. Uses the default metal-json backend (GPU where a device
    /// exists, the CPU oracle otherwise — both produce the same tape).
    #[test]
    fn metal_stats_match_the_shim_on_small_documents() {
        let metal = metal_json::Parser::new().expect("metal-json parser (GPU or CPU oracle)");
        let sj = SjParser::new();
        let cases: &[&[u8]] = &[
            br#"[1,2.5,"x"]"#,
            br#"{"k":[true,false,null]}"#,
            b"42",
            br#""just a string""#,
            br#"{"a":[1,2.5],"b":"x\n"}"#, // escapes: unescaped length counts
            br#"[{},[],{"":""},-0.0,18446744073709551615,-9223372036854775808]"#,
            br#"{"dup":1,"dup":2,"nested":{"deep":[[[42]]]}}"#,
        ];
        for &case in cases {
            let doc = metal.parse(case).expect("metal-json parses");
            let ours = metal_stats(&doc);
            let theirs = sj
                .parse(&PaddedBuf::from_slice(case))
                .expect("simdjson parses");
            assert_eq!(
                ours,
                theirs,
                "stats diverge on {:?}",
                String::from_utf8_lossy(case)
            );
        }
    }

    /// The walk skips number payload words as entries but XORs their bits —
    /// pinned against hand-computed values (mirrors the shim's tiny-doc
    /// test above).
    #[test]
    fn metal_stats_values_are_exact() {
        let metal = metal_json::Parser::new().expect("metal-json parser");
        let doc = metal.parse(br#"[1,2.5,"x"]"#).unwrap();
        let stats = metal_stats(&doc);
        assert_eq!(stats.node_count, 4); // array + 2 numbers + 1 string
        assert_eq!(stats.string_bytes, 1);
        assert_eq!(stats.number_xor, 1u64 ^ 2.5f64.to_bits());
    }
}
