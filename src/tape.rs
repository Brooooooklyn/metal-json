//! Tape format v1: the typed, simdjson-layout parse result.
//!
//! This module is the **canonical definition** of the tape layout. Two other
//! artifacts mirror it and must stay in lock-step:
//!
//! - `shaders/tape_types.h` — the same constants for MSL kernels. The
//!   [`msl_header_layout_lock`](self#cross-language-layout-lock) unit test
//!   parses that header and asserts every `MJ_*` constant matches the Rust
//!   value, so a divergent edit fails `cargo test`.
//! - `docs/tape-format.md` — the prose spec with a worked example; the
//!   `worked_example_matches_tape_format_doc` test pins that example
//!   word-for-word and byte-for-byte.
//!
//! # Layout summary
//!
//! The tape is a sequence of `u64` words. Every *tape entry* packs an ASCII
//! tag into the top byte and a 56-bit payload below it:
//!
//! ```text
//! word = ((tag as u64) << 56) | payload      payload <= 2^56 - 1
//! ```
//!
//! - `tape[0]`: tag [`TAG_ROOT`], payload = index of the *final* root word.
//!   The final word: tag [`TAG_ROOT`], payload 0.
//! - `{` / `[` open words: payload bits 0..32 = index **one past** the
//!   matching close word; bits 32..56 = number of direct children (object
//!   members or array elements), saturated at [`CONTAINER_COUNT_MAX`].
//! - `}` / `]` close words: payload bits 0..32 = index of the matching open
//!   word.
//! - `"` string words: payload = byte offset into the string buffer where a
//!   `[u32 LE length][utf8 bytes][NUL]` record starts. Content is already
//!   unescaped. Keys and value strings use the same encoding.
//! - `l` / `u` / `d` numbers occupy **two** words: the marker (payload 0)
//!   then the raw `i64` / `u64` / `f64` bits. Type selection mirrors
//!   simdjson: no `.`/`e`/`E` and fits `i64` → `l`; else fits `u64` → `u`;
//!   else `d`.
//! - `t` / `f` / `n` literals: one word, payload 0.

use core::ops::Deref;

// ---------------------------------------------------------------------------
// Layout constants (mirrored in shaders/tape_types.h — keep in lock-step)
// ---------------------------------------------------------------------------

/// Bits the tag byte is shifted left by inside a tape word.
pub const TAG_SHIFT: u32 = 56;

/// Mask selecting the 56 payload bits of a tape word.
pub const PAYLOAD_MASK: u64 = (1 << TAG_SHIFT) - 1;

/// Container payload bits 0..32: matching-word index.
pub const CONTAINER_INDEX_MASK: u64 = 0xFFFF_FFFF;

/// Container payload bits 32..56 hold the direct-child count.
pub const CONTAINER_COUNT_SHIFT: u32 = 32;

/// Maximum storable child count; larger counts saturate to this value
/// (simdjson parity — consumers must treat it as "this many or more").
pub const CONTAINER_COUNT_MAX: u32 = 0xFF_FFFF;

/// Maximum string-buffer offset representable in a string word (56 bits).
pub const STRING_OFFSET_MASK: u64 = PAYLOAD_MASK;

/// String record header: `u32` little-endian unescaped length.
pub const STRING_RECORD_HEADER_BYTES: usize = 4;

/// String record trailer: a single NUL byte after the content.
pub const STRING_RECORD_TRAILER_BYTES: usize = 1;

// ---------------------------------------------------------------------------
// Tags (ASCII, identical to simdjson's tape characters)
// ---------------------------------------------------------------------------

/// Root word (`tape[0]` and the final word).
pub const TAG_ROOT: u8 = b'r';
/// Object open.
pub const TAG_START_OBJECT: u8 = b'{';
/// Object close.
pub const TAG_END_OBJECT: u8 = b'}';
/// Array open.
pub const TAG_START_ARRAY: u8 = b'[';
/// Array close.
pub const TAG_END_ARRAY: u8 = b']';
/// String (key or value); payload = string-buffer record offset.
pub const TAG_STRING: u8 = b'"';
/// Signed 64-bit integer marker; next word holds the `i64` bits.
pub const TAG_INT64: u8 = b'l';
/// Unsigned 64-bit integer marker; next word holds the `u64` value.
pub const TAG_UINT64: u8 = b'u';
/// IEEE-754 double marker; next word holds the `f64` bits.
pub const TAG_DOUBLE: u8 = b'd';
/// Literal `true`.
pub const TAG_TRUE: u8 = b't';
/// Literal `false`.
pub const TAG_FALSE: u8 = b'f';
/// Literal `null`.
pub const TAG_NULL: u8 = b'n';

// ---------------------------------------------------------------------------
// Encode helpers
// ---------------------------------------------------------------------------

/// Pack a tag and a 56-bit payload into one tape word.
///
/// Debug-asserts that `payload` fits in 56 bits; in release the payload is
/// masked (the tag byte can never be corrupted by an oversized payload).
#[inline]
#[must_use]
pub const fn make_entry(tag: u8, payload: u64) -> u64 {
    debug_assert!(payload <= PAYLOAD_MASK, "tape payload exceeds 56 bits");
    ((tag as u64) << TAG_SHIFT) | (payload & PAYLOAD_MASK)
}

/// `tape[0]`: root word pointing at the index of the final root word.
#[inline]
#[must_use]
pub const fn make_root(final_root_index: u64) -> u64 {
    make_entry(TAG_ROOT, final_root_index)
}

/// The final tape word: root tag, payload 0.
#[inline]
#[must_use]
pub const fn make_final_root() -> u64 {
    make_entry(TAG_ROOT, 0)
}

/// Container open word (`{` or `[`).
///
/// `end_index` is the tape index **one past** the matching close word;
/// `count` is the number of direct children (object members or array
/// elements), saturated at [`CONTAINER_COUNT_MAX`].
#[inline]
#[must_use]
pub const fn make_open(tag: u8, end_index: u32, count: u32) -> u64 {
    debug_assert!(
        tag == TAG_START_OBJECT || tag == TAG_START_ARRAY,
        "make_open requires '{{' or '['"
    );
    let saturated = if count > CONTAINER_COUNT_MAX {
        CONTAINER_COUNT_MAX
    } else {
        count
    };
    make_entry(
        tag,
        ((saturated as u64) << CONTAINER_COUNT_SHIFT) | (end_index as u64),
    )
}

/// Container close word (`}` or `]`); `open_index` is the tape index of the
/// matching open word.
#[inline]
#[must_use]
pub const fn make_close(tag: u8, open_index: u32) -> u64 {
    debug_assert!(
        tag == TAG_END_OBJECT || tag == TAG_END_ARRAY,
        "make_close requires '}}' or ']'"
    );
    make_entry(tag, open_index as u64)
}

/// String word; `offset` is the byte offset of the record in the
/// [`StringBuffer`] (56 bits).
#[inline]
#[must_use]
pub const fn make_string(offset: u64) -> u64 {
    debug_assert!(
        offset <= STRING_OFFSET_MASK,
        "string offset exceeds 56 bits"
    );
    make_entry(TAG_STRING, offset)
}

/// `i64` marker word; must be followed by [`int64_bits`] of the value.
#[inline]
#[must_use]
pub const fn make_int64_marker() -> u64 {
    make_entry(TAG_INT64, 0)
}

/// `u64` marker word; must be followed by the value itself.
#[inline]
#[must_use]
pub const fn make_uint64_marker() -> u64 {
    make_entry(TAG_UINT64, 0)
}

/// `f64` marker word; must be followed by [`double_bits`] of the value.
#[inline]
#[must_use]
pub const fn make_double_marker() -> u64 {
    make_entry(TAG_DOUBLE, 0)
}

/// Literal `true` word.
#[inline]
#[must_use]
pub const fn make_true() -> u64 {
    make_entry(TAG_TRUE, 0)
}

/// Literal `false` word.
#[inline]
#[must_use]
pub const fn make_false() -> u64 {
    make_entry(TAG_FALSE, 0)
}

/// Literal `null` word.
#[inline]
#[must_use]
pub const fn make_null() -> u64 {
    make_entry(TAG_NULL, 0)
}

/// Value word following a [`make_int64_marker`]: the raw two's-complement
/// bits of the integer.
#[inline]
#[must_use]
pub const fn int64_bits(value: i64) -> u64 {
    value as u64
}

/// Value word following a [`make_double_marker`]: the raw IEEE-754 bits.
#[inline]
#[must_use]
pub const fn double_bits(value: f64) -> u64 {
    value.to_bits()
}

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------

/// Tag byte of a tape word.
#[inline]
#[must_use]
pub const fn tag(word: u64) -> u8 {
    (word >> TAG_SHIFT) as u8
}

/// 56-bit payload of a tape word.
#[inline]
#[must_use]
pub const fn payload(word: u64) -> u64 {
    word & PAYLOAD_MASK
}

/// Index of the final root word, from `tape[0]`.
#[inline]
#[must_use]
pub const fn root_final_index(word: u64) -> u64 {
    payload(word)
}

/// From an open word: tape index one past the matching close word.
#[inline]
#[must_use]
pub const fn container_end_index(word: u64) -> u32 {
    (word & CONTAINER_INDEX_MASK) as u32
}

/// From an open word: direct-child count (saturated at
/// [`CONTAINER_COUNT_MAX`]).
#[inline]
#[must_use]
pub const fn container_count(word: u64) -> u32 {
    ((word >> CONTAINER_COUNT_SHIFT) & (CONTAINER_COUNT_MAX as u64)) as u32
}

/// From a close word: tape index of the matching open word.
#[inline]
#[must_use]
pub const fn container_open_index(word: u64) -> u32 {
    (word & CONTAINER_INDEX_MASK) as u32
}

/// From a string word: byte offset of the record in the [`StringBuffer`].
#[inline]
#[must_use]
pub const fn string_offset(word: u64) -> u64 {
    payload(word)
}

/// Reinterpret the value word after an `l` marker as `i64`.
#[inline]
#[must_use]
pub const fn int64_from_bits(word: u64) -> i64 {
    word as i64
}

/// Reinterpret the value word after a `d` marker as `f64`.
#[inline]
#[must_use]
pub const fn double_from_bits(word: u64) -> f64 {
    f64::from_bits(word)
}

// ---------------------------------------------------------------------------
// TapeBuffer
// ---------------------------------------------------------------------------

/// Owns the tape words.
///
/// M1 stores a plain `Vec<u64>`. From M2 on, the words may instead live in a
/// shared-storage [`GpuBuffer`](crate::metal::GpuBuffer) written by the GPU
/// pipeline; the storage is therefore fully encapsulated — reading goes
/// through `Deref<Target = [u64]>` and building through
/// [`push`](Self::push)/[`set`](Self::set), so swapping the backing store is
/// not an API change.
#[derive(Clone, Default)]
pub struct TapeBuffer {
    words: Vec<u64>,
}

impl TapeBuffer {
    /// Empty tape.
    #[must_use]
    pub const fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Empty tape with room for `capacity` words.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity),
        }
    }

    /// A tape over already-finished words (the GPU backend's copy-out path;
    /// `words` must be a complete tape-format-v1 encoding). M5 TODO: a
    /// zero-copy variant over the shared `GpuBuffer` replaces this copy.
    pub(crate) fn from_words(words: Vec<u64>) -> Self {
        Self { words }
    }

    /// Append a word, returning its tape index.
    #[inline]
    pub fn push(&mut self, word: u64) -> usize {
        let index = self.words.len();
        self.words.push(word);
        index
    }

    /// Overwrite the word at `index` (used to patch open words once the
    /// matching close index and child count are known).
    ///
    /// # Panics
    /// If `index` is out of bounds.
    #[inline]
    pub fn set(&mut self, index: usize, word: u64) {
        self.words[index] = word;
    }

    /// The tape words. Equivalent to the `Deref` view; handy where deref
    /// coercion does not kick in.
    #[inline]
    #[must_use]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Remove all words, keeping the allocation.
    pub fn clear(&mut self) {
        self.words.clear();
    }
}

impl Deref for TapeBuffer {
    type Target = [u64];

    #[inline]
    fn deref(&self) -> &[u64] {
        &self.words
    }
}

impl std::fmt::Debug for TapeBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeBuffer")
            .field("len", &self.words.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StringBuffer
// ---------------------------------------------------------------------------

/// Owns the unescaped string bytes referenced by `"` tape words.
///
/// Each string is stored as a *record*: `[u32 LE length][content][NUL]`. The
/// length counts only the content bytes; the trailing NUL is a C-string
/// convenience (content may itself contain NUL from `\u0000`, which is why
/// the explicit length exists). Like [`TapeBuffer`], the backing store is
/// encapsulated so it can move into a GPU buffer later without API change.
///
/// Records can be placed two ways:
///
/// - [`append_record`](Self::append_record) packs densely (handy for
///   hand-built test tapes);
/// - [`append_record_at`](Self::append_record_at) /
///   [`pad_to`](Self::pad_to) place records at offsets allocated by the
///   pipeline's raw-length prefix sum (see `docs/tape-format.md`), where
///   escapes that shrink a string leave a **gap** before the next record.
///   The reference backend zero-fills gaps deterministically.
#[derive(Clone, Default)]
pub struct StringBuffer {
    bytes: Vec<u8>,
}

impl StringBuffer {
    /// Empty string buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Empty string buffer with room for `capacity` bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    /// A string buffer over already-finished record bytes (the GPU
    /// backend's copy-out path; `bytes` must hold `[u32 LE len][content]
    /// [NUL]` records at the offsets the tape's `"` words carry). M5 TODO:
    /// a zero-copy variant over the shared `GpuBuffer` replaces this copy.
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Append a `[u32 LE length][content][NUL]` record and return the byte
    /// offset of its start — the value to store via [`make_string`].
    ///
    /// `content` must already be unescaped.
    ///
    /// # Panics
    /// If `content` is longer than `u32::MAX` bytes or the record would push
    /// an offset past 56 bits (both far beyond supported input sizes).
    pub fn append_record(&mut self, content: &[u8]) -> u64 {
        let len = u32::try_from(content.len()).expect("string longer than u32::MAX bytes");
        let offset = self.bytes.len() as u64;
        assert!(
            offset <= STRING_OFFSET_MASK,
            "string buffer offset exceeds 56 bits"
        );
        self.bytes
            .reserve(STRING_RECORD_HEADER_BYTES + content.len() + STRING_RECORD_TRAILER_BYTES);
        self.bytes.extend_from_slice(&len.to_le_bytes());
        self.bytes.extend_from_slice(content);
        self.bytes.push(0);
        offset
    }

    /// Zero-fill the buffer up to (exactly) `offset` bytes.
    ///
    /// Used by the reference pipeline to realize the raw-length prefix-sum
    /// offset scheme of `docs/tape-format.md`: when an escape shrinks a
    /// string, the bytes between the end of its record and the start of the
    /// next allocated slot (or the buffer end) are a gap, zero-filled here
    /// so the reference output is deterministic. A no-op when the buffer is
    /// already `offset` bytes long.
    ///
    /// # Panics
    /// If `offset` is smaller than the current length (records are placed in
    /// document order; padding never moves backwards).
    pub fn pad_to(&mut self, offset: u64) {
        let target = usize::try_from(offset).expect("string offset exceeds usize");
        assert!(
            target >= self.bytes.len(),
            "pad_to({target}) would shrink the buffer (len {})",
            self.bytes.len()
        );
        self.bytes.resize(target, 0);
    }

    /// Append a `[u32 LE length][content][NUL]` record **at** byte `offset`,
    /// zero-filling any gap between the current buffer end and `offset`,
    /// and return `offset` (the value to store via [`make_string`]).
    ///
    /// This is how the reference pipeline places records at the offsets the
    /// raw-length prefix sum allocated (gaps appear when escapes shrink an
    /// earlier string).
    ///
    /// # Panics
    /// As [`pad_to`](Self::pad_to) and
    /// [`append_record`](Self::append_record).
    pub fn append_record_at(&mut self, offset: u64, content: &[u8]) -> u64 {
        self.pad_to(offset);
        self.append_record(content)
    }

    /// Content bytes of the record starting at `offset` (as returned by
    /// [`append_record`](Self::append_record) / decoded by
    /// [`string_offset`]).
    ///
    /// # Panics
    /// If `offset` does not point at a complete record (programmer error —
    /// offsets must come from this buffer's tape).
    #[must_use]
    pub fn record_bytes(&self, offset: u64) -> &[u8] {
        let start = usize::try_from(offset).expect("string offset exceeds usize");
        let header: [u8; STRING_RECORD_HEADER_BYTES] = self.bytes
            [start..start + STRING_RECORD_HEADER_BYTES]
            .try_into()
            .expect("string record header out of bounds");
        let len = u32::from_le_bytes(header) as usize;
        let content_start = start + STRING_RECORD_HEADER_BYTES;
        &self.bytes[content_start..content_start + len]
    }

    /// Content of the record at `offset` as `&str`.
    ///
    /// # Panics
    /// As [`record_bytes`](Self::record_bytes), plus if the content is not
    /// valid UTF-8 (the pipeline validates UTF-8 before strings reach the
    /// buffer, so this is also programmer error).
    #[must_use]
    pub fn record_str(&self, offset: u64) -> &str {
        core::str::from_utf8(self.record_bytes(offset))
            .expect("string record content is not valid UTF-8")
    }

    /// Total bytes in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True if no records have been appended.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Raw buffer contents (records in document order, with zero-filled
    /// gaps wherever the offset scheme left room — see
    /// [`append_record_at`](Self::append_record_at)).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Remove all records, keeping the allocation.
    pub fn clear(&mut self) {
        self.bytes.clear();
    }
}

impl std::fmt::Debug for StringBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringBuffer")
            .field("len", &self.bytes.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TAGS: [u8; 12] = [
        TAG_ROOT,
        TAG_START_OBJECT,
        TAG_END_OBJECT,
        TAG_START_ARRAY,
        TAG_END_ARRAY,
        TAG_STRING,
        TAG_INT64,
        TAG_UINT64,
        TAG_DOUBLE,
        TAG_TRUE,
        TAG_FALSE,
        TAG_NULL,
    ];

    #[test]
    fn tag_values_are_the_ascii_tape_characters() {
        assert_eq!(TAG_ROOT, 0x72);
        assert_eq!(TAG_START_OBJECT, 0x7B);
        assert_eq!(TAG_END_OBJECT, 0x7D);
        assert_eq!(TAG_START_ARRAY, 0x5B);
        assert_eq!(TAG_END_ARRAY, 0x5D);
        assert_eq!(TAG_STRING, 0x22);
        assert_eq!(TAG_INT64, 0x6C);
        assert_eq!(TAG_UINT64, 0x75);
        assert_eq!(TAG_DOUBLE, 0x64);
        assert_eq!(TAG_TRUE, 0x74);
        assert_eq!(TAG_FALSE, 0x66);
        assert_eq!(TAG_NULL, 0x6E);
    }

    #[test]
    fn entry_round_trips_every_tag_and_payload_extremes() {
        for t in ALL_TAGS {
            for p in [0u64, 1, 0xDEAD_BEEF, CONTAINER_INDEX_MASK, PAYLOAD_MASK] {
                let word = make_entry(t, p);
                assert_eq!(tag(word), t, "tag round trip for {t:#x} payload {p:#x}");
                assert_eq!(payload(word), p, "payload round trip for {t:#x}");
            }
        }
    }

    #[test]
    fn max_payload_does_not_bleed_into_tag() {
        let word = make_entry(TAG_STRING, PAYLOAD_MASK);
        assert_eq!(tag(word), TAG_STRING);
        assert_eq!(payload(word), PAYLOAD_MASK);
        assert_eq!(word, 0x22FF_FFFF_FFFF_FFFF);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "tape payload exceeds 56 bits")]
    fn oversized_payload_panics_in_debug() {
        let _ = make_entry(TAG_STRING, PAYLOAD_MASK + 1);
    }

    #[test]
    fn open_word_round_trips_index_and_count() {
        let word = make_open(TAG_START_OBJECT, 12, 2);
        assert_eq!(tag(word), TAG_START_OBJECT);
        assert_eq!(container_end_index(word), 12);
        assert_eq!(container_count(word), 2);
        assert_eq!(word, 0x7B00_0002_0000_000C);

        let word = make_open(TAG_START_ARRAY, u32::MAX, 0);
        assert_eq!(tag(word), TAG_START_ARRAY);
        assert_eq!(container_end_index(word), u32::MAX);
        assert_eq!(container_count(word), 0);
    }

    #[test]
    fn open_word_count_saturates_at_24_bits() {
        // Below the cap: stored exactly.
        let word = make_open(TAG_START_ARRAY, 7, CONTAINER_COUNT_MAX - 1);
        assert_eq!(container_count(word), CONTAINER_COUNT_MAX - 1);

        // At the cap: stored exactly.
        let word = make_open(TAG_START_ARRAY, 7, CONTAINER_COUNT_MAX);
        assert_eq!(container_count(word), CONTAINER_COUNT_MAX);

        // One past and far past: saturated, and the end index is untouched.
        for over in [CONTAINER_COUNT_MAX + 1, u32::MAX] {
            let word = make_open(TAG_START_OBJECT, 7, over);
            assert_eq!(
                container_count(word),
                CONTAINER_COUNT_MAX,
                "count {over:#x}"
            );
            assert_eq!(container_end_index(word), 7);
            assert_eq!(tag(word), TAG_START_OBJECT);
        }
    }

    #[test]
    fn close_word_round_trips_open_index() {
        for index in [0u32, 1, 0x00FF_FFFF, u32::MAX] {
            let word = make_close(TAG_END_OBJECT, index);
            assert_eq!(tag(word), TAG_END_OBJECT);
            assert_eq!(container_open_index(word), index);
        }
        assert_eq!(tag(make_close(TAG_END_ARRAY, 3)), TAG_END_ARRAY);
        assert_eq!(make_close(TAG_END_ARRAY, 3), 0x5D00_0000_0000_0003);
    }

    #[test]
    fn string_word_round_trips_56_bit_offsets() {
        for offset in [0u64, 1, 0xABCD_EF01, STRING_OFFSET_MASK] {
            let word = make_string(offset);
            assert_eq!(tag(word), TAG_STRING);
            assert_eq!(string_offset(word), offset, "offset {offset:#x}");
        }
    }

    #[test]
    fn root_words() {
        let first = make_root(12);
        assert_eq!(tag(first), TAG_ROOT);
        assert_eq!(root_final_index(first), 12);
        assert_eq!(first, 0x7200_0000_0000_000C);

        let last = make_final_root();
        assert_eq!(tag(last), TAG_ROOT);
        assert_eq!(payload(last), 0);
        assert_eq!(last, 0x7200_0000_0000_0000);

        // A giant (56-bit) final index survives.
        assert_eq!(root_final_index(make_root(PAYLOAD_MASK)), PAYLOAD_MASK);
    }

    #[test]
    fn number_markers_have_zero_payload() {
        assert_eq!(tag(make_int64_marker()), TAG_INT64);
        assert_eq!(payload(make_int64_marker()), 0);
        assert_eq!(tag(make_uint64_marker()), TAG_UINT64);
        assert_eq!(payload(make_uint64_marker()), 0);
        assert_eq!(tag(make_double_marker()), TAG_DOUBLE);
        assert_eq!(payload(make_double_marker()), 0);
    }

    #[test]
    fn int64_value_word_round_trips() {
        for v in [0i64, 1, -1, 42, i64::MIN, i64::MAX] {
            assert_eq!(int64_from_bits(int64_bits(v)), v, "i64 {v}");
        }
        assert_eq!(int64_bits(-1), u64::MAX);
    }

    #[test]
    fn double_value_word_round_trips_bit_exactly() {
        for v in [
            0.0f64,
            -0.0,
            2.5,
            -1.0e308,
            f64::MIN_POSITIVE,
            5e-324, // smallest subnormal
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let word = double_bits(v);
            assert_eq!(double_from_bits(word).to_bits(), v.to_bits(), "f64 {v:e}");
        }
        // -0.0 and 0.0 stay distinct at the bit level.
        assert_ne!(double_bits(0.0), double_bits(-0.0));
        // NaN payload bits are preserved verbatim.
        let weird_nan = f64::from_bits(0x7FF8_0000_DEAD_BEEF);
        assert_eq!(
            double_from_bits(double_bits(weird_nan)).to_bits(),
            0x7FF8_0000_DEAD_BEEF
        );
        assert_eq!(double_bits(2.5), 0x4004_0000_0000_0000);
    }

    #[test]
    fn literal_words() {
        assert_eq!(tag(make_true()), TAG_TRUE);
        assert_eq!(payload(make_true()), 0);
        assert_eq!(tag(make_false()), TAG_FALSE);
        assert_eq!(payload(make_false()), 0);
        assert_eq!(tag(make_null()), TAG_NULL);
        assert_eq!(payload(make_null()), 0);
    }

    #[test]
    fn tape_buffer_push_set_and_deref() {
        let mut tape = TapeBuffer::new();
        assert!(tape.is_empty());
        assert_eq!(tape.push(make_root(0)), 0);
        assert_eq!(tape.push(make_null()), 1);
        assert_eq!(tape.push(make_final_root()), 2);
        assert_eq!(tape.len(), 3);

        // Patch tape[0] the way the builder patches open words.
        tape.set(0, make_root(2));
        assert_eq!(root_final_index(tape[0]), 2);

        // Deref view and as_words agree.
        let via_deref: &[u64] = &tape;
        assert_eq!(via_deref, tape.as_words());
        assert_eq!(
            tape.as_words(),
            &[make_root(2), make_null(), make_final_root()]
        );

        tape.clear();
        assert!(tape.is_empty());
    }

    #[test]
    fn string_buffer_record_layout() {
        let mut buf = StringBuffer::new();
        assert!(buf.is_empty());

        // Empty string: 4-byte zero length + NUL.
        let off_empty = buf.append_record(b"");
        assert_eq!(off_empty, 0);
        assert_eq!(buf.as_bytes(), &[0, 0, 0, 0, 0]);
        assert_eq!(buf.record_bytes(off_empty), b"");

        // "a": next record starts right after.
        let off_a = buf.append_record(b"a");
        assert_eq!(off_a, 5);
        assert_eq!(buf.record_str(off_a), "a");
        assert_eq!(&buf.as_bytes()[5..11], &[1, 0, 0, 0, b'a', 0]);

        // Interior NUL (from \u0000) is carried by the explicit length.
        let off_nul = buf.append_record(b"x\0y");
        assert_eq!(off_nul, 11);
        assert_eq!(buf.record_bytes(off_nul), b"x\0y");
        assert_eq!(buf.record_str(off_nul), "x\0y");

        // Total length: 5 + 6 + (4 + 3 + 1).
        assert_eq!(buf.len(), 19);

        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn append_record_at_zero_fills_gaps() {
        let mut buf = StringBuffer::new();

        // First record at 0, no gap.
        assert_eq!(buf.append_record_at(0, b"ab"), 0);
        assert_eq!(buf.len(), 7);

        // Next slot allocated at 10 (as if the first string's raw length
        // were 5): bytes 7..10 are a zero-filled gap.
        assert_eq!(buf.append_record_at(10, b"c"), 10);
        assert_eq!(
            buf.as_bytes(),
            &[
                2, 0, 0, 0, b'a', b'b', 0, // record "ab"
                0, 0, 0, // gap
                1, 0, 0, 0, b'c', 0, // record "c"
            ]
        );
        // Records decode through the gap.
        assert_eq!(buf.record_bytes(0), b"ab");
        assert_eq!(buf.record_str(10), "c");

        // Exact-offset placement (gap of zero bytes) is fine.
        assert_eq!(buf.append_record_at(16, b""), 16);
        assert_eq!(buf.len(), 21);
    }

    #[test]
    fn pad_to_extends_with_zeros_and_is_idempotent() {
        let mut buf = StringBuffer::new();
        buf.append_record(b"x"); // 6 bytes
        buf.pad_to(9); // trailing gap, e.g. raw len 4 → slot 9
        assert_eq!(buf.len(), 9);
        assert_eq!(&buf.as_bytes()[6..], &[0, 0, 0]);
        // Padding to the current length is a no-op.
        buf.pad_to(9);
        assert_eq!(buf.len(), 9);
        assert_eq!(buf.record_str(0), "x");
    }

    #[test]
    #[should_panic(expected = "would shrink the buffer")]
    fn pad_to_panics_on_backward_offsets() {
        let mut buf = StringBuffer::new();
        buf.append_record(b"hello");
        buf.pad_to(3);
    }

    #[test]
    fn string_buffer_length_prefix_is_little_endian_u32() {
        let mut buf = StringBuffer::new();
        let content = vec![b'z'; 0x0102]; // 258 bytes: exercises the second LE byte
        let off = buf.append_record(&content);
        assert_eq!(off, 0);
        assert_eq!(&buf.as_bytes()[..4], &[0x02, 0x01, 0x00, 0x00]);
        assert_eq!(buf.record_bytes(off), &content[..]);
        // Trailing NUL after the content.
        assert_eq!(buf.as_bytes()[4 + 0x0102], 0);
        assert_eq!(buf.len(), 4 + 0x0102 + 1);
    }

    #[test]
    fn string_buffer_multibyte_utf8_round_trips() {
        let mut buf = StringBuffer::new();
        let s = "héllo \u{1F600} wörld";
        let off = buf.append_record(s.as_bytes());
        assert_eq!(buf.record_str(off), s);
    }

    /// The worked example from docs/tape-format.md, word for word and byte
    /// for byte: `{"a":[1,2.5],"b":"x\n"}`. If this test changes, the doc
    /// must change with it (and vice versa).
    ///
    /// String record offsets follow the raw-length prefix-sum scheme:
    /// slot size = raw bytes between the quotes + 5 ("a" → 6 at 0,
    /// "b" → 6 at 6, "x\n" → raw 3 → 8 at 12). The escape shrinks the last
    /// record to 7 bytes, so the buffer ends with one zero gap byte.
    #[test]
    fn worked_example_matches_tape_format_doc() {
        let mut tape = TapeBuffer::new();
        let mut strings = StringBuffer::new();

        // Build the way the reference pipeline does: placeholders for words
        // whose payloads are only known once the matching close is seen,
        // string records placed at prefix-sum-allocated offsets.
        let root = tape.push(0); // [0] placeholder root
        let obj_open = tape.push(0); // [1] placeholder '{'
        let off_a = strings.append_record_at(0, b"a"); // raw len 1 → slot 6
        tape.push(make_string(off_a)); // [2] "a"
        let arr_open = tape.push(0); // [3] placeholder '['
        tape.push(make_int64_marker()); // [4] 'l'
        tape.push(int64_bits(1)); // [5] 1
        tape.push(make_double_marker()); // [6] 'd'
        tape.push(double_bits(2.5)); // [7] 2.5
        let arr_close = tape.push(make_close(TAG_END_ARRAY, arr_open as u32)); // [8] ']'
        tape.set(
            arr_open,
            make_open(TAG_START_ARRAY, (arr_close + 1) as u32, 2),
        );
        let off_b = strings.append_record_at(6, b"b"); // raw len 1 → slot 6
        tape.push(make_string(off_b)); // [9] "b"
        let off_xn = strings.append_record_at(12, b"x\n"); // unescaped: 'x', LF
        tape.push(make_string(off_xn)); // [10] "x\n"
        strings.pad_to(20); // raw len 3 → slot 8: one trailing gap byte
        let obj_close = tape.push(make_close(TAG_END_OBJECT, obj_open as u32)); // [11] '}'
        tape.set(
            obj_open,
            make_open(TAG_START_OBJECT, (obj_close + 1) as u32, 2),
        );
        let final_root = tape.push(make_final_root()); // [12] 'r'
        tape.set(root, make_root(final_root as u64));

        // Exact words as documented in docs/tape-format.md.
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C, // [0]  r -> 12
            0x7B00_0002_0000_000C, // [1]  { end=12 count=2
            0x2200_0000_0000_0000, // [2]  " offset=0  ("a")
            0x5B00_0002_0000_0009, // [3]  [ end=9 count=2
            0x6C00_0000_0000_0000, // [4]  l
            0x0000_0000_0000_0001, // [5]  1
            0x6400_0000_0000_0000, // [6]  d
            0x4004_0000_0000_0000, // [7]  2.5
            0x5D00_0000_0000_0003, // [8]  ] open=3
            0x2200_0000_0000_0006, // [9]  " offset=6  ("b")
            0x2200_0000_0000_000C, // [10] " offset=12 ("x\n")
            0x7D00_0000_0000_0001, // [11] } open=1
            0x7200_0000_0000_0000, // [12] r -> 0
        ];
        assert_eq!(tape.as_words(), &expected);

        // Exact string buffer bytes as documented (20 = 6 + 6 + 8; the
        // last slot holds a 7-byte record + 1 zero gap byte).
        let expected_strings: [u8; 20] = [
            0x01, 0x00, 0x00, 0x00, 0x61, 0x00, // "a"
            0x01, 0x00, 0x00, 0x00, 0x62, 0x00, // "b"
            0x02, 0x00, 0x00, 0x00, 0x78, 0x0A, 0x00, // "x\n"
            0x00, // gap: the \n escape shrank 3 raw bytes to 2
        ];
        assert_eq!(strings.as_bytes(), &expected_strings);

        // And the decode helpers walk it correctly.
        assert_eq!(root_final_index(tape[0]), 12);
        assert_eq!(container_end_index(tape[1]), 12);
        assert_eq!(container_count(tape[1]), 2);
        assert_eq!(strings.record_str(string_offset(tape[2])), "a");
        assert_eq!(container_end_index(tape[3]), 9);
        assert_eq!(container_count(tape[3]), 2);
        assert_eq!(int64_from_bits(tape[5]), 1);
        assert_eq!(double_from_bits(tape[7]), 2.5);
        assert_eq!(container_open_index(tape[8]), 3);
        assert_eq!(strings.record_str(string_offset(tape[10])), "x\n");
        assert_eq!(container_open_index(tape[11]), 1);
    }

    // -----------------------------------------------------------------
    // Cross-language layout lock
    // -----------------------------------------------------------------

    /// Parse every `constant constexpr <type> NAME = <int>;` line of an MSL
    /// header into `(name, value)` pairs. Comments (`// ...`) are ignored;
    /// decimal and `0x` hex literals are accepted, with optional `u`/`l`
    /// suffixes.
    fn parse_msl_constants(header: &str) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        for raw in header.lines() {
            let line = raw.split("//").next().unwrap_or("").trim();
            let Some(rest) = line.strip_prefix("constant constexpr ") else {
                continue;
            };
            let decl = rest
                .split(';')
                .next()
                .unwrap_or_else(|| panic!("missing ';' in tape_types.h line: {raw}"));
            let (lhs, rhs) = decl
                .split_once('=')
                .unwrap_or_else(|| panic!("missing '=' in tape_types.h line: {raw}"));
            let name = lhs
                .split_whitespace()
                .last()
                .unwrap_or_else(|| panic!("missing name in tape_types.h line: {raw}"))
                .to_owned();
            let text = rhs.trim().trim_end_matches(['u', 'U', 'l', 'L']).to_owned();
            let value =
                if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                    u64::from_str_radix(hex, 16)
                } else {
                    text.parse::<u64>()
                }
                .unwrap_or_else(|e| panic!("cannot parse value of {name} ({text:?}): {e}"));
            out.push((name, value));
        }
        out
    }

    /// THE cross-language layout lock: shaders/tape_types.h must define
    /// exactly the constants below with exactly these values. Any one-sided
    /// edit — Rust or MSL — fails this test.
    #[test]
    fn msl_header_layout_lock() {
        let header = include_str!("../shaders/tape_types.h");
        let parsed = parse_msl_constants(header);
        assert!(
            !parsed.is_empty(),
            "parsed no constants from shaders/tape_types.h — \
             header format changed without updating the test parser?"
        );

        let expected: &[(&str, u64)] = &[
            ("MJ_TAPE_TAG_SHIFT", TAG_SHIFT as u64),
            ("MJ_TAPE_PAYLOAD_MASK", PAYLOAD_MASK),
            ("MJ_TAG_ROOT", TAG_ROOT as u64),
            ("MJ_TAG_START_OBJECT", TAG_START_OBJECT as u64),
            ("MJ_TAG_END_OBJECT", TAG_END_OBJECT as u64),
            ("MJ_TAG_START_ARRAY", TAG_START_ARRAY as u64),
            ("MJ_TAG_END_ARRAY", TAG_END_ARRAY as u64),
            ("MJ_TAG_STRING", TAG_STRING as u64),
            ("MJ_TAG_INT64", TAG_INT64 as u64),
            ("MJ_TAG_UINT64", TAG_UINT64 as u64),
            ("MJ_TAG_DOUBLE", TAG_DOUBLE as u64),
            ("MJ_TAG_TRUE", TAG_TRUE as u64),
            ("MJ_TAG_FALSE", TAG_FALSE as u64),
            ("MJ_TAG_NULL", TAG_NULL as u64),
            ("MJ_CONTAINER_INDEX_MASK", CONTAINER_INDEX_MASK),
            ("MJ_CONTAINER_COUNT_SHIFT", CONTAINER_COUNT_SHIFT as u64),
            ("MJ_CONTAINER_COUNT_MAX", CONTAINER_COUNT_MAX as u64),
            ("MJ_STRING_OFFSET_MASK", STRING_OFFSET_MASK),
            (
                "MJ_STRING_RECORD_HEADER_BYTES",
                STRING_RECORD_HEADER_BYTES as u64,
            ),
            (
                "MJ_STRING_RECORD_TRAILER_BYTES",
                STRING_RECORD_TRAILER_BYTES as u64,
            ),
        ];

        // Every expected constant must exist in the header with the Rust value.
        for (name, want) in expected {
            let got = parsed
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("shaders/tape_types.h is missing constant {name}"));
            assert_eq!(
                got.1, *want,
                "shaders/tape_types.h {name} = {:#x}, but src/tape.rs says {want:#x}",
                got.1
            );
        }

        // And the header must not define layout constants Rust does not know
        // about (catches drift in the other direction).
        for (name, value) in &parsed {
            assert!(
                expected.iter().any(|(n, _)| n == name),
                "shaders/tape_types.h defines {name} = {value:#x}, \
                 which src/tape.rs's layout-lock test does not know about — \
                 add it to both sides"
            );
        }

        // No duplicate definitions.
        for (i, (name, _)) in parsed.iter().enumerate() {
            assert!(
                !parsed[..i].iter().any(|(n, _)| n == name),
                "shaders/tape_types.h defines {name} twice"
            );
        }
    }
}
