//! `Value<'doc>` cursor API over a parsed [`Document`].
//!
//! A [`Value`] is a copyable `(document, tape index)` pair pointing at one
//! value word on the tape (format spec: `docs/tape-format.md`). Navigation
//! is allocation-free:
//!
//! - [`get`](Value::get) / [`entries`](Value::entries) walk an object's
//!   direct members, skipping nested containers in O(1) via the end-index
//!   payload of open words;
//! - [`at`](Value::at) / [`elements`](Value::elements) do the same for
//!   arrays;
//! - [`len`](Value::len) is O(1) from the open word's count bits, except
//!   for saturated counts ([`CONTAINER_COUNT_MAX`] means "this many or
//!   more"), where it falls back to iterating for the exact count.

use core::iter::FusedIterator;

use crate::document::Document;
use crate::tape::{
    self, CONTAINER_COUNT_MAX, TAG_DOUBLE, TAG_FALSE, TAG_INT64, TAG_NULL, TAG_START_ARRAY,
    TAG_START_OBJECT, TAG_STRING, TAG_TRUE, TAG_UINT64,
};

/// The eight kinds a JSON tape value can have.
///
/// `Int64`/`UInt64`/`Double` mirror the tape's three number tags
/// (`l`/`u`/`d`); see `docs/tape-format.md` for the selection rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueKind {
    /// `{...}`
    Object,
    /// `[...]`
    Array,
    /// `"..."`
    String,
    /// Number stored as `i64` (tape tag `l`).
    Int64,
    /// Number stored as `u64` (tape tag `u`).
    UInt64,
    /// Number stored as `f64` (tape tag `d`).
    Double,
    /// `true` or `false`.
    Bool,
    /// `null`.
    Null,
}

/// A lightweight, copyable cursor pointing at one value on a document's
/// tape.
///
/// Obtained from [`Document::root`] and refined with [`get`](Self::get) /
/// [`at`](Self::at) / the iterators. All accessors are non-panicking on
/// kind mismatch: they return `None` (or an empty iterator) instead.
#[derive(Clone, Copy)]
pub struct Value<'doc> {
    doc: &'doc Document,
    tape_index: usize,
}

impl<'doc> Value<'doc> {
    /// Cursor at `tape_index`, which must hold a *value* word (open word,
    /// string, number marker, or literal — never a close/root word or the
    /// second word of a number).
    pub(crate) fn new(doc: &'doc Document, tape_index: usize) -> Self {
        Self { doc, tape_index }
    }

    /// The tape word this cursor points at.
    #[inline]
    fn word(&self) -> u64 {
        self.doc.tape()[self.tape_index]
    }

    /// The word following this one (the value word of a two-word number).
    #[inline]
    fn next_word(&self) -> u64 {
        self.doc.tape()[self.tape_index + 1]
    }

    /// Which kind of JSON value this cursor points at.
    ///
    /// # Panics
    /// If the cursor points at a non-value word — impossible for cursors
    /// produced by this crate; it would indicate a corrupted tape.
    #[must_use]
    pub fn kind(&self) -> ValueKind {
        match tape::tag(self.word()) {
            TAG_START_OBJECT => ValueKind::Object,
            TAG_START_ARRAY => ValueKind::Array,
            TAG_STRING => ValueKind::String,
            TAG_INT64 => ValueKind::Int64,
            TAG_UINT64 => ValueKind::UInt64,
            TAG_DOUBLE => ValueKind::Double,
            TAG_TRUE | TAG_FALSE => ValueKind::Bool,
            TAG_NULL => ValueKind::Null,
            other => unreachable!(
                "tape index {} holds non-value tag {other:#04x} — corrupted tape",
                self.tape_index
            ),
        }
    }

    /// Look up an object member by key. `None` if this is not an object or
    /// no member has that key.
    ///
    /// Linear scan over the direct members in document order (nested
    /// containers are skipped in O(1)); with duplicate keys, the **first**
    /// match wins, like simdjson's DOM `at_key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Value<'doc>> {
        self.entries().find_map(|(k, v)| (k == key).then_some(v))
    }

    /// The array element at `index` (0-based). `None` if this is not an
    /// array or the index is out of bounds.
    ///
    /// Linear scan over the direct elements (each skipped in O(1)).
    #[must_use]
    pub fn at(&self, index: usize) -> Option<Value<'doc>> {
        self.elements().nth(index)
    }

    /// The string content if this is a string value (already unescaped),
    /// else `None`. May contain interior NULs (from `\u0000`).
    #[must_use]
    pub fn as_str(&self) -> Option<&'doc str> {
        let word = self.word();
        (tape::tag(word) == TAG_STRING)
            .then(|| self.doc.strings().record_str(tape::string_offset(word)))
    }

    /// The number if this value has kind [`ValueKind::Int64`], else `None`.
    /// No cross-kind conversion (use [`as_f64`](Self::as_f64) to widen).
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        (tape::tag(self.word()) == TAG_INT64).then(|| tape::int64_from_bits(self.next_word()))
    }

    /// The number if this value has kind [`ValueKind::UInt64`], else
    /// `None`. No cross-kind conversion (use [`as_f64`](Self::as_f64) to
    /// widen).
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        (tape::tag(self.word()) == TAG_UINT64).then(|| self.next_word())
    }

    /// The number as `f64`. `Double` values are returned bit-exactly;
    /// `Int64`/`UInt64` values are widened (lossless up to 2⁵³; beyond
    /// that the cast rounds to nearest, matching simdjson's `get_double`).
    /// `None` for non-number kinds.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match tape::tag(self.word()) {
            TAG_DOUBLE => Some(tape::double_from_bits(self.next_word())),
            TAG_INT64 => Some(tape::int64_from_bits(self.next_word()) as f64),
            TAG_UINT64 => Some(self.next_word() as f64),
            _ => None,
        }
    }

    /// The boolean if this is `true` or `false`, else `None`.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match tape::tag(self.word()) {
            TAG_TRUE => Some(true),
            TAG_FALSE => Some(false),
            _ => None,
        }
    }

    /// `true` iff this value is `null`.
    #[must_use]
    pub fn is_null(&self) -> bool {
        tape::tag(self.word()) == TAG_NULL
    }

    /// Number of direct children: members for objects, elements for
    /// arrays; `None` for scalars.
    ///
    /// O(1) from the open word's count bits — except when the stored count
    /// is saturated at [`CONTAINER_COUNT_MAX`] ("this many or more"), in
    /// which case the container is walked for the exact count.
    #[must_use]
    pub fn len(&self) -> Option<usize> {
        let word = self.word();
        match tape::tag(word) {
            TAG_START_OBJECT => Some(match tape::container_count(word) {
                CONTAINER_COUNT_MAX => self.entries().count(),
                exact => exact as usize,
            }),
            TAG_START_ARRAY => Some(match tape::container_count(word) {
                CONTAINER_COUNT_MAX => self.elements().count(),
                exact => exact as usize,
            }),
            _ => None,
        }
    }

    /// `Some(true)` for `{}` / `[]`, `Some(false)` for non-empty
    /// containers, `None` for scalars. Always O(1) — an empty container's
    /// close word directly follows its open word.
    #[must_use]
    pub fn is_empty(&self) -> Option<bool> {
        let word = self.word();
        match tape::tag(word) {
            TAG_START_OBJECT | TAG_START_ARRAY => {
                Some(tape::container_end_index(word) as usize == self.tape_index + 2)
            }
            _ => None,
        }
    }

    /// Iterator over an object's direct members as `(key, value)` pairs,
    /// in document order (duplicates included verbatim). Empty if this
    /// value is not an object.
    ///
    /// Nested containers are skipped in O(1) via their end-index payload,
    /// so a full iteration costs O(direct members), not O(descendants).
    /// Saturated child counts do not affect iteration — it runs to the
    /// container's close word.
    #[must_use]
    pub fn entries(&self) -> ObjectIter<'doc> {
        let word = self.word();
        if tape::tag(word) == TAG_START_OBJECT {
            ObjectIter {
                doc: self.doc,
                cursor: self.tape_index + 1,
                end: tape::container_end_index(word) as usize - 1,
            }
        } else {
            ObjectIter {
                doc: self.doc,
                cursor: 0,
                end: 0,
            }
        }
    }

    /// Iterator over an array's direct elements, in document order. Empty
    /// if this value is not an array.
    ///
    /// Same O(1) container-skip and saturation behavior as
    /// [`entries`](Self::entries).
    #[must_use]
    pub fn elements(&self) -> ArrayIter<'doc> {
        let word = self.word();
        if tape::tag(word) == TAG_START_ARRAY {
            ArrayIter {
                doc: self.doc,
                cursor: self.tape_index + 1,
                end: tape::container_end_index(word) as usize - 1,
            }
        } else {
            ArrayIter {
                doc: self.doc,
                cursor: 0,
                end: 0,
            }
        }
    }
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("tape_index", &self.tape_index)
            .field("kind", &self.kind())
            .finish()
    }
}

/// Tape index immediately after the value starting at `index`: O(1) for
/// containers (open words store the index one past their close word),
/// `index + 2` for two-word numbers, `index + 1` for everything else.
fn skip_value(tape: &[u64], index: usize) -> usize {
    let word = tape[index];
    match tape::tag(word) {
        TAG_START_OBJECT | TAG_START_ARRAY => tape::container_end_index(word) as usize,
        TAG_INT64 | TAG_UINT64 | TAG_DOUBLE => index + 2,
        _ => index + 1,
    }
}

/// Iterator over an object's direct members; see [`Value::entries`].
#[derive(Clone, Debug)]
pub struct ObjectIter<'doc> {
    doc: &'doc Document,
    /// Tape index of the next member's key word.
    cursor: usize,
    /// Tape index of the object's close word (exclusive bound).
    end: usize,
}

impl<'doc> Iterator for ObjectIter<'doc> {
    type Item = (&'doc str, Value<'doc>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let tape = self.doc.tape();
        let key_word = tape[self.cursor];
        debug_assert_eq!(
            tape::tag(key_word),
            TAG_STRING,
            "object member at tape index {} does not start with a string key",
            self.cursor
        );
        let key = self.doc.strings().record_str(tape::string_offset(key_word));
        let value = Value::new(self.doc, self.cursor + 1);
        self.cursor = skip_value(tape, self.cursor + 1);
        Some((key, value))
    }
}

impl FusedIterator for ObjectIter<'_> {}

/// Iterator over an array's direct elements; see [`Value::elements`].
#[derive(Clone, Debug)]
pub struct ArrayIter<'doc> {
    doc: &'doc Document,
    /// Tape index of the next element's value word.
    cursor: usize,
    /// Tape index of the array's close word (exclusive bound).
    end: usize,
}

impl<'doc> Iterator for ArrayIter<'doc> {
    type Item = Value<'doc>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let value = Value::new(self.doc, self.cursor);
        self.cursor = skip_value(self.doc.tape(), self.cursor);
        Some(value)
    }
}

impl FusedIterator for ArrayIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{
        StringBuffer, TAG_END_ARRAY, TAG_END_OBJECT, TapeBuffer, int64_bits, make_close,
        make_double_marker, make_false, make_final_root, make_int64_marker, make_null, make_open,
        make_root, make_string, make_true, make_uint64_marker,
    };

    // -----------------------------------------------------------------
    // Hand-built tape helper (exercises tape.rs the way the pipelines do)
    // -----------------------------------------------------------------

    /// Builds a valid tape by hand: placeholder root + open words patched on
    /// close, child counts tracked per container (members for objects,
    /// elements for arrays).
    struct TapeBuilder {
        tape: TapeBuffer,
        strings: StringBuffer,
        /// (open word index, open tag, direct-child count so far)
        opens: Vec<(usize, u8, u32)>,
    }

    impl TapeBuilder {
        fn new() -> Self {
            let mut tape = TapeBuffer::new();
            tape.push(0); // placeholder root, patched in finish()
            Self {
                tape,
                strings: StringBuffer::new(),
                opens: Vec::new(),
            }
        }

        /// Count a new direct child of the innermost container. For objects
        /// the member was already counted at its key, so only arrays count
        /// values.
        fn note_value(&mut self) {
            if let Some(top) = self.opens.last_mut()
                && top.1 == TAG_START_ARRAY
            {
                top.2 += 1;
            }
        }

        fn open_object(&mut self) {
            self.note_value();
            let index = self.tape.push(0);
            self.opens.push((index, TAG_START_OBJECT, 0));
        }

        fn open_array(&mut self) {
            self.note_value();
            let index = self.tape.push(0);
            self.opens.push((index, TAG_START_ARRAY, 0));
        }

        fn close(&mut self) {
            let (open_index, open_tag, count) = self.opens.pop().expect("close without open");
            let close_tag = if open_tag == TAG_START_OBJECT {
                TAG_END_OBJECT
            } else {
                TAG_END_ARRAY
            };
            let close_index = self.tape.push(make_close(close_tag, open_index as u32));
            self.tape.set(
                open_index,
                make_open(open_tag, (close_index + 1) as u32, count),
            );
        }

        /// Object member key (counts the member).
        fn key(&mut self, key: &str) {
            let top = self.opens.last_mut().expect("key outside an object");
            assert_eq!(top.1, TAG_START_OBJECT, "key outside an object");
            top.2 += 1;
            let offset = self.strings.append_record(key.as_bytes());
            self.tape.push(make_string(offset));
        }

        fn string(&mut self, content: &str) {
            self.string_bytes(content.as_bytes());
        }

        /// Raw-byte string value (e.g. unescaped `\u0000` → interior NUL).
        fn string_bytes(&mut self, content: &[u8]) {
            self.note_value();
            let offset = self.strings.append_record(content);
            self.tape.push(make_string(offset));
        }

        fn int(&mut self, value: i64) {
            self.note_value();
            self.tape.push(make_int64_marker());
            self.tape.push(int64_bits(value));
        }

        fn uint(&mut self, value: u64) {
            self.note_value();
            self.tape.push(make_uint64_marker());
            self.tape.push(value);
        }

        fn double(&mut self, value: f64) {
            self.note_value();
            self.tape.push(make_double_marker());
            self.tape.push(tape::double_bits(value));
        }

        fn boolean(&mut self, value: bool) {
            self.note_value();
            self.tape
                .push(if value { make_true() } else { make_false() });
        }

        fn null(&mut self) {
            self.note_value();
            self.tape.push(make_null());
        }

        fn finish(mut self) -> Document {
            assert!(self.opens.is_empty(), "unclosed container");
            let final_index = self.tape.push(make_final_root());
            self.tape.set(0, make_root(final_index as u64));
            Document::from_parts(self.tape, self.strings)
        }
    }

    /// The worked example of docs/tape-format.md:
    /// `{"a":[1,2.5],"b":"x\n"}` (the `\n` already unescaped).
    fn worked_example() -> Document {
        let mut b = TapeBuilder::new();
        b.open_object();
        b.key("a");
        b.open_array();
        b.int(1);
        b.double(2.5);
        b.close();
        b.key("b");
        b.string("x\n");
        b.close();
        b.finish()
    }

    // -----------------------------------------------------------------
    // Builder sanity: it must reproduce the pinned worked example exactly
    // -----------------------------------------------------------------

    #[test]
    fn builder_reproduces_the_worked_example_tape() {
        let doc = worked_example();
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C,
            0x7B00_0002_0000_000C,
            0x2200_0000_0000_0000,
            0x5B00_0002_0000_0009,
            0x6C00_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x6400_0000_0000_0000,
            0x4004_0000_0000_0000,
            0x5D00_0000_0000_0003,
            0x2200_0000_0000_0006,
            0x2200_0000_0000_000C,
            0x7D00_0000_0000_0001,
            0x7200_0000_0000_0000,
        ];
        assert_eq!(doc.tape(), &expected);
    }

    // -----------------------------------------------------------------
    // Kinds and scalar accessors
    // -----------------------------------------------------------------

    #[test]
    fn every_scalar_kind_and_accessor() {
        // ["s", 1, -7, 9223372036854775808, 2.5, true, false, null]
        let mut b = TapeBuilder::new();
        b.open_array();
        b.string("s");
        b.int(1);
        b.int(-7);
        b.uint(9_223_372_036_854_775_808); // i64::MAX + 1 → 'u' tag
        b.double(2.5);
        b.boolean(true);
        b.boolean(false);
        b.null();
        b.close();
        let doc = b.finish();
        let root = doc.root();

        assert_eq!(root.kind(), ValueKind::Array);
        assert_eq!(root.len(), Some(8));
        assert_eq!(root.is_empty(), Some(false));

        let v = root.at(0).unwrap();
        assert_eq!(v.kind(), ValueKind::String);
        assert_eq!(v.as_str(), Some("s"));

        let v = root.at(1).unwrap();
        assert_eq!(v.kind(), ValueKind::Int64);
        assert_eq!(v.as_i64(), Some(1));

        let v = root.at(2).unwrap();
        assert_eq!(v.kind(), ValueKind::Int64);
        assert_eq!(v.as_i64(), Some(-7));

        let v = root.at(3).unwrap();
        assert_eq!(v.kind(), ValueKind::UInt64);
        assert_eq!(v.as_u64(), Some(9_223_372_036_854_775_808));

        let v = root.at(4).unwrap();
        assert_eq!(v.kind(), ValueKind::Double);
        assert_eq!(v.as_f64(), Some(2.5));

        let v = root.at(5).unwrap();
        assert_eq!(v.kind(), ValueKind::Bool);
        assert_eq!(v.as_bool(), Some(true));

        let v = root.at(6).unwrap();
        assert_eq!(v.kind(), ValueKind::Bool);
        assert_eq!(v.as_bool(), Some(false));

        let v = root.at(7).unwrap();
        assert_eq!(v.kind(), ValueKind::Null);
        assert!(v.is_null());

        // Out of bounds.
        assert!(root.at(8).is_none());
    }

    #[test]
    fn accessors_are_none_on_kind_mismatch() {
        let mut b = TapeBuilder::new();
        b.open_array();
        b.int(1);
        b.uint(u64::MAX);
        b.double(0.5);
        b.string("x");
        b.boolean(true);
        b.null();
        b.close();
        let doc = b.finish();
        let root = doc.root();

        let int = root.at(0).unwrap();
        let uint = root.at(1).unwrap();
        let double = root.at(2).unwrap();
        let string = root.at(3).unwrap();
        let boolean = root.at(4).unwrap();
        let null = root.at(5).unwrap();

        // Strict: no cross-kind number conversion outside as_f64.
        assert_eq!(int.as_u64(), None);
        assert_eq!(int.as_str(), None);
        assert_eq!(int.as_bool(), None);
        assert!(!int.is_null());
        assert_eq!(uint.as_i64(), None);
        assert_eq!(double.as_i64(), None);
        assert_eq!(double.as_u64(), None);
        assert_eq!(string.as_i64(), None);
        assert_eq!(string.as_f64(), None);
        assert_eq!(boolean.as_f64(), None);
        assert_eq!(boolean.as_str(), None);
        assert_eq!(null.as_bool(), None);
        assert_eq!(null.as_f64(), None);

        // Scalars have no length, no emptiness, no children.
        for v in [int, uint, double, string, boolean, null] {
            assert_eq!(v.len(), None);
            assert_eq!(v.is_empty(), None);
            assert_eq!(v.entries().count(), 0);
            assert_eq!(v.elements().count(), 0);
            assert!(v.get("k").is_none());
            assert!(v.at(0).is_none());
        }

        // Containers are not scalars.
        assert_eq!(root.as_i64(), None);
        assert_eq!(root.as_u64(), None);
        assert_eq!(root.as_f64(), None);
        assert_eq!(root.as_str(), None);
        assert_eq!(root.as_bool(), None);
        assert!(!root.is_null());
    }

    #[test]
    fn as_f64_widens_from_int_kinds() {
        let mut b = TapeBuilder::new();
        b.open_array();
        b.int(5);
        b.int(-3);
        b.int(i64::MAX); // > 2^53: rounds to nearest like simdjson
        b.uint(u64::MAX);
        b.double(-0.0);
        b.close();
        let doc = b.finish();
        let root = doc.root();

        assert_eq!(root.at(0).unwrap().as_f64(), Some(5.0));
        assert_eq!(root.at(1).unwrap().as_f64(), Some(-3.0));
        assert_eq!(root.at(2).unwrap().as_f64(), Some(i64::MAX as f64));
        assert_eq!(root.at(3).unwrap().as_f64(), Some(u64::MAX as f64));
        // Double passthrough is bit-exact: -0.0 keeps its sign bit.
        assert_eq!(
            root.at(4).unwrap().as_f64().unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
    }

    // -----------------------------------------------------------------
    // Objects: get, entries, duplicates, unicode, misses
    // -----------------------------------------------------------------

    #[test]
    fn object_get_and_entries_on_the_worked_example() {
        let doc = worked_example();
        let root = doc.root();
        assert_eq!(root.kind(), ValueKind::Object);
        assert_eq!(root.len(), Some(2));

        let a = root.get("a").unwrap();
        assert_eq!(a.kind(), ValueKind::Array);
        assert_eq!(a.len(), Some(2));
        assert_eq!(a.at(0).unwrap().as_i64(), Some(1));
        assert_eq!(a.at(1).unwrap().as_f64(), Some(2.5));

        let b = root.get("b").unwrap();
        assert_eq!(b.as_str(), Some("x\n"));

        // Miss: absent key, prefix of a key, and key of a nested member.
        assert!(root.get("c").is_none());
        assert!(root.get("").is_none());
        assert!(root.get("ab").is_none());

        let keys: Vec<&str> = root.entries().map(|(k, _)| k).collect();
        assert_eq!(keys, ["a", "b"]);
    }

    #[test]
    fn duplicate_keys_first_match_wins() {
        // {"k":1,"k":2,"other":3}
        let mut b = TapeBuilder::new();
        b.open_object();
        b.key("k");
        b.int(1);
        b.key("k");
        b.int(2);
        b.key("other");
        b.int(3);
        b.close();
        let doc = b.finish();
        let root = doc.root();

        assert_eq!(root.get("k").unwrap().as_i64(), Some(1));
        // But iteration sees both, verbatim, in document order.
        let pairs: Vec<(&str, i64)> = root
            .entries()
            .map(|(k, v)| (k, v.as_i64().unwrap()))
            .collect();
        assert_eq!(pairs, [("k", 1), ("k", 2), ("other", 3)]);
        assert_eq!(root.len(), Some(3));
    }

    #[test]
    fn unicode_and_exotic_keys() {
        // {"héllo":1, "😀":2, "": 3, "nul\0key":4}
        let mut b = TapeBuilder::new();
        b.open_object();
        b.key("héllo");
        b.int(1);
        b.key("😀");
        b.int(2);
        b.key("");
        b.int(3);
        b.key("nul\0key");
        b.int(4);
        b.close();
        let doc = b.finish();
        let root = doc.root();

        assert_eq!(root.get("héllo").unwrap().as_i64(), Some(1));
        assert_eq!(root.get("😀").unwrap().as_i64(), Some(2));
        assert_eq!(root.get("").unwrap().as_i64(), Some(3));
        assert_eq!(root.get("nul\0key").unwrap().as_i64(), Some(4));
        // Similar-but-different keys miss.
        assert!(root.get("hello").is_none());
        assert!(root.get("nulkey").is_none());

        let keys: Vec<&str> = root.entries().map(|(k, _)| k).collect();
        assert_eq!(keys, ["héllo", "😀", "", "nul\0key"]);
    }

    #[test]
    fn string_values_with_interior_nul() {
        let mut b = TapeBuilder::new();
        b.open_array();
        b.string_bytes(b"x\0y"); // unescaped \u0000
        b.close();
        let doc = b.finish();
        assert_eq!(doc.root().at(0).unwrap().as_str(), Some("x\0y"));
    }

    // -----------------------------------------------------------------
    // Empty containers
    // -----------------------------------------------------------------

    #[test]
    fn empty_object_and_array() {
        // {"o":{},"a":[]}
        let mut b = TapeBuilder::new();
        b.open_object();
        b.key("o");
        b.open_object();
        b.close();
        b.key("a");
        b.open_array();
        b.close();
        b.close();
        let doc = b.finish();
        let root = doc.root();

        let o = root.get("o").unwrap();
        assert_eq!(o.kind(), ValueKind::Object);
        assert_eq!(o.len(), Some(0));
        assert_eq!(o.is_empty(), Some(true));
        assert_eq!(o.entries().count(), 0);
        assert!(o.get("anything").is_none());

        let a = root.get("a").unwrap();
        assert_eq!(a.kind(), ValueKind::Array);
        assert_eq!(a.len(), Some(0));
        assert_eq!(a.is_empty(), Some(true));
        assert_eq!(a.elements().count(), 0);
        assert!(a.at(0).is_none());
    }

    #[test]
    fn empty_containers_as_root() {
        let mut b = TapeBuilder::new();
        b.open_object();
        b.close();
        let doc = b.finish();
        assert_eq!(doc.root().len(), Some(0));
        assert_eq!(doc.root().is_empty(), Some(true));

        let mut b = TapeBuilder::new();
        b.open_array();
        b.close();
        let doc = b.finish();
        assert_eq!(doc.root().len(), Some(0));
        assert_eq!(doc.root().is_empty(), Some(true));
    }

    // -----------------------------------------------------------------
    // Iteration skips nested containers in O(1)
    // -----------------------------------------------------------------

    #[test]
    fn object_iteration_skips_nested_containers() {
        // {"a":{"x":1,"y":[1,2,3]}, "b":[10,[20],{"z":30}], "c":7}
        let mut b = TapeBuilder::new();
        b.open_object();
        b.key("a");
        b.open_object();
        b.key("x");
        b.int(1);
        b.key("y");
        b.open_array();
        b.int(1);
        b.int(2);
        b.int(3);
        b.close();
        b.close();
        b.key("b");
        b.open_array();
        b.int(10);
        b.open_array();
        b.int(20);
        b.close();
        b.open_object();
        b.key("z");
        b.int(30);
        b.close();
        b.close();
        b.key("c");
        b.int(7);
        b.close();
        let doc = b.finish();
        let root = doc.root();

        // Direct members only — nothing from inside "a" or "b" leaks out.
        let summary: Vec<(&str, ValueKind)> = root.entries().map(|(k, v)| (k, v.kind())).collect();
        assert_eq!(
            summary,
            [
                ("a", ValueKind::Object),
                ("b", ValueKind::Array),
                ("c", ValueKind::Int64),
            ]
        );
        assert_eq!(root.len(), Some(3));
        assert_eq!(root.get("c").unwrap().as_i64(), Some(7));

        // And the nested values are intact when navigated into.
        let a = root.get("a").unwrap();
        assert_eq!(a.len(), Some(2));
        assert_eq!(a.get("y").unwrap().len(), Some(3));
        assert_eq!(a.get("y").unwrap().at(2).unwrap().as_i64(), Some(3));

        let b_val = root.get("b").unwrap();
        let kinds: Vec<ValueKind> = b_val.elements().map(|v| v.kind()).collect();
        assert_eq!(
            kinds,
            [ValueKind::Int64, ValueKind::Array, ValueKind::Object]
        );
        assert_eq!(b_val.at(1).unwrap().at(0).unwrap().as_i64(), Some(20));
        assert_eq!(b_val.at(2).unwrap().get("z").unwrap().as_i64(), Some(30));
    }

    #[test]
    fn deeply_nested_arrays_navigate_correctly() {
        // [[[[42]]]]
        let mut b = TapeBuilder::new();
        for _ in 0..4 {
            b.open_array();
        }
        b.int(42);
        for _ in 0..4 {
            b.close();
        }
        let doc = b.finish();

        let mut v = doc.root();
        for _ in 0..3 {
            assert_eq!(v.kind(), ValueKind::Array);
            assert_eq!(v.len(), Some(1));
            v = v.at(0).unwrap();
        }
        assert_eq!(v.at(0).unwrap().as_i64(), Some(42));
    }

    #[test]
    fn iterators_are_fused_and_copyable_cursors_stay_valid() {
        let doc = worked_example();
        let root = doc.root();

        let mut it = root.entries();
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert!(it.next().is_none()); // fused

        // Value is Copy: both copies navigate independently.
        let a = root.get("a").unwrap();
        let a2 = a;
        assert_eq!(a.at(0).unwrap().as_i64(), Some(1));
        assert_eq!(a2.at(1).unwrap().as_f64(), Some(2.5));
    }

    // -----------------------------------------------------------------
    // Count saturation: 0xFFFFFF means "this many or more" → iterate
    // -----------------------------------------------------------------

    #[test]
    fn saturated_array_count_falls_back_to_iteration() {
        // Hand-build an array whose open word stores the saturated count
        // even though it only has 3 elements (as a >0xFFFFFF-element GPU
        // tape would for its real count).
        let mut tape = TapeBuffer::new();
        tape.push(0); // root placeholder
        let open = tape.push(0);
        for v in [10i64, 20, 30] {
            tape.push(make_int64_marker());
            tape.push(int64_bits(v));
        }
        let close = tape.push(make_close(TAG_END_ARRAY, open as u32));
        tape.set(
            open,
            make_open(TAG_START_ARRAY, (close + 1) as u32, CONTAINER_COUNT_MAX),
        );
        let final_index = tape.push(make_final_root());
        tape.set(0, make_root(final_index as u64));
        let doc = Document::from_parts(tape, StringBuffer::new());
        let root = doc.root();

        // The stored count really is saturated...
        assert_eq!(tape::container_count(doc.tape()[1]), CONTAINER_COUNT_MAX);
        // ...so len() iterates and reports the exact count.
        assert_eq!(root.len(), Some(3));
        assert_eq!(root.is_empty(), Some(false));
        let values: Vec<i64> = root.elements().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(values, [10, 20, 30]);
        assert_eq!(root.at(2).unwrap().as_i64(), Some(30));
        assert!(root.at(3).is_none());
    }

    #[test]
    fn saturated_object_count_falls_back_to_iteration() {
        // {"p":true,"q":null} with a saturated stored count.
        let mut tape = TapeBuffer::new();
        let mut strings = StringBuffer::new();
        tape.push(0);
        let open = tape.push(0);
        let off_p = strings.append_record(b"p");
        tape.push(make_string(off_p));
        tape.push(make_true());
        let off_q = strings.append_record(b"q");
        tape.push(make_string(off_q));
        tape.push(make_null());
        let close = tape.push(make_close(TAG_END_OBJECT, open as u32));
        tape.set(
            open,
            make_open(TAG_START_OBJECT, (close + 1) as u32, CONTAINER_COUNT_MAX),
        );
        let final_index = tape.push(make_final_root());
        tape.set(0, make_root(final_index as u64));
        let doc = Document::from_parts(tape, strings);
        let root = doc.root();

        assert_eq!(tape::container_count(doc.tape()[1]), CONTAINER_COUNT_MAX);
        assert_eq!(root.len(), Some(2));
        assert_eq!(root.get("p").unwrap().as_bool(), Some(true));
        assert!(root.get("q").unwrap().is_null());
        assert!(root.get("r").is_none());
        let keys: Vec<&str> = root.entries().map(|(k, _)| k).collect();
        assert_eq!(keys, ["p", "q"]);
    }

    // -----------------------------------------------------------------
    // Root scalars and Debug
    // -----------------------------------------------------------------

    #[test]
    fn scalar_roots() {
        let mut b = TapeBuilder::new();
        b.int(-99);
        let doc = b.finish();
        assert_eq!(doc.root().kind(), ValueKind::Int64);
        assert_eq!(doc.root().as_i64(), Some(-99));
        assert_eq!(doc.root().len(), None);

        let mut b = TapeBuilder::new();
        b.string("top");
        let doc = b.finish();
        assert_eq!(doc.root().as_str(), Some("top"));

        let mut b = TapeBuilder::new();
        b.null();
        let doc = b.finish();
        assert!(doc.root().is_null());
    }

    #[test]
    fn debug_shows_kind_and_index() {
        let doc = worked_example();
        let dbg = format!("{:?}", doc.root());
        assert!(dbg.contains("Object"), "{dbg}");
        assert!(dbg.contains("tape_index: 1"), "{dbg}");
    }
}
