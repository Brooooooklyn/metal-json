//! `Document`: owns the parse result (tape + string buffer).
//!
//! A [`Document`] is self-contained — it never borrows the input bytes. It
//! owns exactly two things: the tape words ([`TapeBuffer`]) and the
//! unescaped string records ([`StringBuffer`]) that `"` tape words point
//! into. The layout of both is tape format v1, specified in
//! `docs/tape-format.md` and pinned by the tests in `src/tape.rs`.
//!
//! From M2 on, the buffers may live in shared-storage GPU memory written by
//! the Metal pipeline; [`TapeBuffer`]/[`StringBuffer`] already encapsulate
//! their backing store for exactly that swap, so nothing here changes.

use crate::tape::{self, StringBuffer, TapeBuffer};
use crate::value::Value;

/// A parsed JSON document: the tape and the unescaped string buffer.
///
/// Produced by [`Parser::parse`](crate::Parser::parse). Navigation goes
/// through [`root`](Self::root), which returns a copyable [`Value`] cursor;
/// [`tape`](Self::tape) and [`strings`](Self::strings) expose the raw
/// buffers for power users and tests.
#[derive(Debug)]
pub struct Document {
    tape: TapeBuffer,
    strings: StringBuffer,
}

impl Document {
    /// Assemble a document from a finished tape + string buffer pair.
    ///
    /// The pair must be a complete, valid tape-format-v1 encoding (root
    /// words present, containers patched); both backends guarantee that
    /// before constructing a `Document` — and on the GPU backend the
    /// rejection contract guarantees the buffers are never even copied out
    /// on a failed parse.
    pub(crate) fn from_parts(tape: TapeBuffer, strings: StringBuffer) -> Self {
        debug_assert!(
            tape.len() >= 3,
            "a valid tape has at least 3 words (root, value, root)"
        );
        debug_assert_eq!(
            tape::tag(tape[0]),
            tape::TAG_ROOT,
            "tape[0] must be the root word"
        );
        debug_assert_eq!(
            tape::root_final_index(tape[0]) as usize,
            tape.len() - 1,
            "tape[0] must point at the final root word"
        );
        Self { tape, strings }
    }

    /// Cursor to the root value (the word after `tape[0]`).
    #[must_use]
    pub fn root(&self) -> Value<'_> {
        Value::new(self, 1)
    }

    /// The raw tape words, for power users and tests.
    ///
    /// See `docs/tape-format.md` for the word encoding. String words hold
    /// offsets into [`strings`](Self::strings).
    #[must_use]
    pub fn tape(&self) -> &[u64] {
        self.tape.as_words()
    }

    /// The unescaped string buffer the tape's `"` words point into, for
    /// power users and tests.
    #[must_use]
    pub fn strings(&self) -> &StringBuffer {
        &self.strings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{
        TAG_ROOT, make_final_root, make_null, make_root, make_string, root_final_index, tag,
    };
    use crate::value::ValueKind;

    /// Smallest valid document: `null` → `[r→2, n, r→0]`.
    fn null_doc() -> Document {
        let mut tape = TapeBuffer::new();
        tape.push(make_root(2));
        tape.push(make_null());
        tape.push(make_final_root());
        Document::from_parts(tape, StringBuffer::new())
    }

    /// The M5 pin: a `Document` is `Send + Sync` no matter which backing
    /// store its buffers use (the GPU-backed storage carries audited
    /// `Send`/`Sync` impls on `GpuBuffer` and an `Arc<ScratchPool>`).
    /// Compile-time; covers both storage variants because the bound is on
    /// the type, not a value.
    #[test]
    fn document_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Document>();
        assert_send_sync::<TapeBuffer>();
        assert_send_sync::<StringBuffer>();
    }

    #[test]
    fn root_is_the_word_after_tape_zero() {
        let doc = null_doc();
        assert_eq!(doc.root().kind(), ValueKind::Null);
        assert!(doc.root().is_null());
    }

    #[test]
    fn tape_exposes_the_raw_words() {
        let doc = null_doc();
        let words = doc.tape();
        assert_eq!(words.len(), 3);
        assert_eq!(tag(words[0]), TAG_ROOT);
        assert_eq!(root_final_index(words[0]), 2);
        assert_eq!(words, &[make_root(2), make_null(), make_final_root()]);
    }

    #[test]
    fn strings_exposes_the_string_buffer() {
        // Document with root value `"hé"`.
        let mut tape = TapeBuffer::new();
        let mut strings = StringBuffer::new();
        tape.push(make_root(2));
        let off = strings.append_record("hé".as_bytes());
        tape.push(make_string(off));
        tape.push(make_final_root());
        let doc = Document::from_parts(tape, strings);

        assert_eq!(doc.root().kind(), ValueKind::String);
        assert_eq!(doc.root().as_str(), Some("hé"));
        // The raw buffer is reachable and decodes the same record.
        assert_eq!(doc.strings().record_str(off), "hé");
        // Exact record bytes: [u32 LE len = 3][h é(2 bytes)][NUL].
        assert_eq!(
            doc.strings().as_bytes(),
            &[3, 0, 0, 0, 0x68, 0xC3, 0xA9, 0x00]
        );
    }
}
