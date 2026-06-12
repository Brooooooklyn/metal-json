//! serde deserialization over parsed metal-json documents.
//!
//! This module does not parse JSON text. It materializes Rust data models
//! from the tape/string buffers already produced by [`Parser`](crate::Parser).
//! Keeping the [`Document`] alive lets serde borrow string
//! fields directly from the document.
//!
//! JSON integers within the `i64`/`u64` range deserialize into `i128`/
//! `u128` targets exactly (including values above 2⁵³). Integers beyond
//! that range were stored as `f64` at parse time — same as simdjson — so
//! `i128`/`u128` targets reject them with an invalid-type error, unlike
//! `serde_json`, which parses the full 128-bit range.

use core::fmt;

use ::serde::Deserializer as _;
use ::serde::de::value::BorrowedStrDeserializer;
use ::serde::de::{
    self, DeserializeSeed, EnumAccess, Error as _, MapAccess, SeqAccess, VariantAccess, Visitor,
};

use crate::document::Document;
use crate::tape::CONTAINER_COUNT_MAX;
use crate::value::{ArrayIter, ObjectIter, Value, ValueKind};

/// Result type for serde materialization from a parsed document.
pub type Result<T> = core::result::Result<T, DeserializeError>;

/// serde deserialization error produced while walking a parsed document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeserializeError {
    message: String,
}

impl DeserializeError {
    /// The human-readable error message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DeserializeError {}

impl de::Error for DeserializeError {
    fn custom<T>(msg: T) -> Self
    where
        T: fmt::Display,
    {
        Self {
            message: msg.to_string(),
        }
    }
}

/// Deserialize the root value of `document`.
///
/// `T` may borrow strings from `document`.
pub fn from_document<'de, T>(document: &'de Document) -> Result<T>
where
    T: ::serde::Deserialize<'de>,
{
    from_value(document.root())
}

/// Deserialize `value`.
///
/// `T` may borrow strings from the value's backing document.
pub fn from_value<'de, T>(value: Value<'de>) -> Result<T>
where
    T: ::serde::Deserialize<'de>,
{
    T::deserialize(value)
}

fn kind_name(kind: ValueKind) -> &'static str {
    match kind {
        ValueKind::Object => "object",
        ValueKind::Array => "array",
        ValueKind::String => "string",
        ValueKind::Int64 | ValueKind::UInt64 | ValueKind::Double => "number",
        ValueKind::Bool => "boolean",
        ValueKind::Null => "null",
    }
}

fn invalid_type(value: Value<'_>, expected: &'static str) -> DeserializeError {
    DeserializeError::custom(format!(
        "invalid type: {}, expected {expected}",
        kind_name(value.kind())
    ))
}

fn invalid_tape(value: Value<'_>, expected: &'static str) -> DeserializeError {
    DeserializeError::custom(format!(
        "corrupted tape: {} value is missing its {expected}",
        kind_name(value.kind())
    ))
}

fn string_value<'de>(value: Value<'de>) -> Result<&'de str> {
    value
        .as_str()
        .ok_or_else(|| invalid_tape(value, "string payload"))
}

fn i64_value(value: Value<'_>) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid_tape(value, "i64 payload"))
}

fn u64_value(value: Value<'_>) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| invalid_tape(value, "u64 payload"))
}

fn f64_value(value: Value<'_>) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| invalid_tape(value, "f64 payload"))
}

fn bool_value(value: Value<'_>) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid_tape(value, "boolean payload"))
}

/// `Some(exact)` size hint, or `None` when the stored count is saturated
/// at [`CONTAINER_COUNT_MAX`] ("this many or more") — serde caps capacity
/// hints anyway, so a pre-walk of a huge container just to count it would
/// be a wasted O(n) tape pass.
fn container_hint(value: Value<'_>) -> Result<Option<usize>> {
    let count = value
        .raw_container_count()
        .ok_or_else(|| invalid_tape(value, "container length"))?;
    Ok((count < CONTAINER_COUNT_MAX).then_some(count as usize))
}

impl<'de> de::Deserializer<'de> for Value<'de> {
    type Error = DeserializeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.kind() {
            ValueKind::Object => visitor.visit_map(ObjectAccess::new(self)?),
            ValueKind::Array => visitor.visit_seq(ArrayAccess::new(self)?),
            ValueKind::String => visitor.visit_borrowed_str(string_value(self)?),
            ValueKind::Int64 => visitor.visit_i64(i64_value(self)?),
            ValueKind::UInt64 => visitor.visit_u64(u64_value(self)?),
            ValueKind::Double => visitor.visit_f64(f64_value(self)?),
            ValueKind::Bool => visitor.visit_bool(bool_value(self)?),
            ValueKind::Null => visitor.visit_unit(),
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_unit()
        } else {
            Err(invalid_type(self, "null"))
        }
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::Array {
            visitor.visit_seq(ArrayAccess::new(self)?)
        } else {
            Err(invalid_type(self, "array"))
        }
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() != ValueKind::Array {
            return Err(invalid_type(self, "array"));
        }
        visit_seq_exact(self, len, visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_tuple(len, visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::Object {
            visitor.visit_map(ObjectAccess::new(self)?)
        } else {
            Err(invalid_type(self, "object"))
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // Like serde_json, structs accept an array of positional fields in
        // declaration order as well as the usual keyed object.
        match self.kind() {
            ValueKind::Object => visitor.visit_map(ObjectAccess::new(self)?),
            ValueKind::Array => visit_seq_exact(self, fields.len(), visitor),
            _ => Err(invalid_type(self, "object or array")),
        }
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.kind() {
            ValueKind::String => {
                visitor.visit_enum(BorrowedStrDeserializer::new(string_value(self)?))
            }
            ValueKind::Object => {
                if self.len() != Some(1) {
                    return Err(DeserializeError::custom(
                        "invalid enum: expected an object with exactly one variant key",
                    ));
                }
                let (variant, value) = self.entries().next().ok_or_else(|| {
                    DeserializeError::custom(
                        "corrupted tape: enum object reports one variant but has no entries",
                    )
                })?;
                visitor.visit_enum(ExternalEnum { variant, value })
            }
            _ => Err(invalid_type(self, "string or single-key object")),
        }
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Some(s) = self.as_str() {
            visitor.visit_borrowed_bytes(s.as_bytes())
        } else {
            self.deserialize_seq(visitor)
        }
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Some(s) = self.as_str() {
            visitor.visit_byte_buf(s.as_bytes().to_vec())
        } else {
            self.deserialize_seq(visitor)
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Some(s) = self.as_str() {
            visitor.visit_borrowed_str(s)
        } else {
            Err(invalid_type(self, "identifier string"))
        }
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let _ = self;
        visitor.visit_unit()
    }

    // Direct happy paths for the common scalar requests: one tag check,
    // then the matching visit. The mismatch fallback is `deserialize_any`,
    // which preserves its cross-kind semantics exactly (e.g. an `f64`
    // target still sees `visit_i64`/`visit_u64` for integer tape values,
    // so serde widens; wrong kinds get the visitor's own rejection).

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::Bool {
            visitor.visit_bool(bool_value(self)?)
        } else {
            self.deserialize_any(visitor)
        }
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::Int64 {
            visitor.visit_i64(i64_value(self)?)
        } else {
            self.deserialize_any(visitor)
        }
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::UInt64 {
            visitor.visit_u64(u64_value(self)?)
        } else {
            self.deserialize_any(visitor)
        }
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::Double {
            visitor.visit_f64(f64_value(self)?)
        } else {
            self.deserialize_any(visitor)
        }
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.kind() == ValueKind::String {
            visitor.visit_borrowed_str(string_value(self)?)
        } else {
            self.deserialize_any(visitor)
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    ::serde::forward_to_deserialize_any! {
        i8 i16 i32 i128 u8 u16 u32 u128 f32 char
    }
}

/// Drive `visitor.visit_seq` over `value`'s elements and reject any the
/// visitor did not consume. Tuple and positional-struct visitors stop after
/// their declared arity, so extras would otherwise be silently dropped;
/// serde_json rejects them when it expects the closing `]`.
fn visit_seq_exact<'de, V>(value: Value<'de>, len: usize, visitor: V) -> Result<V::Value>
where
    V: Visitor<'de>,
{
    let mut access = ArrayAccess::new(value)?;
    let out = visitor.visit_seq(&mut access)?;
    if access.iter.next().is_none() {
        Ok(out)
    } else {
        Err(DeserializeError::invalid_length(
            value.len().unwrap_or(len),
            &"fewer elements in array",
        ))
    }
}

struct ArrayAccess<'de> {
    iter: ArrayIter<'de>,
    /// Exact elements left, or `None` when the stored count is saturated.
    remaining: Option<usize>,
}

impl<'de> ArrayAccess<'de> {
    fn new(value: Value<'de>) -> Result<Self> {
        Ok(Self {
            iter: value.elements(),
            remaining: container_hint(value)?,
        })
    }
}

impl<'de> SeqAccess<'de> for ArrayAccess<'de> {
    type Error = DeserializeError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        let Some(value) = self.iter.next() else {
            return Ok(None);
        };
        self.remaining = self.remaining.map(|n| n.saturating_sub(1));
        seed.deserialize(value).map(Some)
    }

    fn size_hint(&self) -> Option<usize> {
        self.remaining
    }
}

struct ObjectAccess<'de> {
    iter: ObjectIter<'de>,
    pending_value: Option<Value<'de>>,
    /// Exact members left, or `None` when the stored count is saturated.
    remaining: Option<usize>,
}

impl<'de> ObjectAccess<'de> {
    fn new(value: Value<'de>) -> Result<Self> {
        Ok(Self {
            iter: value.entries(),
            pending_value: None,
            remaining: container_hint(value)?,
        })
    }
}

impl<'de> MapAccess<'de> for ObjectAccess<'de> {
    type Error = DeserializeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: DeserializeSeed<'de>,
    {
        let Some((key, value)) = self.iter.next() else {
            return Ok(None);
        };
        self.pending_value = Some(value);
        self.remaining = self.remaining.map(|n| n.saturating_sub(1));
        seed.deserialize(MapKeyDeserializer { key }).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: DeserializeSeed<'de>,
    {
        let value = self.pending_value.take().ok_or_else(|| {
            DeserializeError::custom("serde requested a map value before requesting its key")
        })?;
        seed.deserialize(value)
    }

    fn size_hint(&self) -> Option<usize> {
        self.remaining
    }
}

/// Deserializer for object keys and enum variant names.
///
/// Keys are always strings on the tape, normally delivered with
/// `visit_borrowed_str` — `BorrowedStrDeserializer` rather than
/// `key.into_deserializer()`, which drops the 'de lifetime via `visit_str`,
/// so keys can be deserialized as `&'de str` borrowed from the document.
/// But, like serde_json's map-key deserializer, integer and bool targets
/// parse the key text (JSON object keys are always quoted, so `{"1":...}`
/// is how an integer-keyed map serializes), and newtype-struct keys unwrap
/// to the inner type.
struct MapKeyDeserializer<'de> {
    key: &'de str,
}

/// `deserialize_*` that parses the key text for the requested target,
/// falling back to the borrowed string (and the visitor's own type error)
/// when the key does not parse.
macro_rules! deserialize_parsed_key {
    ($($method:ident => $visit:ident)*) => {
        $(fn $method<V>(self, visitor: V) -> Result<V::Value>
        where
            V: Visitor<'de>,
        {
            match self.key.parse() {
                Ok(parsed) => visitor.$visit(parsed),
                Err(_) => visitor.visit_borrowed_str(self.key),
            }
        })*
    };
}

impl<'de> de::Deserializer<'de> for MapKeyDeserializer<'de> {
    type Error = DeserializeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.key)
    }

    deserialize_parsed_key! {
        deserialize_bool => visit_bool
        deserialize_i8 => visit_i8
        deserialize_i16 => visit_i16
        deserialize_i32 => visit_i32
        deserialize_i64 => visit_i64
        deserialize_i128 => visit_i128
        deserialize_u8 => visit_u8
        deserialize_u16 => visit_u16
        deserialize_u32 => visit_u32
        deserialize_u64 => visit_u64
        deserialize_u128 => visit_u128
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // BorrowedStrDeserializer is a unit-variant-only EnumAccess.
        visitor.visit_enum(BorrowedStrDeserializer::new(self.key))
    }

    ::serde::forward_to_deserialize_any! {
        f32 f64 char str string bytes byte_buf option unit unit_struct seq
        tuple tuple_struct map struct identifier ignored_any
    }
}

struct ExternalEnum<'de> {
    variant: &'de str,
    value: Value<'de>,
}

impl<'de> EnumAccess<'de> for ExternalEnum<'de> {
    type Error = DeserializeError;
    type Variant = ExternalVariant<'de>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: DeserializeSeed<'de>,
    {
        // MapKeyDeserializer for the same reasons as map keys: variant
        // identifiers may be deserialized as `&'de str`, and serde_json
        // parses non-string variant keys (e.g. integer-tagged enums).
        let variant = seed.deserialize(MapKeyDeserializer { key: self.variant })?;
        Ok((variant, ExternalVariant { value: self.value }))
    }
}

struct ExternalVariant<'de> {
    value: Value<'de>,
}

impl<'de> VariantAccess<'de> for ExternalVariant<'de> {
    type Error = DeserializeError;

    fn unit_variant(self) -> Result<()> {
        if self.value.is_null() {
            Ok(())
        } else {
            Err(invalid_type(self.value, "null variant payload"))
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: DeserializeSeed<'de>,
    {
        seed.deserialize(self.value)
    }

    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // deserialize_tuple, not deserialize_seq, so extra elements are
        // rejected here too.
        self.value.deserialize_tuple(len, visitor)
    }

    fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // Same object-or-positional-array acceptance as deserialize_struct.
        match self.value.kind() {
            ValueKind::Object => visitor.visit_map(ObjectAccess::new(self.value)?),
            ValueKind::Array => visit_seq_exact(self.value, fields.len(), visitor),
            _ => Err(invalid_type(self.value, "object or array")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{
        StringBuffer, TAG_END_ARRAY, TAG_END_OBJECT, TAG_START_ARRAY, TAG_START_OBJECT, TapeBuffer,
        int64_bits, make_close, make_final_root, make_int64_marker, make_open, make_root,
        make_string, make_true,
    };

    // -----------------------------------------------------------------
    // size_hint bookkeeping: exact counts decrement to zero, saturated
    // counts ("this many or more") yield no hint at all
    // -----------------------------------------------------------------

    /// `[10, 20, 30]` whose open word stores `count` (hand-built like the
    /// saturated-count tests in `value.rs`, since only a >0xFFFFFF-element
    /// GPU tape would store a saturated count for real).
    fn array_doc(count: u32) -> Document {
        let mut tape = TapeBuffer::new();
        tape.push(0); // root placeholder
        let open = tape.push(0);
        for v in [10i64, 20, 30] {
            tape.push(make_int64_marker());
            tape.push(int64_bits(v));
        }
        let close = tape.push(make_close(TAG_END_ARRAY, open as u32));
        tape.set(open, make_open(TAG_START_ARRAY, (close + 1) as u32, count));
        let final_index = tape.push(make_final_root());
        tape.set(0, make_root(final_index as u64));
        Document::from_parts(tape, StringBuffer::new())
    }

    /// `{"p":true,"q":true}` whose open word stores `count`.
    fn object_doc(count: u32) -> Document {
        let mut tape = TapeBuffer::new();
        let mut strings = StringBuffer::new();
        tape.push(0); // root placeholder
        let open = tape.push(0);
        for key in ["p", "q"] {
            let offset = strings.append_record(key.as_bytes());
            tape.push(make_string(offset));
            tape.push(make_true());
        }
        let close = tape.push(make_close(TAG_END_OBJECT, open as u32));
        tape.set(open, make_open(TAG_START_OBJECT, (close + 1) as u32, count));
        let final_index = tape.push(make_final_root());
        tape.set(0, make_root(final_index as u64));
        Document::from_parts(tape, strings)
    }

    /// Visitor recording the access's `size_hint` before the first item
    /// and after every yielded item, draining the container.
    struct HintRecorder;

    impl<'de> Visitor<'de> for HintRecorder {
        type Value = Vec<Option<usize>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a container")
        }

        fn visit_seq<A>(self, mut seq: A) -> core::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut hints = vec![seq.size_hint()];
            while seq.next_element::<de::IgnoredAny>()?.is_some() {
                hints.push(seq.size_hint());
            }
            Ok(hints)
        }

        fn visit_map<A>(self, mut map: A) -> core::result::Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut hints = vec![map.size_hint()];
            while map
                .next_entry::<de::IgnoredAny, de::IgnoredAny>()?
                .is_some()
            {
                hints.push(map.size_hint());
            }
            Ok(hints)
        }
    }

    #[test]
    fn exact_counts_give_size_hints_that_count_down_per_item() {
        let doc = array_doc(3);
        let hints = doc.root().deserialize_any(HintRecorder).expect("seq");
        assert_eq!(hints, [Some(3), Some(2), Some(1), Some(0)]);

        let doc = object_doc(2);
        let hints = doc.root().deserialize_any(HintRecorder).expect("map");
        assert_eq!(hints, [Some(2), Some(1), Some(0)]);
    }

    #[test]
    fn saturated_counts_give_no_size_hint() {
        let doc = array_doc(CONTAINER_COUNT_MAX);
        let hints = doc.root().deserialize_any(HintRecorder).expect("seq");
        assert_eq!(hints, [None, None, None, None]);

        let doc = object_doc(CONTAINER_COUNT_MAX);
        let hints = doc.root().deserialize_any(HintRecorder).expect("map");
        assert_eq!(hints, [None, None, None]);
    }
}
