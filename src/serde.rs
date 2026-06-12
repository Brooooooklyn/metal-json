//! serde deserialization over parsed metal-json documents.
//!
//! This module does not parse JSON text. It materializes Rust data models
//! from the tape/string buffers already produced by [`Parser`](crate::Parser).
//! Keeping the [`Document`](crate::Document) alive lets serde borrow string
//! fields directly from the document.

use core::fmt;

use ::serde::Deserializer as _;
use ::serde::de::{
    self, DeserializeSeed, EnumAccess, Error as _, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};

use crate::document::Document;
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

fn container_len(value: Value<'_>) -> Result<usize> {
    value
        .len()
        .ok_or_else(|| invalid_tape(value, "container length"))
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

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
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
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_map(visitor)
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
            ValueKind::String => visitor.visit_enum(string_value(self)?.into_deserializer()),
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

    ::serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
    }
}

struct ArrayAccess<'de> {
    iter: ArrayIter<'de>,
    remaining: usize,
}

impl<'de> ArrayAccess<'de> {
    fn new(value: Value<'de>) -> Result<Self> {
        Ok(Self {
            iter: value.elements(),
            remaining: container_len(value)?,
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
        self.remaining = self.remaining.saturating_sub(1);
        seed.deserialize(value).map(Some)
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct ObjectAccess<'de> {
    iter: ObjectIter<'de>,
    pending_value: Option<Value<'de>>,
    remaining: usize,
}

impl<'de> ObjectAccess<'de> {
    fn new(value: Value<'de>) -> Result<Self> {
        Ok(Self {
            iter: value.entries(),
            pending_value: None,
            remaining: container_len(value)?,
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
        self.remaining = self.remaining.saturating_sub(1);
        seed.deserialize(key.into_deserializer()).map(Some)
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
        Some(self.remaining)
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
        let variant = seed.deserialize(self.variant.into_deserializer())?;
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

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.value.deserialize_seq(visitor)
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.value.deserialize_map(visitor)
    }
}
