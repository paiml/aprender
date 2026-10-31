//! Deserializer half: reads the bincode 1.x legacy layout.
use crate::{Error, Result};
use serde::de::{self, DeserializeSeed, IntoDeserializer, Visitor};
use std::io::Read;

pub(crate) struct Deserializer<R: Read> {
    r: R,
}

impl<R: Read> Deserializer<R> {
    pub(crate) fn new(r: R) -> Self {
        Self { r }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).map_err(Error::Io)?;
        Ok(b)
    }
    fn len(&mut self) -> Result<usize> {
        let n = u64::from_le_bytes(self.take()?);
        usize::try_from(n).map_err(|_| Error::LengthOverflow(n))
    }
    /// Read exactly `n` bytes without trusting `n` for the allocation: a corrupt length
    /// prefix on a short input fails with an I/O error instead of reserving `n` bytes.
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let got = (&mut self.r)
            .take(n as u64)
            .read_to_end(&mut out)
            .map_err(Error::Io)?;
        if got == n {
            Ok(out)
        } else {
            Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()))
        }
    }
}

macro_rules! fixed {
    ($($m:ident $v:ident $t:ty),* $(,)?) => {$(
        fn $m<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
            visitor.$v(<$t>::from_le_bytes(self.take()?))
        }
    )*};
}

impl<'de, R: Read> de::Deserializer<'de> for &mut Deserializer<R> {
    type Error = Error;

    fixed!(deserialize_i8 visit_i8 i8, deserialize_i16 visit_i16 i16, deserialize_i32 visit_i32 i32,
           deserialize_i64 visit_i64 i64, deserialize_i128 visit_i128 i128,
           deserialize_u8 visit_u8 u8, deserialize_u16 visit_u16 u16, deserialize_u32 visit_u32 u32,
           deserialize_u64 visit_u64 u64, deserialize_u128 visit_u128 u128,
           deserialize_f32 visit_f32 f32, deserialize_f64 visit_f64 f64);

    fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value> {
        Err(Error::NotSelfDescribing)
    }
    fn deserialize_ignored_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value> {
        Err(Error::NotSelfDescribing)
    }
    fn deserialize_identifier<V: Visitor<'de>>(self, _: V) -> Result<V::Value> {
        Err(Error::NotSelfDescribing)
    }
    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.take::<1>()?[0] {
            0 => visitor.visit_bool(false),
            1 => visitor.visit_bool(true),
            b => Err(Error::InvalidBool(b)),
        }
    }
    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let first = self.take::<1>()?[0];
        let width = match first {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return Err(Error::InvalidChar),
        };
        let mut buf = [first, 0, 0, 0];
        if width > 1 {
            self.r.read_exact(&mut buf[1..width]).map_err(Error::Io)?;
        }
        let s = std::str::from_utf8(&buf[..width]).map_err(|_| Error::InvalidChar)?;
        s.chars()
            .next()
            .map_or(Err(Error::InvalidChar), |c| visitor.visit_char(c))
    }
    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_string(visitor)
    }
    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let n = self.len()?;
        let s = String::from_utf8(self.bytes(n)?).map_err(|_| Error::InvalidUtf8)?;
        visitor.visit_string(s)
    }
    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_byte_buf(visitor)
    }
    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let n = self.len()?;
        visitor.visit_byte_buf(self.bytes(n)?)
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.take::<1>()?[0] {
            0 => visitor.visit_none(),
            1 => visitor.visit_some(self),
            b => Err(Error::InvalidOptionTag(b)),
        }
    }
    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_unit()
    }
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_newtype_struct(self)
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let n = self.len()?;
        visitor.visit_seq(Counted { de: self, left: n })
    }
    fn deserialize_tuple<V: Visitor<'de>>(self, n: usize, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(Counted { de: self, left: n })
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        n: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_tuple(n, visitor)
    }
    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let n = self.len()?;
        visitor.visit_map(Counted { de: self, left: n })
    }
    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_tuple(fields.len(), visitor)
    }
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_enum(self)
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}

/// Upper bound on the `size_hint` handed to a `Visitor`. The length prefix is untrusted: a
/// visitor that does `Vec::with_capacity(hint)` would otherwise allocate from a corrupt `u64`.
/// The hint is advisory, so capping it never changes what is decoded (same as serde's own
/// `cautious` helper, which caps at 4096 elements).
pub(crate) const MAX_SIZE_HINT: usize = 4096;

struct Counted<'a, R: Read> {
    de: &'a mut Deserializer<R>,
    left: usize,
}

impl<'de, R: Read> de::SeqAccess<'de> for Counted<'_, R> {
    type Error = Error;
    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.left.min(MAX_SIZE_HINT))
    }
}

impl<'de, R: Read> de::MapAccess<'de> for Counted<'_, R> {
    type Error = Error;
    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }
    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        seed.deserialize(&mut *self.de)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.left.min(MAX_SIZE_HINT))
    }
}

impl<'de, R: Read> de::EnumAccess<'de> for &mut Deserializer<R> {
    type Error = Error;
    type Variant = Self;
    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self)> {
        let idx = u32::from_le_bytes(self.take()?);
        let v = seed.deserialize(IntoDeserializer::<Error>::into_deserializer(idx))?;
        Ok((v, self))
    }
}

impl<'de, R: Read> de::VariantAccess<'de> for &mut Deserializer<R> {
    type Error = Error;
    fn unit_variant(self) -> Result<()> {
        Ok(())
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value> {
        seed.deserialize(self)
    }
    fn tuple_variant<V: Visitor<'de>>(self, n: usize, visitor: V) -> Result<V::Value> {
        de::Deserializer::deserialize_tuple(self, n, visitor)
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        de::Deserializer::deserialize_tuple(self, fields.len(), visitor)
    }
}
