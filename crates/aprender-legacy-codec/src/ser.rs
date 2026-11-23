//! Serializer half: writes the bincode 1.x legacy layout.
use crate::{Error, Result};
use serde::ser::{self, Serialize};
use std::io::Write;

pub(crate) struct Serializer<W: Write> {
    w: W,
}

impl<W: Write> Serializer<W> {
    pub(crate) fn new(w: W) -> Self {
        Self { w }
    }
    fn put(&mut self, b: &[u8]) -> Result<()> {
        self.w.write_all(b).map_err(Error::Io)
    }
    fn len(&mut self, n: usize) -> Result<()> {
        self.put(&(n as u64).to_le_bytes())
    }
}

macro_rules! fixed {
    ($($m:ident $t:ty),* $(,)?) => {$(
        fn $m(self, v: $t) -> Result<()> { self.put(&v.to_le_bytes()) }
    )*};
}

impl<W: Write> ser::Serializer for &mut Serializer<W> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fixed!(serialize_i8 i8, serialize_i16 i16, serialize_i32 i32, serialize_i64 i64,
           serialize_i128 i128, serialize_u8 u8, serialize_u16 u16, serialize_u32 u32,
           serialize_u64 u64, serialize_u128 u128, serialize_f32 f32, serialize_f64 f64);

    fn serialize_bool(self, v: bool) -> Result<()> {
        self.put(&[u8::from(v)])
    }
    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        self.put(v.encode_utf8(&mut buf).as_bytes())
    }
    fn serialize_str(self, v: &str) -> Result<()> {
        self.len(v.len())?;
        self.put(v.as_bytes())
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        self.len(v.len())?;
        self.put(v)
    }
    fn serialize_none(self) -> Result<()> {
        self.put(&[0])
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<()> {
        self.put(&[1])?;
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<()> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<()> {
        Ok(())
    }
    fn serialize_unit_variant(self, _: &'static str, idx: u32, _: &'static str) -> Result<()> {
        self.put(&idx.to_le_bytes())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, _: &'static str, v: &T) -> Result<()> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        idx: u32,
        _: &'static str,
        v: &T,
    ) -> Result<()> {
        self.put(&idx.to_le_bytes())?;
        v.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self> {
        self.len(len.ok_or(Error::SequenceMustHaveLength)?)?;
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        idx: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self> {
        self.put(&idx.to_le_bytes())?;
        Ok(self)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self> {
        self.len(len.ok_or(Error::SequenceMustHaveLength)?)?;
        Ok(self)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        idx: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self> {
        self.put(&idx.to_le_bytes())?;
        Ok(self)
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}

macro_rules! elements {
    ($tr:ident, $f:ident) => {
        impl<W: Write> ser::$tr for &mut Serializer<W> {
            type Ok = ();
            type Error = Error;
            fn $f<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<()> {
                Ok(())
            }
        }
    };
}
elements!(SerializeSeq, serialize_element);
elements!(SerializeTuple, serialize_element);
elements!(SerializeTupleStruct, serialize_field);
elements!(SerializeTupleVariant, serialize_field);

impl<W: Write> ser::SerializeMap for &mut Serializer<W> {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<()> {
        k.serialize(&mut **self)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        v.serialize(&mut **self)
    }
    fn end(self) -> Result<()> {
        Ok(())
    }
}

macro_rules! named {
    ($tr:ident) => {
        impl<W: Write> ser::$tr for &mut Serializer<W> {
            type Ok = ();
            type Error = Error;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                _: &'static str,
                v: &T,
            ) -> Result<()> {
                v.serialize(&mut **self)
            }
            fn end(self) -> Result<()> {
                Ok(())
            }
        }
    };
}
named!(SerializeStruct);
named!(SerializeStructVariant);
