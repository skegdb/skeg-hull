//! Common header and section table.
//!
//! Every hull file starts with a 64-byte header followed by a section
//! table. The header alone is enough to identify the format and its
//! version; the section table maps logical sections to byte ranges in
//! the file. Tools that don't understand the body can still read both.
//!
//! All integer fields are little-endian.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::error::Error;
use crate::format::{
    FILE_CHECKSUM_LEN, FormatId, HEADER_LEN, MAGIC, SECTION_ENTRY_LEN,
};

/// A section table entry: where the payload of a logical section lives
/// in the file, and how to verify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionEntry {
    /// Section type id. Namespaced per format.
    pub type_id: u16,
    /// Reserved flag bits. Zero in v0.1.
    pub flags: u16,
    /// Offset of the payload, relative to file start.
    pub offset: u64,
    /// Payload length in bytes.
    pub length: u64,
    /// CRC32C of the payload.
    pub checksum: u32,
}

impl SectionEntry {
    fn read_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < SECTION_ENTRY_LEN {
            return Err(Error::Truncated);
        }
        let type_id = u16::from_le_bytes(buf[0..2].try_into().unwrap());
        let flags = u16::from_le_bytes(buf[2..4].try_into().unwrap());
        let offset = u64::from_le_bytes(buf[4..12].try_into().unwrap());
        let length = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let checksum = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        Ok(Self {
            type_id,
            flags,
            offset,
            length,
            checksum,
        })
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..2].copy_from_slice(&self.type_id.to_le_bytes());
        out[2..4].copy_from_slice(&self.flags.to_le_bytes());
        out[4..12].copy_from_slice(&self.offset.to_le_bytes());
        out[12..20].copy_from_slice(&self.length.to_le_bytes());
        out[20..24].copy_from_slice(&self.checksum.to_le_bytes());
        out[24..28].fill(0);
    }
}

/// File header plus its section table.
#[derive(Debug, Clone)]
pub struct Header {
    /// Top-level format family.
    pub format: FormatId,
    /// Format version inside that family.
    pub version: u16,
    /// Reserved flag bits. Zero in v0.1.
    pub flags: u32,
    /// Unix seconds when the file was written.
    pub created_at: i64,
    /// Body length in bytes: section table + payloads (excludes header
    /// and trailing file-level checksum).
    pub body_len: u64,
    /// Section table.
    pub sections: Vec<SectionEntry>,
}

impl Header {
    /// Section table offset for v0.1: immediately after the header.
    pub const SECTION_TABLE_OFFSET: u32 = HEADER_LEN as u32;

    /// Total on-disk size of the section table for this header.
    pub fn section_table_len(&self) -> u64 {
        (self.sections.len() as u64) * (SECTION_ENTRY_LEN as u64)
    }

    /// Offset of the first byte after header+section table.
    pub fn first_payload_offset(&self) -> u64 {
        HEADER_LEN as u64 + self.section_table_len()
    }

    /// Read a header and its section table from `reader`. Leaves the
    /// stream positioned immediately after the section table (i.e. at
    /// the first payload byte).
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        reader.seek(SeekFrom::Start(0))?;

        let mut buf = [0u8; HEADER_LEN];
        reader.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::Truncated
            } else {
                Error::Io(e)
            }
        })?;

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if magic != MAGIC {
            return Err(Error::MagicMismatch { found: magic });
        }

        let format_id_raw = u16::from_le_bytes(buf[8..10].try_into().unwrap());
        let format = FormatId::from_u16(format_id_raw).ok_or(Error::UnknownFormat {
            format_id: format_id_raw,
        })?;
        let version = u16::from_le_bytes(buf[10..12].try_into().unwrap());
        let flags = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let created_at = i64::from_le_bytes(buf[16..24].try_into().unwrap());
        let body_len = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let section_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let section_table_offset =
            u32::from_le_bytes(buf[36..40].try_into().unwrap());
        // bytes 40..64 reserved, ignored.

        if section_table_offset != Self::SECTION_TABLE_OFFSET {
            return Err(Error::Malformed(format!(
                "unexpected section table offset {section_table_offset}; v0.1 requires {}",
                Self::SECTION_TABLE_OFFSET
            )));
        }

        reader.seek(SeekFrom::Start(section_table_offset as u64))?;
        let mut table_buf = vec![0u8; (section_count as usize) * SECTION_ENTRY_LEN];
        reader.read_exact(&mut table_buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::Truncated
            } else {
                Error::Io(e)
            }
        })?;

        let mut sections = Vec::with_capacity(section_count as usize);
        for i in 0..(section_count as usize) {
            let off = i * SECTION_ENTRY_LEN;
            sections.push(SectionEntry::read_from(&table_buf[off..off + SECTION_ENTRY_LEN])?);
        }

        Ok(Self {
            format,
            version,
            flags,
            created_at,
            body_len,
            sections,
        })
    }

    /// Write the header and section table starting at position 0.
    /// `body_len` and `sections` are taken from `self`; callers are
    /// responsible for populating them with the correct payload offsets
    /// and lengths *before* calling write.
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        writer.seek(SeekFrom::Start(0))?;

        let mut buf = [0u8; HEADER_LEN];
        buf[0..8].copy_from_slice(&MAGIC);
        buf[8..10].copy_from_slice(&self.format.as_u16().to_le_bytes());
        buf[10..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.created_at.to_le_bytes());
        buf[24..32].copy_from_slice(&self.body_len.to_le_bytes());
        buf[32..36].copy_from_slice(&(self.sections.len() as u32).to_le_bytes());
        buf[36..40].copy_from_slice(&Self::SECTION_TABLE_OFFSET.to_le_bytes());
        // bytes 40..64 reserved, zero.
        writer.write_all(&buf)?;

        let mut table = vec![0u8; self.sections.len() * SECTION_ENTRY_LEN];
        for (i, sec) in self.sections.iter().enumerate() {
            let off = i * SECTION_ENTRY_LEN;
            sec.write_to(&mut table[off..off + SECTION_ENTRY_LEN]);
        }
        writer.write_all(&table)?;

        Ok(())
    }
}

/// Total length of the trailing file-level checksum block.
pub const TRAILER_LEN: u64 = FILE_CHECKSUM_LEN as u64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_header() -> Header {
        Header {
            format: FormatId::Saga,
            version: 1,
            flags: 0,
            created_at: 1_700_000_000,
            body_len: 56 + 100,
            sections: vec![
                SectionEntry {
                    type_id: 1,
                    flags: 0,
                    offset: HEADER_LEN as u64 + 56,
                    length: 50,
                    checksum: 0xdeadbeef,
                },
                SectionEntry {
                    type_id: 2,
                    flags: 0,
                    offset: HEADER_LEN as u64 + 56 + 50,
                    length: 50,
                    checksum: 0xcafebabe,
                },
            ],
        }
    }

    #[test]
    fn header_roundtrip() {
        let h = make_header();
        let mut buf = Cursor::new(Vec::new());
        h.write(&mut buf).unwrap();
        // Pad up to expected body so read can position past section table.
        buf.get_mut().resize(HEADER_LEN + 56 + 100, 0);
        buf.set_position(0);
        let read = Header::read(&mut buf).unwrap();
        assert_eq!(read.format, h.format);
        assert_eq!(read.version, h.version);
        assert_eq!(read.flags, h.flags);
        assert_eq!(read.created_at, h.created_at);
        assert_eq!(read.body_len, h.body_len);
        assert_eq!(read.sections, h.sections);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = vec![0u8; HEADER_LEN];
        buf[0..8].copy_from_slice(b"NOTHULL!");
        let mut c = Cursor::new(buf);
        let err = Header::read(&mut c).unwrap_err();
        match err {
            Error::MagicMismatch { found } => assert_eq!(&found, b"NOTHULL!"),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_format() {
        let mut buf = vec![0u8; HEADER_LEN];
        buf[0..8].copy_from_slice(&MAGIC);
        buf[8..10].copy_from_slice(&777u16.to_le_bytes());
        buf[10..12].copy_from_slice(&1u16.to_le_bytes());
        buf[36..40].copy_from_slice(&Header::SECTION_TABLE_OFFSET.to_le_bytes());
        let mut c = Cursor::new(buf);
        let err = Header::read(&mut c).unwrap_err();
        assert!(matches!(err, Error::UnknownFormat { format_id: 777 }));
    }

    #[test]
    fn rejects_truncated() {
        let mut c = Cursor::new(vec![0u8; 10]);
        let err = Header::read(&mut c).unwrap_err();
        assert!(matches!(err, Error::Truncated));
    }
}
