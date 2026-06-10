//! KvCacheV1 reader. Validates magic, format, version, file-level CRC,
//! and per-section CRCs before returning a populated [`KvCache`].

use std::io::{Read, Seek, SeekFrom};

use crate::Result;
use crate::checksum::crc32c;
use crate::compat::{Compatibility, check_compatibility};
use crate::error::Error;
use crate::format::{FILE_CHECKSUM_LEN, FormatVersion, KV_CACHE_V1};
use crate::header::Header;
use crate::kv_cache::{KvCache, KvCacheShape, blob, identity, section_id};

/// Read and validate a KvCacheV1 file from `reader`.
///
/// Verifies in order: trailing CRC32C, header magic+format+version, section
/// table layout, per-section CRC32C. Returns a populated [`KvCache`] or an
/// error describing what failed.
pub fn read_kv_cache<R: Read + Seek>(reader: &mut R) -> Result<KvCache> {
    reader.seek(SeekFrom::Start(0))?;
    let mut all = Vec::new();
    reader.read_to_end(&mut all)?;

    if all.len() < FILE_CHECKSUM_LEN + 64 {
        return Err(Error::Truncated);
    }

    // 1. Verify trailing CRC.
    let body_end = all.len() - FILE_CHECKSUM_LEN;
    let stored_trailer = u32::from_le_bytes(all[body_end..].try_into().unwrap());
    let computed_trailer = crc32c(&all[..body_end]);
    if stored_trailer != computed_trailer {
        return Err(Error::FileChecksumFailed);
    }

    // 2. Parse header + section table.
    let mut header_cursor = std::io::Cursor::new(&all[..body_end]);
    let header = Header::read(&mut header_cursor)?;

    let found = FormatVersion {
        format: header.format,
        version: header.version,
    };
    if found.format != KV_CACHE_V1.format {
        return Err(Error::FormatMismatch {
            found,
            expected: KV_CACHE_V1,
        });
    }
    match check_compatibility(found) {
        Compatibility::Compatible | Compatibility::CompatibleWithWarnings(_) => {}
        Compatibility::Incompatible { found, .. } => {
            return Err(Error::UnsupportedVersion {
                format: found.format,
                version: found.version,
            });
        }
    }

    // 3. Locate required sections by type id.
    let mut id_entry = None;
    let mut shape_entry = None;
    let mut blob_entry = None;
    for sec in &header.sections {
        match sec.type_id {
            section_id::IDENTITY => id_entry = Some(*sec),
            section_id::SHAPE => shape_entry = Some(*sec),
            section_id::BLOB => blob_entry = Some(*sec),
            _ => {} // unknown section id ignored (forward-compat)
        }
    }
    let id_entry = id_entry.ok_or(Error::MissingSection {
        type_id: section_id::IDENTITY,
    })?;
    let shape_entry = shape_entry.ok_or(Error::MissingSection {
        type_id: section_id::SHAPE,
    })?;
    let blob_entry = blob_entry.ok_or(Error::MissingSection {
        type_id: section_id::BLOB,
    })?;

    // 4. Read + checksum every section payload.
    let id_bytes = read_section(&all, id_entry.offset, id_entry.length)?;
    if crc32c(id_bytes) != id_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: id_entry.type_id,
        });
    }
    let (model_fp, adapter_id, prefix_hash, meta) = identity::decode(id_bytes)?;

    let shape_bytes = read_section(&all, shape_entry.offset, shape_entry.length)?;
    if crc32c(shape_bytes) != shape_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: shape_entry.type_id,
        });
    }
    let shape = KvCacheShape::decode(shape_bytes)?;

    let blob_bytes = read_section(&all, blob_entry.offset, blob_entry.length)?;
    if crc32c(blob_bytes) != blob_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: blob_entry.type_id,
        });
    }
    let (kv_blob, next_logits) = blob::decode(blob_bytes)?;

    // Size-validation invariant: kv_blob.len() must exactly match
    // shape.expected_blob_bytes(). A mismatch means either the file was
    // forged, the writer was buggy, or partial corruption slipped past
    // CRC32C (extremely unlikely but defense in depth). LLM inference fed
    // a wrong-sized KV produces silent garbage — fail loud here.
    let expected = shape.expected_blob_bytes();
    let got = kv_blob.len() as u64;
    if got != expected {
        return Err(Error::Malformed(format!(
            "kv_blob size mismatch: shape declares {expected} bytes, blob has {got}"
        )));
    }

    Ok(KvCache {
        model_fp,
        adapter_id,
        prefix_hash,
        shape,
        meta,
        kv_blob,
        next_logits,
    })
}

fn read_section(all: &[u8], offset: u64, length: u64) -> Result<&[u8]> {
    let start = offset as usize;
    let end = start
        .checked_add(length as usize)
        .ok_or_else(|| Error::Malformed("section length overflow".into()))?;
    if end > all.len() - FILE_CHECKSUM_LEN {
        return Err(Error::Truncated);
    }
    Ok(&all[start..end])
}
