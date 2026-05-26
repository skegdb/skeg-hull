//! Saga file reader.
//!
//! Validates magic, format, version, file-level CRC, and per-section
//! CRCs before returning a populated [`Saga`]. v0.1 reads the whole
//! file into memory; saga files are small by design.

use std::io::{Read, Seek, SeekFrom};

use crate::Result;
use crate::checksum::crc32c;
use crate::compat::{Compatibility, check_compatibility};
use crate::error::Error;
use crate::format::{FILE_CHECKSUM_LEN, FormatVersion, SAGA_V1};
use crate::header::Header;
use crate::saga::{Saga, centroid, meta::SagaMeta, section_id, tags};

/// Read and validate a SagaV1 file from `reader`.
pub fn read_saga<R: Read + Seek>(reader: &mut R) -> Result<Saga> {
    // Slurp the file: saga files are KB- to low-MB-scale.
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
    if found.format != SAGA_V1.format {
        return Err(Error::FormatMismatch {
            found,
            expected: SAGA_V1,
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
    let mut meta_entry = None;
    let mut centroid_entry = None;
    let mut tag_entry = None;
    for sec in &header.sections {
        match sec.type_id {
            section_id::META => meta_entry = Some(*sec),
            section_id::CENTROIDS => centroid_entry = Some(*sec),
            section_id::TAG_AGGREGATE => tag_entry = Some(*sec),
            _ => {} // unknown section id ignored (forward-compat)
        }
    }
    let meta_entry = meta_entry.ok_or(Error::MissingSection {
        type_id: section_id::META,
    })?;
    let centroid_entry = centroid_entry.ok_or(Error::MissingSection {
        type_id: section_id::CENTROIDS,
    })?;
    let tag_entry = tag_entry.ok_or(Error::MissingSection {
        type_id: section_id::TAG_AGGREGATE,
    })?;

    // 4. Read + checksum every section payload.
    let meta_bytes = read_section(&all, meta_entry.offset, meta_entry.length)?;
    if crc32c(meta_bytes) != meta_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: meta_entry.type_id,
        });
    }
    let meta = SagaMeta::decode(meta_bytes)?;

    let centroid_bytes = read_section(&all, centroid_entry.offset, centroid_entry.length)?;
    if crc32c(centroid_bytes) != centroid_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: centroid_entry.type_id,
        });
    }
    let centroids = centroid::decode(
        centroid_bytes,
        meta.centroid_count,
        meta.embedding_dim,
    )?;

    let tag_bytes = read_section(&all, tag_entry.offset, tag_entry.length)?;
    if crc32c(tag_bytes) != tag_entry.checksum {
        return Err(Error::SectionChecksumFailed {
            type_id: tag_entry.type_id,
        });
    }
    let tags = tags::decode(tag_bytes)?;

    Ok(Saga {
        tenant_id: meta.tenant_id,
        built_at: meta.built_at,
        record_count: meta.record_count,
        embedding_dim: meta.embedding_dim,
        centroids,
        tags,
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
