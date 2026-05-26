//! Saga file writer.
//!
//! Builds the on-disk image in memory, then writes header + section
//! table + payloads + trailing file-level CRC32C in one shot. v0.1 is
//! synchronous and assumes sagas fit in memory (typically <1 MB).

use std::io::{Seek, SeekFrom, Write};

use crate::Result;
use crate::checksum::crc32c;
use crate::format::{FILE_CHECKSUM_LEN, HEADER_LEN, SAGA_V1, SECTION_ENTRY_LEN};
use crate::header::{Header, SectionEntry};
use crate::saga::{Saga, centroid, meta::SagaMeta, section_id, tags};

/// Serialise `saga` into `writer` at position 0.
pub fn write_saga<W: Write + Seek>(saga: &Saga, writer: &mut W) -> Result<()> {
    // 1. Build payloads.
    let meta = SagaMeta {
        tenant_id: saga.tenant_id,
        built_at: saga.built_at,
        record_count: saga.record_count,
        embedding_dim: saga.embedding_dim,
        centroid_count: saga.centroids.len() as u32,
    };
    let meta_bytes = meta.encode();
    let centroid_bytes = centroid::encode(&saga.centroids, saga.embedding_dim)?;
    let tag_bytes = tags::encode(&saga.tags)?;

    // 2. Compute offsets.
    let section_count = 3u32;
    let table_len = (section_count as u64) * (SECTION_ENTRY_LEN as u64);
    let meta_offset = HEADER_LEN as u64 + table_len;
    let centroids_offset = meta_offset + meta_bytes.len() as u64;
    let tags_offset = centroids_offset + centroid_bytes.len() as u64;
    let body_len =
        table_len + meta_bytes.len() as u64 + centroid_bytes.len() as u64 + tag_bytes.len() as u64;

    // 3. Build header.
    let header = Header {
        format: SAGA_V1.format,
        version: SAGA_V1.version,
        flags: 0,
        created_at: saga.built_at,
        body_len,
        sections: vec![
            SectionEntry {
                type_id: section_id::META,
                flags: 0,
                offset: meta_offset,
                length: meta_bytes.len() as u64,
                checksum: crc32c(&meta_bytes),
            },
            SectionEntry {
                type_id: section_id::CENTROIDS,
                flags: 0,
                offset: centroids_offset,
                length: centroid_bytes.len() as u64,
                checksum: crc32c(&centroid_bytes),
            },
            SectionEntry {
                type_id: section_id::TAG_AGGREGATE,
                flags: 0,
                offset: tags_offset,
                length: tag_bytes.len() as u64,
                checksum: crc32c(&tag_bytes),
            },
        ],
    };

    // 4. Assemble the full file image so we can compute a trailing CRC
    //    over header+sections without re-reading.
    let total_len = HEADER_LEN as u64 + body_len;
    let mut image = Vec::with_capacity(total_len as usize + FILE_CHECKSUM_LEN);

    {
        let mut cursor = std::io::Cursor::new(&mut image);
        header.write(&mut cursor)?;
    }
    debug_assert_eq!(image.len() as u64, meta_offset);
    image.extend_from_slice(&meta_bytes);
    image.extend_from_slice(&centroid_bytes);
    image.extend_from_slice(&tag_bytes);
    debug_assert_eq!(image.len() as u64, total_len);

    let trailer = crc32c(&image);
    image.extend_from_slice(&trailer.to_le_bytes());

    // 5. Flush to the writer in one call.
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(&image)?;
    writer.flush()?;
    Ok(())
}
