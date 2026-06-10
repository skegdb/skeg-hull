//! KvCacheV1 writer. Builds header + section table + payloads + trailer CRC32C
//! in one shot.

use std::io::{Seek, SeekFrom, Write};

use crate::Result;
use crate::checksum::crc32c;
use crate::format::{FILE_CHECKSUM_LEN, HEADER_LEN, KV_CACHE_V1, SECTION_ENTRY_LEN};
use crate::header::{Header, SectionEntry};
use crate::kv_cache::{KvCache, blob, identity, section_id};

/// Serialise `cache` into `writer` at position 0.
pub fn write_kv_cache<W: Write + Seek>(cache: &KvCache, writer: &mut W) -> Result<()> {
    // 1. Build payloads.
    let identity_bytes = identity::encode(
        &cache.model_fp,
        cache.adapter_id.as_ref(),
        &cache.prefix_hash,
        &cache.meta,
    )?;
    let shape_bytes = cache.shape.encode();
    let blob_bytes = blob::encode(&cache.kv_blob, cache.next_logits.as_deref());

    // 2. Compute offsets.
    let section_count = 3u32;
    let table_len = u64::from(section_count) * (SECTION_ENTRY_LEN as u64);
    let identity_offset = HEADER_LEN as u64 + table_len;
    let shape_offset = identity_offset + identity_bytes.len() as u64;
    let blob_offset = shape_offset + shape_bytes.len() as u64;
    let body_len = table_len
        + identity_bytes.len() as u64
        + shape_bytes.len() as u64
        + blob_bytes.len() as u64;

    // 3. Build header.
    let header = Header {
        format: KV_CACHE_V1.format,
        version: KV_CACHE_V1.version,
        flags: 0,
        created_at: cache.meta.created_at,
        body_len,
        sections: vec![
            SectionEntry {
                type_id: section_id::IDENTITY,
                flags: 0,
                offset: identity_offset,
                length: identity_bytes.len() as u64,
                checksum: crc32c(&identity_bytes),
            },
            SectionEntry {
                type_id: section_id::SHAPE,
                flags: 0,
                offset: shape_offset,
                length: shape_bytes.len() as u64,
                checksum: crc32c(&shape_bytes),
            },
            SectionEntry {
                type_id: section_id::BLOB,
                flags: 0,
                offset: blob_offset,
                length: blob_bytes.len() as u64,
                checksum: crc32c(&blob_bytes),
            },
        ],
    };

    // 4. Assemble the full file image so we can compute a trailing CRC over
    //    header+sections without re-reading.
    let total_len = HEADER_LEN as u64 + body_len;
    let mut image = Vec::with_capacity(total_len as usize + FILE_CHECKSUM_LEN);
    {
        let mut cursor = std::io::Cursor::new(&mut image);
        header.write(&mut cursor)?;
    }
    debug_assert_eq!(image.len() as u64, identity_offset);
    image.extend_from_slice(&identity_bytes);
    image.extend_from_slice(&shape_bytes);
    image.extend_from_slice(&blob_bytes);
    debug_assert_eq!(image.len() as u64, total_len);

    let trailer = crc32c(&image);
    image.extend_from_slice(&trailer.to_le_bytes());

    // 5. Flush.
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(&image)?;
    writer.flush()?;
    Ok(())
}
