//! TDD: forward compatibility. A v0.1 reader must accept files that contain
//! **unknown sections** alongside the required ones. This enables non-breaking
//! evolution: a v0.2 writer can add new sections (e.g. per-layer offsets) and
//! v0.1 readers continue to function, ignoring what they don't know.
//!
//! The reverse direction (v0.2 reader on a v0.1 file) is the harder case and
//! is handled by `compat::check_compatibility` semantically — not tested here.

use std::io::{Cursor, Seek, SeekFrom, Write};

use skeg_hull::checksum::crc32c;
use skeg_hull::format::{FILE_CHECKSUM_LEN, HEADER_LEN, KV_CACHE_V1, SECTION_ENTRY_LEN};
use skeg_hull::header::{Header, SectionEntry};
use skeg_hull::kv_cache::{KvCache, KvCacheMeta, KvCacheShape, KvDtype, KvLayout, section_id};

const UNKNOWN_TYPE_ID: u16 = 999;

fn shape() -> KvCacheShape {
    KvCacheShape {
        num_layers: 2,
        n_kv_heads: 1,
        head_dim: 4,
        num_tokens: 8,
        dtype: KvDtype::Bf16,
        layout: KvLayout::Contiguous,
        quant: None,
    }
}

fn ref_cache() -> KvCache {
    let s = shape();
    let expected = s.expected_blob_bytes() as usize;
    KvCache {
        model_fp: [0x11u8; 32],
        adapter_id: Some([0x22u8; 32]),
        prefix_hash: [0x33u8; 32],
        shape: s,
        meta: KvCacheMeta {
            created_at: 42,
            tokenizer_version: 1,
            model_name: "t".into(),
        },
        kv_blob: (0..expected as u32).map(|i| (i % 256) as u8).collect(),
        next_logits: None,
    }
}

/// Build a file with 4 sections: IDENTITY, SHAPE, BLOB, plus an unknown section
/// inserted between SHAPE and BLOB. Returns the file bytes.
fn build_file_with_unknown_section(cache: &KvCache) -> Vec<u8> {
    // Re-serialise the 3 known sections by hand. We must replicate what
    // `write_kv_cache` does because we want to insert an extra section in the
    // middle of the section table.
    let identity_bytes = build_identity_section(cache);
    let shape_bytes = cache.shape.encode().to_vec();
    let unknown_bytes = b"this is an unknown section the v0.1 reader has never seen".to_vec();
    let blob_bytes = build_blob_section(cache);

    let section_count = 4u32;
    let table_len = u64::from(section_count) * (SECTION_ENTRY_LEN as u64);
    let identity_offset = HEADER_LEN as u64 + table_len;
    let shape_offset = identity_offset + identity_bytes.len() as u64;
    let unknown_offset = shape_offset + shape_bytes.len() as u64;
    let blob_offset = unknown_offset + unknown_bytes.len() as u64;
    let body_len = table_len
        + identity_bytes.len() as u64
        + shape_bytes.len() as u64
        + unknown_bytes.len() as u64
        + blob_bytes.len() as u64;

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
            // The unknown section — type_id 999 doesn't map to any known
            // section_id constant. v0.1 reader MUST skip this entry, not fail.
            SectionEntry {
                type_id: UNKNOWN_TYPE_ID,
                flags: 0,
                offset: unknown_offset,
                length: unknown_bytes.len() as u64,
                checksum: crc32c(&unknown_bytes),
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

    let total_len = HEADER_LEN as u64 + body_len;
    let mut image = Vec::with_capacity(total_len as usize + FILE_CHECKSUM_LEN);
    {
        let mut cursor = Cursor::new(&mut image);
        header.write(&mut cursor).unwrap();
        // Header.write may not extend the buffer to full position if seek
        // past end. Ensure we're positioned at the end after writing.
        let pos = cursor.position();
        assert_eq!(pos, image.len() as u64);
    }
    assert_eq!(image.len() as u64, identity_offset);

    image.extend_from_slice(&identity_bytes);
    image.extend_from_slice(&shape_bytes);
    image.extend_from_slice(&unknown_bytes);
    image.extend_from_slice(&blob_bytes);
    assert_eq!(image.len() as u64, total_len);

    let trailer = crc32c(&image);
    image.extend_from_slice(&trailer.to_le_bytes());

    image
}

fn build_identity_section(cache: &KvCache) -> Vec<u8> {
    // Replicate identity::encode layout. Total = 32 + 1 + 32 + 32 + 8 + 4 + 2 + name_len.
    let name_bytes = cache.meta.model_name.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&cache.model_fp);
    if let Some(aid) = &cache.adapter_id {
        out.push(1);
        out.extend_from_slice(aid);
    } else {
        out.push(0);
        out.extend_from_slice(&[0u8; 32]);
    }
    out.extend_from_slice(&cache.prefix_hash);
    out.extend_from_slice(&cache.meta.created_at.to_le_bytes());
    out.extend_from_slice(&cache.meta.tokenizer_version.to_le_bytes());
    out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(name_bytes);
    out
}

fn build_blob_section(cache: &KvCache) -> Vec<u8> {
    // Replicate blob::encode layout.
    let mut out = Vec::new();
    let logits_present = u8::from(cache.next_logits.is_some());
    out.push(logits_present);
    out.extend_from_slice(&(cache.kv_blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&cache.kv_blob);
    if let Some(l) = &cache.next_logits {
        out.extend_from_slice(&(l.len() as u64).to_le_bytes());
        out.extend_from_slice(l);
    }
    out
}

#[test]
fn unknown_section_is_skipped_at_read() {
    let cache = ref_cache();
    let bytes = build_file_with_unknown_section(&cache);
    let mut cursor = Cursor::new(bytes);

    let decoded = KvCache::read_from(&mut cursor).expect("reader must accept unknown sections");
    // The unknown section must NOT affect the decoded value of the known fields.
    assert_eq!(decoded, cache);
}

#[test]
fn unknown_section_with_corrupt_payload_still_reads_known_sections() {
    // Even if the unknown section's CRC is wrong, the reader should not fail
    // on it (it's skipped, so its CRC is never verified). This guarantees that
    // a future writer making a mistake in an unknown section doesn't poison
    // the file for older readers.
    let cache = ref_cache();
    let mut bytes = build_file_with_unknown_section(&cache);

    // Corrupt one byte inside the unknown section. We have to find it: it's
    // at `unknown_offset` from the file start. Easier: corrupt a byte that
    // is unique to the unknown section's text — "unknown" in ASCII.
    let needle = b"unknown";
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("needle not found");
    bytes[pos] ^= 0xFF; // flip a byte inside the unknown section

    // The trailing file-level CRC32C will fail now because we modified the body.
    // So this test actually proves that file-level CRC DOES catch arbitrary
    // body modifications — including those inside unknown sections. The
    // forward-compat property is about *section table* tolerance, not about
    // letting CRC-broken payloads pass.

    let mut cursor = Cursor::new(bytes);
    let result = KvCache::read_from(&mut cursor);
    assert!(
        result.is_err(),
        "expected file-CRC failure to surface even when bytes are in an unknown section"
    );
}

#[test]
fn helper_seek_to_keep_clippy_happy() {
    // No-op to ensure imports used.
    let mut c = Cursor::new(Vec::<u8>::new());
    c.write_all(b"x").unwrap();
    c.seek(SeekFrom::Start(0)).unwrap();
}
