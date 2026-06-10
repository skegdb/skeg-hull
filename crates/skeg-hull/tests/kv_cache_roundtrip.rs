//! Integration test: roundtrip a KvCacheV1 file. TDD seed for the
//! `kv_cache` module — initially RED, drives the implementation.

use std::io::Cursor;

use skeg_hull::kv_cache::{KvCache, KvCacheMeta, KvCacheShape, KvDtype, KvLayout};

fn sample() -> KvCache {
    // 2-layer dummy KV blob: 1 KV head, head_dim=4, num_tokens=8, bf16.
    // Per-layer K size = 1 * 8 * 4 * 2 = 64 bytes. V same. Layer = 128 bytes.
    // Total KV blob = 2 layers * 128 = 256 bytes.
    let kv_blob = (0u8..=255).collect::<Vec<u8>>();

    KvCache {
        model_fp: [0xAA; 32],
        adapter_id: Some([0xBB; 32]),
        prefix_hash: [0xCC; 32],
        shape: KvCacheShape {
            num_layers: 2,
            n_kv_heads: 1,
            head_dim: 4,
            num_tokens: 8,
            dtype: KvDtype::Bf16,
            layout: KvLayout::Contiguous,
            quant: None,
        },
        meta: KvCacheMeta {
            created_at: 1_716_800_000,
            tokenizer_version: 3,
            model_name: "llama-3.1-8b-instruct".into(),
        },
        kv_blob,
        next_logits: Some(vec![0u8, 1, 2, 3, 4, 5, 6, 7]),
    }
}

#[test]
fn cursor_roundtrip_is_exact() {
    let original = sample();
    let mut buf = Cursor::new(Vec::<u8>::new());
    original.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn no_adapter_roundtrips() {
    let mut sample = sample();
    sample.adapter_id = None;

    let mut buf = Cursor::new(Vec::<u8>::new());
    sample.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();
    assert_eq!(sample, decoded);
    assert_eq!(decoded.adapter_id, None);
}

#[test]
fn no_next_logits_roundtrips() {
    let mut sample = sample();
    sample.next_logits = None;

    let mut buf = Cursor::new(Vec::<u8>::new());
    sample.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();
    assert_eq!(sample, decoded);
}

#[test]
fn file_path_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cache.kvc");

    let original = sample();
    original.write_to_path(&path).unwrap();
    let decoded = KvCache::read_from_path(&path).unwrap();

    assert_eq!(original, decoded);
}

#[test]
fn kv_blob_is_byte_exact() {
    let original = sample();
    let mut buf = Cursor::new(Vec::<u8>::new());
    original.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();

    // Byte-exact equality on the KV blob is the load-bearing invariant —
    // if a single byte is reordered, LLM inference produces garbage.
    assert_eq!(original.kv_blob, decoded.kv_blob);
}

#[test]
fn corrupted_kv_blob_is_rejected() {
    let original = sample();
    let mut buf = Cursor::new(Vec::<u8>::new());
    original.write_to(&mut buf).unwrap();

    // Flip a byte in the KV blob payload region. The header sits at offset
    // 0..64, the section table at 64..120 (2 entries × 28 bytes). The KV
    // blob is in the body — flipping a byte in the middle of the file
    // should land inside one of the sections.
    let raw = buf.get_mut();
    let mid = raw.len() / 2;
    raw[mid] ^= 0xFF;

    buf.set_position(0);
    let result = KvCache::read_from(&mut buf);
    assert!(
        result.is_err(),
        "expected corruption to be detected by CRC32C check"
    );
}

#[test]
fn unknown_magic_is_rejected() {
    let original = sample();
    let mut buf = Cursor::new(Vec::<u8>::new());
    original.write_to(&mut buf).unwrap();

    // Smash the magic.
    let raw = buf.get_mut();
    raw[0] = b'X';

    buf.set_position(0);
    let result = KvCache::read_from(&mut buf);
    assert!(result.is_err(), "expected unknown magic to be rejected");
}
