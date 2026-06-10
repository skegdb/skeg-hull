//! TDD seed: size validation between `shape.expected_blob_bytes()` and the
//! actual `kv_blob.len()`. A forged or corrupt file with mismatched sizes
//! must be rejected at decode time — otherwise it produces silent garbage
//! in LLM inference (load-bearing correctness invariant).

use std::io::Cursor;

use skeg_hull::kv_cache::{KvCache, KvCacheMeta, KvCacheShape, KvDtype, KvLayout};

fn shape_with(num_tokens: u32) -> KvCacheShape {
    KvCacheShape {
        num_layers: 2,
        n_kv_heads: 1,
        head_dim: 4,
        num_tokens,
        dtype: KvDtype::Bf16,
        layout: KvLayout::Contiguous,
        quant: None,
    }
}

fn meta() -> KvCacheMeta {
    KvCacheMeta {
        created_at: 0,
        tokenizer_version: 1,
        model_name: "t".into(),
    }
}

#[test]
fn read_rejects_blob_smaller_than_shape() {
    // Shape says 8 tokens × 2 layers × 1 head × 4 dim × 2 byte × 2 (K+V) = 256 bytes.
    // We put a 100-byte blob in. Writer must reject (write-side validation)
    // OR reader must reject (read-side validation). At least one — preferably both.
    let cache = KvCache {
        model_fp: [0u8; 32],
        adapter_id: None,
        prefix_hash: [0u8; 32],
        shape: shape_with(8),
        meta: meta(),
        kv_blob: vec![0u8; 100], // 100 != 256
        next_logits: None,
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    match cache.write_to(&mut buf) {
        Ok(_) => {
            // Writer permissive: reader MUST catch it.
            buf.set_position(0);
            let result = KvCache::read_from(&mut buf);
            assert!(
                result.is_err(),
                "reader must reject blob size mismatch when writer didn't"
            );
        }
        Err(_) => {} // Writer-side validation: also acceptable.
    }
}

#[test]
fn read_rejects_blob_larger_than_shape() {
    // Inverse: shape says 256 bytes, blob has 1024.
    let cache = KvCache {
        model_fp: [0u8; 32],
        adapter_id: None,
        prefix_hash: [0u8; 32],
        shape: shape_with(8),
        meta: meta(),
        kv_blob: vec![0u8; 1024],
        next_logits: None,
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    match cache.write_to(&mut buf) {
        Ok(_) => {
            buf.set_position(0);
            let result = KvCache::read_from(&mut buf);
            assert!(result.is_err(), "reader must reject oversized blob");
        }
        Err(_) => {}
    }
}

#[test]
fn read_accepts_blob_matching_shape() {
    // Sanity: the validation only fires on mismatch. A correctly-sized blob
    // must still round-trip.
    let shape = shape_with(8);
    let expected = shape.expected_blob_bytes() as usize;
    let cache = KvCache {
        model_fp: [0u8; 32],
        adapter_id: None,
        prefix_hash: [0u8; 32],
        shape,
        meta: meta(),
        kv_blob: vec![0xA7u8; expected],
        next_logits: None,
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    cache.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();
    assert_eq!(decoded, cache);
}

#[test]
fn zero_token_blob_is_accepted() {
    // Edge: num_tokens=0 → expected_blob_bytes=0 → kv_blob must be empty.
    // This is the state of a cache that hasn't been written to yet.
    let shape = shape_with(0);
    assert_eq!(shape.expected_blob_bytes(), 0);

    let cache = KvCache {
        model_fp: [0u8; 32],
        adapter_id: None,
        prefix_hash: [0u8; 32],
        shape,
        meta: meta(),
        kv_blob: Vec::new(),
        next_logits: None,
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    cache.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = KvCache::read_from(&mut buf).unwrap();
    assert_eq!(decoded, cache);
}

#[test]
fn zero_token_with_nonempty_blob_is_rejected() {
    // Edge: shape says 0 tokens but blob has bytes — corruption.
    let cache = KvCache {
        model_fp: [0u8; 32],
        adapter_id: None,
        prefix_hash: [0u8; 32],
        shape: shape_with(0),
        meta: meta(),
        kv_blob: vec![0u8; 10],
        next_logits: None,
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    match cache.write_to(&mut buf) {
        Ok(_) => {
            buf.set_position(0);
            assert!(
                KvCache::read_from(&mut buf).is_err(),
                "reader must reject nonzero blob when shape declares 0 tokens"
            );
        }
        Err(_) => {}
    }
}
