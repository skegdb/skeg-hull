//! Integration test: build a saga, write it to disk via the atomic
//! writer, read it back, verify byte-for-byte equality.

use std::io::Cursor;

use skeg_hull::saga::{Centroid, Saga, TagEntry};

fn sample() -> Saga {
    Saga {
        tenant_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        built_at: 1_700_000_000,
        record_count: 1234,
        embedding_dim: 8,
        centroids: vec![
            Centroid {
                cluster_size: 100,
                vector: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
            Centroid {
                cluster_size: 50,
                vector: vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            },
            Centroid {
                cluster_size: 25,
                vector: vec![2.71_f32; 8],
            },
        ],
        tags: vec![
            TagEntry {
                count: 200,
                tag: "code".into(),
            },
            TagEntry {
                count: 150,
                tag: "design".into(),
            },
            TagEntry {
                count: 7,
                tag: "übung".into(),
            },
        ],
    }
}

#[test]
fn cursor_roundtrip_is_exact() {
    let saga = sample();
    let mut buf = Cursor::new(Vec::<u8>::new());
    saga.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = Saga::read_from(&mut buf).unwrap();
    assert_eq!(decoded, saga);
}

#[test]
fn file_roundtrip_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent-a.saga");
    let saga = sample();
    saga.write_to_path(&path).unwrap();

    let decoded = Saga::read_from_path(&path).unwrap();
    assert_eq!(decoded, saga);

    // Atomic rewrite leaves no temp files behind.
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .collect();
    assert_eq!(entries.len(), 1, "unexpected entries: {entries:?}");
}

#[test]
fn detects_trailing_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent-b.saga");
    sample().write_to_path(&path).unwrap();

    // Flip the last byte (the trailing CRC).
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    let err = Saga::read_from_path(&path).unwrap_err();
    assert!(
        matches!(err, skeg_hull::Error::FileChecksumFailed),
        "got {err:?}"
    );
}

#[test]
fn detects_section_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent-c.saga");
    sample().write_to_path(&path).unwrap();

    // Flip a centroid byte. We need to keep the trailing CRC valid
    // otherwise we'd hit FileChecksumFailed first - so recompute it.
    let mut bytes = std::fs::read(&path).unwrap();
    // Centroid section starts at offset 64 + 3*28 + 64 = 212.
    bytes[300] ^= 0x01;
    let new_trailer = crc32c::crc32c(&bytes[..bytes.len() - 4]);
    let last = bytes.len() - 4;
    bytes[last..].copy_from_slice(&new_trailer.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let err = Saga::read_from_path(&path).unwrap_err();
    assert!(
        matches!(err, skeg_hull::Error::SectionChecksumFailed { .. }),
        "got {err:?}"
    );
}

#[test]
fn empty_saga_roundtrip() {
    let saga = Saga::empty([0; 16], 16);
    let mut buf = Cursor::new(Vec::<u8>::new());
    saga.write_to(&mut buf).unwrap();
    buf.set_position(0);
    let decoded = Saga::read_from(&mut buf).unwrap();
    assert_eq!(decoded, saga);
}
