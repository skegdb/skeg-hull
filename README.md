# skeg-hull

> *The on-disk format of skeg. Stable, versioned, readable by tools
> that don't need the engine.*

`skeg-hull` is the watertight shell of the skeg ecosystem: a small,
versioned binary format with a fixed 64-byte header, a section table,
per-section CRC32C checksums, and a trailing file-level CRC. Anyone can
read a hull file with `Read + Seek`; no engine dependency required.

## v0.1 scope

Three format identifiers are reserved (`KV`, `Vamana`, `Saga`) but only
**SagaV1** ships with reader and writer. KV and Vamana are placeholders
that will be implemented when (and if) skeg-core adopts hull as its
storage layer.

- Header + section table with per-section CRC32C and a trailing file-level CRC32C.
- `atomic_write(path, |f| { ... })` helper (write-to-temp + fsync + rename + fsync-dir).
- Forward-compat: a v0.2 reader will be able to open v0.1 files; unknown reserved bits are ignored on read.
- Compatibility matrix surfaced as `Compatibility::Compatible | CompatibleWithWarnings | Incompatible`.

## Saga round-trip

```rust
use skeg_hull::saga::{Centroid, Saga, TagEntry};

let saga = Saga {
    tenant_id: [0xab; 16],
    built_at: 1_700_000_000,
    record_count: 1_000,
    embedding_dim: 8,
    centroids: vec![Centroid { cluster_size: 50, vector: vec![0.1; 8] }],
    tags: vec![TagEntry { count: 25, tag: "code".into() }],
};

saga.write_to_path("agent-a.saga")?;
let back = Saga::read_from_path("agent-a.saga")?;
assert_eq!(back, saga);
```

## File layout

```text
┌──────────────────────────────────────────────────┐
│ Header (64 bytes, magic SKEGHULL, format+ver)    │
├──────────────────────────────────────────────────┤
│ Section table (N × 28 bytes)                     │
├──────────────────────────────────────────────────┤
│ Section payloads (per-section CRC32C)            │
├──────────────────────────────────────────────────┤
│ Trailing CRC32C (4 bytes)                        │
└──────────────────────────────────────────────────┘
```

## Building

```sh
cargo build --workspace
cargo test --workspace
```

## License

Apache-2.0.
