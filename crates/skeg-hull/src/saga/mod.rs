//! SagaV1 format: hansa's condensed memory digest.
//!
//! A saga is the cheap "is this peer worth querying?" summary. Each file
//! holds a small set of cluster centroids plus a tag aggregate. Tens to
//! hundreds of KB per saga; meant to be read eagerly.
//!
//! File layout (all little-endian):
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │ Header (64 bytes, magic SKEGHULL, format Saga,v1)│
//! ├──────────────────────────────────────────────────┤
//! │ Section table (3 entries × 28 bytes = 84 bytes)  │
//! ├──────────────────────────────────────────────────┤
//! │ SagaMeta section (64 bytes)                      │
//! │   tenant_id [16] built_at[i64] count[u64]        │
//! │   dim[u32] centroid_count[u32] reserved[24]      │
//! ├──────────────────────────────────────────────────┤
//! │ Centroids section: N × (cluster_size[u32] +      │
//! │                          vector f32[dim])        │
//! ├──────────────────────────────────────────────────┤
//! │ TagAggregate section: count[u32] +               │
//! │   M × (count[u32] + len[u16] + utf8 bytes)       │
//! ├──────────────────────────────────────────────────┤
//! │ Trailing CRC32C (4 bytes)                        │
//! └──────────────────────────────────────────────────┘
//! ```

mod centroid;
mod meta;
mod read;
mod tags;
mod write;

pub use centroid::Centroid;
pub use meta::{META_LEN, SagaMeta};
pub use read::read_saga;
pub use tags::TagEntry;
pub use write::write_saga;

use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::Result;
use crate::atomic::atomic_write;

/// Section type ids inside a SagaV1 file.
pub mod section_id {
    /// SagaMeta block.
    pub const META: u16 = 1;
    /// Centroids block.
    pub const CENTROIDS: u16 = 2;
    /// Tag aggregate block.
    pub const TAG_AGGREGATE: u16 = 3;
}

/// A saga in memory: enough to describe a peer's digest, score it
/// against a query, and serialise to SagaV1 on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Saga {
    /// 16-byte vault (tenant) identifier.
    pub tenant_id: [u8; 16],
    /// Unix seconds when the saga was built.
    pub built_at: i64,
    /// Records present in the vault at build time.
    pub record_count: u64,
    /// Embedding dimension. All centroid vectors share this length.
    pub embedding_dim: u32,
    /// Cluster centroids derived from the vault's vectors.
    pub centroids: Vec<Centroid>,
    /// Top tags with counts, sorted by count descending.
    pub tags: Vec<TagEntry>,
}

impl Saga {
    /// Construct an empty saga for a vault that hasn't been digested yet.
    pub fn empty(tenant_id: [u8; 16], embedding_dim: u32) -> Self {
        Self {
            tenant_id,
            built_at: 0,
            record_count: 0,
            embedding_dim,
            centroids: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Number of centroids in this saga.
    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// Serialise into the given writer at position 0.
    pub fn write_to<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        write_saga(self, writer)
    }

    /// Atomically write the saga to `path` via [`atomic_write`].
    pub fn write_to_path(&self, path: &Path) -> Result<()> {
        atomic_write(path, |f| {
            // `File` does not implement Seek when opened for write-only,
            // but `File` opened with create_new returns a handle that
            // does. The atomic_write helper opened it without
            // truncation; we wrap in a BufWriter? No - we need Seek.
            // `File` implements Seek directly when read+write or write
            // permissions are set on POSIX. We open with `write(true)`
            // which is fine.
            self.write_to(f)
        })
    }

    /// Deserialise from a `Read + Seek` source.
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        read_saga(reader)
    }

    /// Read a saga from `path`.
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }
}
