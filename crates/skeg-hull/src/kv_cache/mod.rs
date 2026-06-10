//! KvCacheV1 format: persistent LLM KV cache (skeg-kv-cache M0 prototype).
//!
//! Stores the post-prefill KV state of an LLM so that the next request with
//! the same `(model_fp, adapter_id, prefix_hash)` triple can skip prefill.
//!
//! File layout (all little-endian):
//!
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │ Header (64 bytes, magic SKEGHULL, format KvCache,v1)│
//! ├────────────────────────────────────────────────────┤
//! │ Section table (3 entries × 28 bytes = 84 bytes)    │
//! ├────────────────────────────────────────────────────┤
//! │ Identity section: model_fp[32] +                   │
//! │   adapter_present[u8] + adapter_id[32] +           │
//! │   prefix_hash[32] + meta block (variable)          │
//! ├────────────────────────────────────────────────────┤
//! │ Shape section: num_layers/n_kv_heads/head_dim/     │
//! │   num_tokens (u32 each) + dtype/layout (u8 each)   │
//! ├────────────────────────────────────────────────────┤
//! │ Blob section: kv_blob bytes + (optional) logits    │
//! ├────────────────────────────────────────────────────┤
//! │ Trailing CRC32C (4 bytes)                          │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! v0.1 (M0) is monolithic: the whole KV blob is one section. Future versions
//! (`hold-design.md`) split it per-layer with per-chunk CRC for partial reads.

mod blob;
mod identity;
mod read;
mod shape;
mod write;

pub use blob::Blob;
pub use identity::KvCacheMeta;
pub use read::read_kv_cache;
pub use shape::{KvCacheShape, KvDtype, KvLayout, QuantParams};
pub use write::write_kv_cache;

use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::Result;
use crate::atomic::atomic_write;

/// Section type ids inside a KvCacheV1 file.
pub mod section_id {
    /// Identity block: model_fp, adapter_id, prefix_hash, model_meta.
    pub const IDENTITY: u16 = 1;
    /// Shape block: layers/heads/dim/tokens/dtype/layout.
    pub const SHAPE: u16 = 2;
    /// KV blob + optional next_logits.
    pub const BLOB: u16 = 3;
}

/// In-memory representation of a KvCacheV1 file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCache {
    /// blake3 hash of the model base weights.
    pub model_fp: [u8; 32],
    /// blake3 hash of LoRA weights, or `None` if no adapter.
    pub adapter_id: Option<[u8; 32]>,
    /// blake3 hash of the canonical token sequence.
    pub prefix_hash: [u8; 32],
    /// Tensor shape metadata.
    pub shape: KvCacheShape,
    /// Model identity and timestamp.
    pub meta: KvCacheMeta,
    /// Raw KV bytes, layer-major: layer0_K | layer0_V | layer1_K | layer1_V | ...
    /// M0 keeps this monolithic; M1+ may split per-layer with per-chunk CRC.
    pub kv_blob: Vec<u8>,
    /// Optional logits for the next token (ds4 trick: persist alongside the
    /// KV so restage can skip 1 decode step).
    pub next_logits: Option<Vec<u8>>,
}

impl KvCache {
    /// Total payload bytes (KV blob + optional logits). Excludes header,
    /// section table, identity/shape metadata.
    pub fn payload_bytes(&self) -> u64 {
        self.kv_blob.len() as u64 + self.next_logits.as_ref().map_or(0, |l| l.len() as u64)
    }

    /// Serialise into the given writer at position 0.
    pub fn write_to<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        write_kv_cache(self, writer)
    }

    /// Atomically write to `path` via [`atomic_write`].
    pub fn write_to_path(&self, path: &Path) -> Result<()> {
        atomic_write(path, |f| self.write_to(f))
    }

    /// Deserialise from a `Read + Seek` source.
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        read_kv_cache(reader)
    }

    /// Read a KvCache from `path` using `read`/`pread` ordinary I/O. ds4-style
    /// path: no VM mapping added, useful when the host process already mmaps
    /// large weight files. Default reader; equivalent to [`Self::read_from_path`].
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }

    /// Read a KvCache from `path` using a read-only mmap **without madvise**.
    ///
    /// **NOT RECOMMENDED for cold-cache reads** — Gate G3 bench (2026-05-27)
    /// measures plain mmap at **3× slower than pread** on cold 1 GiB blob.
    /// The kernel's default readahead window is too small to hide page-fault
    /// latency on large blobs. Prefer [`Self::read_from_path_mmap_sequential`]
    /// or [`Self::read_from_path_mmap_willneed`].
    ///
    /// This method exists for benchmarking and for the rare case where the
    /// caller knows the file is already warm in page cache.
    pub fn read_from_path_mmap(path: &Path) -> Result<Self> {
        read_mmap_impl(path, None)
    }

    /// Read a KvCache from `path` using mmap with `madvise(MADV_SEQUENTIAL)`.
    ///
    /// **Default for `HullStorage`** per Gate G3 re-flip (2026-05-27, second
    /// iteration): wins 1.95× over pread on cold 1 GiB blob. Tells the kernel
    /// to do aggressive sequential readahead, hiding page-fault latency.
    pub fn read_from_path_mmap_sequential(path: &Path) -> Result<Self> {
        read_mmap_impl(path, Some(memmap2::Advice::Sequential))
    }

    /// Read a KvCache from `path` using mmap with `madvise(MADV_WILLNEED)`.
    ///
    /// More aggressive than [`Self::read_from_path_mmap_sequential`]: instructs
    /// the kernel to pre-fault all pages immediately. On Gate G3 bench wins
    /// 1.85× over pread on cold 1 GiB and 3.20× on cold 256 MiB. Worth trying
    /// when you know the blob will be fully consumed.
    pub fn read_from_path_mmap_willneed(path: &Path) -> Result<Self> {
        read_mmap_impl(path, Some(memmap2::Advice::WillNeed))
    }
}

fn read_mmap_impl(path: &Path, advice: Option<memmap2::Advice>) -> Result<KvCache> {
    use std::io::Cursor;
    let f = std::fs::File::open(path)?;
    let mmap = open_mmap(&f)?;
    if let Some(adv) = advice {
        // advise is best-effort; if the kernel rejects it, log the error but
        // proceed with the read anyway — correctness is unaffected.
        let _ = mmap.advise(adv);
    }
    let mut cursor = Cursor::new(&mmap[..]);
    KvCache::read_from(&mut cursor)
}

#[allow(unsafe_code)]
fn open_mmap(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: caller guarantees the file is single-writer at-rest. memmap2 docs
    // call this out as "unsafe because of TOCTOU on file truncation"; we accept
    // this trade because atomic_write ensures consumers see fully-written files.
    Ok(unsafe { memmap2::Mmap::map(file)? })
}
