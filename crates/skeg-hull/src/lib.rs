#![deny(unsafe_code)]
#![warn(missing_docs)]

//! `skeg-hull` - the on-disk format of skeg.
//!
//! Hull is the watertight boundary between data (durable) and engine
//! (replaceable). It defines a small family of binary formats - each
//! starting with the same 64-byte header and a section table - and ships
//! readers and writers that work against `Read + Seek` / `Write + Seek`
//! types. No engine dependency.
//!
//! In v0.1 only the `SagaV1` format is fully implemented; `KvV1` and
//! `VamanaV1` are reserved format IDs without reader/writer impls, kept
//! for forward-compatible header decoding.
//!
//! Concurrent multi-process reads are safe when writers replace files via
//! [`atomic::atomic_write`] (write-to-temp + rename + fsync). Mutating
//! files in place voids that guarantee.

pub mod atomic;
pub mod checksum;
pub mod compat;
pub mod error;
pub mod format;
pub mod header;
pub mod saga;

pub use atomic::atomic_write;
pub use compat::{Compatibility, check_compatibility};
pub use error::Error;
pub use format::{FormatId, FormatVersion, KV_V1, SAGA_V1, VAMANA_V1};
pub use header::{Header, SectionEntry};

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Re-exports for ergonomic consumption.
pub mod prelude {
    pub use crate::{
        Compatibility, Error, FormatId, FormatVersion, Header, KV_V1, Result, SAGA_V1,
        SectionEntry, VAMANA_V1, atomic_write, check_compatibility,
    };
}
