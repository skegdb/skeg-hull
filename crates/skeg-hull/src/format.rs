//! Format identifiers and version pairs.
//!
//! Each format has its own numeric ID and its own version, evolving
//! independently. v0.1 reserves three IDs (Kv, Vamana, Saga) but ships
//! reader/writer support only for Saga.

/// Top-level format identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u16)]
pub enum FormatId {
    /// KV tier records (engine-side; not implemented in v0.1).
    Kv = 1,
    /// Vamana graph + vectors (engine-side; not implemented in v0.1).
    Vamana = 2,
    /// Hansa saga digest (implemented in v0.1).
    Saga = 3,
    /// LLM KV cache for inference state persistence (skeg-kv-cache, M0).
    KvCache = 4,
}

impl FormatId {
    /// Decode a raw u16 into a known format ID.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            1 => Some(Self::Kv),
            2 => Some(Self::Vamana),
            3 => Some(Self::Saga),
            4 => Some(Self::KvCache),
            _ => None,
        }
    }

    /// Raw numeric value as stored on disk.
    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

/// A `(format, version)` pair, surfaced at the API to keep callers from
/// constructing invalid combinations.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FormatVersion {
    /// Top-level format family.
    pub format: FormatId,
    /// Numeric version inside that family. Bumped on incompatible changes.
    pub version: u16,
}

/// KV format, v1. Reserved; not implemented in v0.1.
pub const KV_V1: FormatVersion = FormatVersion {
    format: FormatId::Kv,
    version: 1,
};

/// Vamana format, v1. Reserved; not implemented in v0.1.
pub const VAMANA_V1: FormatVersion = FormatVersion {
    format: FormatId::Vamana,
    version: 1,
};

/// Saga format, v1. Implemented in v0.1.
pub const SAGA_V1: FormatVersion = FormatVersion {
    format: FormatId::Saga,
    version: 1,
};

/// LLM KV cache format, v1. Implemented in skeg-kv-cache M0 prototype.
pub const KV_CACHE_V1: FormatVersion = FormatVersion {
    format: FormatId::KvCache,
    version: 1,
};

/// Magic bytes prefixing every hull file. Eight bytes, ASCII.
pub const MAGIC: [u8; 8] = *b"SKEGHULL";

/// Header size in bytes. Fixed across all formats so headers can be read
/// without knowing the body.
pub const HEADER_LEN: usize = 64;

/// Section table entry size in bytes.
pub const SECTION_ENTRY_LEN: usize = 28;

/// Trailing file-level checksum size (CRC32C).
pub const FILE_CHECKSUM_LEN: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_id_roundtrip() {
        for id in [
            FormatId::Kv,
            FormatId::Vamana,
            FormatId::Saga,
            FormatId::KvCache,
        ] {
            assert_eq!(FormatId::from_u16(id.as_u16()), Some(id));
        }
        assert!(FormatId::from_u16(0).is_none());
        assert!(FormatId::from_u16(999).is_none());
    }

    #[test]
    fn magic_is_eight_bytes() {
        assert_eq!(MAGIC.len(), 8);
        assert_eq!(&MAGIC, b"SKEGHULL");
    }
}
