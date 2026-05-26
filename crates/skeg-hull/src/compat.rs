//! Compatibility matrix between this reader and a file's declared format.
//!
//! For every known format the reader of version N reads files of
//! versions 1..=N. v0.1 only knows version 1 of every format, so the
//! matrix is trivial; the abstraction is in place so v0.2+ can extend
//! it without changing call sites.

use crate::format::{FormatId, FormatVersion, KV_V1, SAGA_V1, VAMANA_V1};

/// Result of checking a file's declared format against this reader.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Compatibility {
    /// Reader supports this format/version pair, proceed normally.
    Compatible,
    /// Reader supports it but observed something unusual (e.g. nonzero
    /// reserved fields suggesting a forward-compatible writer).
    CompatibleWithWarnings(Vec<String>),
    /// Reader does not support this format/version pair.
    Incompatible {
        /// What the file declared.
        found: FormatVersion,
        /// All format/version pairs this reader supports.
        supported: Vec<FormatVersion>,
    },
}

/// Highest version this reader supports for each format.
const fn highest_supported(format: FormatId) -> u16 {
    match format {
        FormatId::Kv => KV_V1.version,
        FormatId::Vamana => VAMANA_V1.version,
        FormatId::Saga => SAGA_V1.version,
    }
}

/// Set of format/version pairs this reader supports.
pub fn supported() -> Vec<FormatVersion> {
    vec![KV_V1, VAMANA_V1, SAGA_V1]
}

/// Check whether this reader can open a file with the given declared
/// format and version.
pub fn check_compatibility(found: FormatVersion) -> Compatibility {
    let max = highest_supported(found.format);
    if found.version == 0 || found.version > max {
        Compatibility::Incompatible {
            found,
            supported: supported(),
        }
    } else {
        Compatibility::Compatible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_for_known_versions() {
        for fv in supported() {
            assert_eq!(check_compatibility(fv), Compatibility::Compatible);
        }
    }

    #[test]
    fn incompatible_for_future_version() {
        let future = FormatVersion {
            format: FormatId::Saga,
            version: 99,
        };
        match check_compatibility(future) {
            Compatibility::Incompatible { found, .. } => assert_eq!(found, future),
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn incompatible_for_zero_version() {
        let bad = FormatVersion {
            format: FormatId::Saga,
            version: 0,
        };
        assert!(matches!(
            check_compatibility(bad),
            Compatibility::Incompatible { .. }
        ));
    }
}
