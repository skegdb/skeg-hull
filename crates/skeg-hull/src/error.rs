//! Hull error type.

use crate::format::{FormatId, FormatVersion};

/// Hull error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// I/O failure while reading or writing.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File magic bytes did not match `SKEGHULL`.
    #[error("magic mismatch: found {found:02x?}, expected SKEGHULL")]
    MagicMismatch {
        /// The eight bytes that were read.
        found: [u8; 8],
    },

    /// File declared a format ID that this reader does not know.
    #[error("unknown format id: {format_id}")]
    UnknownFormat {
        /// Raw format ID from the file header.
        format_id: u16,
    },

    /// File is in a known format but at an unsupported version.
    #[error("unsupported version: format={format:?}, version={version}")]
    UnsupportedVersion {
        /// Format family.
        format: FormatId,
        /// Version inside that family.
        version: u16,
    },

    /// A section's payload CRC32C did not match the section table entry.
    #[error("section checksum failed: type_id={type_id}")]
    SectionChecksumFailed {
        /// Section type id (format-specific).
        type_id: u16,
    },

    /// The trailing file-level CRC32C did not match.
    #[error("file checksum failed")]
    FileChecksumFailed,

    /// File appears truncated (not enough bytes for the declared layout).
    #[error("file truncated")]
    Truncated,

    /// File is malformed in a way more specific errors don't cover.
    #[error("malformed: {0}")]
    Malformed(String),

    /// A section that was required by the format was not present.
    #[error("missing required section: type_id={type_id}")]
    MissingSection {
        /// Section type id expected.
        type_id: u16,
    },

    /// The file's declared format does not match what the caller asked for.
    #[error("format mismatch: file is {found:?}, caller expected {expected:?}")]
    FormatMismatch {
        /// What the file declared.
        found: FormatVersion,
        /// What the caller wanted.
        expected: FormatVersion,
    },
}
