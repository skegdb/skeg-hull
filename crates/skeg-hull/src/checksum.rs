//! CRC32C helpers.
//!
//! Hull uses CRC32C (Castagnoli) at two levels: per-section payload
//! checksums and a trailing file-level checksum. Hardware acceleration on
//! x86_64 and aarch64 makes the cost negligible for the file sizes hull
//! targets.

/// Compute CRC32C over a buffer.
pub fn crc32c(buf: &[u8]) -> u32 {
    crc32c::crc32c(buf)
}

/// Streaming CRC32C builder. Useful when computing a checksum over a body
/// that is being written incrementally.
#[derive(Debug, Default, Clone, Copy)]
pub struct Crc32cBuilder {
    state: u32,
}

impl Crc32cBuilder {
    /// Start a fresh builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes into the builder.
    pub fn update(&mut self, buf: &[u8]) {
        self.state = crc32c::crc32c_append(self.state, buf);
    }

    /// Finalise and return the CRC32C value.
    pub fn finalize(self) -> u32 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_matches_oneshot() {
        let data = b"the rigging holds the sails";
        let oneshot = crc32c(data);
        let mut b = Crc32cBuilder::new();
        b.update(&data[..5]);
        b.update(&data[5..15]);
        b.update(&data[15..]);
        assert_eq!(b.finalize(), oneshot);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(Crc32cBuilder::new().finalize(), 0);
    }
}
