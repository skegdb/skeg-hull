//! SagaMeta section layout.

use crate::Result;
use crate::error::Error;

/// On-disk size of the SagaMeta block.
pub const META_LEN: usize = 64;

/// Parsed SagaMeta fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SagaMeta {
    /// 16-byte vault id (matches rigging::VaultId on-disk).
    pub tenant_id: [u8; 16],
    /// Unix seconds when the saga was built.
    pub built_at: i64,
    /// Records present in the vault at build time.
    pub record_count: u64,
    /// Embedding dimension. All centroid vectors share this length.
    pub embedding_dim: u32,
    /// Number of centroids stored in the Centroids section.
    pub centroid_count: u32,
}

impl SagaMeta {
    /// Encode the block into `META_LEN` bytes.
    pub fn encode(&self) -> [u8; META_LEN] {
        let mut buf = [0u8; META_LEN];
        buf[0..16].copy_from_slice(&self.tenant_id);
        buf[16..24].copy_from_slice(&self.built_at.to_le_bytes());
        buf[24..32].copy_from_slice(&self.record_count.to_le_bytes());
        buf[32..36].copy_from_slice(&self.embedding_dim.to_le_bytes());
        buf[36..40].copy_from_slice(&self.centroid_count.to_le_bytes());
        // bytes 40..64 reserved, zero.
        buf
    }

    /// Decode a SagaMeta block from its on-disk bytes.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() != META_LEN {
            return Err(Error::Malformed(format!(
                "SagaMeta block expected {META_LEN} bytes, got {}",
                buf.len()
            )));
        }
        let mut tenant_id = [0u8; 16];
        tenant_id.copy_from_slice(&buf[0..16]);
        let built_at = i64::from_le_bytes(buf[16..24].try_into().unwrap());
        let record_count = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let embedding_dim = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let centroid_count = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        // bytes 40..64 reserved, ignored.
        Ok(Self {
            tenant_id,
            built_at,
            record_count,
            embedding_dim,
            centroid_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_roundtrip() {
        let m = SagaMeta {
            tenant_id: [7; 16],
            built_at: 1_700_000_000,
            record_count: 42,
            embedding_dim: 768,
            centroid_count: 16,
        };
        let bytes = m.encode();
        assert_eq!(bytes.len(), META_LEN);
        let decoded = SagaMeta::decode(&bytes).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn meta_rejects_short_buffer() {
        let err = SagaMeta::decode(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }
}
