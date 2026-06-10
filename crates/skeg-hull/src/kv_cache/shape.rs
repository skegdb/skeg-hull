//! Shape section: tensor metadata for the KV cache.

use crate::Result;
use crate::error::Error;

/// KV element dtype, encoded as a single byte in the shape section.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum KvDtype {
    /// IEEE 754 binary16 (half precision).
    F16 = 1,
    /// Brain float 16 (truncated FP32 mantissa).
    Bf16 = 2,
    /// 8-bit float (E4M3 or E5M2; spec defers to the runtime).
    F8 = 3,
    /// 8-bit integer (quantized; scales live alongside in caller-provided metadata).
    Q8 = 4,
}

impl KvDtype {
    /// Raw numeric value as stored on disk.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode a raw byte into a known dtype, or error.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::F16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::F8),
            4 => Ok(Self::Q8),
            other => Err(Error::Malformed(format!("unknown KvDtype: {other}"))),
        }
    }

    /// Bytes per element. Q8/F8 are 1 byte; F16/BF16 are 2 bytes.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F8 | Self::Q8 => 1,
        }
    }
}

/// In-memory layout of the KV cache, encoded as a single byte in the shape section.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum KvLayout {
    /// Single tensor per layer, all tokens contiguous on the sequence axis.
    /// This is what mlx-lm `KVCache` produces (Day-1 finding).
    Contiguous = 1,
    /// List of fixed-size blocks per layer (PagedAttention-style). Reserved
    /// for M1+; M0 prototype always writes Contiguous.
    Paged = 2,
}

impl KvLayout {
    /// Raw numeric value as stored on disk.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode a raw byte into a known layout, or error.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Contiguous),
            2 => Ok(Self::Paged),
            other => Err(Error::Malformed(format!("unknown KvLayout: {other}"))),
        }
    }
}

/// Quantization parameters. Present only when the KV blob is quantized
/// via mlx-lm `QuantizedKVCache` (or equivalent). `dtype` on the parent
/// shape refers to the *scale/bias* dtype in that case (typically bf16);
/// the packed-element dtype is implicit (`uint32` per mlx-lm).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct QuantParams {
    /// Number of contiguous head-dim elements per shared scale+bias.
    pub group_size: u32,
    /// Bits per quantized element (typically 4 or 8).
    pub bits: u8,
}

/// Tensor shape metadata. Used to validate that a cache hit matches the
/// model the runtime is loaded with.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct KvCacheShape {
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of KV heads per layer (GQA-aware; not Q heads).
    pub n_kv_heads: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Sequence length (tokens) covered by this cache.
    pub num_tokens: u32,
    /// Element dtype. When `quant.is_some()` this is the scale/bias dtype.
    pub dtype: KvDtype,
    /// In-memory layout.
    pub layout: KvLayout,
    /// Quantization metadata. `None` means the blob is plain `dtype` elements.
    pub quant: Option<QuantParams>,
}

impl KvCacheShape {
    /// Expected size of the KV blob in bytes.
    ///
    /// - Unquantized: `num_layers × n_kv_heads × head_dim × num_tokens × element_bytes × 2`.
    /// - Quantized: per K or V per layer = `packed + 2×scales`, where
    ///   `packed = n_kv_heads × num_tokens × head_dim × bits / 8` and
    ///   `scales = n_kv_heads × num_tokens × (head_dim/group_size) × scale_bytes`.
    pub fn expected_blob_bytes(&self) -> u64 {
        let n_layers = u64::from(self.num_layers);
        let n_heads = u64::from(self.n_kv_heads);
        let head_dim = u64::from(self.head_dim);
        let n_tokens = u64::from(self.num_tokens);
        match self.quant {
            None => {
                n_layers * n_heads * head_dim * n_tokens * (self.dtype.element_bytes() as u64) * 2
            }
            Some(q) => {
                let bits = u64::from(q.bits);
                let group_size = u64::from(q.group_size);
                let scale_bytes = self.dtype.element_bytes() as u64;
                let packed = n_heads * n_tokens * head_dim * bits / 8;
                let scales = n_heads * n_tokens * head_dim / group_size * scale_bytes;
                let per_kv = packed + 2 * scales;
                n_layers * 2 * per_kv
            }
        }
    }

    /// Encode to a variable-length section payload (little-endian). 18 bytes
    /// unquantized, 23 bytes when `quant.is_some()` (5-byte trailer:
    /// `group_size u32 + bits u8`).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(if self.quant.is_some() { 23 } else { 18 });
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.n_kv_heads.to_le_bytes());
        out.extend_from_slice(&self.head_dim.to_le_bytes());
        out.extend_from_slice(&self.num_tokens.to_le_bytes());
        out.push(self.dtype.as_u8());
        out.push(self.layout.as_u8());
        if let Some(q) = self.quant {
            out.extend_from_slice(&q.group_size.to_le_bytes());
            out.push(q.bits);
        }
        out
    }

    /// Decode a shape section payload. Accepts 18 bytes (unquantized) or 23
    /// bytes (quantized with `group_size u32 + bits u8` trailer).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 18 && bytes.len() != 23 {
            return Err(Error::Malformed(format!(
                "KvCacheShape expects 18 or 23 bytes, got {}",
                bytes.len()
            )));
        }
        let quant = if bytes.len() == 23 {
            Some(QuantParams {
                group_size: u32::from_le_bytes(bytes[18..22].try_into().unwrap()),
                bits: bytes[22],
            })
        } else {
            None
        };
        Ok(Self {
            num_layers: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            n_kv_heads: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            head_dim: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            num_tokens: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            dtype: KvDtype::from_u8(bytes[16])?,
            layout: KvLayout::from_u8(bytes[17])?,
            quant,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_roundtrip() {
        let s = KvCacheShape {
            num_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            num_tokens: 16384,
            dtype: KvDtype::Bf16,
            layout: KvLayout::Contiguous,
            quant: None,
        };
        let bytes = s.encode();
        assert_eq!(bytes.len(), 18);
        let decoded = KvCacheShape::decode(&bytes).unwrap();
        assert_eq!(s, decoded);
    }

    #[test]
    fn shape_roundtrip_quantized() {
        let s = KvCacheShape {
            num_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            num_tokens: 1024,
            dtype: KvDtype::Bf16,
            layout: KvLayout::Contiguous,
            quant: Some(QuantParams {
                group_size: 64,
                bits: 4,
            }),
        };
        let bytes = s.encode();
        assert_eq!(bytes.len(), 23);
        let decoded = KvCacheShape::decode(&bytes).unwrap();
        assert_eq!(s, decoded);
        assert_eq!(decoded.quant.unwrap().bits, 4);
        assert_eq!(decoded.quant.unwrap().group_size, 64);
    }

    #[test]
    fn expected_size_llama_3_1_8b_16k() {
        // GQA: 8 KV heads, head_dim=128, num_tokens=16384, BF16, 32 layers
        // Expected blob = 32 × 8 × 128 × 16384 × 2 × 2 ≈ 2.14 GB
        let s = KvCacheShape {
            num_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            num_tokens: 16384,
            dtype: KvDtype::Bf16,
            layout: KvLayout::Contiguous,
            quant: None,
        };
        assert_eq!(s.expected_blob_bytes(), 2_147_483_648);
    }

    #[test]
    fn expected_size_q4_quant() {
        // 1 layer, 1 head, head_dim=128, num_tokens=4, group=64, bits=4, bf16 scales.
        // packed per K|V: 1×4×128×4/8 = 256 bytes
        // scales per K|V: 1×4×128/64 × 2 (bf16) = 16 bytes ; biases same = 16
        // per layer K+V: 2×(256 + 32) = 576 bytes
        let s = KvCacheShape {
            num_layers: 1,
            n_kv_heads: 1,
            head_dim: 128,
            num_tokens: 4,
            dtype: KvDtype::Bf16,
            layout: KvLayout::Contiguous,
            quant: Some(QuantParams {
                group_size: 64,
                bits: 4,
            }),
        };
        assert_eq!(s.expected_blob_bytes(), 576);
    }

    #[test]
    fn decode_rejects_22_bytes() {
        // 22 bytes is neither plain (18) nor quantized (23) — must error.
        let bytes = [0u8; 22];
        assert!(KvCacheShape::decode(&bytes).is_err());
    }
}
