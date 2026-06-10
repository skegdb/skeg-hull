//! Identity section: model fingerprint + adapter id + prefix hash + model meta.

use crate::Result;
use crate::error::Error;

/// Model identity and timestamp metadata.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct KvCacheMeta {
    /// Unix seconds when the cache was built.
    pub created_at: i64,
    /// Tokenizer version used to encode the prefix. Bumps on tokenizer change.
    pub tokenizer_version: u32,
    /// Human-readable model name. Informational; the canonical identity
    /// is `model_fp` (hash of weights).
    pub model_name: String,
}

/// Identity payload layout (variable length due to model_name).
///
/// ```text
/// model_fp           [32 bytes]
/// adapter_present    [1 byte; 0 = None, 1 = Some]
/// adapter_id         [32 bytes; zeros if not present]
/// prefix_hash        [32 bytes]
/// created_at         [8 bytes, i64 LE]
/// tokenizer_version  [4 bytes, u32 LE]
/// model_name_len     [2 bytes, u16 LE]
/// model_name         [N bytes, utf8]
/// ```
pub(super) fn encode(
    model_fp: &[u8; 32],
    adapter_id: Option<&[u8; 32]>,
    prefix_hash: &[u8; 32],
    meta: &KvCacheMeta,
) -> Result<Vec<u8>> {
    let name_bytes = meta.model_name.as_bytes();
    if name_bytes.len() > u16::MAX as usize {
        return Err(Error::Malformed(format!(
            "model_name too long ({} bytes, max 65535)",
            name_bytes.len()
        )));
    }

    let mut out = Vec::with_capacity(32 + 1 + 32 + 32 + 8 + 4 + 2 + name_bytes.len());
    out.extend_from_slice(model_fp);
    if let Some(aid) = adapter_id {
        out.push(1);
        out.extend_from_slice(aid);
    } else {
        out.push(0);
        out.extend_from_slice(&[0u8; 32]);
    }
    out.extend_from_slice(prefix_hash);
    out.extend_from_slice(&meta.created_at.to_le_bytes());
    out.extend_from_slice(&meta.tokenizer_version.to_le_bytes());
    out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(name_bytes);
    Ok(out)
}

pub(super) type DecodedIdentity = ([u8; 32], Option<[u8; 32]>, [u8; 32], KvCacheMeta);

pub(super) fn decode(bytes: &[u8]) -> Result<DecodedIdentity> {
    const FIXED: usize = 32 + 1 + 32 + 32 + 8 + 4 + 2;
    if bytes.len() < FIXED {
        return Err(Error::Malformed(format!(
            "identity section too short: {} < {}",
            bytes.len(),
            FIXED
        )));
    }
    let model_fp: [u8; 32] = bytes[0..32].try_into().unwrap();
    let adapter_present = bytes[32];
    let adapter_raw: [u8; 32] = bytes[33..65].try_into().unwrap();
    let adapter_id = match adapter_present {
        0 => None,
        1 => Some(adapter_raw),
        other => {
            return Err(Error::Malformed(format!(
                "invalid adapter_present byte: {other}"
            )));
        }
    };
    let prefix_hash: [u8; 32] = bytes[65..97].try_into().unwrap();
    let created_at = i64::from_le_bytes(bytes[97..105].try_into().unwrap());
    let tokenizer_version = u32::from_le_bytes(bytes[105..109].try_into().unwrap());
    let name_len = u16::from_le_bytes(bytes[109..111].try_into().unwrap()) as usize;
    if bytes.len() < FIXED + name_len {
        return Err(Error::Malformed(format!(
            "identity section truncated: need {} bytes for model_name, have {}",
            name_len,
            bytes.len() - FIXED
        )));
    }
    let model_name = std::str::from_utf8(&bytes[FIXED..FIXED + name_len])
        .map_err(|e| Error::Malformed(format!("model_name not valid utf-8: {e}")))?
        .to_owned();
    Ok((
        model_fp,
        adapter_id,
        prefix_hash,
        KvCacheMeta {
            created_at,
            tokenizer_version,
            model_name,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip_with_adapter() {
        let meta = KvCacheMeta {
            created_at: 1_716_800_000,
            tokenizer_version: 3,
            model_name: "llama-3.1-8b-instruct".into(),
        };
        let model_fp = [0x11u8; 32];
        let adapter = [0x22u8; 32];
        let prefix = [0x33u8; 32];

        let bytes = encode(&model_fp, Some(&adapter), &prefix, &meta).unwrap();
        let (m, a, p, md) = decode(&bytes).unwrap();
        assert_eq!(m, model_fp);
        assert_eq!(a, Some(adapter));
        assert_eq!(p, prefix);
        assert_eq!(md, meta);
    }

    #[test]
    fn identity_roundtrip_no_adapter() {
        let meta = KvCacheMeta {
            created_at: 0,
            tokenizer_version: 1,
            model_name: String::new(),
        };
        let model_fp = [0u8; 32];
        let prefix = [0xFFu8; 32];

        let bytes = encode(&model_fp, None, &prefix, &meta).unwrap();
        let (_, a, _, _) = decode(&bytes).unwrap();
        assert_eq!(a, None);
    }
}
