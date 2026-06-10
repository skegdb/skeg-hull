//! Blob section: KV bytes (mandatory) + optional logits.
//!
//! Layout:
//!
//! ```text
//! logits_present  [1 byte; 0 = None, 1 = Some]
//! kv_len          [8 bytes, u64 LE]
//! kv_blob         [kv_len bytes]
//! logits_len      [8 bytes, u64 LE; absent if logits_present == 0]
//! logits          [logits_len bytes; absent if logits_present == 0]
//! ```

use crate::Result;
use crate::error::Error;

/// Just a namespace marker. Re-exported for documentation locality.
pub struct Blob;

pub(super) fn encode(kv: &[u8], next_logits: Option<&[u8]>) -> Vec<u8> {
    let mut cap = 1 + 8 + kv.len();
    if let Some(l) = next_logits {
        cap += 8 + l.len();
    }
    let mut out = Vec::with_capacity(cap);
    out.push(if next_logits.is_some() { 1 } else { 0 });
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    out.extend_from_slice(kv);
    if let Some(l) = next_logits {
        out.extend_from_slice(&(l.len() as u64).to_le_bytes());
        out.extend_from_slice(l);
    }
    out
}

pub(super) fn decode(bytes: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    if bytes.len() < 9 {
        return Err(Error::Malformed(format!(
            "blob section too short for header: {}",
            bytes.len()
        )));
    }
    let logits_present = bytes[0];
    let kv_len = u64::from_le_bytes(bytes[1..9].try_into().unwrap()) as usize;
    if bytes.len() < 9 + kv_len {
        return Err(Error::Malformed(format!(
            "blob section truncated: need {} kv bytes, have {}",
            kv_len,
            bytes.len() - 9
        )));
    }
    let kv = bytes[9..9 + kv_len].to_vec();
    let cursor = 9 + kv_len;

    let logits = match logits_present {
        0 => {
            if bytes.len() != cursor {
                return Err(Error::Malformed(format!(
                    "blob section has trailing bytes when logits_present=0: {} extra",
                    bytes.len() - cursor
                )));
            }
            None
        }
        1 => {
            if bytes.len() < cursor + 8 {
                return Err(Error::Malformed(
                    "blob section truncated before logits length".into(),
                ));
            }
            let logits_len =
                u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap()) as usize;
            if bytes.len() != cursor + 8 + logits_len {
                return Err(Error::Malformed(format!(
                    "blob section logits length mismatch: declared {logits_len}, actual {}",
                    bytes.len() - cursor - 8
                )));
            }
            Some(bytes[cursor + 8..cursor + 8 + logits_len].to_vec())
        }
        other => {
            return Err(Error::Malformed(format!(
                "invalid logits_present byte: {other}"
            )));
        }
    };
    Ok((kv, logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip_with_logits() {
        let kv = vec![1u8, 2, 3, 4, 5];
        let logits = vec![10u8, 20, 30];
        let bytes = encode(&kv, Some(&logits));
        let (rkv, rlogits) = decode(&bytes).unwrap();
        assert_eq!(rkv, kv);
        assert_eq!(rlogits, Some(logits));
    }

    #[test]
    fn blob_roundtrip_without_logits() {
        let kv = vec![1u8, 2, 3];
        let bytes = encode(&kv, None);
        let (rkv, rlogits) = decode(&bytes).unwrap();
        assert_eq!(rkv, kv);
        assert_eq!(rlogits, None);
    }
}
