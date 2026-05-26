//! Tag aggregate block layout.

use crate::Result;
use crate::error::Error;

/// Aggregated tag with its occurrence count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagEntry {
    /// Times this tag appeared across the vault.
    pub count: u32,
    /// Tag string. Max 65535 bytes when encoded as UTF-8.
    pub tag: String,
}

/// Encode the full tag aggregate section.
///
/// Section layout:
/// - `u32` entry count
/// - for each entry: `u32` count, `u16` tag byte length, then the UTF-8 bytes
pub fn encode(entries: &[TagEntry]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4 + entries.len() * 8);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        let bytes = e.tag.as_bytes();
        if bytes.len() > u16::MAX as usize {
            return Err(Error::Malformed(format!(
                "tag too long: {} bytes (max {})",
                bytes.len(),
                u16::MAX
            )));
        }
        out.extend_from_slice(&e.count.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Decode the tag aggregate section from its on-disk bytes.
pub fn decode(buf: &[u8]) -> Result<Vec<TagEntry>> {
    if buf.len() < 4 {
        return Err(Error::Malformed("tag aggregate section too short".into()));
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut cur = 4usize;
    for i in 0..count {
        if cur + 6 > buf.len() {
            return Err(Error::Malformed(format!(
                "tag aggregate truncated at entry {i}"
            )));
        }
        let entry_count = u32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let len = u16::from_le_bytes(buf[cur..cur + 2].try_into().unwrap()) as usize;
        cur += 2;
        if cur + len > buf.len() {
            return Err(Error::Malformed(format!(
                "tag aggregate body truncated at entry {i}"
            )));
        }
        let tag = std::str::from_utf8(&buf[cur..cur + len])
            .map_err(|e| Error::Malformed(format!("invalid UTF-8 in tag {i}: {e}")))?
            .to_owned();
        cur += len;
        out.push(TagEntry {
            count: entry_count,
            tag,
        });
    }
    if cur != buf.len() {
        return Err(Error::Malformed(format!(
            "tag aggregate has {} trailing bytes",
            buf.len() - cur
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_roundtrip() {
        let entries = vec![
            TagEntry {
                count: 7,
                tag: "α".into(),
            },
            TagEntry {
                count: 3,
                tag: "long-tag-with-dashes".into(),
            },
            TagEntry {
                count: 1,
                tag: "".into(),
            },
        ];
        let bytes = encode(&entries).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, entries);
    }

    #[test]
    fn rejects_truncated() {
        let err = decode(&[0u8; 2]).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }

    #[test]
    fn rejects_oversized_tag() {
        let entries = vec![TagEntry {
            count: 1,
            tag: "x".repeat(u16::MAX as usize + 1),
        }];
        let err = encode(&entries).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }
}
