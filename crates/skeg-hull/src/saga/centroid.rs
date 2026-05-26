//! Centroid block layout.

use crate::Result;
use crate::error::Error;

/// A single cluster centroid.
#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    /// Number of vectors that mapped to this centroid at build time.
    pub cluster_size: u32,
    /// Centroid vector. Length must equal the saga's embedding_dim.
    pub vector: Vec<f32>,
}

/// Size in bytes of one centroid entry given the embedding dimension.
pub fn entry_size(dim: u32) -> usize {
    4 + (dim as usize) * 4
}

/// Encode all centroids back-to-back into a freshly allocated buffer.
pub fn encode(centroids: &[Centroid], dim: u32) -> Result<Vec<u8>> {
    let entry = entry_size(dim);
    let mut out = Vec::with_capacity(centroids.len() * entry);
    for c in centroids {
        if c.vector.len() != dim as usize {
            return Err(Error::Malformed(format!(
                "centroid vector len {} != saga dim {dim}",
                c.vector.len()
            )));
        }
        out.extend_from_slice(&c.cluster_size.to_le_bytes());
        for &v in &c.vector {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

/// Decode `count` centroids of dimension `dim` from `buf`.
pub fn decode(buf: &[u8], count: u32, dim: u32) -> Result<Vec<Centroid>> {
    let entry = entry_size(dim);
    let expected = (count as usize) * entry;
    if buf.len() != expected {
        return Err(Error::Malformed(format!(
            "centroid section length {} != expected {expected}",
            buf.len()
        )));
    }
    let mut out = Vec::with_capacity(count as usize);
    let mut cur = 0usize;
    for _ in 0..count {
        let cluster_size = u32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let mut vector = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let v = f32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap());
            vector.push(v);
            cur += 4;
        }
        out.push(Centroid {
            cluster_size,
            vector,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_roundtrip() {
        let cs = vec![
            Centroid {
                cluster_size: 10,
                vector: vec![0.1, 0.2, 0.3, 0.4],
            },
            Centroid {
                cluster_size: 99,
                vector: vec![-1.0, 0.0, 1.0, 2.0],
            },
        ];
        let bytes = encode(&cs, 4).unwrap();
        assert_eq!(bytes.len(), 2 * (4 + 4 * 4));
        let decoded = decode(&bytes, 2, 4).unwrap();
        assert_eq!(decoded, cs);
    }

    #[test]
    fn rejects_wrong_dim() {
        let cs = vec![Centroid {
            cluster_size: 1,
            vector: vec![1.0; 3],
        }];
        let err = encode(&cs, 4).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }

    #[test]
    fn rejects_wrong_length_on_decode() {
        let err = decode(&[0u8; 10], 1, 4).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)));
    }
}
