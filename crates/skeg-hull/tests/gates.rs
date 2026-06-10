//! Performance gates for skeg-hull (F.21 scrubber + saga roundtrip).
//!
//! Run with:
//!   cargo test --release --test gates -p skeg-hull
//!
//! Gates skip in debug mode; release-only thresholds set with 2-3x
//! headroom over best-of-N on M-series Apple Silicon.

use std::path::PathBuf;
use std::time::Instant;

use skeg_hull::saga::{Centroid, Saga, TagEntry};
use skeg_hull::scrub::scrub_file;

fn skip_unless_release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("[gates] skipping in debug mode");
        true
    } else {
        false
    }
}

// ── Thresholds ──────────────────────────────────────────────────────

/// Scrubbing a saga with 64 centroids @ dim=128. The biggest section
/// is `centroids` at ~33 KB. Best-of-50 under 1 ms.
const GATE_SCRUB_64_CENTROIDS_MS: u128 = 1;

/// Scrubbing a saga with 128 centroids @ dim=768. ~393 KB centroids
/// section. Best-of-50 under 5 ms (CRC32C hardware-accelerated).
const GATE_SCRUB_128_768_MS: u128 = 5;

// ── Helpers ─────────────────────────────────────────────────────────

fn build_saga(centroid_count: usize, dim: u32) -> Saga {
    let centroids: Vec<Centroid> = (0..centroid_count)
        .map(|i| Centroid {
            cluster_size: 50,
            vector: (0..dim)
                .map(|d| ((i + d as usize) as f32) * 0.001)
                .collect(),
        })
        .collect();
    Saga {
        tenant_id: [0x42; 16],
        embedding_dim: dim,
        record_count: 100_000,
        centroids,
        tags: vec![TagEntry {
            count: 500,
            tag: "topic".into(),
        }],
        built_at: 1_700_000_000,
    }
}

fn write_saga(tmp: &tempfile::TempDir, label: &str, saga: &Saga) -> PathBuf {
    let path = tmp.path().join(format!("{label}.saga"));
    saga.write_to_path(&path).unwrap();
    path
}

// ── Gates ───────────────────────────────────────────────────────────

#[test]
fn gate_scrub_64_centroids_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = write_saga(&tmp, "saga-64-128", &build_saga(64, 128));
    // Warm-up FS cache.
    for _ in 0..3 {
        let _ = scrub_file(&path);
    }
    let mut best_ms = u128::MAX;
    for _ in 0..50 {
        let t = Instant::now();
        let report = scrub_file(&path);
        best_ms = best_ms.min(t.elapsed().as_millis());
        assert!(report.is_clean());
    }
    eprintln!(
        "[gate] scrub(64c, dim=128) best-of-50 = {best_ms} ms (cap {GATE_SCRUB_64_CENTROIDS_MS})"
    );
    assert!(
        best_ms <= GATE_SCRUB_64_CENTROIDS_MS,
        "scrub 64c best-of-50 = {best_ms} ms, gate {GATE_SCRUB_64_CENTROIDS_MS} ms"
    );
}

#[test]
fn gate_scrub_128_centroids_dim768_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = write_saga(&tmp, "saga-128-768", &build_saga(128, 768));
    for _ in 0..3 {
        let _ = scrub_file(&path);
    }
    let mut best_ms = u128::MAX;
    for _ in 0..50 {
        let t = Instant::now();
        let report = scrub_file(&path);
        best_ms = best_ms.min(t.elapsed().as_millis());
        assert!(report.is_clean());
    }
    eprintln!(
        "[gate] scrub(128c, dim=768) best-of-50 = {best_ms} ms (cap {GATE_SCRUB_128_768_MS})"
    );
    assert!(
        best_ms <= GATE_SCRUB_128_768_MS,
        "scrub 128c@768 best-of-50 = {best_ms} ms, gate {GATE_SCRUB_128_768_MS} ms"
    );
}
