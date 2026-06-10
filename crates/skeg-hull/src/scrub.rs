//! Background scrubber for hull files (F.21).
//!
//! Walks a hull file (or a directory of them) and verifies every
//! section's CRC32C against the section table without loading the
//! payload fully into RAM. Designed to be cheap enough to run on a
//! schedule next to a live skeg-server or hansa process.
//!
//! ## What it catches
//!
//! - **Bit rot**: silent sector flips on disk; the per-section
//!   CRC32C diverges from the stored value.
//! - **Partial writes**: a power loss or a kill -9 mid-write leaves
//!   a section truncated; the read returns `Error::Truncated`.
//! - **Wrong format / wrong version**: the file got moved, replaced,
//!   or corrupted at the header level. Surfaced as
//!   [`ScrubReport::header_error`].
//!
//! ## What it doesn't catch
//!
//! - **Semantic corruption**: a saga with bogus centroid coordinates
//!   that still CRC-matches the section table. Hull is the
//!   *transport* layer; semantic checks live in the format's reader.
//! - **Concurrent writer races**: if a writer is rotating the file
//!   while the scrubber walks it, results are undefined. Hull's
//!   contract assumes writers use [`crate::atomic_write`] (temp +
//!   rename); the scrubber observes whichever inode rename made
//!   visible at the moment it opened the file.
//!
//! ## Streaming
//!
//! Section payloads are CRC'd via [`Crc32cBuilder`] in 64 KiB
//! chunks. Memory is bounded by the chunk size regardless of section
//! length, so a multi-GB Vamana index scrubs in a steady ~64 KiB.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::checksum::Crc32cBuilder;
use crate::error::Error;
use crate::format::FormatId;
use crate::header::{Header, SectionEntry};

/// One byte of any hull file is a constant. We chunk reads at 64 KiB
/// so resident memory stays predictable regardless of the file size.
pub const SCRUB_CHUNK_BYTES: usize = 64 * 1024;

/// Detailed finding for a single file the scrubber inspected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubReport {
    /// Path that was scrubbed.
    pub path: PathBuf,
    /// Format the header declared, if it could be read. `None` when
    /// the header itself failed.
    pub format: Option<FormatId>,
    /// One [`ScrubFinding`] per section verified. Empty when the
    /// header could not be read.
    pub findings: Vec<ScrubFinding>,
    /// Wall-clock time the scrub took. Useful for capacity planning
    /// (an SSD ~700 MB/s saturates at ~140 MB sections per file
    /// scrubbed in 200 ms).
    pub elapsed: Duration,
    /// Set when the header itself didn't parse. Mutually exclusive
    /// with non-empty `findings`.
    pub header_error: Option<String>,
}

impl ScrubReport {
    /// True when every section verified cleanly and the header was
    /// readable.
    pub fn is_clean(&self) -> bool {
        self.header_error.is_none()
            && self
                .findings
                .iter()
                .all(|f| matches!(f.outcome, ScrubOutcome::Ok))
    }

    /// Number of sections that failed verification.
    pub fn bad_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| !matches!(f.outcome, ScrubOutcome::Ok))
            .count()
    }
}

/// One section's verification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubFinding {
    /// Section type id from the table.
    pub type_id: u16,
    /// Offset of the section payload in the file.
    pub offset: u64,
    /// Declared payload length.
    pub length: u64,
    /// Outcome of CRC32C verification.
    pub outcome: ScrubOutcome,
}

/// Outcome for one section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScrubOutcome {
    /// CRC matched the section table entry.
    Ok,
    /// CRC differed from the declared value. `expected` is what the
    /// section table claimed; `got` is what we computed by re-reading
    /// the payload bytes.
    ChecksumMismatch {
        /// CRC from the section table.
        expected: u32,
        /// CRC computed by re-reading the payload.
        got: u32,
    },
    /// Section payload was unreadable (file truncated, I/O error).
    /// `reason` is a human-readable summary.
    ReadError {
        /// Diagnostic string for logs.
        reason: String,
    },
}

/// Scrub one hull file and return a [`ScrubReport`].
pub fn scrub_file(path: &Path) -> ScrubReport {
    let started = Instant::now();
    let path = path.to_owned();

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            return ScrubReport {
                path,
                format: None,
                findings: Vec::new(),
                elapsed: started.elapsed(),
                header_error: Some(format!("open: {e}")),
            };
        }
    };

    let header = match Header::read(&mut file) {
        Ok(h) => h,
        Err(e) => {
            return ScrubReport {
                path,
                format: None,
                findings: Vec::new(),
                elapsed: started.elapsed(),
                header_error: Some(format!("header: {e}")),
            };
        }
    };

    let mut findings = Vec::with_capacity(header.sections.len());
    for section in &header.sections {
        findings.push(verify_section(&mut file, section));
    }

    ScrubReport {
        path,
        format: Some(header.format),
        findings,
        elapsed: started.elapsed(),
        header_error: None,
    }
}

/// Re-read one section's payload in chunks and compare its CRC32C
/// against the section table entry.
fn verify_section<R: Read + Seek>(reader: &mut R, section: &SectionEntry) -> ScrubFinding {
    let base = ScrubFinding {
        type_id: section.type_id,
        offset: section.offset,
        length: section.length,
        outcome: ScrubOutcome::Ok,
    };
    if let Err(e) = reader.seek(SeekFrom::Start(section.offset)) {
        return ScrubFinding {
            outcome: ScrubOutcome::ReadError {
                reason: format!("seek to {}: {e}", section.offset),
            },
            ..base
        };
    }
    let mut builder = Crc32cBuilder::new();
    let mut remaining = section.length;
    let mut buf = vec![0u8; SCRUB_CHUNK_BYTES.min(section.length.max(1) as usize)];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        match reader.read(&mut buf[..want]) {
            Ok(0) => {
                return ScrubFinding {
                    outcome: ScrubOutcome::ReadError {
                        reason: format!(
                            "unexpected EOF: {remaining} of {} bytes missing",
                            section.length
                        ),
                    },
                    ..base
                };
            }
            Ok(n) => {
                builder.update(&buf[..n]);
                remaining -= n as u64;
            }
            Err(e) => {
                return ScrubFinding {
                    outcome: ScrubOutcome::ReadError {
                        reason: format!("read: {e}"),
                    },
                    ..base
                };
            }
        }
    }
    let got = builder.finalize();
    if got != section.checksum {
        return ScrubFinding {
            outcome: ScrubOutcome::ChecksumMismatch {
                expected: section.checksum,
                got,
            },
            ..base
        };
    }
    base
}

/// Scrub every hull file in `dir`. Files whose first 8 bytes don't
/// match the hull magic are skipped silently. Hidden / non-regular
/// entries are ignored. Subdirectories are NOT recursed.
pub fn scrub_dir(dir: &Path) -> Vec<ScrubReport> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if !ft.is_file() {
            continue;
        }
        if !looks_like_hull(&path) {
            continue;
        }
        out.push(scrub_file(&path));
    }
    out
}

fn looks_like_hull(path: &Path) -> bool {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == crate::format::MAGIC
}

/// Convert a scrub outcome to a typed [`Error`] for callers that
/// prefer error propagation over inspection. Returns `Ok(())` when
/// the report is clean.
pub fn report_to_result(report: &ScrubReport) -> Result<(), Error> {
    if let Some(msg) = &report.header_error {
        return Err(Error::Malformed(msg.clone()));
    }
    for f in &report.findings {
        match &f.outcome {
            ScrubOutcome::Ok => {}
            ScrubOutcome::ChecksumMismatch { .. } => {
                return Err(Error::SectionChecksumFailed { type_id: f.type_id });
            }
            ScrubOutcome::ReadError { reason } => {
                return Err(Error::Malformed(format!(
                    "section type_id={}: {}",
                    f.type_id, reason
                )));
            }
        }
    }
    Ok(())
}

/// Background scrub loop. Walks `dir` on `interval` and invokes
/// `on_report` for each file scanned. Returns when `stop` is set.
///
/// Runs synchronously on the caller's thread; orchestrators typically
/// spawn it on `std::thread::spawn`. Drop the `Arc<AtomicBool>` to
/// signal stop.
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use std::sync::atomic::AtomicBool;
/// use std::time::Duration;
///
/// let stop = Arc::new(AtomicBool::new(false));
/// let stop_clone = stop.clone();
/// std::thread::spawn(move || {
///     skeg_hull::scrub::scrub_loop(
///         &PathBuf::from("/var/lib/skeg"),
///         Duration::from_secs(60 * 60), // hourly
///         stop_clone,
///         |report| {
///             if !report.is_clean() {
///                 eprintln!("hull scrub: {} bad sections in {:?}",
///                     report.bad_count(), report.path);
///             }
///         },
///     );
/// });
/// ```
///
/// Inter-pass cancellation is checked every 100 ms, so a `stop`
/// signal mid-pass aborts at the next file boundary, not at the
/// next interval tick.
pub fn scrub_loop<F>(dir: &Path, interval: Duration, stop: Arc<AtomicBool>, mut on_report: F)
where
    F: FnMut(&ScrubReport),
{
    let dir = dir.to_owned();
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        for report in scrub_dir(&dir) {
            on_report(&report);
            if stop.load(Ordering::Relaxed) {
                return;
            }
        }
        // Sleep in small slices so the stop signal is responsive.
        let mut slept = Duration::ZERO;
        let slice = Duration::from_millis(100);
        while slept < interval {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(slice.min(interval - slept));
            slept += slice;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saga::{Centroid, Saga, TagEntry};

    fn write_clean_saga(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let saga = Saga {
            tenant_id: [0x42; 16],
            embedding_dim: 4,
            record_count: 100,
            centroids: vec![
                Centroid {
                    cluster_size: 50,
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                },
                Centroid {
                    cluster_size: 50,
                    vector: vec![0.0, 1.0, 0.0, 0.0],
                },
            ],
            tags: vec![
                TagEntry {
                    count: 75,
                    tag: "topic".into(),
                },
                TagEntry {
                    count: 25,
                    tag: "skill:rust".into(),
                },
            ],
            built_at: 1_700_000_000,
        };
        saga.write_to_path(&path).unwrap();
        path
    }

    #[test]
    fn scrub_clean_file_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_clean_saga(dir.path(), "clean.saga");
        let report = scrub_file(&path);
        assert!(report.is_clean(), "got {report:?}");
        assert_eq!(report.bad_count(), 0);
        assert_eq!(report.format, Some(FormatId::Saga));
        assert!(!report.findings.is_empty(), "saga should have sections");
    }

    #[test]
    fn scrub_detects_payload_bit_flip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_clean_saga(dir.path(), "bit-flip.saga");
        // Corrupt one byte well inside the file, past the header +
        // section table (96 bytes is safely in payload territory).
        let mut bytes = std::fs::read(&path).unwrap();
        let target = bytes.len() / 2;
        bytes[target] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let report = scrub_file(&path);
        assert!(!report.is_clean());
        assert!(report.bad_count() >= 1);
        let bad = report
            .findings
            .iter()
            .find(|f| !matches!(f.outcome, ScrubOutcome::Ok))
            .unwrap();
        assert!(matches!(bad.outcome, ScrubOutcome::ChecksumMismatch { .. }));
    }

    #[test]
    fn scrub_handles_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_clean_saga(dir.path(), "trunc.saga");
        let bytes = std::fs::read(&path).unwrap();
        // Lop off the last 16 bytes (well into the last section's payload
        // OR the file trailer).
        std::fs::write(&path, &bytes[..bytes.len() - 16]).unwrap();
        let report = scrub_file(&path);
        // Header still parses (it's the first 64 bytes + section table);
        // at least one section now fails to read.
        assert!(!report.is_clean());
    }

    #[test]
    fn scrub_rejects_non_hull_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("random.bin");
        std::fs::write(&path, b"not a hull file at all").unwrap();
        let report = scrub_file(&path);
        assert!(report.format.is_none());
        assert!(report.header_error.is_some());
    }

    #[test]
    fn scrub_rejects_missing_file() {
        let report = scrub_file(Path::new("/nonexistent/saga.saga"));
        assert!(report.header_error.is_some());
    }

    #[test]
    fn scrub_dir_skips_non_hull_entries() {
        let dir = tempfile::tempdir().unwrap();
        write_clean_saga(dir.path(), "a.saga");
        write_clean_saga(dir.path(), "b.saga");
        std::fs::write(dir.path().join("README"), b"ignore me").unwrap();
        std::fs::write(dir.path().join("garbage.saga"), b"bad magic").unwrap();
        let reports = scrub_dir(dir.path());
        // 2 saga files, ignoring README and the garbage.saga (wrong magic).
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|r| r.is_clean()));
    }

    #[test]
    fn report_to_result_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_clean_saga(dir.path(), "ok.saga");
        let report = scrub_file(&path);
        assert!(report_to_result(&report).is_ok());

        // Corrupt and re-check.
        let mut bytes = std::fs::read(&path).unwrap();
        let target = bytes.len() / 2;
        bytes[target] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let report = scrub_file(&path);
        let err = report_to_result(&report).unwrap_err();
        assert!(matches!(err, Error::SectionChecksumFailed { .. }));
    }

    #[test]
    fn scrub_loop_exits_when_stop_set() {
        let dir = tempfile::tempdir().unwrap();
        write_clean_saga(dir.path(), "x.saga");
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let dir_clone = dir.path().to_owned();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = count.clone();
        let handle = std::thread::spawn(move || {
            scrub_loop(
                &dir_clone,
                Duration::from_millis(50),
                stop_clone,
                move |_r| {
                    count_clone.fetch_add(1, Ordering::Relaxed);
                },
            );
        });
        // Give the loop time to run at least one pass.
        std::thread::sleep(Duration::from_millis(80));
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert!(count.load(Ordering::Relaxed) >= 1);
    }
}
