//! Atomic file replacement helper.
//!
//! The pattern is write-to-temp + fsync(temp) + rename + fsync(dir).
//! POSIX rename is atomic; readers see either the old file or the new
//! file, never a partial state. Hull files are expected to be replaced
//! this way so multi-process readers (notably hansa's membrane) can mmap
//! peer files safely.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;

/// Per-process monotonic counter used to disambiguate `atomic_write` temp
/// files across threads of the same process. Combined with PID + nanos
/// timestamp it produces unique names without coordination.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write a file atomically: build the content with `f` into a temp file
/// in the same directory, fsync it, then rename onto the target. The
/// containing directory is fsynced too so the rename is durable.
///
/// The temp file is named `<target>.tmp.<pid>.<nanos>.<counter>` so that
/// **cross-thread** writers in the same process don't collide on the temp
/// path. Bug fixed 2026-05-27: the prior `<target>.tmp.<pid>` scheme caused
/// EEXIST when two threads raced, and the loser's cleanup `remove_file`
/// could delete the winner's in-flight temp.
///
/// On success the temp is renamed onto the target; on failure the temp is
/// removed best effort.
pub fn atomic_write<P, F>(target: P, f: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut File) -> Result<()>,
{
    let target = target.as_ref();
    let dir = target.parent().ok_or_else(|| {
        crate::Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "target has no parent directory",
        ))
    })?;
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut tmp_name = target
        .file_name()
        .ok_or_else(|| {
            crate::Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "target has no file name",
            ))
        })?
        .to_os_string();
    tmp_name.push(format!(".tmp.{pid}.{nanos}.{counter}"));
    let tmp_path: PathBuf = dir.join(tmp_name);

    let result = (|| -> Result<()> {
        let mut tmp = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f(&mut tmp)?;
        tmp.flush()?;
        tmp.sync_all()?;
        drop(tmp);

        std::fs::rename(&tmp_path, target)?;

        // fsync the directory to make the rename durable. Best effort:
        // some filesystems (notably Windows) don't allow opening dirs;
        // we skip silently if so.
        if let Ok(dir_handle) = File::open(dir) {
            let _ = dir_handle.sync_all();
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn writes_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");

        atomic_write(&path, |f| {
            f.write_all(b"first")?;
            Ok(())
        })
        .unwrap();
        let mut s = String::new();
        File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        assert_eq!(s, "first");

        atomic_write(&path, |f| {
            f.write_all(b"second-and-longer")?;
            Ok(())
        })
        .unwrap();
        let mut s = String::new();
        File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        assert_eq!(s, "second-and-longer");
    }

    #[test]
    fn failure_removes_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let r: Result<()> = atomic_write(&path, |_f| Err(crate::Error::Malformed("nope".into())));
        assert!(r.is_err());
        assert!(!path.exists());
        // No leftover .tmp.<pid>.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(entries.is_empty(), "temp file left behind: {entries:?}");
    }
}
