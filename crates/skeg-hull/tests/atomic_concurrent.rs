//! TDD: `atomic_write` must support concurrent writers from the same process.
//!
//! Bug discovered 2026-05-27 by `skeg-kv-cache/tests/concurrent_access.rs`:
//! the original `atomic_write` used a per-PID temp file name (`<target>.tmp.<pid>`),
//! which collides across threads of the same process. Symptom: `EEXIST` on
//! tmp open, followed by the loser's cleanup `remove_file` deleting the winner's
//! in-flight temp, causing the winner's rename to fail with `NotFound`.
//!
//! Fix: temp name must be unique per atomic_write *call*, not per process.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use skeg_hull::atomic_write;

#[test]
fn many_threads_writing_same_target_all_succeed() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("contended.bin");

    const N_WRITERS: usize = 16;
    const ITERS_PER_WRITER: usize = 25;

    let success = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for w in 0..N_WRITERS {
        let target = target.clone();
        let success = Arc::clone(&success);
        handles.push(std::thread::spawn(move || {
            for i in 0..ITERS_PER_WRITER {
                let payload = format!("writer={w} iter={i}");
                let r = atomic_write(&target, |f| {
                    use std::io::Write;
                    f.write_all(payload.as_bytes())?;
                    Ok(())
                });
                if r.is_ok() {
                    success.fetch_add(1, Ordering::Relaxed);
                } else {
                    panic!("writer {w} iter {i} failed: {:?}", r.err());
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("writer thread panicked");
    }

    let total = success.load(Ordering::Relaxed);
    assert_eq!(
        total,
        N_WRITERS * ITERS_PER_WRITER,
        "every atomic_write call must succeed"
    );

    // The final file content is whichever writer's payload won the last race.
    // Just check it's a valid one (not torn / empty).
    let content = std::fs::read_to_string(&target).unwrap();
    assert!(
        content.starts_with("writer="),
        "expected valid payload, got {content:?}"
    );
}

#[test]
fn no_temp_files_leak_after_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("clean.bin");

    const N: usize = 32;
    let mut handles = Vec::new();
    for w in 0..N {
        let target = target.clone();
        handles.push(std::thread::spawn(move || {
            atomic_write(&target, |f| {
                use std::io::Write;
                f.write_all(format!("w={w}").as_bytes())?;
                Ok(())
            })
            .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    std::thread::sleep(Duration::from_millis(20));

    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path() != target)
        .collect();
    assert!(
        entries.is_empty(),
        "leftover temp files after concurrent writes: {:?}",
        entries.iter().map(|e| e.path()).collect::<Vec<_>>()
    );
}
