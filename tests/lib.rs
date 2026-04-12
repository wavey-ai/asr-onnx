use anyhow::Result;
use asr_onnx::{Config, JobMeta, TranscriberPool};
use edit_distance::edit_distance;
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::{fs, mem, path::Path, time::Duration};
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn normalize(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

#[test]
fn test_transcribe_all_npy() -> Result<()> {
    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_line_number(true))
        .with(EnvFilter::from_default_env().add_directive("debug".parse().unwrap()))
        .try_init();

    let data_dir = Path::new("testdata");
    if !data_dir.exists() {
        panic!("testdata dir missing");
    }

    let model_path = "model/nemo-parakeet_tdt_ctc_110m.onnx";
    let vocab_path = "model/vocab.txt";
    if !Path::new(model_path).exists() || !Path::new(vocab_path).exists() {
        panic!("model or vocab missing");
    }

    let mut feats = Vec::new();
    for entry in fs::read_dir(data_dir)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("npy") {
            let n = p.file_name().unwrap().to_str().unwrap().to_owned();
            match read_npy::<_, Array2<f32>>(&p) {
                Ok(a) => feats.push((n, a)),
                Err(e) => warn!("{:?} {:?}", p, e),
            }
        }
    }
    if feats.is_empty() {
        panic!("no npy files");
    }

    let device_ids = &[0];
    let cfg = Config::default().with_num_sessions(1);
    let pool = TranscriberPool::new(model_path, vocab_path, device_ids, cfg.clone())?;
    for _ in 0..device_ids.len() * cfg.num_sessions {
        pool.ready().recv()?;
    }

    for (name, mel) in &feats {
        let meta = JobMeta {
            seq: 0,
            chunk_id: 0,
        };
        pool.submit(name.clone(), mel.clone(), meta)?;
    }

    let rx = pool.result_rx().clone();
    let mut times = Vec::new();
    let mut dists = Vec::new();
    for (i, (n, _)) in feats.iter().enumerate() {
        use crossbeam_channel::RecvTimeoutError;
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(r) => {
                info!("{} \"{}\" {:.2}ms", r.name, r.text, r.task_time_ms);
                times.push(r.task_time_ms);
                let exp = n.trim_end_matches(".npy").replace('_', " ");
                dists.push(edit_distance(&normalize(&exp), &normalize(&r.text)));
            }
            Err(RecvTimeoutError::Timeout) => panic!("timeout {}", i),
            Err(RecvTimeoutError::Disconnected) => panic!("chan closed"),
        }
    }

    if times.len() > 20 {
        let avg = times.iter().skip(20).sum::<f64>() / (times.len() - 20) as f64;
        info!("avg after 20 {:.2}ms", avg);
    }
    let ed_avg = dists.iter().sum::<usize>() as f64 / dists.len() as f64;
    info!("avg edit dist {:.2}", ed_avg);

    mem::forget(pool);
    Ok(())
}
