use anyhow::{Result, bail};
use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};
use bytes::Bytes;
use crossbeam_channel::{Receiver, Sender, bounded};
use ndarray::{Array2, Array3, ArrayD, Ix3, IxDyn, s};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRTExecutionProvider,
};
use ort::logging::LogLevel;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor as OrtTensor;
use ort::value::ValueType;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::{
    fs,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};
use tracing::{debug, error, info, warn};

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn session_from_provider(path: &Path, provider: ExecutionProviderDispatch) -> Result<Session> {
    session_from_providers(path, [provider])
}

fn session_from_providers(
    path: &Path,
    providers: impl AsRef<[ExecutionProviderDispatch]>,
) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Info)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?
        .with_intra_threads(1)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

fn concrete_shape(value_type: &ValueType) -> Option<Vec<usize>> {
    let shape = value_type.tensor_shape()?;
    Some(
        shape
            .iter()
            .map(|dim| if *dim > 0 { *dim as usize } else { 1 })
            .collect(),
    )
}

fn last_positive_dim(value_type: &ValueType) -> Option<usize> {
    value_type
        .tensor_shape()?
        .iter()
        .rev()
        .find(|dim| **dim > 0)
        .map(|dim| *dim as usize)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub sample_rate: usize,
    pub window: usize,
    pub hop: usize,
    pub min_duration_s: usize,
    pub opt_duration_s: usize,
    pub max_duration_s: usize,
    pub blank_idx: usize,
    pub num_sessions: usize,
    pub trt_cache_dir: String,
    pub trt_components: String,
    pub trt_workspace_bytes: usize,
    pub trt_builder_optimization_level: u8,
    pub trt_fp16: bool,
    pub trt_detailed_build_log: bool,
    pub features_size: usize,
    pub subsampling_factor: usize,
    pub max_symbols_per_step: usize,
    pub force_ctc: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            window: 400,
            hop: 160,
            min_duration_s: 1,
            opt_duration_s: 4,
            max_duration_s: 60,
            blank_idx: 1024,
            num_sessions: 4,
            trt_cache_dir: "./.trt_cache".into(),
            trt_components: "encoder,joint_enc".into(),
            trt_workspace_bytes: 4 * 1024 * 1024 * 1024,
            trt_builder_optimization_level: 5,
            trt_fp16: true,
            trt_detailed_build_log: false,
            features_size: 128,
            subsampling_factor: 8,
            max_symbols_per_step: 10,
            force_ctc: false,
        }
    }
}

impl Config {
    pub fn apply_env_overrides(mut self) -> Self {
        if let Ok(v) = std::env::var("ASR_ONNX_PROFILE_MIN_S") {
            if let Ok(n) = v.parse::<usize>() {
                self.min_duration_s = n.max(1);
            }
        }
        let max_env = std::env::var("ASR_ONNX_PROFILE_MAX_S")
            .or_else(|_| std::env::var("ASR_ONNX_PROFILE_SECONDS"));
        if let Ok(v) = max_env {
            if let Ok(n) = v.parse::<usize>() {
                self.max_duration_s = n.max(1);
            }
        }
        if self.max_duration_s < self.min_duration_s {
            std::mem::swap(&mut self.min_duration_s, &mut self.max_duration_s);
        }
        self.min_duration_s = self.min_duration_s.max(1);
        self.max_duration_s = self.max_duration_s.max(self.min_duration_s);
        self.opt_duration_s =
            (((self.min_duration_s + self.max_duration_s) as f32) / 2.0).round() as usize;
        if let Ok(v) = std::env::var("ASR_ONNX_TRT_COMPONENTS") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                self.trt_components = trimmed.to_string();
            }
        }
        if let Ok(v) = std::env::var("ASR_ONNX_TRT_WORKSPACE_BYTES") {
            if let Ok(n) = v.parse::<usize>() {
                self.trt_workspace_bytes = n.max(1 << 20);
            }
        }
        if let Ok(v) = std::env::var("ASR_ONNX_TRT_BUILDER_OPT_LEVEL") {
            if let Ok(n) = v.parse::<u8>() {
                self.trt_builder_optimization_level = n.min(5);
            }
        }
        if let Ok(v) = std::env::var("ASR_ONNX_TRT_FP16") {
            self.trt_fp16 = matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES");
        }
        if let Ok(v) = std::env::var("ASR_ONNX_TRT_DETAILED_BUILD_LOG") {
            self.trt_detailed_build_log =
                matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES");
        }
        self
    }
    pub fn with_num_sessions(mut self, v: usize) -> Self {
        self.num_sessions = v;
        self
    }
    pub fn min_seq_len(&self) -> usize {
        (self.sample_rate * self.min_duration_s - self.window) / self.hop + 1
    }
    pub fn opt_seq_len(&self) -> usize {
        (self.sample_rate * self.opt_duration_s - self.window) / self.hop + 1
    }
    pub fn max_seq_len(&self) -> usize {
        (self.sample_rate * self.max_duration_s - self.window) / self.hop + 1
    }
    pub fn hop(&self) -> usize {
        self.hop
    }
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn uses_tensorrt_for(&self, component: ModelComponent) -> bool {
        self.trt_components
            .split(',')
            .map(|item| item.trim().to_ascii_lowercase())
            .any(|item| item == "all" || item == component.config_token())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelComponent {
    Encoder,
    Decoder,
    JointEnc,
    JointPred,
    JointNet,
}

impl ModelComponent {
    fn config_token(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decoder => "decoder",
            Self::JointEnc => "joint_enc",
            Self::JointPred => "joint_pred",
            Self::JointNet => "joint_net",
        }
    }

    fn cache_prefix(self) -> &'static str {
        self.config_token()
    }
}

fn encoded_steps_for(seq_len: usize, subsampling_factor: usize) -> usize {
    seq_len.div_ceil(subsampling_factor.max(1)).max(1)
}

fn input_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    let session = session_from_provider(path, CPUExecutionProvider::default().build())?;
    Ok(session
        .inputs()
        .iter()
        .map(|input| {
            (
                input.name().to_string(),
                concrete_shape(input.dtype()).unwrap_or_default(),
            )
        })
        .collect())
}

fn format_profile_entry(name: &str, dims: &[usize]) -> String {
    format!(
        "{}:{}",
        name,
        dims.iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn tensorrt_profile(
    component: ModelComponent,
    config: &Config,
    shapes: &[(String, Vec<usize>)],
    seconds: usize,
) -> Option<String> {
    let seq_len = ((config.sample_rate * seconds) - config.window) / config.hop + 1;
    let time_encoded = encoded_steps_for(seq_len, config.subsampling_factor);
    let mut entries = Vec::new();

    for (name, shape) in shapes {
        if shape.is_empty() {
            continue;
        }
        let mut dims = shape.clone();
        match component {
            ModelComponent::Encoder if name == "audio_signal" && dims.len() >= 3 => {
                dims[0] = 1;
                dims[1] = config.features_size;
                dims[2] = seq_len;
                entries.push(format_profile_entry(name, &dims));
            }
            ModelComponent::JointEnc if name == "encoder_outputs" && dims.len() >= 3 => {
                dims[0] = 1;
                dims[1] = time_encoded;
                entries.push(format_profile_entry(name, &dims));
            }
            _ => {}
        }
    }

    if entries.is_empty() {
        None
    } else {
        Some(entries.join(","))
    }
}

fn provider_chain(
    component: ModelComponent,
    device_id: usize,
    config: &Config,
    cache_dir: &str,
    shapes: Option<&[(String, Vec<usize>)]>,
) -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    if config.uses_tensorrt_for(component) {
        let mut tensorrt = TensorRTExecutionProvider::default()
            .with_device_id(device_id as i32)
            .with_engine_cache(true)
            .with_engine_cache_path(cache_dir)
            .with_engine_cache_prefix(component.cache_prefix())
            .with_timing_cache(true)
            .with_timing_cache_path(cache_dir)
            .with_max_workspace_size(config.trt_workspace_bytes)
            .with_builder_optimization_level(config.trt_builder_optimization_level)
            .with_force_sequential_engine_build(true)
            .with_layer_norm_fp32_fallback(true)
            .with_detailed_build_log(config.trt_detailed_build_log);
        if config.trt_fp16 {
            tensorrt = tensorrt.with_fp16(true);
        }
        if let Some(shapes) = shapes {
            if let Some(min_shapes) =
                tensorrt_profile(component, config, shapes, config.min_duration_s)
            {
                tensorrt = tensorrt.with_profile_min_shapes(min_shapes);
            }
            if let Some(opt_shapes) =
                tensorrt_profile(component, config, shapes, config.opt_duration_s)
            {
                tensorrt = tensorrt.with_profile_opt_shapes(opt_shapes);
            }
            if let Some(max_shapes) =
                tensorrt_profile(component, config, shapes, config.max_duration_s)
            {
                tensorrt = tensorrt.with_profile_max_shapes(max_shapes);
            }
        }
        providers.push(tensorrt.build().fail_silently());
    }

    providers.push(
        CUDAExecutionProvider::default()
            .with_device_id(device_id as i32)
            .build()
            .error_on_failure(),
    );
    providers
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabStyle {
    WordPiece,
    SentencePiece,
    Gpt2Bpe,
    CharCtc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampMode {
    Run,
    Peak,
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    pub timestamp_mode: TimestampMode,
    pub blank_merge_threshold_frames: usize,
    pub filter_specials: bool,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            timestamp_mode: TimestampMode::Run,
            blank_merge_threshold_frames: 0,
            filter_specials: true,
        }
    }
}

#[derive(Clone)]
pub struct VocabDecoder {
    labels: Arc<Vec<String>>,
    blank_idx: usize,
    style: VocabStyle,
    specials: HashSet<String>,
}

impl VocabDecoder {
    pub fn new<P: AsRef<Path>>(vocab_path: P, blank_idx: usize) -> Result<Self> {
        let txt = fs::read_to_string(vocab_path)?;
        let mut labels: Vec<String> = txt
            .lines()
            .map(|l| l.trim_end_matches('\r').to_owned())
            .collect();
        if blank_idx > labels.len() {
            bail!(
                "blank_idx {} out of range for vocab len {}",
                blank_idx,
                labels.len()
            );
        }
        if blank_idx == labels.len() {
            labels.push(String::new());
        } else {
            labels[blank_idx] = String::new();
        }
        let style = Self::detect_style(&labels);
        let specials = Self::default_specials();
        Ok(Self {
            labels: Arc::new(labels),
            blank_idx,
            style,
            specials,
        })
    }

    pub fn new_from_tokens_file<P: AsRef<Path>>(tokens_path: P) -> Result<Self> {
        let txt = fs::read_to_string(tokens_path)?;
        let tokens: Vec<String> = txt
            .lines()
            .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
            .collect();
        let blank_idx = tokens.len() - 1;
        let mut labels = tokens;
        if blank_idx < labels.len() {
            labels[blank_idx] = String::new();
        }
        let style = Self::detect_style(&labels);
        let mut specials = Self::default_specials();
        specials.insert("<blk>".into());
        Ok(Self {
            labels: Arc::new(labels),
            blank_idx,
            style,
            specials,
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.labels.len()
    }
    pub fn blank(&self) -> usize {
        self.blank_idx
    }

    fn detect_style(labels: &[String]) -> VocabStyle {
        if labels.iter().any(|s| s.starts_with("##")) {
            VocabStyle::WordPiece
        } else if labels.iter().any(|s| s.chars().next() == Some('▁')) {
            VocabStyle::SentencePiece
        } else if labels.iter().any(|s| s.starts_with('Ġ')) {
            VocabStyle::Gpt2Bpe
        } else {
            VocabStyle::CharCtc
        }
    }

    fn default_specials() -> HashSet<String> {
        ["<unk>", "<pad>", "<blank>", "<ctc_blank>", "<blk>"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn is_special_token(&self, s: &str) -> bool {
        s.is_empty() || self.specials.contains(s) || (s.starts_with('<') && s.ends_with('>'))
    }

    fn is_space_token(&self, s: &str) -> bool {
        match self.style {
            VocabStyle::CharCtc => s == "|" || s == "<space>" || s == " ",
            VocabStyle::SentencePiece => s == "▁",
            _ => false,
        }
    }

    fn normalize_piece<'a>(&self, s: &'a str) -> (&'a str, bool, bool) {
        match self.style {
            VocabStyle::WordPiece => (
                s.strip_prefix("##").unwrap_or(s),
                !s.starts_with("##"),
                s.starts_with("##"),
            ),
            VocabStyle::SentencePiece => {
                let start = s.starts_with('▁');
                (s.trim_start_matches('▁'), start, !start)
            }
            VocabStyle::Gpt2Bpe => {
                let start = s.starts_with('Ġ');
                (s.trim_start_matches('Ġ'), start, !start)
            }
            VocabStyle::CharCtc => (s, false, false),
        }
    }

    fn is_closing_punct(s: &str) -> bool {
        matches!(
            s,
            "." | "," | "!" | "?" | ";" | ":" | "%" | ")" | "]" | "}" | "'"
        ) || s == "'"
            || s == "\""
            || s == "\u{201D}"
            || s == "\u{2019}"
    }

    fn is_apostrophe(s: &str) -> bool {
        s == "'" || s == "'"
    }

    fn maybe_insert_space_before_piece(out: &mut String, starts_word: bool, piece: &str) {
        if out.is_empty() {
            return;
        }
        if starts_word {
            if Self::is_closing_punct(piece) {
                if out.ends_with(' ') {
                    out.pop();
                }
            } else {
                out.push(' ');
            }
        }
    }

    pub fn decode_ids_with_timestamps(
        &self,
        ids: &[usize],
        emit_t: &[usize],
        enc_steps: usize,
        in_frames: usize,
        hop: usize,
        sample_rate: usize,
    ) -> (String, Vec<(String, u32, u32)>) {
        if ids.is_empty() {
            return (String::new(), Vec::new());
        }
        let ms_per_step = (hop as f32 / sample_rate as f32)
            * 1000.0
            * (in_frames as f32 / enc_steps.max(1) as f32);
        let mut pieces: Vec<(String, u32, u32)> = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            if id == self.blank_idx {
                continue;
            }
            let tok = &self.labels[id];
            if self.is_special_token(tok) {
                continue;
            }
            let s = (emit_t[i] as f32 * ms_per_step).round() as u32;
            let e = ((emit_t.get(i + 1).copied().unwrap_or(emit_t[i]) as f32) * ms_per_step).round()
                as u32;
            pieces.push((tok.clone(), s, e.max(s + 1)));
        }
        let mut words: Vec<(String, u32, u32)> = Vec::new();
        let mut curr = String::new();
        let mut w_s: u32 = 0;
        let mut w_e: u32 = 0;
        let mut first = true;
        for (tok, s, e) in pieces {
            if tok == "▁" {
                if !first && !curr.is_empty() {
                    words.push((curr.clone(), w_s, w_e));
                    curr.clear();
                    first = true;
                }
                continue;
            }
            let (piece, starts_word, _) = self.normalize_piece(&tok);
            if Self::is_closing_punct(piece) || Self::is_apostrophe(piece) {
                if first {
                    curr.push_str(piece);
                    w_s = s;
                    w_e = e;
                    first = false;
                } else {
                    curr.push_str(piece);
                    w_e = e;
                }
                continue;
            }
            if matches!(
                self.style,
                VocabStyle::WordPiece | VocabStyle::SentencePiece | VocabStyle::Gpt2Bpe
            ) && starts_word
            {
                if !first && !curr.is_empty() {
                    words.push((curr.clone(), w_s, w_e));
                    curr.clear();
                }
                curr.push_str(piece);
                w_s = s;
                w_e = e;
                first = false;
            } else {
                if first {
                    w_s = s;
                    first = false;
                }
                curr.push_str(piece);
                w_e = e;
            }
        }
        if !first && !curr.is_empty() {
            words.push((curr.clone(), w_s, w_e));
        }
        let mut text = String::new();
        for (i, (w, _, _)) in words.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(w);
        }
        (text.trim().to_string(), words)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobMeta {
    pub seq: u32,
    pub chunk_id: u64,
}

#[derive(Clone, Debug)]
pub struct TranscriptionJob {
    pub job_id: u64,
    pub name: String,
    pub features: Arc<Array2<f32>>,
    pub meta: JobMeta,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TranscriptionResult {
    pub job_id: u64,
    pub name: String,
    pub text: String,
    pub task_time_ms: f64,
    pub gpu_time_ms: f64,
    pub session_id: usize,
    pub meta: JobMeta,
    pub words: Vec<(String, u32, u32)>,
}

pub fn marshal(tr: &TranscriptionResult) -> Bytes {
    Bytes::from(encode_to_vec(tr, config::standard()).expect("serialize"))
}
pub fn unmarshal(bytes: Bytes) -> TranscriptionResult {
    decode_from_slice(&bytes, config::standard())
        .expect("deserialize")
        .0
}

trait AsrEngine: Send {
    fn run(
        &mut self,
        features: &Array2<f32>,
        cfg: &Config,
        vocab: &VocabDecoder,
    ) -> Result<(String, Vec<(String, u32, u32)>, f64)>;
}

struct TdtEngine {
    encoder: Session,
    decoder: Session,
    joint_pred: Session,
    joint_enc: Session,
    joint_net: Session,
    blank_id: usize,
    hidden_size: usize,
    vocab_len: usize,
    state0_shape: Vec<usize>,
    state1_shape: Vec<usize>,
    joint_net_separate_inputs: bool,
}

impl TdtEngine {
    fn new(
        encoder: Session,
        decoder: Session,
        joint_pred: Session,
        joint_enc: Session,
        joint_net: Session,
        blank_id: usize,
        vocab_len: usize,
    ) -> Self {
        let hidden_size = decoder
            .inputs()
            .get(2)
            .and_then(|input| last_positive_dim(input.dtype()))
            .or_else(|| {
                joint_net
                    .inputs()
                    .first()
                    .and_then(|input| last_positive_dim(input.dtype()))
            })
            .or_else(|| {
                joint_pred
                    .outputs()
                    .first()
                    .and_then(|output| last_positive_dim(output.dtype()))
            })
            .unwrap_or(640);
        let state0_shape = decoder
            .inputs()
            .get(2)
            .and_then(|input| concrete_shape(input.dtype()))
            .unwrap_or_else(|| vec![2, 1, hidden_size]);
        let state1_shape = decoder
            .inputs()
            .get(3)
            .and_then(|input| concrete_shape(input.dtype()))
            .unwrap_or_else(|| vec![2, 1, hidden_size]);
        let joint_net_separate_inputs = joint_net.inputs().len() >= 2;
        Self {
            encoder,
            decoder,
            joint_pred,
            joint_enc,
            joint_net,
            blank_id,
            hidden_size,
            vocab_len,
            state0_shape,
            state1_shape,
            joint_net_separate_inputs,
        }
    }

    fn log_softmax(input: &ndarray::ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let x_minus_max = input.mapv(|v| v - max);
        let exps = x_minus_max.mapv(f32::exp);
        let sum_exp: f32 = exps.iter().sum();
        let log_sum = sum_exp.ln();
        Ok(x_minus_max.mapv(|v| v - log_sum))
    }

    fn encoder_infer(&mut self, features: &Array2<f32>) -> Result<Array3<f32>> {
        let (rows, cols) = features.dim();
        let length = OrtTensor::from_array(([1], vec![cols as i64]))?;
        let flat: Vec<f32> = features.iter().cloned().collect();
        let audio_signal = OrtTensor::from_array(([1, rows as i64, cols as i64], flat))?;
        let outputs = self.encoder.run([audio_signal.into(), length.into()])?;
        if outputs.len() == 0 {
            bail!("missing encoder output");
        }
        let enc = &outputs[0];
        let arr = enc.try_extract_array::<f32>()?;
        let arr3: Array3<f32> = arr.into_owned().into_dimensionality::<Ix3>()?;
        Ok(arr3.permuted_axes([0, 2, 1]))
    }

    fn joint_enc_infer(&mut self, encoder_output: &Array3<f32>) -> Result<ArrayD<f32>> {
        let encoder_view = encoder_output.view().into_dyn();
        let tensor = OrtTensor::from_array(encoder_view.to_owned())?;
        let outputs = self.joint_enc.run([tensor.into()])?;
        if outputs.len() == 0 {
            bail!("missing joint_enc output");
        }
        let out = &outputs[0];
        Ok(out.try_extract_array::<f32>()?.to_owned())
    }

    fn decoder_infer(
        &mut self,
        target: i64,
        state0: &ArrayD<f32>,
        state1: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        let targets = OrtTensor::from_array(([1, 1], vec![target as i32]))?;
        let target_length = OrtTensor::from_array(([1], vec![1i32]))?;
        let state0 = OrtTensor::from_array(state0.clone())?;
        let state1 = OrtTensor::from_array(state1.clone())?;
        let outputs = self.decoder.run([
            targets.into(),
            target_length.into(),
            state0.into(),
            state1.into(),
        ])?;
        if outputs.len() < 4 {
            bail!("decoder output count {} too small", outputs.len());
        }
        let out = &outputs[0];
        let mut dec = out.try_extract_array::<f32>()?;
        dec = dec.permuted_axes(IxDyn(&[0, 2, 1]));
        let s0 = &outputs[2];
        let s1 = &outputs[3];
        Ok((
            dec.to_owned(),
            s0.try_extract_array::<f32>()?.to_owned(),
            s1.try_extract_array::<f32>()?.to_owned(),
        ))
    }

    fn joint_pred_infer(&mut self, decoder_output: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input = OrtTensor::from_array(decoder_output.clone())?;
        let outputs = self.joint_pred.run([input.into()])?;
        if outputs.len() == 0 {
            bail!("missing joint_pred output");
        }
        let out = &outputs[0];
        Ok(out.try_extract_array::<f32>()?.to_owned())
    }

    fn joint_net_infer(&mut self, f: &ArrayD<f32>, g: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let outputs = if self.joint_net_separate_inputs {
            let ff = f.to_shape(IxDyn(&[1, 1, self.hidden_size]))?;
            let gg = g.to_shape(IxDyn(&[1, 1, self.hidden_size]))?;
            let f_input = OrtTensor::from_array(ff.to_owned())?;
            let g_input = OrtTensor::from_array(gg.to_owned())?;
            self.joint_net.run([f_input.into(), g_input.into()])?
        } else {
            let ff = f.to_shape(IxDyn(&[1, 1, 1, self.hidden_size]))?;
            let gg = g.to_shape(IxDyn(&[1, 1, 1, self.hidden_size]))?;
            let input = &ff + &gg;
            let input = OrtTensor::from_array(input.to_owned())?;
            self.joint_net.run([input.into()])?
        };
        if outputs.len() == 0 {
            bail!("missing joint_net output");
        }
        let out = &outputs[0];
        let res = out.try_extract_array::<f32>()?;
        Self::log_softmax(&res)
    }
}

impl AsrEngine for TdtEngine {
    fn run(
        &mut self,
        features: &Array2<f32>,
        cfg: &Config,
        vocab: &VocabDecoder,
    ) -> Result<(String, Vec<(String, u32, u32)>, f64)> {
        let start = Instant::now();
        let (_, cols) = features.dim();
        let encoder_output = self.encoder_infer(features)?;
        let (_, t_enc, _) = encoder_output.dim();
        let encoder_output_projected = self.joint_enc_infer(&encoder_output)?;
        let gpu_ms = start.elapsed().as_secs_f64() * 1e3;

        let mut state0 = ArrayD::zeros(IxDyn(&self.state0_shape));
        let mut state1 = ArrayD::zeros(IxDyn(&self.state1_shape));

        let mut ids: Vec<usize> = Vec::new();
        let mut emits: Vec<usize> = Vec::new();

        let mut time_index = 0usize;
        let mut safe_time_index = 0usize;
        let last_timestamps = t_enc.saturating_sub(1);
        let mut active = t_enc > 0;
        let mut advance = false;
        let mut label = self.blank_id as i64;

        while active {
            let (mut decoder_output, state0_next, state1_next) =
                self.decoder_infer(label, &state0, &state1)?;
            decoder_output = self.joint_pred_infer(&decoder_output)?;
            let f_slice = encoder_output_projected
                .slice(s![0, safe_time_index, ..])
                .to_owned()
                .into_dyn();
            let logits = self.joint_net_infer(&f_slice, &decoder_output)?;
            let vocab_slice = logits.slice(s![0, 0, 0, ..self.vocab_len]);
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in vocab_slice.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            label = best_idx as i64;

            let dur_slice = logits.slice(s![0, 0, 0, self.vocab_len..]);
            let mut dur_idx = 0usize;
            let mut dur_val = f32::NEG_INFINITY;
            for (i, &v) in dur_slice.iter().enumerate() {
                if v > dur_val {
                    dur_val = v;
                    dur_idx = i;
                }
            }
            let mut duration = dur_idx as i32;
            let mut time_index_current_labels = time_index;
            if duration == 0 {
                duration = 1;
            }
            time_index += duration as usize;
            safe_time_index = time_index.min(last_timestamps);
            if time_index >= last_timestamps {
                active = false;
            }
            if active && (label as usize) == self.blank_id {
                advance = true;
            }

            while advance {
                time_index_current_labels = time_index;
                let f_slice = encoder_output_projected
                    .slice(s![0, safe_time_index, ..])
                    .to_owned()
                    .into_dyn();
                let logits = self.joint_net_infer(&f_slice, &decoder_output)?;
                let vocab_slice = logits.slice(s![0, 0, 0, ..self.vocab_len]);
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in vocab_slice.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = i;
                    }
                }
                label = best_idx as i64;
                let dur_slice = logits.slice(s![0, 0, 0, self.vocab_len..]);
                let mut dur_idx = 0usize;
                let mut dur_val = f32::NEG_INFINITY;
                for (i, &v) in dur_slice.iter().enumerate() {
                    if v > dur_val {
                        dur_val = v;
                        dur_idx = i;
                    }
                }
                let mut duration = dur_idx as i32;
                if (label as usize) == self.blank_id && duration == 0 {
                    duration = 1;
                }
                time_index += duration as usize;
                safe_time_index = time_index.min(last_timestamps);
                if time_index >= last_timestamps {
                    active = false;
                }
                if active && (label as usize) == self.blank_id {
                    advance = true;
                } else {
                    advance = false;
                }
            }

            state0 = state0_next;
            state1 = state1_next;
            if (label as usize) != self.blank_id {
                ids.push(label as usize);
                emits.push(time_index_current_labels);
            }
        }

        let (text, words) =
            vocab.decode_ids_with_timestamps(&ids, &emits, t_enc, cols, cfg.hop, cfg.sample_rate);
        Ok((text, words, gpu_ms))
    }
}

pub struct SessionPool {
    submit_tx: Sender<TranscriptionJob>,
    result_rx: Receiver<TranscriptionResult>,
    ready_rx: Receiver<()>,
}

impl SessionPool {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        vocab: P,
        device_id: usize,
        config: Config,
    ) -> Result<Self> {
        let (tx_infer, rx_infer) = bounded::<TranscriptionJob>(4096);
        let (tx_res, rx_res) = bounded::<TranscriptionResult>(4096);
        let (ready_tx, ready_rx) = bounded::<()>(1024);
        let config = config.apply_env_overrides();
        let cache_dir = format!("{}/dev{}", config.trt_cache_dir, device_id);
        let _ = fs::create_dir_all(&cache_dir);

        for s in 0..config.num_sessions {
            let rx = rx_infer.clone();
            let tx = tx_res.clone();
            let ready_tx_clone = ready_tx.clone();
            let mdir = model_dir.as_ref().to_owned();
            let v = vocab.as_ref().to_owned();
            let c = config.clone();
            let cache_dir = cache_dir.clone();
            let did = device_id;
            let sid = s;

            thread::spawn(move || {
                let result: Result<()> = (|| {
                    info!(device = did, session = sid, "session start");
                    let model_path = Path::new(&mdir);
                    if !model_path.is_dir() {
                        bail!("TDT requires a model directory with component ONNX files");
                    }

                    // TDT models require CUDA Execution Provider; abort early if unavailable
                    if !CUDAExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false)
                    {
                        panic!(
                            "CUDA Execution Provider not available: TDT models require CUDA-enabled ONNX Runtime"
                        );
                    }

                    let enc_fp32 = model_path.join("encoder.onnx");
                    let enc_fp16 = model_path.join("encoder.fp16.onnx");
                    let enc_int8 = model_path.join("encoder.int8.onnx");

                    let dec_fp32 = model_path.join("decoder.onnx");
                    let dec_fp16 = model_path.join("decoder.fp16.onnx");
                    let dec_int8 = model_path.join("decoder.int8.onnx");

                    let joint_enc_fp32 = model_path.join("joint.enc.onnx");
                    let joint_enc_fp16 = model_path.join("joint.enc.fp16.onnx");
                    let joint_enc_int8 = model_path.join("joint.enc.int8.onnx");

                    let joint_pred_fp32 = model_path.join("joint.pred.onnx");
                    let joint_pred_fp16 = model_path.join("joint.pred.fp16.onnx");
                    let joint_pred_int8 = model_path.join("joint.pred.int8.onnx");

                    let joint_net_fp32 = model_path.join("joint.joint_net.onnx");
                    let joint_net_fp16 = model_path.join("joint.joint_net.fp16.onnx");
                    let joint_net_int8 = model_path.join("joint.joint_net.int8.onnx");

                    let tokens_path = model_path.join("tokens.txt");

                    let enc_path = if enc_fp16.exists() {
                        enc_fp16.clone()
                    } else if enc_fp32.exists() {
                        enc_fp32.clone()
                    } else {
                        enc_int8.clone()
                    };
                    let dec_path = if dec_fp16.exists() {
                        dec_fp16.clone()
                    } else if dec_fp32.exists() {
                        dec_fp32.clone()
                    } else {
                        dec_int8.clone()
                    };
                    let joint_enc_path = if joint_enc_fp16.exists() {
                        joint_enc_fp16.clone()
                    } else if joint_enc_fp32.exists() {
                        joint_enc_fp32.clone()
                    } else {
                        joint_enc_int8.clone()
                    };
                    let joint_pred_path = if joint_pred_fp16.exists() {
                        joint_pred_fp16.clone()
                    } else if joint_pred_fp32.exists() {
                        joint_pred_fp32.clone()
                    } else {
                        joint_pred_int8.clone()
                    };
                    let joint_net_path = if joint_net_fp16.exists() {
                        joint_net_fp16.clone()
                    } else if joint_net_fp32.exists() {
                        joint_net_fp32.clone()
                    } else {
                        joint_net_int8.clone()
                    };

                    if !enc_path.exists()
                        || !dec_path.exists()
                        || !joint_enc_path.exists()
                        || !joint_pred_path.exists()
                        || !joint_net_path.exists()
                    {
                        bail!(
                            "Missing required TDT component(s) in {}",
                            model_path.display()
                        );
                    }

                    let vocab_decoder = if tokens_path.exists() {
                        VocabDecoder::new_from_tokens_file(&tokens_path)?
                    } else {
                        VocabDecoder::new(&v, c.blank_idx)?
                    };
                    let vocab_len = vocab_decoder.vocab_len();
                    info!(
                        device = did,
                        session = sid,
                        vocab_len,
                        blank_idx = vocab_decoder.blank(),
                        "vocab loaded"
                    );

                    let trt_available = TensorRTExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false);
                    info!(
                        device = did,
                        session = sid,
                        trt_available,
                        trt_components = %c.trt_components,
                        "execution providers ready"
                    );

                    let enc_shapes = if c.uses_tensorrt_for(ModelComponent::Encoder) {
                        match input_shapes(&enc_path) {
                            Ok(shapes) => Some(shapes),
                            Err(error) => {
                                warn!(
                                    device = did,
                                    session = sid,
                                    error = %error,
                                    "failed to inspect encoder shapes for TensorRT; continuing without explicit profiles"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };
                    let joint_enc_shapes = if c.uses_tensorrt_for(ModelComponent::JointEnc) {
                        match input_shapes(&joint_enc_path) {
                            Ok(shapes) => Some(shapes),
                            Err(error) => {
                                warn!(
                                    device = did,
                                    session = sid,
                                    error = %error,
                                    "failed to inspect joint encoder shapes for TensorRT; continuing without explicit profiles"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let enc = session_from_providers(
                        &enc_path,
                        provider_chain(
                            ModelComponent::Encoder,
                            did,
                            &c,
                            &cache_dir,
                            enc_shapes.as_deref(),
                        ),
                    )?;

                    let dec = session_from_providers(
                        &dec_path,
                        provider_chain(ModelComponent::Decoder, did, &c, &cache_dir, None),
                    )?;

                    let jenc = session_from_providers(
                        &joint_enc_path,
                        provider_chain(
                            ModelComponent::JointEnc,
                            did,
                            &c,
                            &cache_dir,
                            joint_enc_shapes.as_deref(),
                        ),
                    )?;

                    let jpred = session_from_providers(
                        &joint_pred_path,
                        provider_chain(ModelComponent::JointPred, did, &c, &cache_dir, None),
                    )?;

                    let jnet = session_from_providers(
                        &joint_net_path,
                        provider_chain(ModelComponent::JointNet, did, &c, &cache_dir, None),
                    )?;

                    let mut engine = TdtEngine::new(
                        enc,
                        dec,
                        jpred,
                        jenc,
                        jnet,
                        vocab_decoder.blank(),
                        vocab_len,
                    );
                    let mut decoder_opt: Option<VocabDecoder> = None;
                    let _ = ready_tx_clone.send(());

                    loop {
                        let job = match rx.recv() {
                            Ok(j) => j,
                            Err(_) => break,
                        };
                        debug!(
                            device = did,
                            session = sid,
                            job_id = job.job_id,
                            seq = job.meta.seq,
                            rows = job.features.dim().0,
                            cols = job.features.dim().1,
                            "infer start"
                        );
                        let start = Instant::now();
                        if decoder_opt.is_none() {
                            decoder_opt = Some(if tokens_path.exists() {
                                VocabDecoder::new_from_tokens_file(&tokens_path)?
                            } else {
                                VocabDecoder::new(&v, c.blank_idx)?
                            });
                            let d = decoder_opt.as_ref().unwrap();
                            info!(
                                device = did,
                                session = sid,
                                vocab_len = d.vocab_len(),
                                blank_idx = d.blank(),
                                "decoder init"
                            );
                        }
                        let d = decoder_opt.as_mut().unwrap();
                        let (text, words, gpu_time_ms) = engine.run(&job.features, &c, d)?;
                        if text.trim().is_empty() {
                            warn!(
                                device = did,
                                session = sid,
                                job_id = job.job_id,
                                seq = job.meta.seq,
                                words = words.len(),
                                "Empty ASR transcription text"
                            );
                        }
                        let task_time_ms = start.elapsed().as_secs_f64() * 1e3;
                        debug!(
                            device = did,
                            session = sid,
                            job_id = job.job_id,
                            seq = job.meta.seq,
                            gpu_ms = gpu_time_ms,
                            task_ms = task_time_ms,
                            text_len = text.len(),
                            words = words.len(),
                            "infer done"
                        );
                        let _ = tx.send(TranscriptionResult {
                            job_id: job.job_id,
                            name: job.name,
                            text,
                            task_time_ms,
                            gpu_time_ms,
                            session_id: did,
                            meta: job.meta.clone(),
                            words,
                        });
                    }

                    info!(device = did, session = sid, "session end");
                    Ok(())
                })();
                if let Err(e) = result {
                    error!(device = did, session = sid, ?e, "session thread error");
                }
            });
        }

        Ok(Self {
            submit_tx: tx_infer,
            result_rx: rx_res,
            ready_rx,
        })
    }

    pub fn submit(&self, job: TranscriptionJob) -> Result<()> {
        debug!(
            job_id = job.job_id,
            seq = job.meta.seq,
            rows = job.features.dim().0,
            cols = job.features.dim().1,
            "pool submit"
        );
        self.submit_tx.send(job)?;
        Ok(())
    }

    pub fn result_rx(&self) -> &Receiver<TranscriptionResult> {
        &self.result_rx
    }
    pub fn ready_rx(&self) -> &Receiver<()> {
        &self.ready_rx
    }
}

#[derive(Clone)]
pub struct TranscriberPool {
    job_counter: Arc<AtomicU64>,
    submit_tx: Sender<TranscriptionJob>,
    result_rx: Receiver<TranscriptionResult>,
    ready_rx: Receiver<()>,
}

impl TranscriberPool {
    pub fn new<P: AsRef<Path>>(
        model: P,
        vocab: P,
        device_ids: &[usize],
        config: Config,
    ) -> Result<Self> {
        fn parse_device_range_from_env(key: &str) -> Option<Vec<usize>> {
            let raw = std::env::var(key).ok()?;
            let s = raw.trim();
            if s.is_empty() {
                return None;
            }
            let spec = s.replace("..=", "..");
            let mut parts = spec.split("..");
            let start = parts.next()?.trim().parse::<usize>().ok()?;
            let end = parts.next()?.trim().parse::<usize>().ok()?;
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            Some((lo..=hi).collect())
        }

        let effective_device_ids: Vec<usize> = if device_ids.is_empty() {
            parse_device_range_from_env("ASR_DEVICE_RANGE").unwrap_or_else(|| vec![0])
        } else {
            device_ids.to_vec()
        };
        info!(device_ids=?effective_device_ids, sessions = config.num_sessions, "pool init");

        let (submit_tx, submit_rx) = bounded::<TranscriptionJob>(10_000);
        let (result_tx, result_rx) = bounded::<TranscriptionResult>(10_000);
        let (ready_tx, ready_rx) = bounded::<()>(1024);
        let job_counter = Arc::new(AtomicU64::new(0));
        let pools: Arc<Mutex<Vec<SessionPool>>> = Arc::new(Mutex::new(Vec::new()));

        for &id in &effective_device_ids {
            let pool = SessionPool::new(&model, &vocab, id, config.clone())?;
            let rx = pool.result_rx().clone();
            let tx = result_tx.clone();
            let pool_ready_rx = pool.ready_rx().clone();
            let ready_tx_clone = ready_tx.clone();
            thread::spawn(move || {
                for r in rx {
                    let _ = tx.send(r);
                }
            });
            thread::spawn(move || {
                for _ in pool_ready_rx {
                    let _ = ready_tx_clone.send(());
                }
            });
            pools.lock().unwrap().push(pool);
        }

        {
            let pools_ref = Arc::clone(&pools);
            let submit_rx_cloned = submit_rx.clone();
            thread::spawn(move || {
                let mut idx = 0usize;
                for job in submit_rx_cloned {
                    let guard = pools_ref.lock().unwrap();
                    if !guard.is_empty() {
                        let _ = guard[idx % guard.len()].submit(job);
                        idx = idx.wrapping_add(1);
                    }
                }
            });
        }

        Ok(Self {
            job_counter,
            submit_tx,
            result_rx,
            ready_rx,
        })
    }

    pub fn submit(&self, name: String, features: Array2<f32>, meta: JobMeta) -> Result<u64> {
        let id = self.job_counter.fetch_add(1, Ordering::SeqCst);
        let job = TranscriptionJob {
            job_id: id,
            name,
            features: Arc::new(features),
            meta,
        };
        debug!(job_id = id, seq = job.meta.seq, "pool enqueue");
        self.submit_tx.send(job)?;
        Ok(id)
    }

    pub fn result_rx(&self) -> &Receiver<TranscriptionResult> {
        &self.result_rx
    }
    pub fn ready(&self) -> &Receiver<()> {
        &self.ready_rx
    }
}
