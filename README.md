# asr-onnx

`asr-onnx` consumes a split TDT export:

- `encoder.onnx`
- `decoder.onnx`
- `joint.enc.onnx`
- `joint.pred.onnx`
- `joint.joint_net.onnx`
- `tokens.txt`

This is not the same asset layout used by `parakeet-rs`, which consumes the stock two-file NeMo export (`encoder` + `decoder_joint`).

### Transcribe test sentences

Run `make testdata` to download the test audio from S3.

Alternatively, copy the `.npy` features from the `asr-torch` `testdir`.

From the `asr-onnx` directory:

```
docker build -t asr-onnx .
docker run --rm --gpus all \
  -v "$(pwd)/testdata":/workspace/testdata \
  -v "$(pwd)/model":/workspace/model \
  --entrypoint /test_entrypoint.sh \
  asr-onnx
```

### Export a split TDT model

The exporter loads a NeMo/Hugging Face Parakeet TDT checkpoint and writes the split ONNX bundle that `asr-onnx` expects.

Example:

```bash
make export-model MODEL=nvidia/parakeet-tdt-0.6b-v3 OUT=model/parakeet-tdt-0.6b-v3
```

Direct invocation:

```bash
python3 export/export_parakeet_tdt.py \
  --model-name nvidia/parakeet-tdt-0.6b-v3 \
  --output-dir model/parakeet-tdt-0.6b-v3
```

The exporter writes:

- `encoder.onnx`
- `encoder.onnx.data`
- `decoder.onnx`
- `decoder.onnx.data`
- `joint.enc.onnx`
- `joint.enc.onnx.data`
- `joint.pred.onnx`
- `joint.pred.onnx.data`
- `joint.joint_net.onnx`
- `joint.joint_net.onnx.data`
- `tokens.txt`
- `vocab.txt`
- `export.json`

The featurizer trace remains in `asr-torch`; this repo only handles the ONNX side of the pipeline.

### Export Cohere Transcribe ONNX graphs

The Cohere export path writes the encoder and decoder graphs used during the ONNX spike work:

- `encoder.onnx`
- `decoder_last_token.onnx`
- `decoder_prefill.onnx`
- `decoder_cached_step.onnx`
- Hugging Face tokenizer / processor metadata
- model config JSON
- prompt / token IDs in `export.json`
- `export.json`

Example:

```bash
python3 export/export_cohere_transcribe.py \
  --source CohereLabs/cohere-transcribe-03-2026 \
  --output-dir model/cohere-transcribe-03-2026 \
  --device cuda
```

The model is gated on Hugging Face, so the export host must already be authenticated for `CohereLabs/cohere-transcribe-03-2026`.

This export path does not mean the current Rust runtime can already serve Cohere. `asr-onnx` still only executes the split TDT bundle (`encoder` / `decoder` / `joint.*`) at runtime today.

### Export a staged Cohere bundle

If you want one bundle directory for the standalone Linux box, use the wrapper:

```bash
python3 export/export_cohere_bundle.py \
  --source CohereLabs/cohere-transcribe-03-2026 \
  --output-dir ../asr-api/models/cohere-transcribe-03-2026 \
  --device cuda \
  --num-devices 1
```

This stages:

- the Cohere ONNX encoder / decoder graphs
- Hugging Face processor / tokenizer metadata
- model config JSON
- `frontend_export.json` and, when traceable, `frontend.onnx`
- `featurizer_cuda*.pt` and `trace_report.json` when the frontend trace succeeds
- `bundle_export.json`
- a refreshed `SHA256SUMS`

Use `--skip-featurizer` or `--skip-frontend` if you only want part of the bundle.

### Export Cohere frontend assets

Use the frontend exporter to save the Hugging Face processor/tokenizer/config next to the ONNX graphs and to attempt a waveform frontend export:

```bash
python3 export/export_cohere_frontend.py \
  --source CohereLabs/cohere-transcribe-03-2026 \
  --output-dir model/cohere-transcribe-03-2026
```

The script always writes `frontend_export.json`. If the Cohere frontend resolves to a traceable Torch module, it also writes `frontend.onnx`. If not, the report captures that the frontend is still Python-only.

### Recreate the export environment

The exact GPU export environment used on `scratch-fm-gpu-de-fra-2-1` is checked in at:

- `python/requirements-export-cu128-torch210.lock.txt`

Bootstrap it with:

```bash
./python/setup-export-env.sh
```

That creates `.venv-export` and installs the pinned Python package set used for the ONNX asset export. The lockfile is a sanitized freeze of the working host environment; it intentionally omits unrelated editable local packages.
