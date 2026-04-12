# asr-onnx

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
