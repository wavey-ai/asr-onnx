.PHONY: test
test:
	cargo test -- --nocapture

.PHONY: model
model:
	aws s3 cp s3://aldea-models/nemo-parakeet_tdt_ctc_110m.onnx model/nemo-parakeet_tdt_ctc_110m.onnx

.PHONY: testdata
testdata:
	aws s3 cp s3://harvard-lines/ ./testdata --recursive --exclude "*" --include "*.npy"


