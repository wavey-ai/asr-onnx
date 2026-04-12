.PHONY: test
test:
	cargo test -- --nocapture

MODEL ?= nvidia/parakeet-tdt-0.6b-v3
OUT ?= model/exported

.PHONY: model
model:
	aws s3 cp s3://aldea-models/nemo-parakeet_tdt_ctc_110m.onnx model/nemo-parakeet_tdt_ctc_110m.onnx

.PHONY: testdata
testdata:
	aws s3 cp s3://harvard-lines/ ./testdata --recursive --exclude "*" --include "*.npy"

.PHONY: export-model
export-model:
	python3 export/export_parakeet_tdt.py --model-name "$(MODEL)" --output-dir "$(OUT)"

