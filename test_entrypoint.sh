#!/bin/bash
set -e
cd /workspace/asr-onnx
cargo test --release -- --nocapture
