export RUST_LOG=info

export LIBTORCH=/home/ubuntu/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
export ORT_LIB_LOCATION=/home/ubuntu/onnxruntime/build/Linux/Release

# !!!!!!!!!! FML
export ORT_STRATEGY=system
export LD_LIBRARY_PATH=$ORT_LIB_LOCATION:$LD_LIBRARY_PATH


