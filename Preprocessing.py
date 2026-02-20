from onnxruntime.quantization.preprocess import quant_pre_process

model_fp32 = "new_model/best.onnx"
model_prep = "new_model/best_preprocessed.onnx"

try:
    quant_pre_process(
        model_fp32,
        model_prep,
        skip_optimization=False,
        skip_symbolic_shape=False
    )
    print("Pre-processing complete: 'best_preprocessed.onnx' created.")
except Exception as e:
    print(f"Pre-processing failed: {e}")
