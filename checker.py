import onnx
from collections import Counter

model = onnx.load("best_preprocessed_excluded.onnx")
types = Counter()

for t in model.graph.initializer:
    types[t.data_type] += 1

print(types)



# ###################################



# from onnxruntime.quantization.preprocess import quant_pre_process

# model_fp32 = "best (1).onnx"
# model_prep = "best_preprocessed.onnx"

# try:
#     quant_pre_process(
#         model_fp32,
#         model_prep,
#         skip_optimization=False,
#         skip_symbolic_shape=False
#     )
#     print("✅ Pre-processing complete: 'best_preprocessed.onnx' created.")
# except Exception as e:
#     print(f"❌ Pre-processing failed: {e}")
