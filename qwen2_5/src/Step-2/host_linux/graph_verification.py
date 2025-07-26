import onnx
import onnxruntime as ort
import numpy as np

model_mha_path= "/nexa/qnn-expr/llama32-compute/qwen3_model/Step-2/host_linux/assets/artifacts/ar1-cl4096/src/onnx/qwen3.onnx"
model_sha_old_path = "/nexa/qnn-expr/llama32-compute/qwen3_model/Step-2/host_linux/assets/artifacts/ar1-cl4096/1_of_1/sha_output/ar1-cl4096_1_of_1.onnx"
model_sha_new_path = "modified_sha_model.onnx"

# Get input and output names from the model and inspect expected shapes
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# print(f"Model inputs: {[inp.name for inp in session.get_inputs()]}")
# print(f"Model outputs: {[out.name for out in session.get_outputs()]}")

# Print detailed input shape information
# for inp in session.get_inputs():
#     print(f"Input '{inp.name}': shape={inp.shape}, type={inp.type}")

# Generate test inputs based on expected shapes from the model
# Based on the error and input hints, using the correct shapes:
input_ids_int64 = np.random.randint(0, 10, size=(1, 1), dtype=np.int64)  # int64[1,1]
attention_mask = np.random.rand(1, 1, 1, 4096).astype(np.float32)  # float32[1,1,1,4096]
position_ids_cos = np.random.rand(1, 1, 1, 64).astype(np.float32)  # float32[1,1,1,64]
position_ids_sin = np.random.rand(1, 1, 1, 64).astype(np.float32)  # float32[1,1,1,64]
past_key_0_in = np.random.rand(1, 8, 128, 4095).astype(np.float32)  # float32[1,8,128,4095]
past_value_0_in = np.random.rand(1, 8, 4095, 128).astype(np.float32)  # float32[1,8,4095,128]

print("Generated all input tensors")
print(f"Input shapes: input_ids={input_ids_int64.shape}, attention_mask={attention_mask.shape}")
print(f"Position shapes: cos={position_ids_cos.shape}, sin={position_ids_sin.shape}")
print(f"Past KV shapes: key={past_key_0_in.shape}, value={past_value_0_in.shape}")

# Prepare input dictionary for inference
inputs = {
    "input_ids": input_ids_int64, 
    "position_ids_cos": position_ids_cos, 
    "position_ids_sin": position_ids_sin,
    "attention_mask": attention_mask,  
    "past_key_0_in": past_key_0_in, 
    "past_value_0_in": past_value_0_in
}

# Create inference session directly from file path to avoid 2GB protobuf limit
session = ort.InferenceSession(model_mha_path, providers=["CPUExecutionProvider"])
print("Created ONNX Runtime session for model_sha_new from file")
output_mha = session.run(None, inputs)
#print(output_mha)
# print stats of output_mha[0]
print("output_mha[0] stats: min = ", np.min(output_mha[0]), "max = ", np.max(output_mha[0]), "mean = ", np.mean(output_mha[0]))

# check another file
session = ort.InferenceSession(model_sha_old_path, providers=["CPUExecutionProvider"])
print("Created ONNX Runtime session for model_sha_old from file")
inputs["past_key_0_in"] = inputs["past_key_0_in"].transpose(1, 0, 2, 3)  # [1,8,128,4095] -> [8,1,128,4095]
inputs["past_value_0_in"] = inputs["past_value_0_in"].transpose(1, 0, 2, 3)  # [1,8,4095,128] -> [8,1,4095,128]
output_sha_old = session.run(None, inputs)
#print(output_sha_old)
# print stats of output_sha_old[0]
print("output_sha_old[0] stats: min = ", np.min(output_sha_old[0]), "max = ", np.max(output_sha_old[0]), "mean = ", np.mean(output_sha_old[0]))

# check another file
session = ort.InferenceSession(model_sha_new_path, providers=["CPUExecutionProvider"])
print("Created ONNX Runtime session for model_sha_new from file")
output_sha_new = session.run(None, inputs)
#print(output_sha_new)
# print stats of output_sha_new[0]
print("output_sha_new[0] stats: min = ", np.min(output_sha_new[0]), "max = ", np.max(output_sha_new[0]), "mean = ", np.mean(output_sha_new[0]))