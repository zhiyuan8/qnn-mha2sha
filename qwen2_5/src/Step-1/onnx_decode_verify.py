import warnings

warnings.filterwarnings("ignore")


import torch
import sys
import os
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../qwen2_torch"))
os.environ["HF_TOKEN"] = "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"


script_dir = os.path.dirname(os.path.abspath(__file__))
qwen2_torch_path = os.path.join(os.path.dirname(script_dir), "qwen2_torch")
sys.path.append(qwen2_torch_path)

from hf_tokenizers import Tokenizer
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load the tokenizer and model
tokenizer_json_path = "/workspace/qnn-expr/experiments/qwen2_tokenizer.json"
tokenizer = Tokenizer(tokenizer_json_path)

eos_tokens = {151645}


from modeling_qwen2 import QNNQwen2, QNNLLMUtils

model = QNNQwen2.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model.qnn_init()
llm_config = model.config


SEQ_LEN = 128
KV_LEN = 4096 - SEQ_LEN  # We have concat during model inference, so we set 4096 = SEQ_LEN + KV_LEN
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

qnn_llm_utils = QNNLLMUtils(seq_len=SEQ_LEN, kv_len=KV_LEN, device=device, config=llm_config)
eos_token_id = {151645, 151643}

input_text = (
    "<|im_start|>user\nIntroduce Newton's first law of motion. Be short and concise.<|im_end|>"
    "\n<|im_start|>assistant\n"
)
input_ids_list = tokenizer.encode(input_text)  # a list of ints

n_past = 0
curr_len = len(input_ids_list)

if curr_len < SEQ_LEN:
    input_ids_list = input_ids_list + [151645] * (SEQ_LEN - curr_len)

input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
attention_mask = qnn_llm_utils.get_attention_mask(n_past, curr_len)
position_ids = qnn_llm_utils.get_position_ids(n_past, SEQ_LEN)
cos, sin = qnn_llm_utils.get_cos_sin(attention_mask, position_ids)  # attn_mask as dummpy input to get device
all_layer_kv_caches = qnn_llm_utils.get_kv_cache()
last_token_indices = torch.tensor([n_past + curr_len - 1], dtype=torch.long, device=device)


generated_ids = []
from tqdm import tqdm

## Here, we demonstrate the onnx model usage.
onnx_model = ort.InferenceSession(
    "/workspace/qnn-expr/llama32-compute/qwen2_mha_model/Step-2/host_linux/assets/models_ar_n/ar128-cl4096/onnx/qwen2.onnx"
)
inputs_keys = ["input_ids", "position_ids_cos", "position_ids_sin", "attention_mask"]
kv_inputs_keys = []
for i in range(llm_config.num_hidden_layers):
    kv_inputs_keys.append(f"past_key_{i}_in")
    kv_inputs_keys.append(f"past_value_{i}_in")
inputs_keys.extend(kv_inputs_keys)


print(f"input_ids: {input_ids}")
output_dir = "/workspace/sha_to_david"
input_data_dir = output_dir + "/input_data"
os.makedirs(input_data_dir, exist_ok=True)

print(f"input_ids: {input_ids}")
print(f"curr_len: {curr_len}")
import numpy as np

with torch.no_grad():
    for i in tqdm(range(1)):
        # prepare input value to ONNX, should be numpy array
        input_values = [input_ids, cos, sin, attention_mask, *all_layer_kv_caches]
        input_values = [input_value.cpu().numpy() for input_value in input_values]
        inputs_to_onnx = dict(zip(inputs_keys, input_values))

        ## save for the qnn-net-run sample
        for key, val in inputs_to_onnx.items():
            dtype = np.float32 if key == "input_ids" else np.float32
            val.astype(dtype).tofile(os.path.join(input_data_dir, f"{key}.raw"))

        outputs = onnx_model.run(None, inputs_to_onnx)

        # First output is logits, rest are new KV caches
        # output[0] has shape (1, seq_len, vocab_size)
        outputs = [torch.from_numpy(output).to(device) for output in outputs]
        logits = outputs[0]
        logits = logits[:, last_token_indices, :]

        next_token_id = torch.argmax(logits, dim=-1)
        next_token_id = next_token_id.unsqueeze(-1)
        generated_ids.append(next_token_id.item())

        if i == 0:
            print(f"logits stats, min, max and mean: {logits.min()}, {logits.max()}, {logits.mean()}")

        all_layer_kv_caches = qnn_llm_utils.update_kv_cache(all_layer_kv_caches, outputs[1:], n_past, curr_len)
        n_past += curr_len
        curr_len = 1
        # next_input_ids = [next_token_id] + [151645] * (SEQ_LEN - 1)
        # input_ids = torch.tensor(next_input_ids, dtype=torch.long, device=device).unsqueeze(
        #     0
        # )  # set bs_size = 1 manually
        # last_token_indices = torch.tensor([curr_len - 1], dtype=torch.long, device=device)
        # attention_mask = qnn_llm_utils.get_attention_mask(n_past, curr_len)
        # position_ids = qnn_llm_utils.get_position_ids(n_past, SEQ_LEN)
        # cos, sin = qnn_llm_utils.get_cos_sin(attention_mask, position_ids)

        # if next_token_id.item() in eos_tokens:
        #     break

generated_text = tokenizer.decode(generated_ids)
print(f"generated text: {generated_text} and generated ids: {generated_ids}")


## Now, prepare for the AR1 input data
SEQ_LEN = 1
KV_LEN = 4096 - SEQ_LEN
qnn_ar1_llm_utils = QNNLLMUtils(seq_len=SEQ_LEN, kv_len=KV_LEN, device=device, config=llm_config)
next_input_ids = [next_token_id] + [151645] * (SEQ_LEN - 1)
input_ids = torch.tensor(next_input_ids, dtype=torch.long, device=device).unsqueeze(0)
attention_mask = qnn_ar1_llm_utils.get_attention_mask(n_past, curr_len)
position_ids = qnn_ar1_llm_utils.get_position_ids(n_past, SEQ_LEN)
cos, sin = qnn_ar1_llm_utils.get_cos_sin(attention_mask, position_ids)

print("Now prepare the all layer kv caches")

for i in range(len(all_layer_kv_caches)):
    cache = all_layer_kv_caches[i]
    if i % 2 == 0:  # Even indices: shape [1, 2, 128, 3968] -> [1, 2, 128, 4095]
        padding = torch.zeros(
            cache.shape[0],
            cache.shape[1],
            cache.shape[2],
            4095 - cache.shape[3],
            dtype=cache.dtype,
            device=cache.device,
        )
        all_layer_kv_caches[i] = torch.cat([cache, padding], dim=3)
    else:  # Odd indices: shape [1, 2, 3968, 128] -> [1, 2, 4095, 128]
        padding = torch.zeros(
            cache.shape[0],
            cache.shape[1],
            4095 - cache.shape[2],
            cache.shape[3],
            dtype=cache.dtype,
            device=cache.device,
        )
        all_layer_kv_caches[i] = torch.cat([cache, padding], dim=2)

print(f"Updated cache shapes:")
for i, cache in enumerate(all_layer_kv_caches[:4]):  # Print first 4 to verify pattern
    print(f"all_layer_kv_caches[{i}].shape: {cache.shape}")


input_values = [
    input_ids.cpu().numpy(),
    cos.cpu().numpy(),
    sin.cpu().numpy(),
    attention_mask.cpu().numpy(),
    *[cache.cpu().numpy() for cache in all_layer_kv_caches],
]
input_keys = ["input_ids", "position_ids_cos", "position_ids_sin", "attention_mask", *kv_inputs_keys]

input_to_ar1 = dict(zip(input_keys, input_values))
for key, val in input_to_ar1.items():
    print(f"key: {key}, val: {val.shape}")
    val.astype(np.float32).tofile(os.path.join(input_data_dir, f"{key}.raw"))
