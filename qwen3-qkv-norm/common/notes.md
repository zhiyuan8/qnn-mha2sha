# G2G

## What is ONNX?

ONNX is an open standard format for representing machine learning models. It is platform-, framework-, and hardware-agnostic.

### Core Components

1. **Computation Graph**
   - Directed acyclic graph (DAG) of operation nodes (e.g., `Add`, `MatMul`, `Relu`)
   - Each node has named inputs/outputs and references a specific ONNX operator

2. **Tensor Initializers**
   - Weights, biases, and constants used by operations

3. **Metadata Container**
   - Input/output tensor shapes and types
   - Versioning and domain information

#### Example Structure

```yaml
graph:
  inputs:
    - name: input_1
      shape: [1, 3, 224, 224]
  outputs:
    - name: output_1
      shape: [1, 1000]
  nodes:
    - op_type: Conv
      inputs: [input_1, conv1_weight, conv1_bias]
      outputs: [conv1_out]
    - op_type: Relu
      inputs: [conv1_out]
      outputs: [relu1_out]
    # ...
  initializers:
    - name: conv1_weight
      data: FloatTensor (shape: [64, 3, 7, 7])
    - name: conv1_bias
      data: FloatTensor (shape: [64])
```

ONNX models are stored as `.onnx` protobuf files (serialized using Protocol Buffers).

---

## MHA2SHA

**MHA2SHA** is a tool (using ONNX, post-AIMET export) to:

1. Convert Multi-Head Attention (MHA) ops to Single-Head Attention (SHA) ops in a new ONNX file.
    - Supports LLMs exceeding the 2GB protobuf limit via ONNX's `save_as_external_data` API.
2. Propagate encodings to these new ops in a new encodings file.
3. Convert Linear ops to Conv2d (1x1) ops.

### Core Steps

1. Identify all MHA patterns in the model.
2. Extract information from each pattern.
3. Create SHA replacements for each MHA block.
4. Update the model graph to use the new SHA blocks.
5. Perform cleanup and final optimizations.

---

### Code Structure for `mha2sha`

| Folder / File                                 | Description                                                                                                                        |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `mha2sha/converter.py`                        | Main conversion pipeline. Handles model loading, orchestrates conversion, applies transformations, manages input/output.           |
| `mha2sha/prequant_adaption.py`                | Pre-quantization adaptation logic. Prepares or modifies the model before quantization.                                             |
| `mha2sha/encoding_mapper.py`                  | Handles mapping between model encodings and internal representations.                                                              |
| `mha2sha/optimizer.py`                        | Core optimization logic for converting MHA to SHA. Main algorithms and transformation passes.                                      |
| `mha2sha/defs/`                               | Definitions, constants, and possibly schema files used throughout the codebase.                                                    |
| `mha2sha/htp_optimizations/`                  | Specialized optimizations for HTP (Hexagon Tensor Processor) or similar hardware.                                                  |
| `mha2sha/optimizer_extension/`                | Extensions to the main optimizer (e.g., RoPE, past key values, advanced attention mechanisms).                                     |
| `mha2sha/transformations/`                    | Modular transformation passes applied to the model graph.                                                                          |
| `mha2sha/utils/`                              | Utility functions and helpers for ONNX graph manipulation, logging, and other common tasks.                                        |

---

### Linear → Conv2d Conversion

- See: `common/utilities/module_converter.py`
- Linear is replaced by Conv2d with kernel size 1.

```python
def linear_to_conv2d(linear_module):
    try:
        conv2d_module = torch.nn.Conv2d(linear_module.in_features, linear_module.out_features, kernel_size=1)
        conv2d_module.weight.data.copy_(linear_module.weight[:, :, None, None])
        if linear_module.bias is not None:
            conv2d_module.bias.data.copy_(linear_module.bias)
    except AttributeError:
        warnings.warn(f'Expect input module to be an instance of torch.nn.Linear but got {type(linear_module)}, conversion won`t take any effects!')
    return conv2d_module
```

---

### Reshape and Permute Handling

- See: `common/G2G/MHA2SHA/src/python/mha2sha/htp_optimizations/linear_to_conv.py`

**Before Conv2d:**
- Input reshaped to `[1, 1, seq_len, vector_inp_dim]` (or `[1, 1, vector_inp_dim, seq_len]`)
- Permuted to `[1, vector_inp_dim, seq_len, 1]` (BCHW format)

**After Conv2d:**
- Output permuted back: `[1, vector_out_dim, seq_len, 1]` → `[1, 1, seq_len, vector_out_dim]`
- Squeezed if needed to match original shape.

```python
reshape_node, reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(input_node, [1, 1, -1, vector_inp_dim])
transpose_to_BCHW = self.mha2sha_optim._op_factory.get_transpose_op(reshape_node, [0, 3, 2, 1])
# ... Conv2d ...
transpose_to_BSE = self.mha2sha_optim._op_factory.get_transpose_op(conv_as_linear_node, [0, 3, 2, 1])
```

---

### MHA to SHA Conversion

- `get_qkv_conv`: Trace up the model graph to find Conv nodes for Q, K, and V.
- `create_sha_qkv_convs`: Create Conv operations for each attention head.
- `create_sha_conv_with_rope`: Create Conv operations for each attention head with RoPE.

---

### Encoding Mapping Merge

Why `_merge_encodings_mappings` and `_merge_encodings_mapping_files`?  
See `common/G2G/MHA2SHA/src/python/mha2sha/converter.py`

1. **Prequant Adaptation:**  
   - May change or add encodings for some tensors, producing a mapping file (e.g., `prequant_encoding_map.json`).
2. **MHA2SHA Conversion:**  
   - Replaces MHA with SHA, changing tensor names and creating new ones. Generates a new mapping file (e.g., `mha_to_sha_name_mapping.json`).

The final model needs a single, unified encoding mapping covering all tensors.

---

### What is BCHW?

- **B**atch, **C**hannels, **H**eight, **W**idth (standard Conv2d input format)
- Input is reshaped/permuted to BCHW before Conv2d, and restored after.

---

## How to Use: `mha2sha_how_to.ipynb`

`mha2sha` can be run in Python or via command line.

### Supported `--base-llm` Architectures

See `common/G2G/MHA2SHA/src/python/mha2sha/utils/base_llm_configs.py`

| `base_llm`      | Description                                                                                   |
|-----------------|----------------------------------------------------------------------------------------------|
| llama2          | Standard Llama2 config; no GQA, no LoRA, standard attention, strict checks, RoPE enabled.    |
| llama2_lora     | Llama2 + LoRA: enables LoRA logic, NCHW alignment, MHA conv, and LoRA alpha from input.      |
| llama3          | Like Llama2, but with GQA enabled (for Llama3's architecture).                               |
| sd_2.1          | All flags off except NCHW alignment and MHA conv (for Stable Diffusion 2.1).                 |
| sd_2.1_lora     | All flags off except LoRA-specific ones (for LoRA-adapted Stable Diffusion 2.1).             |

### Example Command

Entry: `common/G2G/MHA2SHA/src/python/mha2sha/converter.py`

```bash
mha2sha-onnx-converter \
  --model-name example \
  --sha-export-path [PATH-TO-EXPORT] \
  --exported-model-path [PATH-TO-MODEL] \
  --exported-model-encoding-path [PATH-TO-ENCODINGS] \
  --base-llm llama2 \
  --mha-conv      # Treat model as already using Conv layers for MHA
  --nchw-aligned  # Treat model as using NCHW (channels-first) for Q/K/V projections
```

#### Flag Explanations

- `model`: Loaded ONNX model object (`ModelProto`)
- `model_path`: Path to ONNX model file
- `model_name`: Output model name (auto-generated if not provided)
- `sha_export_path`: Directory for converted model and artifacts
- `exported_model_encoding_path`: Path to AIMET encoding file (for quantization/encoding propagation)
- `prepared_model`: Boolean, if model is "prepared" (affects pre-processing)
- `handle_rope_ops`: Handle Rotary Position Embedding (RoPE) ops
- `handle_past_key_value`: Handle past key/value cache logic
- `replace_linear_with_conv`: Replace Linear layers with Conv1x1 (HTP optimization)
- `disable_auto_attn_finder`: Disable automatic attention pattern finder
- `position_ids`: Path/flag for position ids (cos/sin) in model
- `mha_conv`: Treat model as already using Conv layers for MHA
- `nchw_aligned`: Treat model as using NCHW (channels-first) for Q/K/V projections
- `lora_model`: Model uses LoRA adaptors
- `lora_adaptor_list`: YAML file listing LoRA adaptors and encodings
- `create_input_lists`: Create `input_list.txt` and save test vectors
- `no_verification`: Skip verification step
- `log_level`: Logging verbosity (`warn`, `verbose`, `info`, `error`, `fatal`)
- `strict`: Strictly enforce golden RoPE pattern
- `build_ar`: Specify new AR (Attention Ratio) for AR-N to AR-M conversion

    - AR (Attention Ratio): Number of attention heads in a Multi-Head Attention (MHA) layer.
    - AR-N: Model originally has N attention heads (e.g., AR-8 means 8 heads).
    - AR-M: Convert the model to have M attention heads (e.g., AR-1 means single-head attention).

---

### Automatic Attention Finder

The "automatic attention finder" scans a model (typically ONNX) to automatically detect Multi-Head Attention (MHA) patterns.  
See `G2G/MHA2SHA/src/python/mha2sha/utils/auto_mha_finder.py`.

| Finder Type         | Pattern Searched For                                 |
|---------------------|-----------------------------------------------------|
| AttentionFinder     | [Linear, MatMul, Softmax, MatMul, Linear]           |
| AttentionFinderConv | [Conv, MatMul, Softmax, MatMul, Conv]               |
| QuickAttentionFinder| [MatMul, Softmax, MatMul]                           |

---

### Why `handle_rope_ops` for Llama2?

Rotation is parameterized by precomputed cosine and sine values, which depend on the position index and the dimension.  
See `common/G2G/MHA2SHA/src/python/mha2sha/optimizer_extension/rope_extension.py`.

---

## Utilities Code Structure

Located in `common/G2G/MHA2SHA/src/python/mha2sha/utils/`:

| File / Folder                | Description                                                                                      |
|------------------------------|--------------------------------------------------------------------------------------------------|
| auto_mha_finder.py           | Logic to automatically detect MHA patterns in ONNX models.                                       |
| base_llm_configs.py          | Base configuration flags and presets for different LLM architectures.                            |
| base_attn_encoding_mapper.py | Base class and utilities for mapping attention encodings during conversion.                      |
| clean.py                     | Utilities for cleaning up ONNX graphs and optimizing the model after conversion.                 |
| encoding_mapper_utils.py     | Helper functions for encoding mapping, merging, and propagation.                                 |
| logger.py                    | Logging utilities for standardized, colorized, and level-based logging.                          |
| lora_adaptor_converter.py    | Logic for handling LoRA adaptors and converting their encodings and weights.                     |
| op_factory.py                | Factory for creating ONNX operation nodes (e.g., Reshape, Transpose, Conv2d).                    |
| pretty_print.py              | Functions for pretty-printing ONNX graphs, nodes, and mappings for debugging.                    |
| utils.py                     | General-purpose utility functions (e.g., file I/O, name generation, misc helpers).              |
| arg_parser.py                | Command-line argument parsing logic for the MHA2SHA tool and its scripts.                        |
| attention_patterns.py        | Definitions and utilities for describing and matching attention block patterns in ONNX graphs.    |
