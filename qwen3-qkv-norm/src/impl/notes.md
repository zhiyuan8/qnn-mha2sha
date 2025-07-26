# `impl/` folder

| File Name                | Description                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `dependencymanager.py`   | Manages dependencies between different components or steps in the model pipeline. Ensures correct execution order and resolves requirements for each step. |
| `qairtcompile.py`        | Handles the compilation of Q-AIR-T (Quantized AI Runtime) models. This includes parsing, optimizing, and preparing models for execution.                |
| `qairtexecute.py`        | Responsible for executing compiled Q-AIR-T models. Manages runtime, input/output handling, and execution flow.                                      |
| `stepsanitychecker.py`   | Performs sanity checks on the steps or stages of the pipeline. Validates correctness, order, and integrity of each step before execution.             |


# `qairtcompile.py`
Main workflow:
- Export models for different ARs (prepare_exports)
- Split exported ONNX models into subgraphs (split_model)
- Convert MHA to SHA format for each split (mha_to_sha_conversion)
- Convert ONNX to QNN DLC format (convert)
- Quantize DLCs (quantize)
- Generate context binaries for weight sharing (generate_context_binary)


Q1. Why `make_config_file` on the fly


Because Each split may have different graph names and settings:** The configuration must reflect the specific graphs (sub-models) being processed for that split, including their names and any weight sharing settings.

Q2. What does `input_output_tensor_mem_type` mean?


The `--input_output_tensor_mem_type` argument (with value `memhandle`) is passed to the `qnn-context-binary-generator` tool. It specifies how the input and output tensors' memory is managed when running the model on the target device. In summary, this argument controls how the model runtime will access and manage the memory for input and output tensors, which can impact performance and compatibility with the target hardware. 

Q3. Step-by-step summary with input/output for each step in QairtCompile

| Step                        | Input Files                                                                                                                      | Output Files                                                                                                 | Output Count (if ARNs=[1,128], num_splits=N) | Output Differs by         | Notes / Details                                                                                                 |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| **prepare_exports**         | - 1 base ONNX<br>- 1 base encoding<br>(from `${self.LLAMA_MODELS}`)                                                             | - 1 ONNX per AR<br>- 1 encoding per AR<br>(in `${workfolder}/models_ar_n/ar{arn}-cl{CL}/onnx/`)              | 2 ONNX, 2 encodings                         | AR, context length         | Modifies ONNX/encoding for each AR and context length using `change_hardcoding`.                                |
| **split_model**             | - 1 exported ONNX per AR<br>- 1 encoding per AR<br>(from previous step)                                                          | - `num_splits` ONNX per AR<br>- 1 symlinked encoding per split<br>- 1 symlinked ONNX per split<br>- test_vectors/ | 2×N ONNX, 2×N encodings (symlinked)         | AR, split                  | Each ONNX is a subgraph (partition) of the exported ONNX. Encodings and test vectors are symlinked per split.   |
| **mha_to_sha_conversion**   | - Split ONNX per AR, per split (except split 1)<br>(from previous step)                                                          | - 1 SHA ONNX per AR, per split (except split 1)<br>- 1 SHA encoding per AR, per split (except split 1 which is for input_ids to input_embeds)        | 2×(N-1) ONNX, 2×(N-1) encodings             | AR, split, SHA             | Converts MHA to SHA format for each split except split 1. Updates encoding accordingly.                         |
| **convert**                 | - For split 1: split ONNX + encoding<br>- For other splits: SHA ONNX + SHA encoding<br>(from previous steps)                      | - 1 DLC per AR, per split<br>(in `${split_work_dir}/converted_model/`)                                        | 2×N DLC                                     | AR, split                  | Converts ONNX (or SHA ONNX) to QNN DLC format (binary for deployment).                                          |
| **quantize**                | - 1 DLC per AR, per split<br>(from previous step)                                                                                | - 1 quantized DLC per AR, per split<br>(in `${split_work_dir}/compiled_model/`)                               | 2×N quantized DLC                           | AR, split                  | Quantizes weights/activations/biases to lower bitwidths.                                                        |
| **generate_context_binary** | - 2 quantized DLCs per split (one for each AR)<br>(from previous step)                                                           | - 1 context binary per split<br>(in `${workfolder}/artifacts/serialized_binaries/`)                           | N binaries                                  | split                      | Combines quantized DLCs for both ARs into a single context binary for weight sharing.                           |