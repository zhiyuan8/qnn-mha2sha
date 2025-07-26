<a name="readme-top"></a>

<div align="center">
<img src="imgs/brain_logo.png" alt="logo" width="140" height="auto"/>
</br>
  <h3><b>MHA2SHA</b></h3>
</div>

# 📗 Table of Contents

- [📖 About the Project](#about-project)
  - [Key Features](#key-features)
- [💻 Getting Started](#getting-started)
  - [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Run tests](#run-tests)
  - [Linting](#linter)
- [👥 Authors](#authors)
- [📝 License](#license)

<!-- PROJECT DESCRIPTION -->

# 📖 MHA2SHA <a name="about-project"></a>

**MHA2SHA** is a tool using [ONNX](https://stability.ai/news/stable-diffusion-public-release) for after [AIMET](https://github.com/quic/aimet) export, to convert Multi Head Attention operations into Single Head Attention operations, propogate encodings to these new operations, and convert Linear operations into Conv 1x1 operations for HTP backend support. This tool is mainly used for LLM's such as [LLaMA](https://llama.meta.com/) and [Stable Diffusion](https://stability.ai/news/stable-diffusion-public-release). A list of models supported and known to work are listed below.

The LWD for this tool can be found [here](https://confluence.qualcomm.com/confluence/pages/viewpage.action?pageId=1411336257).
<!-- Features -->

### Key Features <a name="key-features"></a>

- **MHA -> SHA**
    - Able to find MHA's inside a model and split them by the number of heads in each MHA into SHA's. This improves the latency of the model on device.
- **RoPE**
    - Some models - like LLaMA - have RoPE operations inside their MHA's. RoPE is handled py using this tool with the `--handle-rope-ops` flag.
- **Past Key/Value**
    - Models may have past key/value optimizations as apart of MHA. This can be handled with `--handle-past-key-value`.
- **KV Cache / BERT**
    - Models with LLM optimizations such as KV Cache and BERT are also handled by the tool. 
- **LORA adaptor**
    - Models with LORA adaptor on q, k, v branches will be handled with `--lora-model` flag.
- **Multiple LORA adaptor**
    - Multiple LORA adaptors and encodings can be passed in with lora mha model together and convert multiple lora adpators at the same time with `--lora-adaptor-list` to yaml file to list of LORA adaptor. See `/src/python/mha2sha/defs/lora_adaptor_list_example.yaml` for example.
- **GQA (Group query attention) models**
    - GQA models will be handled with `--gqa-model` flag. Noted that lora + GQA model is not supported 
    at the moment.
- **Encodings Propogated**
    - As this tool is used after AIMET, encodings that were exported are able to map to the SHA's created during the split of the MHA. This allows the model to have the exact same accuracy but minimize full pipeline time.
- **Linear -> Conv**
    - HTP backend requires Linear operations to updated to Convs, as the Conv operation is optimized for HTP. This tool can automatically update these and maintain names of tensors to match encodings. As this is an optional optimization, it can be enabled with the `--replace-linear-to-conv` flag.
- **Saving Large Models**
    - LLM's that exceed the 2GB protobuf limit are able to saved via ONNX's `save_as_external_data` API.
- **MHA-conv**
  - If the mha model has already applied with linear to conv, use the `--mha-conv` option. This will allow auto-mha-finder to search mha with conv and create sha-conv pattern instead. Also if `--nchw-aligned` (and `--handle-rope-ops`) are enabled, it will create an golden sha-conv pattern without redundent ops.
- **NCHW aligned model**
  - Models with input to Q, K, V proj and Out_proj are in NCHW format are NCHW aligned model. For LLMs,
  NCHW model expects input to have shape [ N, vector_dim, 1, context length ] (BD1L format)
- **Efficient One Step SHA-Conv for NCHW Aligned Model**
  - For models that are NCHW aligned and enable `--nchw-aligned` to let MHA2SHA ONNX optimizer knows it is NCHW aligned. Then the optmizer will create a node efficient sha-conv pattern from pattern designed in `aimet_extension/optimizer_extension.py`.
- **Create Dummy Data (Inputs & Goldens)**
  - At the end of this tool, the original model is compared to the exported on with random input data with ONNXRuntime. These outputs are saved under the `--sha-export-path` directory along with the outputs and an `input_list.txt` file for quick testing when go on device.
- **JSON of MHA Encoding Names to SHA Encoding Names Saved**
    - A JSON file of original MHA encoding names mapped to the new SHA encoding names is saved for debugging purposes.
- **ARX to ARN/AR1**
    - Update a models AR. For example go from AR64 to AR8.
- **Optimize o_proj**
    - Remove redundant Transpose/Reshape between sha head concat and o_porj in mha-conv model.
- **ALiBi**
    - Handles ALiBi style position ids.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## 💻 Getting Started <a name="getting-started"></a>

To get started, please take a look at the point below.

### Prerequisites

In order to run this project you need:
- `aimet-dev-torch-gpu` docker
    - If you don't have access to the `aimet-dev-torch-gpu` docker, you can install the dependencies via the [requirements.txt](requirements.txt). Please note by differences for GPU vs CPU support.
- ONNX minimum version >= 1.14.1
- python >= 3.10

### Setup

Clone this repository to your desired folder. You can use normal `git clone` convention.

After cloning, an [env_setup.sh](bin/env_setup.sh) script can be used to set the paths for this tool.

An example of how to run the script is as follows:
```bash
source env_setup.sh
```
> 💡Tip
>
> For more information about the script, run:
> ```bash
> source env_setup.sh -h
> ```

With the paths setup, you can now access the MHA2SHA from anywhere by running [mha2sha-onnx-converter](bin/mha2sha-onnx-converter) from your CLI. More on this below.

### Usage

The main entry point into this tool is the [mha2sha-onnx-converter](bin/mha2sha-onnx-converter). This file takes in args that can be used to adapt the tool as need. The args are listed below.

<details>
<summary><strong>MHA2SHA Args</strong></summary>

| Arg                              | Description                                                                       | Default           |
| :-----------------------------   | :-------------------------------------------------------------------------------  | :---------------: |
| `--base-llm`                     | Will set all above flags based on a LLMs architecture. For example, "llama2" will set `--handle-rope-ops` `--replace-linear-with-conv`, etc|     None          |
| `--build-ar`                     | Build a model with a new AR value. No verification between MHA vs SHA is available|                   |
| `--create-input-lists`           | Creates a Linux/On device input list txt, saves the input data used and goldens. Replace "/" in tensor name to "_" if there's any.  |       False       |
| `--disable-auto-attn-finder`     | Disable auto attention finder in step 5                                           |       False       |
| `--exported-model-encoding-path` | AIMET encoding path                                                               |                   |
| `--exported-model-path`          | Path to model                                                                     |                   |
| `--gqa-model`                    | MHA model has GQA model architechture                                             |       False       |
| `--handle-alibi`                 | Enable handling of ALiBi position ids                                             |       False       |
| `--handle-rope-ops`              | Enable handling of RoPE ops                                                       |       False       |
| `--handle-past-key-value`        | Enable handling of past key/values                                                |       False       |
| `--llm-model`                    | Model is a LLM model                                                              |       False       |
| `--log-level`                    | Sets the log level. Valid values are: 'warn', 'verbose', 'info', 'error', 'fatal' |        info       |
| `--lora-adaptor-list`| Path to list of LORA adaptors.         | None
| `--lora-alpha-from-input`        | Add lora alpha as model input at prequnat model adaption stage                                                  |       False       |
| `--lora-model`                   | MHA model has lora adaptor                                                        |       False       |
| `--mha-conv`                     | MHA model is a conv model                                                         |       False       |
| `--model-name`                   | Model name to auto generate one if not provided                                   |                   |
| `--nchw-aligned`                 | MHA model input to q, k, v proj are aligned to NCHW format                        |       False       |
| `--no-verification`              | Does **not** run extra steps for verification of the model and encoding mappings  |       False       |
| `--optimize-o-proj`              | Graph optimization between sha head concat and o_proj for mha-conv model. (`--mha-conv` option is required)  |       False       |
| `--prepared-model`               | Enable changes for prepared model                                                 |       False       |
| `--position-ids`                 | Path to the positions ids (cos/sin)                                               |       False       |
| `--replace-linear-with-conv`     | Enable linear to conv conversion                                                  |       False       |
| `--sha-export-path`              | Path for export of artifacts                                                      |  ./export_model   |
| `--skip-mha2sha`              | Skip mha2sha conversion, and do pre-quant model adaption only.                                                      |  False   |
| `--seq-len`                      | Sequence length to encode the input string (0 to disable padding)                 |        128        |

</details>


An example to run the project is as follows - please make sure to have run the [env_setup.sh](bin/env_setup.sh) script:

```bash
mha2sha-onnx-converter \
--model-name example \
--exported-model-encoding-path [PATH-TO-ENCODINGS] \
--exported-model-path [PATH-TO-MODEL] \
--base-llm llama2
```

Users can also interact pythonically.

```python
from mha2sha.converter import MHA2SHAConverter

converter = MHA2SHAConverter(
    model_name="example",
    sha_export_path="./exports",
    model_or_path="path/to/model",
    **kwargs  # Where kwargs are flags for the converter to parse
)

model, verification_status = converter.convert()  # verification_status only returns if '--no-verification' is set to False
```

<!-- RUN TESTS -->

## Running Tests <a name="run-tests"></a>

This project has unit tests of small subgraphs of models. For example, we will run 2 head/layer variations of models for sanity checks. We *DO NOT* run full LLM's due to the size. Below is a list of these smaller variations we run today. More will be added as they become available. However, these are **STILL** too large to add to this repository. Users can run the [symlink-models-for-test.sh](test/symlink-models-for-test.sh) to symlink the models to test into the testing directory. Below is a list a few models that are available.

| Model Name             |
| :--------------------- |
| LLaMA 7B - KV Cache    |
| LLaMA 7B - BERT Cache  |


Model are run by consuming a config under the [configs](test/python/configs) directory. Here, the models name, location, etc, are provided and tests are automatically generated. The script will always compare the original models outputs against the newly created one. These test leverage this.

#### pytest Args
Below is a list of arguments that the user can use to change the running of the pytests. By default pytests `-s` flag is always on to show the logs of the tests.

| Arg                  | Description                                                          | Default           |
| :------------------  | :------------------------------------------------------------------- | :---------------: |
| `--fitlered-configs` | Will run only the models given in a comma separated format. Models are named by the basename file in there onnx model file           |      None         |
| `--save-artifacts`   | Saves the artifacts of the run. By default, the results are deleted  |      False        |

To run tests, run the following command:

```bash
pytest test/python
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LINTING -->

## Linting <a name="linter"></a>

For developers who would like to raise a PR to this repo are asked to run [lintrunner](https://github.com/suo/lintrunner) on the files that are updated. This is to make sure all the code follows the same style across different developers.

To run, first make sure that you have installed prerequistes in [requirements.txt](requirements.txt).
Next, run the following if this is your first time running `lintrunner`.
```bash
lintrunner init
```

#### Warning
Make sure to run `lintrunner` in the root of MHA2SHA. `lintrunner` needs access to files here.

```bash
# For checking, simply run
lintrunner <path/to/file>

# To apply the changes to the file
lintrunner -a <path/to/file>
```

If your file is successfully linted, you will see the following:

<img src="imgs/lintrunner_success.png" alt="lint"/>

<!-- MODELS SUPPORTED -->

## 🤖 Models <a name="models"></a>

A subset of models that have been tested/are being worked on, are listed below.

| Model Name                     | Availablity  | Full Run Time[^1]  |            Conversion Time[^2]            |  Onnx Conversion  |
| :----------------------------- | :----------: | :----------------: |  :-------------------------------------:  |  :-------------:  | 
| LLaMA 1B                       |     ✅       |  Not Tested        |                 Not Tested                |      Passed       |
| Stable Diffusion 2.1           |     ✅       |  Not Tested        |                 Not Tested                |      Passed       |
| LLaMA 7B - AR64                |     ✅       |  86 min            |                  38 min                   |
| LLaMA 7B 4K                    |     ✅       |  112 min           |                  40 min                   |
| Stable Diffusion XL - MHA-conv |     ✅       |   35 min           |                  16 min                   |                   |
| LLaMA 7B MHA-conv 2layer[^3]   |     ✅       |  Not Tested        |                Not Tested                 |
| LLaMA3 4K - GQA                |     ✅         |           Not Tested           |                    Not Tested                        |
| LLaMA2-lora                |     ✅         |           Not Tested           |                    Not Tested                        |
| SD 2.1-lora                |     ✅         |           Not Tested           |                    Not Tested                        |
| Llama v2+v3 AR-N to AR-N                |      🛠️   |           Not Tested           |                    Not Tested                        |

For more in depth profiling of this script, please see below on a split of the LLaMA 7B - AR64 found [here](profiling)

Below is a list of models and their priority. If other models are needed to be supported, please talk to the proper channels.

| Model Name                  |                     Priority                               |   Ask Date  |
| :-------------------------- | :--------------------------------------------------------: | :---------: |
| Llama3 4K - GQA/VQ          |        ${{\color{Red}{\textsf{HIGH}}}}\$                   |     5/24    |
| Google Gemini Nano2         |        ${{\color{Limegreen}{\textsf{LOW}}}}\$              |     N/A     |



<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- AUTHORS -->

## 👥 Authors <a name="authors"></a>

👤 **Mu-Chein Hsu**

- GitHub: [@muchhsu](https://github.qualcomm.com/muchhsu)
- Email: [muchhsu@qti.qualcomm.com](muchhsu@qti.qualcomm.com)

👤 **Matthew Ernst**

- GitHub: [@ernst](https://github.qualcomm.com/ernst)
- Email: [ernst@qti.qualcomm.com](ernst@qti.qualcomm.com)

👤 **Tejwinder Singh**

- GitHub: [@tejwinde](https://github.qualcomm.com/tejwinde)
- Email: [tejwinde@qti.qualcomm.com](tejwinde@qti.qualcomm.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 📝 License <a name="license"></a>

```
 Qualcomm Technologies, Inc. Proprietary
 (c) 2024 Qualcomm Technologies, Inc. All rights reserved.

 All data and information contained in or disclosed by this repository are
 confidential and proprietary information of Qualcomm Technologies, Inc., and
 all rights therein are expressly reserved. By accepting this material, the
 recipient agrees that this material and the information contained therein
 are held in confidence and in trust and will not be used, copied, reproduced
 in whole or in part, nor its contents revealed in any manner to others
 without the express written permission of Qualcomm Technologies, Inc.
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[^1]: **Including** FP Verification, encoding sanity checks saving and reloading the model into ONNXRuntime.
[^2]: **Excluding** FP Verification, encoding sanity checks saving and reloading the model into ONNXRuntime.
[^3]: The LLaMA 7B MHA-conv 2layer models are provided by Kanghwan. Refer to models mentioned in: https://github.qualcomm.com/ernst/MHA2SHA/pull/16.
