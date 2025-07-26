import os
import subprocess
import concurrent.futures
import time
from pathlib import Path
# setup whether using multithread or single thread to compile
go_parallel = True

workfolder = os.getcwd()

QNN_SDK_ROOT = "/opt/qcom/aistack/qairt/2.34.2.250528/"
assert os.path.exists(QNN_SDK_ROOT) == True,"QNN_SDK_ROOT path does not exist"
os.environ['QNN_SDK_ROOT'] = QNN_SDK_ROOT

import sys
sys.path.append(workfolder+'/../../../common/G2G')
sys.path.append(workfolder+'/../../../common/G2G/split_onnx_utils')
sys.path.append(workfolder+'/../../../common/')
from utilities.nsptargets import NspTargets
from utilities.profiler import event_marker

# Set up nsp target specification
nsp_target = NspTargets.Windows.GEN2 

CL = 4096
ARNs = [1]
EXPORT_AR = 2073
EXPORT_CONTEXT_LENGTH = 4096
onnx_name = "qwen3"
# Model splitting is not required to run TinyLlama (w4a16) on NSP, so set num_splits=1
num_splits = 1

splits = range(1, num_splits+1)
arn_list = [ arn for arn in ARNs for i in splits ]
split_idxs = [i for arn in ARNs for i in splits]
print('All task list:', [f"ar{arn}-{n}" for arn,n in zip(arn_list,split_idxs)])

import os

qnn_env = os.environ.copy()
qnn_env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
qnn_env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
qnn_env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + qnn_env["PATH"]
qnn_env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
qnn_env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
qnn_env["LLM"] = "1"
qnn_env["split_embedding"] = "0"
qnn_env["split_lmhead"] = "0"
os.environ = qnn_env


mha2sha_root = workfolder+"/../../../common/G2G/MHA2SHA"
g2g_env = os.environ.copy()
g2g_env["PYTHONPATH"] = os.pathsep.join([g2g_env.get("PYTHONPATH", ""), os.path.join(mha2sha_root, "src/python")])
g2g_env["PATH"] = os.pathsep.join([g2g_env.get("PATH", ""), os.path.join(mha2sha_root, "bin")])
print(f"MHA2SHA tool root set to: {mha2sha_root}")

def thread_g2g(arn, split):
    import sys
    sys.path.append(os.path.join(workfolder, "../../../common/G2G"))  # Add G2G root
    sys.path.append(os.path.join(workfolder, "../../../common/G2G/MHA2SHA/src/python"))  # Add MHA2SHA python src
    from mha2sha.converter import MHA2SHAConverter

    model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
    split_work_dir = os.path.join(model_artifact, f"{split}_of_{num_splits}")
    name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
    os.makedirs(split_work_dir, exist_ok=True)
    sha_folder = f"sha_output/"
    os.makedirs(sha_folder, exist_ok=True)
    name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
    print("loading model from ", name)
    flags = {
        "exported_model_encoding_path": f"{model_artifact}/src/onnx/{onnx_name}.encodings",
        "base_llm": "llama3",
        "mha_conv": True,
        "nchw_aligned": True,
        "log_level": "debug",
        "handle_internal_rmsnorm": True
    }

    print(f"MHA2SHAConverter {name} running...")
    converter = MHA2SHAConverter(
        model_name=name,
        sha_export_path=sha_folder,
        model_or_path=f"{model_artifact}/split_onnx/{name}.onnx",
        **flags
    )
    sha_model, verification_status = converter.convert()
    print(f"MHA2SHAConverter {name} done. Verification status: {verification_status}")

# with event_marker(f'mha2sha'):
#     with concurrent.futures.ProcessPoolExecutor(max_workers = len(arn_list) if go_parallel else 1) as executor:
#         results = executor.map(thread_g2g, arn_list,split_idxs)
# print(f"All mha2sha convert done.")

for arn, split in zip(arn_list, split_idxs):
    thread_g2g(arn, split)