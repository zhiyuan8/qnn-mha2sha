# Demo Notebook: LLaMA Model Inference on Snapdragon Windows Devices

## Platform requirements
This notebook is intended to run on a machine with:
  * Windows 11
  * QNN SDK 2.28.0
  * NSP Device

## Environment setup (venv)
    * install python >= 3.10
    * install virtualenv
    * create an environment (python -m virtualenv llama_env)
    * activate environment (.\llama_env\bin\activate.ps1)

## Pre-requisites
Prior to using this notebook you will need the following artifacts:
* QNN prepared models (model.serialized.bin from Step-2)

Paths to where these artifacts are stored will need to be specified in the notebook by assigning to
corresponding variables

## Launch Jupyter Notebook
To install additional dependencies and launch the jupyter notebook server run the launch script:
```powershell
.\launch_nb.ps1
```

Once the server starts, you will be presented a URL(s) that you may copy and paste into your web browser.

From the web browser, click on the Jupyter Notebook `qnn_model_execution.ipynb` and follow the instructions therein.

*NOTE*: Make sure to update the `QNN_SDK` and `WEIGHT_SHARE_CONTEXT_BINARY_PATH`inside of Example #3 Notebook.
