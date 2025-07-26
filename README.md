# qnn-mha2sha
This repo contains `MHA2SHA` for:
- Orignial MHA2SHA for `Qwen2`
- Modified MHA2SHA for `Qwen3` with RMS Norm for `q-proj` and `k-proj`
- Modified MHA2SHA for `Qwen3` with RMS Norm for `q-proj` and `k-proj`, and RMSNorm Cast op removed, and `rmsnorm_update` changed.
- Modified MHA2SHA for `Qwen3` with RMS Norm for `q-proj`, `k-proj` and `v-proj`, in preperation for Gemma3n model
- Modified MHA2SHA for `Clip` text encoder
- Modified MHA2SHA for `Clip` vision encoder

# Validation
For each MHA2SHA, keep in mind that we need to validate:
1. onnx validation error is within 1e-4, between SHA and MHA in full precision
2. SHA onnx graph is correct
3. MHA2SHA encoding mapping file is correct
4. SHA encoding file is correct, weight and activation encodings match the MHA encoding info.

# File structure
each folder contains qnn step1 and step2 for AIMET.

# Documentation
- [MHA2SHA for Qwen3](https://oc1rr3jgj3d.sg.larksuite.com/wiki/MthAwx2jLivWT8kUMC5leLkYgPg?from=from_copylink)
- [MHA2SHA Notes](https://oc1rr3jgj3d.sg.larksuite.com/docx/UJ3rdD5JNoHVCyxup9KlHTEUgu8)