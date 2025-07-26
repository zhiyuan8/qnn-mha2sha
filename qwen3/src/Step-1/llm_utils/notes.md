# `llm_utils/` folder

| File Name                  | Description                                                                                                                      |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| mixed_precision_overrides.py | Contains utilities or logic to override or customize mixed-precision (e.g., float16/float32) behavior for model operations, likely to optimize performance or memory usage during inference or training. |
| qcllama_adaptation.py        | Implements adaptation logic for QCLlama models, possibly including model conversion, quantization, or compatibility layers to work with specific hardware or frameworks. |
| test_vectors.py              | Provides test vectors (input/output pairs) and utilities for validating model correctness, quantization, or other transformations. Useful for regression and unit testing. |
| wikitext_dataloader.py       | Implements a data loader for the WikiText dataset, handling data preprocessing, batching, and feeding data into the model for training or evaluation. |
| forward_pass_wrapper.py      | Contains a wrapper for the model's forward pass, possibly adding hooks, instrumentation, or custom logic for running inference or collecting statistics. |
| notes.md                     | Markdown file for developer notes, documentation, or explanations related to the utilities in this folder.                      |

# `forward_pass_wrapper.py`
Q1: why flatten_tensors?


`flatten_tensors` is a recursive generator function that takes a possibly deeply nested structure (tuple/list of tensors, or just a tensor) and yields all tensors in a flat sequence.  

---

Q2: why get zero padded_kv_values, padd to left or right side?


`get_padded_kv_values` creates zero-initialized key and value tensors for the transformer’s key-value cache, used for attention.
- **Why:**  
  - When running inference with a kv-cache, you may need to pad the cache to a fixed size (e.g., for batching or to match model expectations).
- **Pad to left or right?**  
  - check `prepare_inputs`, 


---

Q3: what does decoder attention mask shape look like? why choose -50.0? What does _expand_mask do? what does prepare_combined_attention_mask do?


- **Why -50.0?**  
  - -50.0 is used as a large negative value to mask out positions in the attention matrix.
  - In softmax, large negative values (e.g., -50 or -1e9) make the probability close to zero, effectively masking those positions.
  - -50.0 is a compromise: large enough to mask, but not so large as to cause numerical instability.
- **What does _expand_mask do?**  
  - `_expand_mask` takes a `[batch, seq_len]` mask and expands it to `[batch, 1, tgt_seq_len, src_seq_len]`.
  - It inverts the mask (1 becomes 0, 0 becomes 1), then fills masked positions with `mask_neg` (e.g., -50.0).
  - This is used to mask out padding tokens in the attention computation.
- **What does prepare_combined_attention_mask do?**  
  - It combines the causal mask (prevents attending to future tokens) and the padding mask (prevents attending to padding tokens).
  - The result is a mask that enforces both autoregressive (causal) and padding constraints for the decoder.

---

Q4: explain me details and steps of prepare_inputs, what are the input variables and their shapes?


`prepare_inputs` prepares all the necessary inputs for a model forward pass, including padding, kv-cache, and attention masks.

**Inputs:**
- `input_ids` (Tensor): `[batch, seq_len]`
- `attention_mask` (Tensor): `[batch, seq_len + kv_len]`
- `position_embeddings_cos` (Tensor): `[batch, 1, seq_len, head_dim//2]`
- `position_embeddings_sin` (Tensor): `[batch, 1, seq_len, head_dim//2]`
- `past_key_values`: kv-cache, tuple of tuples of tensors

**Steps:**
1. **Validation:**  
   - Ensures only one of `input_text`, `input_ids`, or `input_embeddings` is provided.
   - Checks kv-cache and embedding converter validity.

2. **Tokenization (if input_text):**  
   - Tokenizes text to get `input_ids` and `attention_mask`.

3. **Input Preparation:**  
   - If using embeddings, gets or computes `input_embeddings` and ensures correct dtype/device.
   - Otherwise, uses `input_ids` and ensures correct dtype/device.
   - Shape after this:  
     - `input`: `[batch, seq_len]` or `[batch, seq_len, embed_dim]`

4. **Batch/Length Calculation:**  
   - Gets `batch_size` and `input_length` from input.

5. **KV-Cache Length:**  
   - Gets `kv_length` from `past_key_values` if present.

6. **Attention Mask:**  
   - If not provided, creates a mask of ones for `[batch, input_length + kv_length]`.
   - Ensures mask is tensor, correct dtype/device.

7. **Input/Mask Length Validation:**  
   - Checks that input, mask, and attention lengths are consistent.

8. **Padding Inputs:**  
   - If `input_length < num_tokens`, pads input on the left with EOS token (or zeros for embeddings) to `[batch, num_tokens]`.

9. **Padding Attention Mask:**  
   - Pads attention mask to `[batch, max_tokens]` (left for kv-cache, right for input padding).

10. **KV-Cache Padding:**  
    - Pads kv-cache to desired length (`max_tokens - num_tokens`) with zeros.

11. **Final Attention Mask Padding:**  
    - Further pads attention mask for kv-cache padding.

12. **Final Validation:**  
    - Checks that input, attention mask, and kv-cache have correct shapes.

13. **Position IDs:**  
    - Computes position ids from attention mask.
    - If using position embedding input, gets RoPE embeddings.

14. **Combined Mask:**  
    - If using combined mask input, prepares combined attention mask.

15. **Input Dict Construction:**  
    - Builds input dict for the model, including all required fields (input_ids/inputs_embeds, attention_mask, position_ids, kv-cache, etc.).

**Output:**  
- Returns `(inputs_dict, kvcache_info_bundle)`.

---

Q5: explain me details of `slice_inputs_and_run_successive_kvcache_inference`?

**A5:**  
This function enables inference on long sequences by slicing them into manageable chunks and running them sequentially, updating the kv-cache at each step.

**How it works:**
- **Inputs:**
  - `fpm`: The forward pass manager (LLMForwardPassManager).
  - `input_ids` or `input_embeds`: The full input sequence (tokens or embeddings).
  - `**kwargs`: Additional arguments (e.g., attention_mask).

- **Steps:**
  1. **Determine Input Length:**  
     - Gets the total sequence length from `input_ids` or `input_embeds`.

  2. **Chunking:**  
     - Iterates over the input in reverse, in steps of `fpm.num_tokens` (the model’s chunk size).
     - For each chunk, slices the input to the current window.

  3. **Attention Mask Handling:**  
     - If an attention mask is provided, slices it to match the current chunk.

  4. **Forward Pass:**  
     - Calls the forward pass manager (`fpm`) on the current chunk, passing the current kv-cache (from previous step).
     - Gets logits and updated kv-cache.

  5. **Output Accumulation:**  
     - Concatenates logits from each chunk to build the full output.
     - Updates `past_key_values` in `kwargs` for the next chunk.

- **Output:**  
  - Returns a dict with concatenated logits and the final kv-cache.

**Use case:**  
- This is useful for models with a limited context window (e.g., 2048 tokens), allowing you to process longer sequences by feeding them in chunks and carrying forward the kv-cache.
