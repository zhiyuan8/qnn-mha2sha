import onnx
import networkx as nx

model_mha = onnx.load("/nexa/qnn-expr/llama32-compute/qwen3_model/Step-2/host_linux/assets/models/onnx/qwen3.onnx")
model_sha = onnx.load("/nexa/qnn-expr/llama32-compute/qwen3_model/Step-2/host_linux/assets/artifacts/ar1-cl4096/1_of_1/sha_output/ar1-cl4096_1_of_1.onnx")

import onnx
from onnx import helper, numpy_helper
import copy
import itertools

# 1.  -------- helpers --------------------------------------------------------

def collect_nodes_by_prefix(model: onnx.ModelProto, prefix: str):
    """
    Return (nodes, initializers) for the sub-graph whose *node names*
    start with `prefix`.

    * Every node whose **name** starts with `prefix` is selected.
    * Every initializer is kept if either
        • its name is referenced (as an input OR output) by those nodes, OR
        • its own name starts with the same prefix.
    """
    nodes = [n for n in model.graph.node if n.name.startswith(prefix)]

    # tensor names touched by those nodes (inputs *and* outputs)
    referenced = {t for n in nodes for t in itertools.chain(n.input, n.output)}

    inits = [
        t for t in model.graph.initializer
        if t.name in referenced or t.name.startswith(prefix)
    ]
    return nodes, inits

def clone_subgraph(nodes, inits, old_pref: str, new_pref: str):
    """
    Deep-clone `nodes` + `inits`, renaming every occurrence of `old_pref`
    in tensor / node names to `new_pref`.
    Returns (new_nodes, new_inits, name_map).
    """
    name_map = {}

    # ─ clone initializers ────────────────────────────────────────────────────
    new_inits = []
    for t in inits:
        nt = copy.deepcopy(t)
        nt.name = t.name.replace(old_pref, new_pref)
        name_map[t.name] = nt.name
        new_inits.append(nt)

    # ─ clone nodes ───────────────────────────────────────────────────────────
    new_nodes = []
    for n in nodes:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        nn.name = n.name.replace(old_pref, new_pref)
        nn.input[:]  = [name_map.get(x, x.replace(old_pref, new_pref)) for x in n.input]
        nn.output[:] = [name_map.get(x, x.replace(old_pref, new_pref)) for x in n.output]
        new_nodes.append(nn)

    return new_nodes, new_inits, name_map


def insert_after(model: onnx.ModelProto, anchor_name: str, new_nodes, new_inits):
    """
    Physically insert `new_nodes` (with `new_inits`) immediately after the
    node called `anchor_name`.  Redirect every consumer of the anchor’s
    original output so they now read from the *last* node of the cloned block.
    """
    # locate anchor -----------------------------------------------------------
    for idx, n in enumerate(model.graph.node):
        if n.name == anchor_name:
            anchor = n
            break
    else:
        raise ValueError(f"{anchor_name!r} not found")

    old_out = anchor.output[0]

    # wire first / last nodes -------------------------------------------------
    new_nodes[0].input[0] = old_out
    new_out = new_nodes[-1].output[0]

    # insert nodes + initializers --------------------------------------------
    for n in reversed(new_nodes):
        model.graph.node.insert(idx + 1, n)
    model.graph.initializer.extend(new_inits)

    # re-route downstream consumers ------------------------------------------
    for n in model.graph.node:
        if n is anchor or n in new_nodes:
            continue
        n.input[:] = [new_out if x == old_out else x for x in n.input]

# 2.  -------- grab the template sub-graphs -----------------------------------

rms2_nodes, rms2_inits = collect_nodes_by_prefix(model_mha, "/rms_norm_2/")
rms4_nodes, rms4_inits = collect_nodes_by_prefix(model_mha, "/rms_norm_4/")

# 3.  -------- iterate over heads ---------------------------------------------

heads_q = [f"attn_0_head_{i}_query_Conv" for i in range(16)]
heads_k = [f"attn_0_head_{i}_key_Conv"   for i in range(8)]

for h_name in heads_q:
    sub_nodes, sub_inits, _ = clone_subgraph(rms2_nodes, rms2_inits,
                                             "/rms_norm_2/", f"/{h_name}_rms_norm/")
    insert_after(model_sha, h_name, sub_nodes, sub_inits)

for h_name in heads_k:
    sub_nodes, sub_inits, _ = clone_subgraph(rms4_nodes, rms4_inits,
                                             "/rms_norm_4/", f"/{h_name}_rms_norm/")
    insert_after(model_sha, h_name, sub_nodes, sub_inits)

# 4.  -------- save -----------------------------------------------------------

onnx.save_model(model_sha,
                "modified_sha_model.onnx",
                save_as_external_data=True,
                location="modified_sha_model.data")
