#!/usr/bin/env python3
# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2023 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================

# A set of utilities to split onnx files and generate various artifacts

import inspect
import sys
import os
import re
import itertools
import json
import onnx
import math
import numpy as np
import struct
import torch
import pickle
import collections
import shutil
import pathlib

from typing import List

from datetime import datetime
from difflib import get_close_matches
from onnx.numpy_helper import to_array, from_array

from split_onnx import OnnxSplitter, save_model


def _test_vector_pickle_file(source, modeldir='.'):
    return os.path.join(modeldir,  source)

def _input_dir(name, source='qt'):
    return f'test_inputs_{name}' if source=='qt' else f'fp_test_inputs_{name}'

def _input_list_file(name, source='qt'):
    return f'input_list_{name}.txt' if source=='qt' else f'fp_input_list_{name}.txt'

def _golden_output_dir(name, source='qt'):
    return f'test_golden_outputs_{name}' if source=='qt' else f'fp_test_golden_outputs_{name}'

def _per_layer_output_dir(name, source='qt'):
    return f'test_per_layer_outputs_{name}' if source=='qt' else f'fp_test_per_layer_outputs_{name}'

def sqnr(org_out, quant_out, eps=1e-20):
    # for single data
    if org_out.shape != quant_out.shape:
        raise RuntimeError(f"Shapes mismatch, {org_out.shape} != {quant_out.shape}, Invalid SQNR measurement")
    quant_error = org_out - quant_out
    exp_noise = (quant_error ** 2).mean() + eps
    exp_signal = (org_out ** 2).mean()
    sqnr = (exp_signal / exp_noise) + eps
    sqnr_db = 10 * np.log10(sqnr)
    return sqnr_db

def mse(signal, noisy):
    return np.power(noisy - signal, 2).mean()

def cosine_similarity(signal, noisy):
    signal, noisy = signal.reshape(-1), noisy.reshape(-1)
    return np.dot(signal, noisy)/(np.linalg.norm(signal) * np.linalg.norm(noisy))

def _mkdir(dirs, clean=False):
    if clean and os.path.exists(dirs):
        shutil.rmtree(dirs, ignore_errors=True)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def _np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def _dtype_suffix(t):
    return str(t.dtype)

def _dump(tensor, filename):
    if isinstance(filename, (tuple, list)):
        assert len(filename) == len(tensor), f"len({len(filename)} != len(tensor):{len(tensor)}), filename:{filename}"
        for atensor, aname in zip(tensor, filename):
            _dump(atensor, aname)
        return

    path = os.path.dirname(filename)
    if len(path)>0:
        _mkdir(path)
    nptensor =  _np(tensor)
    if nptensor.dtype != np.float32:
        split = os.path.basename(filename).split('.')
        assert len(split)>1, f"No extension in '{filename}'?"
        split.insert(-1, _dtype_suffix(nptensor))
        filename_suffix = os.path.join(path, '.'.join(split))
        nptensor.tofile(filename_suffix)
    nptensor.astype(np.float32).tofile(filename)

def _target_name(name, deco_digit=True, using_qairt_workflow=False):
    name = f'_{name}' if deco_digit and name.isdigit() else name
    # name = name.replace('.', '_')
    if not using_qairt_workflow:
        name = name.replace('/', '-')
    return name

def _load_pickle(pklfile):
    with open(pklfile, 'rb') as f:
        return pickle.load(f)

def size_of_pickles(d):
    def _size(d):
        if isinstance(d, np.ndarray):
            return math.prod(d.shape) * d.itemsize
        if isinstance(d, torch.Tensor):
            return math.prod(d.shape) * d.element_size()
        if isinstance(d, (list, tuple)):
            return sum(_size(i) for i in d)
        if isinstance(d, dict):
            return sum(_size(v) for _,v in d.items())
        if isinstance(d, str):
            return len(d)
        assert False, f'Unexpected type({type(d)})'
    return _size(d)


class Metric:
    def __init__(self, a, b):
        self.sqnr, self.mse, self.cossim = self.calc(a, b)

    def calc(self, a, b):
        if isinstance(a, (list,tuple)):
            return tuple(np.mean([self.calc(ai, bi) for ai, bi in zip(a, b)], axis=0))
        if isinstance(a, torch.Tensor) and not torch.is_floating_point(a):
            a, b = a.to(torch.float), b.to(torch.float)
        return (float(sqnr(a,b)), float(mse(a, b)), float(cosine_similarity(a,b)))

    def __repr__(self):
        return f'{self.sqnr:.1f}dB|{self.mse:.5f}|{self.cossim:.3f}'


class TestVectorBase:
    def __init__(self, filename):
        self.filename = filename
        self.num_test_vectors = 0

    @staticmethod
    def load(picklefile, onnxfile, max_count=1000, using_qairt_workflow=False):
        max_count=int(os.getenv('MAX_SAMPLES', max_count))
        return TracerTestVector(picklefile, onnxfile, max_count, using_qairt_workflow=using_qairt_workflow)

    @property
    def count(self):
        return self.num_test_vectors

    def get(self, name, i):
        raise RuntimeError("get(): Not implemented")

    def has(self, name):
        raise RuntimeError("has(): Not implemented")

    def get_batches(self, list_of_names: List[List[str]]) -> List[List[np.array]]:
        raise RuntimeError("get_batches(): Not implemented")

    def add(self, name, tensor):
        if len(tensor.shape) == 1:
            tensor = _np(tensor).reshape(1, -1)
        if name not in self.data:
            self.data[name] = _np(tensor)
        else:
            self.data[name] = np.concatenate([self.data[name], tensor])

    def prepare_dummy_inputs(self, onnxmodel, missing_tensors):
        np.random.seed(0)
        missing = {i.name:i for i in onnxmodel.graph.value_info if i.name in missing_tensors}
        assert len(missing) == len(missing_tensors), f"Missing value_info: {[i for i in missing_tensors if i not in missing]}"
        dummy_inputs = {}
        for name, info in missing.items():
            assert info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "Just float :)"
            shape = [i.dim_value for i in info.type.tensor_type.shape.dim]
            shape[0] = self.num_test_vectors
            dummy = np.random.randn(*shape) - 0.5   # make it [-0.5, 0.5)
            dummy_inputs[name] = dummy
        return dummy_inputs


class WordTrie():
    def __init__(self):
        self.children = {}
        self.name = None
        self.end = False

    @staticmethod
    def from_strings(strings, sep='_/'):
        root = WordTrie()
        for string in strings:
            words = re.split(f'[{sep}]', string)
            words = [word for word in words if word != '']
            root._add(string, words, 0)
        return root

    def match(self, string, sep='_/'):
        words = re.split(f'[{sep}]', string)
        return self._get(words, 0)

    def _add(self, fullname, words, offset):
        if offset == len(words):
            self.end = True
            self.name = fullname
        else:
            if words[offset] not in self.children:
                self.children[words[offset]] = WordTrie()
            self.children[words[offset]]._add(fullname, words, offset+1)

    def _get(self, words, offset):
        if offset == len(words):
            return self.name if self.end else None
        if words[offset] in self.children:
            return self.children[words[offset]]._get(words, offset+1)
        return None


class TracerTestVector(TestVectorBase):
    '''
    # fp_0.pkl
    {
        "0": {
            "input_ids": "([2048], torch.int64)",
            "attention_mask": "([2048], torch.int64)",
            "position_ids": "([2048], torch.int64)",
            "logits": "([1, 50257], torch.float32)",
            "output_key_values": [
                [
                "([20, 2048, 128], torch.float32)",
                "([20, 2048, 128], torch.float32)"
                ],
                ...,
            ],
            "past_key_values": [
                [
                    "([20, 2047, 128], torch.float32)",
                    "([20, 2047, 128], torch.float32)"
                ],
                ....,
            ],
            "transformer.h.0.ln_1": { "input": "([1, 768], torch.float32)" },
            "transformer.h.0.ln_2": { "input": "([1, 768], torch.float32)" },
            ...,
            "transformer.ln_f": { "input": "([1, 768], torch.float32)" }
        },
    }
    #fp_1.pkl
    {
        "1": {...},
    }
    '''
    def __init__(self, picklefile, onnxfile, max_count, using_qairt_workflow=False):
        super().__init__(picklefile)
        self.onnxfile = onnxfile

        def _name(node):
            if node.name.endswith(f'/{node.op_type}'):
                return node.name[:-(len(node.op_type)+1)]
            return node.name

        self.onnxmodel = _load_model(onnxfile)
        # rstrip because torch.1.13 appends op_type to its name, which causes get_close_matches fail
        self.onnxnodes = {_name(node):node for node in self.onnxmodel.graph.node}
        self.wordtrie = WordTrie.from_strings(list(self.onnxnodes.keys()))
        self.producer = {output:node for node in self.onnxmodel.graph.node for output in node.output}
        self.consumers = collections.defaultdict(list)
        for node in self.onnxmodel.graph.node:
            for inputtensor in node.input:
                self.consumers[inputtensor].append(node)

        self.test_vector_name_to_node_name = {}
        self.intermediate_tensor_mapping = {} # (tensor name, test vector name)

        self.input_names, self.output_names = get_onnx_input_output_names(onnxfile, onnxmodel=self.onnxmodel, deco_digit=False, using_qairt_workflow=using_qairt_workflow)

        self.max_count = max_count
        gen_file_names = (f'{picklefile}_{i}.pkl' for i in range(1))
        self.picklefiles = list(itertools.takewhile(lambda name: os.path.exists(name), gen_file_names))
        self.num_test_vectors = len(self.picklefiles)
        self.has_intermediate_tensor_dumps = True

        # walk through 1st batch to get intermediate tensor dump names
        batch = _load_pickle(self.picklefiles[0])['0']
        dict(self._flatten_and_rename(batch)) # to get name and shapes

    def get_batches(self, list_of_names: List[List[str]]) -> List[List[np.array]]:
        for batch_index, picklefile in enumerate(self.picklefiles):
            if batch_index == self.max_count: break
            batch_name = f'{batch_index}'

            data = _load_pickle(picklefile)
            assert batch_name in data, f'key:"{batch_name}" not found in {picklefile}'
            batch = data[batch_name]

            loaded = dict(self._flatten_and_rename(batch))

            nickname = {}
            for k,v in loaded.items():
                if not k.startswith('model'): continue
                k = k.replace('model.', '/model/') if k.startswith('model.') else k.replace('model', '/model')
                k = k.replace('.input_layernorm', '/input_layernorm')
                k = k.replace('.post_attention_layernorm', '/post_attention_layernorm')
                k = k.replace('.cast', '/cast/Cast') if '.cast' in k else k.replace('Cast', 'Cast/Cast')
                nickname[k] = v
            loaded.update(nickname)

            not_found =  [name for names in list_of_names for name in names if name not in loaded]
            if not_found:
                dummy = self.prepare_dummy_inputs(self.onnxmodel, not_found)
                print("#"*80)
                print(f"# Adding dummy test vector for {not_found}!!!")
                print("#"*80)
                loaded.update(dummy)

            # all the names we were looking for were in the pkl file
            assert all(name in loaded for names in list_of_names for name in names), \
                f'Not found: {[name for names in list_of_names for name in names if name not in loaded]}'
            yield [[loaded[name] for name in names] for names in list_of_names]

    def _get_bestfit_node_name(self, test_vector_name):
        # Try fast rule first
        bestfit = self.wordtrie.match(test_vector_name)
        if bestfit:
            return bestfit

        bestfit = get_close_matches(test_vector_name, self.onnxnodes.keys(), n=1, cutoff=0.7)
        if bestfit:
            bestfit = bestfit[0]
            return bestfit
        return None

    def _flatten_and_rename(self, batch):
        def _add_mapping(name, torch_name, tensor):
            self.intermediate_tensor_mapping[name] = {'name':torch_name, 'shape':f"{list(tensor.shape)}"}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                yield (key, _np(value))
            elif key == 'inputs':
                yield (key, value)
            elif key == 'per_layer_inputs':
                # per-layer inputs, injectected for testing purpose in the AIMET pipeline
                continue
            elif key == 'position_ids' and isinstance(value, (tuple, list)):
                # Handle RoPE embedding, untuple and rename
                yield from [(f'position_ids_cos', value[0]), (f'position_ids_sin', value[1])]
            elif key == 'past_key_values':
                assert isinstance(value, (tuple, list)), f'Unexpected past_key_values type: {type(value)}'
                if isinstance(value[0][0], (tuple, list)):
                    num_layers, num_heads = len(value), len(value[0][0])
                    for layer in range(num_layers):
                        yield from [(f'past_key_{layer}_h{head}_in', v) for head, v in enumerate(value[layer][0])]
                        yield from [(f'past_value_{layer}_h{head}_in', v) for head, v in enumerate(value[layer][1])]
                else:
                    yield from [(f'past_key_{i}_in', v[0]) for i, v in enumerate(value)]
                    yield from [(f'past_value_{i}_in', v[1]) for i, v in enumerate(value)]
            elif key == 'output_key_values' or key == 'output_values':
                assert isinstance(value, (tuple, list)), f'Unexpected past_key_values type: {type(value)}'
                if isinstance(value[0][0][0], (tuple, list)):  # for some exports I made this mistake :(
                    value = value[0]
                if isinstance(value[0][0], (tuple, list)):
                    num_layers, num_heads = len(value), len(value[0][0])
                    for layer in range(num_layers):
                        yield from [(f'past_key_{layer}_h{head}_out', v) for head, v in enumerate(value[layer][0])]
                        yield from [(f'past_value_{layer}_h{head}_out', v) for head, v in enumerate(value[layer][1])]
                else:
                    yield from [(f'past_key_{i}_out', v[0]) for i, v in enumerate(value)]
                    yield from [(f'past_value_{i}_out', v[1]) for i, v in enumerate(value)]
            elif isinstance(value, dict):
                # intermediate tensor dumps
                if key not in self.test_vector_name_to_node_name:
                    bestfit = self._get_bestfit_node_name(key)
                    if bestfit:
                        print(f"Mapping test vector '{key}' to '{self.onnxnodes[bestfit].name}'")
                        self.test_vector_name_to_node_name[key] = bestfit
                if key in self.test_vector_name_to_node_name:
                    node = self.onnxnodes[self.test_vector_name_to_node_name[key]]
                    if 'input' in value: # assuming input[0]
                        if node.op_type == 'Cast':
                            # Becasue QNN converter fuses 'Cast' into its input node,
                            #  we don't see input names of 'Cast' in the graph
                            _add_mapping(node.input[0], f'{key}.input', value['input'])
                            yield (node.input[0], value['input'])
                            _add_mapping(node.output[0], f'{key}.input', value['input'])
                            yield (node.output[0], value['input'])
                        else:
                            _add_mapping(node.input[0], f'{key}.input', value['input'])
                            yield (node.input[0], value['input'])
                        # For backward?
                        #yield (key, value['input'])
                    if 'output' in value: # assuming output[0]
                        # duplicate some tensors
                        consumers = self.consumers[node.output[0]]
                        if len(consumers) == 1 and consumers[0].op_type == 'Cast':
                            _add_mapping(consumers[0].output[0], f'{key}.output', value['output'])
                            yield (f'{consumers[0].output[0]}', value['output'])
                        elif node.op_type == 'Concat' and len(consumers) == 1 and consumers[0].op_type == 'Transpose':
                            _add_mapping(f'{consumers[0].output[0]}.nhwc', f'{key}.output', value['output'])
                            yield (f'{consumers[0].output[0]}.nhwc', value['output'])
                        else:
                            _add_mapping(node.output[0], f'{key}.output', value['output'])
                            yield (node.output[0], value['output'])

                else:
                    assert 'input' in value and len(value) == 1, f"Unexpected intermediate tensor dump:{key}"
                    _add_mapping(key, f'{key}.input', value['output'])
                    yield (key, value['input'])
            else:
                assert False, f"Unexpected key:{key}, value:{type(value)}"


def save_per_layer_mapping(per_layer_dir, mapping, using_qairt_workflow=False):
    with open(os.path.join(per_layer_dir, 'per_layer_mapping.json'), 'wt') as f:
        f.write(json.dumps(mapping, indent=2))
    with open(os.path.join(per_layer_dir, 'per_layer_mapping.sh'), 'wt') as f:
        f.write('#!/bin/bash')
        f.write(f'per_layer_tensor_names=( {" ".join(mapping.keys())} )')
        f.write('declare -A per_layer_mapping')
        for tensor, node in mapping.items():
            filename = _target_name(tensor, using_qairt_workflow=using_qairt_workflow)
            f.write(f'per_layer_mapping["{filename}"]="{node["name"]}"')

def prepare_test_vector_by_source(modelname, onnxfile, modeldir='.', source='qt', using_qairt_workflow=False):
    input_names, output_names = get_onnx_input_output_names(onnxfile, using_qairt_workflow=using_qairt_workflow)
    picklefile = _test_vector_pickle_file(source, modeldir)
    tv = TestVectorBase.load(picklefile, onnxfile, using_qairt_workflow=using_qairt_workflow)

    input_dir = _input_dir(modelname, source)
    input_list_file = _input_list_file(modelname, source)
    per_layer_dir = _per_layer_output_dir(modelname, source)
    output_dir = _golden_output_dir(modelname, source)
    _mkdir(os.path.join(modeldir,input_dir))
    _mkdir(os.path.join(modeldir,output_dir))
    if tv.has_intermediate_tensor_dumps:
        _mkdir(os.path.join(modeldir,per_layer_dir))

    tensor_names = [input_names, output_names]
    if tv.has_intermediate_tensor_dumps:
        intermediate_tensor_names = list(tv.intermediate_tensor_mapping.keys())
        tensor_names.append(intermediate_tensor_names)
        save_per_layer_mapping(os.path.join(modeldir, per_layer_dir), tv.intermediate_tensor_mapping, using_qairt_workflow=using_qairt_workflow)

    print(f'Save as {input_list_file}')
    with open(os.path.join(modeldir, f'{input_list_file}'), 'wt') as input_list:
        for batch, tensors in enumerate(tv.get_batches(tensor_names)):
            input_tensors = tensors[0]
            input_dump_names = [f'{modeldir}/{input_dir}/{batch}/{name}.raw' for name in input_names]
            _dump(input_tensors, input_dump_names)
            input_list_names = [f'{name}:={input_dir}/{batch}/{name}.raw' for name in input_names]
            input_list.write(' '.join(input_list_names))

            if len(tensors) > 1:
                output_tensors = tensors[1]
                output_dump_names = [f'{modeldir}/{output_dir}/Result_{batch}/{_target_name(name, using_qairt_workflow=using_qairt_workflow)}.raw' for name in output_names]
                _dump(output_tensors, output_dump_names)

            if tv.has_intermediate_tensor_dumps:
                intermediate_tensors = tensors[2]
                full_path_names = [f'{modeldir}/{per_layer_dir}/{batch}/{_target_name(name, using_qairt_workflow=using_qairt_workflow)}.raw' for name in intermediate_tensor_names]
                _dump(intermediate_tensors, full_path_names)


def prepare_test_vector(modelname, onnxfile, modeldir='.', using_qairt_workflow=False):
    prepare_test_vector_by_source(modelname, onnxfile, modeldir=modeldir, source='qt', using_qairt_workflow=using_qairt_workflow)
    prepare_test_vector_by_source(modelname, onnxfile, modeldir=modeldir, source='fp', using_qairt_workflow=using_qairt_workflow)


def prepare_test_vector_fp(modelname, onnxfile, modeldir='.', using_qairt_workflow=False):
    prepare_test_vector_by_source(modelname, onnxfile, modeldir=modeldir, source='fp', using_qairt_workflow=using_qairt_workflow)


def get_onnx_input_output_names(onnxfile, onnxmodel=None, deco_digit=True, using_qairt_workflow=False):
    onnxmodel = _load_model(onnxfile) if onnxmodel is None else onnxmodel
    input_names = [_target_name(i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow) for i in onnxmodel.graph.input]
    output_names = [_target_name(i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow) for i in onnxmodel.graph.output]
    return input_names, output_names

def get_split_tensors(onnxfile, onnxmodel=None, include_first_input=True):
    '''
    Model topology
            │ ←─────────  layers[0]  ────────────→ │       │ ←─────────  layers[-1]  ─────────────→ │
            │                                      │       │                                        │
    embed ────┬──────────── add0 ─┬─────────── add1 ── ┄┄┄  ─┬─────────────── add ─┬───────────── add ─── lmhead
            ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑       ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑
            │                                        │       │                                        │
            │                                        │       │                                        │
            valid splitting points
    '''
    def can_visit(src, dst):
        if seq[src] < seq[dst]: return False
        stack, visited = collections.deque([src]), set()
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            visited.add(cur)
            next_nodes = [producers[tensor] for tensor in nodes[cur].input if producers[tensor] is not None]
            for name in next_nodes:
                if name not in visited and seq[name] >= seq[dst]:
                    stack.append(name)
        return False

    def is_residual_add(nodename, strict):
        if nodes[nodename].op_type != 'Add': return False
        a, b = [producers[tensor] for tensor in nodes[nodename].input]
        if a is None or b is None: return False
        begin, end = (a, b) if seq[a] < seq[b] else (b, a)
        if strict and (begin == 'graph_input' or nodes[begin].op_type != 'Add'):
            return False
        return can_visit(end, begin)

    def get_add0(add1):
        a, b = [producers[tensor] for tensor in nodes[add1].input]
        add0 = a if seq[a] < seq[b] else b
        assert is_residual_add(add0, strict=False)
        return add0

    def get_layer0_input(add0):
        a, b = [producers[tensor] for tensor in nodes[add0].input]
        return a if seq[a] < seq[b] else b

    def get_nodes():
        model = _load_model(onnxfile) if onnxmodel is None else onnxmodel
        nodes = {i.name: i for i in model.graph.node}
        seq = {i.name: idx for idx, i in enumerate(model.graph.node)}
        producers = collections.defaultdict(lambda: None)
        producers.update({i.output[0]: i.name for i in model.graph.node})
        producers.update({i.name: 'graph_input' for i in model.graph.input})
        seq['graph_input'] = -1
        return nodes, seq, producers
    nodes, seq, producers = get_nodes()

    residual_add_names = [name for name in nodes.keys() if is_residual_add(name, strict=True)]
    if len(residual_add_names) % 2 == 1:
        # 'add0' is missing in residual_adds
        add0 = get_add0(residual_add_names[0])
        residual_add_names.insert(0, add0)

    output_tensors = []
    if include_first_input:
        layer0_input = get_layer0_input(residual_add_names[0])
        output_tensors.append(nodes[layer0_input].output[0])
    output_tensors += [nodes[node].output[0] for i, node in enumerate(residual_add_names) if i%2==1]

    return output_tensors


def concatenate_signals(signals_per_head, signal_type, output_names):
    # signal_type = key  or  value
    # concatenate the key/value signals from the same layer

    # print("signal_type = ", signal_type)

    # layer_signals = [num_layers][signals]
    # [0] - [k00, k01, ... k031]
    # ...

    layer_signals = {}  # layer number -> [] signals for that layer number
    for i in range(len(output_names)):
        if not f"{signal_type}_" in output_names[i]: # if it's not the correct signal type skip
            continue
        split_name = output_names[i].split("_")
        layer_number = int(split_name[split_name.index(signal_type)+1])  # figure out the layer number
        if layer_number in layer_signals: # if there's already a list for this layer
            layer_signals[layer_number].append(signals_per_head[i])   # append to it
        else :   # otherwise make a new list
            layer_signals[layer_number]= [signals_per_head[i]]

    concat_signal = [np.concatenate(ls) for ls in layer_signals.values()]

    return concat_signal


def check_output(modelname, onnxfile, output_dir, csvfilename=None, source='qt', using_qairt_workflow=False):
    if csvfilename is None:
        now = datetime.now()
        csvfilename=f'output-{now.strftime("%m%d%Y-%H%M%S")}.csv'
    print(csvfilename)

    _, output_names = get_onnx_input_output_names(onnxfile, deco_digit=True, using_qairt_workflow=using_qairt_workflow)

    # print("output names = ", output_names)
    # print("len output names = ", len(output_names))

    layer_names = []
    for name in output_names :
        if "key_" in name or "value_" in name:
            name_split = name.split("_")

            if "key_" in name :
                layer_name = "_".join(name_split[name_split.index("key"): name_split.index("key") + 2])
            elif "value_" in name :
                layer_name = "_".join(name_split[name_split.index("value"): name_split.index("value") + 2])

            if "past_" + layer_name not in layer_names:
                layer_names.append("past_" + layer_name)
        else :
            layer_names.append(name)

    print("layer names = ", layer_names)

    golden_dir = _golden_output_dir(modelname, source)

    with open(csvfilename, 'wt') as csvfile:

        csvfile.write(','.join([f'#'] + [f'{i},,' for i in layer_names]) + '\n')
        csvfile.write(f','.join(['#'] + ['SQNR,MSE,CosSim' for _ in layer_names]) + '\n')
        print(f"{' ':>5} || " + ' || '.join(f'{i:29}' for i in layer_names))
        head = f"{'SQNR':>7} | {'MSE':9} | {'CosSim':7}"
        print(f"{'#':>5} || " + ' || '.join(f'{head:29}' for _ in layer_names))
        for idx in range(1000000):
            if not os.path.exists(f'{output_dir}/Result_{idx}/{output_names[0]}.raw'):
                break

            # collect all the signals from each of the heads
            nan = float('nan')
            signals_per_head = []
            for i in range(len(output_names)):
                fname = f'{golden_dir}/Result_{idx}/{output_names[i]}.raw'
                signals_per_head.append(np.fromfile(fname, np.float32) if os.path.exists(fname) else None)
            # print("signals per head len = ", len(signals_per_head))

            key_signal = concatenate_signals(signals_per_head, "key", output_names)
            # print("len key signal = ", len(key_signal))
            val_signal = concatenate_signals(signals_per_head, "value", output_names)
            # print("len val signal = ", len(val_signal))
            # interleave the lists as k1, v1, k2, v2, ...
            signal = [val for pair in zip(key_signal, val_signal) for val in pair]
            # append the signal for the logits
            signal = [signals_per_head[0]] + signal
            # print("len signal = ", len(signal))


            # collect all the noisy signals from each of the heads
            noisy_per_head = [np.fromfile(f'{output_dir}/Result_{idx}/{o}.raw', np.float32) for o in output_names]
            # print("noisy per head len =  ", len(noisy_per_head))

            key_noisy = concatenate_signals(noisy_per_head, "key", output_names)
            # print("len key noisy = ", len(key_noisy))
            val_noisy = concatenate_signals(noisy_per_head, "value", output_names)
            # print("len val noisy = ", len(val_noisy))
            # interleave the lists as k1, v1, k2, v2, ...
            noisy = [val for pair in zip(key_noisy, val_noisy) for val in pair]
            # prepend noisy logits
            noisy  = [noisy_per_head[0]] + noisy
            # print("len noisy = ", len(noisy))


            ss = [sqnr(s, n) if s is not None else nan for s, n in zip(signal, noisy)]
            ms = [mse(s, n)  if s is not None else nan for i, (s, n) in enumerate(zip(signal, noisy))]
            cs = [cosine_similarity(s, n) if s is not None else nan for s, n in zip(signal, noisy)]

            print(f'{idx%20+1:5} || ' +
                ' || '.join([f"{s:7.3f} | {m:<9.7f} | {c:7.4f}" for s,m,c in zip(ss, ms, cs)])
                )
            csvfile.write(f'{idx%20+1},'
                + ','.join([f"{s:f},{m:f},{c:f}" for s,m,c in zip(ss, ms, cs)])
                + f'\n'
                )

        if os.path.exists('check-output.csv'): os.unlink('check-output.csv')
        os.symlink(csvfilename, 'check-output.csv')


def _load_model(onnxfile, load_external_data=False, model_cache={}):
    if onnxfile not in model_cache:
        print(f'Loading {onnxfile}', file=sys.stderr)
        model_cache[onnxfile] = onnx.load(onnxfile, load_external_data=load_external_data)
    return model_cache[onnxfile]



def get_input_layout(onnxfile, using_qairt_workflow=False):
    def _dims(vi):
        return ','.join([str(dim.dim_value) for dim in vi.type.tensor_type.shape.dim])

    layout = os.getenv('INPUT_LAYOUT', 'NONTRIVIAL')
    supported = ['NCDHW', 'NDHWC', 'NCHW', 'NHWC', 'NFC', 'NCF', 'NTF', 'TNF', 'NF', 'NC', 'F', 'NONTRIVIAL']
    assert layout in supported, f'Unexpected layout:{layout}, it should be one of {supported}'
    no_layout_option = [] #'attention_mask', 'position_ids']

    onnxmodel = _load_model(onnxfile, load_external_data=False)
    if using_qairt_workflow:
        input_info = [("--source_model_input_layout", i.name, layout)
                      for i in onnxmodel.graph.input if not any(j in i.name for j in no_layout_option)]
    else:
        input_info = [("--input_layout", i.name, layout, "--input_dim", i.name, _dims(i))
                      for i in onnxmodel.graph.input if not any(j in i.name for j in no_layout_option)]
    return input_info


def _load_encoding(encodingfile, no_merge=False):
    all = {}
    if encodingfile is not None:
        print(f'Loading {encodingfile}')
        with open(encodingfile) as json_file:
            quant_encoding_dict = json.load(json_file)
        if no_merge:
            return quant_encoding_dict
        all.update(quant_encoding_dict['activation_encodings'])
        all.update(quant_encoding_dict['param_encodings'])
    return all

def _save_encoding(encodings, encodingfile):
    print(f'Saving {encodingfile}')
    with open(encodingfile, 'wt') as json_file:
        quant_encoding_dict = json.dump(encodings, json_file, indent=4, sort_keys=True)


def embed_forecast_token_embeddings(onnxmodel, forecast_token_embeddings, base_dir):

    embedding_table_name, = [node.input[0] for node in onnxmodel.graph.node if node.op_type == 'Gather']
    embedding_table_proto, = [i for i in onnxmodel.graph.initializer if i.name == embedding_table_name]
    embedding_table = to_array(embedding_table_proto, base_dir=base_dir)

    assert embedding_table.shape[1] == forecast_token_embeddings.shape[1], f'Mismatching token embedding size'
    new_embedding_table = np.concatenate((embedding_table, forecast_token_embeddings), axis=0)
    onnxmodel.graph.initializer.remove(embedding_table_proto)
    onnxmodel.graph.initializer.append(from_array(new_embedding_table, embedding_table_proto.name))

def quantize(f, encoding):
    def _round(x):
        sign = np.where(x < 0, -1, 1).astype(np.float32)
        return np.floor(np.abs(x) + 0.5) * sign

    def _quantize(f, scale, offset, dtype=np.uint8):
        q = _round((f / scale - offset))
        return q.clip(np.iinfo(dtype).min,np.iinfo(dtype).max).astype(dtype)

    def _dequantize(q, scale, offset):
        return (q.astype(np.float32) + offset) * scale

    scale, offset = encoding[0]['scale'], encoding[0]['offset']
    assert encoding[0]['bitwidth'] == 8
    f = np.array(f)
    q = _quantize(f, scale, offset)
    dq = _dequantize(q, scale, offset)
    dB = Metric(f, dq)
    return q, dB

def save_kv_cache(kvcache, encodings, filename, num_layers=10000,check_sqnr=True):
    key_value_encodings = [[encodings[f'past_key_{layer}_in'], encodings[f'past_value_{layer}_in']] for layer in range(num_layers)]
    if check_sqnr:
        for layer, ((key, value), (key_enc, value_enc)) in enumerate(zip(kvcache, key_value_encodings)):
            key_q, key_sqnr = quantize(key, key_enc)
            value_q, value_sqnr = quantize(value, value_enc)
            #print(f'layer[{layer}]: key {key_sqnr}, value {value_sqnr}')
    key_q = [quantize(cache[0], encoding[0])[0] for cache, encoding in zip(kvcache, key_value_encodings)]
    value_q = [quantize(cache[1], encoding[1])[0] for cache, encoding in zip(kvcache, key_value_encodings)]

    # Save as cache file for Qualla
    key_cache = np.concatenate(key_q)
    value_cache = np.concatenate(value_q)

    # from qualla/scripts/store-load-kvcache.py
    CACHE_FILE_SPEC="IIBxHHH"
    CACHE_FILE_SPEC_SIZE = struct.calcsize(CACHE_FILE_SPEC)
    DATATYPES = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, None, np.float16, np.float32, np.float64, np.bool8]
    assert CACHE_FILE_SPEC_SIZE == 16

    with open(filename, "wb") as handle:
        dtype = DATATYPES.index(key_cache.dtype)
        n_layer, n_head, n_tok, n_kv_dim = value_cache.shape
        num_tensors = n_layer * 2
        handle.write(struct.pack(CACHE_FILE_SPEC, num_tensors, 0xc0de, dtype, n_head, n_kv_dim, n_tok))
        key_cache.tofile(handle)
        value_cache.tofile(handle)

def process_ssd_param(onnxfile, encodingfile, ssd_param_filename, transposed_key_cache=True, output_dir='.'):

    ssd_param = torch.load(ssd_param_filename, map_location='cpu')
    prefix = ssd_param['forecast_prefix'].to(torch.float32)
    n_layer, _, _, heads, len_prefix, head_dim = prefix.shape
    prefix = tuple([(prefix[ndx_layer][0], prefix[ndx_layer][1]) for ndx_layer in range(n_layer)])
    if transposed_key_cache:
        prefix = tuple([(kv[0].permute(0,1,3,2), kv[1]) for kv in prefix])

    forecast_kvcache_prefix = prefix
    forecast_token_embeddings = ssd_param['forecast_embedding'].to(torch.float32)
    print(f"SSD: the number of forecast tokens = {len(forecast_token_embeddings)}, length of the forecast prefix = {len_prefix}")

    # Append forecast token embeddings
    onnxmodel = _load_model(onnxfile)
    embed_forecast_token_embeddings(onnxmodel, forecast_token_embeddings, base_dir=os.path.dirname(onnxfile))
    num_layers = len([i for i in onnxmodel.graph.output if i.name.startswith('past_value_')])

    # Quantize kvcache prefix
    encodings = _load_encoding(encodingfile)
    cache_filename = os.path.join(output_dir, 'kv-cache.primary.qnn-htp')
    save_kv_cache(forecast_kvcache_prefix, encodings, cache_filename, num_layers=num_layers)
    print(f'Save forecast kvcache prefix as {cache_filename}')


def split_onnx_by_names(onnxfile, modelname, pickle_filedir, *list_of_output_tensors, output_dir='.', onnxmodel=None, encoding_file=None, embed_ssd_params_file=None, using_qairt_workflow=False):

    if embed_ssd_params_file is not None:
        if not os.path.exists(embed_ssd_params_file):
            raise RuntimeError(f"{embed_ssd_params_file} doesn't exist")

        if not encoding_file or not os.path.exists(encoding_file):
            raise RuntimeError(f"Encoding({encoding_file}) file not found")
        process_ssd_param(onnxfile, encoding_file, embed_ssd_params_file, output_dir=output_dir)

    onnx_to_artifacts_map = dict()
    onnxmodel = _load_model(onnxfile, load_external_data=False) if onnxmodel is None else onnxmodel
    splitter = OnnxSplitter(onnxmodel, verbose=False)
    base_dir = os.path.dirname(onnxfile)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    skip_onnx = os.getenv('SKIP_ONNX', '') == '1'
    skip_test_vector=os.getenv('SKIP_TEST_VECTOR', '') == '1'

    list_of_output_tensors = [i.split(',') for i in list_of_output_tensors]
    num_splits = len(list_of_output_tensors) + 1
    pathlib.Path(f'{output_dir}/split_onnx').mkdir(parents=True, exist_ok=True)

    # 1. split model
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f'{modelname}_{i+1}_of_{num_splits}'
        input_tensor_names = [i.name for i in subgraph.input]
        output_tensor_names = [i.name for i in subgraph.output]
        new_model_info.append([new_basename, input_tensor_names, output_tensor_names])

        if skip_onnx: continue

        submodel = onnx.helper.make_model(subgraph, opset_imports=onnxmodel.opset_import)
        if not using_external_data and submodel.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
            onnx.checker.check_model(submodel)

        if using_external_data:
            onnx.load_external_data_for_model(submodel, base_dir=base_dir)

        newonnxfile = f'{output_dir}/split_onnx/{new_basename}.onnx'
        print(f'Saving {newonnxfile}')
        save_model(submodel, newonnxfile, using_external_data)

    if skip_test_vector:
        print(f'Skip saving test vector')
        return

    for source in ['qt']:
        # 2. prepare test vector
        picklefile = _test_vector_pickle_file(source, pickle_filedir)
        print(picklefile)
        tv = TestVectorBase.load(picklefile, onnxfile, using_qairt_workflow=using_qairt_workflow)

        for i, (new_basename, input_names, output_names)  in enumerate(new_model_info):
            sample_dir = _input_dir(new_basename, source)
            golden_output_dir = _golden_output_dir(new_basename, source)

            if i > 0:
                # Don't dump input_ids and position_ids
                # These 2 inputs seem to be removed by QNN converter because only its shape is being used!!!
                input_names = [i for i in input_names if i not in ['input_ids', 'position_ids']]

            # Prepare inputs
            with open(f'{output_dir}/input_list_{new_basename}.txt', 'wt') as input_list:
                for batch, (input_tensors, output_tensors) in enumerate(tv.get_batches([input_names, output_names])):
                    os.makedirs(f'{output_dir}/{sample_dir}', exist_ok=True)
                    filenames = [f'{output_dir}/{sample_dir}/{batch}/{_target_name(tensor, using_qairt_workflow=using_qairt_workflow)}.raw' for tensor in input_names]
                    _dump(input_tensors, filenames)
                    input_list.write(' '.join([f"{_target_name(tensor, using_qairt_workflow=using_qairt_workflow)}:={filename}" for tensor, filename in zip(input_names, filenames)]))

                    # Prepare golden outputs
                    output_dump_names = [f'{output_dir}/{golden_output_dir}/Result_{batch}/{_target_name(tensor, using_qairt_workflow=using_qairt_workflow)}.raw' for tensor in output_names]
                    _dump(output_tensors, output_dump_names)

            onnx_to_artifacts_map[new_basename] = {'input_list': input_list.name, 'golden_output_vectors': output_dump_names}
    return onnx_to_artifacts_map

def _get_lm_head_sizes(onnxmodel):
    "Get dimensions of the LM head : embedding_size, vocab_size"
    lm_head_weight_name  = next(node.input[1] for node in reversed(onnxmodel.graph.node) if node.op_type in ('Conv', 'MatMul', 'Gemm'))
    lm_head_weight, = [i for i in onnxmodel.graph.initializer if lm_head_weight_name == i.name]
    if len(lm_head_weight.dims) == 2:
        embedding_size, vocab_size = lm_head_weight.dims
    else:
        lm_head, = [i for i in onnxmodel.graph.node if lm_head_weight.name in i.input]
        if lm_head.op_type == 'Conv':
            attr_group = [i.i for i in lm_head.attribute if i.name == 'group']
            group = attr_group[0] if len(attr_group) == 1 else 1
            grouped_vocab, group_size, _, _ = lm_head_weight.dims
            vocab_size, embedding_size = grouped_vocab//group, group * group_size
        elif lm_head.op_type == 'MatMul':
            group, group_size, vocab_size = lm_head_weight.dims
            embedding_size = group * group_size
        else:
            raise RuntimeError(f'Unexpected lm_head op_type:{lm_head}')

    return embedding_size, vocab_size

def is_encoding_present(encodings, in_tensor):
    if isinstance(encodings, list):
        for encoding in encodings:
            if encoding['name'] == in_tensor:
                return True
    elif isinstance(encodings, dict):
        return in_tensor in encodings
    else:
        raise TypeError("Encodings data must be a list or a dictionary")
    return False

def get_encoding(encodings, in_tensor):
    if isinstance(encodings, list):
        for encoding in encodings:
            if encoding['name'] == in_tensor:
                return encoding
    elif isinstance(encodings, dict):
        if in_tensor in encodings:
            return encodings[in_tensor]
    return None

def fill_input_encodings_of_split(onnxmodel, encodingfile, output_tensor_list):
    changed = False
    encodings = _load_encoding(encodingfile, no_merge=True)
    enc_act, enc_param = encodings['activation_encodings'], encodings['param_encodings']
    producer = {tensor: node for node in onnxmodel.graph.node for tensor in node.output}
    for split_tensor in output_tensor_list:
        if not is_encoding_present(enc_act, split_tensor):
            assert split_tensor in producer
            input_tensor = producer[split_tensor].input[0] # use only 1st input
            while not is_encoding_present(enc_act, input_tensor) and not is_encoding_present(enc_param, input_tensor):
                input_tensor = producer[input_tensor].input[0]
            input_encoding = {}
            if is_encoding_present(enc_act, input_tensor):
                input_encoding = get_encoding(enc_act, input_tensor)
            else:
                input_encoding = get_encoding(enc_param, input_tensor)
            if isinstance(enc_act, list):
                input_encoding['name'] = split_tensor
                enc_act.append(input_encoding)
            else:
                enc_act[split_tensor] = input_encoding
            print(f'Copy encoding {input_tensor} -> {split_tensor}')
            changed = True

    if changed:
        backup = f'{encodingfile}.bak'
        if not os.path.exists(backup):
            shutil.move(encodingfile, backup)
        _save_encoding(encodings, encodingfile)

def split_onnx(onnxfile, modelname, pickle_filedir, num_splits, output_dir='./', split_embedding=False, encoding_file=None, embed_ssd_params_file=None, using_qairt_workflow=False):

    def _is_cache(layer, name):
        return re.search(f'past_(key|value)_{layer}_', name) != None

    num_splits = int(num_splits)

    onnxmodel = _load_model(onnxfile, load_external_data=False)
    input_names, output_names = get_onnx_input_output_names(onnxfile, onnxmodel=onnxmodel, deco_digit=False, using_qairt_workflow=using_qairt_workflow)
    output_tensor_list = get_split_tensors(onnxfile, onnxmodel=onnxmodel, include_first_input=split_embedding)
    print('Per_layer_output_names:', output_tensor_list)


    # Infer the shape of per-layer tensors
    onnx_input_names = [i.name for i in onnxmodel.graph.input]
    if 'input_ids' in onnx_input_names:
        input_ids, = [i for i in onnxmodel.graph.input if i.name == 'input_ids']
        batch_size, seq_length = [i.dim_value for i in input_ids.type.tensor_type.shape.dim]
    elif any(name in onnx_input_names for name in ('inputs_embeds', 'input_embeds')):
        input_embeds, = [i for i in onnxmodel.graph.input if i.name in ('inputs_embeds', 'input_embeds')]
        batch_size, seq_length, embed_dim = [i.dim_value for i in input_embeds.type.tensor_type.shape.dim]
        split_embedding = False # if input_embeds is the input to onnx model that means embedding layer is not existed
    else:
        raise ValueError("input_ids and input_embeds are not found in the onnx input:", onnx_input_names)

    embedding_size, vocab_size = _get_lm_head_sizes(onnxmodel)
    print(f'Using per-layer output shape: {[batch_size, seq_length, embedding_size]}')

    per_layer_output_value_info = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [batch_size, seq_length, embedding_size]) for name in output_tensor_list]
    onnxmodel.graph.value_info.extend(per_layer_output_value_info)

    num_layers = len(output_tensor_list)
    num_layers_per_split = num_layers//num_splits
    past_key_values = {layer:[output for output in output_names if _is_cache(layer, output)] for layer in range(num_layers)}

    names_to_split = []
    if split_embedding:
        first_output_tensors = output_tensor_list[0].split(',')
        fill_input_encodings_of_split(onnxmodel, encoding_file, first_output_tensors)
        names_to_split.append(output_tensor_list[0])
        output_tensor_list.pop(0)

    num_layers = len(output_tensor_list)
    num_layers_per_split = (num_layers // (num_splits-1)) if split_embedding else (num_layers // num_splits)
    past_key_values = {layer:[output for output in output_names if _is_cache(layer, output)] for layer in range(num_layers)}

    for layer_end in range(num_layers_per_split,num_layers,num_layers_per_split):
        outputs = [output_tensor_list[layer_end-1]]
        for layer in range(layer_end-num_layers_per_split, layer_end):
            outputs += past_key_values[layer]
        names_to_split.append(','.join(outputs))

    print('Names_to_split', names_to_split)
    assert num_splits == len(names_to_split)+1, f"Failed to split into {num_splits} pieces!"
    return split_onnx_by_names(onnxfile, modelname, pickle_filedir, *names_to_split,
                    output_dir=output_dir, onnxmodel=onnxmodel,
                    encoding_file=encoding_file, embed_ssd_params_file=embed_ssd_params_file,
                    using_qairt_workflow=using_qairt_workflow)


def _short(s):
    words = s.split('_')
    return ''.join(list(words[0][:2]) + [w[0] for w in words[1:]])

def add_shortcuts(commands):
    new = dict(commands)
    new.update({_short(k): v for k, v in commands.items()})
    return new

def usage(commands):
    desc = '\n  '.join([f'{cmd}{inspect.signature(func)}, short:{_short(cmd)}' for cmd, func in commands.items()])

    print(f'Usage "helper.py command [arguments, ...]\n'
    'Available commands\n'
    f'  {desc}'
    )
    exit(1)


if __name__ == '__main__':

    commands = {
        'prepare_test_vector': prepare_test_vector,
        'prepare_test_vector_fp': prepare_test_vector_fp,
        'check_output': check_output,
        'print_input_layout': get_input_layout,
        'split_onnx': split_onnx,
    }

    if len(sys.argv) < 2:
        usage(commands)
    newcommands = add_shortcuts(commands)

    cmd = sys.argv[1]
    if cmd not in newcommands:
        bestfit = get_close_matches(cmd, newcommands.keys(), n=1, cutoff=0.7)
        if not bestfit:
            usage(commands)
        cmd = bestfit[0]

    newcommands[cmd](*sys.argv[2:])
