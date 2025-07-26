#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import io
import itertools
import os
import pickle
import shutil

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import torch


def apply_fix(onnxmodel, base_dir, fix):

    changed = [0]
    def fix_value_info_proto(valueinfoproto):
        before = [dim.dim_value for dim in valueinfoproto.type.tensor_type.shape.dim]
        for dim in valueinfoproto.type.tensor_type.shape.dim:
            if dim.dim_value in fix:
                dim.dim_value = fix[dim.dim_value]
        after = [dim.dim_value for dim in valueinfoproto.type.tensor_type.shape.dim]
        if before != after:
            print(f'{before} => {after} : {valueinfoproto.name}')
            changed[0] += 1

    def fix_tensor_proto_in_attribute(tensor_proto):
        tensor = numpy_helper.to_array(tensor_proto, base_dir=base_dir)
        if any(i in tensor for i in fix.keys()):
            arr = copy.copy(tensor)
            for k, v in fix.items():
                arr[arr==k] = v
            print(f'{tensor.tolist()} => {arr.tolist()} : {tensor_proto.name}')
            new_tensor = numpy_helper.from_array(arr, tensor_proto.name)
            tensor_proto.raw_data = new_tensor.raw_data
            changed[0] += 1

    def fix_tensor_proto_in_initializer(tensor_proto):
        tensor = numpy_helper.to_array(tensor_proto, base_dir=base_dir)
        if len(tensor.shape) == 1 and tensor.shape[0] in fix and (tensor == np.arange(tensor.shape[0])).all():
            new_shape = [fix[i] if i in fix else i for i in tensor.shape]
            print(f'{tensor_proto.name} {tensor.shape}, => {new_shape}')
            arr = np.arange(new_shape[0], dtype=tensor.dtype)
            new_tensor = numpy_helper.from_array(arr, tensor_proto.name)
            changed[0] += 1
            return new_tensor
        if (tensor == 1).all() and any(i in list(tensor.shape) for i in fix.keys()):
            new_shape = [fix[i] if i in fix else i for i in tensor.shape]
            print(f'{tensor_proto.name} {tensor.shape}, => {new_shape}')
            arr = np.ones(new_shape).astype(tensor.dtype)
            new_tensor = numpy_helper.from_array(arr, tensor_proto.name)
            changed[0] += 1
            return new_tensor
        return None

    print("Checking graph input/output/value_info")
    for vip in itertools.chain(*(onnxmodel.graph.input, onnxmodel.graph.output, onnxmodel.graph.value_info)):
        fix_value_info_proto(vip)

    print("Checking initializer")
    to_remove = []
    for tp in onnxmodel.graph.initializer:
        new_tp = fix_tensor_proto_in_initializer(tp)
        if new_tp:
            to_remove.append(tp)
            onnxmodel.graph.initializer.append(new_tp)
    for tp in to_remove:
            onnxmodel.graph.initializer.remove(tp)

    print("Checking node attributes")
    for node in onnxmodel.graph.node:
        for attr in node.attribute:
            if attr.type != 4: continue
            fix_tensor_proto_in_attribute(attr.t)

    print(f"Done. Fixed {changed[0]} occurrences")
    return changed[0]


def usage():
    print('Usage: change_hardcoding.py model.onnx --fix 73,128 --fix 373,2048 --fix 300,1920')
    exit(1)

def save_model(model, newonnxfile, using_external_data=False):
    kwargs = {}
    if using_external_data or model.ByteSize()>onnx.checker.MAXIMUM_PROTOBUF:
        dirname = os.path.dirname(newonnxfile)
        location = os.path.basename(newonnxfile).replace('.onnx', '.data')
        kwargs['save_as_external_data'] = True
        kwargs['all_tensors_to_one_file'] = True
        kwargs['location'] = location
        if os.path.exists(os.path.join(dirname, kwargs['location'])):
            os.unlink(os.path.join(dirname, kwargs['location']))

    onnx.save(model, newonnxfile, **kwargs)


def fix_shapes(data, fix):

    def _resize(tensor):
        if any(i in fix for i in list(tensor.shape)):
            new_shape = [fix[i] if i in fix else i for i in tensor.shape]
            print(f'Resize {list(tensor.shape)} => {new_shape}')
            if isinstance(tensor, np.ndarray):
                tensor = tensor.resize(new_shape)
            else:
                if tensor.requires_grad:
                    tensor = tensor.detach()
                nptensor = tensor.cpu().numpy().copy()
                nptensor.resize(new_shape)
                tensor = torch.tensor(nptensor)

        return tensor

    def _fix(d):
        if isinstance(d, np.ndarray): return _resize(d)
        if isinstance(d, torch.Tensor): return _resize(d)
        if isinstance(d, list):
            return [_fix(i) for i in d]
        if isinstance(d, tuple):
            return tuple(_fix(i) for i in d)
        if isinstance(d, dict):
            return {f'{k}':_fix(v) for k,v in d.items()}
        return d
    return _fix(data)


def files(path, ext):
    return [os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files if file.endswith(ext)]

def _mkdir(path):
    os.makedirs(path, exist_ok=True)

def remap_path(oldpath, root, newroot):
    pfx = os.path.commonpath([oldpath, root])
    newpath = f'{newroot}{oldpath[len(pfx):]}'
    _mkdir(os.path.dirname(newpath))
    return newpath


def _load_pickle(pklfile):
    map_location = 'cpu'
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda x: torch.load(io.BytesIO(x), map_location=map_location)
            return super().find_class(module, name)
    with open(pklfile, 'rb') as f:
        return CustomUnpickler(f).load()

def execute(input_path,output_path,fix_list):
    fix = {}
    for opt in fix_list:
        old, new = [int(i) for i in opt.split(',')]
        if old != new:
            fix[old] = new

    onnxfiles = files(os.path.join(input_path, 'onnx'), '.onnx')
    picklefiles = files(os.path.join(input_path, 'test_vectors'), '.pkl')

    to_copy = files(os.path.join(input_path, 'onnx'), '.encodings')
    to_copy += files(os.path.join(input_path, 'onnx'), '.json')
    to_copy += files(os.path.join(input_path, 'onnx'), '.yaml')
    to_copy += files(input_path, '.json')
    to_copy += files(input_path, '.encodings')

    # remove temp... if exists
    onnxfiles = [i for i in onnxfiles if not i.startswith('temp')]

    for onnxfile in onnxfiles:
        base_dir = os.path.dirname(onnxfile)
        newonnxfile = remap_path(onnxfile, input_path, output_path)

        onnxmodel = onnx.load(onnxfile, load_external_data=False)
        changes = apply_fix(onnxmodel, base_dir, fix)
        print(f'Saving as {newonnxfile}, with {changes} changes')
        save_model(onnxmodel, newonnxfile)

    for picklefile in picklefiles:
        try:
            print(f'Loading {picklefile}')
            data = _load_pickle(picklefile)

            newdata = fix_shapes(data, fix)
            newpicklefile=remap_path(picklefile, input_path, output_path)
            with open(newpicklefile, 'wb') as f:
                print(f'Saving {newpicklefile}')
                pickle.dump(newdata, f)

        except ModuleNotFoundError as e:
            print(f'Skip {picklefile}, error: {e}')

    to_copy = filter(lambda filename: not filename.endswith("_torch.encodings"), to_copy)
    for src in to_copy:
        newfile=remap_path(src, input_path, output_path)
        print(f'Copying {newfile}')
        shutil.copyfile(src, newfile)
