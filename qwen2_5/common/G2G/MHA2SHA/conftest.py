# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import glob
import os
from typing import List

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--filtered-configs",
        action="store",
        help="Run only the models specified in comma separated format",
    )
    parser.addoption(
        "--save-artifacts",
        action="store_true",
        default=False,
        help="By default, the test artifacts are saved to temp directory, this will save them in 'test-artifacts'",
    )


@pytest.fixture
def save_artifacts(request):
    return request.config.getoption("--save-artifacts")


def pytest_collection_modifyitems(config, items):
    # If the user is at the root dir, no need to run all tests. Just full flow.
    if "/".join(os.path.normpath(os.getcwd()).split("/")[-3:]) == "mha2sha/python/test":
        selected_items = []
        deselected_items = []
        for item in items:
            if "test_full_run_onnxruntime.py" in item.nodeid:
                selected_items.append(item)
            else:
                deselected_items.append(item)
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items


def pytest_generate_tests(metafunc):
    if "test_full_run_onnxruntime" == metafunc.definition.module.__name__:
        config_files = glob.glob(os.path.abspath("./configs/*.json"))
        if filtered_configs := metafunc.config.option.filtered_configs:
            config_files = get_selected_config_files(filtered_configs, config_files)
        metafunc.parametrize("config_file", config_files)


def get_selected_config_files(filtered_config_files: str, config_files: List[str]):
    filters = [f.strip().lower() for f in filtered_config_files.split(",")]
    base_config_filenames = [os.path.basename(c).split(".")[0] for c in config_files]
    selected_config_files = [
        c
        for c, b in zip(config_files, base_config_filenames)
        if any(b.startswith(f) for f in filters)
    ]

    if not selected_config_files:
        raise ValueError(
            f"No models found for the given filter(s): {filtered_config_files}\n"
            f"Available filters are: {', '.join(base_config_filenames)}"
        )
    return selected_config_files
