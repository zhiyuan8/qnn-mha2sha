# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import rich
import rich.markdown

from mha2sha import __version__
from mha2sha.utils.base_llm_configs import base_llms_to_flags

from mha2sha.utils.logger import log_assert, log_debug, log_error, log_warning
from mha2sha.utils.pretty_print import (
    bold_text,
    create_rich_table,
    true_false_color,
    PrettyPrintConstants,
)


class ExplicitlySetArgumentParser(argparse.ArgumentParser):

    set_args = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        return_args = super().parse_args(*args, **kwargs)
        args = args[0] if args else sys.argv[1:]
        args = [
            # --not-strict isn't really a flag, it just set --strict
            # to False.
            "--strict" if arg == "--not-strict" else arg
            for arg in args
            if arg.startswith("--")
        ]

        ExplicitlySetArgumentParser.set_args.update({unflagify(arg) for arg in args})

        return return_args


def converter_arg_parser() -> argparse.ArgumentParser:
    """Parses arguments for a MHA2SHA Converter run.

    Argument parser specific for the entry run into the MHA2SHA converter. Use `-h`
    for more information on available arguments.
    """

    # Please keep args alphabetical
    parser = ExplicitlySetArgumentParser()
    parser.add_argument(
        "--ar-num",
        default=None,
        type=int,
        help="Manually overwrite ar num and bypass ar num detection.",
    )
    parser.add_argument(
        "--base-llm",
        help="Mapping of LLM models to the default needed flags",
        required=False,
    )
    parser.add_argument(
        "--build-ar",
        type=check_ar_value,
        help="Builds a SHA model with a different AR provided. AR value must be >=1.",
        required=False,
    )
    parser.add_argument(
        "--create-input-lists",
        action="store_true",
        help="Creates a Linux/On device input list txt, saves the input data used and goldens.",
    )
    parser.add_argument(
        "--disable-auto-attn-finder",
        action="store_true",
        help="Disable auto attention finder in step 5.",
    )
    parser.add_argument(
        "--exported-model-encoding-path",
        default=None,
        help="aimet encoding path",
    )
    parser.add_argument(
        "--exported-model-path",
        default=".",
        help="ONNX model path",
        required=True,
    )
    parser.add_argument(
        "--gqa-model",
        action="store_true",
        help="Model has GQA architecture.",
    )
    parser.add_argument(
        "--handle-alibi",
        action="store_true",
        help="Model has ALiBi position embeddings"
    )
    parser.add_argument(
        "--handle-past-key-value",
        action="store_true",
        help="Enable handling of past key/value in LLM's",
    )
    parser.add_argument(
        "--handle-rope-ops",
        action="store_true",
        help="Enable handling of RoPE ops.",
    )
    parser.add_argument(
        "--handle-internal-rmsnorm",
        action="store_true",
        help="Enable handling of RMSNorm pattern on Q and K branches",
    )
    parser.add_argument(
        "--llm-model",
        action="store_true",
        help="Enables handling of LLM specific models.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Sets the log level. Default is 'info'.\nValid values are: 'warn', 'verbose', 'info', 'error', 'fatal'.",
    )
    parser.add_argument(
        "--lora-adaptor-list-path",
        default=None,
        help="Path to lora adaptor list yaml file",
    )
    parser.add_argument(
        "--lora-alpha-from-input",
        action="store_true",
        help="Lora alpha from model input. Use this option when --lora-model is also used.",
    )
    parser.add_argument(
        "--lora-model",
        action="store_true",
        help="Model has lora adaptor.",
    )
    parser.add_argument(
        "--mha-conv",
        action="store_true",
        help="Conv MHA model",
    )
    parser.add_argument(
        "--mha-lora-tensor-path",
        default=None,
        help="Path to MHA LoRA tensor name.",
        required=False,
    )
    parser.add_argument(
        "--model-name",
        required=False,
        default=None,
        help="Model name to generate",
    )
    parser.add_argument(
        "--nchw-aligned",
        action="store_true",
        help="Model is aligned to NCHW format.",
    )
    parser.add_argument(
        "--handle-r3-matrix",
        action="store_true",
        help="Enable handling of R3 matrix in RoPE ops",
    )
    parser.add_argument(
        "--no-verification",
        action="store_true",
        help="Does not run extra steps for verification of the model and encoding mappings",
    )
    parser.add_argument(
        "--not-strict",
        dest="strict",
        action="store_false",
        help="Relaxes RoPE pattern split, will no map encodings but will still split model.",
    )
    parser.add_argument(
        "--optimize-o-proj",
        action="store_true",
        help="Optimize sha head concat -> o_proj pattern for mha-conv model.",
    )
    parser.add_argument(
        "--prepared-model",
        action="store_true",
        help="Enable changes for prepared model.",
    )
    parser.add_argument(
        "--position-ids",
        default=None,
        help="Path to the positions ids (cos/sin)",
    )
    parser.add_argument(
        "--replace-linear-with-conv",
        action="store_true",
        help="enable linear to conv conversion",
    )
    parser.add_argument(
        "--sha-export-path",
        default=Path.cwd() / "exported_model",
        help="Path to export SHA artifacts",
    )
    parser.add_argument(
        "--skip-mha2sha",
        action="store_true",
        help="Skip mha2sha",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Set by default, will strictly enforce Golden RoPE pattern.",
    )
    parser.set_defaults(strict=True)

    return parser


def check_ar_value(value: int) -> int:
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")

    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is less than 1")

    return value


def pretty_print_args(args_: dict, model_byte_size: str) -> None:
    """Pretty prints the args chosen for a run

    Heavily inspired from Keras 'print_summary'
    https://github.com/keras-team/keras/blob/1c60668f6bdd05dab619806e7b2dc25d3ed4ccbf/keras/src/utils/summary_utils.py#L98

    Args:
        args_:
            Dict of the args and the value set to them
        model_byte_size:
            Model size in bytes
    """
    # Sort the args alphabetically
    args_ = dict(sorted(args_.items(), key=lambda x: x[0].lower()))

    args_always_explicit = {
        "base_llm",
        "exported_model_encoding_path",
        "exported_model_path",
        "log_level",
        "model_name"
        "sha_export_path",
    }

    user_overwritten_args = dict()

    byte_size_caption = (
        bold_text(" Model byte size: ")
       + bold_text(f"{model_byte_size:,}", color=PrettyPrintConstants.Q_BLUE)
    )
    version_caption = (
        bold_text(" Version: ")
        + bold_text(__version__, color=PrettyPrintConstants.Q_BLUE)
    )

    # Flags to set value table
    table = create_rich_table(
        title=bold_text(" Qualcomm MHA2SHA", color=PrettyPrintConstants.Q_BLUE),
        caption=f"{version_caption}\n{byte_size_caption}"
    )

    for arg, val in args_.items():
        pp_arg = rich.markup.escape(flagify(arg)[1:-1])  # [1:-1] remove ''
        if pp_arg == "--ar-num":
            pp_arg += " [ To be deprecate] "

        color = None

        # If a user explicitly set a flag and 'base_llm' was set, then
        # we want to warn the user in the table.
        if args_["base_llm"] and arg in (
            ExplicitlySetArgumentParser.set_args - args_always_explicit
        ):
            user_overwritten_args[pp_arg] = (_args_to_default_value[arg], val)
            pp_val = f":warning:  [bold red blink2]{val}[/]"

        else:
            # Coloring set flags True to Green and False flags to red.
            if isinstance(val, bool):
                color = true_false_color(val)

            # Anything that is not a bool will have qoutes around it.
            elif val is not None:
                if arg == "exported_model_encoding_path":
                    val = os.path.basename(val)
                val = f'"{val}"'

            pp_val = bold_text(f"{val}", color=color)
        table.add_row(pp_arg, pp_val)

    console = rich.console.Console(highlight=True)
    console.print(table)
    console.print()  # separation
    console.print()  # separation

    if user_overwritten_args:
        pretty_print_warning_table(
            console,
            args_["base_llm"],
            user_overwritten_args,
        )


def pretty_print_warning_table(
    console,
    base_llm: str,
    overwritten_args: Dict[str, Tuple[str]]
) -> None:
    """Pretty prints a warning table of user overwritten flags.

    If a user passes in the 'base_llm' and overwrites a flag,
    a table will be printed with the changes and the defaults.

    Args:
        console:
            Rich Console to print with.
        base_llm:
            Base LLM that was overwritten.
        overwritten_args:
            Args overwritten with the default and set value.
    """
    warning_explanation = """
    The [italic]--base-llm[/] flag is used to turn on necessary flags for specific model
    architectures. For example, [italic]--base-llm llama2[/] would implictly turn on flags
    such as [italic]--handle-rope-ops[/]. However, some of these flags were overwritten
    by user provided arguments.
    """
    try:
        table = create_rich_table(
            # extra space needed initially, otherwise overlaps
            title=f":warning:  --base-llm '{base_llm}' Mapped Args Overwritten :warning:",
            headers=["Flag", f"'{base_llm}' Default", "User Provided"],
            positions=[0.3, 0.6, 0.8],
            alignment=["left", "middle", "left"]
        )
    except ValueError:
        log_error("Insufficient console width to print table.")
        return

    for flag, (default, override) in overwritten_args.items():
        table.add_row(
            flag,
            bold_text(default, true_false_color(default)),
            bold_text(override, true_false_color(override))
        )
    console.print(table)

    console.print(
        rich.panel.Panel(warning_explanation, width=table.width)
    )

def flagify(var: str) -> str:
    """Converts a python variable into a flag look

    Ex: handle_rope_ops -> --handle-rope-ops

    Args:
        var:
            Variable to change to a flag.

    Returns:
        A variable to a flag look.
    """

    return f"'--{var.replace('_', '-')}'"


def unflagify(flag: str) -> str:
    """Converts a flag to a python variable

    Ex: --handle-rope-ops -> handle_rope_ops

    Args:
        flag:
            Flag to convert.

    Returns:
        Flag converted to python varaible.
    """
    return flag[2:].lower().replace("-", "_")


_ARGS_TO_IGNORE = [
    "--help",
    "--sha-export-path",
    "--model-name",
    "--exported-model-path",
]

_args_to_default_value = {
    unflagify(action.option_strings[0]): action.default
    for action in converter_arg_parser()._actions
    if action.option_strings[-1] not in _ARGS_TO_IGNORE
}

AvailableConverterArgs = Enum(
    "AvailableConverterArgs",
    {
        arg_name.upper(): arg_name
        for arg_name in _args_to_default_value.keys()
    },
)

AvailableConverterArgs.default_value = (
    lambda self: _args_to_default_value[
        self.value.lower()
])


def filter_args(**passed_in_args) -> Dict:
    """Filters passed in args and returns the args updated

    If a user passes in '--base-llm' we will turn on the corresponding models
    flags. However, if a user explicitly passes in a value that conflicts
    against the base-llm, we will NOT update that flag.

    Args:
        passed_in_args:
            Args from the command line or through python code.

    Returns:
        Args passed in filtered and updated.
    """

    args_to_iter = [
        arg
        for arg in list(AvailableConverterArgs)
        if arg != AvailableConverterArgs.BASE_LLM
    ]
    args_to_return = {}

    if not ExplicitlySetArgumentParser.set_args:
        ExplicitlySetArgumentParser.set_args.update(
            {arg_name for arg_name in passed_in_args.keys()}
        )

    def override_arg(x):
        return x not in ExplicitlySetArgumentParser.set_args

    base_llm = passed_in_args.pop(
        "base_llm", AvailableConverterArgs.BASE_LLM.default_value()
    )
    args_to_return["base_llm"] = base_llm

    for available_arg in args_to_iter:
        # .value -> arg name in enum (value is apart of enum)
        # .default_value() -> default value for the arg
        arg_name = available_arg.value
        default_val = available_arg.default_value()

        # First we set the attr to the arg passed in or the default
        # value.
        passed_in_arg_val = passed_in_args.pop(arg_name, default_val)
        args_to_return[arg_name] = passed_in_arg_val

        # Second, if we have a "base-llm" flag passed in then we can
        # override an arg if that arg is NOT in the explicitly set args.
        # Otherwise, we set what the user passed in.
        if base_llm:
            base_llm_val = base_llms_to_flags[base_llm].get(arg_name, None)
            if base_llm_val is not None:
                if override_arg(arg_name):
                    log_debug(
                        f"{flagify(arg_name)} overridden to '{base_llm_val}'"
                    )
                    args_to_return[arg_name] = base_llm_val
                else:
                    log_warning(
                        f"Flag {flagify(arg_name)} is set to '{base_llm_val}' "
                        f"for '{base_llm}' but got '{passed_in_arg_val}' "
                        "explicitly. Setting flag to explict value."
                    )

    # All args are popped and if there are any left, we are not
    # expecting them.
    log_assert(
        not passed_in_args,
        "The following arguments are not available: "
        f"'{', '.join(passed_in_args.keys())}'",
    )

    return args_to_return
