# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import List, Optional
import shutil

import rich
import rich.table
import rich.markdown


class PrettyPrintConstants:
    DEFAULT_LINE_LENGTH = 88
    GREEN = "40"
    RED = "160"
    Q_BLUE = "57"
    MAX_WIDTH = 4


def create_rich_table(
    title: Optional[str] = None,
    caption: Optional[str] = None,
    headers: Optional[List[str]] = ["Flag", "Value"],
    positions: Optional[List[float]] = [0.45, 1.0],
    alignment: Optional[List[str]] = ["left", "middle"]
) -> rich.table:
    """Creates a blank rich table"""

    num_headers = len(headers)
    for attr in (positions, alignment):
        assert num_headers == len(attr)

    line_length = min(
        PrettyPrintConstants.DEFAULT_LINE_LENGTH, shutil.get_terminal_size().columns - 4
    )

    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < PrettyPrintConstants.MAX_WIDTH:
            raise ValueError("Insufficient console width to print Flag-Value Table.")
        column_widths.append(width)
        current += width

    columns = []
    for i, name in enumerate(headers):
        column = rich.table.Column(
            name,
            justify=alignment[i],
            width=column_widths[i],
        )
        columns.append(column)

    return rich.table.Table(
        title=title,
        caption=caption,
        *columns,
        width=line_length,
        show_lines=True
    )


def bold_text(x, color=None):
    """Bolds text using rich markup."""
    if color:
        return f"[bold][color({color})]{x}[/][/]"
    return f"[bold]{x}[/]"

def true_false_color(x):
    return PrettyPrintConstants.GREEN if x else PrettyPrintConstants.RED
