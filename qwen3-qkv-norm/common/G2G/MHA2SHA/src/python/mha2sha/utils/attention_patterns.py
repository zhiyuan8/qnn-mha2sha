# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

attention_patterns = [
    {
        "pattern": ["MatMul", "Div", "Add", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Div", "Where", "Add", "Softmax", "MatMul"],
    },
    {
        "pattern": ["MatMul", "Div", "Add", "Add", "Softmax", "MatMul"],
    },
    {
        "pattern": ["MatMul", "Add", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Div", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Where", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Mul", "Softmax", "MatMul"]  # SD 2.1
    },
    {
        "pattern": ["MatMul", "Div", "Add", "Softmax", "Cast", "Cast", "MatMul"]  # LLaMA v2 SS
    },
    {
        "pattern": ["MatMul", "Div", "Transpose", "Add", "Transpose", "Softmax", "MatMul"]  # LLaMA v2 BERT
    }
]
