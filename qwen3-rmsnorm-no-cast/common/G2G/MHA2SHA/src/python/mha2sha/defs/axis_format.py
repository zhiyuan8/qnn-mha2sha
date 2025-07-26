# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class AxisFormat(object):
    """
    Contains axis commonly used axis orders along with
    permute order to go to/from this well-defined formats
    """

    # Batch,Channel,Depth,Height,Width.
    # With one batch and three spatial dimensions. Used for 5D ops.
    NCDHW = "NCDHW"
    # Batch,Depth,Height,Width,Channel. With one batch and
    # three spatial dimensions. Used for 5D ops. This is
    # the native data order for SNPE ops.
    NDHWC = "NDHWC"
    # Batch,Channel,Spatial. With one batch and
    # two spatial dimensions, equivalent to NCHW
    NCS = "NCHW"
    # Batch,Spatial,Channel. With one batch and two spatial dimensions,
    # equivalent to NHWC. This is the native data order for SNPE ops with
    # output feature maps.
    NSC = "NHWC"
    # Batch,Channel,Feature. With one batch and one spatial dimension,
    # Used for Conv1D, BatchNorm1D etc.
    NCF = "NCF"
    # Batch,Feature,Channel. With one batch and one spatial dimension,
    # Used for Conv1D, BatchNorm1D etc. This is the native data order
    # for SNPE ops.
    NFC = "NFC"
    # Time,Batch,Feature.
    TNF = "TNF"
    # Batch,Time,Feature. This is the native data order for SNPE RNN ops.
    NTF = "NTF"
    # Batch,Feature.
    NF = "NF"
    # Batch,Channels. Used with Reduce Ops to identify that
    # reduce happened across Spatial dimensions
    NC = "NC"
    # used by Constant Op and 1D tensor
    ANY = "ANY"
    # Op specific data format.
    NONTRIVIAL = "NONTRIVIAL"
    # Enum value used by buffers which have not yet undergone axis tracking.
    NOT_YET_DEFINED = "NOT_YET_DEFINED"
    # Enum value used by null buffer
    NULL = "NULL"
