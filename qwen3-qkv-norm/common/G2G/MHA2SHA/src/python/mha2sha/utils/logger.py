# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging

# From QAIRT Converter Utils

LOGGER = None
HANDLER = None
LOG_LEVEL = logging.INFO

# Custom Logging
logging.VERBOSE = LOG_LEVEL_VERBOSE = 5

# add the custom log-levels
logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")


def setup_logging(debug_lvl, name="MHA2SHA"):
    global LOGGER
    global HANDLER
    global LOG_LEVEL

    debug_lvl = debug_lvl.lower()
    if debug_lvl == "info":
        LOG_LEVEL = logging.INFO
    elif debug_lvl == "debug":
        LOG_LEVEL = logging.DEBUG
    elif debug_lvl == "verbose":
        LOG_LEVEL = logging.VERBOSE
    else:
        log_assert("Unknown debug level provided. Got {}", debug_lvl)

    if LOGGER is None:
        LOGGER = logging.getLogger(name)
    LOGGER.setLevel(LOG_LEVEL)
    LOGGER.propagate = False

    if HANDLER is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(lineno)d - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        HANDLER = handler
    HANDLER.setLevel(LOG_LEVEL)


def log_assert(cond, msg, *args):
    assert cond, msg.format(*args)


def log_debug(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))


def log_verbose(msg, *args):
    def verbose(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.VERBOSE):
            LOGGER._log(logging.VERBOSE, msg, args, kwargs)

    verbose(msg.format(*args))


def log_error(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))


def log_info(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))


def log_warning(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))
