# !/bin/bash
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# This script is inspired from the QAIRT env_setup.sh

function usage()
{
  clean_up_error;
  cat << EOF
Script sets up environment variables for the MHA2SHA tool.

USAGE:
  $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h]

EOF
}

function clean_up_error()
{
  if [[ -n "$OLD_PYTHONPATH"  ]]
  then
    export PYTHONPATH=$OLD_PYTHONPATH
  fi
}


OLD_PYTHONPATH=$PYTHONPATH

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "[ERROR] This file should be run with 'source'"
  usage;
  return 1;
fi

OPTIND=1
while getopts "h?s:m:" opt; do
    case "$opt" in
    h)
        usage;
        return 0
        ;;
    \?)
    usage;
    return 1 ;;
    esac
done


# Get the source dir of the env_setup.sh script
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
MHA2SHA_ROOT=$(readlink -f ${SOURCEDIR}/..)

export PYTHONPATH=$PYTHONPATH:$MHA2SHA_ROOT/src/python
export PATH=$MHA2SHA_ROOT/bin:$PATH

echo "MHA2SHA tool root set to:- "$MHA2SHA_ROOT

python_version=$(python --version)
echo "Python Version:- $python_version"

# Clean up the local variables
unset mha2sha_root python_version OLD_PYTHONPATH SOURCEDIR MHA2SHA_ROOT
