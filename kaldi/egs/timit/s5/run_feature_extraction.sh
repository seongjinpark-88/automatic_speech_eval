#!/usr/bin/env bash

# This script generates MFCCs and cmvn statistics

# ARGUMENTS
# REQUIRED
# -c <path> = path to `.conf` file
# -d [no argument] = wav file dir is TRUE in kaldi_config.json
# Must have *at least* one of `-n` or `-t`
# OPTIONAL
# -j <int> = number of processors to use, default=8
# -q <string> = non-vanilla hyperparameters to `compute_cmvn_stats.sh`, in the form "--fake-dims 13:14"

# OUTPUTS
# Creates:
    # `mfcc/` directory for the `mfcc`s from training data
    # `exp/` for logs
    # `data/{train,test}dir/{feats,cmvn}.scp` which are required when running `run_train_phones.sh`

############################
##BEGIN parse params##
############################

# all params
all_params="\
    mfcc_config \
    data_present \
    num_processors \
    non_vanilla_compute_cmvn_stats_hyperparameters"

# parse parameter file
for param in ${all_params}; do
    param_name=${param}
    declare ${param}="$(python utils/parse_config.py $1 $0 ${param_name})"
done

############################
##END parse params##
############################

# source file with path information
. path.sh

data_dir=${data_present}

# update variables for train/test dir
# train_dir=
# test_dir=

# if [ "${train_data_present}" = true ]; then
#     train_dir=train_dir
# fi

# if [ "${test_data_present}" = true ]; then
#     test_dir=test_dir
# fi

printf "Timestamp in HH:MM:SS (24 hour format)\n";
date +%T
printf "\n"

# run fix_data_dir just in case
#    utils/fix_data_dir.sh ${dir}

# make mfccs
./steps/make_mfcc.sh --nj ${num_processors} \
    --mfcc-config ${mfcc_config} \
    data/${data_dir} \
    exp/make_mfcc/${data_dir} \
    mfcc \
    || (printf "\n####\n#### ERROR: make_mfcc.sh \n####\n\n" && exit 1);

# compute cmvn stats
./steps/compute_cmvn_stats.sh \
    ${non_vanilla_compute_cmvn_stats_hyperparameters} \
    data/${data_dir} \
    exp/make_mfcc/${data_dir} \
    mfcc \
    || (printf "\n####\n#### ERROR: compute_cmvn_stats.sh \n####\n\n" && exit 1);

printf "Timestamp in HH:MM:SS (24 hour format)\n";
date +%T
printf "\n"

python ${KALDI_PATH}/egs/timit/s5/utils/parse_config.py $1 $0 > mfcc/kaldi_config_args.json
