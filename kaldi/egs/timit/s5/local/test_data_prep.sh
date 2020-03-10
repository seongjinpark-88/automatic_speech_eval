#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Argument should be test dir"
    exit 1;
fi 

currentdir=`pwd`
dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT

export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi


# First check if the wav directories exist (these can either be upper-
# or lower-cased)

if [ ! -d $*/wav ]; then
    echo "timit_data_prep.sh: Spot check of command line argument failed"
    echo "Command line argument must be absolute pathname to WAV directory"
    echo "with name like /my_dir_name/asr"
    exit 1;
fi

# Now check what case the directory structure is
asr_dir=$1/wav

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.

tr '[:lower:]' '[:upper:]' < $conf/asr_spk.list > $tmpdir/asr_spk

cd $dir

find $asr_dir | grep -f $tmpdir/asr_spk > asr_sph.flist

sed -e 's:.*/\(.*\)/\(.*\).wav$:\2:i' asr_sph.flist \
  > $tmpdir/asr_sph.uttids
paste $tmpdir/asr_sph.uttids asr_sph.flist \
  | sort -k1,1 > asr_sph.scp

cat asr_sph.scp | awk '{print $1}' > asr.uttids

# Create wav.scp
# awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < asr_sph.scp > asr_wav.scp
awk '{printf("%s %s\n", $1, $2);}' < asr_sph.scp > asr_wav.scp

# Make the utt2spk and spk2utt files.
cut -f1 -d'_'  asr.uttids | paste -d' ' asr.uttids - > asr.utt2spk
cat asr.utt2spk | $utils/utt2spk_to_spk2utt.pl > asr.spk2utt || exit 1;

cd $currentdir

mkdir -p data/asr_dir
cp $dir/asr_wav.scp data/asr_dir/wav.scp || exit 1;
cp $dir/asr.spk2utt data/asr_dir/spk2utt || exit 1;
cp $dir/asr.utt2spk data/asr_dir/utt2spk || exit 1;