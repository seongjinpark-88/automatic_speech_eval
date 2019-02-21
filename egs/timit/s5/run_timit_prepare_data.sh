#!/bin/bash

# This script prepares the data (audio, transcripts, phone list, language model)
# for timit data

##########################
## DEFINE timit location##
##########################

printf "Timestamp in HH:MM:SS (24 hour format\n";
date +%T 
printf "\n"


timit=${KALDI_TIMIT_PATH}/raw_data

${KALDI_TIMIT_PATH}/local/timit_data_prep.sh $timit
${KALDI_TIMIT_PATH}/local/timit_prepare_dict.sh

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
srcdir=${KALDI_TIMIT_PATH}/data/local/data
lmdir=${KALDI_TIMIT_PATH}/data/local/nist_lm
tmpdir=${KALDI_TIMIT_PATH}/data/local/lm_tmp
lexicon=${KALDI_TIMIT_PATH}/data/local/dict/lexicon.txt
mkdir -p $tmpdir

for x in train dev test; do
  mkdir -p data/$x
  cp $srcdir/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp $srcdir/$x.text data/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/$x/spk2utt $srcdir/$x.spk2gender > data/$x/spk2gender || exit 1;
  cp $srcdir/${x}.stm data/$x/stm
  cp $srcdir/${x}.glm data/$x/glm
  ${KALDI_TIMIT_PATH}/utils/validate_data_dir.sh --no-feats data/$x || exit 1;
done

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

for lm_suffix in bg; do
  test=${KALDI_TIMIT_PATH}/data/lang_test_${lm_suffix}
  mkdir -p $test
  cp -r ${KALDI_TIMIT_PATH}/data/lang/* $test

  gunzip -c $lmdir/lm_phone_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst
  fstisstochastic $test/G.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

 # Everything below is only for diagnostic.
 # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
 # this might cause determinization failure of CLG.
 # #0 is treated as an empty word.
  mkdir -p $tmpdir/g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
  fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
   fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
  fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' &&
    echo "Language model has cycles with empty words" && exit 1
  rm -r $tmpdir/g
done

${KALDI_TIMIT_PATH}/utils/validate_lang.pl ${KALDI_TIMIT_PATH}/data/lang_test_bg || exit 1

# for ASR lab 2017
# rm ${KALDI_TIMIT_PATH}/data/train/utt2spk
# rm ${KALDI_TIMIT_PATH}/data/test/utt2spk

echo "Succeeded in formatting data."
rm -r $tmpdir
