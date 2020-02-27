#!/usr/bin/env bash
# usage:
# bash train.sh gpuid cpt-id
# e.g. bash train.sh 0 20191019
# or bash train.sh 0,1 20191019

set -eu

cpt_dir=workspace/conv_tasnet
# resume_file=workspace/conv_tasnet/20191027/last.pt.tar
epochs=100
# constrainted by GPU number & memory
batch_size=14 # when using 2 gpus
# batch_size=6 # when using only 1 gpu
cache_size=24

[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1 
# explaination of: [ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1 
# $# denotes for the number of inputs of this .sh script
# $0 denotes for the script name, i.e., train.sh
# $1 denotes for the first input, i.e., gpuid, e.g. 0 or 0,1 
# $2 is the second input, i.e., cpt-id, e.g. a directory for saving files created during training. Better to be date.
#  if the number of input of train.sh is not equal to 2, then echo "script error: ....... "

./nnet/train_parallel.py \
  --gpus $1 \
  --epochs $epochs \
  --batch-size $batch_size \
  --checkpoint $cpt_dir/$2 \
  --num-workers 6 \
  > $2.train.log 2>&1
#   --resume $resume_file \

# explaination of: > $2.train.log 2>&1
# $2 is the second input. This sentence means to write all standard error or standard output to log file named as $2.train.log