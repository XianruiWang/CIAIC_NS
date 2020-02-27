#!/usr/bin/bash


dir_root='/home/ningpan/work/SpeechSeperation/2019APSIPA/conv-tasnet/workspace/Audios'
dir_train_audio=$dir_root/train # directory of training data
dir_train_scp=$dir_root/train_scp # directory for training scp file
dir_cv_scp=$dir_root/cv_scp # directory for cross validation scp file

dir_names=("mix" "clean" "noise")

if [ -d "$dir_train_scp" ]; then
  rm -rf $dir_train_scp
fi
if [ -d "$dir_cv_scp" ]; then
  rm -rf $dir_cv_scp
fi

mkdir $dir_train_scp
mkdir $dir_cv_scp

audio_list=$(ls $dir_train_audio/mix)

(
echo "creating scp file for training and cross validation"
cnt=0
for nn in $audio_list; do
    let "cnt+=1"
    let "flag=$cnt%10" 
    if [ $flag -eq 0 ] # 9:1 training data: cv data
    then
        echo "cv scp $cnt corresponding wav is $nn"
        echo "${nn::-4} and $nn"
        for dir_na in ${dir_names[@]}; do
            echo ${nn::-4} $dir_train_audio/$dir_na/$nn >> "$dir_cv_scp/$dir_na.scp"
        done
    else
        echo "train scp $cnt corresponding wav is $nn"
        for dir_na in ${dir_names[@]}; do
            echo ${nn::-4} $dir_train_audio/$dir_na/$nn >> "$dir_train_scp/$dir_na.scp"
        done
    fi
done
)|| exit 1
