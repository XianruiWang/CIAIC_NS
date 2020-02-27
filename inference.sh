#!/usr/bin/env bash

cpt_dir=workspace/conv_tasnet/20191209 # directory to the well-trained model
# scp_dir=../mixture2clean_dnn/workspace2/Audios/test/Woman_story # directory to test set
scp_dir=../conv-tasnet/workspace/Audios/test/TIMIT
# dump_dir=workspace/Audios_Enh/conv_tasnet/Woman_story # directory to dump enhanced audios
dump_dir=workspace/Audios_Enh/conv_tasnet/TIMIT_PP_Case3 # directory to dump enhanced audios
if [ ! -d $dump_dir ]; then
    mkdir -p $dump_dir
fi


noise_names=('102' '109' '110' '112' '114' '174' '177' '412' '295' '406' '409' '400' '300' '325' '250' 'babble' 'buccaneer1' 'buccaneer2' 'destroyerengine' 'destroyerops' 'f16' 'factory1' 'factory2' 'hfchannel' 'leopard' 'm109' 'machinegun' 'pink' 'volvo' 'white' 'modulate_white')
SNR=('-5db' '0db' '5db' '10db' '15db')
# SNR=('5db' '10db' '15db')

for noise_na in ${noise_names[@]}
do
    for snr in ${SNR[@]}
    do
        scp_path=$scp_dir/$noise_na/$snr/mix.scp
        dump_path=$dump_dir/$noise_na/$snr
        if [ ! -d $dump_path ]; then
            mkdir -p $dump_path
        fi
        (
        ./nnet/separate.py --checkpoint $cpt_dir --input $scp_path --gpu 2 --dump-dir $dump_path --fs 16000 \
        > $dump_path/separate.log 2>&1 
        )
    done
done
