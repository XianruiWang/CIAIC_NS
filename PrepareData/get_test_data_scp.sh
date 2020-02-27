#!/usr/bin/bash

# dir_root='/home/ningpan/work/SpeechSeperation/2019APSIPA/conv-tasnet/workspace/Audios/test/TIMIT'
# dir_root='/home/ningpan/work/SpeechSeperation/2019APSIPA/conv-tasnet/workspace/Audios/test/MRT
dir_root='/home/ningpan/work/SpeechSeperation/2019APSIPA/mixture2clean_dnn/workspace2/Audios/test/MRT_mTurk'
# noise_names=('babble' 'buccaneer1' 'buccaneer2' 'destroyerengine' 'destroyerops' 'f16' 'factory1' 'factory2' 'hfchannel' 'leopard' 'm109' 'machinegun' 'pink' 'volvo' 'white' '102' '109' '110' '112' '114' '174' '177' '412' '295' '406' '409' '400' '300' '325' '250')
# SNR=('-5db' '0db' '5db' '10db' '15db')
noise_names=('babble' 'pink')
SNR=('10db')

for noise_na in ${noise_names[@]}
do
    for snr in ${SNR[@]}
    do
        dir=$dir_root/$noise_na/$snr
        (
        for x in mix clean; do
              echo "creating scp file in $dir/$x"
              # find wav names
              for nn in `find  $dir/$x -name "*.wav" | sort -u | xargs -I {} basename {} .wav`; do # `` is equal to $() 
                  echo $nn $dir/$x/$nn.wav >> $dir/wav.scp
              done 
              sort $dir/wav.scp -o $dir/wav.scp
              # rename scp file and move to the directory $dir
              if [ "$x" == "mix" ]
              then
                  mv $dir/wav.scp $dir/mix.scp
              else
                  mv $dir/wav.scp $dir/clean.scp
              fi

              echo "scp file in data/$x has been created"
        done
        )
    done
done


     

