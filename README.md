# CIAIC_NS
CIAIC_NoiseSuppression
Single channel noise reduction based on TCN.

Users are limited to members in CIAIC.

Requirements: 
python3

torch==1.0.0

tqdm==4.26.0

numpy==1.15.2

scipy==1.1.0

librosa

soundfile

##################################################################

Step 1: PrepareData

python3 prepare_data.py

###################################################################

Step 2: Prepare scp file for training, cross validation and test

bash get_train_cv_scp

bash get_test_data_scp

###################################################################

Step 3: Training

bash train_parallel.sh GPUid model_save_DIR

e.g. bash train_parallel.sh 0,1 MyModel

Best trained model and log files will be saved in MyModel.

###################################################################

Step 4: Test

bash inference.sh
