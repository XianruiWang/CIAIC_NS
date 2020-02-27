fs = 16000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
num_spks = 2

# network configure
nnet_conf = {
    "L": 40,
    "N": 512,
    "X": 8,
    "R": 4,
    "B": 512,
    "H": 512,
    "P": 3,
    "norm": "BN",
    "num_spks": num_spks,
    "non_linear": "relu"
}

# data configure:
train_dir = "workspace/Audios/train_scp/"
dev_dir = "workspace/Audios/cv_scp/"

# local data in kiwi
# train_dir = "/data/users/ningning/conv-tasnet-16k/Audios/Audios/train/"
# dev_dir = "/data/users/ningning/conv-tasnet-16k/Audios/Audios/cv/"

train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    [train_dir + "clean.scp",train_dir + "noise.scp" ],
    "sample_rate":
    fs,
}

dev_data = {
    "mix_scp": dev_dir + "mix.scp",
    "ref_scp":
    [dev_dir + "clean.scp",dev_dir + "noise.scp" ],
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200  # batch number
}
