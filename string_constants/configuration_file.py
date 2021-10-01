"""
    # Data Preprocess Arguments
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--noise_path', type=str, default='./noise', help="noise dataset directory")
data_arg.add_argument('--train_path', type=str, default='./train_tisv', help="train dataset directory")
data_arg.add_argument('--test_path', type=str, default='./test_tisv', help="test dataset directory")
data_arg.add_argument('--tdsv', type=str2bool, default=False, help="text dependent or not")
data_arg.add_argument('--sr', type=int, default=8000, help="sampling rate")
data_arg.add_argument('--nfft', type=int, default=512, help="fft kernel size")
data_arg.add_argument('--window', type=int, default=0.025, help="window length (ms)")
data_arg.add_argument('--hop', type=int, default=0.01, help="hop size (ms)")
data_arg.add_argument('--tdsv_frame', type=int, default=80, help="frame number of utterance of tdsv")
data_arg.add_argument('--tisv_frame', type=int, default=180, help="max frame number of utterances of tdsv")

# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=128, help="hidden state dimension of lstm")
model_arg.add_argument('--proj', type=int, default=64, help="projection dimension of lstm")
model_arg.add_argument('--num_layer', type=int, default=3, help="number of lstm layers")
model_arg.add_argument('--restore', type=str2bool, default=False, help="restore model or not")
model_arg.add_argument('--model_path', type=str, default='./tisv_model', help="model directory to save or load")
model_arg.add_argument('--model_num', type=int, default=6, help="number of ckpt file to load")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=False, help="train session or not(test session)")
train_arg.add_argument('--N', type=int, default=4, help="number of speakers of batch")
train_arg.add_argument('--M', type=int, default=5, help="number of utterances per speaker")
train_arg.add_argument('--noise_filenum', type=int, default=16, help="how many noise files will you use")
train_arg.add_argument('--loss', type=str, default='softmax', help="loss type (softmax or contrast)")
train_arg.add_argument('--optim', type=str.lower, default='sgd', help="optimizer type")
train_arg.add_argument('--lr', type=float, default=1e-2, help="learning rate")
train_arg.add_argument('--beta1', type=float, default=0.5, help="beta1")
train_arg.add_argument('--beta2', type=float, default=0.9, help="beta2")
train_arg.add_argument('--iteration', type=int, default=100000, help="max iteration")
train_arg.add_argument('--comment', type=str, default='', help="any comment")
"""

config = {
    # audio parameters
    "sr": 8000,
    "nfft": 512,
    "window": 0.025,
    "hop": 0.01,
    "tdsv_frame": 80,
    "tisv_frame": 180,
    "n_mels": 40,

    # model build
    "hidden": 128,
    "embed_dim": 64,
    "num_layer": 3,
    "restore": False,
    "model_path": "models/tisv_model",
    "model_num": 6,

    # model training
    "train": True,
    "speaker_num": 4,
    "utter_num": 5,
    "noise_filenum": 16,
    "loss": "softmax",
    "optim": "sgd",
    "lr": 1e-2,
    "beta1": 0.5,
    "beta2": 0.9,
    "iteration": 100000,
    "comment": "",
    "train_path": "C:\\gkv\\My Stuff\\ML_AI\\Voice_Encoder\\Resemblyzer\\audio_data\\spectrograms\\train",

    # general
    "small_error": 1e-6,
}
