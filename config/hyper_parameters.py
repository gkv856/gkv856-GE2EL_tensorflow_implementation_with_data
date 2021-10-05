import torch


def get_hyper_parameters():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hparam_dict = {

        # genereal parameters
        "is_training_mode": True,
        "device": device,

        # data:
        "raw_data": 'C:\\gkv\\My Stuff\\ML_AI\\GE2EL_tf\\audio_data\\raw\\librispeech_test-other',
        "train_spectrogram_path": "C:\\gkv\\My Stuff\\ML_AI\\GE2EL_tf\\audio_data\\spectrograms\\train",
        "test_spectrogram_path": "C:\\gkv\\My Stuff\\ML_AI\GE2EL_tf\\audio_data\\spectrograms\\test",
        # Model path for testing, inference, or resuming training
        "model_path": "C:\\gkv\\My Stuff\\ML_AI\\GE2EL_tf\\static\\pre_trained_models\\",
        "log_file": "C:\\gkv\\My Stuff\\ML_AI\\GE2EL_tf\\static\\logs\\log.txt",
        "checkpoint_dir": "C:\\gkv\\My Stuff\\ML_AI\\GE2EL_tf\\static\\chk_pts\\",

        "is_data_preprocessed": True,
        "sr": 16000,

        # For mel spectrogram preprocess
        "nfft": 512,
        "window": 0.025,  # (s)
        "hop": 0.01,  # (s)
        "n_mels": 40,  # Number of mel energies
        "tisv_frame": 180,  # Max number of time steps in input after preprocess

        # model hyper parameters
        "hidden": 768,  # Number of LSTM hidden layer units
        "num_layer": 3,  # Number of LSTM layers
        "proj": 256,  # Embedding size

        # train:
        "training_N": 4,  # Number of speakers in batch
        "training_M": 5,  # Number of utterances per speaker
        "training_num_workers": 0,  # number of workers for dataloader
        "lr": 0.01,
        "training_epochs": 950,  # Max training speaker epoch
        "log_interval": 30,  # Epochs before printing progress
        "checkpoint_interval": 120,  # Save model after x speaker epochs
        "restore_existing_model": False,  # Resume training from previous model path
        "verbose": True,

        # test:
        "test_N": 4,  # Number of speakers in batch
        "test_M": 6,  # Number of utterances per speaker
        "test_num_workers": 8,  # number of workers for data laoder
        "test_epochs": 10,  # testing speaker epochs

        # small error
        "small_err": 1e-6,

    }
    return hparam_dict


class Dict_with_dot_notation(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dict_with_dot_notation(value)
            self[key] = value


class Hyper_parameters(Dict_with_dot_notation):

    def __init__(self, hp_dict=None):
        super(Dict_with_dot_notation, self).__init__()

        if hp_dict is None:
            hp_dict = get_hyper_parameters()

        hp_dotdict = Dict_with_dot_notation(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dict_with_dot_notation.__getitem__
    __setattr__ = Dict_with_dot_notation.__setitem__
    __delattr__ = Dict_with_dot_notation.__delitem__


hyper_paramaters = Hyper_parameters()
