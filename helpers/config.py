from collections import namedtuple
import json
import torch

DEFAULTS = {
    'exp_name': 'SD-RSIC',
    'images_dir': 'data/RSICD/images',
    'captions_file': 'data/RSICD/dataset.json',
    'vocabulary_path': 'vocab.txt',
    'vocabulary_size': 50000,
    'emb_dim': 512,
    'decoder_dim': 512,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'embed_model': 'resnet152',
    'epochs': 150,
    'batch_size': 512,
    'decoder_lr': 8e-5,
    'print_freq': 100,
    'checkpoint': 'auto',
    'checkpoints_path': 'checkpoints',
    'results_file': 'results.json',
    'summarization_model_path': 'summarization.tar',
}

Config = namedtuple('Config', list(DEFAULTS.keys()))

class ConfigFactory():
    @staticmethod
    def fetch(config_file='config.json'):
        config = Config(**DEFAULTS)
        with open(config_file, 'r') as file:
            contents = json.loads(file.read())

        config = config._replace(**contents)
        return config
