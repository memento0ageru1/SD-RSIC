import os
import torch

class Checkpoint():
    @staticmethod
    def load(config, path):
        if not path[0] == '/':
            path = os.path.join(config.checkpoints_path, path)

        data = torch.load(path)
        data['epoch'] += 1

        return data

    def __init__(self, config, state):
        self.config = config
        self.state = state

    def state_dict(self):
        return {
            'epoch': self.state.epoch,
            'epochs_since_improvement': self.state.epochs_since_improvement,
            'bleu4': self.state.bleu4,
            'decoder': self.state.decoder,
            'decoder_optimizer': self.state.decoder_optimizer,
            'encoder': self.state.encoder,
            'encoder_optimizer': self.state.encoder_optimizer
        }

    def save(self, is_best=False):
        file_name = 'checkpoint_' + self.config.exp_name + '.pth.tar'
        torch.save(self.state_dict(), self.__build_full_path(file_name))

        if is_best:
            file_name = 'best_' + file_name
            torch.save(self.state_dict(), self.__build_full_path(file_name))

    def __build_full_path(self, file_name):
        return os.path.join(self.config.checkpoints_path, file_name)
