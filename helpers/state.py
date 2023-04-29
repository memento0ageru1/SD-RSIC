import os
import torch

from models import EncoderCNN, DecoderRNN
from .checkpoint import Checkpoint

class State():
    def __init__(self, epoch, epochs_since_improvement, bleu4, decoder,
                 decoder_optimizer, encoder, encoder_optimizer):
        self.epoch = epoch
        self.epochs_since_improvement = epochs_since_improvement
        self.bleu4 = bleu4
        self.decoder = decoder
        self.decoder_optimizer = decoder_optimizer
        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer

class StateFactory():
    @staticmethod
    def create(config, vocab_size, best=False):
        if config.checkpoint == 'auto':
            return StateFactory.autodetect_checkpoint(config, vocab_size, best)

        if config.checkpoint is not None:
            return State(**Checkpoint.load(config, config.checkpoint))

        return StateFactory.build(config, vocab_size)

    @staticmethod
    def autodetect_checkpoint(config, vocab_size, best):
        file_name = 'checkpoint_' + config.exp_name + '.pth.tar'
        if best:
            file_name = 'best_' + file_name

        expected_path = os.path.join(config.checkpoints_path, file_name)
        expected_path = os.path.abspath(expected_path)

        if os.path.isfile(expected_path):
            return State(**Checkpoint.load(config, expected_path))

        return StateFactory.build(config, vocab_size)

    @staticmethod
    def build(config, vocab_size):
        """Builds initial state"""
        decoder = DecoderRNN(
            embed_dim=config.emb_dim,
            decoder_dim=config.decoder_dim,
            vocab_size=vocab_size,
            dropout=0.5
        )

        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad, decoder.parameters()),
            lr=config.decoder_lr
        )

        encoder = EncoderCNN(config.emb_dim, config.embed_model)
        decoder = decoder.to(config.device)
        encoder = encoder.to(config.device)
        return State(
            epoch=0,
            epochs_since_improvement=0,
            bleu4=0.,
            decoder=decoder,
            decoder_optimizer=decoder_optimizer,
            encoder=encoder,
            encoder_optimizer=None
        )
