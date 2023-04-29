import os
import json
import argparse
from helpers import (
    ConfigFactory, Vocabulary, StateFactory, Trainer, BeamValidator,
    Checkpoint, adjust_learning_rate
)

def save_results(config, epoch, train_loss, bleu4):
    contents = []

    if os.path.isfile(config.results_file):
        with open(config.results_file, 'r') as file:
            contents = json.loads(file.read())

    contents.append({
        'epoch': epoch,
        'experiment': config.exp_name,
        'train_loss': train_loss.avg,
        'bleu4': bleu4
    })

    with open(config.results_file, 'w+') as file:
        file.write(json.dumps(contents))

def train(config_file):
    config = ConfigFactory.fetch(config_file)

    vocab = Vocabulary.load(config.vocabulary_path, config.vocabulary_size)

    state = StateFactory.create(config, len(vocab))

    trainer = Trainer(config, state, vocab)
    validator = BeamValidator(config, state, vocab)

    for epoch in range(state.epoch, config.epochs):
        state.epoch = epoch

        if state.epochs_since_improvement == 20 and state.epoch < 30:
            break
        if state.epochs_since_improvement > 0 and state.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(state.decoder_optimizer, 0.8)

        trainer.step()
        current_bleu4 = validator.step()

        is_best = current_bleu4 > state.bleu4
        state.bleu4 = max(current_bleu4, state.bleu4)

        if not is_best:
            state.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (state.epochs_since_improvement,))
        else:
            state.epochs_since_improvement = 0

        Checkpoint(config, state).save(is_best)
        save_results(config, epoch, trainer.tracker['losses'], current_bleu4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('config', nargs='?', default='config.json',
                        help='json config file')

    args = parser.parse_args()
    train(args.config)
