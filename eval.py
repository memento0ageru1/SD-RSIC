"""Calculate BLEU scores for test split"""

import argparse
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from helpers import (
    ConfigFactory, Vocabulary, StateFactory, Tester
)


def ids_to_sentence(vocab, ids):
    blacklist = list(map(vocab, ['<start>', '<pad>', '<end>']))
    return ' '.join([vocab.id2word(id) for id in ids if id not in blacklist])

def evaluate(config_file):
    config = ConfigFactory.fetch(config_file)

    vocab = Vocabulary.load(config.vocabulary_path, config.vocabulary_size)
    state = StateFactory.create(config, len(vocab), best=True)
    tester = Tester(config, state, vocab, split='test')

    references_index = tester.dataset.fetch_sentences()

    img_ids, img_paths, predictions, references = tester.step(beam_size=4, return_results=True)

    weights = [
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (0.33, 0.33, 0.33, 0.0),
        (0.25, 0.25, 0.25, 0.25)
    ]

    smoothing_function = SmoothingFunction().method1

    bleu_scores = [
        corpus_bleu(references, predictions, weights=weight, smoothing_function=smoothing_function)
        for weight in weights
    ]

    results = []

    for img_id, img_path, prediction, tokenized_ref in zip(img_ids, img_paths, predictions, references):
        img_bleus = [
            corpus_bleu([tokenized_ref], [prediction],
                        weights=weight, smoothing_function=smoothing_function)
            for weight in weights
        ]

        results.append({
            'img_id': img_id,
            'img_path': img_path,
            'prediction': ids_to_sentence(vocab, prediction),
            'bleu_scores': img_bleus,
            'references': references_index[img_id]
        })

    result = {
        'bleu': bleu_scores,
        'results': results
    }
    test_result_file = config.results_file.split('.json')[0] + '_test.json'
    with open(test_result_file, 'w+') as file:
        file.write(json.dumps(result))
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('config', nargs='?', default='config.json',
                        help='json config file')

    args = parser.parse_args()
    evaluate(args.config)