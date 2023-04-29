from helpers import Tester

class BeamValidator(Tester):
    def __init__(self, config, state, vocab):
        super().__init__(
            config=config,
            state=state,
            vocab=vocab,
            split='val',
            bleu_weights=(0.25, 0.5, 0.15, 0.10)
        )

    def step(self):
        return super(BeamValidator, self).step(
            beam_size=4,
            return_results=False
        )
