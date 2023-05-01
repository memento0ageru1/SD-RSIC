import torch
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import CaptioningDataset, CaptioningDataLoader

from torch.nn.functional import log_softmax

class Tester(object):
    def __init__(self, config, state, vocab, split='test', bleu_weights=(0.25, 0.25, 0.25, 0.25)):
        self.config = config
        self.state = state
        self.vocab = vocab
        self.split = split
        self.bleu_weights = bleu_weights

        self.dataset, self.test_loader = self.__build_test_loader()

    @torch.no_grad()
    def step(self, beam_size, return_results=False):
        self.state.decoder.eval()
        self.state.encoder.eval()

        predictions = []
        references = []
        all_img_ids = []
        filenames = []

        for img_ids, imgs, _, _ in self.test_loader:
            imgs = imgs.to(self.config.device)

            imgs = self.state.encoder(imgs)

            beam_groups = [
                BeamGroup(self.config, img_id, imgs[idx], beam_size, self.vocab)
                for idx, img_id in enumerate(img_ids)
            ]

            encoder_out = [group.combined_encoder_out() for group in beam_groups]
            encoder_out = torch.cat(encoder_out)

            step = 1
            hidden_state = encoder_out
            cell_state = encoder_out
            
            while step < 50 and any(map(lambda group: group.is_active(), beam_groups)):
                active_beams = [
                    group.beam_sentences()
                    for group in beam_groups if group.is_active()
                ]
                active_beams = torch.cat(active_beams).to(self.config.device)

                encoder_out = [group.combined_encoder_out() for group in beam_groups]
                encoder_out = torch.cat(encoder_out)

                embeddings = self.state.decoder.embedding(active_beams)[:, -1, :]

                decode_features = torch.cat([embeddings, encoder_out], dim=1)

                hidden_state, cell_state = self.state.decoder.decode_step(
                    decode_features, (hidden_state, cell_state)
                )

                scores = self.state.decoder.classifier(hidden_state)
                scores = log_softmax(scores, dim=1)

                splits = [group.active_beams_size() for group in beam_groups]
                splitted_scores = torch.split(scores, splits)

                preserved_indices = []
                current_index = 0

                for group_scores, beam_group in zip(splitted_scores, beam_groups):
                    next_offset = beam_group.active_beams_size()

                    indices = beam_group.step(group_scores)
                    preserved_indices.extend([idx + current_index for idx in indices])

                    current_index += next_offset

                hidden_state = hidden_state[preserved_indices]
                cell_state = cell_state[preserved_indices]

                step += 1

            for beam_group in beam_groups:
                predictions.append(beam_group.best_sentence())

            references.extend(self.dataset.fetch_references(img_ids))
            filenames.append(self.dataset.fetch_imgname(img_ids))
            all_img_ids.extend(img_ids)

        smoothing_function = SmoothingFunction().method1
        bleu4 = corpus_bleu(references, predictions, smoothing_function=smoothing_function,
                            weights=self.bleu_weights)

        if return_results:
            return all_img_ids, filenames, predictions, references

        template = '\n * BEAM SEARCH TESTING - BLEU-4 - {bleu}\n'
        print(template.format(bleu=bleu4))

        return bleu4

    def __build_test_loader(self):
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(224,interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])

        dataset = CaptioningDataset(
            self.config.images_dir,
            self.config.captions_file,
            self.vocab,
            split=self.split,
            transform=transform
        )

        return dataset, CaptioningDataLoader(dataset, batch_size=self.config.batch_size)

class BeamGroup():
    def __init__(self, config, image_id, encoder_out, beam_size, vocab):
        self.config = config
        self.image_id = image_id
        self.encoder_out = encoder_out
        self.beam_size = beam_size
        self.vocab = vocab

        self.end_token = vocab('<end>')

        self.active_beams = self.__initialize_beams()
        self.beam_scores = [0 for _ in range(beam_size)]

        self.finished_sentences = []

    def step(self, scores):
        scores = scores.cpu()
        next_beams = set()

        for beam_index, beam in enumerate(self.active_beams):
            next_scores, next_ids = scores[beam_index].topk(self.beam_size)
            next_ids = next_ids.view(self.beam_size, 1)

            for next_id, next_score in zip(next_ids, next_scores):
                potential_beam = (
                    tuple(torch.cat((beam, next_id)).tolist()),
                    self.beam_scores[beam_index] + next_score.item()
                )

                next_beams.add(potential_beam)

        next_beams = list(next_beams)
        next_beams.sort(key=lambda next_beam: -next_beam[1])
        next_beams = next_beams[:self.active_beams_size()]

        self.active_beams = [
            torch.Tensor(next_beam[0]).long()
            for next_beam in next_beams
        ]

        self.beam_scores = [next_beam[1] for next_beam in next_beams]

        preserved_indices = []

        for idx, beam in enumerate(self.active_beams):
            if not beam[-1].item() == self.end_token:
                preserved_indices.append(idx)

        self.finished_sentences.extend([
            result
            for idx, result in enumerate(zip(self.active_beams, self.beam_scores))
            if idx not in preserved_indices
        ])

        self.active_beams = [
            beam for idx, beam in enumerate(self.active_beams) if idx in preserved_indices
        ]

        return preserved_indices

    def best_sentence(self):
        blacklist = list(map(self.vocab, ['<start>', '<pad>', '<end>']))

        if len(self.finished_sentences) == 0:
            self.finish_all_sentences()

        best_sentence, _ = max(self.finished_sentences, key=lambda x: x[1])
        best_sentence = [idx.item() for idx in best_sentence if idx.item() not in blacklist]

        return best_sentence

    def combined_encoder_out(self):
        return self.encoder_out.unsqueeze(0).expand(self.active_beams_size(),-1)

    def beam_sentences(self):
        if self.is_active():
            return torch.stack(self.active_beams, dim=0)

        return None

    def active_beams_size(self):
        return len(self.active_beams)

    def is_active(self):
        return self.active_beams_size() > 0

    def finish_all_sentences(self):
        self.finished_sentences = list(zip(self.active_beams, self.beam_scores))

        return None

    def __initialize_beams(self):
        return [
            torch.Tensor([self.vocab('<start>')]).long()
            for idx in range(self.beam_size)
        ]
