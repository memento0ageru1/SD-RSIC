import time
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import CaptioningDataset, CaptioningDataLoader
from helpers import AverageMeter, accuracy

class Validator():
    def __init__(self, config, state, vocab):
        self.config = config
        self.state = state
        self.vocab = vocab

        self.dataset, self.val_loader = self.__build_val_loader()
        self.criterion = CrossEntropyLoss().to(config.device, non_blocking=True)

        self.tracker = None

    @torch.no_grad()
    def step(self):
        self.__initialize_tracker()

        self.state.decoder.eval()
        self.state.encoder.eval()

        references = [] 
        hypotheses = []

        token_blacklist = map(self.vocab, ['<start>', '<end>', '<pad>'])

        for batch_idx, (img_ids, imgs, caps, caplens) in enumerate(self.val_loader):
            imgs = imgs.to(self.config.device, non_blocking=True)
            caps = caps.to(self.config.device, non_blocking=True)
            caplens = caplens.to(self.config.device, non_blocking=True)

            imgs = self.state.encoder(imgs)
            scores, decode_lengths, weights = self.state.decoder(imgs, caps, caplens)
            targets = caps[:, 1:]

            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = self.criterion(scores, targets)

            top5 = accuracy(scores, targets, 5)

            self.__track_loss_updated(loss.item(), sum(decode_lengths))
            self.__track_accuracy_updated(top5, sum(decode_lengths))
            self.__track_batch_finished(batch_idx)

            references.extend(self.dataset.fetch_references(img_ids))

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = []
            for index, prediction in enumerate(preds):
                current_prediction = prediction[:decode_lengths[index]]
                current_prediction = [
                    token for token in current_prediction if token not in token_blacklist
                ]
                temp_preds.append(current_prediction)

            hypotheses.extend(temp_preds)
            del loss
            del scores

        smoothing_function = SmoothingFunction().method1
        bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)

        template = '\n * LOSS - {loss.avg:.3f}, '\
                   'TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'
        print(template.format(
            loss=self.tracker['losses'],
            top5=self.tracker['top5accs'],
            bleu=bleu4
        ))

        return bleu4

    def __build_val_loader(self):
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(224,interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])

        dataset = CaptioningDataset(
            self.config.images_dir,
            self.config.captions_file,
            self.vocab,
            split='val',
            transform=transform
        )

        return dataset, CaptioningDataLoader(dataset, batch_size=self.config.batch_size)

    def __initialize_tracker(self):
        self.tracker = {
            'start': time.time(),
            'batch_time': AverageMeter(),
            'losses': AverageMeter(),
            'top5accs': AverageMeter()
        }

    def __track_loss_updated(self, loss, count):
        self.tracker['losses'].update(loss, count)

    def __track_accuracy_updated(self, top5accuracy, count):
        self.tracker['top5accs'].update(top5accuracy, count)

    def __track_batch_finished(self, batch_index):
        self.tracker['batch_time'].update(time.time() - self.tracker['start'])
        self.tracker['start'] = time.time()

        if batch_index % self.config.print_freq == 0:
            template = 'Validation: [{0}/{1}]\t' \
                       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'

            print(template.format(
                batch_index,
                len(self.val_loader),
                batch_time=self.tracker['batch_time'],
                loss=self.tracker['losses'],
                top5=self.tracker['top5accs']
            ))
