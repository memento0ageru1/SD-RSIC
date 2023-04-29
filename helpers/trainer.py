import time

from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import CaptioningDataset, CaptioningDataLoader
from helpers import AverageMeter, clip_gradient, accuracy
from summarization import SummarizationProvider

class Trainer():
    def __init__(self, config, state, vocab):
        self.config = config
        self.state = state
        self.vocab = vocab

        dataset, self.train_loader = self.__build_train_loader()
        self.criterion = CrossEntropyLoss().to(config.device, non_blocking=True)

        self.summarization_provider = SummarizationProvider(config, dataset, vocab)
        
        self.tracker = None

    def step(self):
        self.__initialize_tracker()

        self.state.decoder.train()
        self.state.encoder.train()

        for batch_idx, (img_ids, imgs, caps, caplens) in enumerate(self.train_loader):
            self.__track_data_loaded()
            imgs = imgs.to(self.config.device, non_blocking=True)
            caps = caps.to(self.config.device, non_blocking=True)
            caplens = caplens.to(self.config.device, non_blocking=True)

            imgs = self.state.encoder(imgs)
        
            caption_probs, decode_lengths, weights = self.state.decoder(imgs, caps, caplens)

            targets = caps[:, 1:]

            caption_probs = pack_padded_sequence(caption_probs, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            weights = pack_padded_sequence(weights, decode_lengths, batch_first=True).data
            summarization_probs = self.summarization_provider(img_ids, decode_lengths)
            summarization_probs = summarization_probs.data.to(self.config.device)
            final_probs = (1 - weights) * summarization_probs + weights * caption_probs

            loss = self.criterion(final_probs, targets)

            self.state.decoder_optimizer.zero_grad()
            if self.state.encoder_optimizer is not None:
                self.state.encoder_optimizer.zero_grad()
            loss.backward()

            clip_gradient(self.state.decoder_optimizer, 5.0)
            if self.state.encoder_optimizer is not None:
                clip_gradient(self.state.encoder_optimizer, 5.0)

            self.state.decoder_optimizer.step()
            if self.state.encoder_optimizer is not None:
                self.state.encoder_optimizer.step()

            top5 = accuracy(final_probs, targets, 5)

            self.__track_loss_updated(loss.item(), sum(decode_lengths))
            self.__track_accuracy_updated(top5, sum(decode_lengths))
            self.__track_accuracy_updated(top5, sum(decode_lengths))
            self.__track_batch_finished(batch_idx)
            del loss
            del final_probs

    def __build_train_loader(self):
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
            split='train',
            transform=transform,
        )

        return dataset, CaptioningDataLoader(dataset, batch_size=self.config.batch_size)

    def __initialize_tracker(self):
        self.tracker = {
            'start': time.time(),
            'batch_time': AverageMeter(),
            'data_time': AverageMeter(),
            'losses': AverageMeter(),
            'top5accs': AverageMeter()
            }

    def __track_data_loaded(self):
        self.tracker['data_time'].update(time.time() - self.tracker['start'])

    def __track_loss_updated(self, loss, count):
        self.tracker['losses'].update(loss, count)

    def __track_accuracy_updated(self, top5accuracy, count):
        self.tracker['top5accs'].update(top5accuracy, count)

    def __track_batch_finished(self, batch_index):
        self.tracker['batch_time'].update(time.time() - self.tracker['start'])
        self.tracker['start'] = time.time()

        if batch_index % self.config.print_freq == 0:
            template = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'
            print(template.format(    
                self.state.epoch,
                batch_index,
                len(self.train_loader),
                batch_time=self.tracker['batch_time'],
                data_time=self.tracker['data_time'],
                loss=self.tracker['losses'],
                top5=self.tracker['top5accs']
            ))           
