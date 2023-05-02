import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class CaptioningDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=True, num_workers=2):
        self.pad_value = dataset.vocab('<pad>')
        super(CaptioningDataLoader,self).__init__(dataset=dataset, pin_memory=True, batch_size=batch_size, shuffle=shuffle, num_workers=10, collate_fn=self.__collate_fn)
    ##original dataloader num_workers=2

    def __collate_fn(self, data):
        data.sort(key=lambda x: len(x[2]), reverse=True)
        img_ids, images, captions = zip(*data)

        images = torch.stack(images, 0)

        lengths = torch.Tensor([len(caption) for caption in captions]).long()
        targets = pad_sequence(captions, batch_first=True, padding_value=self.pad_value)

        return img_ids, images, targets, lengths
