import os
import json
from collections import defaultdict

import nltk
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class CaptioningDataset(Dataset):
    def __init__(self, root, caption_file, vocab, split='train', transform=ToTensor()):
        self.vocab = vocab
        self.transform = transform
        self.examples = []
        self.split = split

        self.__prepare_examples(root, caption_file)

        if split != 'train':
            self.references_index = self.__prepare_reference_index()
        else:
            self.references_index = None

    def __getitem__(self, index):
        example = self.examples[index]

        image = Image.open(example['image_path']).convert('RGB')
        image = self.transform(image)

        return example['img_id'], image, example['tokens']

    def __len__(self):
        return len(self.examples)

    def fetch_references(self, img_ids):
        if self.references_index is None:
            self.references_index = self.__prepare_reference_index()

        return [self.references_index[id] for id in img_ids]

    def fetch_sentences(self):
        index = defaultdict(list)

        for example in self.examples:
            sentence = example['sentence']
            index[example['img_id']].append(sentence)

        return index
        
    ##new:获取img_name
    def fetch_imgname(self, img_id):
    
        filepaths = []
    
        for id in img_id:
            print(id)
            for example in self.examples:
                if example['img_id'] == id:
                    filepath = example['image_path']
                    break
            print(filepath)
            filepaths.append(filepath)
        return filepaths
    ###

    def __prepare_examples(self, root, caption_file):
        tokenizer = nltk.tokenize.TreebankWordTokenizer()

        for image_object in self.__image_objects(caption_file):
            if image_object['split'] != self.split:
                continue

            for sentence_object in image_object['sentences']:
                example = self.__build_example(image_object, sentence_object, root, tokenizer)
                self.examples.append(example)

    def __build_example(self, image_object, sentence_object, root, tokenizer):
        sentence = sentence_object['raw']
        tokens = self.__tokenize_sentence(sentence, tokenizer)

        full_image_path = os.path.join(root, image_object['filename'])

        return {
            'img_id': image_object['imgid'],
            'sentence_id': sentence_object['sentid'],
            'sentence': sentence,
            'image_path': full_image_path,
            'tokens': tokens
        }

    def __tokenize_sentence(self, sentence, tokenizer):
        tokens = tokenizer.tokenize(sentence.lower())

        word_ids = []
        word_ids.append(self.vocab('<start>'))
        word_ids.extend([self.vocab(token) for token in tokens])
        word_ids.append(self.vocab('<end>'))

        return torch.Tensor(word_ids).long()

    def __prepare_reference_index(self):
        index = defaultdict(list)

        for example in self.examples:
            tokens = example['tokens'][1:-1] # without <start> and <end>
            index[example['img_id']].append(tokens.tolist())

        return index

    @staticmethod
    def __image_objects(caption_file):
        with open(caption_file, 'r') as file:
            json_contents = json.loads(file.read())['images']

        return tqdm(json_contents, desc='Loading Captioning Dataset contents file')
