class Vocabulary():
    @staticmethod
    def load(path, size):
        with open(path, 'r') as file:
            lines = file.readlines()

        vocab = Vocabulary()

        vocab.add_word('<unk>')
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')

        index = 0

        while index < len(lines) and len(vocab) != size:
            token, _count = lines[index].split(' ')
            vocab.add_word(token)

            index += 1

        return vocab

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def word2id(self, word):
        return self.__call__(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
