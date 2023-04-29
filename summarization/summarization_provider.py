import sys
import numpy
import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import Sequential
from summarization.models import Model, HIDDEN_DIM
from helpers import AverageMeter


def get_enc_data(batch, config):
    batch_size = len(batch.enc_lens)
    enc_batch = torch.from_numpy(batch.enc_batch).long()
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float()

    enc_lens = batch.enc_lens

    ct_e = torch.zeros(batch_size, 2 * HIDDEN_DIM)

    enc_batch = enc_batch.to(config.device)
    enc_padding_mask = enc_padding_mask.to(config.device)

    ct_e = ct_e.to(config.device)

    enc_batch_extend_vocab = None
    if batch.enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long()
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(config.device)

    extra_zeros = None
    if batch.max_art_oovs > 0:
        extra_zeros = torch.zeros(batch_size, batch.max_art_oovs)
        extra_zeros = extra_zeros.to(config.device)

    return enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e

class Evaluate():
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.__setup_model()

    @torch.no_grad()
    def evaluate_batch(self, batched_examples):
        start_id = self.vocab('<start>')
        unk_id = self.vocab('<unk>')

        (
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab,
            extra_zeros, ct_e
        ) = get_enc_data(batched_examples, self.config)

        enc_batch = self.model.embeds(enc_batch)

        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
        x_t = torch.Tensor(len(enc_out)).fill_(start_id).long().to(self.config.device)
        s_t = (enc_hidden[0], enc_hidden[1])
        sum_temporal_srcs = None
        prev_s = None

        distributions = []
        mask = torch.Tensor(len(enc_out)).fill_(1).long().to(self.config.device)

        decoder_macs = 0.
        summarization_max_dec_steps = 7
        for _ in range(summarization_max_dec_steps):
            x_t = self.model.embeds(x_t)

            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(
                x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros,
                enc_batch_extend_vocab, sum_temporal_srcs, prev_s
            )

            distributions.append(final_dist[:, :self.vocab_size])

            multi_dist = Categorical(final_dist)
            x_t = multi_dist.sample()

            is_oov = (x_t >= self.vocab_size).long()
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * unk_id

        del x_t, s_t, ct_e, sum_temporal_srcs, prev_s, enc_out, enc_hidden, mask
        return torch.stack(distributions, dim=1)

    def __setup_model(self):
        self.model = Model(self.config, len(self.vocab))
        self.model = self.model.to(self.config.device)
        checkpoint = torch.load(
            self.config.summarization_model_path,
            map_location=self.config.device
        )

        self.model.load_state_dict(checkpoint["model_dict"])

class Example(object):
    def __init__(self, config, article, vocab):
        self.vocab = vocab
        summarization_max_enc_steps = 55
        article_words = article.split()
        if len(article_words) > summarization_max_enc_steps:
            article_words = article_words[:summarization_max_enc_steps]

        self.enc_len = len(article_words)

        self.enc_input = [vocab(word) for word in article_words]

        self.enc_input_extend_vocab, self.article_oovs = self.__article2ids(article_words)

        self.original_article = article

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)

    def __article2ids(self, article_words):
        ids = []
        oovs = []
        unk_id = self.vocab('<unk>')
        for word in article_words:
            i = self.vocab.word2id(word)

            if i == unk_id:
                if word not in oovs:
                    oovs.append(word)
                oov_num = oovs.index(word)
                ids.append(len(self.vocab) + oov_num)
            else:
                ids.append(i)

        return ids, oovs


class Batch(object):
    def __init__(self, config, example_list, vocab):
        self.config = config
        self.batch_size = len(example_list)
        self.pad_id = vocab.word2id('<pad>')

        self.init_encoder_seq(example_list)

    def init_encoder_seq(self, example_list):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for example in example_list:
            example.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = numpy.zeros((self.batch_size, max_enc_seq_len), dtype=numpy.int32)
        self.enc_lens = numpy.zeros((self.batch_size), dtype=numpy.int32)
        self.enc_padding_mask = numpy.zeros((self.batch_size, max_enc_seq_len), dtype=numpy.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        self.art_oovs = [ex.article_oovs for ex in example_list]
        self.enc_batch_extend_vocab = numpy.zeros(
            (self.batch_size, max_enc_seq_len),
            dtype=numpy.int32
        )

        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

class SummarizationProvider():
    def __init__(self, config, rsicd_dataset, vocab, verbose=True):
        self.config = config
        self.rsicd_dataset = rsicd_dataset
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.verbose = verbose

        self.eps = 1 / len(vocab)

        self.__index_images()

    def __call__(self, img_ids, lengths):
        batch_size = len(img_ids)
        max_length = lengths[0]

        all_probs = torch.zeros(batch_size, max_length, self.vocab_size)

        for idx, img_id in enumerate(img_ids):
            probabilities = self.image_index[img_id].to_dense()
            target_length = min([max_length, probabilities.shape[0]])           
            all_probs[idx, :target_length, :] = probabilities[:target_length, :]

        return pack_padded_sequence(all_probs, lengths, batch_first=True)

    def __index_images(self):
        results = self.__get_sorted_examples()

        eval_processor = Evaluate(self.config, self.vocab)
        generated_output = []

        for batch in self.__batches(results):
            batch_output = eval_processor.evaluate_batch(batch)
            batch_output = map(self.__normalize_output, batch_output)

            generated_output.extend(batch_output)

        index = [(results[index][0], generated_output[index]) for index in range(len(results))]
        self.image_index = dict(index)

        del eval_processor
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()

    def __normalize_output(self, probabilities):
        mask = (probabilities > self.eps).float()
        probabilities = probabilities * mask

        return probabilities.to_sparse().cpu()

    def __batches(self, results):
        batch_size = 1000
        current_slice = 0

        while current_slice * batch_size < len(results):
            start_index = current_slice * batch_size
            end_index = start_index + batch_size

            if self.verbose:
                sys.stdout.write(
                    "\r\033[K Processing elements {} to {}...".format(start_index, end_index)
                )
                sys.stdout.flush()

            examples = results[start_index:end_index]
            batch = Batch(self.config, [x[1] for x in examples], self.vocab)

            yield batch
            del batch

            current_slice += 1

        if self.verbose:
            print(" Done!")

    def __get_sorted_examples(self):
        img_ids = []
        examples = []

        for img_id, annotations in self.rsicd_dataset.fetch_sentences().items():
            img_ids.append(img_id)
            article = ' '.join(annotations)

            examples.append(Example(self.config, article, self.vocab))

        results = list(zip(img_ids, examples))
        results.sort(key=lambda x: -x[1].enc_len)

        return results
