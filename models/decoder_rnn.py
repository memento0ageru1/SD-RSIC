import torch
from torch.nn import Module, Embedding, Dropout, LSTMCell, Linear, Sigmoid, Parameter
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = Embedding(vocab_size, embed_dim).to('cuda', non_blocking=True)  # embedding layer 
        self.dropout = Dropout(p=self.dropout).to('cuda', non_blocking=True)

        self.decode_step = LSTMCell(embed_dim * 2, decoder_dim, bias=True).to('cuda', non_blocking=True) #
        self.weight_step = LSTMCell(embed_dim, 1, bias=True).to('cuda', non_blocking=True)
        self.weight_init_hidden = Linear(embed_dim, 1).to('cuda', non_blocking=True)
        self.weight_init_cell = Linear(embed_dim, 1).to('cuda', non_blocking=True)
        self.sigmoid = Sigmoid().to('cuda', non_blocking=True)
        self.classifier = Linear(decoder_dim, vocab_size).to('cuda', non_blocking=True)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = Parameter(embeddings)

    def init_weight_hidden_state(self, encoder_out):
        hidden_state = self.weight_init_hidden(encoder_out) 
        cell_state = self.weight_init_cell(encoder_out)
        return hidden_state, cell_state

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        embeddings = self.embedding(encoded_captions) 

        hidden_state = encoder_out
        cell_state = encoder_out
        weight_hidden_state, weight_cell_state = self.init_weight_hidden_state(encoder_out)
      
        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_lengths = max(decode_lengths)
            
        predictions, weights = self.__init_result_tensors(
            batch_size, max_decode_lengths
        )
       
        for step, batch_size_t, step_embeddings in self.__time_steps(max_decode_lengths, decode_lengths, embeddings):
            weight_hidden_state, weight_cell_state = self.weight_step(
                encoder_out[:batch_size_t],
                (weight_hidden_state[:batch_size_t], weight_cell_state[:batch_size_t])
            )
            decode_features = torch.cat([step_embeddings, encoder_out[:batch_size_t]], dim=1)

            hidden_state, cell_state = self.decode_step(
                decode_features,
                (hidden_state[:batch_size_t], cell_state[:batch_size_t])
            )

            preds = self.classifier(self.dropout(hidden_state))
            predictions[:batch_size_t, step, :] = preds

            weight = self.sigmoid(weight_hidden_state)
            replicated_weight = torch.stack([weight] * self.vocab_size, dim=1)
            weights[:batch_size_t, step, :] = replicated_weight.squeeze().to(DEVICE)

        return predictions, decode_lengths, weights

    def __init_result_tensors(self, batch_size, max_caption_length):
        predictions = torch.zeros(batch_size, max_caption_length, self.vocab_size)
        weights = torch.zeros(batch_size, max_caption_length, self.vocab_size)
        return predictions.to(DEVICE), weights.to(DEVICE)

    @staticmethod
    def __time_steps(max_decode_lengths, decode_lengths, embeddings):
        for step in range(max_decode_lengths):
            batch_size = sum([length > step for length in decode_lengths])

            step_embeddings = embeddings[:batch_size, step, :]

            yield step, batch_size, step_embeddings
