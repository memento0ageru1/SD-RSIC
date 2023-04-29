import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import relu, softmax

HIDDEN_DIM = 512
EMBED_DIM = 256

def init_lstm_wt(lstm, config):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            weight = getattr(lstm, name)
            weight.data.uniform_(
                -0.02,
                0.02
            )
        elif 'bias' in name:
            bias = getattr(lstm, name)
            bias_size = bias.size(0)
            start, end = bias_size // 4, bias_size // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)

def init_linear_wt(linear, config):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)

def init_wt_normal(wt, config):
    wt.data.normal_(std=1e-4)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.lstm = nn.LSTM(
            EMBED_DIM, HIDDEN_DIM,
            num_layers=1, batch_first=True, bidirectional=True
        )
        init_lstm_wt(self.lstm, config)

        self.reduce_h = nn.Linear(
            HIDDEN_DIM * 2,
            HIDDEN_DIM
        )
        init_linear_wt(self.reduce_h, config)
        self.reduce_c = nn.Linear(
            HIDDEN_DIM * 2,
            HIDDEN_DIM
        )
        init_linear_wt(self.reduce_c, config)

    def forward(self, x, seq_lens):
        enc_out = pack_padded_sequence(x, seq_lens, batch_first=True)
        enc_out, enc_hid = self.lstm(enc_out)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = enc_out.contiguous()                          
        hidden_state, cell_state = enc_hid                      
        hidden_state = torch.cat(list(hidden_state), dim=1)     
        cell_state = torch.cat(list(cell_state), dim=1)
        h_reduced = relu(self.reduce_h(hidden_state))
        c_reduced = relu(self.reduce_c(cell_state))
        return enc_out, (h_reduced, c_reduced)


class EncoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.W_h = nn.Linear(
            HIDDEN_DIM * 2,
            HIDDEN_DIM * 2, bias=False
        )
        self.W_s = nn.Linear(
            HIDDEN_DIM * 2,
            HIDDEN_DIM * 2
        )
        self.v = nn.Linear(
            HIDDEN_DIM * 2,
            1, bias=False
        )


    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
        et = self.W_h(h)                        
        dec_fea = self.W_s(st_hat).unsqueeze(1) 
        et = et + dec_fea
        et = torch.tanh(et)                     
        et = self.v(et).squeeze(2)              

        exp_et = torch.exp(et)
        if sum_temporal_srcs is None:
            et1 = exp_et
            sum_temporal_srcs = torch.Tensor(et.size()).fill_(1e-10)
            sum_temporal_srcs = sum_temporal_srcs.to(self.config.device) + exp_et
        else:
            et1 = exp_et/sum_temporal_srcs 
            sum_temporal_srcs = sum_temporal_srcs + exp_et

        at = et1 * enc_padding_mask
        normalization_factor = at.sum(1, keepdim=True)
        at = at / normalization_factor

        at = at.unsqueeze(1)                    
        ct_e = torch.bmm(at, h)
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)

        return ct_e, at, sum_temporal_srcs

class DecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.W_prev = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM, bias=False
        )
        self.W_s = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM
        )
        self.v = nn.Linear(HIDDEN_DIM, 1, bias=False)

    def forward(self, s_t, prev_s):
        if prev_s is None:
            ct_d = torch.zeros(s_t.size()).to(self.config.device)
            prev_s = s_t.unsqueeze(1)
        else:
            et = self.W_prev(prev_s)
            dec_fea = self.W_s(s_t).unsqueeze(1)
            et = et + dec_fea
            et = torch.tanh(et)
            et = self.v(et).squeeze(2)
            at = softmax(et, dim=1).unsqueeze(1) 
            ct_d = torch.bmm(at, prev_s).squeeze(1)
            prev_s = torch.cat([prev_s, s_t.unsqueeze(1)], dim=1)

        return ct_d, prev_s

class Decoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.config = config

        self.enc_attention = EncoderAttention(config)
        self.dec_attention = DecoderAttention(config)
        self.x_context = nn.Linear(
            HIDDEN_DIM * 2 + EMBED_DIM,
            EMBED_DIM
        )

        self.lstm = nn.LSTMCell(
            EMBED_DIM,
            HIDDEN_DIM
        )
        init_lstm_wt(self.lstm, config)

        self.p_gen_linear = nn.Linear(
            HIDDEN_DIM * 5 + EMBED_DIM,
            1
        )

        self.V = nn.Linear(
            HIDDEN_DIM * 4,
            HIDDEN_DIM
        )
        self.V1 = nn.Linear(HIDDEN_DIM, vocab_size)
        init_linear_wt(self.V1, config)

    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s):
        x = self.x_context(torch.cat([x_t, ct_e], dim=1))
        s_t = self.lstm(x, s_t)

        dec_h, dec_c = s_t
        st_hat = torch.cat([dec_h, dec_c], dim=1)
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out, enc_padding_mask, sum_temporal_srcs)

        ct_d, prev_s = self.dec_attention(dec_h, prev_s)

        p_gen = torch.cat([ct_e, ct_d, st_hat, x], 1)
        p_gen = self.p_gen_linear(p_gen)
        p_gen = torch.sigmoid(p_gen)

        out = torch.cat([dec_h, ct_e, ct_d], dim=1)
        out = self.V(out)
        out = self.V1(out)
        vocab_dist = softmax(out, dim=1)
        vocab_dist = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        if extra_zeros is not None:
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s

class Model(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, vocab_size)
        self.embeds = nn.Embedding(vocab_size, EMBED_DIM)
        init_wt_normal(self.embeds.weight, config)

        self.encoder = self.encoder.to(config.device)
        self.decoder = self.decoder.to(config.device)
        self.embeds = self.embeds.to(config.device)
