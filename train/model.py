import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.use_cuda = config.use_cuda

        self.n_layers = config.encoder_n_layers
        self.hidden_size = config.encoder_hidden_size

        self.embedding = nn.Embedding(config.input_lang_n_words, config.encoder_hidden_size)
        self.gru = nn.GRU(config.encoder_hidden_size, config.encoder_hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, config):
        super(AttnDecoderRNN, self).__init__()
        self.use_cuda = config.use_cuda

        self.hidden_size = config.decoder_hidden_size
        self.output_size = config.output_lang_n_words
        self.n_layers = config.decoder_n_layers
        self.dropout_p = config.decoder_dropout_p
        self.max_length = config.max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
