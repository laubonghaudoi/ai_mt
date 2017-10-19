'''
Simple seq2seq model for translation
'''
import random

import torch
from torch.autograd import Variable

from lang import prepareData
from model import EncoderRNN, AttnDecoderRNN
from config import Config
from train import trainIters

config = Config()

input_lang, output_lang, pairs = prepareData('eng', 'zh', config)
config.input_lang_n_words = input_lang.n_words
config.output_lang_n_words = output_lang.n_words
print(random.choice(pairs))


encoder1 = EncoderRNN(config)
attn_decoder1 = AttnDecoderRNN(config)

if config.use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, pairs, input_lang, output_lang, config)

evaluateRandomly(encoder1, attn_decoder1)