'''
Simple seq2seq model for translation
'''
import random

import torch
from torch.autograd import Variable

from lang import prepareData
from model import EncoderRNN, AttnDecoderRNN
from config import Config
from train import trainIters, evaluateRandomly

config = Config()

input_lang, output_lang, pairs = prepareData('en', 'zh', config)
config.input_lang_n_words = input_lang.n_words
config.output_lang_n_words = output_lang.n_words
print(random.choice(pairs))


encoder = EncoderRNN(config)
decoder = AttnDecoderRNN(config)

if config.use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder, decoder, pairs, input_lang, output_lang, config)

evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, config)