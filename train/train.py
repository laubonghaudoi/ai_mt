import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from util import asMinutes, timeSince, showPlot


SOS_token = 0
EOS_token = 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence, config):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if config.use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair, input_lang, output_lang, config):
    input_variable = variableFromSentence(input_lang, pair[0], config)
    target_variable = variableFromSentence(output_lang, pair[1], config)
    return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(
        config.max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if config.use_cuda else encoder_outputs

    loss = 0

    # Run the encoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random(
    ) < config.teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, pairs, input_lang, output_lang, config):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(
        encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.SGD(
        decoder.parameters(), lr=config.learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), input_lang, output_lang, config)
                      for i in range(config.n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, config.n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion, config)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % config.print_every == 0:
            print_loss_avg = print_loss_total / config.print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / config.n_iters),
                                         iter, iter / config.n_iters * 100, print_loss_avg))

        if iter % config.plot_every == 0:
            plot_loss_avg = plot_loss_total / config.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, config):
    input_variable = variableFromSentence(input_lang, sentence, config)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(
        config.max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if config.use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(config.max_length, config.max_length)

    for di in range(config.max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, config):
    for i in range(config.evaluate_n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang, config)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
