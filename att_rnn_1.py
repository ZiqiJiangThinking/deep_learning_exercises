# goal:
# implement in pytorch from scratch the NLP model in Neural Machine Translation by Jointly Learning to Align and Translate

# reference: 
# paper: 
#        Neural Machine Translation by Jointly Learning to Align and Translate
#        Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
#        https://arxiv.org/abs/1409.0473
# test functions:
#        https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import math
import random
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# plt.switch_backend('agg')

from util import readLangs, split_train_test, timeSince, tensorsFromPair, tensorFromSentence


# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

train_pairs, eval_pairs = split_train_test(data = pairs, train_fraction = 0.7)
print(random.choice(train_pairs))

# Encoder
class EncoderRNN(nn.Module):
    '''
    Params:
        input_size: the number of tokens in source vocabulary
        encoder_hidden_size: the desired dimension of encoder
        decoder_hidden_size: the desired dimension of decoder
    Inputs:
        input: a input tensor, (batch_size, 1)
        hidden_state: the hidden state from previous encoder node, (D∗num_layers,batch_size,output_size)
                      D = 2 if bidirectional, otherwise, 1
    Return:
        output: all hidden states of the input sequence, back and forwards, (batch_size=1, sequence_length, encoder_hidden_size)
        hidden_state: final forward and backward hidden states passed through a linear layer (batch_size, sequence_length, decoder_hidden_size)
        hidden_state_de_ini: the initialized hidden state for decoder, (batch_size, 1, decoder_hidden_size)
                             The paper uses the 1st backwards direction hidden state
    '''
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size):
        super(EncoderRNN, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.emb = nn.Embedding(num_embeddings = input_size, 
                                embedding_dim = encoder_hidden_size)# emd (input_shape, embedding_dim)
        self.gru = nn.GRU(input_size = encoder_hidden_size, 
                          hidden_size = encoder_hidden_size, 
                          bidirectional = True,
                          batch_first=True) # input (batch_size, sequence_length, input_size)
        self.fc = nn.Linear(encoder_hidden_size, decoder_hidden_size)
    
    # for each time t, both source input and hidden state from t-1 are needed
    def forward(self, input, hidden_state):
        # (∗,embedding_dim = encoder_hidden_size in this case), where * is the input shape
        embedded = self.emb(input)
        # output (batch_size, sequence_length, D*output_size), D=2 if bidirectional, otherwise 1
        # hidden_state (D∗num_layers,batch_size,output_size)
        output, hidden_state = self.gru(embedded, hidden_state)

        # the encoder hidden_states are saved in output
        # the backward hidden_state of 1st token in the sequence is used as the initial hidden_state of decoder
        # For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        hidden_state_de_ini = self.fc(hidden_state)[1]
        return output, hidden_state, hidden_state_de_ini



# Attention
class BahdandauAtt(nn.Module):
    '''
    Params:
        encoder_hidden_size: the desired dimension of encoder, align with en/decoder's parameter
        decoder_hidden_size: the desired dimension of decoder, align with en/decoder's parameter
    Inputs:
        decoder_hidden_state: the hidden state from previous decoder node (batch_size, 1, decoder_hidden_size)
        encoder_hidden_states: all the hidden states from encoder, (batch_size, encoded_sequence_length, output_size*D)
    Return:
        context: the weighted encoder hidden states (batch_size, 1, encoder_hidden_size*2)
    '''
    def __init__(self, decoder_hidden_size, encoder_hidden_size):
        super(BahdandauAtt, self).__init__()
        self.fc = nn.Linear(decoder_hidden_size+encoder_hidden_size*2, 1, bias = False)
        self.softmax = nn.Softmax()
    def forward(self, decoder_hidden_state, encoder_hidden_states):
        # encoder_hidden_states (batch_size, encoded_sequence_length, output_size*D)
        num_encoder_hidden_states = encoder_hidden_states.size()[1]
        # decoder_hidden_states (batch_size, encoded_sequence_length, output_size*D)
        decoder_hidden_states = decoder_hidden_state.repeat(1, num_encoder_hidden_states, 1)
        hidden_states = torch.cat((decoder_hidden_states, encoder_hidden_states), dim=2)
        energy = self.fc(hidden_states).tanh()
        # attention (batch_size, sequence_length, 1)
        # attention expect (batch_size, 1, sequence_length)
        attention = self.softmax(energy).view(1, -1, num_encoder_hidden_states)
        # attention expect (batch_size, 1, sequence_length)
        # encoder_hidden_states (batch_size, sequence_length, encoder_hidden_size*D)
        context = torch.bmm(attention, encoder_hidden_states)
        # context (batch_size, 1, encoder_hidden_size*2)
        return context

# Decoder
class AttDecoderRNN(nn.Module):
    '''
    Params:
        output_size: the number of tokens in target vocabulary
        encoder_hidden_size: the desired dimension of encoder, align with en/decoder's parameter
        decoder_hidden_size: the desired dimension of decoder, align with en/decoder's parameter
    Inputs:
        input: previous decoder ouput tensor (batch_size, 1)
        decoder_hidden_state: the hidden state from previous decoder node (batch_size, 1, decoder_hidden_size)
        encoder_hidden_states: all the hidden states from encoder, (batch_size, encoded_sequence_length, output_size*D)
    Return:
        prediction: the log likelihood of each class (batch_size, 1, output_size)
        decoder_hidden_state: the hidden state of current decoder node (batch_size, 1, decoder_hidden_size)
        context: the weighted encoder hidden states (batch_size, 1, encoder_hidden_size*2) 
                 this is output to check decoder's functionality
    '''
    def __init__(self, output_size, decoder_hidden_size, encoder_hidden_size, dropout):
        super(AttDecoderRNN, self).__init__()
        self.output_size = output_size
        self.emb = nn.Embedding(num_embeddings = output_size, embedding_dim = decoder_hidden_size)
        self.gru = nn.GRU(input_size = decoder_hidden_size+encoder_hidden_size*2, hidden_size = decoder_hidden_size)
        self.attention = BahdandauAtt(decoder_hidden_size, encoder_hidden_size)
        # fc_out encoder_hidden_size*2+decoder_hidden_size+embed_dim
        self.fc_out = nn.Linear(encoder_hidden_size*2+decoder_hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(2)
    
    def forward(self, input, decoder_hidden_state, encoder_hidden_states):
        # embedded (batch_size, 1, embedding_dim)
        embedded = self.emb(input)
        embedded = self.dropout(embedded)
        # context (batch_size, 1, encoder_hidden_size*2)
        context = self.attention(decoder_hidden_state, encoder_hidden_states)
        # decoder_input (batch_size, 1, encoder_hidden_size*2+decoder_hidden_size)
        decoder_input = torch.cat((embedded, context), dim = 2)
        # decoder_hidden_state (batch_size, 1, decoder_hidden_size)
        # output (batch_size, 1, decoder_hidden_size)
        
        output, decoder_hidden_state = self.gru(decoder_input, decoder_hidden_state)
        # pred_input (batch_size, 1, decoder_hidden_size*2+encoder_hidden_size*2)
        pred_input = torch.cat((embedded, context, output), dim = 2)
        # prediction (batch_size, 1, output_size)
        prediction = self.logsoftmax(self.fc_out(pred_input))
        return prediction, decoder_hidden_state, context


# Model
class ModelRNN(nn.Module):
    '''
    Params:
        input_size: the number of tokens in source vocabulary
        output_size: the number of tokens in target vocabulary
        encoder_hidden_size: the desired dimension of encoder, align with en/decoder's parameter
        decoder_hidden_size: the desired dimension of decoder, align with en/decoder's parameter
        dropout: the desired dropout ratio
    Inputs:
        input_tensor: the tensor containing all words in the source sentence
        target_tensor: the tensor containing all words in the target sentence
        teacher_forcing_ratio: [0, 1], the probablity of using the label word tensor instead of decoder output
                               as the input of the next decoder unit
    Return:
        decoder_outputs: the tensor containing the translation of the source sentence
    '''
    def __init__(self, input_size, output_size, decoder_hidden_size, encoder_hidden_size, dropout):
        super(ModelRNN, self).__init__()
        self.encoder = EncoderRNN(input_size = input_size, 
                                   decoder_hidden_size = decoder_hidden_size, 
                                   encoder_hidden_size = encoder_hidden_size).to(device)
        self.decoder = AttDecoderRNN(output_size = output_size, 
                                    decoder_hidden_size = decoder_hidden_size, 
                                    encoder_hidden_size = encoder_hidden_size, 
                                    dropout=dropout).to(device)

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio = 0.5):
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(input_length, self.encoder.encoder_hidden_size*2, device=device)
        decoder_outputs = torch.zeros(target_length, self.decoder.output_size, device=device)

        encoder_hidden = torch.zeros(2, 1, self.encoder.encoder_hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, hidden_state_de_ini = self.encoder(
                input_tensor[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = hidden_state_de_ini
        encoder_outputs = encoder_outputs.view(1, -1, self.encoder.encoder_hidden_size*2)

        # initialization
        decoder_hidden = decoder_hidden.view(1, -1, self.encoder.encoder_hidden_size)

        for di in range( target_length):
            decoder_output, decoder_hidden, decoder_context = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output (batch_size, ouput_size)
            decoder_output = decoder_output.squeeze(1)
            decoder_outputs[di] = decoder_output

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[di].view(1,-1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_token:
                    break
                decoder_input = decoder_input.view(1,-1)
        return decoder_outputs

# Training

def train(input_tensor, target_tensor, model, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio = 0.5):
    '''
    Functionality: 
        train on one sentence which includes multiple tokens
    Input:
        input_tensor: the tensor containing all words in the source sentence
        target_tensor: the tensor containing all words in the target sentence
        model: the ModelRNN object which gives the loglikelihood for each class 
               # class = # of token in target language
        encoder_optimizer: the optimizer for encoder
        decoder_optimizer: the decoder for encoder
        criterion: the loss function to evaluate the goodness of fit
        teacher_forcing_ratio: [0, 1], the probablity of using the label word tensor instead of decoder output
                               as the input of the next decoder unit
    Return:
        the average loss per token in this sentence
    '''
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    decoder_outputs = model(input_tensor = input_tensor, 
                            target_tensor = target_tensor, 
                            teacher_forcing_ratio = teacher_forcing_ratio)
    
    decoder_outputs_size = decoder_outputs.size(0)

    for di in range(decoder_outputs_size):
        loss += criterion(decoder_outputs[di].unsqueeze(0), target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / decoder_outputs_size



def trainIters(model, n_iters, print_every=1000, learning_rate=1e-5):
    '''
    Functionality: 
        iterate train() for specified times and print result periodically
    Input:
        model: the ModelRNN object which gives the loglikelihood for each class 
               # class = # of token in target language
        n_iters: the desired number of iterations for training
        print_every: the desired cadence of print out tentative result
    '''
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    training_pairs = [tensorsFromPair(random.choice(train_pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, model, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

# Test functions
def testEncoder(encoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]

        encoder_hidden = torch.zeros(2, 1, encoder.encoder_hidden_size, device=device)
        encoder_outputs = torch.zeros(input_length, encoder.encoder_hidden_size*2, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, hidden_state_de_ini = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        return encoder_hidden, encoder_outputs
    
def testEncoderRandomly(encoder, n=2):
    for i in range(n):
        print('---------------------------------------')
        print('-------------------', i, '-------------------')
        print('---------------------------------------')
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        encoder_hidden, encoder_outputs = testEncoder(encoder, pair[0])
        print('-------------encoder_hidden-------------')
        print(encoder_hidden)
        print('-------------encoder_outputs-------------')
        print(encoder_outputs)

# print('=======================================')
# print('=================Encoder===============')
# print('=======================================')
# encoder_hidden_size = 256
# decoder_hidden_size = 256
# encoder1 = EncoderRNN(input_lang.n_words, encoder_hidden_size, decoder_hidden_size).to(device)
# testEncoderRandomly(encoder1)


def testDecoder(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        print('input_tensor', input_tensor)
        input_length = input_tensor.size()[0]
        print('input_length', input_length)
        encoder_hidden = torch.zeros(2, 1, encoder.encoder_hidden_size, device=device)

        encoder_outputs = torch.zeros(max_length, encoder.encoder_hidden_size*2, device=device)
        for ei in range(input_length):
            print('looking at input_tensor', ei, input_tensor[ei].unsqueeze(0))
            encoder_output, encoder_hidden, hidden_state_de_ini = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = hidden_state_de_ini

        # encoder_outputs (sequence_length, output_size*2)
        # encoder_outputs expected (batch_size, sequence_length, output_size*2) so that it can be concat later
        encoder_outputs = encoder_outputs.view(1, -1, encoder.encoder_hidden_size*2)
        decoder_hidden = decoder_hidden.view(1, -1, encoder.encoder_hidden_size)
        decoded_words = []
        decoder_contexts = torch.zeros(max_length, encoder.encoder_hidden_size*2)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_context = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_contexts[di] = decoder_context.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            # which dimension should be squeezed?
            decoder_input = topi.squeeze(0)

        return decoded_words, decoder_contexts[:di + 1]

def testDecoderRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, decoder_contexts = testDecoder(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# print('=======================================')
# print('=================Decoder===============')
# print('=======================================')
# encoder_hidden_size = 256
# decoder_hidden_size = 256
# encoder1 = EncoderRNN(input_lang.n_words, encoder_hidden_size, decoder_hidden_size).to(device)
# attn_decoder1 = AttDecoderRNN(output_lang.n_words, decoder_hidden_size, encoder_hidden_size, dropout=0.1).to(device)
# testDecoderRandomly(encoder1, attn_decoder1)

# Evaluation
def evaluate(input_tensor, target_tensor, model, criterion, max_length=MAX_LENGTH, teacher_forcing_ratio = 0.5):
    '''
    Functionality: 
        evaluate on one sentence which includes multiple tokens
    Input:
        input_tensor: the tensor containing all words in the source sentence
        target_tensor: the tensor containing all words in the target sentence
        model: the ModelRNN object which gives the loglikelihood for each class 
               # class = # of token in target language
        encoder_optimizer: the optimizer for encoder
        decoder_optimizer: the decoder for encoder
        criterion: the loss function to evaluate the goodness of fit
        teacher_forcing_ratio: [0, 1], the probablity of using the label word tensor instead of decoder output
                               as the input of the next decoder unit
    Return:
        the average loss per token in this sentence
    '''
    loss = 0

    decoder_outputs = model(input_tensor = input_tensor, 
                            target_tensor = target_tensor, 
                            teacher_forcing_ratio = teacher_forcing_ratio)
    
    decoder_outputs_size = decoder_outputs.size(0)

    for di in range(decoder_outputs_size):
        loss += criterion(decoder_outputs[di].unsqueeze(0), target_tensor[di])

    return loss.item() / decoder_outputs_size

def evaluateIters(model, n_iters, print_every=1000):
    '''
    Functionality: 
        iterate evaluate() for specified times and print result periodically
    Input:
        model: the ModelRNN object which gives the loglikelihood for each class 
               # class = # of token in target language
        n_iters: the desired number of iterations for evaluation
        print_every: the desired cadence of print out tentative result
    '''
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    criterion = nn.NLLLoss()
    evaling_pairs = [tensorsFromPair(random.choice(eval_pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]
    
    for iter in range(1, n_iters + 1):
        eval_pair = evaling_pairs[iter - 1]
        input_tensor = eval_pair[0]
        target_tensor = eval_pair[1]

        loss = evaluate(input_tensor, target_tensor, model, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


print('=======================================')
print('================= Train ===============')
print('=======================================')
modelrnn = ModelRNN(input_size = input_lang.n_words, 
                    output_size = output_lang.n_words, 
                    decoder_hidden_size = 256, 
                    encoder_hidden_size = 256, 
                    dropout = 0.2)

print(trainIters(modelrnn, n_iters=75000, print_every=5000,learning_rate=1e-5))

print('========================================')
print('=================Evaluate===============')
print('========================================')
print(evaluateIters(modelrnn, n_iters=5000, print_every=1000))


# print out result:
# Reading lines...
# Read 135842 sentence pairs
# Trimmed to 10599 sentence pairs
# Counting words...
# Counted words:
# fra 4345
# eng 2803
# ['je suis determine a devenir un scientifique .', 'i am determined to be a scientist .']
# ['je suis enchantee de vous rencontrer .', 'i m delighted to meet you .']
# =======================================
# ================= Train ===============
# =======================================
# main_simplified.py:205: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
#   attention = self.softmax(energy).view(1, -1, num_encoder_hidden_states)
# 4m 1s (- 56m 27s) (5000 6%) 4.8794
# 8m 14s (- 53m 36s) (10000 13%) 3.7579
# 12m 28s (- 49m 54s) (15000 20%) 3.4864
# 16m 42s (- 45m 56s) (20000 26%) 3.3212
# 20m 59s (- 41m 58s) (25000 33%) 3.2066
# 25m 20s (- 38m 0s) (30000 40%) 3.1515
# 29m 48s (- 34m 4s) (35000 46%) 3.0750
# 34m 16s (- 29m 59s) (40000 53%) 3.0212
# 38m 44s (- 25m 49s) (45000 60%) 2.9970
# 43m 12s (- 21m 36s) (50000 66%) 2.9247
# 47m 41s (- 17m 20s) (55000 73%) 2.8909
# 52m 13s (- 13m 3s) (60000 80%) 2.8666
# 56m 45s (- 8m 43s) (65000 86%) 2.8364
# 61m 16s (- 4m 22s) (70000 93%) 2.8153
# 65m 49s (- 0m 0s) (75000 100%) 2.7848
# None
# ========================================
# =================Evaluate===============
# ========================================
# 0m 8s (- 0m 32s) (1000 20%) 2.8295
# 0m 16s (- 0m 24s) (2000 40%) 2.8269
# 0m 25s (- 0m 16s) (3000 60%) 2.8000
# 0m 33s (- 0m 8s) (4000 80%) 2.8358
# 0m 41s (- 0m 0s) (5000 100%) 2.8867
# None