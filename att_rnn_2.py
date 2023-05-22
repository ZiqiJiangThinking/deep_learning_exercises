# goal:
# implement in pytorch the NLP model in Neural Machine Translation by Jointly Learning to Align and Translate
# use torchtext and Dataloader

# reference: 
# paper: 
#        Neural Machine Translation by Jointly Learning to Align and Translate
#        Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
#        https://arxiv.org/abs/1409.0473

# seq2seq translation from scratch:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# batched seq2seq 
# https://github.com/pengyuchen/PyTorch-Batch-Seq2seq/blob/master/seq2seq_translation_tutorial.py

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

plt.switch_backend('agg')

from util_1 import normalizeString, split_train_val_test, timeSince, tensorsFromPair, tensorFromSentence


# Load data
SRC_LANGUAGE = 'eng'
TGT_LANGUAGE = 'fra'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
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
        p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def readData(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = filterPairs(pairs)
    return pairs

from torchtext.data import to_map_style_dataset
file_path = 'data/{0}-{1}.txt'.format(SRC_LANGUAGE, TGT_LANGUAGE)
pairs = readData('eng', 'fra')
# print(pairs[10])
# ['wait !', 'attends !']

# data_iter = to_map_style_dataset(open(file_name,'r'))
# print(data_iter)
def tokenizer(l):
    return [normalizeString(s) for s in l.split(' ')]

# Build Vocabulary
from torchtext.vocab import build_vocab_from_iterator
# build an iteratable
def yield_tokens_lang(pairs, lang):
    for pair in pairs:
        yield pair[lang-1].strip().split()

train_pairs, eval_pairs, test_pairs = split_train_val_test(data = pairs, 
                                                           train_fraction = 0.7, 
                                                           eval_fraction = 0.2)
# print('one random training example:',random.choice(train_pairs))
# ['i let you catch me .', 'je vous ai laisses m attraper .']

print('one random validation example:',random.choice(eval_pairs))

vocab_transform = {}
token_transform = {}


vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens_lang(train_pairs, lang = 1), 
                                        specials=special_symbols)
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens_lang(train_pairs, lang = 2), 
                                        specials=special_symbols)
vocab_transform[SRC_LANGUAGE].set_default_index(vocab_transform[SRC_LANGUAGE]["<unk>"])
vocab_transform[TGT_LANGUAGE].set_default_index(vocab_transform[TGT_LANGUAGE]["<unk>"])
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print('len of lang1 vocab',SRC_VOCAB_SIZE)
print('len of lang2 vocab',TGT_VOCAB_SIZE)
# len of lang1 vocab 11597
# len of lang2 vocab 18667

# text processing pipeline with the tokenizer and vocabulary
token_transform[SRC_LANGUAGE] = lambda x: vocab_transform[SRC_LANGUAGE](tokenizer(x))
token_transform[TGT_LANGUAGE] = lambda x: vocab_transform[TGT_LANGUAGE](tokenizer(x))


# print(token_transform[SRC_LANGUAGE]('here is the an example'))
# print(token_transform[TGT_LANGUAGE]('ca alors !'))
# [56, 9, 5, 76, 1435]
# [41, 342, 0]

# Convert to DataLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
                                                #Tokenization + Numericalization
    text_transform[ln] = sequential_transforms(token_transform[ln], 
                                                # Add BOS/EOS and create tensor
                                               tensor_transform)


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

BATCH_SIZE = 8


# print(text_transform[SRC_LANGUAGE](train_pairs[10][0]))
# tensor([  2,   8,   6, 540,   0,   3])

from torch.utils.data import Dataset
class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]
        # source = text_transform[SRC_LANGUAGE](self.pairs[idx][0])
        # target = text_transform[TGT_LANGUAGE](self.pairs[idx][1])
        # return torch.tensor(source), torch.tensor(target)
    
train_iter = PairDataset(train_pairs)
val_iter = PairDataset(eval_pairs)
test_iter = PairDataset(test_pairs)

# print(len(train_iter))
# print(train_iter[0])


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
train_dataloader = DataLoader(train_iter, shuffle = False, 
                              batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataloader = DataLoader(val_iter, shuffle = False, 
                              batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_iter, shuffle = False, 
                              batch_size=BATCH_SIZE, collate_fn=collate_fn)


# for src, tgt in train_dataloader:
#     src # (source_padded_sentence_length, batch_size)
#     tgt # (target_padded_sentence_length, batch_size)
#     print(tgt)
#     break

## model definition
# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size):
        super(EncoderRNN, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        # emd_output (input_shape, embedding_dim)
        self.emb = nn.Embedding(num_embeddings = input_size, 
                                embedding_dim = encoder_hidden_size)
        # gru_input
        # input: (batch_size, sequence_length=1, <input_size>)
        # h0: (D∗num_layers=1, batch_size, <output_size>)
        # gru_output
        # output: (batch_size, sequence_length=1, D*<output_size>)
        # hn: (D∗num_layers=1, batch_size, <output_size>)
        # D = 2 if bidirectional
        # for pytorch, if sequence_length of input > 1, 
        # output contains all processed hidden states 
        # while hidden_state only contains the last one
        self.gru = nn.GRU(input_size = encoder_hidden_size, 
                          hidden_size = encoder_hidden_size, 
                          bidirectional = True,
                          batch_first=True) 
        # fc_input (input_shape, <input_size>)
        # fc_output (input_shape, <output_size>)
        # fc_input is encoder_hidden_size because this uses only <- h1
        self.fc = nn.Linear(encoder_hidden_size, decoder_hidden_size)
    
    def forward(self, input, hidden_state):
        embedded = self.emb(input)
        output, hidden_state = self.gru(embedded, hidden_state)
        # taking the backward hidden_state, will be used for decoder initialization
        # hidden_state_de_ini (batch, sequance_length=1, decoder_hidden_size)
        hidden_state_de_ini = self.fc(hidden_state)[1].unsqueeze(1)
        return output, hidden_state, hidden_state_de_ini


class BahdandauAtt(nn.Module):
    def __init__(self, decoder_hidden_size, encoder_hidden_size):
        super(BahdandauAtt, self).__init__()
        self.fc = nn.Linear(decoder_hidden_size+encoder_hidden_size*2, 1, bias = False)
        self.softmax = nn.Softmax()
    def forward(self, decoder_hidden_state, encoder_hidden_states):
        # print('decoder_hidden_state>>>>>', decoder_hidden_state.size())
        # print('encoder_hidden_states>>>>>', encoder_hidden_states.size())
        # encoder_hidden_states (encoded_sequence_length = length of padded sentence, 
        #                        batch_size, 
        #                        D*hidden_size)
        num_encoder_hidden_states = encoder_hidden_states.size()[0]
        total_encoder_hidden_size = encoder_hidden_states.size()[2]
        # decoder_hidden_state original (1, batch_size, hidden_size*D)
        # decoder_hidden_states expected (encoded_sequence_length, batch_size, hidden_size)
        decoder_hidden_states = decoder_hidden_state.repeat(num_encoder_hidden_states, 1, 1)
        # hidden_states (encoded_sequence_length, batch_size, D*hidden_size+hidden_size)
        hidden_states = torch.cat((decoder_hidden_states, encoder_hidden_states), dim=2)
        # energy (encoded_sequence_length, batch_size, 1)
        energy = self.fc(hidden_states).tanh().squeeze(2)
        energy = torch.transpose(energy, 0, 1).unsqueeze(1)
        # attention expect (batch_size, 1, sequence_length)
        attention = self.softmax(energy)
        # encoder_hidden_states (batch_size, sequence_length, encoder_hidden_size*D)
        # mat1 (b*n*m), mat2 (b*m*p) => output (b*n*p)
        # b = batch_size
        # n = 1
        # p = encoder_hidden_size*D
        # attention expected (batch_size, 1, m)
        # encoder_hidden_states expected (batch_size, m, encoder_hidden_size*D)
        encoder_hidden_states = encoder_hidden_states.view(-1, 
                                                           num_encoder_hidden_states, 
                                                           total_encoder_hidden_size)
        context = torch.bmm(attention, encoder_hidden_states)
        # context (batch_size, 1, encoder_hidden_size*D)
        return context
    
class AttDecoderRNN(nn.Module): 
    def __init__(self, output_size, decoder_hidden_size, encoder_hidden_size, dropout):
        super(AttDecoderRNN, self).__init__()
        self.output_size = output_size
        self.emb = nn.Embedding(num_embeddings = output_size, 
                                embedding_dim = decoder_hidden_size)
        # gru_input
        # input: (batch_size, sequence_length=1, <input_size>)
        # h0: (D=1∗num_layers=1, batch_size, <output_size>)
        # gru_output
        # output: (batch_size, sequence_length=1, D=1*<output_size>)
        # hn: (D=1∗num_layers=1, batch_size, <output_size>)
        # D = 1 if not bidirectional
        self.gru = nn.GRU(input_size = decoder_hidden_size+encoder_hidden_size*2, 
                          hidden_size = decoder_hidden_size,
                          batch_first = True)
        self.attention = BahdandauAtt(decoder_hidden_size, encoder_hidden_size)
        # fc_out encoder_hidden_size*2+decoder_hidden_size+embed_dim
        self.fc_out = nn.Linear(encoder_hidden_size*2+decoder_hidden_size*2, 
                                output_size)
        self.dropout = nn.Dropout(dropout)
        # LogSoftmax: the shape of output = input
        # dim = 2: dimension along which LogSoftmax will be computed.
        self.logsoftmax = nn.LogSoftmax(2)
    
    def forward(self, input, decoder_hidden_state, encoder_hidden_states):
        # embedded (batch_size, 1, embedding_dim=decoder_hidden_size)
        embedded = self.emb(input)
        embedded = self.dropout(embedded)
        # context (batch_size, 1, encoder_hidden_size*D)
        context = self.attention(decoder_hidden_state, encoder_hidden_states)
        # decoder_input (batch_size, 1, encoder_hidden_size*D+decoder_hidden_size)
        decoder_input = torch.cat((embedded, context), dim = 2)
        # technically, output and decoder_hidden_state have identical, 
        # but output's shape is easier for later concatenation
        output, decoder_hidden_state = self.gru(decoder_input, decoder_hidden_state)
        # pred_input (batch_size, 1, decoder_hidden_size*2+encoder_hidden_size*2)
        pred_input = torch.cat((embedded, context, output), dim = 2)
        # prediction (batch_size, 1, output_size)
        prediction = self.logsoftmax(self.fc_out(pred_input))
        return prediction, decoder_hidden_state, context
    
class ModelRNN(nn.Module):
    def __init__(self, input_size, output_size, decoder_hidden_size, 
                 encoder_hidden_size, dropout):
        super(ModelRNN, self).__init__()
        self.encoder = EncoderRNN(input_size = input_size, 
                                   decoder_hidden_size = decoder_hidden_size, 
                                   encoder_hidden_size = encoder_hidden_size).to(device)
        self.decoder = AttDecoderRNN(output_size = output_size, 
                                    decoder_hidden_size = decoder_hidden_size, 
                                    encoder_hidden_size = encoder_hidden_size, 
                                    dropout=dropout).to(device)

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio = 0.5):
        '''
        input_tensor (input_sentence_length, batch_size)
        target_tensor (target_sentence_length, batch_size)
        '''
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        batch_size = input_tensor.size(1)

        encoder_outputs = torch.zeros(input_length, batch_size, 
                                      self.encoder.encoder_hidden_size*2, 
                                      device=device)
        decoder_outputs = torch.zeros(target_length, batch_size,
                                      self.decoder.output_size, 
                                      device=device)
        # encoder initialization
        # encoder_hidden expected (D=2∗num_layers=1, batch_size, <output_size>)
        encoder_hidden = torch.zeros(2, batch_size, self.encoder.encoder_hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, hidden_state_de_ini = self.encoder(
                input_tensor[ei].unsqueeze(1), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        
        # decoder_input expected (batch_size, 1)
        decoder_input = input_tensor[0].unsqueeze(1)

        # when bidirectional, the last hidden_state of gru gives you 
        # the last hidden state for the forward direction 
        # but the first hidden state of the backward direction
        # https://discuss.pytorch.org/t/output-of-a-gru-layer/92186/6
        decoder_hidden = hidden_state_de_ini

        # encoder_outputs original (batch_size, input_length, encoder_hidden_size*2)
        # encoder_outputs expected (batch_size, input_length, D*hidden_size)
        # encoder_outputs = encoder_outputs.view(1, -1, self.encoder.encoder_hidden_size*2)

        # decoder initialization
        # decoder_hidden(ini) original (batch_size, 1, encoder_hidden_size*D=1)
        # decoder_hidden expected (D=1∗num_layers=1, batch_size, encoder_hidden_size*D=1)
        decoder_hidden = decoder_hidden.view(1, -1, self.encoder.encoder_hidden_size)

        for di in range( target_length):
            decoder_output, decoder_hidden, decoder_context = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output original (batch_size, 1, ouput_size)
            # decoder_output expected (batch_size, ouput_size)
            # so squeeze
            decoder_output = decoder_output.squeeze(1)
            decoder_outputs[di] = decoder_output

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                # target_tensor[di] (1, batch_size)
                # decoder_input expected (batch_size, 1)
                decoder_input = target_tensor[di].view(-1,1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                # decoder_output (batch_size, ouput_size)
                # topv is the top1 values (batch_size,1)
                # topi is the top1 indexes (batch_size,1)
                topv, topi = decoder_output.topk(1, dim = 1)
                # detach from history as input
                decoder_input = topi.detach()
        return decoder_outputs
    
model = ModelRNN(input_size = SRC_VOCAB_SIZE, 
                 output_size = TGT_VOCAB_SIZE, 
                 decoder_hidden_size = 256, 
                 encoder_hidden_size = 256, 
                 dropout = 0.2)

def train(src, tgt, model, encoder_optimizer, decoder_optimizer, criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    target_length = tgt.size()[0]
    decoder_outputs = model(input_tensor = src, target_tensor = tgt)
    decoder_length = decoder_outputs.size()[0]

    for i in range(decoder_length):
        decoder_output = decoder_outputs[i]
        tgt_output = tgt[i]
        # prediction: (batch_size, num_of_class)
        # label: (batch_size)
        loss += criterion(decoder_output, tgt_output)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / target_length

def evaluate(dataloader, model):

    total = 0
    correct = 0

    ignore = [UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX]

    for src, tgt in dataloader:
        decoder_outputs = model(input_tensor = src, target_tensor = tgt)
        batch_size = decoder_outputs.size()[1]
        topv, topi = decoder_outputs.topk(1, dim = 2)
        # topi expected (batch_size, sequence_length)
        topi = topi.squeeze(2).transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        total = total + batch_size
        for di in range(batch_size):
            prediction = topi[di]
            y = tgt[di]
            prediction = [word for word in prediction if word not in ignore]
            y = [word for word in y if word not in ignore]
            total = total + 1
            if prediction == y:
                correct = correct + 1
    print('val accuracy:{0}%'.format(round(100*correct/total, 2)))

def trainIters(model, epochs, train_loader, val_loader, learning_rate=0.01):
    start = time.time()

    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.decoder.parameters()),
                                  lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        print_loss_total = 0
        for src, tgt in train_loader:
            loss = train(src, tgt, model, encoder_optimizer, decoder_optimizer, criterion)            
            print_loss_total += loss

        print('epochs: '+str(epoch))
        print('total train loss: '+str(print_loss_total))
        evaluate(val_loader, model)
    print('End training')

trainIters(model=model, 
           epochs=10, 
           train_loader=train_dataloader, 
           val_loader=val_dataloader, 
           learning_rate=1e-5)

evaluate(test_dataloader, model)
