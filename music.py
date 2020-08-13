# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt

from utils import *

PATH = f'data'

log = logger(True) 
log("We're logging :)") #Log will print whatever it's given if this first arg is true

# Auto-reloading of modules in iPython
# %load_ext autoreload
# %autoreload 2

# CONSTS
SAVE_EVERY = 20
SEQ_SIZE = 64
RANDOM_SEED = 11
VALIDATION_SIZE = 0.15
LR = 1e-3
N_EPOCHS = 5 #WAS 100
NUM_LAYERS, HIDDEN_SIZE = 2, 150
DROPOUT_P = 0
model_type = 'lstm'

use_cuda = False #torch.cuda.is_available(); log("use_cuda is: %s" % use_cuda)

torch.manual_seed(RANDOM_SEED)
INPUT = f'data/music.txt'  # Music
RESUME =False# True                                   # r e s u m i n g (Micha: Remove)
CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(model_type, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)

PLOT = True

# READ IN DATA FILE
f = open(INPUT,"r")
data, buffer = [], ''
store = False
for line in f:
    if line == '<start>\n':
        buffer += line
    elif line == '<end>\n':
        buffer += line
        data += [buffer]
        buffer = ''
    else:
        buffer += line
f.close()

# We only want songs which are at least as big as our batch size +1
data = [ song for song in data if len(song) > SEQ_SIZE + 10 ]

print(data[0])
print('=====> Data loaded')

char_idx = ''.join(set(list(open(INPUT,'r').read())))
char_list = list(char_idx)

# NOW SPLIT INTO TRAIN/VALIDATION SETS
num_train = len(data)
indices = list(range(num_train))
split_idx = int(np.floor(VALIDATION_SIZE * num_train))

# Shuffle data and split
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)
train_idxs, valid_idxs = indices[split_idx:], indices[:split_idx]

train_len, valid_len = len(train_idxs), len(valid_idxs)
log('Number of unique characters: %s' % len(char_idx))
log('Original data length: %s' % len(data))
log('Training data length: %s'% train_len)
log('Validation data length: %s' % valid_len)
assert(train_len + valid_len == len(data)), 'Train_len + valid_len should == len(data)'

# Some utils
def tic(): return time.time()

def toc(tic, msg=None):
    s = time.time() - tic
    m = int(s / 60)
    if msg:
        return '{}m {}s {}'.format(m, int(s - (m * 60)), msg)
    return '{}m {}s'.format(m, int(s - (m * 60)))

# Gives us a random slice of size SEQ_SIZE + 1 so we can get a train/target.
def rand_slice(data, slice_len=SEQ_SIZE):
    d_len = len(data)
    s_idx = random.randint(0, d_len - slice_len)
    e_idx = s_idx + slice_len + 1
    return data[s_idx:e_idx]

test_slice = rand_slice(data[0])
log(test_slice)

def seq_to_tensor(seq):
    '''
    create tensor from char seq
    '''
    out = torch.zeros(len(seq)).long()
    for i, c in enumerate(seq):
        out[i] = char_idx.index(c)

    if use_cuda:
        out.cuda()

    return out

t = seq_to_tensor(test_slice)
log('T is a: ', type(t), ' of size ', len(t))

def train_slice(data, slice_len=50):
    '''
    creates a random training set
    '''
    slice_i = rand_slice(data, slice_len=slice_len)
    seq = seq_to_tensor(slice_i[:-1])
    target = seq_to_tensor(slice_i[1:])
    return Variable(seq), Variable(target)

def train_batch(data, b_size=100, slice_len=50):
    batch_seq = torch.zeros(b_size, slice_len).long()
    batch_target = torch.zeros(b_size, slice_len).long()
    for idx in range(b_size):
        seq, target = train_slice(data, slice_len=slice_len)
        batch_seq[idx] = seq.data
        batch_target[idx] = target.data
    return Variable(batch_seq), Variable(batch_target)

# Given a song, return a sequence/target as a variable
def song_to_seq_target(song):
    a_slice = rand_slice(song)
    seq = seq_to_tensor(a_slice[:-1])
    target = seq_to_tensor(a_slice[1:])
    assert(len(seq) == len(target)), 'SEQ AND TARGET MISMATCH'
    return Variable(seq), Variable(target)

s, t = song_to_seq_target(data[0])
log(s.size())
assert(t.data[0] == s.data[1])

class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model='gru', num_layers=1):
        super(MusicRNN, self).__init__()
        self.model = model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embeddings = nn.Embedding(input_size, hidden_size)
        if self.model == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif self.model == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            raise NotImplementedError
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.drop = nn.Dropout(p=DROPOUT_P)
        
    def init_hidden(self):
        if self.model == 'lstm':
            if(use_cuda):
                self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()),
                                Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda()))
            else:
                self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                                Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        elif self.model == 'gru':
            if(use_cuda):
                self.hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda())
            else:
                self.hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))

        
    def forward(self, seq):
        embeds = self.embeddings(seq.view(1, -1))
        rnn_out, self.hidden = self.rnn(embeds.view(1,1,-1), self.hidden)
        rnn_out = self.drop(rnn_out)
        output = self.out(rnn_out.view(1,-1))
        return output

def some_pass(seq, target, fit=True):
    model.init_hidden() # Zero out the hidden layer
    model.zero_grad()   # Zero out the gradient
    some_loss = 0
    
    for i, c in enumerate(seq):

        if(use_cuda):
            output = model(c.cuda())
            some_loss += loss_function(output, torch.unsqueeze(target[i],dim=0).cuda())
        else:
            output = model(c)
            some_loss += loss_function(output, torch.unsqueeze(target[i],dim=0))
        
    if fit:
        some_loss.backward()
        optimizer.step()
    
    return some_loss.data / len(seq)

# Model
if RESUME:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    checkpoint = torch.load('./checkpoint/' + CHECKPOINT + '.t4')
    model = checkpoint['model']
    loss = checkpoint['loss']
    v_loss = checkpoint['v_loss']
    losses = checkpoint['losses']
    v_losses = checkpoint['v_losses']
    start_epoch = checkpoint['epoch']
    
else:
    print('==> Building model..')
    in_size, out_size = len(char_idx), len(char_idx)
    model = MusicRNN(in_size, HIDDEN_SIZE, out_size, model_type, NUM_LAYERS)
    loss, v_loss = 0, 0
    losses, v_losses = [], []
    start_epoch = 0

if use_cuda:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# optimizer = torch.optim.Adagrad(model.parameters())
# optimizer = torch.optim.RMSprop(model.parameters())

loss_function = nn.CrossEntropyLoss()

# Train
time_since = tic()
for epoch in range(start_epoch, N_EPOCHS):
    # Training
    for i, song_idx in enumerate(train_idxs):
        this_loss = some_pass(*song_to_seq_target(data[song_idx]))
        loss += this_loss
        
        msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
             epoch, (i+1)/len(train_idxs)*100, i, toc(time_since), this_loss)
        sys.stdout.write(msg)
        sys.stdout.flush()
    print()
    losses.append(loss / len(train_idxs))
        
    # Validation
    for i, song_idx in enumerate(valid_idxs):
        this_loss = some_pass(*song_to_seq_target(data[song_idx]), fit=False)
        v_loss += this_loss
        
        msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Time: {} Loss: {:.4}'.format(
             epoch, (i+1)/len(valid_idxs)*100, i, toc(time_since), this_loss)
        sys.stdout.write(msg)
        sys.stdout.flush()
    print()
    v_losses.append(v_loss / len(valid_idxs))
    
    # Save checkpoint.
    if epoch % SAVE_EVERY == 0 and start_epoch != epoch or epoch == N_EPOCHS - 1:
        print('=======>Saving..')
        state = {
            'model': model.modules if use_cuda else model,
            'loss': losses[-1],
            'v_loss': v_losses[-1],
            'losses': losses,
            'v_losses': v_losses,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t%s' % epoch)
        torch.save(state, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)
    
    # Reset loss
    loss, v_loss = 0, 0

if PLOT:
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=0)      # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=12)   # fontsize of the figure title
    plt.plot(losses, label='Training Loss')
    plt.plot(v_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

def write_song(prime_str='<start>', max_len=1000, temp=0.8):
    model.init_hidden()
    
    # "build up" hidden state using the beginging of a song '<start>'
    creation = '<start>'
    prime = Variable(seq_to_tensor(creation))
    for i in range(len(prime)-1):
        _ = model(prime[i])

    # Generate rest of sequence
    for j in range(max_len):
        out = model(Variable(seq_to_tensor(creation[-1]))).data.view(-1)
        
        out = np.array(np.exp(out/temp))
        dist = out / np.sum(out)

        # Add predicted character to string and use as next input        
        creation += char_idx[np.random.choice(len(dist), p=dist)]
        if creation[-5:] == '<end>':
            break

    return creation

log(write_song(max_len=1000, temp=0.8))

# # Heatmap generation
# # Needed to make this 100x100 in order to be able to print it out as a nice lookin grid...
# heatmap = np.zeros((100,100))
# for neuron_idx in range(len(char_idx)-1):
#     for j, char in enumerate(char_idx):
#         #model.init_hidden()
#         out = model(Variable(seq_to_tensor(char))).data.view(-1)[neuron_idx]
#         heatmap[neuron_idx,j] = out

# # Adapted from https://stackoverflow.com/questions/25071968/heatmap-with-text-in-each-cell-with-matplotlibs-pyplot
# plt.rc('font', size=100)          # controls default text sizes
# plt.rc('axes', titlesize=100)     # fontsize of the axes title
# plt.rc('axes', labelsize=0)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=100)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=100)    # fontsize of the tick labels
# plt.rc('legend', fontsize=100)    # legend fontsize
# plt.rc('figure', titlesize=100)  # fontsize of the figure title

# title = "Heatmap For Music RNN"
# xlabel= "Character"
# ylabel="Neuron ID"
# data =  np.reshape(heatmap[0,:], (10,-1))
# plt.figure(figsize=np.shape(heatmap))
# plt.title(title)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# c = plt.pcolor(data, edgecolors='k', linewidths=4, cmap='RdBu_r', vmin=-1.0, vmax=1.0)

# def show_values(pc, fmt="%.2f", **kw):
#     pc.update_scalarmappable()
#     ax = pc.axes
#     for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
#         x, y = p.vertices[:-2, :].mean(0)
#         if np.all(color[:3] > 0.5):
#             color = (0.0, 0.0, 0.0)
#         else:
#             color = (1.0, 1.0, 1.0)
#         idx = int(x-.5) + 10*int(y-.5)
#         if idx < len(char_idx):
#             ax.text(x, y, repr(char_list[idx])[1:-1], fontsize=180, ha="center", va="center", color=color, **kw)

# show_values(c)
# plt.colorbar(c)

# plt.show()

CHECKPOINT