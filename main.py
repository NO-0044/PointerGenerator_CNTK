from model import create_model,create_basic_model
#from train import train,create_reader,get_vocab
from basic_train import basic_train,create_reader,get_vocab
import cntk as C
import numpy as np
import h_p
from h_p import *

print('start build voc')
vocab, i2w, w2i = get_vocab('data/voc_50k.txt')
vocab = [x[0] for x in vocab]
h_p.vocab_dim = len(vocab)
h_p.max_extended_vocab_dim = vocab_dim + 400
h_p.sentence_start = np.array([i == w2i['<s>'] for i in range(vocab_dim)], dtype=np.float32)
h_p.sentence_end_index = vocab.index('</s>')
print('voc builded')

#C.try_set_default_device(C.cpu())
print('start build model')
train_reader = create_reader('data/stories_training.ctf', True)
valid_reader = create_reader('data/stories_validation.ctf', True)
model = create_basic_model(vocab_dim)
print('model created')
#C.cntk_py.set_gpumemory_allocation_trace_level(1)
basic_train(train_reader, train_reader, vocab, w2i, model, max_epochs=100, epoch_size=114845200)