from model import create_model
#from train import train,create_reader,get_vocab
from train import train,create_reader,get_vocab,para_train
import cntk as C
import numpy as np
import h_p
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--use_point",help="whether to use point generator",action="store_true")
    parser.add_argument("-e","--encoder_layer_num",help="encoder layer num",type=int,default=1)
    parser.add_argument("-d", "--decoder_layer_num", help="decoder layer num", type=int, default=1)
    parser.add_argument("-H", "--hidden_dim", help="hidden dim", type=int, default=256)
    parser.add_argument("-a", "--att_dim", help="attention dim", type=int, default=128)
    parser.add_argument("-b", "--minibatch_size", help="mini batch", type=int, default=3000)
    parser.add_argument("--epoch_size",type=int,default=638924687)
    parser.add_argument("-c","--use_coverage",help="whether to use coverage",action="store_true")
    args = parser.parse_args()
    if args.use_point:
        h_p.use_point = True
    if args.use_coverage:
        h_p.use_coverage = True
    h_p.hidden_dim = args.hidden_dim
    h_p.attention_dim = args.att_dim
    h_p.num_decoder_layer = args.decoder_layer_num
    h_p.num_encoder_layer = args.encoder_layer_num
    print('start build voc')
    vocab, i2w, w2i = get_vocab('data/voc_50k.txt')
    vocab = [x[0] for x in vocab]
    h_p.sentence_start = np.array([i == w2i['<s>'] for i in range(h_p.vocab_dim)], dtype=np.float32)
    h_p.sentence_end_index = vocab.index('</s>')
    print('voc builded')
    if h_p.use_point:
        if h_p.use_coverage:
            print('start build model with point and coverage')
        else:
            print('start build model with point')
    else:
        print('start build basic model')
    print('---------------------')
    print('encoder layers:%d' % h_p.num_encoder_layer)
    print('decoder layers:%d' % h_p.num_decoder_layer)
    print('hidden dim:%d' % h_p.hidden_dim)
    print('attention dim:%d' % h_p.attention_dim)
    print('voc size:%d' % h_p.vocab_dim)
    print('max extended voc:%d' % (h_p.max_extended_vocab_dim-h_p.vocab_dim))
    print('---------------------')
    model = create_model()
    print('model created')
    #C.cntk_py.set_gpumemory_allocation_trace_level(1)
    para_train(vocab, w2i, model, max_epochs=2, epoch_size=638924687, minibatch_size=args.minibatch_size)
    #train(vocab, w2i, model, max_epochs=1, epoch_size=args.epoch_size, minibatch_size=args.minibatch_size)
