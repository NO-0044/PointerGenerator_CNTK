import cntk as C
from h_p import *

def create_model(vocab_dim):
    embed = C.layers.Embedding(embedding_dim, name='embed')

    with C.layers.default_options(enable_self_stabilization=True):
        encode = C.layers.Sequential([
            embed,
            C.layers.Stabilizer(),
            (C.layers.Recurrence(C.layers.LSTM(hidden_dim), return_full_state=True),
             C.layers.Recurrence(C.layers.LSTM(hidden_dim), return_full_state=True,go_backwards=True)),
        ])

    with C.layers.default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = C.layers.Stabilizer()
        stab_out = C.layers.Stabilizer()
        # output voc distribution
        voc_out = C.layers.Sequential([
            C.layers.Dense(hidden_dim, activation=None),
            C.layers.Dense(vocab_dim, activation=C.softmax)
        ],name='Pvocab')

        # convert encoder input h,c
        h_dense = C.layers.Dense(hidden_dim, activation=C.relu, name='en2de_h')
        c_dense = C.layers.Dense(hidden_dim, activation=C.relu, name='en2de_c')

        # decoder rec block
        rec_block = C.layers.LSTM(hidden_dim)
        attention_model = C.layers.AttentionModel(attention_dim, name='attention_model')

        # dense layer for Pgen
        pgen_h_att = C.layers.Dense(1, activation=None)
        pgen_h = C.layers.Dense(1, activation=None)
        pgen_x = C.layers.Dense(1, activation=None)

        @C.Function
        def decode(input, extended_input, de_in):
            # encode input
            encoded_input = encode(input)
            encoded_h = C.splice(encoded_input[0], encoded_input[2])
            encoded_h = h_dense(encoded_h)
            encoded_c = C.splice(encoded_input[1][-1], encoded_input[3][-1])
            encoded_c = c_dense(encoded_c)

            x = embed(de_in)
            x = stab_in(x)
            r = C.layers.RecurrenceFrom(rec_block, return_full_state=True)(encoded_h, encoded_c, x)

            h_att = attention_model(encoded_input.outputs[0], r[0])
            voc_dist = voc_out(C.splice(r[0], h_att))

            #att_w = h_att.attention_weights
            att_w = C.sequence.unpack(h_att.attention_weights, padding_value=0, no_mask_output=True)
            #tmp = C.sequence.broadcast_as(C.sequence.unpack(extended_input, padding_value=0, no_mask_output=True), att_w)
            tmp = C.sequence.unpack(extended_input, padding_value=0, no_mask_output=True)
            #att_dist = C.reduce_sum(att_w*tmp, axis=0)
            att_dist = C.reduce_sum(C.element_select(tmp, att_w, tmp), axis=0)
            att_dist = C.reshape(att_dist, (), 0, 1, name='att_dist')

            pgen = C.sigmoid(pgen_h_att(h_att) + pgen_h(r[0]) + pgen_x(x))
            voc_dist = C.pad(voc_dist, pattern=[(0, 400)], mode=C.ops.CONSTANT_PAD, constant_value=0)
            extend_dist = pgen*voc_dist + (1 - pgen)*att_dist
            extend_dist = stab_out(extend_dist)

            return (extend_dist, att_dist)

    return decode

def create_basic_model(vocab_dim):

    num_layers = 3
    embed = C.layers.Embedding(embedding_dim, name='embed')

    with C.layers.default_options(enable_self_stabilization=True):
        encode = C.layers.Sequential([
            embed,
            C.layers.Stabilizer(),
            C.layers.For(range(num_layers-1), lambda:[
                (C.layers.Recurrence(C.layers.LSTM(hidden_dim)),
                 C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=True)),
                C.splice
            ]),
            (C.layers.Recurrence(C.layers.LSTM(hidden_dim), return_full_state=True),
             C.layers.Recurrence(C.layers.LSTM(hidden_dim), return_full_state=True, go_backwards=True)),
        ])

    with C.layers.default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = C.layers.Stabilizer()
        stab_out = C.layers.Stabilizer()
        # decode
        decode_first = C.layers.RecurrenceFrom(C.layers.LSTM(hidden_dim))
        decode = C.layers.For(range(num_layers - 1), lambda:C.layers.Recurrence(C.layers.LSTM(hidden_dim)))
        # output voc distribution
        voc_out = C.layers.Dense(vocab_dim, activation=None)

        # convert encoder input h,c
        h_dense = C.layers.Dense(hidden_dim, activation=C.relu, name='en2de_h')
        c_dense = C.layers.Dense(hidden_dim, activation=C.relu, name='en2de_c')

        attention_model = C.layers.AttentionModel(attention_dim, name='attention_model')

        @C.Function
        def decoder(en_in, de_in):
            # encode input
            encoded_input = encode(en_in)
            encoded_h = C.splice(encoded_input[0], encoded_input[2])
            encoded_h = h_dense(encoded_h)
            encoded_c = C.splice(encoded_input[1][-1], encoded_input[3][-1])
            encoded_c = c_dense(encoded_c)

            x = embed(de_in)
            x = stab_in(x)
            r = decode_first(encoded_h, encoded_c, x)
            r = decode(r)

            h_att = attention_model(encoded_h, r[0])
            voc_dist = voc_out(C.splice(r[0], h_att))

            return voc_dist

    return decoder
