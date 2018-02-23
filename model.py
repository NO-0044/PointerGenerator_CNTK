import cntk as C
import h_p
from train import EncoderSequence,DecoderSequence

def create_model():
    embed = C.layers.Embedding(h_p.embedding_dim, name='embed')

    with C.layers.default_options(enable_self_stabilization=True):
        if h_p.num_encoder_layer > 1:
            encoder = C.layers.Sequential([
                embed,
                C.layers.Stabilizer(),
                (C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim)),
                C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim), go_backwards=True)),
                C.splice,
                C.layers.For(range(h_p.num_encoder_layer - 2), lambda: [
                    C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim)),
                ]),
                C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim), return_full_state=True)
            ])
        else:
            encoder = C.layers.Sequential([
                embed,
                C.layers.Stabilizer(),
                (C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim), return_full_state=True),
                 C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim), go_backwards=True, return_full_state=True))
            ])

    with C.layers.default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = C.layers.Stabilizer()
        stab_out = C.layers.Stabilizer()
        # output voc distribution
        voc_out = C.layers.Sequential([
            C.layers.Stabilizer(),
            C.layers.Dense(h_p.hidden_dim, activation=None),
            C.layers.Dense(h_p.vocab_dim, activation=C.softmax)
        ],name='Pvocab')

        # if encoder just has one bi-layer, then convert encoder input h,c
        if h_p.num_encoder_layer == 1:
            h_dense = C.layers.Stabilizer() >> C.layers.Dense(h_p.hidden_dim, activation=C.relu, name='en2de_h')
            c_dense = C.layers.Stabilizer() >> C.layers.Dense(h_p.hidden_dim, activation=C.relu, name='en2de_c')

        # decoder rec block
        decoder_first_block = C.layers.LSTM(h_p.hidden_dim)
        if h_p.num_decoder_layer > 1:
            decoder = C.layers.Sequential([
                C.layers.For(range(h_p.num_decoder_layer - 2), lambda: C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim))),
                C.layers.Recurrence(C.layers.LSTM(h_p.hidden_dim),return_full_state=True)
            ])
        attention_model = C.layers.AttentionModel(h_p.attention_dim, name='attention_model')

        # dense layer for Pgen
        pgen_h_att = C.layers.Stabilizer() >> C.layers.Dense(1, activation=None)
        pgen_h = C.layers.Stabilizer() >> C.layers.Dense(1, activation=None)
        pgen_x = C.layers.Stabilizer() >> C.layers.Dense(1, activation=None)

        if h_p.use_point:
            if h_p.use_coverage:
                def s2s_point_coverage(input, extended_input, de_in):
                    # encode input
                    encoded_input = encoder(input)
                    if h_p.num_encoder_layer == 1:
                        encoded_h = C.splice(encoded_input[0], encoded_input[2])
                        encoded_h = h_dense(encoded_h)
                        encoded_c = C.splice(C.sequence.last(encoded_input[1]), C.sequence.last(encoded_input[3]))
                        encoded_c = c_dense(encoded_c)
                    else:
                        encoded_h = encoded_input[0]
                        encoded_c = C.sequence.last(encoded_input[1])

                    x = embed(de_in)
                    x = stab_in(x)
                    if h_p.num_decoder_layer == 1:
                        r = C.layers.RecurrenceFrom(decoder_first_block, return_full_state=True)(
                            C.sequence.last(encoded_h), encoded_c, x)
                    else:
                        r = C.layers.RecurrenceFrom(decoder_first_block)(C.sequence.last(encoded_h), encoded_c, x)
                        r = decoder(r)

                    print('decoder ouput:%s'%repr(r))
                    @C.Function
                    def att_cov(dh,dw,dc,x):
                        h_att = attention_model(encoded_h, x)
                        att_w = h_att.attention_weights
                        c = att_w + dc
                        c = C.to_sequence(c)
                        return h_att+0*dh,att_w+0*dw,c
                    h_att, att_w, c = C.layers.Recurrence(att_cov,return_full_state=True)(r[0]).outputs
                    print('cov attention:\n%s\n%s\n%s' %(repr(h_att),repr(att_w),repr(c)))
                    #covloss = C.element_min(c,att_w)
                    voc_dist = voc_out(C.splice(r[0], h_att))
                    att_w = C.reshape(att_w, (-1), name='att_w')
                    #print('att_w:\n%s' % repr(att_w))
                    #unpacked_att_w = C.sequence.unpack(att_w, padding_value=0,no_mask_output=True, name='unpack_att_w')
                    #print('unpacked_att_w:\n%s' % repr(unpacked_att_w))
                    #extended_input_unpacked = C.sequence.unpack(extended_input, padding_value=0,no_mask_output=True,
                                                                #name='unpack_extended_in')
                    #print("extended_input_unpacked:%s" % repr(extended_input_unpacked))
                    #att_dist_unpacked = C.times(unpacked_att_w, extended_input_unpacked)
                    #print("att_dist_unpacked:%s" % repr(att_dist_unpacked))
                    #att_dist = C.to_sequence_like(att_dist_unpacked, att_w)
                    #print("att_dist:%s" % repr(att_dist))
                    #pgen = C.sigmoid(pgen_h_att(h_att) + pgen_h(r[0]) + pgen_x(x))
                    voc_dist = C.pad(voc_dist, pattern=[(0, h_p.max_extended_vocab_dim-h_p.vocab_dim)], mode=C.ops.CONSTANT_PAD, constant_value=0)
                    #extend_dist = pgen * voc_dist# + (1 - pgen) * att_dist
                    #extend_dist = stab_out(extend_dist)
                    #print('extended_dist:\n%s'%repr(extend_dist))
                    return voc_dist,h_att, att_w, c,extended_input#,covloss
            else:
                def s2s_point(input, extended_input, de_in):
                    # encode input
                    encoded_input = encoder(input)
                    if h_p.num_encoder_layer == 1:
                        encoded_h = C.splice(encoded_input[0], encoded_input[2])
                        encoded_h = h_dense(encoded_h)
                        encoded_c = C.splice(C.sequence.last(encoded_input[1]), C.sequence.last(encoded_input[3]))
                        encoded_c = c_dense(encoded_c)
                    else:
                        encoded_h = encoded_input[0]
                        encoded_c = C.sequence.last(encoded_input[1])

                    x = embed(de_in)
                    x = stab_in(x)
                    if h_p.num_decoder_layer == 1:
                        r = C.layers.RecurrenceFrom(decoder_first_block,return_full_state=True)(C.sequence.last(encoded_h), encoded_c, x)
                    else:
                        r = C.layers.RecurrenceFrom(decoder_first_block)(C.sequence.last(encoded_h), encoded_c, x)
                        r = decoder(r)

                    h_att = attention_model(encoded_h, r[0])
                    voc_dist = voc_out(C.splice(r[0], h_att))

                    att_w = C.reshape(h_att.attention_weights,(-1),name='att_w')
                    #print('att_w:\n%s'%repr(att_w))
                    unpacked_att_w,_ = C.sequence.unpack(att_w, padding_value=0,name='unpack_att_w').outputs
                    #print('unpacked_att_w:%s'%repr(unpacked_att_w))
                    extended_input_unpacked,_ = C.sequence.unpack(extended_input, padding_value=0,name='unpack_extended_in').outputs
                    #print("extended_input_unpacked:%s"%repr(extended_input_unpacked))
                    att_dist_unpacked = C.times(unpacked_att_w, extended_input_unpacked)
                    #print("att_dist_unpacked:%s"%repr(att_dist_unpacked))
                    att_dist = C.to_sequence_like(att_dist_unpacked, att_w)
                    #print("att_dist:%s"%repr(att_dist))
                    pgen = C.sigmoid(pgen_h_att(h_att) + pgen_h(r[0]) + pgen_x(x))
                    voc_extend_dist = C.pad(voc_dist, pattern=[(0, h_p.max_extended_vocab_dim-h_p.vocab_dim)], mode=C.ops.CONSTANT_PAD, constant_value=0)
                    extend_dist = pgen*voc_extend_dist + (1 - pgen)*att_dist
                    #extend_dist = stab_out(extend_dist)
                    #print('extended_dist:%s' % repr(extend_dist))
                    return extend_dist,voc_dist,pgen
        else:
            @C.Function
            def s2s(input, de_in):
                # encode input
                print("input:%s" % repr(input))
                print("de_in:%s" % repr(de_in))
                encoded_input = encoder(input)
                if h_p.num_encoder_layer == 1:
                    encoded_h = C.splice(encoded_input[0], encoded_input[2])
                    encoded_h = h_dense(encoded_h)
                    encoded_c = C.splice(C.sequence.last(encoded_input[1]), C.sequence.last(encoded_input[3]))
                    encoded_c = c_dense(encoded_c)
                else:
                    encoded_h = encoded_input[0]
                    encoded_c = C.sequence.last(encoded_input[1])

                x = embed(de_in)
                x = stab_in(x)
                if h_p.num_decoder_layer == 1:
                    r = C.layers.RecurrenceFrom(decoder_first_block, return_full_state=True)(C.sequence.last(encoded_h),
                                                                                             encoded_c, x)
                else:
                    r = C.layers.RecurrenceFrom(decoder_first_block)(C.sequence.last(encoded_h), encoded_c, x)
                    r = decoder(r)

                h_att = attention_model(encoded_h, r[0])
                voc_dist = voc_out(C.splice(r[0], h_att))
                return voc_dist

    if h_p.use_point:
        if h_p.use_coverage:
            return s2s_point_coverage
        else:
            return s2s_point
    else:
        return s2s
