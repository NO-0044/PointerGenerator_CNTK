import cntk as C
import numpy as np
from h_p import *
import time

def get_vocab(path):
    vocab = [w.strip().split('\t') for w in open(path, encoding='utf-8').readlines()]
    i2w = {int(i): w for w, i in vocab}
    w2i = {w: int(i) for w, i in vocab}

    return (vocab, i2w, w2i)

def create_reader(path, is_training):
    print(vocab_dim)
    print(max_extended_vocab_dim)
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        en_in=C.io.StreamDef(field='S0', shape=vocab_dim, is_sparse=True),
        de_in=C.io.StreamDef(field='S2', shape=vocab_dim, is_sparse=True),
    )), randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)


enAxis = C.Axis('enAxis')
deAxis = C.Axis('deAxis')
EncoderSequence = C.layers.SequenceOver[enAxis]
DecoderSequence = C.layers.SequenceOver[deAxis]

def create_basic_criterion_function(model):
    @C.Function
    @C.layers.Signature(input=EncoderSequence[C.layers.Tensor[vocab_dim]],
                        de_in=DecoderSequence[C.layers.Tensor[vocab_dim]])
    def criterion(input, de_in):
        # criterion function must drop the <s> from the labels
        postprocessed_de_in = C.sequence.slice(de_in,1 ,0)
        z = model(input, postprocessed_de_in)
        print(repr(z))
        ce = C.cross_entropy_with_softmax(z, postprocessed_de_in)
        print(repr(ce))
        return ce

    return criterion

def create_basic_model_train(s2smodel,sentence_start):
    # model used in training (history is known from labels)
    # note: the labels must NOT contain the initial <s>
    @C.Function
    def model_train(input, de_in): # (input*, labels*) --> (word_logp*)

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        past_labels = C.layers.Delay(initial_state=C.Constant(sentence_start))(de_in)
        return s2smodel(input, past_labels)
    return model_train

def basic_train(train_reader, valid_reader, vocab, w2i, s2smodel, max_epochs, epoch_size):
    # create the training wrapper for the s2smodel, as well as the criterion function
    sentence_start = np.array([i == w2i['<s>'] for i in range(vocab_dim)], dtype=np.float32)
    model_train = create_basic_model_train(s2smodel,sentence_start)
    criterion = create_basic_criterion_function(model_train)

    # also wire in a greedy decoder so that we can properly log progress on a validation example
    # This is not used for the actual training process.
    # model_greedy = create_model_greedy(s2smodel)

    # Instantiate the trainer object to drive the model training
    minibatch_size = 3000
    lr = 0.001 if use_attention else 0.005
    learner = C.fsadagrad(model_train.parameters,
                          # apply the learning rate as if it is a minibatch of size 1
                          lr=C.learning_parameter_schedule_per_sample([lr] * 2 + [lr / 2] * 3 + [lr / 4], epoch_size),
                          momentum=C.momentum_schedule(0.9366416204111472, minibatch_size=minibatch_size),
                          gradient_clipping_threshold_per_sample=2.3,
                          gradient_clipping_with_truncation=True)
    trainer = C.Trainer(None, criterion, learner)

    # Get minibatches of sequences to train with and perform model training
    total_samples = 0
    eval_freq = 100

    # print out some useful training information
    C.logging.log_number_of_parameters(model_train);
    print()
    progress_printer = C.logging.ProgressPrinter(tag='Training')

    # a hack to allow us to print sparse vectors
    # sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    start = time.clock()
    last = start
    n = 1
    for epoch in range(max_epochs):
        mbs = 0
        while total_samples < epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)
            #print(mb_train[train_reader.streams.en_in].shape)

            # do the training
            trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.en_in],
                                     criterion.arguments[1]: mb_train[train_reader.streams.de_in]})

            progress_printer.update_with_trainer(trainer, with_metric=False)  # log progress
            total_samples += mb_train[train_reader.streams.en_in].num_samples
            if mbs == 100:
                print("avg_loss since last:%f"%progress_printer.avg_loss_since_last())
                print('processed %d output samples, %d input sample in total'%(progress_printer.samples_since_last,total_samples) + " :{0:.0f}%".format(total_samples*1.0/epoch_size * 100))
                print('%f seconds, total %f hours'%(time.clock()-last,(time.clock() - start)/3600))
                print('----------------------------------------------')
                progress_printer.reset_last()
                last = time.clock()
                mbs = 0
            #if time.clock() - start >= 1800*n:
                #print('processed %d samples, cost %f seconds'%(total_samples,time.clock() - start))
                #n += 1
            mbs += 1
        # log a summary of the stats for the epoch
        print('------------------------------------')
        print('finished epoch %d'%epoch)
        progress_printer.epoch_summary(with_metric=False)
        print(time.clock()-start)
        print('------------------------------------')

    # done: save the final model
    model_path = "model_%d.cmf" % epoch
    print("Saving final model to '%s'" % model_path)
    s2smodel.save(model_path)
    print("%d epochs complete." % max_epochs)
