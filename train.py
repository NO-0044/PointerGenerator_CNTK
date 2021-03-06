import cntk as C
from cntk.train.training_session import *
import numpy as np
import h_p
import time

def get_vocab(path):
    vocab = [w.strip().split('\t') for w in open(path, encoding='utf-8').readlines()]
    i2w = {int(i): w for w, i in vocab}
    w2i = {w: int(i) for w, i in vocab}

    return (vocab, i2w, w2i)

def create_reader(path, is_training,total_number_of_sweeps):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        en_in=C.io.StreamDef(field='S0', shape=h_p.vocab_dim, is_sparse=True),
        en_in_extended=C.io.StreamDef(field='S1', shape=h_p.max_extended_vocab_dim, is_sparse=True),
        de_in=C.io.StreamDef(field='S2', shape=h_p.vocab_dim, is_sparse=True),
        target=C.io.StreamDef(field='S3', shape=h_p.max_extended_vocab_dim, is_sparse=True)
    )), randomize=is_training,max_sweeps=total_number_of_sweeps,multithreaded_deserializer=True)


enAxis = C.Axis('enAxis')
deAxis = C.Axis('deAxis')
EncoderSequence = C.layers.SequenceOver[enAxis]
DecoderSequence = C.layers.SequenceOver[deAxis]

def create_criterion_function(model,is_train,start=None,end = None):
    if h_p.use_point:
        if is_train:
            @C.Function
            @C.layers.Signature(input=EncoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                extended_input=EncoderSequence[C.layers.Tensor[h_p.max_extended_vocab_dim]],
                                de_in=DecoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                target=DecoderSequence[C.layers.Tensor[h_p.max_extended_vocab_dim]])
            def point_criterion_train(input, extended_input, de_in, target):
                # criterion function must drop the <s> from the labels
                postprocessed_de_in = C.sequence.slice(de_in,1 ,0)
                postprocessed_target = C.sequence.slice(target,1 ,0)
                z = model(input, extended_input, postprocessed_de_in)
                print('model outputs')
                print(repr(z))
                #if h_p.use_coverage:
                    #ce = C.negate(C.reduce_sum(postprocessed_target*C.log(z[0]), axis=-1), name='log_loss')#+\
                         #h_p.cov_lambda*C.reduce_sum(z[1], axis=-1,name='cov_loss')
                #else:
                ce = C.negate(C.reduce_sum(postprocessed_target*C.log(z[0]), axis=-1), name='loss')
                error = C.metrics.classification_error(z[0],postprocessed_target)
                #error = z[2]
                print(repr(ce))
                return (ce,error)
            return point_criterion_train
        else:
            @C.Function
            @C.layers.Signature(input=EncoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                extended_input=EncoderSequence[C.layers.Tensor[h_p.max_extended_vocab_dim]],
                                de_in=DecoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                target=DecoderSequence[C.layers.Tensor[h_p.max_extended_vocab_dim]])
            def point_criterion_eval(input, extended_input, de_in, target):
                # criterion function must drop the <s> from the labels
                postprocessed_de_in = C.sequence.slice(de_in,1 ,0)
                postprocessed_target = C.sequence.slice(target,1 ,0)
                z = model(input, extended_input, postprocessed_de_in)
                #if h_p.use_coverage:
                    #ce = C.negate(C.reduce_sum(postprocessed_target*C.log(z[0]), axis=-1), name='log_loss')#+\
                         #h_p.cov_lambda*C.reduce_sum(z[1], axis=-1,name='cov_loss')
                #else:
                #unfold = C.layers.UnfoldFrom(lambda history: model(input,extended_input,history)[1] >> C.hardmax,
                               #until_predicate=lambda w: w[...,end],
                               #length_increase=0.5)
                #z = unfold(initial_state=start, dynamic_axes_like=input)
                ce = C.negate(C.reduce_sum(postprocessed_target*C.log(z[0]), axis=-1), name='loss')
                error = C.metrics.classification_error(z[0],postprocessed_target)
                return (ce,error)
            return point_criterion_eval
    else:
        if is_train:
            @C.Function
            @C.layers.Signature(input=EncoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                de_in=DecoderSequence[C.layers.Tensor[h_p.vocab_dim]])
            def criterion_train(input, de_in):
                # criterion function must drop the <s> from the labels
                postprocessed_de_in = C.sequence.slice(de_in, 1, 0)
                z = model(input, postprocessed_de_in)
                print(repr(z))
                ce = C.negate(C.reduce_sum(postprocessed_de_in * C.log(z), axis=-1), name='loss')
                error = C.metrics.classification_error(postprocessed_de_in,z)
                return (ce,error)
            return criterion_train
        else:
            @C.Function
            @C.layers.Signature(input=EncoderSequence[C.layers.Tensor[h_p.vocab_dim]],
                                de_in=DecoderSequence[C.layers.Tensor[h_p.vocab_dim]])
            def criterion_eval(input, de_in):
                postprocessed_de_in = C.sequence.slice(de_in, 1, 0)
                unfold = C.layers.UnfoldFrom(lambda history: model(input, history) >> C.hardmax,
                               until_predicate=lambda w: w[...,end],
                               length_increase=0.5)
                z = unfold(initial_state=start, dynamic_axes_like=input)
                print(repr(z))
                ce = C.negate(C.reduce_sum(postprocessed_de_in * C.log(z), axis=-1), name='loss')
                error = C.metrics.classification_error(postprocessed_de_in,z)
                return (ce,error)
            return criterion_eval

def create_model_train(s2smodel,sentence_start):
    # model used in training (history is known from labels)
    # note: the labels must NOT contain the initial <s>
    if h_p.use_point:
        @C.Function
        def point_model_train(input, extended_input, de_in): # (input*, labels*) --> (word_logp*)

            # The input to the decoder always starts with the special label sequence start token.
            # Then, use the previous value of the label sequence (for training) or the output (for execution).
            past_labels = C.layers.Delay(initial_state=sentence_start)(de_in)
            return s2smodel(input, extended_input, past_labels)
    else:
        @C.Function
        def model_train(input, de_in):  # (input*, labels*) --> (word_logp*)

            # The input to the decoder always starts with the special label sequence start token.
            # Then, use the previous value of the label sequence (for training) or the output (for execution).
            past_labels = C.layers.Delay(initial_state=sentence_start)(de_in)
            return s2smodel(input, past_labels)

    if h_p.use_point:
        return point_model_train
    else:
        return model_train

def create_trainer(s2smodel,model_train,criterion,epoch_size, num_quantization_bits, block_size, warm_up, progress_writers):
    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule       = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=epoch_size)
    mms               = [0]*20 + [0.9983347214509387]*20 + [0.9991670137924583]
    mm_schedule       = C.learners.momentum_schedule_per_sample(mms, epoch_size=epoch_size)
    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")
    local_learner = C.learners.momentum_sgd(model_train.parameters,
                                            lr_schedule, mm_schedule)
    if block_size != None:
        parameter_learner = C.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = C.train.distributed.data_parallel_distributed_learner(local_learner, 
                                                                                  num_quantization_bits=num_quantization_bits, 
                                                                                  distributed_after=warm_up)
    # Create trainer
    return C.Trainer(None,criterion, parameter_learner, progress_writers)

def cv_call(index, average_error, cv_num_samples, cv_num_minibatches):
    #print("on validation:")
    #print("index:%d\taverage error:%f"%(index,average_error))
    return False

def para_train(vocab, w2i, s2smodel, max_epochs, epoch_size, minibatch_size):
    train_reader = create_reader('data/train.ctf', False,max_epochs)
    val_reader = create_reader('data/valid.ctf',False,1)
    sentence_start = C.Constant(np.array([i == w2i['<s>'] for i in range(h_p.vocab_dim)], dtype=np.float32))
    sentence_end_index = vocab.index('</s>')
    model_train = create_model_train(s2smodel,sentence_start)
    criterion = create_criterion_function(model_train,True)
    val_criterion = create_criterion_function(model_train,False)
    distributed_sync_report_freq = None
    progress_writers = [C.logging.ProgressPrinter(
        freq=50,
        tag='Training',
        log_to_file=None,
        rank=C.train.distributed.Communicator.rank(),
        gen_heartbeat=False,
        metric_is_pct = False,
        num_epochs=max_epochs,
        distributed_freq=distributed_sync_report_freq)]
    progress_writers.append(C.logging.TensorBoardProgressWriter(
        freq=100,
        log_dir="tensorboard_log/",
        rank=C.train.distributed.Communicator.rank(),
        model=None))
    trainer = create_trainer(s2smodel,model_train,criterion,epoch_size,32,3200,0,progress_writers)
    if h_p.use_point:
        input_map = {criterion.arguments[0]: train_reader.streams.en_in,
                     criterion.arguments[1]: train_reader.streams.en_in_extended,
                     criterion.arguments[2]: train_reader.streams.de_in,
                     criterion.arguments[3]: train_reader.streams.target}
    else:
        input_map = {criterion.arguments[0]: train_reader.streams.en_in,
                     criterion.arguments[1]:train_reader.streams.de_in}
    val_cfg = CrossValidationConfig(
                                      minibatch_source = val_reader,
                                      frequency=(1,C.DataUnit.minibatch),
                                      minibatch_size = 3000,
                                      #max_samples = 9999,
                                      callback = cv_call,
                                      criterion = val_criterion)
    training_session(
        trainer=trainer, mb_source = train_reader,
        model_inputs_to_streams = input_map,
        max_samples = max_epochs*epoch_size, 
        mb_size = minibatch_size,
        progress_frequency = epoch_size,
        checkpoint_config = CheckpointConfig(frequency = 500,
                                             filename = "checkpoint/point_model",
                                             restore = False),
        cv_config = val_cfg
    ).train()
    C.train.distributed.Communicator.finalize()

def train(vocab, w2i, s2smodel, max_epochs, epoch_size,minibatch_size):
    # create the training wrapper for the s2smodel, as well as the criterion function
    train_reader = create_reader('data/train.ctf', False,max_epochs)
    val_reader = create_reader('data/valid.ctf', False,1)
    sentence_start = np.array([i == w2i['<s>'] for i in range(h_p.vocab_dim)], dtype=np.float32)
    model_train = create_model_train(s2smodel,sentence_start)
    criterion = create_criterion_function(model_train, True)
    val_criterion = create_criterion_function(model_train, False)
    lr = 0.001
    learner = C.momentum_sgd(model_train.parameters,
                             # apply the learning rate as if it is a minibatch of size 1
                             lr=C.learning_parameter_schedule_per_sample([lr] * 2 + [lr / 2] * 3 + [lr / 4],
                                                                         epoch_size),
                             momentum=C.momentum_schedule(0.9366416204111472, minibatch_size=minibatch_size),
                             gradient_clipping_threshold_per_sample=2.3,
                             gradient_clipping_with_truncation=True)
    progress_writers = [C.logging.ProgressPrinter(
        freq=50,
        tag='Training',
        log_to_file="log/log",
        num_epochs=max_epochs)]
    progress_writers.append(C.logging.TensorBoardProgressWriter(
        freq=100,
        log_dir="tensorboard_log/",
        model=None))
    trainer = C.Trainer(None, criterion, learner,progress_writers=progress_writers)
    if h_p.use_point:
        input_map = {criterion.arguments[0]: train_reader.streams.en_in,
                     criterion.arguments[1]: train_reader.streams.en_in_extended,
                     criterion.arguments[2]: train_reader.streams.de_in,
                     criterion.arguments[3]: train_reader.streams.target}
    else:
        input_map = {criterion.arguments[0]: train_reader.streams.en_in,
                     criterion.arguments[1]: train_reader.streams.de_in}
    val_cfg = CrossValidationConfig(
                                      minibatch_source = val_reader,
                                      frequency=(100,C.DataUnit.minibatch),
                                      minibatch_size = 4000,
                                      callback=cv_call,
                                      criterion = val_criterion)
    training_session(
        trainer=trainer, mb_source = train_reader,
        model_inputs_to_streams = input_map,
        max_samples = max_epochs*epoch_size,
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(frequency = int(epoch_size/50),
                                             filename = "checkpoint/point_checkpoint",
                                             restore = True),
        cv_config = val_cfg
    ).train()
    s2smodel.save("origin_model.cmf")
