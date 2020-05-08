import tensorflow as tf
import numpy as np

import config
import time

class ChatBotModel:
    """
    This class defines all the functions needed to build tensorflow model
    """
    def __init__(self, forward_only, batch_size):
        self.fw_only = forward_only
        self.batch_size = batch_size
    
    def _create_placeholders(self):
        #creates placeholder for TF graph
        print('Create placeholders')
        ##feed actual training input to the model without any predefined definition in shape
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i)) \
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape= [None], name='decoder{}'.format(i)) \
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]
        ##targets does not define <start> tag which is the start of the decoding script
        self.targets = self.decoder_inputs[1:]
    
    def _inference_(self):
        ##Create variables with given name in tf graph using get_variables
        #Sampled softmax loss can be used only when number of samples is
        #less than the vocab size
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w), 
                                              biases=b, 
                                              inputs=logits, 
                                              labels=labels, 
                                              num_sampled=config.NUM_SAMPLES, 
                                              num_classes=config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])
        
    def _create_loss(self):
        start = time.time()
        def _seq2seq_f(encoder_input, decoder_input, do_decode):
            #serattr sets the name for the object
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self) 
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_input,
                    decoder_input,
                    self.cell,
                    num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols = config.DEC_VOCAB,
                    embedding_size= config.HIDDEN_SIZE,
                    output_projection= self.output_projection,
                    feed_previous=do_decode) #True takes decoding outputs from prev layer
        
        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    self.targets,
                    self.decoder_masks, #weight for the targets
                    config.BUCKETS,
                    lambda x,y: _seq2seq_f(x,y,False),
                    softmax_loss_function=self.softmax_loss_function)
            #If output projection is used, the output should be projected for the 
            #decoding script
            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)
            
        print('Total time taken: {}'.format(time.time() - start))
        
        
    def _create_optimizer(self):
        ##Gradient descend optimizer
        with tf.variable_scope('training') as scope:
            ## Several norms - https://stackoverflow.com/questions/44796793/difference-between-tf-clip-by-value-and-tf-clip-by-global-norm-for-rnns-and-how/44798131
            #global step keeps count of number of batches seen by the model while training
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()#return all variables created by trainable=True
                self.gradient_norms = [] #l2 normalization of the variables to prevent gradient exploding
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):
                    #tf.gradients return sum(dy/dx)
                    #print(self.losses[bucket])
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM) #max gradient value
                    self.gradient_norms.append(norm)
                    ##default apply_gradients is gradientdescendent method
                    #print(self.global_step)
                    #print(clipped_grads)
                    #print(trainables)
                    self.train_ops.append(self.optimizer.apply_gradients(list(zip(clipped_grads, trainables)),  global_step=self.global_step))
                    
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()
                    
    def _create_summary(self):
        pass
    
    def build_graph(self):
        self._create_placeholders()
        self._inference_()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()