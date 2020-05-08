"""
This file has functions which are used in building the model.
"""
import os
import config
import random
import numpy as np

def load_data(encoding_filename, decoding_filename, max_training_size = None):
    encoding_file = open(os.path.join(config.PROCESSED_PATH, encoding_filename), 'r')
    decoding_file = open(os.path.join(config.PROCESSED_PATH, decoding_filename), 'r')
    encode, decode = encoding_file.readline(), decoding_file.readline()
    #create list with empty buckets
    data_buckets = [[] for i in range(len(config.BUCKETS))]
    i = 0
    while encode and decode:
        if (i+1) % 10000 == 0:
            print('Bucketing conversation number', i)
        encode_ids = [int(ids) for ids in encode.split()]
        decode_ids = [int(ids) for ids in decode.split()]

        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encoding_file.readline(), decoding_file.readline()
        i += 1

    return data_buckets

def _pad_input(_input, size):
    return _input + [config.PAD_ID] * (size - len(_input))

def _reshape_batches(_input, size, batch_size):
    batch_inputs = []
    for length in range(size):
        batch_inputs.append(np.array([_input[batch_id][length] for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs

def get_batch(data_bucket,bucket_id,batch_size = 1):
    """Returns one batch to feed into the model"""
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    padded_encoder, padded_decoder = [], []
    for _ in range(batch_size):
        encoder, decoder = random.choice(data_bucket)
        ##Input of seq2seq encoder is reversed, so the last word will be seen as first word in decoder
        padded_encoder.append(list(reversed(_pad_input(encoder, encoder_size))))
        padded_decoder.append(_pad_input(decoder, decoder_size))

    ##create mini-batches of the input based on batch size
    batch_encoder_inputs = _reshape_batches(padded_encoder, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batches(padded_decoder, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    # zero padded values will be processed normally if not masked
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = padded_decoder[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks