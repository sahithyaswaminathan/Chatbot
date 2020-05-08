"""
This python file contains necessary filename and
Hyperparameters needed for the sequence to sequence model to
build chatbot

Inspired from - https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/assignments/chatbot/config.py

Original paper:
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

"""
DATA_PATH = 'cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

'''
Number of words in vocal threshold
'''
THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TEST_SIZE = 250

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]

#not using it anywhere
CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),\
                ("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),\
                ("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),\
                ("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 32

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 124
ENC_VOCAB = 25746
DEC_VOCAB = 25979

ENC_VOCAB = 25751
DEC_VOCAB = 25964

ENC_VOCAB = 25752
DEC_VOCAB = 25982

ENC_VOCAB = 25754
DEC_VOCAB = 25967

ENC_VOCAB = 25752
DEC_VOCAB = 25968

ENC_VOCAB = 25757
DEC_VOCAB = 25954

ENC_VOCAB = 25739
DEC_VOCAB = 25953

ENC_VOCAB = 25743
DEC_VOCAB = 25976

ENC_VOCAB = 25722
DEC_VOCAB = 25953

ENC_VOCAB = 25739
DEC_VOCAB = 25977

ENC_VOCAB = 25752
DEC_VOCAB = 25980

ENC_VOCAB = 25735
DEC_VOCAB = 25975

ENC_VOCAB = 1422
DEC_VOCAB = 1402

ENC_VOCAB = 1427
DEC_VOCAB = 1416

ENC_VOCAB = 1446
DEC_VOCAB = 1439

ENC_VOCAB = 4872
DEC_VOCAB = 4806

ENC_VOCAB = 2384
DEC_VOCAB = 2373

ENC_VOCAB = 1409
DEC_VOCAB = 1412

ENC_VOCAB = 4884
DEC_VOCAB = 4807
