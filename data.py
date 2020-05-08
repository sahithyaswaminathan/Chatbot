"""
This file contains the code for the pre-processing
of the text before it is sent to the model
"""

import os
import random
import numpy as np
import re
import config
import ast

def get_lines():
    index2line = {}
    line_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    #print(line_path)
    with open(line_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            convo = line.split(' +++$+++ ')
            if len(convo) == 5:
                if convo[4][-1] == '\n':
                    convo[4] = convo[4][:-1]
                index2line[convo[0]] = convo[4]
    return index2line

def get_convo():
    convolist = []
    convo_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    with open(convo_path, 'r') as file:
        lines = file.readlines()[:5000]
        for line in lines:
            items = line.split(' +++$+++ ')
            #print(type(items[3]))
            if len(items) == 4:
                # converts string(list) to list
                #items[3] = items[3].decode("utf-8")
                convo = ast.literal_eval(items[3])
            convolist.append(convo)
    return convolist

def get_question_answers(index2conv, convolist):
    questions, answers = [] , []
    for conv in convolist:
        for index, items in enumerate(conv[:-1]):
            questions.append(index2conv[conv[index]])
            answers.append(index2conv[conv[index+1]])
    assert len(questions) == len(answers)
    return questions, answers

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def prepare_dataset(question, answer):
    #try:
    make_dir(config.PROCESSED_PATH)
    #except OSError:
    #    pass
    #Get random ids for test dataset
    test_ids = random.sample(range(len(question)), config.TEST_SIZE)
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    #open 4 files in write mode
    files = []
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename), 'w'))

    for i in range(len(question)):
        if i in test_ids:
            files[2].write(question[i] + '\n')
            files[3].write(answer[i] + '\n')
        else:
            files[0].write(question[i] + '\n')
            files[1].write(answer[i] + '\n')

    for file in files:
        file.close()
    #return
def preprocess_raw_data():
    print('Starting the movie convo preprocessing')
    index2conv = get_lines()
    convolist = get_convo()
    question, answers = get_question_answers(index2conv, convolist)
    #print(question[:10])
    #print(answers[:10])
    ##Prepare dataset with questions and answers
    prepare_dataset(question, answers)
    #return

def build_tokenizer(line, normalize_digit = True):
    #convert string to standard format
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    ##re.compile creates a pattern object which later can be used to perform
    #other reg functions
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digit:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(filename, normalize_digit = True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))
    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in build_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    sorted_vocab = sorted(vocab, key = vocab.get, reverse = True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                with open('config.py', 'a') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('\n' + 'ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1

def load_vocab(filepath):
    with open(filepath, 'r') as f:
        words = f.read().splitlines() #memory inefficient: but reads without \n character
    return words, {words[i]: i for i in range(len(words))} #get id for each words in vocab

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in build_tokenizer(line)]

def token2id(data, mode):
    """
    Tokens are converted to corresponding index in the vocab
    :param data:
    :param mode:
    :return:
    """
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode
    vocab_path = 'vocab.'+mode

    _,vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')

    lines = in_file.read().splitlines()
    for line in lines:
        #for decoder, check only the start and end <s> & </s>
        if mode == 'dec':
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'dec':
            ids.append(vocab['<\s>'])

        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


def process_data():
    """
    Creates vocabulary and prepares raw data for the model
    :return:
    """
    print('Preparing data for model ready')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

if __name__ == '__main__':
    preprocess_raw_data()
    process_data()