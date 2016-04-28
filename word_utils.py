"""Utilities for manipulating data from Google Knowledge Graph, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import random
import math
import os
import re
import time
import tarfile

from six.moves import urllib
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_POS = "_POS"
_NEG = "_NEG"
_GO = "_GO"

_START_VOCAB = [_PAD, _UNK, _POS, _NEG, _GO]

PAD_ID = 0
UNK_ID = 1
POS_ID = 2
NEG_ID = 3
GO_ID = 4

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
_buckets = [(80, 1), (160, 1), (320, 1), (640, 1), (1280,1)]
unsup_buckets = [(80, 80), (160, 160), (320, 320), (640, 640), (1280,1280)]


class TextLoader():

  def __init__(self,data_dir,batch_size, vocab_size):
    self.buckets = _buckets
    self.unsup_buckets = unsup_buckets
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.pos_path = self.data_dir + ("/train/pos.ids%d" % self.vocab_size)
    self.neg_path = self.data_dir + ("/train/neg.ids%d" % self.vocab_size)
    self.unsup_path = self.data_dir + ("/train/unsup.ids%d" % self.vocab_size)
    self.vocab_path = self.data_dir + ("/vocab%d" % self.vocab_size)
    self.pos_dev_path = self.data_dir + ("/test/pos.ids%d" % self.vocab_size)
    self.neg_dev_path = self.data_dir + ("/test/neg.ids%d" % self.vocab_size)
    self.dev_counter = -1
    
    _, self.re_vocab = self.initialize_vocabulary(self.vocab_path)
    if not (os.path.exists(self.pos_path) and os.path.exists(self.neg_path) and os.path.exists(self.vocab_path) and 
    os.path.exists(self.pos_dev_path) and os.path.exists(self.neg_dev_path) and os.path.exists(self.unsup_path)):
      print ('preparing data') 
      self.pos_path, self.neg_path, self.vocab_path, self.pos_dev_path, self.neg_dev_path, self.unsup_path = self.prepare_data(self.data_dir,self.vocab_size)
    else:
      print ('loading prepared file')
    self.data_set, self.valid_set = self.read_data(self.pos_path, self.neg_path)
    self.train_bucket_sizes = [len(self.data_set[b]) for b in xrange(len(_buckets))]
    self.train_total_size = float(sum(self.train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    self.train_buckets_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.train_total_size
                           for i in xrange(len(self.train_bucket_sizes))]

    #############calculating valid bucket scale#############
    self.valid_bucket_sizes = [len(self.valid_set[b]) for b in xrange(len(_buckets))]
    self.valid_total_size = float(sum(self.valid_bucket_sizes))
    self.valid_buckets_scale = [sum(self.valid_bucket_sizes[:i + 1]) / self.valid_total_size
                           for i in xrange(len(self.valid_bucket_sizes))]                           
    self.read_dev(self.pos_dev_path, self.neg_dev_path)

    self.unsup_set, self.unsup_valid_set = self.read_unsup_data(self.unsup_path)
    #############calculating unsupervised bucket scale#############
    self.unsup_bucket_sizes = [len(self.unsup_set[b]) for b in xrange(len(unsup_buckets))]
    self.unsup_total_size = float(sum(self.unsup_bucket_sizes))
    self.unsup_buckets_scale = [sum(self.unsup_bucket_sizes[:i + 1]) / self.unsup_total_size
                           for i in xrange(len(self.unsup_bucket_sizes))]                              


  def combine_file(self,data_dir):
    """
    Combine data file from single txt to a file with a comment per line
    return conbimed file name
    """
    data = []
    for subdir, dirs, files in os.walk(data_dir):
      for afile in files:
        with gfile.GFile(os.path.join(subdir, afile), mode="r") as f:
          data.append(f.readline())
    print ('reading directory complete')
    with gfile.GFile(data_dir+'.txt', mode="w") as f:
      for element in data:
        f.write(element + '\n')
      print ('Combined file'+os.path.basename(data_dir))
    return os.path.basename(data_dir)


  def basic_tokenizer(self, sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
      words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


  def create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size,
                        tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
      print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
      vocab = {}
      with gfile.GFile(data_path+"/pos.txt", mode="r") as f:
        counter = 0
        for line in f:
          counter += 1
          if counter % 100000 == 0:
            print("  processing line %d" % counter)
          tokens = tokenizer(line) if tokenizer else self.basic_tokenizer(line)
          for w in tokens:
            word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
      with gfile.GFile(data_path+"/neg.txt", mode="r") as f:
        for line in f:
          counter += 1
          if counter % 100000 == 0:
            print("  processing line %d" % counter)
          tokens = tokenizer(line) if tokenizer else self.basic_tokenizer(line)
          for w in tokens:
            word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
      with gfile.GFile(data_path+"/unsup.txt", mode="r") as f:
        for line in f:
          counter += 1
          if counter % 100000 == 0:
            print("  processing line %d" % counter)
          tokens = tokenizer(line) if tokenizer else self.basic_tokenizer(line)
          for w in tokens:
            word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
          vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
          for w in vocab_list:
            vocab_file.write(w + "\n")


  def initialize_vocabulary(self, vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
      rev_vocab = []
      with gfile.GFile(vocabulary_path, mode="r") as f:
        rev_vocab.extend(f.readlines())
      rev_vocab = [line.strip() for line in rev_vocab]
      vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
      return vocab, rev_vocab
    else:
      raise ValueError("Vocabulary file %s not found.", vocabulary_path)


  def sentence_to_token_ids(self, sentence, vocabulary,
                            tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids to word sentence.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
      words = tokenizer(sentence)
    else:
      words = self.basic_tokenizer(sentence)
    if not normalize_digits:
      return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]

  def token_ids_to_sentence(self, batch, bucket_id):
    """Convert a string to list of integers representing token-ids.

    For example, a tokenized sentence [1, 2, 4, 7]  may become into
    ["I", "have", "a", "dog"] and with vocabulary {1 : "I", 2 : "have",
    4:"a", 7: "dog"} this function will return .

    Args:
      sentence: a string, the sentence of token-ids.

    Returns:
      a list of word, the words for the sentence.
    """
    sentence_converted = []
    for outputs in range(self.batch_size):
      sentence_converted.append([])
    for outputs in range(self.buckets[bucket_id][0]):
      for output in range(self.batch_size):
        sentence_converted[output].append(self.re_vocab[batch[outputs][output]])
    return sentence_converted

  def data_to_token_ids(self, data_path, target_path, vocabulary_path,
                        tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
      print("Tokenizing data in %s" % data_path)
      vocab, _ = self.initialize_vocabulary(vocabulary_path)
      with gfile.GFile(data_path, mode="r") as data_file:
        with gfile.GFile(target_path, mode="w") as tokens_file:
          counter = 0
          for line in data_file:
            counter += 1
            if counter % 100000 == 0:
              print("  tokenizing line %d" % counter)
            token_ids = self.sentence_to_token_ids(line, vocab, tokenizer,
                                              normalize_digits)
            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


  def prepare_data(self, data_dir, vocabulary_size):
    """Get knowledge graph data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      vocabulary_size: size of the vocabulary to create and use.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for pos training data-set,
        (2) path to the token-ids for neg training data-set,
        (3) path to the vocabulary file.
        (4) path to the token-ids for pos dev(validation) data-set,
        (5) path to the token-ids for neg dev(validation) data-set,
        (6) path to the token-ids for unsupervised data-set,

    """
    train_path = './data/train/'
    dev_path = './data/test/'
    # Create vocabularies of the appropriate sizes.
    pos_path = self.combine_file(train_path+'pos')
    print (pos_path + ' combined' )
    neg_path = self.combine_file(train_path+'neg')
    print (neg_path + ' combined' )
    unsup_path = self.combine_file(train_path+'unsup')
    print (unsup_path + ' combined' )
    
    pos_path = self.combine_file(dev_path+'pos')
    print (pos_path + ' combined' )
    neg_path = self.combine_file(dev_path+'neg')
    print (neg_path + ' combined' )
    upsup_path = self.combine_file(dev_path+'unsup')
    print (upsup_path + ' combined' )    
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    self.create_vocabulary(vocab_path, train_path, vocabulary_size)

    # Create token ids for the training data.
    pos_train_ids_path = train_path + ("/pos.ids%d" % vocabulary_size)
    neg_train_ids_path = train_path + ("/neg.ids%d" % vocabulary_size)
    unsup_train_ids_path = train_path + ("/unsup.ids%d" % vocabulary_size)
    self.data_to_token_ids(train_path + "/pos.txt", pos_train_ids_path, vocab_path)
    self.data_to_token_ids(train_path + "/neg.txt", neg_train_ids_path, vocab_path)
    self.data_to_token_ids(train_path + "/unsup.txt", unsup_train_ids_path, vocab_path)

    # Create token ids for the development data.
    pos_dev_ids_path = dev_path + ("/pos.ids%d" % vocabulary_size)
    neg_dev_ids_path = dev_path + ("/neg.ids%d" % vocabulary_size)
    self.data_to_token_ids(dev_path + "/pos.txt", pos_dev_ids_path, vocab_path)
    self.data_to_token_ids(dev_path + "/neg.txt", neg_dev_ids_path, vocab_path)

    return (pos_train_ids_path, neg_train_ids_path, vocab_path, 
      pos_dev_ids_path, neg_dev_ids_path, unsup_train_ids_path)


  def read_data(self, pos_path, neg_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    valid_set = [[] for _ in _buckets]
    with tf.gfile.GFile(pos_path, mode="r") as pos_file:
      with tf.gfile.GFile(neg_path, mode="r") as neg_file:
        pos, neg = pos_file.readline(), neg_file.readline()
        counter = 0
        while pos and neg and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          pos_ids = [int(x) for x in pos.split()]
          neg_ids = [int(x) for x in neg.split()]
          for bucket_id, (pos_size, sentiment) in enumerate(_buckets):
            if len(pos_ids) < pos_size:
              if counter % 9 == 0:
                valid_set[bucket_id].append([pos_ids, [0]])
              else:
                data_set[bucket_id].append([pos_ids, [0]])
              break
          for bucket_id, (neg_size, sentiment) in enumerate(_buckets):
            if len(neg_ids) < neg_size:
              if counter % 9 == 0:
                valid_set[bucket_id].append([neg_ids, [1]])
              else:
                data_set[bucket_id].append([neg_ids, [1]])
              break
          pos, neg = pos_file.readline(), neg_file.readline()
    return data_set, valid_set

  def read_unsup_data(self, unsup_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    unsup_set = [[] for _ in _buckets]
    unsup_valid_set = [[] for _ in _buckets]
    with tf.gfile.GFile(unsup_path, mode="r") as unsup_file:
      unsup = unsup_file.readline()
      counter = 0
      while unsup and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        unsup_ids = [int(x) for x in unsup.split()]
        for bucket_id, (unsup_size, sentiment) in enumerate(_buckets):
          if len(unsup_ids) < unsup_size:
            if counter % 9 == 0:
              unsup_set[bucket_id].append(unsup_ids)
            else:
              unsup_valid_set[bucket_id].append(unsup_ids)
            break
        unsup = unsup_file.readline()
    return unsup_set, unsup_valid_set

  def read_dev(self, pos_path, neg_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    dev_set = [[] for _ in _buckets]
    with tf.gfile.GFile(pos_path, mode="r") as pos_file:
      with tf.gfile.GFile(neg_path, mode="r") as neg_file:
        pos, neg = pos_file.readline(), neg_file.readline()
        counter = 0
        while pos and neg and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          pos_ids = [int(x) for x in pos.split()]
          neg_ids = [int(x) for x in neg.split()]
          for bucket_id, (pos_size, sentiment) in enumerate(_buckets):
            if len(pos_ids) < pos_size:
              dev_set[bucket_id].append([pos_ids, [0]])
              break
          for bucket_id, (neg_size, sentiment) in enumerate(_buckets):
            if len(neg_ids) < neg_size:
              dev_set[bucket_id].append([neg_ids, [1]])
              break
          pos, neg = pos_file.readline(), neg_file.readline()
        self.get_dev(dev_set)

  def next_batch(self):
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                     if self.train_buckets_scale[i] > random_number_01])
    x, y, z= self.get_batch(self.data_set, bucket_id)
    return x, y, z

  def next_unsup(self):
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(self.unsup_buckets_scale))
                     if self.unsup_buckets_scale[i] > random_number_01])
    x, y, z= self.get_unsup(self.unsup_set, bucket_id)
    return x, y, z

  def next_unsup_valid(self,bucket_id):
    x, y, z= self.get_unsup(self.unsup_valid_set, bucket_id)
    return x, y

  def next_valid(self,bucket_id):
    x, y, z= self.get_batch(self.valid_set, bucket_id)
    return x, y

  def next_dev(self):
    self.dev_counter += 1
    return self.dev_set_packed[self.dev_counter]

  def get_dev(self, data):
    self.dev_set_packed =[]
    bucket_id = -1
    for encoder_size, decoder_size in self.buckets:
      bucket_id += 1
      while True:
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(self.batch_size):
          encoder_input, decoder_input = data[bucket_id].pop()

          encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
          encoder_inputs.append(list(encoder_input + encoder_pad))

          decoder_inputs.append(decoder_input)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
          batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
          batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        self.dev_set_packed.append([batch_encoder_inputs,batch_decoder_inputs,bucket_id])
        if len(data[bucket_id])<self.batch_size:
          break


  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(encoder_input + encoder_pad))

      # print (decoder_input)
      decoder_inputs.append(decoder_input)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs = [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
    # print (len(batch_decoder_inputs[0]), bucket_id)
    return batch_encoder_inputs, batch_decoder_inputs, bucket_id

  def get_unsup(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.unsup_buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input = random.choice(data[bucket_id])
      decoder_input = encoder_input
      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(encoder_input + encoder_pad))

      # Decoder inputs are padded.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs = [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    return batch_encoder_inputs, batch_decoder_inputs, bucket_id



f = TextLoader('data',4,80000)
x, y, z= f.next_batch()
for e in f.token_ids_to_sentence(x,z):
  print(' '.join(s for s in e))
# print (y[0])
# x, y, z= f.next_dev()
# print (x[1])
# print (y[0])

# x, y, z = f.next_unsup()
# print (x)
# print (y)