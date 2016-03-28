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
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
_buckets = [(5, 2), (10, 5), (20, 10), (40, 20), (80, 40), (160, 40), (320, 40)]


class TextLoader():

  def __init__(self,data_dir,batch_size,seq_length, vocab_size):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.vocab_size = vocab_size
    self.x_path = self.data_dir + ("/xdata.ids%d.x" % self.vocab_size)
    self.y_path = self.data_dir + ("/ydata.ids%d.y" % self.vocab_size)
    self.vocab_path = self.data_dir + ("/vocab%d" % self.vocab_size)
    

    if not (os.path.exists(self.x_path) and os.path.exists(self.y_path) and os.path.exists(self.vocab_path)):
      print ('preparing data')
      self.x_path, self.y_path, self.vocab_path = self.prepare_kg_data(self.data_dir,self.vocab_size)
    else:
      print ('loading prepared file')
    print ('checking integrity of file')
    with gfile.GFile(self.x_path, mode="r") as xf:
      xline = 0
      for line in xf:
        xline += 1
    with gfile.GFile(self.y_path, mode="r") as yf:
      yline = 0
      for line in yf:
        yline += 1
    if xline == yline:
      print ('integrity checking finished')
      print ('preparing data_set')
      self.data_set = self.read_data(self.x_path, self.y_path)
      self.train_bucket_sizes = [len(self.data_set[b]) for b in xrange(len(_buckets))]
      self.train_total_size = float(sum(self.train_bucket_sizes))
      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.
      self.train_buckets_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.train_total_size
                             for i in xrange(len(self.train_bucket_sizes))]
    else:
      raise Exception("error: wrong line_number. Make sure both files have the same line nubmer.")


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
      with gfile.GFile(data_path+"/xdata", mode="r") as f:
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
      with gfile.GFile(data_path+"/ydata", mode="r") as f:
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
    """Convert a string to list of integers representing token-ids.

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


  def prepare_kg_data(self, data_dir, vocabulary_size):
    """Get knowledge graph data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      vocabulary_size: size of the vocabulary to create and use.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for x training data-set,
        (2) path to the token-ids for y training data-set,
        (3) path to the vocabulary file.
    """
    # Get knowlege graph data to the specified directory.
    train_path = data_dir
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    self.create_vocabulary(vocab_path, train_path, vocabulary_size)

    # Create token ids for the training data.
    x_train_ids_path = train_path + ("/xdata.ids%d" % vocabulary_size)
    y_train_ids_path = train_path + ("/ydata.ids%d" % vocabulary_size)
    self.data_to_token_ids(train_path + "/xdata", x_train_ids_path, vocab_path)
    self.data_to_token_ids(train_path + "/ydata", y_train_ids_path, vocab_path)

    # Create token ids for the development data.
    # fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
    # en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
    # data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path)
    # data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path)

    return (x_train_ids_path, y_train_ids_path, vocab_path)


  def read_data(self, source_path, target_path, max_size=None):
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
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          target_ids.append(EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = source_file.readline(), target_file.readline()
    return data_set

  def next_batch(self):
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    self.bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                     if self.train_buckets_scale[i] > random_number_01])
    x, y = self.get_batch(self.data_set, self.bucket_id)
    return x, y

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
    encoder_size, decoder_size = _buckets[self.bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append(decoder_input +
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
    return batch_encoder_inputs, batch_decoder_inputs




f = TextLoader('data',50,50,40000)
x, y = f.next_batch()
print (x[5])

