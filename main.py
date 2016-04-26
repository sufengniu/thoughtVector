from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import gzip
import re

import numpy as np
from sklearn import metrics
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib import skflow

import data_utils

_PAD = "_PAD"
_UNK = "_UNK"
_POS = "_POS"
_NEG = "_NEG"

_START_VOCAB = [_PAD, _UNK, _POS, _NEG]

PAD_ID = 0
UNK_ID = 1
POS_ID = 2
NEG_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
_buckets = [(80, 1), (160, 1), (320, 1), (640, 1), (1280,1)]



# load data
"""
datasets = {'imdb_sup': (data_utils.load_data, data_utils.prepare_data), 
'imdb_unsup': (data_utils.load_data_unsup, data_utils.prepare_data)}

def get_dataset(name):
	return datasets[name][0], datasets[name][1]
"""

flags=tf.app.flags

# RNN configuration
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("dropout", 1, "dropout rates")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
flags.DEFINE_integer("embedding_size", 128, "Size of each model layer.")
flags.DEFINE_integer("depth", 3, "Number of layers in the model.")
flags.DEFINE_integer("vocab_size", 100000, "vocabulary size.")
flags.DEFINE_string("data_dir", "data", "Data directory")
flags.DEFINE_string("train_dir", "data", "Training directory.")
flags.DEFINE_string("model_types", "LSTM", "RNN model types (LSTM, GRU)")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
flags.DEFINE_boolean("validation", False, "Run validation if true")

FLAGS = flags.FLAGS

# testing
def embedding_seq2seq(X, do_decode):
	with variable_scope.variable_scop(scope or "embedding_seq2seq"):
		encoder_cell = rnn_cell.EmbeddingWrapper(cell, embedding_classes=num_encoder_symbols,
				embedding_size=embedding_size)
		encoder_outputs, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtypes=dtype)

		# decoder
		#if isinstance(do_decode, bool):

class rnnAutoEncoder(object):
	def __init__(self, size, buckets, max_gradient_norm=5.0, 
			batch_size=64, learning_rate=0.5, forward_only=False):

		softmax_loss_function = None
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)

		print("Creating %d layers of %d units." % (FLAGS.depth, size))
		if model_types is GRU:
			single_cell = tf.nn.rnn_cell.GRUCell(size)
		elif model_types is LSTM:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		else:
			print("model types not found!")
			sys.exit(1)
			cell = single_cell
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell])

		# The seq2seq function: we use embedding for the input.
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.nn.seq2seq.embedding_rnn_seq2seq(
					encoder_inputs, decoder_inputs, cell, vocab_size,
					o_vocab_size, size, feed_previous=do_decode)

		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []
		for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
													name="encoder{0}".format(i)))
		for i in xrange(buckets[-1][1]):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
													name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(tf.float32, shape=[None],name="weight{0}".format(i)))

		targets = self.decoder_inputs

		if forward_only:
			# print (self.decoder_inputs)
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets,
				self.target_weights, buckets, lambda x, y: seq2seq_f(x, y,False),
				softmax_loss_function=softmax_loss_function)
		else:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets,
				self.target_weights, buckets,
				lambda x, y: seq2seq_f(x, y, False),
				softmax_loss_function=softmax_loss_function)

		params = tf.trainable_variable()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)	# TODO: ADA
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, norm = tf.clip_by_global_norm(gradients,
						max_gradient_norm)
				self.gradient_norms.append(norm)
				self.updates.append(opt.apply_gradients(
					zip(clipped_gradients, params), global_step=self.global_step))

	def layer_step(self, session, X, bucket_id, forward_only):
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
						" %d != %d." % (len(X), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
						" %d != %d." % (len(X), decoder_size))

		target_weights = []
		for _ in range(0,self.batch_size):
			target_weights.append(1.0)

		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = X[l]
		for l in xrange(decoder_size):
			input_feed[self.encoder_inputs[l].name] = X[l]
			input_feed[self.target_weights[l].name] = np.array(target_weights)

		if not forward_only:
			output_feed = [self.updates[bucket_id],
						self.gradient_norms[bucket_id],
						self.losses[bucket_id]]
		else:
			output_feed = [self.losses[bucket_id]]
			for l in xrange(decoder_size):
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None # gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:] # no gradient norm, loss, outputs.

	def step(self, session, encoder_inputs, decoder_inputs, bucket_id, forward_only):
		# Check if the sizes match.
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
							" %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
							" %d != %d." % (len(decoder_inputs), decoder_size))

		# Input feed: encoder inputs, decoder inputs as provided.
		target_weights = []
		for _ in range(0,self.batch_size):
			target_weights.append(1.0)

		
		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = np.array(target_weights)

		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
							self.gradient_norms[bucket_id],  # Gradient norm.
							self.losses[bucket_id]]  # Loss for this batch.
		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

	
	"""
	def create_model_regression(session, size, forward_only=True):
		print("Creating %d layers of %d units." % (FLAGS.depth, size))
		if model_types is GRU:
			single_cell = tf.nn.rnn_cell.GRUCell(size)
		elif model_types is LSTM:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		else:
			print("model types not found!")
			sys.exit(1)
			cell = single_cell
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell])

		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.nn.seq2seq.embedding_attention_seq2seq(
				encoder_inputs, decoder_inputs, cell,
				num_encoder_symbols=source_vocab_size,
				num_decoder_symbols=target_vocab_size,
				embedding_size=size,
				output_projection=output_projection,
				feed_previous=do_decode)

		def softmax_loss_function(inputs, labels):
			with tf.device("/cpu:0"):


		if forward_only:
			# outputs, losses = embedding_seq2seq(X, True)
			losses = []
			outputs = []
			with ops.op_scope(all_inputs, name, "model_with_buckets"):
				for j, bucket in enumerate(buckets):
					with variable_scope.variable_scope(variable_scope.get_variable_scope(),
						reuse=True if j > 0 else None):
					bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
						decoder_inputs[:bucket[1]])
					outputs.append(bucket_outputs)
					if per_example_loss:
						losses.append(sequence_loss_by_example(
							outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
							softmax_loss_function=softmax_loss_function))
					else:
						losses.append(sequence_loss(
							outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
							softmax_loss_function=softmax_loss_function))
		else:
			# outputs, losses = embedding_seq2seq(X, False)

		return model
	"""
"""
def read_data(train, max_size=None):
	data_set = [[] for _ in _buckets
	for train:
		for bucket_id, (source_size) in enumerate(_buckets):
			f len(source_ids) < source_size:
				data_set[bucket_id].append([source_ids])
				break
	return data_set
"""
def create_model_softmax(session, forward_only):
	model = rnnAutoEncoder(
		FLAGS.embedding_size, _buckets, FLAGS.max_gradient_norm,
		FLAGS.batch_size, FLAGS.learning_rate, forward_only=forward_only)
	"""
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	"""
	return model

def main(_):
	""" train a sentiment analysis mode using """
	print ("[*] Preparing IMDB sentiment data... in %s" % FLAGS.data_dir)
	#load_data, prepare_data = get_dataset('imdb_unsup')
	

	with tf.Graph().as_default(), tf.Session() as sess:

		# create model
		print('Loading unsupervise training data')
		#train, valid = load_data(n_words=FLAGS.vocab_size, valid_portion=0.05, maxlen=FLAGS.max_len)
		#train_set = read_data()
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
								for i in xrange(len(train_bucket_sizes))]

		print('create the model')
		model = create_model_softmax(sess, FLAGS.embedding_size, forward_only=False)

		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []

		# unsupervised layer-wise train
		for i in range(FLAGS.depth):
			for j in range(FLGAS.max_epoch):
				random_number_01 = np.random.random_sample()
				bucket_id = min([i for i in xrange(len(train_buckets_scale))
                		if train_buckets_scale[i] > random_number_01])

				start_time = time.time()
				# load each batch
				encoder_inputs, bucket_id = model.get_batch(train, bucket_id)

				_, step_loss, _ = model.layer_step(sess, encoder_inputs, bucket_id, False)
				step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
				loss += step_loss / FLAGS.steps_per_checkpoint
				current_step += 1

				if current_step % FLAGS.steps_per_checkpoint == 0:
					perplexity = math.exp(loss) if loss < 300 else float('inf')
					print ("global step %d learning rate %.4f step-time %.2f perplexity "
				    		"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
							step_time, perplexity))
					previous_losses.append(loss)
					# Save checkpoint and zero timer and loss.
					checkpoint_path = os.path.join(FLAGS.train_dir, "sentiment.ckpt")
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
					step_time, loss = 0.0, 0.0
					# Run evals on development set and print their perplexity.
					for bucket_id in xrange(len(_buckets)):
						encoder_inputs, decoder_inputs, bucket_id = sentiment_data.next_valid()
						_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, bucket_id, True)
						eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
						print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
					sys.stdout.flush()



		print("unsupervised train finished, start supervised tuning")
		# supervised fine-tune
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		while True:
			start_time = time.time()
			# load each batch
			encoder_inputs, decoder_inputs, bucket_id = sentiment_data.next_batch()

			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, bucket_id, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1

			if current_step % FLAGS.steps_per_checkpoint == 0:
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
					"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
						step_time, perplexity))
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "sentiment.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(_buckets)):
					encoder_inputs, decoder_inputs, bucket_id = sentiment_data.next_valid()
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, bucket_id, True)
					eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				sys.stdout.flush()

if __name__ == '__main__':
	tf.app.run()
