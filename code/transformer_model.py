import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Decoder(tf.keras.Model):
	def __init__(self, window_size, vocab_size):

		super(Transformer_Decoder, self).__init__()

		# Define hyperparameters
		self.vocab_size = vocab_size
		self.window_size = window_size
		self.batch_size = 64
		self.embedding_size = 128
		self.num_blocks = 1

		# Define layers
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.embedding  = tf.keras.layers.Embedding(self.vocab_size,  self.embedding_size)
		self.pos_encoder = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.blocks = tf.keras.Sequential(
			*[transformer.Transformer_Block(self.embedding_size) for _ in range(self.num_blocks)]
		)
		self.dense = tf.keras.layers.Dense(self.vocab_size)

	@tf.function
	def call(self, inputs):
		"""
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		#1) Add the positional embeddings to French sentence embeddings
		#2) Pass the French sentence embeddings to the encoder
		#3) Add positional embeddings to the English sentence embeddings
		#4) Pass the English embeddings and output of your encoder, to the decoder
		#5) Apply dense layer(s) to the decoder out to generate probabilities

		tok_embedding = self.embedding(inputs)
		blocks_input = self.pos_encoder(tok_embedding)
		blocks_output = self.blocks(blocks_input)
		output = self.dense(blocks_output)
		return tf.nn.softmax(output)

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		losses = tf.keras.metrics.sparse_categorical_crossentropy(labels, prbs)
		total_loss = tf.math.reduce_sum(tf.boolean_mask(losses,mask))
		return total_loss

	def __call__(self, *args, **kwargs):
		return super(Transformer_Decoder, self).__call__(*args, **kwargs)
