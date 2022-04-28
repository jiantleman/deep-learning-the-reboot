import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Decoder(tf.keras.Model):
	def __init__(self, window_size, vocab_size):

		super(Transformer_Decoder, self).__init__()

		# hyperparameters
		self.vocab_size = vocab_size
		self.window_size = window_size
		self.batch_size = 64
		self.embedding_size = 128
		self.num_blocks = 2

		# layers
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.embedding  = tf.keras.layers.Embedding(self.vocab_size,  self.embedding_size)
		self.pos_encoder = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.decoder_blocks = tf.keras.Sequential(
			[transformer.Transformer_Block(self.embedding_size) for _ in range(self.num_blocks)]
		)
		self.dense = tf.keras.layers.Dense(self.vocab_size)

	@tf.function
	def call(self, inputs):
		"""
		:param inputs: ???
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
		"""

		# combine sentence & positional embeddings
		tok_embedding = self.embedding(inputs)
		blocks_input = self.pos_encoder(tok_embedding)
		# pass to decoder blocks & dense layer(s)
		blocks_output = self.decoder_blocks(blocks_input)
		logits = self.dense(blocks_output)
		
		return tf.nn.softmax(logits)

	def accuracy_function(self, prbs, labels, mask):
		"""
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
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
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		losses = tf.keras.metrics.sparse_categorical_crossentropy(labels, prbs)
		total_loss = tf.math.reduce_sum(tf.boolean_mask(losses,mask))
		return total_loss

	def __call__(self, *args, **kwargs):
		return super(Transformer_Decoder, self).__call__(*args, **kwargs)
