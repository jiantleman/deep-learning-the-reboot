import math
import numpy as np
import tensorflow as tf
import numpy as np


def Attention_Matrix(K, Q, use_mask=True):
	"""
	This functions runs a single attention head.
	:param K: is [batch_size x window_size_keys x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention matrix
	"""

	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	# - Q is [batch_size x window_size_queries x embedding_size]
	# - K is [batch_size x window_size_keys x embedding_size]
	# - Mask is [batch_size x window_size_queries x window_size_keys]


	# Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
	# This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
	# Those weights are then used to create linear combinations of the corresponding values for each query.
	# Those queries will become the new embeddings.

	K_T = tf.transpose(K, perm=[0,2,1])
	QK = tf.einsum('bij,bjk->bik', Q, K_T)
	embedding_size = K.get_shape()[2]
	QK = QK/math.sqrt(embedding_size)
	if use_mask: 
		QK += atten_mask
	return tf.nn.softmax(QK)		# attention matrix


class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		# init weight matrices for K, V, and Q
		self.K = self.add_weight("key_matrix",   shape=[input_size, output_size])
		self.Q = self.add_weight("query_matrix", shape=[input_size, output_size])
		self.V = self.add_weight("value_matrix", shape=[input_size, output_size])
 
	@tf.function
	def call(self, inputs):

		"""
		This functions runs a single attention head.
		:param inputs:
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
		# mult weights and inputs to get keys, values, and queries
		K = tf.tensordot(inputs, self.K, axes=[[2],[0]])
		V = tf.tensordot(inputs, self.V, axes=[[2],[0]])
		Q = tf.tensordot(inputs, self.Q, axes=[[2],[0]])
		# call Attention_Matrix with keys, queries, and self.use_mask
		att_mat = Attention_Matrix(K, Q, self.use_mask)

		return tf.einsum('bij,bjk->bik', att_mat, V)		# apply attention matrix to values

class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Multi_Headed, self).__init__()

		# initialize heads
		self.num_heads = 2
		single_head_size = int(emb_sz/self.num_heads)
		self.heads = [Atten_Head(single_head_size, single_head_size, True) for _ in range(self.num_heads)]
		self.dense = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		FOR CS2470 STUDENTS:
		This functions runs a multiheaded attention layer.
		Requirements:
			- Splits data for 3 different heads into size embed_sz/3
			- Create three different attention heads
			- Each attention head should have input size embed_size and output embed_size/3
			- Concatenate the outputs of these heads together
			- Apply a linear layer
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""

		inputs = tf.split(inputs, self.num_heads, 2)
		output = None
		for i in range(self.num_heads):
			head_output = self.heads[i](inputs[i])
			if output == None:
				output = head_output
			else:			
				output = tf.concat([output, head_output], -1)
		return self.dense(output)
		

class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		This functions creates a feed forward network as described in 3.3
		https://arxiv.org/pdf/1706.03762.pdf
		Requirements:
		- Two linear layers with relu between them
		:param inputs: input tensor [batch_size x window_size x embedding_size]
		:return: tensor [batch_size x window_size x embedding_size]
		"""
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Transformer_Block, self).__init__()

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
		self.self_atten = Multi_Headed(emb_sz)
		self.ff_layer = Feed_Forwards(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		This functions calls a transformer block.
		There are two possibilities for when this function is called.
		    - if self.is_decoder == False, then:
		        1) compute unmasked attention on the inputs
		        2) residual connection and layer normalization
		        3) feed forward layer
		        4) residual connection and layer normalization
		    - if self.is_decoder == True, then:
		        1) compute MASKED attention on the inputs
		        2) residual connection and layer normalization
		        3) computed UNMASKED attention using context
		        4) residual connection and layer normalization
		        5) feed forward layer
		        6) residual layer and layer normalization
		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)
		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""
		atten_out = self.self_atten(inputs)
		atten_out += inputs
		atten_norm = self.layer_norm(atten_out)
		ff_out = self.ff_layer(atten_norm)
		ff_out += atten_norm
		ff_norm = self.layer_norm(ff_out)
		
		return tf.nn.relu(ff_norm)



class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		Adds positional embeddings to word embeddings.
		:param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		return x+self.positional_embeddings
