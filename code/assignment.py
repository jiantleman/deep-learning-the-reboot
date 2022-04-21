import math
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Decoder
import sys


def train(model, inputs, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: French train data (all data for training) of shape (num_sentences, window_size)
	:param train_english: English train data (all data for training) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the French sentences to be used by the encoder,
	# and English sentences to be used by the decoder
	# - The English sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#pty
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]
	print("Training model...")

	num_examples = len(inputs)
	indices = tf.random.shuffle(tf.range(num_examples))
	inputs = tf.gather(inputs, indices)

	num_batches = len(inputs)//model.batch_size
	print("Total number of batches: ", num_batches)
	for i in range(num_batches):
		if(i%100 == 0):
			print("Batch: ", i)
		decoder_input = inputs[i*model.batch_size:(i+1)*model.batch_size][:,:-1]
		decoder_label = inputs[i*model.batch_size:(i+1)*model.batch_size][:,1:]
		mask = np.where(decoder_label != eng_padding_index, 1, 0)

		with tf.GradientTape() as tape:
			probs = model.call(decoder_input)
			loss = model.loss_function(probs, decoder_label, mask)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, inputs, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: French test data (all data for testing) of shape (num_sentences, window_size)
	:param test_english: English test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	print("Testing model...")
	num_batches = len(inputs)//model.batch_size
	print("Total number of batches: ", num_batches)
	losses, accuracies, num_words = [], [], []
	for i in range(num_batches):
		if(i%100 == 0):
			print("Batch: ", i)
		decoder_input = inputs[i*model.batch_size:(i+1)*model.batch_size][:,:-1]
		decoder_label = inputs[i*model.batch_size:(i+1)*model.batch_size][:,1:]
		mask = np.where(decoder_label != eng_padding_index, 1, 0)

		probs = model.call(decoder_input)
		losses.append(model.loss_function(probs, decoder_label, mask))
		accuracies.append(model.accuracy_function(probs, decoder_label, mask).numpy())
		num_words.append(np.sum(mask))
	avg_loss = sum(losses)/sum(num_words)
	avg_acc = np.sum(np.array(accuracies)*np.array(num_words))/sum(num_words)
	return math.exp(avg_loss), avg_acc
	

def main():
	
	print("Running preprocessing...")
	data_dir   = '../data'
	file_names = ('fls.txt', 'els.txt', 'flt.txt', 'elt.txt')
	file_paths = [f'{data_dir}/{fname}' for fname in file_names]
	train_eng, test_eng, _, _, vocab_eng, _ ,eng_padding_index = get_data(*file_paths)
	print("Preprocessing complete.")

	model = Transformer_Decoder(ENGLISH_WINDOW_SIZE, len(vocab_eng))

	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_eng, eng_padding_index)
	perplexity, accuracy = test(model, test_eng, eng_padding_index)
	print("Perplexity: ", perplexity)
	print("Accuracy: ", accuracy)

	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above


if __name__ == '__main__':
	main()
