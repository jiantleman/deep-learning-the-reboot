import math
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Decoder
import sys

TRAIN_RATIO = 0.8

def train(model, inputs, padding_index):
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
		decoder_input = inputs[i*model.batch_size:(i+1)*model.batch_size][:,:-1]
		decoder_label = inputs[i*model.batch_size:(i+1)*model.batch_size][:,1:]
		mask = np.where(decoder_label != padding_index, 1, 0)

		with tf.GradientTape() as tape:
			probs = model.call(decoder_input)
			loss = model.loss_function(probs, decoder_label, mask)

		if(i%100 == 0):
			print("Batch: {}, loss: {}".format(i, loss))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, inputs, padding_index):
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
		mask = np.where(decoder_label != padding_index, 1, 0)

		probs = model.call(decoder_input)
		losses.append(model.loss_function(probs, decoder_label, mask))
		accuracies.append(model.accuracy_function(probs, decoder_label, mask).numpy())
		num_words.append(np.sum(mask))
	avg_loss = sum(losses)/sum(num_words)
	avg_acc = np.sum(np.array(accuracies)*np.array(num_words))/sum(num_words)
	return math.exp(avg_loss), avg_acc

def generate_sentence(word1, length, tokenizer, model, sample_n=5):
	"""
	Takes a model, vocab, selects from the most likely next word from the model's distribution

	:param model: trained RNN model
	:param vocab: dictionary, word to id mapping
	:return: None
	"""

	first_word_index = tokenizer.token_to_id(word1)
	input = np.zeros((1,model.window_size+1))
	input[0,0] = tokenizer.token_to_id("[BOS]")
	input[0,1] = first_word_index
	input[0,2] = tokenizer.token_to_id("[DELIM]")
	
	for i in range(1,length):
		logits = model.call(input[:,:-1])[0,i].numpy()
		top_n = np.argsort(logits)[-sample_n:]
		n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
		out_index = np.random.choice(top_n, p=n_logits)
		input[0,i+1] = out_index
		if out_index == tokenizer.token_to_id("[EOS]"):
			break
	
	input = input.astype(int)
	print(tokenizer.decode(input[0,:i+1]))
	

def main():
	
	print("Running preprocessing...")
	tokenizer, data, pretrain_data = get_data()
	data = np.array(data)
	np.random.shuffle(data)
	train_data = data[0:int(TRAIN_RATIO*len(data))]
	test_data = data[int(TRAIN_RATIO*len(data)):]
	print("Preprocessing complete.")

	model = Transformer_Decoder(WINDOW_SIZE, VOCAB_SIZE)
	model.build((None, model.window_size))
	model.summary()
	if len(sys.argv) == 2:
		model.load_weights(sys.argv[1]).expect_partial()
	else:
		print("Pretraining")
		train(model, pretrain_data, PADDING_INDEX)
		print("Finetuning")
		train(model, train_data, PADDING_INDEX)
		model.save_weights('./checkpoints/my_checkpoint_pretrained')
	
		perplexity, accuracy = test(model, test_data, PADDING_INDEX)
		print("Perplexity: ", perplexity)
		print("Accuracy: ", accuracy)
	
	start_words = ["Monica", "Rachel", "Phoebe", "Joey", "Chandler", "Ross"]
	for word in start_words:
		generate_sentence(word, WINDOW_SIZE, tokenizer, model)


if __name__ == '__main__':
	main()
